from ultralytics import YOLO
import torch
import time
import os
import cv2
import warnings
import threading
from queue import Queue, Empty
import yt_dlp

# Suppress warnings để giảm I/O overhead
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['YOLO_VERBOSE'] = 'False'

# ---------- 1. Load YOLO Model ----------
use_tensorrt = True  # Bật TensorRT để tăng tốc (cần NVIDIA GPU + CUDA)

# Kiểm tra GPU trước để chọn model phù hợp
gpu_info = None
# Kiểm tra GPU có không
if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    gpu_memory = torch.cuda.get_device_properties(idx).total_memory / 1024**3  # GB
    gpu_name = torch.cuda.get_device_name(idx)
    gpu_info = {"idx": idx, "name": gpu_name, "memory": gpu_memory} 
    
    # Tự động chọn model dựa trên VRAM
    if gpu_memory < 4.0:
        print(f"⚠ Detected low-end GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        print("  → Recommending YOLO11n for better performance")
        model_name = "yolo11n"
        use_tensorrt = False
    elif gpu_memory < 6.0:
        print(f"⚠ Detected mid-range GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        print("  → Recommending YOLO11s for balanced performance")
        model_name = "yolo11s"
    else:
        model_name = "yolo11m"
    
    recommended_imgsz = 320
else:
    model_name = "yolo11m"
    recommended_imgsz = 320

tensorrt_engine_path = f"{model_name}.engine"
print(f"Loading model '{model_name}'...")

# Kiểm tra xem có TensorRT engine chưa
tensorrt_enabled = False
if use_tensorrt and torch.cuda.is_available():
    if os.path.exists(tensorrt_engine_path):
        print(f"  → Found TensorRT engine: {tensorrt_engine_path}")
        try:
            model = YOLO(tensorrt_engine_path)
            print("✓ TensorRT model loaded successfully!")
            tensorrt_enabled = True
        except Exception as e:
            print(f"  ⚠ Failed to load TensorRT engine: {e}")
            print("  → Falling back to PyTorch model...")
            model = YOLO(model_name)
            print("✓ PyTorch model loaded successfully!")
    else:
        print(f"  → TensorRT engine not found. Attempting to export to TensorRT...")
        temp_model = YOLO(model_name)
        try:
            # Export sang TensorRT (FP16 để tăng tốc)
            temp_model.export(
                format="engine",
                imgsz=recommended_imgsz,  # Dùng imgsz được recommend cho GPU
                half=True,  # FP16 để tăng tốc
                device=0 if torch.cuda.is_available() else 'cpu',  # Đảm bảo device đúng
                simplify=True,
            )
            print(f"✓ TensorRT export completed! Engine saved: {tensorrt_engine_path}")
            model = YOLO(tensorrt_engine_path)
            print("✓ TensorRT model loaded successfully!")
            tensorrt_enabled = True
        except Exception as e:
            print(f"  ✗ TensorRT export failed: {e}")
            print("  → Falling back to PyTorch model with optimizations...")
            model = temp_model
            # Tối ưu PyTorch model khi fallback: compile với torch.compile()
            if torch.cuda.is_available() and hasattr(torch, 'compile'):
                try:
                    print("  → Compiling PyTorch model with torch.compile for faster inference...")
                    model.model = torch.compile(model.model, mode="reduce-overhead")
                    print("  → Model compiled successfully!")
                except Exception as compile_err:
                    print(f"  → Compilation failed (using eager mode): {compile_err}")
            print("✓ PyTorch model loaded successfully!")
else:
    model = YOLO(model_name)  # Tự động tải về nếu chưa có
    print("✓ Model loaded successfully!")

# Compile model với PyTorch 2.x để tăng tốc (chỉ khi KHÔNG dùng TensorRT)
# TensorRT đã optimize rồi, không cần torch.compile
if not tensorrt_enabled and torch.cuda.is_available() and hasattr(torch, 'compile'):
    try:
        print("Compiling model with torch.compile...")
        model.model = torch.compile(model.model, mode="reduce-overhead")
        print("  → Model compiled successfully")
    except Exception as e:
        print(f"  → Compilation failed (using eager mode): {e}")

# ---------- 2. Device setup ----------
# GPU Utilization Optimization: Batch processing để tăng GPU usage
# Trade-off: Tăng latency nhẹ nhưng tăng throughput đáng kể
USE_BATCH_PROCESSING = True  # True = tích lũy frames rồi batch inference, False = single frame

# Tự động điều chỉnh batch size dựa trên GPU VRAM
if torch.cuda.is_available():
    gpu_memory_gb = gpu_info["memory"] if gpu_info else torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Batch size dựa trên VRAM: ~1GB per batch item
    BATCH_SIZE = min(8, max(1, round(gpu_memory_gb)))
else:
    BATCH_SIZE = 1  # CPU mode

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    idx = gpu_info["idx"]
    gpu_name = gpu_info["name"]
    gpu_memory = gpu_info["memory"]
    print(f"GPU: {gpu_name} (index {idx}, {gpu_memory:.1f} GB VRAM)")
    if USE_BATCH_PROCESSING and BATCH_SIZE > 1:
        print(f"  → Batch size: {BATCH_SIZE} (batch processing mode - higher GPU utilization)")
    else:
        print(f"  → Batch size: {BATCH_SIZE} (single frame mode - low latency)")
    
    # Clear cache ban đầu để có memory sạch
    torch.cuda.empty_cache()
    
    # Pre-allocate một dummy tensor để khởi tạo CUDA context (warm-up)
    # Giúp tránh latency spike ở inference đầu tiên
    print("  → Warming up GPU...")
    dummy_tensor = torch.zeros(1, 3, recommended_imgsz, recommended_imgsz, device=device)
    if hasattr(torch.cuda, 'synchronize'):
        torch.cuda.synchronize()
    del dummy_tensor
    torch.cuda.empty_cache()
    print("  → GPU warmed up!")
else:
    print(f"  → Batch size: 1 (CPU mode)")

# --- Canonical predict device for Ultralytics ---
predict_device = 0 if device == "cuda" else "cpu"

# --- Safe predict wrapper: converts to list, handles CUDA OOM by retrying with batch=1
def safe_predict(**predict_kwargs):
    """
    - Trả về list(results) an toàn
    - Nếu OOM xảy ra, clear cache và retry một lần với batch=1
    - Nếu lỗi khác, trả về []
    """
    try:
        raw = model.predict(**predict_kwargs)
        return list(raw) if not isinstance(raw, list) else raw
    except RuntimeError as e:
        err = str(e).lower()
        if "out of memory" in err:
            print("✗ CUDA OOM during predict — retrying with batch=1")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                predict_kwargs['batch'] = 1
                raw = model.predict(**predict_kwargs)
                return list(raw) if not isinstance(raw, list) else raw
            except Exception as e2:
                print(f"✗ Predict retry failed: {e2}")
                return []
        else:
            print(f"✗ Predict runtime error: {e}")
            return []
    except Exception as e:
        print(f"✗ Unexpected predict error: {e}")
        return []

# --- Unified inference function: xử lý cả batch và single frame
def run_inference(frames, frame_ids, frame_originals, frame_times):
    """
    Chạy inference cho batch frames hoặc single frame
    Trả về: (results, inference_time, inference_end_time)
    """
    if not frames:
        return [], 0, time.time()
    
    batch_size = len(frames)
    inference_start = time.time()
    
    # Dùng lock để đảm bảo thread-safe
    with model_lock:
        # Luôn dùng list để tránh warning với Ultralytics
        source = frames
        # Với batch_size=1, không truyền batch parameter (None) để tránh warning
        batch_param = batch_size if batch_size > 1 else None
        results = safe_predict(
            source=source,
            conf=0.4,
            iou=0.6,
            device=predict_device,
            save=False,
            show=False,
            half=(device=="cuda"),
            imgsz=recommended_imgsz,
            batch=batch_param,
            verbose=False,
            stream=False,
            agnostic_nms=False,
            rect=True,
            max_det=30,
            augment=False,
        )
    
    inference_end_time = time.time()
    inference_time = (inference_end_time - inference_start) / batch_size if batch_size > 0 else 0
    
    return results, inference_time, inference_end_time

# --- Helper function: xử lý results và đưa vào detection_queue
def process_and_queue_results(results, frame_ids, frame_originals, frame_times, inference_time, inference_end_time):
    """
    Xử lý results và đưa vào detection_queue, tự động gọi task_done()
    """
    num_frames = len(frame_ids)
    num_results = len(results) if results else 0
    
    # Xử lý từng result và đưa vào queue
    for i in range(min(num_results, num_frames)):
        result = results[i]
        frame_original = frame_originals[i] if i < len(frame_originals) else None
        
        # Bỏ qua nếu result hoặc frame_original là None hoặc không hợp lệ
        if result is None or frame_original is None:
            continue
        
        # KHÔNG drop frames để giữ timeline đúng với video gốc
        # Nếu queue đầy, đợi (block) thay vì drop để đảm bảo sync với video timeline
        # Việc drop frames sẽ làm timeline bị sai và video chạy nhanh hơn
        # Dùng put() KHÔNG timeout để block khi queue đầy, đảm bảo không mất frames
        # Chỉ block khi queue đầy, không drop để giữ timeline đúng
        detection_queue.put((
            frame_ids[i],
            result,
            frame_original,
            frame_times[i],
            inference_time,
            inference_end_time
        ))
    
    # Gọi task_done() cho tất cả frames đã được xử lý (kể cả khi không có result)
    # Điều này đảm bảo queue tracking đúng và tránh deadlock
    for _ in range(num_frames):
        frame_queue.task_done()

# ---------- 3. Video/Livestream source setup ----------
# Có thể dùng: video file, camera (0), RTSP URL, hoặc YouTube URL
SOURCE = "https://www.youtube.com/watch?v=cH7VBI4QQzA"  # YouTube livestream URL
# SOURCE = "PART01_Introdution/RawData/ShortBadmintonVideo_30fps.mp4"  # Video file
# SOURCE = 0  # Camera/webcam

# Cấu hình visualization
SHOW_VIDEO = True  # Bật/tắt hiển thị video (True = xem real-time, False = chỉ xử lý)
PRINT_EVERY_N_FRAMES = 200  # Chỉ in thông tin mỗi N frames (giảm I/O overhead, tăng FPS)

# Multithreading configuration
FRAME_BUFFER_SIZE = 10  # Số frames tối đa trong frame buffer (Thread 1 -> Thread 2)
DETECTION_BUFFER_SIZE = 6  # Số detections tối đa trong detection buffer (Thread 2 -> Main)

# Xử lý source: kiểm tra xem là YouTube URL, video file, hay camera
is_youtube = isinstance(SOURCE, str) and ("youtube.com" in SOURCE or "youtu.be" in SOURCE)
is_video_file = isinstance(SOURCE, str) and os.path.exists(SOURCE)
is_camera = isinstance(SOURCE, int) or SOURCE == 0

stream_url = SOURCE
target_fps = 30.0  # Default FPS

if is_youtube:
    print("="*60)
    print("YOUTUBE LIVESTREAM MODE")
    print("="*60)
    print(f"YouTube URL: {SOURCE}")
    print("Getting stream URL with yt-dlp...")
    
    try:
        # Lấy stream URL từ YouTube
        def get_youtube_stream_url(youtube_url: str, preferred_height=720):
            """Lấy direct stream URL từ YouTube URL"""
            ydl_opts = {"skip_download": True, "quiet": False}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                formats = info.get("formats") or [info]
                
                # Ưu tiên direct video streams
                direct_candidates = []
                hls_candidates = []
                
                for f in formats:
                    url = f.get("url")
                    if not url:
                        continue
                    
                    protocol = f.get("protocol", "").lower()
                    is_hls = "m3u8" in protocol or "dash" in protocol or "hls" in protocol
                    has_video = f.get("vcodec") != "none"
                    
                    if not has_video:
                        continue
                    
                    if is_hls:
                        hls_candidates.append(f)
                    else:
                        direct_candidates.append(f)
                
                # Ưu tiên direct streams
                candidates = direct_candidates if direct_candidates else hls_candidates
                
                if preferred_height:
                    with_height = [f for f in candidates if f.get("height")]
                    if with_height:
                        # Tìm format gần nhất với preferred_height
                        best = min(with_height, key=lambda x: abs((x.get("height") or 0) - preferred_height))
                        return best.get("url"), best.get("fps") or 30.0
                
                # Fallback: lấy format tốt nhất
                if candidates:
                    return candidates[-1].get("url"), candidates[-1].get("fps") or 30.0
                
                raise RuntimeError("Không tìm thấy stream URL từ YouTube")
        
        stream_url, target_fps = get_youtube_stream_url(SOURCE, preferred_height=720)
        print(f"✓ Stream URL obtained successfully!")
        print(f"  Target FPS: {target_fps:.1f}")
        print(f"  Stream type: {'HLS (m3u8)' if 'm3u8' in stream_url.lower() or 'hls' in stream_url.lower() else 'Direct'}")
        
    except Exception as e:
        print(f"✗ Error getting YouTube stream URL: {e}")
        raise

elif is_video_file:
    print("="*60)
    print("VIDEO FILE MODE")
    print("="*60)
    stream_url = SOURCE
    if not os.path.exists(stream_url):
        raise FileNotFoundError(f"Video file not found: {stream_url}")
    
    # Lấy FPS từ video file
    cap = cv2.VideoCapture(stream_url)
    target_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
elif is_camera:
    print("="*60)
    print("CAMERA MODE")
    print("="*60)
    stream_url = SOURCE
    target_fps = 30.0  # Default camera FPS

print(f"Source: {stream_url}")
print(f"Target FPS: {target_fps:.1f}")

total_start = time.time()

# ---------- MULTITHREADING SETUP ----------
print("="*60)
print("MULTITHREADING MODE")
print("  Thread 1: Frame Grabber (đọc frames từ stream)")
print("  Thread 2: Object Detection (inference với YOLO)")
print("  Main Thread: Display (hiển thị kết quả)")
print(f"  Frame buffer size: {FRAME_BUFFER_SIZE}")
print(f"  Detection buffer size: {DETECTION_BUFFER_SIZE}")
if USE_BATCH_PROCESSING and device == "cuda":
    print(f"  Batch processing: ENABLED (batch_size={BATCH_SIZE})")
else:
    print(f"  Batch processing: DISABLED (batch_size=1)")
print("="*60)

# Queues cho thread-safe communication
frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)  # Thread 1 -> Thread 2
detection_queue = Queue(maxsize=DETECTION_BUFFER_SIZE)  # Thread 2 -> Main

# Flags để control threads
stop_flag = threading.Event()

# Lock để đảm bảo thread-safe khi dùng model
model_lock = threading.Lock()

# Thread 1: Frame Grabber
def frame_grabber_thread():
    """Thread 1: Đọc frames từ stream và đưa vào frame_queue"""
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("✗ Error: Cannot open video source")
        stop_flag.set()
        return
    
    # Tối ưu cho HLS streams: giảm buffer để giảm latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_id = 0
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("✗ End of stream or error reading frame")
            break
        
        frame_id += 1
        frame_time = time.time()
        
        # Đưa frame GỐC vào queue (giữ full resolution để hiển thị đẹp)
        # Pre-processing (resize) sẽ làm trong detection thread (gần GPU hơn)
        # KHÔNG drop frames để giữ timeline đúng với video gốc
        # Dùng put() KHÔNG timeout để block khi queue đầy, đảm bảo không mất frames
        # Chỉ block khi queue đầy, không drop để giữ timeline đúng
        frame_queue.put((frame_id, frame, frame_time))
    
    cap.release()
    stop_flag.set()
    print("Thread 1 (Frame Grabber) stopped")

# Thread 2: Object Detection
def detection_thread():
    """Thread 2: Lấy frames từ frame_queue, inference, đưa kết quả vào detection_queue"""
    # Thread-Safety: Dùng lock để đảm bảo thread-safe khi inference
    # Dùng model_lock để serialize các lần inference
    print("  → Detection thread: Using shared model with lock (thread-safe)")
    
    # Tối ưu GPU memory: Clear cache định kỳ (mỗi N frames)
    clear_cache_interval = 3000  # Clear cache mỗi 3000 frames (~100 giây với 30fps)
    frame_count_detection = 0
    
    # Batch processing để tăng GPU utilization
    batch_frames = []  # Tích lũy frames để batch inference (cũng là frame gốc vì không có PRE_RESIZE)
    batch_frame_ids = []  # Lưu frame_id
    batch_frame_times = []  # Lưu frame_time
    batch_start_time = None  # Thời gian bắt đầu tích lũy batch hiện tại
    # Batch timeout: Cân bằng giữa GPU utilization và latency
    # Timeout ngắn (20ms) → latency thấp, mượt mà cho livestream
    # Timeout dài (100ms) → GPU utilization cao nhưng latency cao, không phù hợp livestream
    batch_timeout = 0.05  # 50ms → cân bằng giữa latency và GPU utilization
    
    while not stop_flag.is_set():
        try:
            # Lấy frame từ queue (timeout để check stop_flag)
            frame_id, frame, frame_time = frame_queue.get(timeout=0.1)
            
            # Tích lũy frames cho batch processing (nếu bật)
            if USE_BATCH_PROCESSING and device == "cuda":
                # Khởi tạo batch_start_time khi bắt đầu batch mới
                if batch_start_time is None:
                    batch_start_time = time.time()
                
                batch_frames.append(frame)
                batch_frame_ids.append(frame_id)
                batch_frame_times.append(frame_time)
                
                # Khi đủ batch size hoặc timeout, thực hiện batch inference
                current_time = time.time()
                should_process_batch = (
                    len(batch_frames) >= BATCH_SIZE or 
                    (len(batch_frames) > 0 and (current_time - batch_start_time) >= batch_timeout)
                )
                
                if should_process_batch:
                    # Batch inference: xử lý nhiều frames cùng lúc → tăng GPU utilization
                    batch_size_processed = len(batch_frame_ids)  # Lưu số frames trước khi xử lý
                    results, inference_time, inference_end_time = run_inference(
                        batch_frames, batch_frame_ids, batch_frames, batch_frame_times
                    )
                    
                    # Xử lý results và đưa vào queue (tự động gọi task_done())
                    process_and_queue_results(
                        results, batch_frame_ids, batch_frames, batch_frame_times,
                        inference_time, inference_end_time
                    )
                    
                    # Reset batch
                    batch_frames = []
                    batch_frame_ids = []
                    batch_frame_times = []
                    batch_start_time = None
                    
                    # Tối ưu GPU memory: Clear cache định kỳ
                    # Tăng counter bằng số frames đã xử lý (không phải số results)
                    frame_count_detection += batch_size_processed
                    if frame_count_detection % clear_cache_interval == 0:
                        torch.cuda.empty_cache()
                else:
                    # Chưa đủ batch, tiếp tục tích lũy
                    # KHÔNG gọi task_done() ở đây vì frame chưa được xử lý
                    # task_done() sẽ được gọi khi batch được xử lý
                    continue
            else:
                # Single frame processing (batch=1) - cho CPU hoặc khi tắt batch processing
                results, inference_time, inference_end_time = run_inference(
                    [frame], [frame_id], [frame], [frame_time]
                )
                
                # Xử lý results và đưa vào queue (tự động gọi task_done())
                process_and_queue_results(
                    results, [frame_id], [frame], [frame_time],
                    inference_time, inference_end_time
                )
                
                # Tối ưu GPU memory: Clear cache định kỳ
                frame_count_detection += 1
                if device == "cuda" and frame_count_detection % clear_cache_interval == 0:
                    torch.cuda.empty_cache()
            
        except Empty:
            # Không có frame mới, kiểm tra xem có batch cần flush không
            if USE_BATCH_PROCESSING and device == "cuda" and len(batch_frames) > 0:
                current_time = time.time()
                should_flush = (
                    (batch_start_time is not None and (current_time - batch_start_time) >= batch_timeout) or
                    stop_flag.is_set()  # Flush ngay khi stop_flag được set
                )
                
                if should_flush:
                    # Flush batch khi timeout hoặc khi stop_flag được set
                    batch_size_processed = len(batch_frame_ids)  # Lưu số frames trước khi xử lý
                    results, inference_time, inference_end_time = run_inference(
                        batch_frames, batch_frame_ids, batch_frames, batch_frame_times
                    )
                    
                    # Xử lý results và đưa vào queue (tự động gọi task_done())
                    process_and_queue_results(
                        results, batch_frame_ids, batch_frames, batch_frame_times,
                        inference_time, inference_end_time
                    )
                    
                    # Reset batch
                    batch_frames = []
                    batch_frame_ids = []
                    batch_frame_times = []
                    batch_start_time = None
                    
                    # Tối ưu GPU memory: Clear cache định kỳ
                    # Tăng counter bằng số frames đã xử lý (không phải số results)
                    frame_count_detection += batch_size_processed
                    if frame_count_detection % clear_cache_interval == 0:
                        torch.cuda.empty_cache()
                    
                    # Nếu stop_flag được set, thoát luôn
                    if stop_flag.is_set():
                        break
            
            # Kiểm tra stop_flag trước khi continue
            if stop_flag.is_set():
                break
            continue  # Không có frame, tiếp tục chờ
        except Exception as e:
            print(f"✗ Error in detection thread: {e}")
            # Quan trọng: Gọi task_done() để tránh deadlock khi queue đầy
            try:
                frame_queue.task_done()
            except Exception:
                pass
            continue
    
    # Flush batch còn lại trước khi dừng thread
    if USE_BATCH_PROCESSING and device == "cuda" and len(batch_frames) > 0:
        print("  → Flushing remaining batch before stopping...")
        batch_size_processed = len(batch_frame_ids)  # Lưu số frames trước khi xử lý
        results, inference_time, inference_end_time = run_inference(
            batch_frames, batch_frame_ids, batch_frames, batch_frame_times
        )
        process_and_queue_results(
            results, batch_frame_ids, batch_frames, batch_frame_times,
            inference_time, inference_end_time
        )
        # Tăng counter cho batch cuối cùng
        frame_count_detection += batch_size_processed
    
    print("Thread 2 (Object Detection) stopped")

# Khởi động threads
stream_url_str = str(stream_url)
print(f"Starting inference with source: {stream_url_str[:80]}{'...' if len(stream_url_str) > 80 else ''}")

thread1 = threading.Thread(target=frame_grabber_thread, daemon=True)
thread2 = threading.Thread(target=detection_thread, daemon=True)

thread1.start()
time.sleep(0.5)  # Đợi thread1 khởi động
thread2.start()

pred_start = time.time()

# ---------- 4. Xử lý kết quả real-time với visualization ----------
total_objects = 0
frame_count = 0
fps_list = []  # Để tính FPS trung bình
frame_intervals = []  # Để tính thời gian giữa các lần hiển thị (cho FPS calculation)
display_latencies = []  # Để tính latency từ khi inference xong đến khi hiển thị

# Khởi tạo OpenCV window nếu cần hiển thị
if SHOW_VIDEO:
    cv2.namedWindow("YOLO Realtime Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Realtime Detection", 1280, 720)

# Main processing loop: Lấy kết quả từ detection_queue
# Rate limiting để sync với video FPS thực tế
target_frame_interval = 1.0 / target_fps if target_fps > 0 else 1.0 / 30.0  # Khoảng thời gian giữa các frame (giây)
prev_display_time = time.time()  # Thời gian hiển thị frame trước
next_frame_time = None  # Thời điểm hiển thị frame tiếp theo (để sync với video timeline)

while not stop_flag.is_set():
    try:
        # Lấy detection result từ queue
        # Queue chứa: (frame_id, result, frame_original, frame_time, inference_time, inference_end_time)
        frame_id, result, frame_original, frame_time, inference_time, inference_end_time = detection_queue.get(timeout=0.1)
        
        # Kiểm tra frame_original ngay từ đầu để tránh tính toán không cần thiết
        if frame_original is None:
            detection_queue.task_done()
            continue  # Bỏ qua frame không hợp lệ
        
        # Rate limiting: Đợi đến đúng thời điểm hiển thị frame tiếp theo để sync với video FPS
        # Điều này đảm bảo video chạy đúng tốc độ với video gốc trên YouTube
        current_time = time.time()
        if next_frame_time is None:
            # Frame đầu tiên: khởi tạo next_frame_time ngay tại thời điểm hiện tại
            next_frame_time = current_time
        else:
            # Đợi đến đúng thời điểm hiển thị frame tiếp theo để sync với video timeline
            # KHÔNG giới hạn sleep time để đảm bảo timeline đúng (không bỏ qua frames)
            wait_time = next_frame_time - current_time
            if wait_time > 0:
                # Sleep để sync với video FPS (không giới hạn để đảm bảo timeline đúng)
                time.sleep(wait_time)
            # Cập nhật thời điểm hiển thị frame tiếp theo dựa trên target_frame_interval
            # Dùng max() để tránh drift khi inference chậm
            next_frame_time = max(current_time, next_frame_time) + target_frame_interval
        
        # Tính latency: từ khi inference xong đến khi hiển thị
        current_display_time = time.time()
        display_latency = current_display_time - inference_end_time
        display_latencies.append(display_latency)  # Lưu latency để tính trung bình
        
        # Tính frame interval: thời gian giữa 2 lần hiển thị
        if frame_count == 0:
            frame_interval = 0
            prev_display_time = current_display_time
        else:
            frame_interval = current_display_time - prev_display_time
            prev_display_time = current_display_time
        
        # Lưu frame interval
        if frame_interval > 0:
            frame_intervals.append(frame_interval)
        
        frame_count += 1
        # Kiểm tra result trước khi truy cập thuộc tính
        num_objects = len(result.boxes) if result and hasattr(result, 'boxes') else 0
        total_objects += num_objects
        
        # Tính FPS
        current_fps = None
        if frame_interval > 0:
            current_fps = 1.0 / frame_interval
            fps_list.append(current_fps)
        
        # Tính avg_fps một lần để dùng cho cả display và print
        avg_fps_display = None
        if len(fps_list) > 0:
            avg_fps_display = sum(fps_list[-30:]) / min(30, len(fps_list))
        
        # Visualization
        if SHOW_VIDEO:
            annotated_frame = None
            
            # Nhánh 1: Dùng result.plot() nếu có
            if result and hasattr(result, 'orig_img') and result.orig_img is not None:
                try:
                    annotated_frame = result.plot()
                except Exception:
                    pass  # Fallback sang nhánh 2
            
            # Nhánh 2: Vẽ thủ công nếu result.plot() không có hoặc lỗi
            if annotated_frame is None:
                annotated_frame = frame_original.copy()
                # Vẽ boxes nếu có
                # Dùng getattr để tránh crash khi boxes=None
                if result and getattr(result, 'boxes', None) is not None and len(result.boxes) > 0:
                    try:
                        boxes_cpu = result.boxes.cpu()
                        boxes = boxes_cpu.xyxy.numpy()
                        confidences = boxes_cpu.conf.numpy()
                        classes = boxes_cpu.cls.numpy().astype(int)
                        
                        # Vẽ boxes và labels
                        for box, conf, cls in zip(boxes, confidences, classes):
                            x1, y1, x2, y2 = box.astype(int)
                            class_name = model.names[cls]
                            label = f"{class_name} {conf:.2f}"
                            
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            (text_width, text_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                            )
                            cv2.rectangle(
                                annotated_frame, 
                                (x1, y1 - text_height - baseline - 5), 
                                (x1 + text_width, y1), 
                                (0, 255, 0), -1
                            )
                            cv2.putText(
                                annotated_frame, label, (x1, y1 - baseline - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
                            )
                    except (AttributeError, IndexError):
                        pass  # Bỏ qua nếu có lỗi
            
            # Chỉ hiển thị nếu có frame hợp lệ
            if annotated_frame is not None:
                # Vẽ text lên frame nếu có FPS data
                if current_fps is not None and avg_fps_display is not None:
                    fps_text = f"FPS: {current_fps:.1f} (avg: {avg_fps_display:.1f})"
                    latency_text = f"Latency: {display_latency*1000:.1f}ms"
                    inference_text = f"Inference: {inference_time*1000:.1f}ms"
                    objects_text = f"Objects: {num_objects}"
                    
                    # Vẽ text lên frame
                    cv2.putText(annotated_frame, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(annotated_frame, latency_text, (10, 55), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(annotated_frame, inference_text, (10, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.putText(annotated_frame, objects_text, (10, 105), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                cv2.imshow("YOLO Realtime Detection", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user (pressed 'q')")
                    stop_flag.set()
                    break
        
        # Print info
        if frame_count % PRINT_EVERY_N_FRAMES == 0 or frame_count <= 5:
            if num_objects > 0 and result and getattr(result, 'boxes', None) is not None:
                boxes_cpu = result.boxes.cpu()
                conf_all = boxes_cpu.conf.numpy()
                cls_all = boxes_cpu.cls.numpy().astype(int)
                
                print(f"Frame {frame_count}: {num_objects} objects", end="")
                if num_objects <= 3:
                    for j in range(num_objects):
                        conf = float(conf_all[j])
                        cls = int(cls_all[j])
                        class_name = model.names[cls]
                        print(f" | {class_name}({conf:.2f})", end="")
                print()
            
            if len(fps_list) > 0:
                avg_frame_interval = sum(frame_intervals[-30:]) / min(30, len(frame_intervals)) * 1000 if frame_intervals else 0
                avg_display_latency = sum(display_latencies[-30:]) / min(30, len(display_latencies)) * 1000 if display_latencies else 0
                avg_fps_print = avg_fps_display if avg_fps_display is not None else sum(fps_list[-30:]) / min(30, len(fps_list))
                print(f"  → FPS: {avg_fps_print:.1f} | Frame interval: {avg_frame_interval:.1f}ms | Display latency: {avg_display_latency:.1f}ms | Inference: {inference_time*1000:.1f}ms")
        
        detection_queue.task_done()
        
    except Empty:
        continue
    except Exception as e:
        print(f"✗ Error in main loop: {e}")
        break

# Đợi threads kết thúc
stop_flag.set()

# Đợi queues empty trước khi join threads (tránh deadlock)
try:
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except Empty:
            break
    while not detection_queue.empty():
        try:
            detection_queue.get_nowait()
            detection_queue.task_done()
        except Empty:
            break
except Exception:
    pass

thread1.join(timeout=2)
thread2.join(timeout=2)

# Đóng OpenCV window
if SHOW_VIDEO:
    cv2.destroyAllWindows()

# Kết thúc đo thời gian inference
pred_end = time.time()
pred_time = pred_end - pred_start

# Giải phóng GPU memory sau khi xử lý xong
if device == "cuda":
    torch.cuda.empty_cache()

# Tính thống kê
avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
avg_frame_interval = sum(frame_intervals) / len(frame_intervals) * 1000 if frame_intervals else 0  # ms
avg_display_latency = sum(display_latencies) / len(display_latencies) * 1000 if display_latencies else 0  # ms
min_display_latency = min(display_latencies) * 1000 if display_latencies else 0  # ms
max_display_latency = max(display_latencies) * 1000 if display_latencies else 0  # ms

total_end = time.time()

print(f"\n{'='*60}")
print(f"REALTIME SUMMARY:")
print(f"  Total frames processed: {frame_count}")
print(f"  Total objects detected: {total_objects}")
print(f"  Target FPS: {target_fps:.1f}")
print(f"  Average FPS: {avg_fps:.2f}")
print(f"  Average frame interval: {avg_frame_interval:.2f}ms")
print(f"  Average display latency: {avg_display_latency:.2f}ms")
print(f"  Min display latency: {min_display_latency:.2f}ms | Max display latency: {max_display_latency:.2f}ms")
print(f"  Total inference time: {pred_time:.2f}s")
print(f"  Total script time: {total_end - total_start:.2f} seconds")
if avg_fps > 0:
    efficiency = (avg_fps / target_fps) * 100
    print(f"  Efficiency: {efficiency:.1f}% (vs target FPS)")
print(f"{'='*60}")
