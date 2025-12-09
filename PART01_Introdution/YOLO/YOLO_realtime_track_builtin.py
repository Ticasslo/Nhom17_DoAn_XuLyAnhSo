from ultralytics import YOLO
import torch
import time
import os
import cv2
import warnings
import threading
from queue import Queue, Empty, Full
import yt_dlp
import numpy as np

# Suppress warnings để giảm I/O overhead
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['YOLO_VERBOSE'] = 'False'

# ---------- Global image size ----------
imgsz = 640  # Dùng chung cho model, resize, warm-up

# ---------- 1. Load YOLO Model ----------
# Kiểm tra GPU trước để chọn model phù hợp
gpu_info = None
if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    gpu_memory = torch.cuda.get_device_properties(idx).total_memory / 1024**3  # GB
    gpu_name = torch.cuda.get_device_name(idx)
    gpu_info = {"idx": idx, "name": gpu_name, "memory": gpu_memory} 
    
    # Tự động chọn model dựa trên VRAM
    if gpu_memory < 4.0:
        print(f"⚠ Detected low-end GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        print("  → Recommending YOLO11s for better performance")
        model_name = "yolo11s"
    elif gpu_memory < 6.0:
        print(f"⚠ Detected mid-range GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        print("  → Recommending YOLO11m for balanced performance")
        model_name = "yolo11m"
    else:
        model_name = "yolo11l"
else:
    model_name = "yolo11n" # CPU mode

print(f"Loading model '{model_name}'...")
model = YOLO(model_name)  # Tự động tải về nếu chưa có
print("✓ PyTorch model loaded successfully!")

# Compile model với PyTorch 2.x để tăng tốc
if torch.cuda.is_available() and hasattr(torch, 'compile'):
    try:
        print("Compiling model with torch.compile...")
        model.model = torch.compile(model.model, mode="reduce-overhead")
        print("  → Model compiled successfully")
    except Exception as e:
        print(f"  → Compilation failed (using eager mode): {e}")

# Đảm bảo model ở eval mode một lần sau khi load/compile
if hasattr(model, "model"):
    try:
        model.model.eval()
        print("  → Model set to eval()")
    except Exception as e:
        print(f"  → Warning: failed to set model to eval(): {e}")

# ---------- 2. Device setup & Realtime Config ----------
# Device setup - khai báo trước khi dùng
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Setup device cho model
if device == "cuda":
    try:
        model.to("cuda")
    except Exception:
        pass
    
    if gpu_info is not None:
        idx = gpu_info["idx"]
        gpu_name = gpu_info["name"]
        gpu_memory = gpu_info["memory"]
        print(f"GPU: {gpu_name} (index {idx}, {gpu_memory:.1f} GB VRAM)")
    
    # Clear cache ban đầu để có memory sạch
    torch.cuda.empty_cache()
    
    # Pre-allocate một dummy tensor để khởi tạo CUDA context (warm-up)
    print("  → Warming up GPU...")
    dummy_tensor = torch.zeros(1, 3, imgsz, imgsz, device=device)
    if hasattr(torch.cuda, 'synchronize'):
        torch.cuda.synchronize()
    del dummy_tensor
    torch.cuda.empty_cache()
    print("  → GPU warmed up!")

# --- Canonical predict device for Ultralytics ---
predict_device = 0 if device == "cuda" else "cpu"

# --- Inference parameters (dùng chung cho tất cả inference calls) ---
INFERENCE_PARAMS = {
    "imgsz": imgsz,
    "conf": 0.4,  # Confidence threshold
    "iou": 0.45,   # IoU threshold
    "device": predict_device,
    "half": device == "cuda",  # FP16 trên GPU để tăng tốc
    "agnostic_nms": True,  # Class-agnostic NMS
    "verbose": False,
    "rect": True,
    "max_det": 30,
    "augment": False
}

# --- Safe track wrapper: single image tracking với model.track() API
def safe_track_single_image(img_bgr, persist=True):
    """
    Trả về Ultralytics Results object (single) hoặc None.
    Dùng model.track() với persist=True để giữ track giữa các frame.
    Gọi model.track(img, ...) và xử lý OOM.
    Dùng torch.inference_mode() để giảm overhead.
    """
    def _track():
        with torch.inference_mode():
            # Dùng model.track() với persist=True và tracker YAML config
            # tracker=TRACKER_YAML sẽ dùng custom config thay vì mặc định
            return model.track(img_bgr, persist=persist, tracker=TRACKER_YAML, **INFERENCE_PARAMS)[0]
    
    try:
        return _track()
    except RuntimeError as e:
        err = str(e).lower()
        if "out of memory" in err:
            print("✗ CUDA OOM during tracking — clearing cache and retrying")
            try:
                torch.cuda.empty_cache()
                return _track()
            except Exception as e2:
                print(f"✗ Retry failed: {e2}")
                return None
        else:
            print(f"✗ Track runtime error: {e}")
            return None
    except Exception as e:
        print(f"✗ Unexpected track error: {e}")
        return None

# --- Inference function: Single frame mode (realtime) với tracking
def run_tracking(frame):
    """
    Chạy tracking cho single frame (realtime mode) - dùng YOLO built-in tracking
    Trả về: (result, inference_time, inference_end_time)
    result là Ultralytics Results object hoặc None
    """
    if frame is None:
        return None, 0, time.time()
    
    inference_start = time.time()
    
    with model_lock:
        result = safe_track_single_image(frame, persist=True)
    
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start
    
    return result, inference_time, inference_end_time

# ---------- 3. Source setup ----------
# Có thể dùng: YouTube URL hoặc Camera (0)
# SOURCE = "https://www.youtube.com/watch?v=cH7VBI4QQzA"  # YouTube livestream URL
SOURCE = 0  # Camera/webcam

# Cấu hình visualization
SHOW_VIDEO = True  # Bật/tắt hiển thị video (True = xem real-time, False = chỉ xử lý)
PRINT_EVERY_N_FRAMES = 200  # Chỉ in thông tin mỗi N frames (giảm I/O overhead, tăng FPS)

# Cấu hình tracker YAML
# Có thể dùng: "bytetrack.yaml" (mặc định), "botsort.yaml", hoặc custom YAML file
# File YAML phải nằm trong thư mục ultralytics/cfg/trackers/ hoặc cùng thư mục với script
TRACKER_YAML_NAME = "custom_bytetrack.yaml"  # Tên file YAML

# Tìm file YAML: ưu tiên cùng thư mục với script, sau đó là ultralytics/cfg/trackers/
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path_local = os.path.join(script_dir, TRACKER_YAML_NAME)
yaml_path_ultralytics = None

# Thử tìm trong ultralytics/cfg/trackers/
try:
    from ultralytics import __file__ as ultralytics_init_file
    ultralytics_dir = os.path.dirname(ultralytics_init_file)
    trackers_dir = os.path.join(ultralytics_dir, "cfg", "trackers")
    yaml_path_ultralytics = os.path.join(trackers_dir, TRACKER_YAML_NAME)
except Exception:
    pass

# Chọn đường dẫn YAML
# YOLO có thể chỉ tìm trong ultralytics/cfg/trackers/, nên ưu tiên copy vào đó hoặc dùng tên file
if os.path.exists(yaml_path_local):
    # Nếu file tồn tại trong cùng thư mục, thử copy vào ultralytics/cfg/trackers/ nếu có thể
    if yaml_path_ultralytics:
        try:
            # Tạo thư mục nếu chưa có
            os.makedirs(os.path.dirname(yaml_path_ultralytics), exist_ok=True)
            # Copy file vào ultralytics/cfg/trackers/ nếu chưa có
            if not os.path.exists(yaml_path_ultralytics):
                import shutil
                shutil.copy2(yaml_path_local, yaml_path_ultralytics)
                print(f"  → Copied tracker YAML to: {yaml_path_ultralytics}")
            TRACKER_YAML = TRACKER_YAML_NAME  # Dùng tên file, YOLO sẽ tìm trong ultralytics/cfg/trackers/
        except Exception as e:
            # Nếu không copy được, dùng đường dẫn tuyệt đối
            TRACKER_YAML = yaml_path_local
            print(f"  → Using tracker YAML from script directory: {yaml_path_local}")
    else:
        TRACKER_YAML = yaml_path_local
        print(f"  → Using tracker YAML from script directory: {yaml_path_local}")
elif yaml_path_ultralytics and os.path.exists(yaml_path_ultralytics):
    TRACKER_YAML = TRACKER_YAML_NAME  # Dùng tên file, YOLO sẽ tìm trong ultralytics/cfg/trackers/
    print(f"  → Found tracker YAML in ultralytics: {yaml_path_ultralytics}")
else:
    # Fallback: dùng mặc định
    TRACKER_YAML = "bytetrack.yaml"
    print(f"  ⚠ Custom tracker YAML not found, using default: {TRACKER_YAML}")
    print(f"     (Searched: {yaml_path_local})")
    if yaml_path_ultralytics:
        print(f"     (Searched: {yaml_path_ultralytics})")

# ---------- Queue Configuration (Realtime) ----------
# Queue size = 1 để luôn giữ frame mới nhất, tránh backlog và latency tích lũy
FRAME_BUFFER_SIZE = 1  # Queue size = 1 (overwrite latest) - latency thấp cho realtime
DETECTION_BUFFER_SIZE = 1  # Queue size = 1 (overwrite latest) - luôn dùng detection mới nhất

# Xử lý source: kiểm tra xem là YouTube URL hay camera
is_youtube = isinstance(SOURCE, str) and ("youtube.com" in SOURCE or "youtu.be" in SOURCE)
is_camera = isinstance(SOURCE, int) or SOURCE == 0

stream_url = SOURCE
target_fps = 30.0  # Default FPS (FPS lý thuyết/mong đợi từ source - có thể không khớp với thực tế)

if is_youtube:
    print("="*60)
    print("YOUTUBE LIVESTREAM MODE - YOLO BUILT-IN TRACKING")
    print(f"  Tracker config: {TRACKER_YAML_NAME}")
    if isinstance(TRACKER_YAML, str) and os.path.exists(TRACKER_YAML):
        print(f"  → Using custom YAML: {TRACKER_YAML}")
    elif TRACKER_YAML == "bytetrack.yaml":
        print(f"  → Using default tracker (custom YAML not found)")
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
                    
                    # Kiểm tra protocol: HLS (m3u8) thường có latency cao hơn direct stream
                    protocol = f.get("protocol", "").lower()
                    is_hls = "m3u8" in protocol or "dash" in protocol or "hls" in protocol
                    # Chỉ lấy format có video (bỏ qua audio-only)
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

elif is_camera:
    print("="*60)
    print("CAMERA MODE - YOLO BUILT-IN TRACKING")
    print(f"  Tracker config: {TRACKER_YAML_NAME}")
    if isinstance(TRACKER_YAML, str) and os.path.exists(TRACKER_YAML):
        print(f"  → Using custom YAML: {TRACKER_YAML}")
    elif TRACKER_YAML == "bytetrack.yaml":
        print(f"  → Using default tracker (custom YAML not found)")
    print("="*60)
    stream_url = SOURCE
    # Ước lượng FPS thực tế từ thiết bị camera
    temp_cap = cv2.VideoCapture(stream_url)
    if temp_cap.isOpened():
        detected_fps = temp_cap.get(cv2.CAP_PROP_FPS)
        temp_cap.release()
        if detected_fps and detected_fps > 1 and detected_fps < 240:
            target_fps = float(detected_fps)
            print(f"Detected camera FPS: {target_fps:.1f}")
        else:
            target_fps = 30.0
            print("Detected camera FPS invalid (<=1 or >240). Using 30 FPS fallback.")
    else:
        target_fps = 30.0
        print("Warning: Unable to open camera for FPS detection. Using 30 FPS fallback.")
else:
    raise ValueError(f"Unsupported source type: {SOURCE}. Please use YouTube URL or camera (0).")

print(f"Source: {stream_url}")
print(f"Target FPS: {target_fps:.1f}")

total_start = time.time()

# ---------- MULTITHREADING SETUP ----------
print("="*60)
print("MULTITHREADING MODE - YOLO BUILT-IN TRACKING")
print("  Thread 1: Frame Grabber (đọc frames từ stream)")
print("  Thread 2: Object Tracking (tracking với YOLO built-in)")
print("  Main Thread: Display (hiển thị kết quả)")
print(f"  Frame buffer size: {FRAME_BUFFER_SIZE} (queue size=1 for low latency)")
print(f"  Detection buffer size: {DETECTION_BUFFER_SIZE} (queue size=1 for latest detection)")
print("="*60)

# Queues cho thread-safe communication
frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)  # Thread A → Thread B (tracking)
display_frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)  # Thread A → Display thread
detection_queue = Queue(maxsize=DETECTION_BUFFER_SIZE)  # Thread B → Display thread (tracking results)

stop_flag = threading.Event()
model_lock = threading.Lock()

# Metrics tracking
queue_drop_count = 0
queue_drop_lock = threading.Lock()

# Thread 1: Frame Grabber
def frame_grabber_thread():
    """Thread 1: Đọc frames từ stream và đưa vào frame_queue"""
    global queue_drop_count
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("✗ Error: Cannot open video source")
        stop_flag.set()
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_id = 0
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("✗ End of stream or error reading frame")
            break
        
        frame_id += 1
        frame_time = time.time()
        
        frame_for_display = frame.copy()
        
        try:
            frame_queue.put((frame_id, frame, frame_time), timeout=0.01)
        except Full:
            with queue_drop_lock:
                queue_drop_count += 1
        
        try:
            display_frame_queue.put((frame_id, frame_for_display, frame_time), timeout=0.01)
        except Full:
            pass
    
    cap.release()
    stop_flag.set()
    print("Thread 1 (Frame Grabber) stopped")

# Thread 2: Object Tracking (dùng YOLO built-in tracking)
def tracking_thread():
    """
    Thread 2: Lấy frames từ frame_queue, tracking với YOLO built-in (model.track())
    - YOLO xử lý được bao nhiêu frame thì xử lý bấy nhiêu, luôn trên frame mới nhất trong queue
    - Đẩy tracking results vào detection_queue cho Display thread
    """
    global queue_drop_count
    
    print("  → Tracking thread: YOLO built-in tracking (model.track() with persist=True)")
    
    clear_cache_interval = 1000
    frame_count_tracking = 0
    
    while not stop_flag.is_set():
        try:
            frame_id, frame, frame_time = frame_queue.get(timeout=0.1)
            frame_queue.task_done()
            
            # Single frame tracking với YOLO built-in
            result, inference_time, inference_end_time = run_tracking(frame)
            
            # Đẩy tracking results vào detection_queue
            payload = (frame_id, result, inference_time, inference_end_time)
            try:
                detection_queue.put(payload, timeout=0.01)
            except Full:
                with queue_drop_lock:
                    queue_drop_count += 1
            
            frame_count_tracking += 1
            
            if device == "cuda" and frame_count_tracking % clear_cache_interval == 0:
                torch.cuda.empty_cache()
            
        except Empty:
            if stop_flag.is_set():
                break
            continue
        except Exception as e:
            print(f"✗ Error in tracking thread: {e}")
            try:
                frame_queue.task_done()
            except Exception:
                pass
            continue
    
    print("Thread 2 (Object Tracking) stopped")

# Khởi động threads
stream_url_str = str(stream_url)
print(f"Starting tracking with source: {stream_url_str[:80]}{'...' if len(stream_url_str) > 80 else ''}")

thread1 = threading.Thread(target=frame_grabber_thread, daemon=True)
thread2 = threading.Thread(target=tracking_thread, daemon=True)

thread1.start()
time.sleep(0.5)
thread2.start()

pred_start = time.time()

# ---------- 4. Xử lý kết quả real-time với visualization ----------
total_objects = 0
frame_count = 0
MAX_FPS_HISTORY = 300
fps_list = []
frame_intervals = []
display_latencies = []
inference_fps_list = []
input_fps_list = []

CLEAR_CACHE_INTERVAL = 1000

if SHOW_VIDEO:
    cv2.namedWindow("YOLO Built-in Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Built-in Tracking", 1280, 720)

# ---------- Display Thread (Main Thread) ----------
prev_display_time = time.time()
prev_capture_time = None

latest_detection = None
latest_detection_lock = threading.Lock()

# Helper functions
def limit_list_size(data_list, max_size):
    """Giới hạn kích thước list, chỉ giữ N giá trị gần nhất"""
    if len(data_list) > max_size:
        return data_list[-max_size:]
    return data_list

def fps_text(val, avg=None):
    """Format FPS text"""
    return f"{val:.1f} (avg: {avg:.1f})" if avg is not None else f"{val:.1f}"

def moving_avg(data_list, window=30):
    """Tính trung bình trượt (moving average)"""
    if not data_list:
        return None
    return sum(data_list[-window:]) / min(window, len(data_list))

def get_track_color(track_id):
    """Tạo màu ổn định từ track_id"""
    hash_val = hash(str(track_id)) % (256**3)
    r = max(100, (hash_val & 0xFF0000) >> 16)
    g = max(100, (hash_val & 0x00FF00) >> 8)
    b = max(100, hash_val & 0x0000FF)
    return (r, g, b)

while not stop_flag.is_set():
    try:
        # Lấy frame mới nhất từ display_frame_queue
        frame_id, frame_original, frame_time = None, None, None
        
        try:
            while True:
                frame_id, frame_original, frame_time = display_frame_queue.get_nowait()
                display_frame_queue.task_done()
        except Empty:
            if frame_original is None:
                try:
                    frame_id, frame_original, frame_time = display_frame_queue.get(timeout=0.01)
                    display_frame_queue.task_done()
                except Empty:
                    continue
        
        if frame_original is None:
            continue
        
        # Check detection_queue non-blocking để lấy tracking result mới nhất
        result = None
        inference_time = 0
        inference_end_time = time.time()
        
        try:
            detection_data = detection_queue.get_nowait()
            frame_id_det, result, inference_time, inference_end_time = detection_data
            detection_queue.task_done()
            with latest_detection_lock:
                latest_detection = (result, inference_time, inference_end_time)
            if inference_time > 0:
                inference_fps_list.append(1.0 / inference_time)
                inference_fps_list = limit_list_size(inference_fps_list, MAX_FPS_HISTORY)
        except Empty:
            with latest_detection_lock:
                if latest_detection is not None:
                    result, inference_time, inference_end_time = latest_detection
        
        # Đo Input FPS thực tế
        if prev_capture_time is not None:
            capture_interval = frame_time - prev_capture_time
            if capture_interval > 0:
                input_fps_list.append(1.0 / capture_interval)
                input_fps_list = limit_list_size(input_fps_list, MAX_FPS_HISTORY)
        prev_capture_time = frame_time
        
        # Tính latency
        current_display_time = time.time()
        display_latency = current_display_time - inference_end_time
        display_latencies.append(display_latency)
        display_latencies = limit_list_size(display_latencies, MAX_FPS_HISTORY)
        
        # Tính frame interval
        if frame_count == 0:
            frame_interval = 0
            prev_display_time = current_display_time
        else:
            frame_interval = current_display_time - prev_display_time
            prev_display_time = current_display_time
        
        if frame_interval > 0:
            frame_intervals.append(frame_interval)
            frame_intervals = limit_list_size(frame_intervals, MAX_FPS_HISTORY)
        
        frame_count += 1
        
        # Tính số objects từ tracking results (YOLO built-in tracking)
        if result and hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            # Kiểm tra xem có track IDs không
            if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                num_objects = len(result.boxes.id)
            else:
                num_objects = len(result.boxes)
        else:
            num_objects = 0
        total_objects += num_objects
        
        # Tính FPS hiển thị
        current_fps = None
        if frame_interval > 0:
            current_fps = 1.0 / frame_interval
            fps_list.append(current_fps)
            fps_list = limit_list_size(fps_list, MAX_FPS_HISTORY)
        
        # Tính trung bình các FPS metrics
        current_inference_fps = 1.0 / inference_time if inference_time > 0 else None
        avg_fps_display = moving_avg(fps_list)
        avg_inference_fps_display = moving_avg(inference_fps_list)
        avg_input_fps_display = moving_avg(input_fps_list)
        current_input_fps = input_fps_list[-1] if input_fps_list else None
        
        # Visualization
        if SHOW_VIDEO:
            annotated_frame = frame_original.copy()
            
            # Vẽ tracking results từ YOLO built-in tracking
            if result and hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                try:
                    boxes_cpu = result.boxes.cpu()
                    boxes = boxes_cpu.xyxy.numpy()
                    confidences = boxes_cpu.conf.numpy()
                    classes = boxes_cpu.cls.numpy().astype(int)
                    
                    # Lấy track IDs nếu có (YOLO built-in tracking)
                    track_ids = None
                    if hasattr(boxes_cpu, 'id') and boxes_cpu.id is not None:
                        track_ids = boxes_cpu.id.numpy()
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = model.names[cls]
                        
                        # Nếu có track ID, hiển thị ID
                        if track_ids is not None and i < len(track_ids):
                            track_id = int(track_ids[i]) if track_ids[i] is not None else -1
                            color = get_track_color(track_id)
                            label = f"ID:{track_id} {class_name} {conf:.2f}"
                        else:
                            color = (0, 255, 0)
                            label = f"{class_name} {conf:.2f}"
                        
                        # Vẽ box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline - 3), (x1 + text_width, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"⚠ Error drawing tracking results: {e}")
            
            if annotated_frame is not None and current_fps is not None and avg_fps_display is not None:
                texts = [
                    (f"Target FPS: {target_fps:.1f}", (0, 200, 255)),
                    (f"Latency: {display_latency*1000:.1f}ms", (0, 255, 0)),
                    (f"Inference: {inference_time*1000:.1f}ms", (255, 255, 0)),
                    (f"Objects: {num_objects}", (255, 255, 0)),
                    (f"Input FPS: {fps_text(current_input_fps, avg_input_fps_display) if current_input_fps else '--'}", (200, 200, 0)),
                    (f"YOLO FPS: {fps_text(current_inference_fps, avg_inference_fps_display) if current_inference_fps else '--'}", (0, 200, 255)),
                    (f"Display FPS: {fps_text(current_fps, avg_fps_display)}", (0, 255, 0))
                ]
                
                for i, (text, color) in enumerate(texts):
                    cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                cv2.imshow("YOLO Built-in Tracking", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user (pressed 'q')")
                    stop_flag.set()
                    break
        
        # Print info
        if frame_count % PRINT_EVERY_N_FRAMES == 0 or frame_count <= 5:
            if num_objects > 0 and result and hasattr(result, 'boxes') and len(result.boxes) > 0:
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
                avg_frame_interval = (moving_avg(frame_intervals) or 0) * 1000
                avg_display_latency = (moving_avg(display_latencies) or 0) * 1000
                avg_fps_print = avg_fps_display or moving_avg(fps_list) or 0
                avg_inference_fps_print = moving_avg(inference_fps_list) or 0
                avg_input_fps_print = moving_avg(input_fps_list) or 0
                print(f"  → Average Display FPS: {avg_fps_print:.1f} | Average YOLO FPS: {avg_inference_fps_print:.1f} | Average Input FPS: {avg_input_fps_print:.1f} | Target FPS: {target_fps:.1f} | Frame interval: {avg_frame_interval:.1f}ms | Display latency: {avg_display_latency:.1f}ms | Inference: {inference_time*1000:.1f}ms")
        
        # Clear GPU cache định kỳ
        if device == "cuda" and frame_count % CLEAR_CACHE_INTERVAL == 0 and frame_count > 0:
            torch.cuda.empty_cache()
        
    except Empty:
        continue
    except Exception as e:
        print(f"✗ Error in main loop: {e}")
        break

# Đợi threads kết thúc
stop_flag.set()

try:
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except Empty:
            break
    while not display_frame_queue.empty():
        try:
            display_frame_queue.get_nowait()
            display_frame_queue.task_done()
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

if thread1.is_alive():
    thread1.join(timeout=2)
if thread2.is_alive():
    thread2.join(timeout=2)

if SHOW_VIDEO:
    cv2.destroyAllWindows()

pred_end = time.time()
pred_time = pred_end - pred_start

if device == "cuda":
    torch.cuda.empty_cache()

# Tính thống kê
avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
avg_frame_interval = sum(frame_intervals) / len(frame_intervals) * 1000 if frame_intervals else 0
avg_display_latency = sum(display_latencies) / len(display_latencies) * 1000 if display_latencies else 0
min_display_latency = min(display_latencies) * 1000 if display_latencies else 0
max_display_latency = max(display_latencies) * 1000 if display_latencies else 0
avg_inference_fps = sum(inference_fps_list) / len(inference_fps_list) if inference_fps_list else 0
avg_input_fps = sum(input_fps_list) / len(input_fps_list) if input_fps_list else 0

total_end = time.time()

print(f"\n{'='*60}")
print(f"REALTIME SUMMARY - YOLO BUILT-IN TRACKING:")
print(f"  Total frames processed: {frame_count}")
print(f"  Total objects detected: {total_objects}")
print(f"  Target FPS: {target_fps:.1f}")
print(f"  Average Display FPS: {avg_fps:.2f}")
print(f"  Average YOLO FPS: {avg_inference_fps:.2f}")
print(f"  Average Input FPS: {avg_input_fps:.2f}")
print(f"  Average frame interval: {avg_frame_interval:.2f}ms")
print(f"  Average display latency: {avg_display_latency:.2f}ms")
print(f"  Min display latency: {min_display_latency:.2f}ms | Max display latency: {max_display_latency:.2f}ms")
print(f"  Total inference time: {pred_time:.2f}s")
print(f"  Total script time: {total_end - total_start:.2f} seconds")
with queue_drop_lock:
    print(f"  Queue drops (frames): {queue_drop_count}")
if avg_fps > 0:
    efficiency = (avg_fps / target_fps) * 100
    print(f"  Efficiency: {efficiency:.1f}% (vs target FPS)")
print(f"{'='*60}")

