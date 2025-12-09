from ultralytics import YOLO
import supervision as sv
import torch
import time
import os
import cv2
import warnings
import threading
from queue import Queue, Empty, Full
import numpy as np

# Suppress warnings để giảm I/O overhead
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['YOLO_VERBOSE'] = 'False'

# ---------- Global image size ----------
imgsz = 640  # Dùng chung cho model, resize, warm-up

# ---------- 1. Load YOLO Model ----------
use_tensorrt = True  # Bật TensorRT để tăng tốc (cần NVIDIA GPU + CUDA)

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
        use_tensorrt = False
    elif gpu_memory < 6.0:
        print(f"⚠ Detected mid-range GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        print("  → Recommending YOLO11m for balanced performance")
        model_name = "yolo11m"
    else:
        model_name = "yolo11l"
else:
    model_name = "yolo11n" # CPU mode

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
                imgsz=imgsz,  # Dùng global imgsz
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
    # Giúp tránh latency spike ở inference đầu tiên
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
    "agnostic_nms": True,  # Class-agnostic NMS (True = NMS across classes)
    "verbose": False, # Bật/tắt verbose mode (True = hiển thị progress bar)
    "rect": True,  # Rectangular training
    "max_det": 30,  # Giới hạn số detection tối đa mỗi frame
    "augment": False
}

# --- Safe predict wrapper: single image inference với model(img) API
def safe_predict_single_image(img_bgr):
    """
    Trả về Ultralytics Results object (single) hoặc None.
    Gọi model(img, ...) và xử lý OOM.
    Dùng torch.inference_mode() để giảm overhead.
    """
    def _predict():
        with torch.inference_mode():
            return model(img_bgr, **INFERENCE_PARAMS)[0]
    
    try:
        return _predict()
    except RuntimeError as e:
        err = str(e).lower()
        if "out of memory" in err:
            print("✗ CUDA OOM during inference — clearing cache and retrying")
            try:
                torch.cuda.empty_cache()
                return _predict()
            except Exception as e2:
                print(f"✗ Retry failed: {e2}")
                return None
        else:
            print(f"✗ Predict runtime error: {e}")
            return None
    except Exception as e:
        print(f"✗ Unexpected predict error: {e}")
        return None

# --- Helper function: Convert YOLO results to supervision Detections format
def results_to_supervision_detections(result):
    """
    Convert Ultralytics YOLO result sang supervision Detections object
    Trả về: supervision Detections object hoặc Detections.empty()
    """
    if result is None or not hasattr(result, "boxes") or len(result.boxes) == 0:
        # Trả về empty Detections
        return sv.Detections.empty()
    try:
        # Supervision Detections cần: xyxy, confidence, class_id
        boxes = result.boxes.xyxy.cpu().numpy()  # (N,4) [x1,y1,x2,y2]
        confidence = result.boxes.conf.cpu().numpy()  # (N,)
        class_id = result.boxes.cls.cpu().numpy().astype(np.int32)  # (N,)
        
        # Tạo Detections object từ arrays
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidence,
            class_id=class_id
        )
        return detections
    except Exception as e:
        print(f"Error converting result to supervision Detections: {e}")
        return sv.Detections.empty()

# --- Inference function: Single frame mode (realtime)
def run_inference(frame):
    """
    Chạy inference cho single frame (realtime mode)
    Trả về: (result, inference_time, inference_end_time)
    result là Ultralytics Results object hoặc None
    """
    if frame is None:
        return None, 0, time.time()
    
    inference_start = time.time()
    
    with model_lock:
        result = safe_predict_single_image(frame)
    
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start
    
    return result, inference_time, inference_end_time

# ---------- 3. Source setup ----------
# Camera/webcam
SOURCE = 0  # Camera index (0 = default camera, 1 = second camera, etc.)

# Cấu hình visualization
SHOW_VIDEO = True  # Bật/tắt hiển thị video (True = xem real-time, False = chỉ xử lý)
PRINT_EVERY_N_FRAMES = 200  # Chỉ in thông tin mỗi N frames (giảm I/O overhead, tăng FPS)

# Cấu hình tracker
USE_TRACKER = True  # Bật/tắt tracker (True = dùng ByteTrack để track objects, False = chỉ detection không có track ID)

# ---------- Queue Configuration (Realtime) ----------
# Queue size = 1 để luôn giữ frame mới nhất, tránh backlog và latency tích lũy
FRAME_BUFFER_SIZE = 1  # Queue size = 1 (overwrite latest) - latency thấp cho realtime
DETECTION_BUFFER_SIZE = 1  # Queue size = 1 (overwrite latest) - luôn dùng detection mới nhất

# Xử lý camera source
stream_url = SOURCE
target_fps = 30.0  # Default FPS (FPS lý thuyết/mong đợi từ camera - có thể không khớp với thực tế)

print("="*60)
print("CAMERA MODE")
print("="*60)
# Ước lượng FPS thực tế từ thiết bị camera
# Mở camera tạm thời để đọc FPS, sau đó đóng lại (thread chính sẽ mở lại)
temp_cap = cv2.VideoCapture(stream_url)
if temp_cap.isOpened():
    detected_fps = temp_cap.get(cv2.CAP_PROP_FPS)
    temp_cap.release()
    # Validate FPS: phải > 1 và < 240 (giới hạn hợp lý cho camera)
    if detected_fps and detected_fps > 1 and detected_fps < 240:
        target_fps = float(detected_fps)
        print(f"Detected camera FPS: {target_fps:.1f}")
    else:
        target_fps = 30.0
        print("Detected camera FPS invalid (<=1 or >240). Using 30 FPS fallback.")
else:
    target_fps = 30.0
    print("Warning: Unable to open camera for FPS detection. Using 30 FPS fallback.")

print(f"Source: {stream_url}")
print(f"Target FPS: {target_fps:.1f}")

total_start = time.time()

# ---------- MULTITHREADING SETUP ----------
print("="*60)
print("MULTITHREADING MODE")
print("  Thread 1: Frame Grabber (đọc frames từ stream)")
print("  Thread 2: Object Detection (inference với YOLO)")
print("  Main Thread: Display (hiển thị kết quả)")
print(f"  Frame buffer size: {FRAME_BUFFER_SIZE} (queue size=1 for low latency)")
print(f"  Detection buffer size: {DETECTION_BUFFER_SIZE} (queue size=1 for latest detection)")
print("="*60)

# Queues cho thread-safe communication
frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)  # Thread A → Thread B (inference)
display_frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)  # Thread A → Display thread (30 FPS)
detection_queue = Queue(maxsize=DETECTION_BUFFER_SIZE)  # Thread B → Display thread (detections)

stop_flag = threading.Event()
model_lock = threading.Lock()

# ---------- Tracker Setup ----------
TRACKER_CONFIG = {
    "track_activation_threshold": 0.6,      # Confidence threshold để tạo track mới (tăng lên để khó tạo track mới, giữ ID ổn định hơn)
    "lost_track_buffer": 150,                # Số frames giữ track khi mất detection (tăng để giữ track lâu hơn khi object ra khỏi khung hình)
    "minimum_matching_threshold": 0.7,      # IoU threshold để match tracks (tăng lên để match chặt chẽ hơn, tránh đổi ID)
    "frame_rate": 30.0                      # Sẽ được cập nhật sau khi biết target_fps
}

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
    
    # Tối ưu cho camera: giảm buffer để giảm latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_id = 0
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("✗ End of stream or error reading frame")
            break
        
        frame_id += 1
        frame_time = time.time()
        
        # Dùng chung frame cho inference, chỉ copy một lần cho display để giảm overhead
        frame_for_display = frame.copy()
        
        # Đẩy frame vào 2 queue: một cho Thread B (inference), một cho Display thread
        # timeout=0.01: non-blocking, nếu queue đầy thì bỏ qua frame này (queue size=1 nên luôn có frame mới nhất)
        try:
            frame_queue.put((frame_id, frame, frame_time), timeout=0.01)
        except Full:
            # Queue đầy = detection thread đang xử lý, frame này sẽ bị drop (đã có frame mới hơn trong queue)
            with queue_drop_lock:
                queue_drop_count += 1
        
        try:
            display_frame_queue.put((frame_id, frame_for_display, frame_time), timeout=0.01)
        except Full:
            # Display queue đầy, bỏ qua (đã có frame mới hơn trong queue, display thread sẽ lấy frame mới nhất)
            pass
    
    cap.release()
    stop_flag.set()
    print("Thread 1 (Frame Grabber) stopped")

# Thread 2: Object Detection (Single Frame Mode)
def detection_thread():
    """
    Thread 2: Lấy frames từ frame_queue, inference với YOLO (single-frame mode)
    - YOLO xử lý được bao nhiêu frame thì xử lý bấy nhiêu, luôn trên frame mới nhất trong queue
    - Đẩy detection vào detection_queue cho Display thread
    """
    global queue_drop_count
    
    print("  → Detection thread: Single frame mode (low latency for realtime)")
    
    clear_cache_interval = 1000
    frame_count_detection = 0
    
    while not stop_flag.is_set():
        try:
            # Lấy frame từ frame_queue (chỉ Thread B đọc từ queue này)
            frame_id, frame, frame_time = frame_queue.get(timeout=0.1)
            frame_queue.task_done()
            
            # Single frame inference
            result, inference_time, inference_end_time = run_inference(frame)
            
            # Đẩy detection vào detection_queue (không cần frame, Display thread đã có từ display_frame_queue)
            payload = (frame_id, result, inference_time, inference_end_time)
            try:
                detection_queue.put(payload, timeout=0.01)
            except Full:
                with queue_drop_lock:
                    queue_drop_count += 1
            
            frame_count_detection += 1
            
            if device == "cuda" and frame_count_detection % clear_cache_interval == 0:
                torch.cuda.empty_cache()
            
        except Empty:
            if stop_flag.is_set():
                break
            continue
        except Exception as e:
            print(f"✗ Error in detection thread: {e}")
            try:
                frame_queue.task_done()
            except Exception:
                pass
            continue
    
    print("Thread 2 (Object Detection) stopped")

# Khởi động threads
print(f"Starting inference with camera: {stream_url}")

thread1 = threading.Thread(target=frame_grabber_thread, daemon=True)
thread2 = threading.Thread(target=detection_thread, daemon=True)

thread1.start()
time.sleep(0.5)  # Đợi thread1 khởi động
thread2.start()

pred_start = time.time()

# ---------- 4. Xử lý kết quả real-time với visualization ----------
total_objects = 0
frame_count = 0
# Giới hạn kích thước các list để tránh memory leak (chỉ giữ N giá trị gần nhất)
MAX_FPS_HISTORY = 300  # Giữ 300 giá trị
fps_list = []  # Để tính FPS trung bình
frame_intervals = []  # Để tính thời gian giữa các lần hiển thị (cho FPS calculation)
display_latencies = []  # Để tính latency từ khi inference xong đến khi hiển thị
inference_fps_list = []  # Theo dõi YOLO FPS (1 / inference_time)
input_fps_list = []  # Theo dõi FPS đọc từ source (thời gian giữa các frame_time)

# Clear GPU cache định kỳ trong main loop (nếu processing dài)
CLEAR_CACHE_INTERVAL = 1000  # Clear cache mỗi 1000 frames

# Khởi tạo OpenCV window nếu cần hiển thị
if SHOW_VIDEO:
    cv2.namedWindow("YOLO Realtime Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Realtime Detection", 1280, 720)

# ---------- Initialize Tracker ----------
tracker = None
if USE_TRACKER:
    TRACKER_CONFIG["frame_rate"] = target_fps
    try:
        tracker = sv.ByteTrack(**TRACKER_CONFIG)
        print("✓ Supervision ByteTrack initialized successfully")
    except Exception as e:
        print(f"⚠ Failed to initialize ByteTrack: {e}")
        print("  → Running without tracker (detection only)")
        tracker = None
else:
    print("  → Tracker disabled (USE_TRACKER = False) - running in detection-only mode")

# ---------- Display Thread (Main Thread) với Tracker ----------
# Display thread hiển thị frame càng nhanh càng tốt để video chạy đúng tốc độ
prev_display_time = time.time()
prev_capture_time = None

# Lưu detection mới nhất để tracker update
latest_detection = None
latest_detection_lock = threading.Lock()

# Helper functions cho visualization
def limit_list_size(data_list, max_size):
    """Giới hạn kích thước list, chỉ giữ N giá trị gần nhất"""
    if len(data_list) > max_size:
        return data_list[-max_size:]
    return data_list

def fps_text(val, avg=None):
    """Format FPS text (định nghĩa một lần, dùng lại nhiều lần)"""
    return f"{val:.1f} (avg: {avg:.1f})" if avg is not None else f"{val:.1f}"

def moving_avg(data_list, window=30):
    """
    Tính trung bình trượt (moving average) của N giá trị gần nhất trong danh sách.
    
    Args:
        data_list: Danh sách các giá trị cần tính trung bình (ví dụ: FPS values)
        window: Số lượng giá trị gần nhất để tính trung bình (mặc định 30 frames)
    
    Returns:
        Trung bình của window giá trị gần nhất, hoặc None nếu danh sách rỗng
    
    Ví dụ:
        fps_list = [25, 26, 24, 27, 25, 28, ...]
        moving_avg(fps_list, window=30)  # Tính trung bình của 30 giá trị FPS gần nhất
        # → Làm mượt số liệu, phản ánh xu hướng gần đây thay vì toàn bộ lịch sử
    """
    if not data_list:
        return None
    # Lấy window giá trị gần nhất (từ cuối danh sách) và tính trung bình
    return sum(data_list[-window:]) / min(window, len(data_list))

def get_track_color(track_id):
    """Tạo màu ổn định từ track_id (hash-based)"""
    hash_val = hash(str(track_id)) % (256**3)
    r = max(100, (hash_val & 0xFF0000) >> 16)
    g = max(100, (hash_val & 0x00FF00) >> 8)
    b = max(100, hash_val & 0x0000FF)
    return (r, g, b)

def draw_box_with_label(frame, x1, y1, x2, y2, label, color, font_scale=0.5):
    """Vẽ bounding box với label"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(frame, (x1, y1 - text_height - baseline - 3), (x1 + text_width, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

while not stop_flag.is_set():
    try:
        # Lấy frame mới nhất từ display_frame_queue (hiển thị càng nhanh càng tốt)
        # Bỏ qua frames cũ trong queue để giảm latency
        frame_id, frame_original, frame_time = None, None, None
        
        # Bỏ qua tất cả frames cũ, chỉ lấy frame mới nhất
        # Strategy: lấy hết frames trong queue (get_nowait), frame cuối cùng là frame mới nhất
        try:
            while True:
                frame_id, frame_original, frame_time = display_frame_queue.get_nowait()
                display_frame_queue.task_done()
        except Empty:
            # Không còn frame nào trong queue, đợi frame mới với timeout ngắn
            if frame_original is None:
                try:
                    frame_id, frame_original, frame_time = display_frame_queue.get(timeout=0.01)
                    display_frame_queue.task_done()
                except Empty:
                    continue  # Queue vẫn rỗng, quay lại vòng lặp chính
        
        if frame_original is None:
            continue
        
        # Check detection_queue non-blocking để lấy detection mới nhất (nếu có)
        result = None
        inference_time = 0
        inference_end_time = time.time()
        
        try:
            # Lấy detection mới nhất từ queue (non-blocking)
            detection_data = detection_queue.get_nowait()
            frame_id_det, result, inference_time, inference_end_time = detection_data
            detection_queue.task_done()
            # Lưu detection mới nhất để dùng cho frame tiếp theo nếu không có detection mới
            with latest_detection_lock:
                latest_detection = (result, inference_time, inference_end_time)
            if inference_time > 0:
                inference_fps_list.append(1.0 / inference_time)
                inference_fps_list = limit_list_size(inference_fps_list, MAX_FPS_HISTORY)
        except Empty:
            # Không có detection mới trong queue, dùng detection cũ đã lưu (đảm bảo luôn có detection để hiển thị)
            with latest_detection_lock:
                if latest_detection is not None:
                    result, inference_time, inference_end_time = latest_detection
        
        # Đo Input FPS thực tế: FPS thực sự đọc được từ source (khác với Target FPS lý thuyết)
        # Input FPS = 1 / (thời gian giữa 2 frames liên tiếp)
        # Ví dụ: Target FPS = 30 (camera báo), nhưng Input FPS = 28 (thực tế đọc được)
        if prev_capture_time is not None:
            capture_interval = frame_time - prev_capture_time
            if capture_interval > 0:
                input_fps_list.append(1.0 / capture_interval)
                input_fps_list = limit_list_size(input_fps_list, MAX_FPS_HISTORY)
        prev_capture_time = frame_time
        
        # Update tracker với detection mới nhất
        # Đảm bảo tracker luôn nhận Detections object (kể cả khi rỗng) để tránh lỗi NoneType
        tracks = None
        if tracker is not None:
            try:
                # Luôn tạo Detections object, kể cả khi không có detection
                if result is not None:
                    detections = results_to_supervision_detections(result)
                else:
                    detections = sv.Detections.empty()
                
                # ByteTrack sẽ tự động predict/interpolate với empty detections nếu không có detection
                tracks = tracker.update_with_detections(detections)
            except Exception as e:
                print(f"⚠ Tracker update error: {e}")
                tracks = None
        
        # Tính latency: từ khi inference xong đến khi hiển thị
        current_display_time = time.time()
        display_latency = current_display_time - inference_end_time
        display_latencies.append(display_latency)  # Lưu latency để tính trung bình
        display_latencies = limit_list_size(display_latencies, MAX_FPS_HISTORY)
        
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
            frame_intervals = limit_list_size(frame_intervals, MAX_FPS_HISTORY)
        
        frame_count += 1
        # Tính số objects từ tracks (đã được ByteTrack xử lý) thay vì từ detections thô
        # Tracks phản ánh số objects đang được track chính xác hơn
        if tracks is not None and len(tracks) > 0:
            num_objects = len(tracks)
        elif result and hasattr(result, 'boxes') and len(result.boxes) > 0:
            # Fallback: nếu không có tracks, dùng detections thô
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
        
        # Tính trung bình các FPS metrics bằng moving average (30 frames gần nhất)
        # Moving average giúp làm mượt số liệu và phản ánh xu hướng gần đây
        current_inference_fps = 1.0 / inference_time if inference_time > 0 else None
        avg_fps_display = moving_avg(fps_list)  # Trung bình Display FPS (30 frames gần nhất)
        avg_inference_fps_display = moving_avg(inference_fps_list)  # Trung bình YOLO FPS (30 frames gần nhất)
        avg_input_fps_display = moving_avg(input_fps_list)  # Trung bình Input FPS (30 frames gần nhất)
        current_input_fps = input_fps_list[-1] if input_fps_list else None
        
        # Visualization
        if SHOW_VIDEO:
            annotated_frame = frame_original.copy()
            
            # Strategy: Ưu tiên vẽ tracks (có track ID) để hiển thị ID ổn định
            # ByteTrack tự động merge detection mới với track prediction
            # Nếu có detection mới → dùng detection (chính xác hơn)
            # Nếu không có detection → dùng track prediction (có thể lệch khi object di chuyển nhanh)
            if tracks is not None and len(tracks) > 0:
                try:
                    boxes = tracks.xyxy
                    tracker_ids = tracks.tracker_id
                    confidences = tracks.confidence
                    class_ids = tracks.class_id
                    
                    for i in range(len(boxes)):
                        try:
                            x1, y1, x2, y2 = map(int, boxes[i])
                            track_id = int(tracker_ids[i]) if tracker_ids[i] is not None else -1
                            score = float(confidences[i])
                            cls_id = int(class_ids[i])
                            
                            color = get_track_color(track_id)
                            class_name = model.names.get(cls_id, 'unknown')
                            label = f"ID:{track_id} {class_name} {score:.2f}"
                            draw_box_with_label(annotated_frame, x1, y1, x2, y2, label, color, 0.5)
                        except Exception as e:
                            pass  # Bỏ qua track lỗi
                except Exception as e:
                    print(f"⚠ Error drawing tracks: {e}")
            
            # Fallback: nếu không có tracks, vẽ detections thô từ YOLO (không có track ID)
            elif result and getattr(result, 'boxes', None) is not None and len(result.boxes) > 0:
                try:
                    # Chuyển boxes từ GPU về CPU trước khi convert sang numpy
                    boxes_cpu = result.boxes.cpu()
                    boxes = boxes_cpu.xyxy.numpy()
                    confidences = boxes_cpu.conf.numpy()
                    classes = boxes_cpu.cls.numpy().astype(int)
                    
                    # Vẽ boxes và labels (màu xanh lá cây cho detections không có track)
                    for box, conf, cls in zip(boxes, confidences, classes):
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = model.names[cls]
                        label = f"{class_name} {conf:.2f}"
                        draw_box_with_label(annotated_frame, x1, y1, x2, y2, label, (0, 255, 0), 0.6)
                except (AttributeError, IndexError):
                    pass  # Bỏ qua nếu có lỗi
            
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
                # Tính trung bình bằng moving average (30 frames gần nhất) cho các metrics
                # Chuyển đổi từ giây sang milliseconds (* 1000) cho frame interval và latency
                avg_frame_interval = (moving_avg(frame_intervals) or 0) * 1000
                avg_display_latency = (moving_avg(display_latencies) or 0) * 1000
                avg_fps_print = avg_fps_display or moving_avg(fps_list) or 0  # Trung bình Display FPS
                avg_inference_fps_print = moving_avg(inference_fps_list) or 0  # Trung bình YOLO FPS
                avg_input_fps_print = moving_avg(input_fps_list) or 0  # Trung bình Input FPS
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

# Đợi queues empty trước khi join threads
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
avg_inference_fps = sum(inference_fps_list) / len(inference_fps_list) if inference_fps_list else 0
avg_input_fps = sum(input_fps_list) / len(input_fps_list) if input_fps_list else 0

total_end = time.time()

print(f"\n{'='*60}")
print(f"REALTIME SUMMARY:")
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
