import os
import time
import cv2
import warnings
import threading
from queue import Queue, Empty, Full
import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# Giảm warning log
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['YOLO_VERBOSE'] = 'False'

# ---------- 1. Camera & hiển thị ----------
SOURCE = 0  # 0 = webcam mặc định
SHOW_VIDEO = True
PRINT_EVERY_N_FRAMES = 200

# ---------- 2. MediaPipe Hand Landmarker ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
HAND_LANDMARKER_MODEL_PATH = os.path.join(script_dir, "hand_landmarker.task")

if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy model MediaPipe: {HAND_LANDMARKER_MODEL_PATH}\n"
        f"Vui lòng tải file .task (hand_landmarker.task) về và đặt cạnh script này."
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=10,  # Tăng lên 10 để detect được nhiều người (mỗi người 2 tay)
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = HandLandmarker.create_from_options(hand_options)

# ---------- 3. YOLO Person Detection (để detect nhiều người) ----------
# Dùng YOLO để detect người (có thể detect nhiều người cùng lúc)
# Model nhẹ cho real-time: yolov8n.pt (nano) hoặc yolov8s.pt (small)
print("Loading YOLO model for person detection...")
person_model = YOLO("yolov8n.pt")  # Tự động tải về nếu chưa có
print("✓ YOLO model loaded successfully!")

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
predict_device = 0 if device == "cuda" else "cpu"
if device == "cuda":
    person_model.to("cuda")
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

# Person detection parameters
PERSON_DETECTION_PARAMS = {
    "imgsz": 640,
    "conf": 0.5,  # Confidence threshold cho person detection
    "iou": 0.45,
    "device": predict_device,
    "half": device == "cuda",
    "verbose": False,
    "max_det": 10,  # Tối đa 10 người trong frame
    "classes": [0],  # Chỉ detect class "person" (class_id = 0 trong COCO)
}

# ---------- 4. Logic Active Person Tracking ----------
# Lưu person_id của người đã làm "Start" (None = chưa có ai active)
active_person_id = None
active_person_lock = threading.Lock()

# Lưu mapping tay → người (dict: hand_idx -> person_id)
# person_id được gán dựa trên pose detection
hand_to_person_map = {}  # {hand_idx: person_id}
hand_to_person_lock = threading.Lock()

# Ngưỡng IoU để map tay vào người (hand bbox phải overlap với person bbox)
HAND_TO_PERSON_IOU_THRESHOLD = 0.1  # IoU tối thiểu để map tay vào người
# Hoặc dùng khoảng cách từ hand center đến person bbox center
HAND_TO_PERSON_DISTANCE_THRESHOLD = 300  # pixels (fallback nếu không có overlap)

# ---------- 5. Filter thresholds ----------
HAND_MIN_AREA_RATIO = 0.0025
HAND_MAX_AREA_RATIO = 0.35

# ---------- 6. Queue & threading setup ----------
FRAME_BUFFER_SIZE = 1
DETECTION_BUFFER_SIZE = 1

stream_url = SOURCE
target_fps = 30.0

print("=" * 60)
print("CAMERA MODE - MediaPipe Hand + YOLO Person (Multi-Person Support)")
print("  - Hand Landmarker: detect tay (tối đa 10 tay)")
print("  - YOLO: detect nhiều người để map tay vào người")
print("  - Active Person: chỉ xử lý gesture từ người đã 'Start'")

temp_cap = cv2.VideoCapture(stream_url)
if temp_cap.isOpened():
    detected_fps = temp_cap.get(cv2.CAP_PROP_FPS)
    temp_cap.release()
    if detected_fps and detected_fps > 1 and detected_fps < 240:
        target_fps = float(detected_fps)
        print(f"Detected camera FPS: {target_fps:.1f}")
    else:
        print("Detected camera FPS invalid (<=1 or >240). Using 30 FPS fallback.")
else:
    print("Warning: Unable to open camera for FPS detection. Using 30 FPS fallback.")

print(f"Source: {stream_url}")
print(f"Target FPS: {target_fps:.1f}")

total_start = time.time()

print("=" * 60)
print("MULTITHREADING MODE - Hand + YOLO Person (Parallel)")
print("  Thread 1: Frame Grabber")
print("  Thread 2: Hand Landmarker (song song)")
print("  Thread 3: YOLO Person Detection (song song)")
print("  Main Thread: Display + Active Person Logic")
print("=" * 60)

frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)
display_frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)
hand_detection_queue = Queue(maxsize=DETECTION_BUFFER_SIZE)
person_detection_queue = Queue(maxsize=DETECTION_BUFFER_SIZE)

stop_flag = threading.Event()
queue_drop_count = 0
queue_drop_lock = threading.Lock()


def frame_grabber_thread():
    """Thread 1: đọc frame từ camera và đưa vào queue."""
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

    try:
    cap.release()
    except Exception:
        pass
    stop_flag.set()
    print("Thread 1 (Frame Grabber) stopped")


def hand_detection_thread():
    """
    Thread 2: chạy Hand Landmarker trên frame.
    Chạy song song với YOLO thread để giảm latency.
    """
    global queue_drop_count

    print("  → Thread 2: Hand Landmarker (parallel)")

    while not stop_flag.is_set():
        try:
            frame_id, frame, frame_time = frame_queue.get(timeout=0.1)
            frame_queue.task_done()
        except Empty:
            if stop_flag.is_set():
                break
            continue

        # Chuẩn bị RGB cho MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Chạy Hand Landmarker
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int(frame_time * 1000)
        t0 = time.time()
        hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
        t1 = time.time()

        rgb_frame.flags.writeable = True

        inference_time = t1 - t0
        payload = (frame_id, hand_result, inference_time, t1)

        try:
            hand_detection_queue.put(payload, timeout=0.01)
        except Full:
            with queue_drop_lock:
                queue_drop_count += 1

    print("Thread 2 (Hand Landmarker) stopped")


def person_detection_thread():
    """
    Thread 3: chạy YOLO Person Detection trên frame.
    Chạy song song với Hand Landmarker thread để giảm latency.
    """
    global queue_drop_count

    print("  → Thread 3: YOLO Person Detection (parallel)")

    while not stop_flag.is_set():
        try:
            frame_id, frame, frame_time = frame_queue.get(timeout=0.1)
            frame_queue.task_done()
        except Empty:
            if stop_flag.is_set():
                break
            continue

        # Chạy YOLO Person Detection
        t0 = time.time()
        with torch.inference_mode():
            person_result = person_model(frame, **PERSON_DETECTION_PARAMS)[0]
        t1 = time.time()

        inference_time = t1 - t0
        payload = (frame_id, person_result, inference_time, t1)

        try:
            person_detection_queue.put(payload, timeout=0.01)
        except Full:
            with queue_drop_lock:
                queue_drop_count += 1

    print("Thread 3 (YOLO Person Detection) stopped")


def calculate_iou(box1, box2):
    """Tính IoU (Intersection over Union) giữa 2 bounding boxes.
    box1, box2: (x1, y1, x2, y2)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Tính intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def calculate_distance(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm (x, y)."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def map_hands_to_persons(hand_landmarks_list, person_boxes, frame_w, frame_h):
    """
    Map tay vào người dựa trên IoU hoặc khoảng cách từ hand bbox đến person bbox.
    YOLO có thể detect nhiều người, mỗi người có person_id riêng.
    
    Args:
        hand_landmarks_list: List các hand landmarks từ MediaPipe
        person_boxes: List các person bounding boxes từ YOLO [(x1, y1, x2, y2, conf, person_id), ...]
        frame_w, frame_h: Kích thước frame
    
    Returns:
        dict: {hand_idx: person_id} hoặc {} nếu không map được
    """
    if not hand_landmarks_list or not person_boxes or len(person_boxes) == 0:
        return {}

    mapping = {}

    # Tính hand bbox cho từng tay
    hand_boxes = []
    for hand_landmarks in hand_landmarks_list:
        if len(hand_landmarks) < 1:
            hand_boxes.append(None)
            continue

        xs = [lm.x * frame_w for lm in hand_landmarks]
        ys = [lm.y * frame_h for lm in hand_landmarks]
        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))
        hand_boxes.append((min_x, min_y, max_x, max_y))

    # Map từng tay vào người gần nhất (dựa trên IoU hoặc distance)
    for hand_idx, hand_bbox in enumerate(hand_boxes):
        if hand_bbox is None:
            continue

        best_iou = 0.0
        best_person_id = None
        best_distance = float("inf")

        # Tính center của hand bbox
        hand_center = (
            (hand_bbox[0] + hand_bbox[2]) / 2,
            (hand_bbox[1] + hand_bbox[3]) / 2,
        )

        for person_box in person_boxes:
            person_id = person_box[5]  # person_id được gán từ YOLO tracking hoặc index
            person_bbox = person_box[:4]  # (x1, y1, x2, y2)

            # Tính IoU
            iou = calculate_iou(hand_bbox, person_bbox)
            if iou > best_iou:
                best_iou = iou
                best_person_id = person_id

            # Tính khoảng cách từ hand center đến person bbox center (fallback)
            person_center = (
                (person_bbox[0] + person_bbox[2]) / 2,
                (person_bbox[1] + person_bbox[3]) / 2,
            )
            dist = calculate_distance(hand_center, person_center)
            if dist < best_distance:
                best_distance = dist

        # Map nếu IoU đủ lớn hoặc khoảng cách đủ gần
        if best_iou >= HAND_TO_PERSON_IOU_THRESHOLD:
            mapping[hand_idx] = best_person_id
        elif best_distance < HAND_TO_PERSON_DISTANCE_THRESHOLD and best_person_id is not None:
            mapping[hand_idx] = best_person_id

    return mapping


stream_url_str = str(stream_url)
print(
    f"Starting MediaPipe Hand + YOLO Person with source: {stream_url_str[:80]}{'...' if len(stream_url_str) > 80 else ''}"
)

thread1 = threading.Thread(target=frame_grabber_thread, daemon=True)
thread2 = threading.Thread(target=hand_detection_thread, daemon=True)
thread3 = threading.Thread(target=person_detection_thread, daemon=True)

thread1.start()
time.sleep(0.5)
thread2.start()
thread3.start()

pred_start = time.time()

# ---------- 7. Hiển thị real-time ----------
total_objects = 0
frame_count = 0
MAX_FPS_HISTORY = 300
fps_list = []
frame_intervals = []
display_latencies = []
inference_fps_list = []
input_fps_list = []

if SHOW_VIDEO:
    cv2.namedWindow("MediaPipe Hand + Pose (Multi-Person)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MediaPipe Hand + Pose (Multi-Person)", 1280, 720)

prev_display_time = time.time()
prev_capture_time = None

latest_hand_detection = None
latest_person_detection = None
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


def get_person_color(person_id):
    """Tạo màu ổn định từ person_id"""
    hash_val = hash(str(person_id)) % (256**3)
    r = max(100, (hash_val & 0xFF0000) >> 16)
    g = max(100, (hash_val & 0x00FF00) >> 8)
    b = max(100, hash_val & 0x0000FF)
    return (r, g, b)


def draw_keypoints(frame, keypoints, color=(0, 255, 255), radius=3, conf_threshold=0.3):
    """Vẽ keypoints lên frame"""
    if keypoints is None or len(keypoints) == 0:
        return

    frame_h, frame_w = frame.shape[:2]
    
    for kp in keypoints:
        if len(kp) >= 2:
            x, y = float(kp[0]), float(kp[1])
            conf = float(kp[2]) if len(kp) > 2 else 1.0

            # Vẽ keypoint nếu confidence đủ và tọa độ hợp lệ (>= 0 và < frame size)
            if conf >= conf_threshold and 0 <= x < frame_w and 0 <= y < frame_h:
                x, y = int(x), int(y)
                cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), radius, color, -1)


def draw_hand_skeleton(frame, keypoints, color=(0, 255, 255), thickness=1, conf_threshold=0.3):
    """Vẽ skeleton connections cho hand keypoints (21 keypoints)"""
    if keypoints is None or len(keypoints) < 21:
        return

    frame_h, frame_w = frame.shape[:2]

    wrist_to_fingers = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
    thumb_chain = [(1, 2), (2, 3), (3, 4)]
    index_chain = [(5, 6), (6, 7), (7, 8)]
    middle_chain = [(9, 10), (10, 11), (11, 12)]
    ring_chain = [(13, 14), (14, 15), (15, 16)]
    pinky_chain = [(17, 18), (18, 19), (19, 20)]

    all_connections = (
        wrist_to_fingers
        + thumb_chain
        + index_chain
        + middle_chain
        + ring_chain
        + pinky_chain
    )

    for start_idx, end_idx in all_connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            kp1 = keypoints[start_idx]
            kp2 = keypoints[end_idx]

            if len(kp1) >= 2 and len(kp2) >= 2:
                x1, y1 = float(kp1[0]), float(kp1[1])
                x2, y2 = float(kp2[0]), float(kp2[1])
                conf1 = float(kp1[2]) if len(kp1) > 2 else 1.0
                conf2 = float(kp2[2]) if len(kp2) > 2 else 1.0

                if (
                    conf1 >= conf_threshold
                    and conf2 >= conf_threshold
                    and 0 <= x1 < frame_w
                    and 0 <= y1 < frame_h
                    and 0 <= x2 < frame_w
                    and 0 <= y2 < frame_h
                ):
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(frame, pt1, pt2, color, thickness)


def draw_person_boxes(frame, person_boxes, colors_dict=None):
    """Vẽ person bounding boxes từ YOLO detection"""
    if person_boxes is None or len(person_boxes) == 0:
        return

    for person_box in person_boxes:
        x1, y1, x2, y2, conf, person_id = person_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Lấy màu cho person_id
        if colors_dict and person_id in colors_dict:
            color = colors_dict[person_id]
        else:
            color = get_person_color(person_id)

        # Vẽ bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"Person {person_id} ({conf:.2f})"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 3),
            (x1 + text_width, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )


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

        # Lấy detection mới nhất từ 2 queue riêng (hand + person chạy song song)
        hand_result = None
        person_result = None
        hand_inference_time = 0
        person_inference_time = 0
        hand_end_time = time.time()  # Khởi tạo để tránh lỗi
        person_end_time = time.time()  # Khởi tạo để tránh lỗi
        inference_end_time = time.time()
        frame_w, frame_h = frame_original.shape[1], frame_original.shape[0]

        # Lấy hand detection
        try:
            hand_data = hand_detection_queue.get_nowait()
            (
                frame_id_hand,
                hand_result,
                hand_inference_time,
                hand_end_time,
            ) = hand_data
            hand_detection_queue.task_done()
            with latest_detection_lock:
                latest_hand_detection = (hand_result, hand_inference_time, hand_end_time)
            if hand_inference_time > 0:
                inference_fps_list.append(1.0 / hand_inference_time)
                inference_fps_list = limit_list_size(inference_fps_list, MAX_FPS_HISTORY)
        except Empty:
            with latest_detection_lock:
                if latest_hand_detection is not None:
                    hand_result, hand_inference_time, hand_end_time = latest_hand_detection

        # Lấy person detection
        try:
            person_data = person_detection_queue.get_nowait()
            (
                frame_id_person,
                person_result,
                person_inference_time,
                person_end_time,
            ) = person_data
            person_detection_queue.task_done()
            with latest_detection_lock:
                latest_person_detection = (person_result, person_inference_time, person_end_time)
        except Empty:
            with latest_detection_lock:
                if latest_person_detection is not None:
                    person_result, person_inference_time, person_end_time = latest_person_detection

        # Tính inference_end_time (max của 2 thread vì chạy song song)
        inference_end_time = max(hand_end_time, person_end_time)

        # Tính tổng inference time (max của 2 thread vì chạy song song)
        inference_time = max(hand_inference_time, person_inference_time)

        # Extract person boxes từ YOLO result và map tay vào người
        person_boxes = []
        if person_result is not None and hasattr(person_result, "boxes") and len(person_result.boxes) > 0:
            # Extract boxes: (x1, y1, x2, y2, conf, person_id)
            boxes_xyxy = person_result.boxes.xyxy.cpu().numpy()  # (N, 4)
            boxes_conf = person_result.boxes.conf.cpu().numpy()  # (N,)
            
            # Gán person_id = index (tạm thời, có thể cải thiện bằng tracking ID)
            # TODO: Nếu dùng YOLO tracking (tracker="bytetrack.yaml"), có thể dùng:
            #   person_id = int(person_result.boxes.id[idx]) if hasattr(person_result.boxes, 'id') else idx
            for idx, (box, conf) in enumerate(zip(boxes_xyxy, boxes_conf)):
                person_id = idx  # Tạm thời dùng index, có thể thay bằng tracking ID
                person_boxes.append((box[0], box[1], box[2], box[3], conf, person_id))

        # Map tay vào người (mỗi frame)
        if hand_result and hand_result.hand_landmarks and person_boxes:
            new_mapping = map_hands_to_persons(
                hand_result.hand_landmarks, person_boxes, frame_w, frame_h
            )
            with hand_to_person_lock:
                hand_to_person_map = new_mapping

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

        # Số bàn tay và số người
        num_hands = 0
        num_persons = len(person_boxes)
        if hand_result and hand_result.hand_landmarks:
            num_hands = len(hand_result.hand_landmarks)

        total_objects += num_hands

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

            # Vẽ person boxes (người) trước
            if person_boxes:
                # Tạo dict màu cho từng person
                person_colors = {}
                for person_box in person_boxes:
                    person_id = person_box[5]
                    person_colors[person_id] = get_person_color(person_id)
                draw_person_boxes(annotated_frame, person_boxes, person_colors)

            # Vẽ hand landmarks + bbox
            if hand_result and hand_result.hand_landmarks:
                try:
                    with hand_to_person_lock:
                        current_mapping = hand_to_person_map.copy()

                    for hand_idx, landmarks in enumerate(hand_result.hand_landmarks):
                        # Chuyển landmarks sang pixel
                        pts = []
                        xs = []
                        ys = []
                        for lm in landmarks:
                            x = lm.x * frame_w
                            y = lm.y * frame_h
                            pts.append((x, y, 1.0))
                            xs.append(x)
                            ys.append(y)

                        pts_np = np.array(pts, dtype=np.float32)

                        # Bounding box
                        min_x, max_x = int(min(xs)), int(max(xs))
                        min_y, max_y = int(min(ys)), int(max(ys))

                        # Filter theo kích thước
                        box_w = max_x - min_x
                        box_h = max_y - min_y
                        if box_w <= 0 or box_h <= 0:
                            continue
                        box_area = box_w * box_h
                        frame_area = float(frame_w * frame_h)
                        area_ratio = box_area / frame_area if frame_area > 0 else 0.0

                        if area_ratio < HAND_MIN_AREA_RATIO or area_ratio > HAND_MAX_AREA_RATIO:
                            continue

                        # Filter handedness
                        handedness_label = "Hand"
                        handedness_score = 1.0
                        if hand_result.handedness and len(hand_result.handedness) > hand_idx:
                            cat = hand_result.handedness[hand_idx][0]
                            handedness_label = f"{cat.category_name}:{cat.score:.2f}"
                            handedness_score = float(cat.score)
                        if handedness_score < 0.6:
                            continue

                        # Lấy person_id từ mapping
                        person_id = current_mapping.get(hand_idx, None)
                        if person_id is not None:
                            person_color = get_person_color(person_id)
                            # Kiểm tra xem tay này có thuộc active person không
                            with active_person_lock:
                                is_active = active_person_id == person_id
                            # Label: Person ID + Hand ID + Left/Right + Active status
                            status_text = " [ACTIVE]" if is_active else ""
                            label = f"P{person_id} H{hand_idx} {handedness_label}{status_text}"
                            
                            # Màu đậm hơn nếu active
                            if is_active:
                                color = person_color
                                thickness = 3
                            else:
                                color = tuple(int(c * 0.5) for c in person_color)  # Màu nhạt hơn
                                thickness = 2
                        else:
                            # Không map được vào người nào
                            person_color = (128, 128, 128)  # Màu xám
                            is_active = False
                            color = person_color  # Giữ nguyên màu xám, không làm tối
                            thickness = 2
                            # Label: Hand ID + Left/Right (không có person_id)
                            label = f"H{hand_idx} {handedness_label} [UNMAPPED]"

                        cv2.rectangle(
                            annotated_frame, (min_x, min_y), (max_x, max_y), color, thickness
                        )
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            annotated_frame,
                            (min_x, min_y - text_height - baseline - 3),
                            (min_x + text_width, min_y),
                            color,
                            -1,
                        )
                        cv2.putText(
                            annotated_frame,
                            label,
                            (min_x, min_y - baseline - 1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                        # Skeleton + keypoints
                        draw_hand_skeleton(annotated_frame, pts_np, color, 1, conf_threshold=0.0)
                        draw_keypoints(annotated_frame, pts_np, color, 3, conf_threshold=0.0)

                except Exception as e:
                    print(f"⚠ Error drawing results: {e}")

            # Overlay info
            if annotated_frame is not None and current_fps is not None and avg_fps_display is not None:
                with active_person_lock:
                    active_text = f"Active Person: {active_person_id}" if active_person_id is not None else "Active Person: None"
                
                texts = [
                    (f"Target FPS: {target_fps:.1f}", (0, 200, 255)),
                    (f"Latency: {display_latency*1000:.1f}ms", (0, 255, 0)),
                    (f"Inference: {inference_time*1000:.1f}ms", (255, 255, 0)),
                    (f"Hands: {num_hands} | Persons: {num_persons}", (255, 255, 0)),
                    (active_text, (0, 255, 255) if active_person_id is not None else (128, 128, 128)),
                    (f"Input FPS: {fps_text(current_input_fps, avg_input_fps_display) if current_input_fps else '--'}", (200, 200, 0)),
                    (f"MediaPipe FPS: {fps_text(current_inference_fps, avg_inference_fps_display) if current_inference_fps else '--'}", (0, 200, 255)),
                    (f"Display FPS: {fps_text(current_fps, avg_fps_display)}", (0, 255, 0)),
                ]

                for i, (text, color) in enumerate(texts):
                    cv2.putText(
                        annotated_frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
                    )

                cv2.imshow("MediaPipe Hand + Pose (Multi-Person)", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nStopped by user (pressed 'q')")
                    stop_flag.set()
                    break
                elif key == ord("s"):
                    # Phím 's' để simulate "Start" gesture từ active person
                    # (Trong thực tế, bạn sẽ detect gesture "Start" từ ML model)
                    with hand_to_person_lock:
                        if hand_to_person_map:
                            # Lấy person_id đầu tiên có tay
                            first_person = list(hand_to_person_map.values())[0]
                            if first_person is not None:
                                with active_person_lock:
                                    active_person_id = first_person
                                print(f"  → Set active person: {first_person} (press 's' to simulate Start gesture)")
                            else:
                                print("  → No valid person mapped to hands")

        # Print info
        if frame_count % PRINT_EVERY_N_FRAMES == 0 or frame_count <= 5:
            if len(fps_list) > 0:
                avg_frame_interval = (moving_avg(frame_intervals) or 0) * 1000
                avg_display_latency = (moving_avg(display_latencies) or 0) * 1000
                avg_fps_print = avg_fps_display or moving_avg(fps_list) or 0
                avg_inference_fps_print = moving_avg(inference_fps_list) or 0
                avg_input_fps_print = moving_avg(input_fps_list) or 0
                print(
                    f"  → Average Display FPS: {avg_fps_print:.1f} | "
                    f"Average MediaPipe FPS: {avg_inference_fps_print:.1f} | "
                    f"Average Input FPS: {avg_input_fps_print:.1f} | "
                    f"Target FPS: {target_fps:.1f} | "
                    f"Frame interval: {avg_frame_interval:.1f}ms | "
                    f"Display latency: {avg_display_latency:.1f}ms | "
                    f"Inference: {inference_time*1000:.1f}ms"
                )

    except Empty:
        continue
    except Exception as e:
        print(f"✗ Error in main loop: {e}")
        break

# Đợi threads kết thúc
stop_flag.set()

# Queue cleanup: dùng get_nowait() với Empty exception thay vì empty() (không thread-safe)
try:
    while True:
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except Empty:
            break
    while True:
        try:
            display_frame_queue.get_nowait()
            display_frame_queue.task_done()
        except Empty:
            break
    while True:
        try:
            hand_detection_queue.get_nowait()
            hand_detection_queue.task_done()
        except Empty:
            break
    while True:
        try:
            person_detection_queue.get_nowait()
            person_detection_queue.task_done()
        except Empty:
            break
except Exception:
    pass

if thread1.is_alive():
    thread1.join(timeout=2)
if thread2.is_alive():
    thread2.join(timeout=2)
if thread3.is_alive():
    thread3.join(timeout=2)

# Đóng landmarker để giải phóng tài nguyên
try:
    hand_landmarker.close()
except Exception:
    pass

if SHOW_VIDEO:
    cv2.destroyAllWindows()

pred_end = time.time()
pred_time = pred_end - pred_start

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
print(f"REALTIME SUMMARY - MEDIAPIPE HAND + YOLO PERSON (MULTI-PERSON):")
print(f"  Backend: MediaPipe Hand Landmarker + YOLO Person Detection")
print(f"  Total frames processed: {frame_count}")
print(f"  Total hands detected: {total_objects}")
print(f"  Target FPS: {target_fps:.1f}")
print(f"  Average Display FPS: {avg_fps:.2f}")
print(f"  Average MediaPipe FPS: {avg_inference_fps:.2f}")
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
