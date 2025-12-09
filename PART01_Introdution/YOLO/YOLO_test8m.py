from ultralytics import YOLO
import torch
import time
import os

# ---------- 1. Load YOLOv8m ----------
model_name = "yolov8m"
print(f"Loading model '{model_name}'...")
model = YOLO(model_name)  # Tự động tải về nếu chưa có
print("✓ Model loaded successfully!")

# Compile model với PyTorch 2.x để tăng tốc
# Chỉ compile khi dùng CUDA và PyTorch >= 2.0
if torch.cuda.is_available() and hasattr(torch, 'compile'):
    try:
        print("Compiling model with torch.compile for faster inference...")
        model.model = torch.compile(model.model, mode="reduce-overhead")
        print("  → Model compiled successfully")
    except Exception as e:
        print(f"  → Compilation failed (using eager mode): {e}")

# ---------- 2. Device setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    idx = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(idx)
    gpu_memory = torch.cuda.get_device_properties(idx).total_memory / 1024**3  # GB
    print(f"GPU: {gpu_name} (index {idx}, {gpu_memory:.1f} GB VRAM)")
    
    # Tự động điều chỉnh batch size dựa trên VRAM
    if gpu_memory >= 16:
        batch_size = 16  # GPU rất mạnh (16GB+) có thể dùng batch lớn
    elif gpu_memory >= 12:
        batch_size = 12
    elif gpu_memory >= 8:
        batch_size = 8  # GPU mạnh có thể dùng batch lớn
    elif gpu_memory >= 6:
        batch_size = 4
    else:
        batch_size = 2  # GPU yếu dùng batch nhỏ
    print(f"  → Auto batch size: {batch_size}")
else:
    batch_size = 1  # CPU chỉ dùng batch=1

# ---------- 3. Video prediction ----------
video_path = "PART01_Introdution/RawData/ShortBadmintonVideo_30fps.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

total_start = time.time()
pred_start = time.time()

results = model.predict(
    source=video_path,
    conf=0.3,  # Confidence threshold (0.3 = 30%)
    iou=0.6,  # IoU threshold cho NMS (giảm từ 0.7 xuống 0.6 để tăng tốc nhẹ, vẫn giữ accuracy tốt)
    device=0 if device=="cuda" else "cpu", # dùng GPU nếu có, nếu không thì dùng CPU
    save=False, # không lưu video kết quả
    show=False, # không hiển thị video kết quả
    half=(device=="cuda"),  # FP16 để tăng tốc ~2x và tiết kiệm VRAM ~50%
    imgsz=416,  # Giảm từ 640 xuống 416 để tăng tốc (giảm accuracy nhẹ)
    batch=batch_size,  # Batch size tự động điều chỉnh theo VRAM
    verbose=True,  # Bật verbose để hiển thị progress bar
    stream=False,  # stream=False nhanh hơn cho video ngắn (load tất cả vào memory)
    agnostic_nms=False,  # Class-agnostic NMS (False = nhanh hơn, phù hợp khi classes khác nhau)
    vid_stride=2,  # Xử lý mọi frame (1 frame / 1 frame) - giữ accuracy tối đa
                   # vid_stride=2: Detect frame 1, bỏ frame 2, detect frame 3, bỏ frame 4, ...
                   # vid_stride=3: Detect frame 1, bỏ frame 2-3, detect frame 4, bỏ frame 5-6, ...
    rect=True,  # Rectangular inference - padding tối thiểu để tăng tốc (mặc định True)
    max_det=50,  # Giảm từ 300 xuống 50 (đủ cho badminton, giảm post-processing time)
    augment=False,  # Tắt TTA để tăng tốc (không cần cho inference thông thường)
    stream_buffer=False,  # Không buffer frames (tối ưu cho real-time, giảm memory)
)

pred_end = time.time()
pred_time = pred_end - pred_start

# Giải phóng GPU memory ngay sau prediction
if device == "cuda":
    torch.cuda.empty_cache()

# ---------- 4. In kết quả chi tiết  ----------
total_objects = 0
for i, result in enumerate(results):
    num_objects = len(result.boxes)
    total_objects += num_objects
    print(f"Frame {i+1}: {num_objects} objects detected")
    
    if num_objects > 0:
        # Batch process tất cả boxes cùng lúc (nhanh hơn nhiều so với từng box)
        # Lấy tất cả data cùng lúc từ GPU về CPU một lần
        boxes_cpu = result.boxes.cpu()
        xyxy_all = boxes_cpu.xyxy.numpy()  # Tất cả boxes
        conf_all = boxes_cpu.conf.numpy()  # Tất cả confidences
        cls_all = boxes_cpu.cls.numpy().astype(int)  # Tất cả classes
        
        # In từng box
        for j in range(num_objects):
            xyxy = xyxy_all[j].tolist()
            conf = float(conf_all[j])
            cls = int(cls_all[j])
            class_name = model.names[cls]
            print(f"  {class_name} (Class {cls}), Conf {conf:.2f}, Box {[round(x, 2) for x in xyxy]}")

total_end = time.time()

print(f"\n{'='*60}")
print(f"SUMMARY: {len(results)} frames, {total_objects} total objects")
print(f"Prediction time: {pred_time:.2f}s")
print(f"Speed: {len(results)/pred_time:.2f} FPS" if pred_time > 0 else "Speed: N/A")
print(f"Total script time: {total_end - total_start:.2f} seconds")
print(f"{'='*60}")
