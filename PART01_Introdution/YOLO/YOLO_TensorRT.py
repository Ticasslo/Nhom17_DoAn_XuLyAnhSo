from ultralytics import YOLO
import torch
import time
import os

# ---------- 1. Kiểm tra device ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
if device == 'cuda':
    current_device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device_index)
    print(f"Using CUDA GPU: {device_name} (index {current_device_index})")
else:
    print("CUDA not available. Using CPU.")

# ---------- 2. Load YOLO model ----------
pt_path = "yolov8m.pt"
model = YOLO(pt_path)

# ---------- 3. Export sang TensorRT engine ----------
engine_path = "yolov8m.engine"
if not os.path.exists(engine_path):
    print("Export YOLOv8m sang TensorRT engine (FP16)...")
    model.export(format="engine", imgsz=640, half=True)  # tạo yolov8m.engine
    print("Export completed!")
else:
    print("TensorRT engine đã tồn tại, bỏ qua export.")

# ---------- 4. Load TensorRT engine ----------
model = YOLO(engine_path)

# ---------- 5. Predict video ----------
video_source = 'PART01_Introdution/RawData/ShortBadmintonVideo_30fps.mp4'

total_start = time.time()
pred_start = time.time()

results = model.predict(
    source=video_source,
    conf=0.3,
    device=0 if device=='cuda' else 'cpu',
    save=True,   # lưu video kết quả
    show=False   # True nếu muốn hiển thị realtime
)

pred_end = time.time()
pred_time = pred_end - pred_start

# ---------- 6. In bounding boxes ----------
print("Results summary:", results)
for box in results[0].boxes:
    print(box)

# ---------- 7. In total time ----------
print(f"Prediction time: {pred_time:.2f} seconds")
print(f"Total script time: {time.time() - total_start:.2f} seconds")
