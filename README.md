# Đồ Án Xử Lý Ảnh Số - Computer Vision & Deep Learning

Đồ án nghiên cứu và ứng dụng các kỹ thuật xử lý ảnh số (Digital Image Processing), Computer Vision và Deep Learning để xây dựng các hệ thống nhận diện và xử lý ảnh trong thời gian thực.

## 📋 Tổng Quan Dự Án

Dự án này bao gồm nhiều module nghiên cứu và ứng dụng các công nghệ:

- **Real-time Hand Detection & Tracking**: MediaPipe hand landmark detection với ESP32-CAM streaming
- **Object Detection & Tracking**: YOLO models (YOLOv8, YOLO11) với ByteTracker
- **American Sign Language (ASL) Recognition**: Nhận diện ngôn ngữ ký hiệu Mỹ
- **Hand Gesture Recognition**: Nhận diện cử chỉ tay với MediaPipe và TensorFlow
- **Sound Source Separation**: Tách nguồn âm thanh từ audio mixture
- **ESP32-CAM Integration**: Stream video qua WiFi từ ESP32-CAM

## 📁 Cấu Trúc Dự Án

```
.
├── MAIN_PROJECT/                          # Phần chính của đồ án
│   ├── esp32cam/
│   │   └── esp32cam.ino                  # ESP32-CAM firmware (ESP32-S3 + OV2640)
│   ├── UI/
│   │   ├── mediapipe_tkinter_esp32.py    # Main app với ESP32 stream + Tkinter GUI
│   │   ├── mediapipe_tkinter_template.py  # Template với webcam + Tkinter GUI
│   │   └── hand_landmarker.task           # MediaPipe hand landmarker model
│   ├── mediapipe_realtime_hand.py         # Non-GUI version (OpenCV window)
│   └── hand_landmarker.task                # MediaPipe model file
│
├── PART01_Introdution/                    # Phần giới thiệu và thử nghiệm
│   ├── mediapipe/                         # Thử nghiệm MediaPipe
│   │   ├── mediapipe_realtime_hand.py     # Hand detection cơ bản
│   │   ├── mediapipe_realtime_hand_with_pose.py  # Hand + Pose detection
│   │   ├── test_isp.py                    # Test image signal processing
│   │   ├── GESTURE_CLASSIFICATION_GUIDE.md # Hướng dẫn gesture classification
│   │   └── hand_landmarker.task           # MediaPipe model
│   │
│   ├── YOLO/                              # Thử nghiệm YOLO object detection
│   │   ├── YOLO_realtime.py               # YOLO real-time với webcam
│   │   ├── YOLO_realtime_cam.py           # YOLO với camera
│   │   ├── YOLO_realtime_cam2.py           # YOLO với camera (version 2)
│   │   ├── YOLO_realtime_hand.py          # YOLO hand detection
│   │   ├── YOLO_realtime_hand_track.py    # YOLO hand detection + tracking
│   │   ├── YOLO_realtime_track_builtin.py # YOLO với built-in tracking
│   │   ├── YOLO_realtime_yt.py            # YOLO với YouTube video
│   │   ├── YOLO_TensorRT.py               # YOLO với TensorRT optimization
│   │   ├── YOLO_test8m.py                 # Test YOLOv8m model
│   │   ├── YOLO_test11m.py                # Test YOLO11m model
│   │   ├── YOLO_test11m_real.py           # Test YOLO11m real-time
│   │   ├── menuUI.py                       # Menu UI cho YOLO
│   │   ├── GetYoutube.py                  # Download YouTube video
│   │   ├── hello.py                       # Hello world script
│   │   ├── custom_bytetrack.yaml          # ByteTracker config
│   │   ├── note.txt                       # Ghi chú về performance
│   │   └── result_box.txt                # Kết quả detection
│   │
│   ├── esp32/                             # Thử nghiệm ESP32
│   │   ├── convert_passive_buzzer.py     # Convert passive buzzer
│   │   ├── fan_test/                      # Test quạt
│   │   └── pass_buzzer_test/              # Test passive buzzer
│   │
│   └── RawData/                           # Dữ liệu mẫu
│       ├── BadmintonPic.jpg
│       ├── object.jpeg
│       ├── ShortBadmintonVideo_30fps.mp4
│       └── skeleton.jpg
│
├── DoAnPython_SignLangugeDetection_Nhom18/  # Dự án ASL Recognition
│   ├── collect_imgs.py                    # Thu thập ảnh dataset
│   ├── data_extraction.py                 # Trích xuất features từ ảnh
│   ├── model_training.py                  # Train model (Random Forest)
│   ├── real_time_prediction.py            # Real-time ASL prediction
│   ├── real_time_pre_with_tin2.py         # Real-time với TensorFlow Lite
│   ├── detectModel.py                     # Detect model
│   ├── signgame.py                        # Game ASL
│   ├── visualization.py                   # Visualization ASL
│   ├── menu.py                            # Menu GUI
│   ├── geminiAI2.py                       # Tích hợp Gemini AI
│   ├── data/                              # Dataset (A-Z, Space, Del, Nothing)
│   ├── asl_dataset_test/                  # Test dataset
│   ├── datatest.csv                       # Test data CSV
│   ├── datatest.pickle                    # Test data pickle
│   ├── modeltest.p                        # Trained model
│   └── READ_ME_PLEASE.txt                 # Hướng dẫn ASL project
│
├── hand-gesture-recognition-mediapipe-main/  # Thư viện tham khảo
│   └── hand-gesture-recognition-mediapipe-main/
│       ├── app.py                         # Hand gesture recognition app
│       ├── keypoint_classification.ipynb  # Jupyter notebook cho classification
│       ├── keypoint_classification_EN.ipynb  # English version
│       ├── point_history_classification.ipynb # Point history classification
│       ├── model/                         # Trained models
│       └── utils/                        # Utility functions
│
├── sound_source_separation/               # Tách nguồn âm thanh
│   ├── Code/                             # Code tách âm thanh
│   └── sound_mixture/                    # Audio files mẫu
│       ├── mix_violin_piano_01.wav
│       └── mix_violin_piano_02.wav
│
├── runs/                                 # Kết quả YOLO detection
│   ├── detect/predict*/                  # Video kết quả detection
│   └── pose/train/                       # Pose training results
│
├── *.pt, *.onnx                          # YOLO model files
│   ├── yolo11m.pt, yolo11m.onnx
│   ├── yolo11n.pt, yolo11n.onnx
│   ├── yolo11n-pose.pt
│   ├── yolo11s.pt
│   ├── yolov8m.pt, yolov8n.pt
│   ├── handkeypoint.pt, handkeypoint2.pt
│   └── ...
│
└── README.md                             # File này
```

## 🎯 Các Module Chính

### 1. **MAIN_PROJECT** - Real-time Hand Detection & Tracking

Module chính của đồ án, tập trung vào nhận diện và theo dõi bàn tay trong thời gian thực.

#### Tính năng:

- **MediaPipe Hand Landmarker**: Phát hiện 21 landmarks trên mỗi bàn tay
- **ESP32-CAM Streaming**: Stream video qua WiFi từ ESP32-S3 + OV2640
- **Tkinter GUI**: Giao diện người dùng với multi-threading
- **EMA Smoothing**: Làm mượt kết quả detection
- **Performance Optimization**: Frame skipping, caching, multi-threading

#### Files chính:

- `UI/mediapipe_tkinter_esp32.py`: Main app với ESP32 stream
- `UI/mediapipe_tkinter_template.py`: Template với webcam
- `mediapipe_realtime_hand.py`: Non-GUI version
- `esp32cam/esp32cam.ino`: ESP32-CAM firmware

#### Cách sử dụng:

```bash
cd MAIN_PROJECT/UI
python mediapipe_tkinter_esp32.py  # Với ESP32-CAM
python mediapipe_tkinter_template.py  # Với webcam
```

### 2. **PART01_Introdution** - Thử Nghiệm và Nghiên Cứu

Phần giới thiệu các kỹ thuật và thử nghiệm các công nghệ khác nhau.

#### 2.1. MediaPipe Experiments (`PART01_Introdution/mediapipe/`)

- **mediapipe_realtime_hand.py**: Hand detection cơ bản
- **mediapipe_realtime_hand_with_pose.py**: Hand + Pose detection
- **test_isp.py**: Test image signal processing
- **GESTURE_CLASSIFICATION_GUIDE.md**: Hướng dẫn tích hợp gesture classification với TensorFlow

#### 2.2. YOLO Object Detection (`PART01_Introdution/YOLO/`)

Thử nghiệm các model YOLO khác nhau:

- **YOLO_realtime.py**: Real-time detection với webcam
- **YOLO_realtime_cam.py, YOLO_realtime_cam2.py**: Với camera
- **YOLO_realtime_hand.py**: Hand detection
- **YOLO_realtime_hand_track.py**: Hand detection + tracking
- **YOLO_realtime_track_builtin.py**: Với built-in tracking
- **YOLO_realtime_yt.py**: Với YouTube video
- **YOLO_TensorRT.py**: TensorRT optimization
- **YOLO_test8m.py, YOLO_test11m.py**: Test các model khác nhau
- **menuUI.py**: Menu UI cho YOLO
- **GetYoutube.py**: Download YouTube video

**Models đã test:**

- YOLOv8 (yolov8m.pt, yolov8n.pt)
- YOLO11 (yolo11m.pt, yolo11n.pt, yolo11s.pt)
- YOLO11 Pose (yolo11n-pose.pt)

**Performance (theo note.txt):**

- YOLOv8m + GPU: ~270-290s cho 1615 frames (30fps video)
- Average FPS: ~12.46 FPS real-time
- Average latency: ~93.99ms

#### 2.3. ESP32 Experiments (`PART01_Introdution/esp32/`)

- **convert_passive_buzzer.py**: Convert passive buzzer
- **fan_test/**: Test quạt
- **pass_buzzer_test/**: Test passive buzzer

#### 2.4. Raw Data (`PART01_Introdution/RawData/`)

Dữ liệu mẫu để test:

- Images: BadmintonPic.jpg, object.jpeg, skeleton.jpg
- Video: ShortBadmintonVideo_30fps.mp4

### 3. **DoAnPython_SignLangugeDetection_Nhom18** - ASL Recognition

Dự án nhận diện ngôn ngữ ký hiệu Mỹ (American Sign Language).

#### Tính năng:

- **Data Collection**: Thu thập ảnh dataset (collect_imgs.py)
- **Feature Extraction**: Trích xuất features từ MediaPipe (data_extraction.py)
- **Model Training**: Train Random Forest model (model_training.py)
- **Real-time Prediction**: Nhận diện ASL real-time (real_time_prediction.py)
- **TensorFlow Lite**: Version tối ưu với TFLite (real_time_pre_with_tin2.py)
- **ASL Game**: Trò chơi ASL (signgame.py)
- **Visualization**: Hình ảnh hóa ASL (visualization.py)
- **Gemini AI Integration**: Tích hợp Gemini AI (geminiAI2.py)
- **GUI Menu**: Giao diện menu (menu.py)

#### Dataset:

- **26 chữ cái**: A-Z
- **Space, Del, Nothing**: Các ký tự đặc biệt
- **Format**: Images (.jpg) trong các folder tương ứng

#### Pipeline:

```
collect_imgs.py → data_extraction.py → model_training.py → real_time_prediction.py
```

#### Cách sử dụng:

```bash
cd DoAnPython_SignLangugeDetection_Nhom18
python menu.py  # Chạy menu chính
python collect_imgs.py  # Thu thập dữ liệu
python data_extraction.py  # Trích xuất features
python model_training.py  # Train model
python real_time_prediction.py  # Real-time prediction
```

### 4. **hand-gesture-recognition-mediapipe-main** - Thư Viện Tham Khảo

Thư viện tham khảo về hand gesture recognition với MediaPipe.

#### Nội dung:

- **app.py**: Hand gesture recognition application
- **keypoint_classification.ipynb**: Jupyter notebook cho classification
- **point_history_classification.ipynb**: Point history classification
- **model/**: Trained models (TFLite)
- **utils/**: Utility functions

### 5. **sound_source_separation** - Tách Nguồn Âm Thanh

Module nghiên cứu tách nguồn âm thanh từ audio mixture.

#### Nội dung:

- **Code/**: Code tách âm thanh
- **sound_mixture/**: Audio files mẫu (violin + piano mix)

### 6. **Model Files** (Root Directory)

Các file model YOLO và hand detection:

- **YOLO Models**:

  - `yolo11m.pt`, `yolo11m.onnx`: YOLO11 Medium
  - `yolo11n.pt`, `yolo11n.onnx`: YOLO11 Nano
  - `yolo11s.pt`: YOLO11 Small
  - `yolo11n-pose.pt`: YOLO11 Nano Pose
  - `yolov8m.pt`, `yolov8n.pt`: YOLOv8 models

- **Hand Detection Models**:
  - `handkeypoint.pt`, `handkeypoint2.pt`: Hand keypoint models

## 🚀 Cài Đặt và Sử Dụng

### Yêu Cầu Hệ Thống

**Python Dependencies:**

```bash
pip install opencv-python mediapipe pillow numpy tkinter scikit-learn tensorflow ultralytics supervision
```

**Arduino/ESP32:**

- Arduino IDE với ESP32 board support
- ESP32-S3 board (N16R8 với 8MB PSRAM)
- OV2640 camera module

### Hướng Dẫn Sử Dụng Từng Module

#### 1. MAIN_PROJECT - Hand Detection

**Bước 1: Cấu hình ESP32-CAM**

```cpp
// esp32cam/esp32cam.ino
const char *ssid = "YOUR_WIFI_SSID";
const char *password = "YOUR_WIFI_PASSWORD";
```

**Bước 2: Tải MediaPipe Model**
Tải `hand_landmarker.task` từ [MediaPipe Model Hub](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

**Bước 3: Chạy ứng dụng**

```bash
cd MAIN_PROJECT/UI
python mediapipe_tkinter_esp32.py
```

#### 2. PART01_Introdution - YOLO Detection

```bash
cd PART01_Introdution/YOLO
python YOLO_realtime.py  # Real-time với webcam
python YOLO_realtime_hand_track.py  # Hand detection + tracking
python menuUI.py  # Menu UI
```

#### 3. ASL Recognition

```bash
cd DoAnPython_SignLangugeDetection_Nhom18
python menu.py  # Menu chính
```

## 🔧 Các Kỹ Thuật Xử Lý Ảnh Số Đã Áp Dụng

### 1. **Image Preprocessing**

- Resize và crop frames
- Color space conversion (RGB, BGR, SRGB)
- Frame normalization

### 2. **Feature Extraction**

- **Hand Landmarks**: 21 điểm landmarks (x, y, z)
- **Object Detection**: Bounding boxes, class probabilities
- **Keypoint Detection**: Pose keypoints

### 3. **Signal Processing**

- **EMA Smoothing**: Exponential Moving Average
- **Frame Skipping**: Giảm CPU load
- **Temporal Filtering**: Làm mượt kết quả theo thời gian

### 4. **Deep Learning Models**

- **MediaPipe**: Hand landmarker, pose detection
- **YOLO**: Object detection (YOLOv8, YOLO11)
- **TensorFlow/Keras**: Gesture classification
- **Random Forest**: ASL classification

### 5. **Multi-threading & Performance**

- Thread-safe queues
- Frame buffering
- Image caching
- GPU acceleration (TensorRT, CUDA)

### 6. **Network Streaming**

- MJPEG protocol
- HTTP chunking
- Reconnection logic
- ESP32-CAM integration

## 📊 Thông Số Kỹ Thuật

### MAIN_PROJECT

- **Detection FPS**: ~15-25 FPS
- **Display FPS**: ~30-60 FPS
- **Latency**: ~50-100ms (end-to-end)
- **Memory**: ~200-500MB

### ESP32-CAM

- **Resolution**: VGA (640x480)
- **JPEG Quality**: 12
- **Frame Rate**: ~20-30 FPS
- **Streaming**: HTTP MJPEG port 80

### YOLO Detection

- **YOLOv8m + GPU**: ~270-290s cho 1615 frames
- **Real-time FPS**: ~12.46 FPS
- **Latency**: ~93.99ms average

## 📚 Tài Liệu Tham Khảo

- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [YOLO Ultralytics](https://docs.ultralytics.com/)
- [ESP32-CAM](https://github.com/espressif/esp32-camera)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## 📄 License

Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

**Lưu ý**: Đây là đồ án nghiên cứu về xử lý ảnh số, computer vision và deep learning. Code được tối ưu cho mục đích học tập và nghiên cứu.
