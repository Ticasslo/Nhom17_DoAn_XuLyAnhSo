# Hướng Dẫn Tích Hợp Gesture Classification với TensorFlow vào MediaPipe Hand Detection

## Tổng Quan

Hướng dẫn này giải thích cách tích hợp gesture classification **sử dụng TensorFlow/Keras** vào file `mediapipe_realtime_hand.py` để nhận diện cử chỉ tay (ví dụ: "A", "B", "Thumbs Up", "OK", etc.)

## Cách Hoạt Động

### 1. Pipeline Tổng Quan

```
Camera → MediaPipe Hand Detection → Extract 42 Features → TensorFlow Model → Gesture Label
```

### 2. Feature Extraction

- **Input**: 21 landmarks từ MediaPipe (mỗi landmark có x, y normalized 0-1)
- **Output**: 42 features (21 x 2 = x, y cho mỗi landmark)
- **Format**: `[x0, y0, x1, y1, x2, y2, ..., x20, y20]` (numpy array, float32)

### 3. TensorFlow Classification Model

- **Model types**:
  - **TensorFlow/Keras** (`.h5` hoặc SavedModel) - cho training và development
  - **TensorFlow Lite** (`.tflite`) - cho production, real-time (khuyến nghị)
- **Input**: 42 features (numpy array, shape: (1, 42) hoặc (42,))
- **Output**: Class probabilities (softmax) → Class index → Class label

## Các Bước Tích Hợp

### Bước 1: Train TensorFlow Model

#### 1.1. Extract Features từ Dataset

Dùng `data_extraction.py` để extract 42 features từ images:

- Input: Images với hand gestures
- Output: CSV hoặc numpy array với 42 columns (features) + 1 label column
- Format: `[x0, y0, x1, y1, ..., x20, y20, label]`

#### 1.2. Train TensorFlow/Keras Model

**File: `train_tensorflow_gesture_model.py`**

```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_pickle('datatest.csv')  # Hoặc CSV
X = data.drop(columns=['label']).values.astype(np.float32)
y = data['label'].values

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
NUM_CLASSES = len(label_encoder.classes_)

# Save label encoder for later use
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Build TensorFlow/Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,), name='input_layer'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation='relu', name='dense_1'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='output_layer')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save Keras model
model.save('gesture_model.h5')
print('✓ Saved: gesture_model.h5')

# Convert to TensorFlow Lite (khuyến nghị cho real-time)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)
print('✓ Saved: gesture_model.tflite')

# Optional: Quantize để giảm size và tăng tốc
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized = converter.convert()
with open('gesture_model_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized)
print('✓ Saved: gesture_model_quantized.tflite (optimized)')
```

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
X = ... # 42 features
y = ... # labels (encoded as integers)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Save model
model.save('gesture_model.h5')  # Keras format
# Hoặc convert sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

3. **Model formats**:
   - **`.h5`**: Keras format (dùng `tf.keras.models.load_model()`)
   - **`.tflite`**: TensorFlow Lite (nhẹ, nhanh, khuyến nghị cho real-time)
   - **SavedModel**: TensorFlow format (`.pb`)

### Bước 2: Load Model trong Code

**Option 1: TensorFlow Lite (Khuyến nghị - nhanh nhất)**

```python
import tensorflow as tf
import pickle

# Load TFLite model
GESTURE_MODEL_PATH = "gesture_model.tflite"
LABEL_ENCODER_PATH = "label_encoder.pkl"  # Để map class ID → label name

gesture_interpreter = tf.lite.Interpreter(model_path=GESTURE_MODEL_PATH)
gesture_interpreter.allocate_tensors()
gesture_input_details = gesture_interpreter.get_input_details()
gesture_output_details = gesture_interpreter.get_output_details()

# Load label encoder để map class ID → label name
try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    GESTURE_CLASS_NAMES = label_encoder.classes_  # Array of class names
    print(f"✓ Loaded {len(GESTURE_CLASS_NAMES)} gesture classes")
except Exception as e:
    print(f"⚠ Label encoder not found: {e}")
    GESTURE_CLASS_NAMES = None
```

**Option 2: Keras Model (.h5)**

```python
import tensorflow as tf
import pickle

# Load Keras model
GESTURE_MODEL_PATH = "gesture_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

gesture_model = tf.keras.models.load_model(GESTURE_MODEL_PATH)

# Load label encoder
try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    GESTURE_CLASS_NAMES = label_encoder.classes_
    print(f"✓ Loaded {len(GESTURE_CLASS_NAMES)} gesture classes")
except Exception as e:
    print(f"⚠ Label encoder not found: {e}")
    GESTURE_CLASS_NAMES = None
```

### Bước 2: Extract Features từ Landmarks

- **Vị trí**: Sau khi có `landmarks` từ MediaPipe (sau line 590)
- **Cách làm**:
  - Lặp qua 21 landmarks
  - Lấy x, y (normalized, không cần convert sang pixel)
  - Append vào list: `[x0, y0, x1, y1, ..., x20, y20]`
- **Lưu ý**: Dùng normalized coordinates (0-1), không phải pixel coordinates

### Bước 3: Predict Gesture với TensorFlow

**Option 1: TensorFlow Lite (Khuyến nghị)**

```python
# Prepare input (shape: (1, 42))
features_array = np.array([features], dtype=np.float32)

# Set input tensor
gesture_interpreter.set_tensor(gesture_input_details[0]['index'], features_array)

# Run inference
gesture_interpreter.invoke()

# Get output probabilities
output_data = gesture_interpreter.get_tensor(gesture_output_details[0]['index'])
probabilities = output_data[0]  # Shape: (NUM_CLASSES,)

# Get predicted class
class_id = np.argmax(probabilities)
confidence = probabilities[class_id] * 100

# Map class ID to label name
if GESTURE_CLASS_NAMES is not None and class_id < len(GESTURE_CLASS_NAMES):
    gesture_label = GESTURE_CLASS_NAMES[class_id]
else:
    gesture_label = f"Class_{class_id}"
```

**Option 2: Keras Model**

```python
# Prepare input (shape: (1, 42))
features_array = np.array([features], dtype=np.float32)

# Predict (verbose=0 để không in log)
predictions = gesture_model.predict(features_array, verbose=0)
probabilities = predictions[0]  # Shape: (NUM_CLASSES,)

# Get predicted class
class_id = np.argmax(probabilities)
confidence = probabilities[class_id] * 100

# Map class ID to label name
if GESTURE_CLASS_NAMES is not None and class_id < len(GESTURE_CLASS_NAMES):
    gesture_label = GESTURE_CLASS_NAMES[class_id]
else:
    gesture_label = f"Class_{class_id}"
```

### Bước 4: Hiển Thị Kết Quả

- **Vị trí**: Trong visualization loop (sau line 639, trước khi vẽ label)
- **Cách làm**:
  - Thêm gesture label vào text hiển thị
  - Hiển thị confidence score
  - Có thể thêm màu sắc khác nhau cho từng gesture

## Best Practices

### 1. Feature Extraction

- ✅ **Dùng normalized coordinates** (0-1) từ MediaPipe
- ✅ **Áp dụng EMA smoothing** trước khi extract (đã có trong code)
- ✅ **Validate features**: Đảm bảo có đủ 42 features
- ❌ **Không dùng pixel coordinates**: Sẽ không scale được

### 2. Model Performance

- ✅ **Dùng TensorFlow Lite**: Nhẹ, nhanh nhất cho real-time (< 5ms)
- ✅ **Cache model**: Load 1 lần, dùng nhiều lần
- ✅ **Batch prediction**: Có thể predict nhiều hands cùng lúc
- ✅ **Optimize model**: Quantization, pruning để giảm size và tăng tốc
- ❌ **Tránh model quá sâu**: Dense layers đơn giản đủ cho gesture classification

### 3. Preprocessing

- ✅ **EMA smoothing**: Giảm jitter (đã có)
- ✅ **Filter low confidence**: Bỏ qua detection không chắc chắn
- ✅ **Normalize**: Đảm bảo features trong range [0, 1]

### 4. Error Handling

- ✅ **Try-except**: Bọc prediction trong try-except
- ✅ **Check model loaded**: Đảm bảo model đã load trước khi predict
- ✅ **Validate features**: Check length = 42 trước khi predict

## Vị Trí Tích Hợp Trong Code

### 1. Import & Load Model (đầu file, sau imports)

**Option 1: TensorFlow Lite (Khuyến nghị)**

```python
# Thêm vào imports
import tensorflow as tf
import pickle

# Load TFLite model (sau khi setup MediaPipe, trước threading)
GESTURE_MODEL_PATH = "path/to/gesture_model.tflite"
LABEL_ENCODER_PATH = "path/to/label_encoder.pkl"

gesture_interpreter = None
gesture_input_details = None
gesture_output_details = None
GESTURE_CLASS_NAMES = None

try:
    # Load TFLite model
    gesture_interpreter = tf.lite.Interpreter(model_path=GESTURE_MODEL_PATH)
    gesture_interpreter.allocate_tensors()
    gesture_input_details = gesture_interpreter.get_input_details()
    gesture_output_details = gesture_interpreter.get_output_details()

    # Load label encoder để map class ID → label name
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
        GESTURE_CLASS_NAMES = label_encoder.classes_

    print(f"✓ TensorFlow Lite gesture model loaded ({len(GESTURE_CLASS_NAMES)} classes)")
except Exception as e:
    print(f"⚠ Gesture model not loaded: {e}")
```

**Option 2: Keras Model (.h5)**

```python
# Thêm vào imports
import tensorflow as tf
import pickle

# Load Keras model
GESTURE_MODEL_PATH = "path/to/gesture_model.h5"
LABEL_ENCODER_PATH = "path/to/label_encoder.pkl"

gesture_model = None
GESTURE_CLASS_NAMES = None

try:
    # Load Keras model
    gesture_model = tf.keras.models.load_model(GESTURE_MODEL_PATH)

    # Load label encoder
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
        GESTURE_CLASS_NAMES = label_encoder.classes_

    print(f"✓ Keras gesture model loaded ({len(GESTURE_CLASS_NAMES)} classes)")
except Exception as e:
    print(f"⚠ Gesture model not loaded: {e}")
```

### 2. Extract Features Function (sau helper functions)

```python
def extract_gesture_features(landmarks):
    """
    Extract 42 features từ MediaPipe landmarks

    Args:
        landmarks: MediaPipe landmarks (list of 21 landmark objects)

    Returns:
        features: List of 42 values [x0, y0, x1, y1, ..., x20, y20]
    """
    features = []
    for lm in landmarks:
        features.append(lm.x)  # Normalized x (0-1)
        features.append(lm.y)  # Normalized y (0-1)
    return features
```

### 3. Predict Gesture (trong visualization loop, sau line 594)

**Option 1: TensorFlow Lite**

```python
# Sau khi có landmarks (sau EMA smoothing)
# Extract features từ original landmarks (normalized)
gesture_features = extract_gesture_features(landmarks)

# Predict gesture với TFLite
gesture_label = "Unknown"
gesture_confidence = 0.0
if gesture_interpreter is not None and len(gesture_features) == 42:
    try:
        # Prepare input (shape: (1, 42))
        features_array = np.array([gesture_features], dtype=np.float32)

        # Set input tensor
        gesture_interpreter.set_tensor(gesture_input_details[0]['index'], features_array)

        # Run inference
        gesture_interpreter.invoke()

        # Get output probabilities
        output_data = gesture_interpreter.get_tensor(gesture_output_details[0]['index'])
        probabilities = output_data[0]  # Shape: (NUM_CLASSES,)

        # Get predicted class
        class_id = np.argmax(probabilities)
        gesture_confidence = probabilities[class_id] * 100

        # Map class ID to label name
        if GESTURE_CLASS_NAMES is not None and class_id < len(GESTURE_CLASS_NAMES):
            gesture_label = GESTURE_CLASS_NAMES[class_id]
        else:
            gesture_label = f"Class_{class_id}"
    except Exception as e:
        print(f"⚠ Gesture prediction error: {e}")
```

**Option 2: Keras Model**

```python
# Sau khi có landmarks (sau EMA smoothing)
# Extract features từ original landmarks (normalized)
gesture_features = extract_gesture_features(landmarks)

# Predict gesture với Keras
gesture_label = "Unknown"
gesture_confidence = 0.0
if gesture_model is not None and len(gesture_features) == 42:
    try:
        # Prepare input (shape: (1, 42))
        features_array = np.array([gesture_features], dtype=np.float32)

        # Predict (verbose=0 để không in log)
        predictions = gesture_model.predict(features_array, verbose=0)
        probabilities = predictions[0]  # Shape: (NUM_CLASSES,)

        # Get predicted class
        class_id = np.argmax(probabilities)
        gesture_confidence = probabilities[class_id] * 100

        # Map class ID to label name
        if GESTURE_CLASS_NAMES is not None and class_id < len(GESTURE_CLASS_NAMES):
            gesture_label = GESTURE_CLASS_NAMES[class_id]
        else:
            gesture_label = f"Class_{class_id}"
    except Exception as e:
        print(f"⚠ Gesture prediction error: {e}")
```

### 4. Hiển Thị Gesture (sau line 639, trong label)

```python
# Thêm gesture vào label
label = f"ID:{hand_idx} {handedness_label} | Gesture: {gesture_label} ({gesture_confidence:.1f}%)"
```

## Tài Liệu Tham Khảo

### Official Documentation

- MediaPipe Hand Landmarks: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- MediaPipe Python API: https://developers.google.com/mediapipe/api/solutions/python

### Tutorials & Examples

- Hand Gesture Recognition: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
- MediaPipe Gesture Classification: https://github.com/google/mediapipe/tree/master/mediapipe/python/solutions

### Model Training với TensorFlow

- **TensorFlow/Keras**: https://www.tensorflow.org/api_docs/python/tf/keras
- **TensorFlow Lite**: https://www.tensorflow.org/lite
- **TFLite Converter**: https://www.tensorflow.org/lite/models/convert
- **Model Optimization**: https://www.tensorflow.org/model_optimization

## Lưu Ý Quan Trọng

1. **Normalized Coordinates**: Luôn dùng normalized (0-1) từ MediaPipe, không convert sang pixel
2. **Feature Order**: Phải đúng thứ tự [x0, y0, x1, y1, ..., x20, y20]
3. **Model Compatibility**: Đảm bảo model được train với cùng format features
4. **Performance**: Gesture prediction nên nhanh (< 5ms) để không ảnh hưởng FPS
5. **Multi-hand**: Có thể predict cho nhiều hands cùng lúc

## Troubleshooting

### Model không load được

- Check file path
- Check pickle format
- Check model structure trong dict

### Prediction sai

- Check feature extraction có đúng format không
- Check model có được train với cùng format không
- Check normalized coordinates (0-1)

### Performance chậm

- ✅ **Dùng TensorFlow Lite** thay vì Keras model (nhanh hơn 2-3x)
- ✅ **Quantize model**: Convert sang int8 để tăng tốc
- ✅ **Giảm số lượng hands detect**: Giảm NUM_HANDS
- ✅ **Cache model**: Load 1 lần, không reload mỗi frame
- ✅ **Optimize model architecture**: Giảm số layers/neurons nếu quá nặng
