import cv2
import numpy as np

# ISP nhẹ nhàng và nhanh
def simple_isp(img):
    # Chuyển float 0..1
    img = img.astype(np.float32) / 255.0

    # White Balance nhẹ (gray-world đơn giản)
    mean = img.mean(axis=(0,1), keepdims=True)
    gray = mean.mean()
    scale = gray / (mean + 1e-6)
    img = np.clip(img * scale, 0, 1)

    # Gamma correction
    img = np.power(img, 1/2.2)

    # Sharpen nhẹ
    blurred = cv2.GaussianBlur(img, (0,0), 2)
    img = np.clip(img + (img - blurred) * 1.0, 0, 1)

    return (img * 255).astype(np.uint8)

# MAIN
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply ISP (nhanh, không lag)
    output = simple_isp(frame)

    cv2.imshow("Output", output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
