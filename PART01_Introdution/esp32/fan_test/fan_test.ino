// Chân kết nối L298N
const int ENA = 18;    // chân PWM điều khiển tốc độ (ENA của L298N)
const int IN1 = 25;    // chiều quay
const int IN2 = 26;    // chiều quay

// PWM config
// Lưu ý: trong core 3.x ta dùng ledcAttach(pin, freq, resolution)
const int freq = 5000;    // tần số PWM 5 kHz
const int resolution = 8; // 8-bit resolution (0..255)

// Thời gian ramp (ms) khi thay đổi tốc độ
const unsigned long rampMillis = 800; // thời gian tăng/giảm tốc (tùy chỉnh)
const int rampSteps = 25;             // số bước ramp (mượt hơn = nhiều bước)

// Thời gian mỗi chế độ chạy (ms)
const unsigned long holdTime = 3000;

// Lưu trạng thái hiện tại (phần trăm) để ramp
static int currentPercent = 0;

void setup() {
  Serial.begin(115200);
  delay(10);
  Serial.println();
  Serial.println("=== ESP32 Fan control (L298N) - Core 3.x compatible ===");

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Gán chiều quay mặc định
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  // Cấu hình LEDC cho pin ENA bằng API mới (merge ledcSetup + ledcAttachPin)
  // bool ok = ledcAttach(pin, freq, resolution);
  if (!ledcAttach(ENA, freq, resolution)) {
    Serial.println("Warning: ledcAttach failed!");
    // Nếu ledcAttach thất bại, chương trình vẫn tiếp tục nhưng PWM không hoạt động.
  }

  // Tắt quạt lúc bắt đầu
  setSpeedPercent(0);
  delay(200);
}

// Chuyển % tốc độ (0..100) -> giá trị PWM (0..255) và viết ra
void setSpeedPercent(int percent) {
  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;
  int pwmVal = map(percent, 0, 100, 0, 255);

  // API mới: ledcWrite(pin, duty)
  if (!ledcWrite(ENA, pwmVal)) {
    // ledcWrite trả về bool theo tài liệu; nếu false thì báo lỗi
    Serial.println("Warning: ledcWrite failed or returned false");
  }
}

// Ramp từ tốc độ hiện tại tới tốc độ đích (percent: 0..100)
// Giúp giảm dòng khởi động mạnh
void rampTo(int targetPercent) {
  if (targetPercent < 0) targetPercent = 0;
  if (targetPercent > 100) targetPercent = 100;

  int start = currentPercent;
  int end = targetPercent;
  if (start == end) return;

  for (int step = 1; step <= rampSteps; ++step) {
    float t = (float)step / (float)rampSteps;
    int now = start + (int)((end - start) * t);
    setSpeedPercent(now);
    delay(rampMillis / rampSteps);
  }
  // đảm bảo giá trị cuối cùng chính xác
  setSpeedPercent(end);
  currentPercent = end;
}

void loop() {
  Serial.println("Quạt tốc độ 30%");
  rampTo(30);
  delay(holdTime);

  Serial.println("Quạt tốc độ 60%");
  rampTo(60);
  delay(holdTime);

  Serial.println("Quạt tốc độ 100%");
  rampTo(100);
  delay(holdTime);

  Serial.println("Tắt quạt");
  rampTo(0);
  delay(holdTime);
}
