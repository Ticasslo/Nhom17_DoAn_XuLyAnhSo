// Happy Birthday trên ESP32 + KY-006 (buzzer passive)
// Compatible với ESP32 Arduino Core 3.x

const int buzzerPin = 25;   // chân GPIO nối + của KY-006
const int tempo = 120;      // beats per minute (thay đổi để nhanh/chậm)

void setup() {
  // Trong Core 3.x: ledcAttach(pin, freq, resolution)
  ledcAttach(buzzerPin, 2000, 8); // pin 25, tần số 2kHz, 8-bit resolution
}

// bảng tần số (Hz)
#define NOTE_C4  261
#define NOTE_D4  294
#define NOTE_E4  329
#define NOTE_F4  349
#define NOTE_G4  392
#define NOTE_A4  440
#define NOTE_B4  494
#define NOTE_C5  523
#define NOTE_D5  587
#define NOTE_E5  659
#define NOTE_F5  698
#define NOTE_G5  784

// Melody: Happy Birthday (nốt)
int melody[] = {
  NOTE_G4, NOTE_G4, NOTE_A4, NOTE_G4, NOTE_C5, NOTE_B4,
  NOTE_G4, NOTE_G4, NOTE_A4, NOTE_G4, NOTE_D5, NOTE_C5,
  NOTE_G4, NOTE_G4, NOTE_G5, NOTE_E5, NOTE_C5, NOTE_B4, NOTE_A4,
  NOTE_F5, NOTE_F5, NOTE_E5, NOTE_C5, NOTE_D5, NOTE_C5
};

// Duration in beats (relative): 4 = quarter note, 8 = eighth, etc.
int beats[] = {
  8, 8, 4, 4, 4, 2,
  8, 8, 4, 4, 4, 2,
  8, 8, 4, 4, 4, 4, 2,
  8, 8, 4, 4, 4, 2
};

const int noteCount = sizeof(melody) / sizeof(melody[0]);

void loop() {
  for (int i = 0; i < noteCount; i++) {
    int note = melody[i];
    int beat = beats[i];

    // tính thời gian nốt theo tempo (ms)
    int wholeNote = (60000 * 4) / tempo; // thời gian 1 whole note
    int duration = wholeNote / beat;

    // phát âm: đặt tần số
    if (note == 0) {
      // rest
      ledcWriteTone(buzzerPin, 0);
    } else {
      ledcWriteTone(buzzerPin, note);
    }

    delay(duration * 0.9); // giữ nốt 90% thời lượng
    // tắt ngắn giữa các nốt (10%)
    ledcWriteTone(buzzerPin, 0);
    delay(duration * 0.1);
  }

  // lặp lại sau 1.5 giây
  delay(1500);
}