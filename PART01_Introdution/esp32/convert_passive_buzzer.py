"""
MusicXML to Arduino Note Converter (with GUI)
Chuyển đổi file .mxl hoặc .musicxml sang mảng nốt nhạc cho Arduino/ESP32

Cài đặt thư viện:
pip install music21

Sử dụng:
python convert.py
"""

from music21 import converter, note, chord, stream
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import threading

# Bảng ánh xạ note -> Arduino define
NOTE_MAP = {
    'C': 'NOTE_C', 'D': 'NOTE_D', 'E': 'NOTE_E', 'F': 'NOTE_F',
    'G': 'NOTE_G', 'A': 'NOTE_A', 'B': 'NOTE_B',
    'C#': 'NOTE_CS', 'D#': 'NOTE_DS', 'F#': 'NOTE_FS', 
    'G#': 'NOTE_GS', 'A#': 'NOTE_AS',
    'C-': 'NOTE_B', 'D-': 'NOTE_CS', 'E-': 'NOTE_DS',
    'F-': 'NOTE_E', 'G-': 'NOTE_FS', 'A-': 'NOTE_GS', 'B-': 'NOTE_AS'
}

# Bảng tần số chuẩn cho Arduino
FREQ_TABLE = {
    'B0': 31, 'C1': 33, 'CS1': 35, 'D1': 37, 'DS1': 39, 'E1': 41, 'F1': 44,
    'FS1': 46, 'G1': 49, 'GS1': 52, 'A1': 55, 'AS1': 58, 'B1': 62,
    'C2': 65, 'CS2': 69, 'D2': 73, 'DS2': 78, 'E2': 82, 'F2': 87,
    'FS2': 93, 'G2': 98, 'GS2': 104, 'A2': 110, 'AS2': 117, 'B2': 123,
    'C3': 131, 'CS3': 139, 'D3': 147, 'DS3': 156, 'E3': 165, 'F3': 175,
    'FS3': 185, 'G3': 196, 'GS3': 208, 'A3': 220, 'AS3': 233, 'B3': 247,
    'C4': 262, 'CS4': 277, 'D4': 294, 'DS4': 311, 'E4': 330, 'F4': 349,
    'FS4': 370, 'G4': 392, 'GS4': 415, 'A4': 440, 'AS4': 466, 'B4': 494,
    'C5': 523, 'CS5': 554, 'D5': 587, 'DS5': 622, 'E5': 659, 'F5': 698,
    'FS5': 740, 'G5': 784, 'GS5': 831, 'A5': 880, 'AS5': 932, 'B5': 988,
    'C6': 1047, 'CS6': 1109, 'D6': 1175, 'DS6': 1245, 'E6': 1319, 'F6': 1397,
    'FS6': 1480, 'G6': 1568, 'GS6': 1661, 'A6': 1760, 'AS6': 1865, 'B6': 1976,
    'C7': 2093, 'CS7': 2217, 'D7': 2349, 'DS7': 2489, 'E7': 2637, 'F7': 2794,
    'FS7': 2960, 'G7': 3136, 'GS7': 3322, 'A7': 3520, 'AS7': 3729, 'B7': 3951,
    'C8': 4186, 'CS8': 4435, 'D8': 4699, 'DS8': 4978
}

def note_to_arduino(pitch_name, octave):
    """Chuyển note thành tên Arduino (NOTE_C4, NOTE_D5,...)"""
    if '#' in pitch_name:
        base = pitch_name.replace('#', '')
        note_name = f"{NOTE_MAP.get(base + '#', 'NOTE_C')}{octave}"
    elif '-' in pitch_name:
        base = pitch_name.replace('-', '')
        note_name = f"{NOTE_MAP.get(base + '-', 'NOTE_C')}{octave}"
    else:
        note_name = f"{NOTE_MAP.get(pitch_name, 'NOTE_C')}{octave}"
    
    return note_name

def note_to_frequency(pitch_name, octave):
    """Chuyển note thành tần số Hz"""
    if '#' in pitch_name:
        freq_key = pitch_name.replace('#', 'S') + str(octave)
    elif '-' in pitch_name:
        if pitch_name == 'D-':
            freq_key = f"CS{octave}"
        elif pitch_name == 'E-':
            freq_key = f"DS{octave}"
        elif pitch_name == 'G-':
            freq_key = f"FS{octave}"
        elif pitch_name == 'A-':
            freq_key = f"GS{octave}"
        elif pitch_name == 'B-':
            freq_key = f"AS{octave}"
        else:
            freq_key = f"{pitch_name[0]}{octave}"
    else:
        freq_key = f"{pitch_name}{octave}"
    
    return FREQ_TABLE.get(freq_key, 440)

def duration_to_beat(duration_type):
    """Chuyển loại nốt thành beat (4=1/4, 8=1/8,...)"""
    duration_map = {
        'whole': 1,
        'half': 2,
        'quarter': 4,
        'eighth': 8,
        '16th': 16,
        '32nd': 32
    }
    return duration_map.get(duration_type, 4)

def convert_musicxml_to_arduino(file_path, output_callback):
    """Chuyển đổi file MusicXML sang Arduino code"""
    
    output_callback(f"🎵 Đang đọc file: {os.path.basename(file_path)}\n")
    
    try:
        score = converter.parse(file_path)
    except Exception as e:
        output_callback(f"❌ Lỗi đọc file: {e}\n")
        return None
    
    parts = score.parts
    if not parts:
        output_callback("❌ Không tìm thấy nốt nhạc trong file!\n")
        return None
    
    melody_part = parts[0]
    
    melody_notes = []
    melody_beats = []
    frequencies = []
    
    output_callback("\n📝 Đang trích xuất nốt nhạc...\n")
    
    for element in melody_part.flatten().notesAndRests:
        if isinstance(element, note.Note):
            pitch_name = element.pitch.name
            octave = element.pitch.octave
            duration_type = element.duration.type
            
            arduino_note = note_to_arduino(pitch_name, octave)
            freq = note_to_frequency(pitch_name, octave)
            beat = duration_to_beat(duration_type)
            
            melody_notes.append(arduino_note)
            frequencies.append(freq)
            melody_beats.append(beat)
            
            output_callback(f"  {pitch_name}{octave} → {arduino_note} ({freq} Hz, beat={beat})\n")
            
        elif isinstance(element, note.Rest):
            melody_notes.append("0")
            frequencies.append(0)
            beat = duration_to_beat(element.duration.type)
            melody_beats.append(beat)
            output_callback(f"  REST (nghỉ, beat={beat})\n")
        
        elif isinstance(element, chord.Chord):
            highest = element.pitches[-1]
            arduino_note = note_to_arduino(highest.name, highest.octave)
            freq = note_to_frequency(highest.name, highest.octave)
            beat = duration_to_beat(element.duration.type)
            
            melody_notes.append(arduino_note)
            frequencies.append(freq)
            melody_beats.append(beat)
            
            output_callback(f"  CHORD → {arduino_note} ({freq} Hz, beat={beat})\n")
    
    output_callback("\n✅ Chuyển đổi hoàn tất!\n")
    output_callback("="*70 + "\n")
    output_callback("📋 CODE ARDUINO/ESP32:\n")
    output_callback("="*70 + "\n\n")
    
    # Tạo code
    unique_notes = sorted(set([n for n in melody_notes if n != "0"]))
    unique_freqs = {}
    for n in unique_notes:
        for i, note_name in enumerate(melody_notes):
            if note_name == n:
                unique_freqs[n] = frequencies[i]
                break
    
    arduino_code = ""
    arduino_code += "// Định nghĩa tần số các nốt\n"
    for note_name in unique_notes:
        arduino_code += f"#define {note_name:<12} {unique_freqs[note_name]}\n"
    
    arduino_code += f"\n// Melody ({len(melody_notes)} nốt)\n"
    arduino_code += "int melody[] = {\n"
    for i in range(0, len(melody_notes), 6):
        chunk = melody_notes[i:i+6]
        arduino_code += "  " + ", ".join(chunk) + ("," if i+6 < len(melody_notes) else "") + "\n"
    arduino_code += "};\n"
    
    arduino_code += f"\n// Độ dài nốt (beats)\n"
    arduino_code += "int beats[] = {\n"
    for i in range(0, len(melody_beats), 10):
        chunk = [str(b) for b in melody_beats[i:i+10]]
        arduino_code += "  " + ", ".join(chunk) + ("," if i+10 < len(melody_beats) else "") + "\n"
    arduino_code += "};\n"
    
    arduino_code += f"\nconst int noteCount = {len(melody_notes)};\n"
    
    output_callback(arduino_code)
    output_callback("\n" + "="*70 + "\n")
    
    # Lưu file
    output_file = file_path.rsplit('.', 1)[0] + '_arduino.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(arduino_code)
    
    output_callback(f"💾 Đã lưu code vào: {output_file}\n")
    
    return arduino_code

class MusicXMLConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 MusicXML to Arduino Converter")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎵 MusicXML to Arduino Converter", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Chọn file MusicXML", padding="10")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state='readonly')
        file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="📁 Chọn file", command=self.browse_file)
        browse_btn.grid(row=0, column=2)
        
        convert_btn = ttk.Button(file_frame, text="🔄 Convert", command=self.convert_file, 
                                style='Accent.TButton')
        convert_btn.grid(row=0, column=3, padx=(5, 0))
        
        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Kết quả", padding="10")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, 
                                                     font=('Consolas', 10))
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        copy_btn = ttk.Button(button_frame, text="📋 Copy code", command=self.copy_to_clipboard)
        copy_btn.grid(row=0, column=0, padx=(0, 5))
        
        clear_btn = ttk.Button(button_frame, text="🗑️ Xóa", command=self.clear_output)
        clear_btn.grid(row=0, column=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Sẵn sàng...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, 
                              anchor=tk.W)
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Initial message
        self.output_text.insert(tk.END, "🎵 Chào mừng bạn đến với MusicXML to Arduino Converter!\n\n")
        self.output_text.insert(tk.END, "Hướng dẫn sử dụng:\n")
        self.output_text.insert(tk.END, "1. Nhấn nút '📁 Chọn file' để chọn file .mxl hoặc .musicxml\n")
        self.output_text.insert(tk.END, "2. Nhấn nút '🔄 Convert' để chuyển đổi\n")
        self.output_text.insert(tk.END, "3. Code Arduino sẽ hiển thị ở đây và tự động lưu file .txt\n")
        self.output_text.insert(tk.END, "4. Nhấn '📋 Copy code' để copy vào Arduino IDE\n\n")
        self.output_text.insert(tk.END, "=" * 70 + "\n")
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Chọn file MusicXML",
            filetypes=[
                ("MusicXML files", "*.mxl *.musicxml *.xml"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file_path_var.set(filename)
            self.status_var.set(f"Đã chọn: {os.path.basename(filename)}")
    
    def output_callback(self, text):
        """Callback để hiển thị output trong text widget"""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def convert_file(self):
        file_path = self.file_path_var.get()
        
        if not file_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file MusicXML!")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Lỗi", f"File không tồn tại: {file_path}")
            return
        
        self.clear_output()
        self.status_var.set("Đang chuyển đổi...")
        
        # Chạy conversion trong thread riêng để không block GUI
        def run_conversion():
            try:
                convert_musicxml_to_arduino(file_path, self.output_callback)
                self.status_var.set("✅ Chuyển đổi hoàn tất!")
            except Exception as e:
                self.output_callback(f"\n❌ Lỗi: {str(e)}\n")
                self.status_var.set("❌ Có lỗi xảy ra!")
        
        thread = threading.Thread(target=run_conversion)
        thread.daemon = True
        thread.start()
    
    def copy_to_clipboard(self):
        text = self.output_text.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("📋 Đã copy vào clipboard!")
        messagebox.showinfo("Thành công", "Đã copy code vào clipboard!")
    
    def clear_output(self):
        self.output_text.delete("1.0", tk.END)

def main():
    root = tk.Tk()
    app = MusicXMLConverterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()