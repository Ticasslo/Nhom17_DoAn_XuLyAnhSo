import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import threading
import time
import os

# ========== CẤU HÌNH ==========
MODEL_PATH = "yolov8m.pt"

# Detection settings
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IMAGE_SIZE = 416
DEFAULT_IOU_THRESHOLD = 0.6
MAX_DETECTIONS = 300

# UI settings
WINDOW_SIZE = "1200x800"
WINDOW_TITLE = "YOLO Object Detection - Badminton"
LEFT_PANEL_WIDTH = 300

# Colors
COLOR_HEADER_BG = "#2c3e50"
COLOR_BG = "#f0f0f0"
COLOR_BUTTON_SELECT = "#3498db"
COLOR_BUTTON_START = "#27ae60"
COLOR_BUTTON_STOP = "#e74c3c"
COLOR_TEXT_DARK = "#1e1e1e"
COLOR_TEXT_LIGHT = "#d4d4d4"

# Batch size thresholds (VRAM in GB) - cho non-stream mode
BATCH_SIZE_16GB = 16
BATCH_SIZE_12GB = 12
BATCH_SIZE_8GB = 8
BATCH_SIZE_6GB = 4
BATCH_SIZE_DEFAULT = 2
BATCH_SIZE_CPU = 1

# Batch size cho real-time streaming mode
# Lưu ý: Với stream=True, batch size lớn làm tăng latency (độ trễ)
# Batch size nhỏ (1-2) phù hợp hơn cho real-time display để giảm latency
# Theo nghiên cứu: batch=1 có latency ~20ms, batch=8 có latency ~68ms
BATCH_SIZE_REALTIME = 1  # Dùng batch=1 cho real-time mượt nhất

# Video file types
VIDEO_FILE_TYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
    ("All files", "*.*")
]

# Image size options
IMAGE_SIZE_OPTIONS = ["320", "416", "640"]


class YOLODetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=COLOR_BG)
        
        # Variables
        self.model = None
        self.video_path = None
        self.results = []
        self.is_processing = False
        self.current_frame_idx = 0
        self.video_cap = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 1
        self.total_objects = 0
        self.class_counts = {}
        self.frame_count = 0
        self.start_time = None
        
        # Setup UI
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg=COLOR_HEADER_BG, height=60)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="YOLO Object Detection System", 
            font=("Arial", 18, "bold"),
            bg=COLOR_HEADER_BG, 
            fg="white"
        )
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg=COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg="white", relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.config(width=LEFT_PANEL_WIDTH)
        
        # Control section
        control_label = tk.Label(
            left_panel, 
            text="Controls", 
            font=("Arial", 14, "bold"),
            bg="white"
        )
        control_label.pack(pady=10)
        
        # Video selection
        video_frame = tk.Frame(left_panel, bg="white")
        video_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(video_frame, text="Video File:", bg="white", font=("Arial", 10)).pack(anchor=tk.W)
        self.video_path_label = tk.Label(
            video_frame, 
            text="No video selected", 
            bg="white", 
            fg="gray",
            wraplength=280,
            justify=tk.LEFT
        )
        self.video_path_label.pack(anchor=tk.W, pady=2)
        
        select_btn = tk.Button(
            video_frame,
            text="Select Video",
            command=self.select_video,
            bg=COLOR_BUTTON_SELECT,
            fg="white",
            font=("Arial", 10),
            relief=tk.FLAT,
            cursor="hand2",
            padx=20,
            pady=5
        )
        select_btn.pack(fill=tk.X, pady=5)
        
        # Detection settings
        settings_frame = tk.LabelFrame(
            left_panel, 
            text="Detection Settings", 
            bg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=10
        )
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(settings_frame, text="Confidence:", bg="white").pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        conf_scale = tk.Scale(
            settings_frame,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.conf_var,
            bg="white"
        )
        conf_scale.pack(fill=tk.X, pady=5)
        
        tk.Label(settings_frame, text="Image Size:", bg="white").pack(anchor=tk.W, pady=(10, 0))
        self.imgsz_var = tk.StringVar(value=str(DEFAULT_IMAGE_SIZE))
        imgsz_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.imgsz_var,
            values=IMAGE_SIZE_OPTIONS,
            state="readonly",
            width=15
        )
        imgsz_combo.pack(fill=tk.X, pady=5)
        
        # Device info
        device_frame = tk.LabelFrame(
            left_panel,
            text="Device Info",
            bg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=10
        )
        device_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.device_label = tk.Label(
            device_frame,
            text=f"Device: {self.device.upper()}",
            bg="white",
            font=("Arial", 9)
        )
        self.device_label.pack(anchor=tk.W)
        
        if self.device == "cuda":
            idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(idx)
            gpu_memory = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            tk.Label(
                device_frame,
                text=f"GPU: {gpu_name}",
                bg="white",
                font=("Arial", 8),
                fg="gray"
            ).pack(anchor=tk.W)
            tk.Label(
                device_frame,
                text=f"VRAM: {gpu_memory:.1f} GB",
                bg="white",
                font=("Arial", 8),
                fg="gray"
            ).pack(anchor=tk.W)
        
        # Action buttons
        action_frame = tk.Frame(left_panel, bg="white")
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.process_btn = tk.Button(
            action_frame,
            text="Start Detection",
            command=self.start_detection,
            bg=COLOR_BUTTON_START,
            fg="white",
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.process_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = tk.Button(
            action_frame,
            text="Stop",
            command=self.stop_detection,
            bg=COLOR_BUTTON_STOP,
            fg="white",
            font=("Arial", 10),
            relief=tk.FLAT,
            cursor="hand2",
            padx=20,
            pady=5,
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        # Progress
        progress_frame = tk.Frame(left_panel, bg="white")
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(progress_frame, text="Progress:", bg="white", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = tk.Label(
            progress_frame,
            textvariable=self.progress_var,
            bg="white",
            font=("Arial", 9),
            fg="gray"
        )
        self.progress_label.pack(anchor=tk.W, pady=2)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=280
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Right panel - Video display and results
        right_panel = tk.Frame(main_container, bg="white", relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video display
        video_display_frame = tk.LabelFrame(
            right_panel,
            text="Video Preview",
            font=("Arial", 12, "bold"),
            bg="white",
            padx=10,
            pady=10
        )
        video_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.video_canvas = tk.Canvas(
            video_display_frame,
            bg="black",
            highlightthickness=0
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results section
        results_frame = tk.LabelFrame(
            right_panel,
            text="Detection Results",
            font=("Arial", 12, "bold"),
            bg="white",
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.BOTH, padx=10, pady=(0, 10))
        results_frame.config(height=200)
        results_frame.pack_propagate(False)
        
        # Stats
        stats_frame = tk.Frame(results_frame, bg="white")
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_label = tk.Label(
            stats_frame,
            text="No results yet",
            bg="white",
            font=("Arial", 10),
            justify=tk.LEFT
        )
        self.stats_label.pack(anchor=tk.W)
        
        # Results text
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=8,
            font=("Consolas", 9),
            bg=COLOR_TEXT_DARK,
            fg=COLOR_TEXT_LIGHT,
            wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def load_model(self):
        """Load YOLO model"""
        try:
            if not os.path.exists(MODEL_PATH):
                messagebox.showerror("Error", f"Model file not found: {MODEL_PATH}")
                return False
            
            self.progress_var.set("Loading model...")
            self.model = YOLO(MODEL_PATH)
            
            # Setup batch size based on GPU
            if self.device == "cuda":
                idx = torch.cuda.current_device()
                gpu_memory = torch.cuda.get_device_properties(idx).total_memory / 1024**3
                if gpu_memory >= 16:
                    self.batch_size = BATCH_SIZE_16GB
                elif gpu_memory >= 12:
                    self.batch_size = BATCH_SIZE_12GB
                elif gpu_memory >= 8:
                    self.batch_size = BATCH_SIZE_8GB
                elif gpu_memory >= 6:
                    self.batch_size = BATCH_SIZE_6GB
                else:
                    self.batch_size = BATCH_SIZE_DEFAULT
            else:
                self.batch_size = BATCH_SIZE_CPU
            
            self.progress_var.set("Model loaded successfully")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return False
    
    def select_video(self):
        """Select video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=VIDEO_FILE_TYPES
        )
        if file_path:
            self.video_path = file_path
            self.video_path_label.config(text=os.path.basename(file_path), fg="black")
            self.process_btn.config(state=tk.NORMAL)
            
            # Load and display first frame
            self.load_video_preview()
    
    def load_video_preview(self):
        """Load first frame of video for preview"""
        if not self.video_path:
            return
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_frame(frame_rgb)
            cap.release()
        except Exception as e:
            print(f"Error loading preview: {e}")
    
    def display_frame(self, frame):
        """Display frame on canvas"""
        try:
            h, w = frame.shape[:2]
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = w, h
            
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
            self.imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_canvas.delete("all")
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
            self.video_canvas.create_image(x, y, anchor=tk.NW, image=self.imgtk)
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def start_detection(self):
        """Start detection in separate thread"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Run detection in separate thread
        thread = threading.Thread(target=self.run_detection, daemon=True)
        thread.start()
    
    def run_detection(self):
        """Run YOLO detection in real-time"""
        try:
            # Reset counters
            self.results = []
            self.total_objects = 0
            self.class_counts = {}
            self.frame_count = 0
            self.current_frame_idx = 0
            
            self.progress_var.set("Initializing detection...")
            self.progress_bar['maximum'] = 100
            self.progress_bar['value'] = 0
            
            self.start_time = time.time()
            
            # Get total frames for progress
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Run prediction with stream=True for real-time processing
            self.progress_var.set("Running detection (real-time)...")
            
            # Use stream=True để hiển thị real-time
            # 
            # QUAN TRỌNG về batch size với stream=True:
            # - Với stream=True, YOLO xử lý từng frame và yield ngay kết quả
            # - Batch size lớn (8-16) làm TĂNG LATENCY (độ trễ):
            #   + Batch=1: latency ~20ms (mượt cho real-time)
            #   + Batch=8: latency ~68ms (chậm hơn, không phù hợp real-time)
            # - Batch size nhỏ (1-2) giảm latency nhưng throughput thấp hơn
            # 
            # Kết luận: Với real-time display, nên dùng batch=1 để giảm latency
            # Batch size lớn chỉ phù hợp khi stream=False (xử lý hết rồi mới hiển thị)
            for result in self.model.predict(
                source=self.video_path,
                conf=self.conf_var.get(),
                iou=DEFAULT_IOU_THRESHOLD,
                device=0 if self.device == "cuda" else "cpu",
                save=False,
                show=False,
                half=(self.device == "cuda"),
                imgsz=int(self.imgsz_var.get()),
                batch=BATCH_SIZE_REALTIME,  # Dùng batch=1 cho real-time (giảm latency)
                verbose=False,
                stream=True,  # Stream mode: yield từng result để hiển thị real-time
                max_det=MAX_DETECTIONS,
                agnostic_nms=False
            ):
                if not self.is_processing:  # Check if stopped
                    break
                
                # Store result
                self.results.append(result)
                self.frame_count += 1
                
                # Process and display this frame immediately
                self.process_frame_result(result, self.frame_count)
                
                # Update progress (thread-safe)
                progress = int((self.frame_count / total_frames) * 100) if total_frames > 0 else 0
                frame_num = self.frame_count  # Capture value for lambda
                self.root.after_idle(lambda p=progress: self.progress_bar.config(value=p))
                self.root.after_idle(lambda fn=frame_num, tf=total_frames: self.progress_var.set(f"Processing frame {fn}/{tf}"))
                
                # Display frame with bounding boxes immediately
                annotated_frame = result.plot()
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                # Use after_idle to update UI from thread safely
                self.root.after_idle(self.display_frame, frame_rgb)
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Final summary
            total_time = time.time() - self.start_time
            fps = self.frame_count / total_time if total_time > 0 else 0
            
            # Update final progress (thread-safe)
            fc = self.frame_count  # Capture values for lambda
            tt = total_time
            fps_val = fps
            self.root.after_idle(lambda: self.progress_bar.config(value=100))
            self.root.after_idle(lambda: self.progress_var.set(f"Completed: {fc} frames in {tt:.2f}s ({fps_val:.2f} FPS)"))
            
            # Update final summary
            self.update_final_summary(total_time, fps)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.progress_var.set("Error occurred")
        finally:
            self.is_processing = False
            self.process_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def process_frame_result(self, result, frame_num):
        """Process a single frame result and update UI in real-time"""
        num_objects = len(result.boxes)
        self.total_objects += num_objects
        
        if num_objects > 0:
            boxes_cpu = result.boxes.cpu()
            cls_all = boxes_cpu.cls.numpy().astype(int)
            
            # Update class counts
            for cls in cls_all:
                class_name = self.model.names[int(cls)]
                self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
            
            # Update stats label in real-time
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            fps = frame_num / elapsed_time if elapsed_time > 0 else 0
            
            stats_text = f"Frame: {frame_num} | Objects: {self.total_objects} | Time: {elapsed_time:.1f}s | FPS: {fps:.1f}"
            self.root.after(0, lambda: self.stats_label.config(text=stats_text))
    
    def update_final_summary(self, total_time, fps):
        """Update final summary after detection completes"""
        results_text = "DETECTION RESULTS (REAL-TIME)\n" + "="*60 + "\n\n"
        results_text += f"Total frames processed: {self.frame_count}\n"
        results_text += f"Total objects detected: {self.total_objects}\n"
        results_text += f"Total time: {total_time:.2f}s\n"
        results_text += f"Average speed: {fps:.2f} FPS\n\n"
        
        if self.class_counts:
            results_text += "Class distribution:\n"
            for class_name, count in sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.total_objects) * 100 if self.total_objects > 0 else 0
                results_text += f"  • {class_name}: {count} ({percentage:.1f}%)\n"
        
        # Update results text
        self.root.after(0, lambda: self.results_text.insert(1.0, results_text))
        
        # Final stats
        final_stats = f"Frames: {self.frame_count} | Objects: {self.total_objects} | Time: {total_time:.2f}s | FPS: {fps:.2f}"
        self.root.after(0, lambda: self.stats_label.config(text=final_stats))
    
    def stop_detection(self):
        """Stop detection"""
        self.is_processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_var.set("Stopped")


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectionUI(root)
    root.mainloop()

