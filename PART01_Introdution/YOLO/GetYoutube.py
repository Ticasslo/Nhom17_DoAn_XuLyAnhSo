"""
Tkinter + yt-dlp + imageio-ffmpeg (ffmpeg binary via pip) example

Phiên bản này sử dụng `imageio-ffmpeg` để lấy đường dẫn ffmpeg đã được đóng gói trong wheel,
như vậy người dùng không cần cài ffmpeg thủ công trên hệ thống.

Tính năng:
 - Lấy stream URL bằng yt-dlp
 - Dùng ffmpeg (binary do imageio-ffmpeg cung cấp) để transcode/ép fps và scale
 - Đọc raw RGB frames từ stdout của ffmpeg, hiển thị lên Tkinter Canvas
 - UI: nhập URL, chọn chất lượng (Auto/1080/720/480/360), cấu hình W/H, FPS, nút Phát/Dừng

Yêu cầu:
    pip install yt-dlp pillow opencv-python imageio-ffmpeg numpy

Chạy: python tk_ffmpeg_ytdlp_imageio.py

Ghi chú:
 - imageio-ffmpeg sẽ cung cấp một binary ffmpeg trong wheel; hàm get_ffmpeg_exe() trả về đường dẫn đó.
 - Nếu muốn dùng ffmpeg hệ thống thay vì imageio-ffmpeg, có thể đổi đường dẫn `ffmpeg_exe`.
"""

import subprocess
import threading
import time
import sys
import tkinter as tk
from tkinter import ttk
from queue import Queue, Empty

import numpy as np
from PIL import Image, ImageTk
import yt_dlp
import imageio_ffmpeg as iioff


def get_url_with_ytdlp(youtube_url: str, preferred_height=None):
    """Trả về direct stream URL chọn format phù hợp (dựa vào preferred_height nếu có).
    Lọc bỏ HLS/DASH manifests và chỉ lấy direct video URLs.
    """
    ydl_opts = {"skip_download": True, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        formats = info.get("formats") or [info]
        
        # Ưu tiên direct video streams, nhưng vẫn cho phép HLS/DASH nếu không có
        # HLS manifests thường có protocol "m3u8" hoặc "m3u8_native"
        # DASH manifests có protocol "http_dash_segments"
        direct_candidates = []
        hls_candidates = []
        
        for f in formats:
            url = f.get("url")
            if not url:
                continue
            
            protocol = f.get("protocol", "").lower()
            is_hls_format = "m3u8" in protocol or "dash" in protocol or "hls" in protocol
            
            # Chỉ lấy video streams (không phải audio only)
            has_video = f.get("vcodec") != "none"
            has_audio = f.get("acodec") != "none"
            
            if not has_video:
                continue  # Bỏ qua audio-only
            
            if is_hls_format:
                hls_candidates.append(f)
            else:
                # Ưu tiên video+audio, nhưng video-only cũng OK
                direct_candidates.append(f)
        
        # Ưu tiên direct streams, fallback sang HLS nếu không có
        if direct_candidates:
            candidates = direct_candidates
            print(f"Tìm thấy {len(direct_candidates)} direct video stream(s)")
        elif hls_candidates:
            candidates = hls_candidates
            print(f"Cảnh báo: Chỉ tìm thấy HLS/DASH manifests ({len(hls_candidates)} format). ffmpeg sẽ xử lý.")
        else:
            # Fallback cuối cùng: lấy bất kỳ format nào có URL
            candidates = [f for f in formats if f.get("url") and f.get("vcodec") != "none"]
            if not candidates:
                raise RuntimeError("Không tìm thấy format video có URL.")
            print(f"Cảnh báo: Dùng format fallback ({len(candidates)} format)")
        
        if preferred_height is not None:
            with_height = [f for f in candidates if f.get("height")]
            if with_height:
                exact = [f for f in with_height if f.get("height") == preferred_height]
                if exact:
                    return exact[-1].get("url")
                le = [f for f in with_height if f.get("height") <= preferred_height]
                if le:
                    le.sort(key=lambda x: x.get("height") or 0)
                    return le[-1].get("url")
                gt = [f for f in with_height if f.get("height") > preferred_height]
                if gt:
                    gt.sort(key=lambda x: x.get("height") or 0)
                    return gt[0].get("url")
        candidates.sort(key=lambda f: (f.get("height") or 0))
        return candidates[-1].get("url")


class FfmpegReader(threading.Thread):
    """Thread chạy ffmpeg và đọc stdout raw frames, đẩy vào queue.
    Dùng binary ffmpeg do imageio-ffmpeg cung cấp (get_ffmpeg_exe()).
    """
    def __init__(self, stream_url, queue, width=640, height=360, target_fps=30, ffmpeg_exe=None):
        super().__init__(daemon=True)
        self.stream_url = stream_url
        self.queue = queue
        self.width = int(width)
        self.height = int(height)
        self.target_fps = int(target_fps)
        self._running = threading.Event()
        self._running.set()
        self.proc = None
        self.frame_size = self.width * self.height * 3
        self.ffmpeg_exe = ffmpeg_exe
        self.stderr_thread = None

    def stop(self):
        self._running.clear()
        if self.stderr_thread:
            self.stderr_thread = None
        if self.proc:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

    def _read_stderr(self):
        """Thread đọc stderr của ffmpeg để tránh deadlock."""
        if not self.proc:
            return
        try:
            stderr = self.proc.stderr
            if stderr:
                # Đọc stderr để tránh buffer đầy
                while self._running.is_set() and self.proc:
                    try:
                        line = stderr.readline()
                        if not line:
                            break
                        # In tất cả để debug
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        if line_str:
                            print(f"ffmpeg: {line_str}")
                    except Exception:
                        break
        except Exception:
            pass

    def run(self):
        ffmpeg_path = self.ffmpeg_exe or iioff.get_ffmpeg_exe()
        print(f"Đang khởi động ffmpeg với URL: {self.stream_url[:80]}...")
        
        # Kiểm tra xem URL có phải là HLS manifest không
        is_hls = "m3u8" in self.stream_url.lower() or "hls" in self.stream_url.lower()
        
        # Thêm các tham số để xử lý stream trực tiếp (HLS/DASH) và giảm latency
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "warning",  # Đổi thành warning để xem thông tin debug
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-strict", "experimental",
        ]
        
        # Thêm tham số cho HLS input nếu cần
        if is_hls:
            print("Phát hiện HLS manifest, dùng tham số HLS đặc biệt")
            # Không cần tham số đặc biệt cho HLS input, ffmpeg tự xử lý
            # Nhưng có thể thêm timeout và retry
            pass
        
        cmd.extend([
            "-i", self.stream_url,
            "-vf", f"fps={self.target_fps},scale={self.width}:{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-"  # Output to stdout
        ])
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered để giảm latency
            )
        except FileNotFoundError:
            print("ffmpeg không được tìm thấy. Hãy kiểm tra imageio-ffmpeg hoặc ffmpeg hệ thống.")
            return
        except Exception as e:
            print("Không thể khởi ffmpeg:", e)
            return

        if self.proc is None:
            return

        # Khởi động thread đọc stderr để tránh deadlock
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()

        sock = self.proc.stdout
        frames_read = 0
        no_data_count = 0
        # Tăng số lần thử cho HLS (cần thời gian tải manifest và segments)
        max_no_data = 500 if is_hls else 100
        
        # Đợi lâu hơn cho HLS để ffmpeg tải manifest và segments đầu tiên
        wait_time = 2.0 if is_hls else 0.5
        print(f"Đợi {wait_time}s để ffmpeg khởi động...")
        time.sleep(wait_time)
        
        while self._running.is_set():
            try:
                if self.proc is None or sock is None:
                    print("ffmpeg process hoặc stdout đã bị None")
                    break
                
                # Kiểm tra xem process có còn chạy không
                if self.proc.poll() is not None:
                    print(f"ffmpeg đã dừng với return code: {self.proc.returncode}")
                    break
                
                raw = sock.read(self.frame_size)
                if not raw or len(raw) < self.frame_size:
                    no_data_count += 1
                    if no_data_count > max_no_data:
                        print(f"Không nhận được data sau {max_no_data} lần thử. Có thể stream URL không hợp lệ.")
                        break
                    if self.proc.poll() is not None:
                        print(f"ffmpeg đã dừng với return code: {self.proc.returncode}")
                        break
                    time.sleep(0.01)
                    continue
                
                # Reset counter khi có data
                no_data_count = 0
                frames_read += 1
                if frames_read == 1:
                    print("Đã nhận được frame đầu tiên!")
                
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                try:
                    if self.queue.qsize() >= 1:
                        try:
                            _ = self.queue.get_nowait()
                        except Empty:
                            pass
                    self.queue.put(frame)
                except Exception:
                    pass
            except Exception as e:
                print(f"Lỗi đọc từ ffmpeg stdout: {e}")
                break
        
        print(f"Đã đọc tổng cộng {frames_read} frames")

        try:
            if self.proc:
                self.proc.kill()
        except Exception:
            pass
        self.proc = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YouTube (yt-dlp) + ffmpeg(imageio) -> Tkinter")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.queue = Queue(maxsize=2)
        self.reader = None

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=6)

        ttk.Label(top, text="YouTube / Stream URL:").pack(side="left")
        self.url_var = tk.StringVar()
        self.entry = ttk.Entry(top, textvariable=self.url_var, width=50)
        self.entry.pack(side="left", padx=6)
        self.entry.bind("<Return>", lambda e: self.on_play())

        self.quality_var = tk.StringVar(value="Auto")
        quality_box = ttk.Combobox(top, textvariable=self.quality_var, state="readonly", width=8)
        quality_box["values"] = ("Auto", "1080", "720", "480", "360")
        quality_box.pack(side="left", padx=(4,6))

        ttk.Label(top, text="W:").pack(side="left")
        self.w_var = tk.IntVar(value=640)
        ttk.Entry(top, textvariable=self.w_var, width=6).pack(side="left", padx=(2,6))
        ttk.Label(top, text="H:").pack(side="left")
        self.h_var = tk.IntVar(value=360)
        ttk.Entry(top, textvariable=self.h_var, width=6).pack(side="left", padx=(2,6))

        self.play_btn = ttk.Button(top, text="Phát", command=self.on_play)
        self.play_btn.pack(side="left", padx=(0,6))
        self.stop_btn = ttk.Button(top, text="Dừng", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left")

        fps_frame = ttk.Frame(self)
        fps_frame.pack(fill="x", padx=6, pady=(4,0))
        ttk.Label(fps_frame, text="Target FPS (ffmpeg):").pack(side="left")
        self.target_fps_var = tk.IntVar(value=30)
        ttk.Entry(fps_frame, textvariable=self.target_fps_var, width=6).pack(side="left", padx=(4,6))

        self.canvas = tk.Canvas(self, width=640, height=360, bg="black")
        self.canvas.pack(fill="both", expand=True)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", padx=8)

        self._imgtk = None
        self._last_show_time = 0.0
        self.update_id = None
        self.after(20, self.update_frame)

    def start_ffmpeg_reader(self, stream_url, width, height, target_fps):
        if self.reader is not None:
            try:
                self.reader.stop()
            except Exception:
                pass
            time.sleep(0.1)
            self.reader = None
        try:
            while True:
                self.queue.get_nowait()
        except Empty:
            pass

        ffmpeg_exe = None
        try:
            ffmpeg_exe = iioff.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None

        self.reader = FfmpegReader(stream_url, self.queue, width=width, height=height, target_fps=target_fps, ffmpeg_exe=ffmpeg_exe)
        self.reader.start()
        self.status_var.set("Connecting (ffmpeg)...")
        self.play_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def on_play(self):
        url = self.url_var.get().strip()
        if not url:
            self.status_var.set("Chưa nhập URL")
            return
        q = self.quality_var.get()
        pref = None if q == "Auto" else int(q)
        try:
            self.status_var.set("Lấy URL với yt-dlp...")
            self.update()
            stream_url = get_url_with_ytdlp(url, preferred_height=pref)
        except Exception as e:
            self.status_var.set(f"yt-dlp error: {e}")
            return

        width = int(self.w_var.get())
        height = int(self.h_var.get())
        target_fps = int(self.target_fps_var.get())
        self.start_ffmpeg_reader(stream_url, width, height, target_fps)

    def on_stop(self):
        if self.reader is not None:
            try:
                self.reader.stop()
            except Exception:
                pass
            self.reader = None
        try:
            while True:
                self.queue.get_nowait()
        except Empty:
            pass
        self.status_var.set("Stopped")
        self.play_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.canvas.delete("all")

    def update_frame(self):
        frame = None
        try:
            while True:
                f = self.queue.get_nowait()
                frame = f
        except Empty:
            pass

        if frame is None:
            if self.reader is not None:
                self.status_var.set("Waiting for frame...")
            self.update_id = self.after(100, self.update_frame)
            return

        try:
            h, w = frame.shape[:2]
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw <= 1 or ch <= 1:
                cw, ch = w, h
            scale = min(cw / w, ch / h)
            nw, nh = int(w * scale), int(h * scale)
            img = Image.fromarray(frame).resize((nw, nh), Image.BILINEAR)
            self._imgtk = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            x = (cw - nw) // 2
            y = (ch - nh) // 2
            self.canvas.create_image(x, y, anchor="nw", image=self._imgtk)
            self.status_var.set("Playing (ffmpeg)")
        except Exception as e:
            print("Lỗi hiển thị frame:", e)

        self.update_id = self.after(10, self.update_frame)

    def on_close(self):
        if self.update_id:
            try:
                self.after_cancel(self.update_id)
            except Exception:
                pass
        if self.reader is not None:
            try:
                self.reader.stop()
            except Exception:
                pass
        time.sleep(0.2)
        self.destroy()


if __name__ == "__main__":
    app = App()
    if len(sys.argv) >= 2:
        app.url_var.set(sys.argv[1])
    app.mainloop()
