"""
Cat vs Dog Detector - Graphical User Interface
Auto-loads trained model on startup, organized model selector on change
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from pathlib import Path
from ultralytics import YOLO
import threading
import time


class CatDogDetectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🐱🐕 Cat vs Dog Detector")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.model = None
        self.model_name = "No model"
        self.current_image_path = None
        self.current_video_path = None
        self.original_image = None
        self.annotated_image = None
        self.current_mode = "image"
        
        # Video variables
        self.is_playing_video = False
        self.stop_video = False

        # Load initial model (auto-load trained)
        self.load_model()
        
        # Create UI
        self.create_ui()

    def load_model(self, show_selector=False):
        """Load YOLO model - auto-load trained on startup, show selector on change"""
        try:
            project_root = Path(__file__).parent
            runs_dir = project_root / "runs" / "train"
            pretrained_dir = project_root / "models" / "pretrained"
            
            # Check for trained models
            trained_model = None
            train_dirs = []
            if runs_dir.exists():
                train_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
                if train_dirs:
                    latest_run = max(train_dirs, key=lambda d: d.stat().st_mtime)
                    model_path = latest_run / "weights" / "best.pt"
                    if model_path.exists():
                        trained_model = (model_path, latest_run.name)
            
            # Check for pretrained models
            pretrained_models = []
            if pretrained_dir.exists():
                pretrained_models = list(pretrained_dir.glob("*.pt"))
            
            # If show_selector requested (Change Model button)
            if show_selector:
                self.show_model_selector(trained_model, train_dirs, pretrained_models)
                return
            
            # Auto-load on startup
            if trained_model:
                self.model = YOLO(str(trained_model[0]))
                self.model_name = f"Self-trained: {trained_model[1]}"
                print(f"✅ Auto-loaded: {self.model_name}")
            elif pretrained_models:
                self.model = YOLO(str(pretrained_models[0]))
                self.model_name = f"Pretrained: {pretrained_models[0].name}"
                print(f"✅ Auto-loaded: {self.model_name}")
            else:
                choice = messagebox.askyesno(
                    "No Models Found",
                    "No models found.\n\nDownload pretrained YOLO model?"
                )
                if choice:
                    self.download_pretrained()
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def show_model_selector(self, trained_model, train_dirs, pretrained_models):
        """Show organized model selector with categories"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select AI Model")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Header
        header = tk.Frame(dialog, bg="#2c3e50", height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="🔄 Select AI Model", font=("Arial", 14, "bold"),
                bg="#2c3e50", fg="white").pack(pady=12)

        # Scrollable frame
        main_frame = tk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Category 1: Self-trained Models
        if train_dirs:
            cat_frame = tk.LabelFrame(scrollable_frame, text="🎓 Self-Trained Models",
                                     font=("Arial", 11, "bold"), padx=10, pady=10)
            cat_frame.pack(fill=tk.X, padx=5, pady=5)

            for train_dir in sorted(train_dirs, key=lambda d: d.stat().st_mtime, reverse=True):
                model_path = train_dir / "weights" / "best.pt"
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    is_current = trained_model and train_dir.name == trained_model[1]
                    
                    btn_frame = tk.Frame(cat_frame, bg="#ecf0f1" if is_current else "white", relief=tk.RAISED, borderwidth=1)
                    btn_frame.pack(fill=tk.X, pady=3)

                    info_frame = tk.Frame(btn_frame, bg="#ecf0f1" if is_current else "white")
                    info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=8)

                    tk.Label(info_frame, text=train_dir.name, font=("Arial", 10, "bold"),
                            bg="#ecf0f1" if is_current else "white", anchor="w").pack(fill=tk.X)

                    tk.Label(info_frame, text=f"📦 {size_mb:.1f}MB" + (" ✓ Current" if is_current else ""),
                            font=("Arial", 8), fg="#27ae60" if is_current else "gray",
                            bg="#ecf0f1" if is_current else "white", anchor="w").pack(fill=tk.X)

                    def load_trained(path=model_path, name=train_dir.name):
                        self.model = YOLO(str(path))
                        self.model_name = f"Self-trained: {name}"
                        self.info_label.config(text=f"Model: {self.model_name} | Mode: {self.current_mode.title()}")
                        messagebox.showinfo("Success", f"Loaded: {name}")
                        dialog.destroy()

                    tk.Button(btn_frame, text="✅ Load" if is_current else "Load", command=load_trained,
                             font=("Arial", 9, "bold"), bg="#27ae60" if is_current else "#3498db",
                             fg="white", padx=15, pady=5).pack(side=tk.RIGHT, padx=10, pady=5)
        else:
            tk.Label(scrollable_frame, text="❌ No self-trained models found\n\nTrain: python training/train_local.py",
                    font=("Arial", 9), fg="gray", justify=tk.CENTER).pack(pady=10)

        # Category 2: Pretrained Models
        cat_frame = tk.LabelFrame(scrollable_frame, text="📦 Pretrained Models (COCO - 80 classes)",
                                 font=("Arial", 11, "bold"), padx=10, pady=10)
        cat_frame.pack(fill=tk.X, padx=5, pady=5)

        if pretrained_models:
            for model_path in sorted(pretrained_models):
                size_mb = model_path.stat().st_size / (1024 * 1024)
                
                btn_frame = tk.Frame(cat_frame, bg="white", relief=tk.RAISED, borderwidth=1)
                btn_frame.pack(fill=tk.X, pady=3)

                info_frame = tk.Frame(btn_frame, bg="white")
                info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=8)

                tk.Label(info_frame, text=model_path.name, font=("Arial", 10, "bold"),
                        bg="white", anchor="w").pack(fill=tk.X)

                tk.Label(info_frame, text=f"📦 {size_mb:.1f}MB", font=("Arial", 8),
                        fg="gray", bg="white", anchor="w").pack(fill=tk.X)

                def load_pretrained(path=model_path):
                    self.model = YOLO(str(path))
                    self.model_name = f"Pretrained: {path.name}"
                    self.info_label.config(text=f"Model: {self.model_name} | Mode: {self.current_mode.title()}")
                    messagebox.showinfo("Success", f"Loaded: {path.name}")
                    dialog.destroy()

                tk.Button(btn_frame, text="Load", command=load_pretrained, font=("Arial", 9, "bold"),
                         bg="#9b59b6", fg="white", padx=15, pady=5).pack(side=tk.RIGHT, padx=10, pady=5)

        # Download section
        download_frame = tk.Frame(cat_frame, bg="#e8f4f8", relief=tk.RAISED, borderwidth=1)
        download_frame.pack(fill=tk.X, pady=5)
        tk.Label(download_frame, text="⬇️ Download new pretrained model", font=("Arial", 9, "bold"), bg="#e8f4f8").pack(pady=5)
        tk.Button(download_frame, text="📥 Download", command=lambda: [dialog.destroy(), self.download_pretrained()],
                 font=("Arial", 9, "bold"), bg="#3498db", fg="white", padx=15, pady=5).pack(pady=5)

        # Bottom buttons
        bottom = tk.Frame(dialog, bg="#ecf0f1")
        bottom.pack(fill=tk.X, pady=5)
        tk.Button(bottom, text="🔍 Browse...", command=lambda: [dialog.destroy(), self.browse_for_model()],
                 font=("Arial", 10), bg="#95a5a6", fg="white", padx=15, pady=5).pack(side=tk.LEFT, padx=10)
        tk.Button(bottom, text="❌ Cancel", command=dialog.destroy,
                 font=("Arial", 10), bg="#e74c3c", fg="white", padx=15, pady=5).pack(side=tk.RIGHT, padx=10)
    
    def browse_for_model(self):
        """Browse for any .pt model file"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")],
            initialdir=str(Path(__file__).parent)
        )
        if file_path:
            self.model = YOLO(file_path)
            self.model_name = f"Custom: {Path(file_path).name}"
            self.info_label.config(text=f"Model: {self.model_name} | Mode: {self.current_mode.title()}")
            messagebox.showinfo("Success", f"Loaded: {Path(file_path).name}")

    def download_pretrained(self):
        """Download pretrained YOLO model"""
        models = {
            "YOLOv8 Nano (Fastest)": "yolov8n.pt",
            "YOLOv8 Small": "yolov8s.pt",
            "YOLOv8 Medium": "yolov8m.pt",
            "YOLOv8 Large": "yolov8l.pt",
            "YOLOv8 XLarge (Best)": "yolov8x.pt"
        }

        dialog = tk.Toplevel(self.root)
        dialog.title("Download Model")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Select model to download:", font=("Arial", 12, "bold"), pady=10).pack()
        tk.Label(dialog, text="(COCO trained - 80 classes)", font=("Arial", 9), fg="gray").pack()

        listbox = tk.Listbox(dialog, font=("Arial", 10), height=8)
        listbox.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        for name in models.keys():
            listbox.insert(tk.END, name)
        listbox.select_set(2)

        def on_download():
            sel = listbox.curselection()
            if sel:
                model_name = models[list(models.keys())[sel[0]]]
                dialog.destroy()

                progress = tk.Toplevel(self.root)
                progress.title("Downloading...")
                progress.geometry("350x80")
                progress.transient(self.root)
                progress.grab_set()
                tk.Label(progress, text=f"Downloading {model_name}...", font=("Arial", 10), pady=20).pack()
                self.root.update()

                try:
                    self.model = YOLO(model_name)
                    self.model_name = f"Pretrained: {model_name}"
                    progress.destroy()
                    messagebox.showinfo("Success", f"Downloaded: {model_name}")
                except Exception as e:
                    progress.destroy()
                    messagebox.showerror("Error", f"Download failed:\n{str(e)}")

        tk.Button(dialog, text="Download & Load", command=on_download, font=("Arial", 11, "bold"),
                 bg="#3498db", fg="white", pady=8).pack(pady=10)

    def create_ui(self):
        """Create the user interface"""
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="🐱 🐕 Cat vs Dog Detector", font=("Arial", 20, "bold"),
                bg="#2c3e50", fg="white").pack(pady=15)

        # Control Panel
        control = tk.Frame(self.root, bg="#ecf0f1", height=100)
        control.pack(fill=tk.X, padx=10, pady=5)
        control.pack_propagate(False)

        # Row 1
        row1 = tk.Frame(control, bg="#ecf0f1")
        row1.pack(pady=3)

        tk.Button(row1, text="📁 Load Image", command=self.load_image, font=("Arial", 10, "bold"),
                 bg="#3498db", fg="white", padx=12, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)
        tk.Button(row1, text="🎥 Load Video", command=self.load_video, font=("Arial", 10, "bold"),
                 bg="#9b59b6", fg="white", padx=12, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)
        
        self.detect_btn = tk.Button(row1, text="🔍 Detect", command=self.detect, font=("Arial", 10, "bold"),
                                    bg="#27ae60", fg="white", padx=12, pady=5, cursor="hand2", state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=4)
        
        self.play_btn = tk.Button(row1, text="▶️ Play", command=self.play_video, font=("Arial", 10, "bold"),
                                 bg="#16a085", fg="white", padx=12, pady=5, cursor="hand2", state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        
        self.stop_btn = tk.Button(row1, text="⏹️ Stop", command=self.stop_video_playback, font=("Arial", 10, "bold"),
                                 bg="#c0392b", fg="white", padx=12, pady=5, cursor="hand2", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        # Row 2
        row2 = tk.Frame(control, bg="#ecf0f1")
        row2.pack(pady=3)

        tk.Button(row2, text="🔄 Change Model", command=lambda: self.load_model(show_selector=True), font=("Arial", 10, "bold"),
                 bg="#9b59b6", fg="white", padx=12, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)
        
        self.save_btn = tk.Button(row2, text="💾 Save", command=self.save_result, font=("Arial", 10, "bold"),
                                 bg="#e67e22", fg="white", padx=12, pady=5, cursor="hand2", state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=4)
        
        tk.Button(row2, text="🗑️ Clear", command=self.clear_all, font=("Arial", 10, "bold"),
                 bg="#e74c3c", fg="white", padx=12, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)

        self.info_label = tk.Label(control, text=f"Model: {self.model_name} | Mode: Image",
                                   font=("Arial", 8), bg="#ecf0f1", fg="#7f8c8d")
        self.info_label.pack()

        # Images Display
        images_frame = tk.Frame(self.root, bg="#f0f0f1")
        images_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel
        left = tk.Frame(images_frame, bg="white", relief=tk.RAISED, borderwidth=2)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        tk.Label(left, text="📷 Original", font=("Arial", 11, "bold"), bg="white").pack(pady=5)
        self.original_canvas = tk.Canvas(left, bg="#ecf0f1")
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel
        right = tk.Frame(images_frame, bg="white", relief=tk.RAISED, borderwidth=2)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=3)
        tk.Label(right, text="🎯 Detection", font=("Arial", 11, "bold"), bg="white").pack(pady=5)
        self.annotated_canvas = tk.Canvas(right, bg="#ecf0f1")
        self.annotated_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results area
        results_frame = tk.Frame(self.root, bg="#ecf0f1")
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(results_frame, text="📊 Results:", font=("Arial", 10, "bold"), bg="#ecf0f1").pack(anchor=tk.W, padx=5)
        
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill=tk.X, padx=5, pady=2)
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(text_frame, height=8, font=("Courier", 9), bg="white",
                                    yscrollcommand=scrollbar.set)
        self.results_text.pack(fill=tk.X)
        scrollbar.config(command=self.results_text.yview)
        self.results_text.config(state=tk.DISABLED)

    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")],
            initialdir=str(Path(__file__).parent / "Pictures")
        )
        if file_path:
            self.current_image_path = file_path
            self.current_mode = "image"
            img = Image.open(file_path)
            self.original_image = img
            self.display_image(self.original_canvas, img)
            self.detect_btn.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.update_results("Image loaded. Click Detect to analyze.")
            self.annotated_canvas.delete("all")
            self.info_label.config(text=f"Model: {self.model_name} | Mode: Image")

    def load_video(self):
        """Load a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")],
            initialdir=str(Path(__file__).parent / "Pictures")
        )
        if file_path:
            self.current_video_path = file_path
            self.current_mode = "video"
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.display_image(self.original_canvas, img)
            cap.release()
            self.detect_btn.config(state=tk.DISABLED)
            self.play_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.update_results("Video loaded. Click Play to process.")
            self.annotated_canvas.delete("all")
            self.info_label.config(text=f"Model: {self.model_name} | Mode: Video")

    def detect(self):
        """Detect objects in image"""
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Load an image first!")
            return

        self.detect_btn.config(state=tk.DISABLED, text="⏳ Processing...")
        self.update_results("Running detection...")
        threading.Thread(target=self._run_detection, daemon=True).start()

    def _run_detection(self):
        """Run detection in thread"""
        try:
            results = self.model.predict(source=self.current_image_path, conf=0.25, save=False, verbose=False)
            annotated = results[0].plot()
            self.annotated_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            
            self.root.after(0, lambda: self.display_image(self.annotated_canvas, self.annotated_image))
            self.root.after(0, lambda: self._show_results(results[0]))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL, text="🔍 Detect"))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))

    def _show_results(self, result):
        """Display detection results"""
        text = "=" * 70 + "\n📊 DETECTION RESULTS\n" + "=" * 70 + "\n\n"
        boxes = result.boxes
        
        if len(boxes) == 0:
            text += "❌ No objects detected.\n"
        else:
            text += f"✅ Detected {len(boxes)} object(s):\n\n"
            counts = {}
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls]
                counts[name] = counts.get(name, 0) + 1
                emoji = "🐱" if "cat" in name.lower() else "🐕" if "dog" in name.lower() else "📦"
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                text += f"{i+1}. {emoji} {name.upper()}\n"
                text += f"   Confidence: {conf*100:.1f}%\n"
                text += f"   Position: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})\n\n"
            
            text += "=" * 70 + "\n📈 SUMMARY:\n"
            for name, count in counts.items():
                emoji = "🐱" if "cat" in name.lower() else "🐕" if "dog" in name.lower() else "📦"
                text += f"   {emoji} {name}: {count}\n"
            text += "=" * 70
        
        self.update_results(text)

    def play_video(self):
        """Play and process video"""
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
        if not self.current_video_path:
            messagebox.showwarning("Warning", "Load a video first!")
            return

        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.stop_video = False
        self.is_playing_video = True
        threading.Thread(target=self._process_video, daemon=True).start()

    def _process_video(self):
        """Process video in thread"""
        try:
            cap = cv2.VideoCapture(self.current_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1.0 / fps if fps > 0 else 0.033
            frame_num = 0
            detections = {}

            while cap.isOpened() and not self.stop_video:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                results = self.model.predict(source=frame, conf=0.25, save=False, verbose=False)
                annotated = results[0].plot()

                img_orig = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_ann = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                self.root.after(0, lambda io=img_orig, ia=img_ann: self._display_frames(io, ia))

                for box in results[0].boxes:
                    name = self.model.names[int(box.cls[0])]
                    detections[name] = detections.get(name, 0) + 1

                if frame_num % 10 == 0:
                    self.root.after(0, lambda f=frame_num, d=dict(detections): self._update_video_stats(f, d))

                time.sleep(delay)

            cap.release()
            self.root.after(0, lambda f=frame_num, d=detections: self._video_done(f, d))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Video processing failed:\n{str(e)}"))
        finally:
            self.is_playing_video = False
            self.root.after(0, self._reset_video_buttons)

    def _display_frames(self, orig, ann):
        """Display video frames"""
        self.display_image(self.original_canvas, orig)
        self.display_image(self.annotated_canvas, ann)

    def _update_video_stats(self, frame, detections):
        """Update video stats"""
        text = f"🎥 Processing frame {frame}...\n\nDetections:\n"
        for name, count in detections.items():
            emoji = "🐱" if "cat" in name.lower() else "🐕" if "dog" in name.lower() else "📦"
            text += f"  {emoji} {name}: {count}\n"
        self.update_results(text)

    def _video_done(self, frames, detections):
        """Video processing complete"""
        text = "=" * 70 + "\n🎥 VIDEO COMPLETE\n" + "=" * 70 + f"\n\nFrames: {frames}\n\n"
        if detections:
            text += "Total detections:\n"
            for name, count in detections.items():
                emoji = "🐱" if "cat" in name.lower() else "🐕" if "dog" in name.lower() else "📦"
                text += f"  {emoji} {name}: {count}\n"
        else:
            text += "No objects detected.\n"
        text += "\n" + "=" * 70
        self.update_results(text)

    def _reset_video_buttons(self):
        """Reset video buttons"""
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def stop_video_playback(self):
        """Stop video playback"""
        self.stop_video = True
        self.update_results("\n⏹️ Video stopped.\n")

    def display_image(self, canvas, image):
        """Display image on canvas"""
        canvas.update()
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        iw, ih = image.size
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)

        canvas.delete("all")
        canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor=tk.NW, image=photo)
        canvas.image = photo

    def save_result(self):
        """Save annotated result"""
        if not self.annotated_image:
            messagebox.showwarning("Warning", "No result to save!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All", "*.*")],
            initialdir=str(Path(__file__).parent / "output" / "test_results")
        )
        if file_path:
            try:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                self.annotated_image.save(file_path)
                messagebox.showinfo("Success", f"Saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{str(e)}")

    def clear_all(self):
        """Clear everything"""
        self.current_image_path = None
        self.current_video_path = None
        self.original_image = None
        self.annotated_image = None
        self.original_canvas.delete("all")
        self.annotated_canvas.delete("all")
        self.detect_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.update_results("Cleared. Load image or video to start.")
        self.info_label.config(text=f"Model: {self.model_name} | Mode: Image")

    def update_results(self, text):
        """Update results text area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = CatDogDetectorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

