"""
Training Results Viewer - Vizualizare Grafice și Rezultate Antrenare

Acest script afișează toate graficele și rezultatele antrenării într-o interfață grafică.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import yaml


class TrainingResultsViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Training Results Viewer - Grafice Antrenare")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")

        # Find training runs
        self.project_root = Path(__file__).parent
        self.runs_dir = self.project_root / "runs" / "train"
        self.training_runs = []
        self.current_run = None

        self.find_training_runs()
        self.create_ui()

        if self.training_runs:
            self.load_run(self.training_runs[0])

    def find_training_runs(self):
        """Find all training runs"""
        if not self.runs_dir.exists():
            return

        self.training_runs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        self.training_runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    def create_ui(self):
        """Create the graphical interface"""

        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="📊 Training Results Viewer",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)

        # Control Panel
        control_frame = tk.Frame(self.root, bg="#ecf0f1", height=80)
        control_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=5)
        control_frame.pack_propagate(False)

        # Training run selector
        selector_frame = tk.Frame(control_frame, bg="#ecf0f1")
        selector_frame.pack(pady=10)

        tk.Label(
            selector_frame,
            text="Select Training Run:",
            font=("Arial", 11, "bold"),
            bg="#ecf0f1"
        ).pack(side=tk.LEFT, padx=5)

        self.run_selector = ttk.Combobox(
            selector_frame,
            values=[run.name for run in self.training_runs],
            state="readonly",
            width=50,
            font=("Arial", 10)
        )
        self.run_selector.pack(side=tk.LEFT, padx=5)
        if self.training_runs:
            self.run_selector.current(0)
        self.run_selector.bind("<<ComboboxSelected>>", self.on_run_selected)

        # Refresh button
        refresh_btn = tk.Button(
            selector_frame,
            text="🔄 Refresh",
            command=self.refresh_runs,
            font=("Arial", 10, "bold"),
            bg="#3498db",
            fg="white",
            padx=10,
            pady=5,
            cursor="hand2"
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)

        # Open folder button
        open_folder_btn = tk.Button(
            selector_frame,
            text="📁 Open Folder",
            command=self.open_folder,
            font=("Arial", 10, "bold"),
            bg="#27ae60",
            fg="white",
            padx=10,
            pady=5,
            cursor="hand2"
        )
        open_folder_btn.pack(side=tk.LEFT, padx=5)

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Overview
        self.overview_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.overview_tab, text="📋 Overview")

        # Tab 2: Training Graphs
        self.graphs_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.graphs_tab, text="📈 Training Graphs")

        # Tab 3: Training Images
        self.images_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.images_tab, text="🖼️ Training Images")

        # Tab 4: Configuration
        self.config_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.config_tab, text="⚙️ Configuration")

        self.setup_tabs()

    def setup_tabs(self):
        """Setup all tabs"""
        # Overview tab
        self.overview_text = tk.Text(
            self.overview_tab,
            font=("Courier", 10),
            bg="white",
            wrap=tk.WORD
        )
        overview_scroll = tk.Scrollbar(self.overview_tab, command=self.overview_text.yview)
        self.overview_text.config(yscrollcommand=overview_scroll.set)
        overview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.overview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Graphs tab
        self.graphs_canvas_frame = tk.Frame(self.graphs_tab, bg="white")
        self.graphs_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Images tab
        images_scroll = tk.Scrollbar(self.images_tab)
        images_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.images_canvas = tk.Canvas(
            self.images_tab,
            bg="white",
            yscrollcommand=images_scroll.set
        )
        self.images_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        images_scroll.config(command=self.images_canvas.yview)

        self.images_inner_frame = tk.Frame(self.images_canvas, bg="white")
        self.images_canvas_window = self.images_canvas.create_window(
            (0, 0),
            window=self.images_inner_frame,
            anchor="nw"
        )

        self.images_inner_frame.bind(
            "<Configure>",
            lambda e: self.images_canvas.configure(scrollregion=self.images_canvas.bbox("all"))
        )

        # Config tab
        self.config_text = tk.Text(
            self.config_tab,
            font=("Courier", 9),
            bg="white",
            wrap=tk.NONE
        )
        config_scroll_y = tk.Scrollbar(self.config_tab, command=self.config_text.yview)
        config_scroll_x = tk.Scrollbar(self.config_tab, orient=tk.HORIZONTAL, command=self.config_text.xview)
        self.config_text.config(yscrollcommand=config_scroll_y.set, xscrollcommand=config_scroll_x.set)
        config_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        config_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.config_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def on_run_selected(self, event=None):
        """Handle run selection"""
        selected = self.run_selector.get()
        for run in self.training_runs:
            if run.name == selected:
                self.load_run(run)
                break

    def refresh_runs(self):
        """Refresh training runs list"""
        self.find_training_runs()
        self.run_selector['values'] = [run.name for run in self.training_runs]
        if self.training_runs:
            self.run_selector.current(0)
            self.load_run(self.training_runs[0])

    def open_folder(self):
        """Open current training folder"""
        if self.current_run:
            import os
            import subprocess
            if os.name == 'nt':  # Windows
                os.startfile(self.current_run)
            elif os.name == 'posix':  # Linux/Mac
                subprocess.run(['xdg-open', str(self.current_run)])

    def load_run(self, run_path):
        """Load training run data"""
        self.current_run = run_path

        # Clear previous data
        self.overview_text.delete(1.0, tk.END)
        self.config_text.delete(1.0, tk.END)

        # Load overview
        self.load_overview(run_path)

        # Load graphs
        self.load_graphs(run_path)

        # Load training images
        self.load_training_images(run_path)

        # Load configuration
        self.load_config(run_path)

    def load_overview(self, run_path):
        """Load overview information"""
        self.overview_text.insert(tk.END, "=" * 80 + "\n")
        self.overview_text.insert(tk.END, f"📊 TRAINING RUN: {run_path.name}\n")
        self.overview_text.insert(tk.END, "=" * 80 + "\n\n")

        # Check for results.csv
        results_csv = run_path / "results.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()

            self.overview_text.insert(tk.END, "📈 TRAINING SUMMARY\n")
            self.overview_text.insert(tk.END, "-" * 80 + "\n\n")

            # Basic info
            self.overview_text.insert(tk.END, f"Total Epochs: {len(df)}\n")

            # Get last row metrics
            if len(df) > 0:
                last_row = df.iloc[-1]

                self.overview_text.insert(tk.END, f"\n📊 FINAL METRICS (Epoch {len(df)}):\n")
                self.overview_text.insert(tk.END, "-" * 80 + "\n")

                # Training metrics
                if 'train/box_loss' in df.columns:
                    self.overview_text.insert(tk.END, f"Train Box Loss:     {last_row['train/box_loss']:.4f}\n")
                if 'train/cls_loss' in df.columns:
                    self.overview_text.insert(tk.END, f"Train Class Loss:   {last_row['train/cls_loss']:.4f}\n")
                if 'train/dfl_loss' in df.columns:
                    self.overview_text.insert(tk.END, f"Train DFL Loss:     {last_row['train/dfl_loss']:.4f}\n")

                self.overview_text.insert(tk.END, "\n")

                # Validation metrics
                if 'metrics/precision(B)' in df.columns:
                    self.overview_text.insert(tk.END, f"Precision:          {last_row['metrics/precision(B)']:.4f} ({last_row['metrics/precision(B)']*100:.2f}%)\n")
                if 'metrics/recall(B)' in df.columns:
                    self.overview_text.insert(tk.END, f"Recall:             {last_row['metrics/recall(B)']:.4f} ({last_row['metrics/recall(B)']*100:.2f}%)\n")
                if 'metrics/mAP50(B)' in df.columns:
                    self.overview_text.insert(tk.END, f"mAP@50:             {last_row['metrics/mAP50(B)']:.4f} ({last_row['metrics/mAP50(B)']*100:.2f}%)\n")
                if 'metrics/mAP50-95(B)' in df.columns:
                    self.overview_text.insert(tk.END, f"mAP@50-95:          {last_row['metrics/mAP50-95(B)']:.4f} ({last_row['metrics/mAP50-95(B)']*100:.2f}%)\n")

                # Best metrics
                self.overview_text.insert(tk.END, f"\n🏆 BEST METRICS:\n")
                self.overview_text.insert(tk.END, "-" * 80 + "\n")

                if 'metrics/precision(B)' in df.columns:
                    best_precision = df['metrics/precision(B)'].max()
                    best_precision_epoch = df['metrics/precision(B)'].idxmax() + 1
                    self.overview_text.insert(tk.END, f"Best Precision:     {best_precision:.4f} ({best_precision*100:.2f}%) at epoch {best_precision_epoch}\n")

                if 'metrics/recall(B)' in df.columns:
                    best_recall = df['metrics/recall(B)'].max()
                    best_recall_epoch = df['metrics/recall(B)'].idxmax() + 1
                    self.overview_text.insert(tk.END, f"Best Recall:        {best_recall:.4f} ({best_recall*100:.2f}%) at epoch {best_recall_epoch}\n")

                if 'metrics/mAP50(B)' in df.columns:
                    best_map50 = df['metrics/mAP50(B)'].max()
                    best_map50_epoch = df['metrics/mAP50(B)'].idxmax() + 1
                    self.overview_text.insert(tk.END, f"Best mAP@50:        {best_map50:.4f} ({best_map50*100:.2f}%) at epoch {best_map50_epoch}\n")

                if 'metrics/mAP50-95(B)' in df.columns:
                    best_map5095 = df['metrics/mAP50-95(B)'].max()
                    best_map5095_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1
                    self.overview_text.insert(tk.END, f"Best mAP@50-95:     {best_map5095:.4f} ({best_map5095*100:.2f}%) at epoch {best_map5095_epoch}\n")

        else:
            self.overview_text.insert(tk.END, "❌ No results.csv found\n")

        # Check for weights
        self.overview_text.insert(tk.END, f"\n📦 MODEL WEIGHTS:\n")
        self.overview_text.insert(tk.END, "-" * 80 + "\n")

        weights_dir = run_path / "weights"
        if weights_dir.exists():
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"

            if best_pt.exists():
                size_mb = best_pt.stat().st_size / (1024 * 1024)
                self.overview_text.insert(tk.END, f"✅ best.pt - {size_mb:.2f} MB\n")
            else:
                self.overview_text.insert(tk.END, "❌ best.pt - Not found\n")

            if last_pt.exists():
                size_mb = last_pt.stat().st_size / (1024 * 1024)
                self.overview_text.insert(tk.END, f"✅ last.pt - {size_mb:.2f} MB\n")
            else:
                self.overview_text.insert(tk.END, "❌ last.pt - Not found\n")
        else:
            self.overview_text.insert(tk.END, "❌ weights/ folder not found\n")

        self.overview_text.insert(tk.END, f"\n📁 LOCATION:\n")
        self.overview_text.insert(tk.END, "-" * 80 + "\n")
        self.overview_text.insert(tk.END, f"{run_path}\n")

    def load_graphs(self, run_path):
        """Load and display training graphs"""
        # Clear previous graphs
        for widget in self.graphs_canvas_frame.winfo_children():
            widget.destroy()

        results_csv = run_path / "results.csv"
        if not results_csv.exists():
            tk.Label(
                self.graphs_canvas_frame,
                text="❌ No results.csv found - Cannot generate graphs",
                font=("Arial", 12),
                bg="white"
            ).pack(pady=50)
            return

        # Load data
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Results: {run_path.name}', fontsize=14, fontweight='bold')

        epochs = range(1, len(df) + 1)

        # Plot 1: Losses
        ax1 = axes[0, 0]
        if 'train/box_loss' in df.columns:
            ax1.plot(epochs, df['train/box_loss'], label='Box Loss', marker='o', markersize=2)
        if 'train/cls_loss' in df.columns:
            ax1.plot(epochs, df['train/cls_loss'], label='Class Loss', marker='s', markersize=2)
        if 'train/dfl_loss' in df.columns:
            ax1.plot(epochs, df['train/dfl_loss'], label='DFL Loss', marker='^', markersize=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Precision & Recall
        ax2 = axes[0, 1]
        if 'metrics/precision(B)' in df.columns:
            ax2.plot(epochs, df['metrics/precision(B)'], label='Precision', marker='o', markersize=2, color='green')
        if 'metrics/recall(B)' in df.columns:
            ax2.plot(epochs, df['metrics/recall(B)'], label='Recall', marker='s', markersize=2, color='blue')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision & Recall')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # Plot 3: mAP
        ax3 = axes[1, 0]
        if 'metrics/mAP50(B)' in df.columns:
            ax3.plot(epochs, df['metrics/mAP50(B)'], label='mAP@50', marker='o', markersize=2, color='purple')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax3.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@50-95', marker='s', markersize=2, color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('mAP')
        ax3.set_title('Mean Average Precision (mAP)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

        # Plot 4: All metrics combined
        ax4 = axes[1, 1]
        if 'metrics/precision(B)' in df.columns:
            ax4.plot(epochs, df['metrics/precision(B)'], label='Precision', marker='o', markersize=1, alpha=0.7)
        if 'metrics/recall(B)' in df.columns:
            ax4.plot(epochs, df['metrics/recall(B)'], label='Recall', marker='s', markersize=1, alpha=0.7)
        if 'metrics/mAP50(B)' in df.columns:
            ax4.plot(epochs, df['metrics/mAP50(B)'], label='mAP@50', marker='^', markersize=1, alpha=0.7)
        if 'metrics/mAP50-95(B)' in df.columns:
            ax4.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@50-95', marker='d', markersize=1, alpha=0.7)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.set_title('All Metrics Combined')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])

        plt.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graphs_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_training_images(self, run_path):
        """Load training sample images"""
        # Clear previous images
        for widget in self.images_inner_frame.winfo_children():
            widget.destroy()

        # Find image files
        image_files = []

        # Look for specific images
        for img_name in ['labels.jpg', 'train_batch0.jpg', 'train_batch1.jpg', 'train_batch2.jpg']:
            img_path = run_path / img_name
            if img_path.exists():
                image_files.append((img_name, img_path))

        if not image_files:
            tk.Label(
                self.images_inner_frame,
                text="❌ No training images found",
                font=("Arial", 12),
                bg="white"
            ).pack(pady=50)
            return

        # Display images
        for idx, (name, img_path) in enumerate(image_files):
            # Image title
            title_label = tk.Label(
                self.images_inner_frame,
                text=f"📷 {name}",
                font=("Arial", 12, "bold"),
                bg="white"
            )
            title_label.pack(pady=(10 if idx == 0 else 20, 5))

            # Load and display image
            try:
                img = Image.open(img_path)

                # Resize if too large
                max_width = 1200
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                photo = ImageTk.PhotoImage(img)

                img_label = tk.Label(self.images_inner_frame, image=photo, bg="white")
                img_label.image = photo  # Keep reference
                img_label.pack(pady=5)

            except Exception as e:
                error_label = tk.Label(
                    self.images_inner_frame,
                    text=f"❌ Error loading image: {e}",
                    font=("Arial", 10),
                    fg="red",
                    bg="white"
                )
                error_label.pack(pady=5)

    def load_config(self, run_path):
        """Load training configuration"""
        args_yaml = run_path / "args.yaml"

        if not args_yaml.exists():
            self.config_text.insert(tk.END, "❌ No args.yaml found\n")
            return

        try:
            with open(args_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            self.config_text.insert(tk.END, "=" * 80 + "\n")
            self.config_text.insert(tk.END, "⚙️ TRAINING CONFIGURATION\n")
            self.config_text.insert(tk.END, "=" * 80 + "\n\n")

            # Format and display config
            for key, value in config.items():
                self.config_text.insert(tk.END, f"{key}: {value}\n")

        except Exception as e:
            self.config_text.insert(tk.END, f"❌ Error loading config: {e}\n")


def main():
    """Main function"""
    root = tk.Tk()
    app = TrainingResultsViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

