"""
NON-INTERACTIVE Training Script for Google Colab
Runs with optimal default settings for YOLOv8 Medium
"""

import os
import platform
import torch
import cv2
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# Reduce thread oversubscription
cv2.setNumThreads(0)
os.environ.setdefault("OMP_NUM_THREADS", "1")


def train_colab_default():
    """Train with optimal default parameters for Colab"""

    print("\n" + "=" * 70)
    print("🎯 AUTOMATIC TRAINING - CATS VS DOGS (COLAB)")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device = 0
    else:
        print("\n⚠️ GPU not available! Using CPU (MUCH slower)")
        device = 'cpu'

    # Dataset
    project_root = Path(__file__).parent
    data_yaml = project_root / "Data_set_Cat_vs_Dog" / "yolo_data" / "data.yaml"

    if not data_yaml.exists():
        print(f"\n❌ Error: Dataset not found at: {data_yaml}")
        return

    print(f"\n✅ Dataset: {data_yaml}")

    # OPTIMAL DEFAULT CONFIGURATION FOR COLAB
    model_file = "yolov8m.pt"
    model_name = "medium"
    epochs = 150
    batch = 8
    lr0 = 0.005
    patience = 75

    # Workers and cache for Colab (Linux)
    num_workers = 4 if platform.system() != "Windows" else 0
    cache_mode = True  # RAM cache on Colab

    print("\n" + "=" * 70)
    print("📋 AUTOMATIC CONFIGURATION")
    print("=" * 70)
    print(f"  Model:       YOLOv8-{model_name}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch:       {batch}")
    print(f"  LR:          {lr0}")
    print(f"  Patience:    {patience}")
    print(f"  Device:      {'GPU (CUDA)' if device == 0 else 'CPU'}")
    print(f"  Workers:     {num_workers}")
    print(f"  Cache:       {cache_mode}")
    print(f"  Target mAP:  95-97%")

    # Duration estimate
    est_hours = (epochs * 0.8 * (16 / batch)) / 60
    print(f"  Duration ~:  {int(est_hours * 60)} minutes ({est_hours:.1f} hours)")

    # Load model
    print(f"\n📥 Loading {model_file}...")
    model = YOLO(model_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'colab_{model_name}_{timestamp}'

    print("\n" + "=" * 70)
    print("🚀 TRAINING STARTS")
    print("=" * 70)
    print(f"\n📁 Results: runs/train/{run_name}/\n")

    try:
        # TRAINING
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=batch,
            device=device,
            # Stability/performance
            workers=num_workers,
            amp=True,
            cache=cache_mode,
            # Run metadata
            project='runs/train',
            name=run_name,
            exist_ok=True,
            save=True,
            save_period=max(5, epochs // 20),
            # Optimization
            patience=patience,
            optimizer='AdamW',
            lr0=lr0,
            lrf=lr0 / 10,
            momentum=0.95,
            weight_decay=0.001,
            warmup_epochs=min(5.0, epochs * 0.05),
            # Augmentation
            hsv_h=0.02,
            hsv_s=0.8,
            hsv_v=0.5,
            degrees=10.0,
            translate=0.15,
            scale=0.7,
            shear=2.0,
            perspective=0.0001,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.05,
            label_smoothing=0.1,
            # Others
            cos_lr=True,
            close_mosaic=min(15, epochs // 10),
            deterministic=True,
            verbose=True,
            seed=42,
            plots=True,
            val=True,
        )

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED!")
        print("=" * 70)

        model_path = f'runs/train/{run_name}'

        # Copy model
        import shutil
        trained_dir = project_root / "models" / "trained"
        trained_dir.mkdir(parents=True, exist_ok=True)

        best_src = Path(f'{model_path}/weights/best.pt')
        best_dst = trained_dir / f'colab_{model_name}_e{epochs}_b{batch}_{timestamp}.pt'

        if best_src.exists():
            shutil.copy2(best_src, best_dst)
            print(f"\n📁 Model saved: {best_dst}")

        # Test validation
        print("\n🔍 Validation on test set...")
        best_model = YOLO(str(best_src))
        test_results = best_model.val(data=str(data_yaml), split='test', device=device)

        metrics = test_results.results_dict
        mAP50 = metrics.get('metrics/mAP50(B)', 0)
        mAP50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)

        print("\n" + "=" * 70)
        print("📊 FINAL RESULTS")
        print("=" * 70)
        print(f"\n  mAP50:     {mAP50:.4f} {'🎯 EXCELLENT!' if mAP50 > 0.95 else '✅ Good!' if mAP50 > 0.90 else '👍 OK'}")
        print(f"  mAP50-95:  {mAP50_95:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")

        print("\n" + "=" * 70)
        print("🎉 SUCCESS!")
        print("=" * 70)
        print(f"\n💡 Use the model:")
        print(f"   model = YOLO('{best_dst}')")
        print(f"   results = model.predict('image.jpg')\n")

        return str(best_dst)

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Antrenare oprită! Progres salvat în: runs/train/{run_name}/")
    except Exception as e:
        print(f"\n❌ Eroare: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_colab_default()

