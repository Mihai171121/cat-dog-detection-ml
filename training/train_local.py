"""
Script de antrenare CONFIGURABIL - TU ALEGI PARAMETRII!
RTX 3060 (6GB VRAM) - High Accuracy
"""

import os
import platform
import torch
import cv2
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# Reduce thread oversubscription for OpenCV/PyTorch to lower RAM pressure on Windows
cv2.setNumThreads(0)
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _default_workers() -> int:
    """Choose a conservative number of DataLoader workers, especially for Windows.
    Windows DataLoader workers can duplicate memory; using fewer avoids OpenCV OOMs.
    """
    try:
        cpus = os.cpu_count() or 2
    except Exception:
        cpus = 2
    if platform.system().lower().startswith("win"):
        # Most stable on Windows is 0-2 workers; prefer 0 for safety on low-RAM systems
        return 0 if cpus <= 8 else 2
    # On Linux/macOS a few workers are fine
    return min(4, max(0, cpus // 2))


def _default_cache_mode() -> str | bool:
    """Use disk caching on Windows to avoid large RAM usage; RAM elsewhere."""
    return "disk" if platform.system().lower().startswith("win") else True


def _is_oom_error(e: Exception) -> bool:
    """Heuristic to detect OOM-like errors from OpenCV/DataLoader on Windows."""
    msg = (str(e) or "").lower()
    signs = [
        "insufficient memory",
        "out of memory",
        "outofmemory",
        "failed to allocate",
        "cv::outofmemory",
        "worker process",
        "dataloader worker",
        "winerror 1455",  # paging file too small
    ]
    return any(s in msg for s in signs)


def train_custom():
    """Antrenează cu parametri alesi de tine interactiv"""

    print("\n" + "=" * 70)
    print("🎯 ANTRENARE CONFIGURABILĂ - PISICI VS CÂINI")
    print("Alege tu parametrii pentru antrenare!")
    print("=" * 70)

    # Verifică GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device = 0
    else:
        print("\n⚠️ GPU nu este disponibil! Folosim CPU (MULT mai lent)")
        device = 'cpu'

    # Dataset
    project_root = Path(__file__).parent.parent  # Go up to project root
    data_yaml = project_root / "Data_set_Cat_vs_Dog" / "yolo_data" / "data.yaml"

    if not data_yaml.exists():
        print(f"\n❌ Eroare: Dataset nu găsit la: {data_yaml}")
        return

    print(f"\n✅ Dataset: {data_yaml}")

    # ========================================================================
    # CONFIGURARE INTERACTIVĂ - ALEGI TU!
    # ========================================================================

    print("\n" + "=" * 70)
    print("⚙️  CONFIGURARE PARAMETRI")
    print("=" * 70)

    # 1. Selectează modelul
    print("\n📦 ALEGE MODELUL YOLOv8:")
    print("  1. YOLOv8n (nano)   - ~6MB,  rapid,      mAP ~85%")
    print("  2. YOLOv8s (small)  - ~22MB, echilibru,  mAP ~90%")
    print("  3. YOLOv8m (medium) - ~50MB, precis,     mAP ~95% ⭐ RECOMANDAT")
    print("  4. YOLOv8l (large)  - ~87MB, foarte precis, mAP ~96%")
    print("  5. YOLOv8x (xlarge) - ~136MB, maxim,     mAP ~97% (necesită >6GB)")

    model_choice = input("\nSelectează modelul (1-5, Enter=3): ").strip() or "3"

    model_map = {
        "1": ("yolov8n.pt", "nano", "85-88%", 16),
        "2": ("yolov8s.pt", "small", "90-93%", 12),
        "3": ("yolov8m.pt", "medium", "95-97%", 8),
        "4": ("yolov8l.pt", "large", "96-98%", 6),
        "5": ("yolov8x.pt", "xlarge", "97-99%", 4)
    }

    model_file, model_name, expected_map, rec_batch = model_map.get(model_choice, model_map["3"])

    pretrained_model = project_root / "models" / "pretrained" / model_file
    if not pretrained_model.exists():
        print(f"\n📥 Descărcare {model_file}...")
        pretrained_model = model_file
    else:
        pretrained_model = str(pretrained_model)

    print(f"✅ Model: YOLOv8-{model_name} (mAP așteptat: {expected_map})")

    # 2. Selectează EPOCILE
    print("\n" + "-" * 70)
    print("📊 ALEGE NUMĂRUL DE EPOCI:")
    print("  Recomandări:")
    print("  • 50-100:   Test rapid      (20-60 min)")
    print("  • 100-150:  Standard        (60-90 min)")
    print("  • 150-200:  High Accuracy   (90-120 min) ⭐ RECOMANDAT")
    print("  • 200-300:  Max Accuracy    (2-3 ore)")

    epochs_input = input(f"\nNumăr epoci (Enter=200): ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() and int(epochs_input) > 0 else 200

    print(f"✅ Epoci: {epochs}")

    # 3. Selectează BATCH SIZE
    print("\n" + "-" * 70)
    print("📦 ALEGE BATCH SIZE:")
    print(f"  RTX 3060 (6GB) - Recomandare pentru YOLOv8-{model_name}:")
    print(f"  • Batch 4:  Maxim precizie  (lent, ~3GB VRAM)")
    print(f"  • Batch 8:  High accuracy   (recomandat, ~4.5GB) ⭐")
    print(f"  • Batch 12: Echilibru       (~5.5GB)")
    print(f"  • Batch 16: Rapid           (~5.8GB, risc OOM)")

    batch_input = input(f"\nBatch size (Enter={rec_batch}): ").strip()
    batch = int(batch_input) if batch_input.isdigit() and int(batch_input) > 0 else rec_batch

    if batch > 16:
        print(f"⚠️  Batch {batch} este prea mare! Risc 'Out of Memory'. Setez la 16.")
        batch = 16

    print(f"✅ Batch: {batch}")

    # 4. Learning Rate (opțional - pentru avansați)
    print("\n" + "-" * 70)
    print("🎓 LEARNING RATE (Enter pentru default optim):")
    print("  • 0.005: Fin, precis      ⭐ RECOMANDAT pentru accuracy")
    print("  • 0.01:  Standard")
    print("  • 0.02:  Rapid")

    lr_input = input("\nLearning rate (Enter=0.005): ").strip()
    lr0 = float(lr_input) if lr_input else 0.005

    print(f"✅ Learning rate: {lr0}")

    # 5. Early Stopping
    patience = min(100, epochs // 2)
    print(f"\n✅ Early stopping: {patience} epoci (oprire auto dacă nu progresează)")

    # Stabilitate DataLoader (Windows): muncitori puțini + cache pe disk
    num_workers = _default_workers()
    cache_mode = _default_cache_mode()
    print("\n" + "-" * 70)
    print("🧠 OPTIMIZARE STABILITATE (DataLoader)")
    print(f"  • workers: {num_workers}  (Windows recomandat <=2)")
    print(f"  • cache:   {cache_mode}")

    # REZUMAT
    print("\n" + "=" * 70)
    print("📋 CONFIGURAȚIE FINALĂ")
    print("=" * 70)
    print(f"  Model:       YOLOv8-{model_name}")
    print(f"  Epoci:       {epochs}")
    print(f"  Batch:       {batch}")
    print(f"  LR:          {lr0}")
    print(f"  Patience:    {patience}")
    print(f"  Device:      {'GPU (CUDA)' if device == 0 else 'CPU'}")
    print(f"  mAP țintă:   {expected_map}")
    print(f"  Workers:     {num_workers}")
    print(f"  Cache:       {cache_mode}")

    # Estimare durată
    time_map = {"nano": 0.3, "small": 0.5, "medium": 0.8, "large": 1.2, "xlarge": 1.8}
    est_hours = (epochs * time_map[model_name] * (16 / batch)) / 60
    print(f"  Durată ~:    {int(est_hours * 60)} minute ({est_hours:.1f} ore)")

    # Confirmare
    print("\n" + "=" * 70)
    confirm = input("🚀 Pornesc antrenarea? (da/Enter): ").strip().lower()
    if confirm and confirm not in ['da', 'yes', 'y', 'd', '']:
        print("❌ Antrenare anulată.")
        return

    # Încarcă model
    print(f"\n📥 Încărcare {model_file}...")
    model = YOLO(pretrained_model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'custom_{model_name}_{timestamp}'
    run_dir = Path('runs/train') / run_name

    print("\n" + "=" * 70)
    print("🚀 ÎNCEPE ANTRENAREA")
    print("=" * 70)
    print(f"\n📁 Rezultate: runs/train/{run_name}/")
    print("💡 Ctrl+C pentru a opri (progresul se salvează automat)\n")

    def _train_with_settings(
        resume: bool = False,
        override_workers: int | None = None,
        override_cache: str | bool | None = None,
        override_batch: int | None = None,
        override_imgsz: int | None = None,
        aug_overrides: dict | None = None,
    ):
        # Local knobs
        _workers = override_workers if override_workers is not None else num_workers
        _cache = override_cache if override_cache is not None else cache_mode
        _batch = override_batch if override_batch is not None else batch
        _imgsz = override_imgsz if override_imgsz is not None else 640

        # Default augment values
        aug = {
            'hsv_h': 0.02,
            'hsv_s': 0.8,
            'hsv_v': 0.5,
            'degrees': 10.0,
            'translate': 0.15,
            'scale': 0.7,
            'shear': 2.0,
            'perspective': 0.0001,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.05,
            'label_smoothing': 0.1,
        }
        if aug_overrides:
            aug.update(aug_overrides)

        return model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=_imgsz,
            batch=_batch,
            device=device,
            # Stabilitate/performanță
            workers=_workers,
            amp=True,
            cache=_cache,
            # Run metadata
            project='runs/train',
            name=run_name,
            exist_ok=True,
            save=True,
            save_period=max(5, epochs // 20),
            # Optimizare
            patience=patience,
            optimizer='AdamW',
            lr0=lr0,
            lrf=lr0 / 10,
            momentum=0.95,
            weight_decay=0.001,
            warmup_epochs=min(5.0, epochs * 0.05),
            # Augmentare
            hsv_h=aug['hsv_h'],
            hsv_s=aug['hsv_s'],
            hsv_v=aug['hsv_v'],
            degrees=aug['degrees'],
            translate=aug['translate'],
            scale=aug['scale'],
            shear=aug['shear'],
            perspective=aug['perspective'],
            fliplr=aug['fliplr'],
            mosaic=aug['mosaic'],
            mixup=aug['mixup'],
            copy_paste=aug['copy_paste'],
            label_smoothing=aug['label_smoothing'],
            # Altele
            cos_lr=True,
            close_mosaic=min(15, epochs // 10),
            deterministic=True,
            verbose=True,
            seed=42,
            plots=True,
            val=True,
            resume=resume,
        )

    try:
        # ANTRENARE
        results = _train_with_settings()

        print("\n" + "=" * 70)
        print("✅ ANTRENARE FINALIZATĂ!")
        print("=" * 70)

        model_path = f'runs/train/{run_name}'

        # Copiere model
        import shutil
        trained_dir = project_root / "models" / "trained"
        trained_dir.mkdir(parents=True, exist_ok=True)

        best_src = Path(f'{model_path}/weights/best.pt')
        best_dst = trained_dir / f'custom_{model_name}_e{epochs}_b{batch}_{timestamp}.pt'

        if best_src.exists():
            shutil.copy2(best_src, best_dst)
            print(f"\n📁 Model salvat: {best_dst}")

        # Validare test
        print("\n🔍 Validare pe test set...")
        best_model = YOLO(str(best_src))
        test_results = best_model.val(data=str(data_yaml), split='test', device=device)

        metrics = test_results.results_dict
        mAP50 = metrics.get('metrics/mAP50(B)', 0)
        mAP50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)

        print("\n" + "=" * 70)
        print("📊 REZULTATE FINALE")
        print("=" * 70)
        print(f"\n  mAP50:     {mAP50:.4f} {'🎯 EXCELENT!' if mAP50 > 0.95 else '✅ Bun!' if mAP50 > 0.90 else '👍 OK'}")
        print(f"  mAP50-95:  {mAP50_95:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")

        print("\n" + "=" * 70)
        print("🎉 SUCCES!")
        print("=" * 70)
        print(f"\n💡 Folosește modelul:")
        print(f"   model = YOLO('{best_dst}')")
        print(f"   results = model.predict('imagine.jpg')\n")

        return str(best_dst)

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Antrenare oprită! Progres salvat în: runs/train/{run_name}/")
    except cv2.error as e:
        # Auto-recovery for OpenCV RAM OOM in DataLoader workers
        print(f"\n❌ Eroare OpenCV: {e}")
        if _is_oom_error(e):
            print("\n🔁 Încerc reluarea cu setări mai sigure: workers=0, cache='disk', batch redus, imgsz 512, augment minim...")
            try:
                safer_batch = max(2, batch // 2)
                aug_overrides = {
                    'mosaic': 0.0,
                    'mixup': 0.0,
                    'copy_paste': 0.0,
                    'hsv_s': 0.5,
                    'hsv_v': 0.4,
                    'degrees': 5.0,
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 1.0,
                    'perspective': 0.00005,
                }
                _ = _train_with_settings(
                    resume=True,
                    override_workers=0,
                    override_cache='disk',
                    override_batch=safer_batch,
                    override_imgsz=512,
                    aug_overrides=aug_overrides,
                )
                print("\n✅ Reluare finalizată cu succes.")
            except Exception as e2:
                print(f"\n❌ Reluarea a eșuat: {e2}")
        else:
            raise
    except Exception as e:
        # Catch DataLoader worker OOM surfaced as RuntimeError
        print(f"\n❌ Eroare: {e}")
        if _is_oom_error(e):
            print("\n🔁 Detectat OOM/DataLoader worker error. Reiau cu setări SAFE (workers=0, cache='disk', batch redus, imgsz 512, augment minim)...")
            try:
                safer_batch = max(2, batch // 2)
                aug_overrides = {
                    'mosaic': 0.0,
                    'mixup': 0.0,
                    'copy_paste': 0.0,
                    'hsv_s': 0.5,
                    'hsv_v': 0.4,
                    'degrees': 5.0,
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 1.0,
                    'perspective': 0.00005,
                }
                _ = _train_with_settings(
                    resume=True,
                    override_workers=0,
                    override_cache='disk',
                    override_batch=safer_batch,
                    override_imgsz=512,
                    aug_overrides=aug_overrides,
                )
                print("\n✅ Reluare finalizată cu succes.")
            except Exception as e2:
                print(f"\n❌ Reluarea a eșuat: {e2}")
        else:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    train_custom()
