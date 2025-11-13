#!/usr/bin/env python3
"""Script to create a valid Jupyter notebook for Google Colab"""

import json
from pathlib import Path

# Create notebook with valid structure
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": [
        {
            "cell_type": "markdown",
            "source": [
                "# 🐱🐶 Train & Test YOLOv8 (Cats vs Dogs) in Google Colab\n",
                "\n",
                "**IMPORTANT: Before running, set up GPU:**\n",
                "1. Runtime → Change runtime type → Hardware accelerator: **GPU** (T4)\n",
                "2. Then run cells in order (1 → 6)\n",
                "\n",
                "---\n",
                "\n",
                "## 📋 What this notebook does:\n",
                "- ✅ Mounts Google Drive\n",
                "- ✅ Installs dependencies (PyTorch + YOLO)\n",
                "- ✅ Trains YOLOv8 Medium model (150 epochs, ~2 hours)\n",
                "- ✅ Displays graphs and metrics\n",
                "- ✅ Tests predictions on images\n",
                "\n",
                "**Results are saved in Google Drive and remain there even if the session closes!**"
            ],
            "metadata": {
                "id": "header"
            }
        },
        {
            "cell_type": "code",
            "source": [
                "# ============================================================================\n",
                "# 1️⃣ MOUNT GOOGLE DRIVE\n",
                "# ============================================================================\n",
                "from google.colab import drive\n",
                "import os\n",
                "\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# 🔧 ADJUST THE PATH to your project in Drive:\n",
                "PROJECT_DIR = '/content/drive/MyDrive/ML Cats vs Dogs'\n",
                "\n",
                "# Check if project exists\n",
                "if os.path.exists(PROJECT_DIR):\n",
                "    print(f'✅ Project found: {PROJECT_DIR}')\n",
                "    !ls -lh \"$PROJECT_DIR\" | head -20\n",
                "else:\n",
                "    print(f'❌ ERROR: Project does not exist at: {PROJECT_DIR}')\n",
                "    print('\\n💡 SOLUTIONS:')\n",
                "    print('   1. Upload the \"ML Cats vs Dogs\" folder to Google Drive (MyDrive)')\n",
                "    print('   2. OR modify PROJECT_DIR above with the correct path')\n",
                "    print('\\nAvailable paths in Drive:')\n",
                "    !ls -lh \"/content/drive/MyDrive/\" | head -20"
            ],
            "metadata": {
                "id": "mount_drive"
            },
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "source": [
                "# ============================================================================\n",
                "# 2️⃣ CHECK GPU & INSTALL DEPENDENCIES\n",
                "# ============================================================================\n",
                "import torch\n",
                "\n",
                "print('=' * 70)\n",
                "print('🔍 GPU CHECK')\n",
                "print('=' * 70)\n",
                "print(f'PyTorch version: {torch.__version__}')\n",
                "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                "\n",
                "if torch.cuda.is_available():\n",
                "    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')\n",
                "    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')\n",
                "    print(f'✅ CUDA version: {torch.version.cuda}')\n",
                "else:\n",
                "    print('❌ GPU IS NOT ACTIVE!')\n",
                "    print('\\n💡 Enable GPU: Runtime → Change runtime type → GPU')\n",
                "\n",
                "print('\\n' + '=' * 70)\n",
                "print('📦 INSTALL DEPENDENCIES')\n",
                "print('=' * 70)\n",
                "\n",
                "# Install (quiet mode)\n",
                "!pip install -q --upgrade pip\n",
                "!pip install -q ultralytics opencv-python-headless matplotlib seaborn pandas PyYAML tqdm\n",
                "\n",
                "import ultralytics\n",
                "print(f'✅ Ultralytics YOLOv8: {ultralytics.__version__}')\n",
                "print('✅ All dependencies are installed!')"
            ],
            "metadata": {
                "id": "check_gpu"
            },
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "source": [
                "# ============================================================================\n",
                "# 3️⃣ CHECK PROJECT FILES\n",
                "# ============================================================================\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "project_dir = Path(PROJECT_DIR)\n",
                "assert project_dir.exists(), f'❌ Project not found: {project_dir}'\n",
                "\n",
                "print('=' * 70)\n",
                "print('📂 CHECK PROJECT STRUCTURE')\n",
                "print('=' * 70)\n",
                "\n",
                "# Essential files\n",
                "required = {\n",
                "    'training/train_colab.py': 'Automatic training script (Colab)',\n",
                "    'training/train_local.py': 'Interactive training script (PC)',\n",
                "    'ui_detector.py': 'Graphical interface for detection',\n",
                "}\n",
                "\n",
                "for f, desc in required.items():\n",
                "    p = project_dir / f\n",
                "    status = '✅' if p.exists() else '❌'\n",
                "    print(f'{status} {f:25s} - {desc}')\n",
                "\n",
                "# Dataset\n",
                "data_yaml = project_dir / 'Data_set_Cat_vs_Dog' / 'yolo_data' / 'data.yaml'\n",
                "if data_yaml.exists():\n",
                "    print(f'\\n✅ Dataset YAML found: {data_yaml}')\n",
                "    !head -15 \"$data_yaml\"\n",
                "else:\n",
                "    print(f'\\n❌ Dataset YAML missing: {data_yaml}')\n",
                "\n",
                "# Create necessary directories\n",
                "for d in ['Pictures', 'models/trained', 'output/test_results']:\n",
                "    (project_dir / d).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print(f'\\n✅ Project verified and ready for training!')"
            ],
            "metadata": {
                "id": "verify_project"
            },
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "source": [
                "# ============================================================================\n",
                "# 4️⃣ TRAIN MODEL - CHOOSE MODE\n",
                "# ============================================================================\n",
                "# You have 2 options:\n",
                "# A) AUTOMATIC - runs with default settings (YOLOv8m, 150 epochs, batch 8)\n",
                "# B) INTERACTIVE - you choose the model, epochs, batch, learning rate\n",
                "\n",
                "%cd \"$PROJECT_DIR\"\n",
                "\n",
                "print('=' * 70)\n",
                "print('🎯 CHOOSE TRAINING MODE')\n",
                "print('=' * 70)\n",
                "print('\\n1️⃣  AUTOMATIC - Fast, no questions (YOLOv8m, 150 epochs, ~2h)')\n",
                "print('2️⃣  INTERACTIVE - You choose all settings (model, epochs, batch, lr)\\n')\n",
                "\n",
                "choice = input('Choose mode (1 or 2): ').strip()\n",
                "\n",
                "if choice == '1':\n",
                "    print('\\n🚀 Starting AUTOMATIC training...')\n",
                "    print('💡 Settings: YOLOv8 Medium, 150 epochs, batch 8, lr 0.005')\n",
                "    print('💡 Progress is automatically saved to Drive every 5-10 epochs')\n",
                "    print('💡 You can close the tab and return later - progress remains saved!\\n')\n",
                "    !python training/train_colab.py\n",
                "else:\n",
                "    print('\\n🚀 Starting INTERACTIVE training...')\n",
                "    print('💡 Answer the questions below to configure training')\n",
                "    print('💡 Press Enter to use the recommended value\\n')\n",
                "    !python training/train_local.py\n",
                "\n",
                "print('\\n✅ Training completed!')\n",
                "print('📁 Results are in:', PROJECT_DIR + '/runs/train/')"
            ],
            "metadata": {
                "id": "run_training"
            },
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "source": [
                "# ============================================================================\n",
                "# 5️⃣ VISUALIZE GRAPHS & METRICS\n",
                "# ============================================================================\n",
                "%cd \"$PROJECT_DIR\"\n",
                "\n",
                "from pathlib import Path\n",
                "import matplotlib.pyplot as plt\n",
                "from IPython.display import Image, display\n",
                "import pandas as pd\n",
                "\n",
                "# Find latest training\n",
                "runs_dir = Path('runs/train')\n",
                "train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)\n",
                "\n",
                "if train_dirs:\n",
                "    latest = train_dirs[-1]\n",
                "    print(f'📊 Results from: {latest.name}\\n')\n",
                "    \n",
                "    # Read CSV with results\n",
                "    results_csv = latest / 'results.csv'\n",
                "    if results_csv.exists():\n",
                "        df = pd.read_csv(results_csv)\n",
                "        df.columns = df.columns.str.strip()\n",
                "        \n",
                "        print('=' * 70)\n",
                "        print('📈 FINAL RESULTS (last epoch)')\n",
                "        print('=' * 70)\n",
                "        last = df.iloc[-1]\n",
                "        print(f\"  Epoch:     {int(last['epoch'])}\")\n",
                "        print(f\"  mAP50:     {last.get('metrics/mAP50(B)', 0):.4f} ({last.get('metrics/mAP50(B)', 0)*100:.2f}%)\")\n",
                "        print(f\"  mAP50-95:  {last.get('metrics/mAP50-95(B)', 0):.4f}\")\n",
                "        print(f\"  Precision: {last.get('metrics/precision(B)', 0):.4f}\")\n",
                "        print(f\"  Recall:    {last.get('metrics/recall(B)', 0):.4f}\")\n",
                "        print('=' * 70 + '\\n')\n",
                "    \n",
                "    # Display graphs\n",
                "    graphs = {\n",
                "        'results.png': '📈 Training Evolution',\n",
                "        'confusion_matrix.png': '🎯 Confusion Matrix',\n",
                "        'labels.jpg': '🏷️ Label Distribution',\n",
                "        'PR_curve.png': '📉 Precision-Recall Curve',\n",
                "        'F1_curve.png': '📊 F1-Score Curve',\n",
                "    }\n",
                "    \n",
                "    for img_file, title in graphs.items():\n",
                "        img_path = latest / img_file\n",
                "        if img_path.exists():\n",
                "            print(f'{title}')\n",
                "            display(Image(filename=str(img_path), width=900))\n",
                "            print('\\n')\n",
                "else:\n",
                "    print('❌ No saved training runs!')"
            ],
            "metadata": {
                "id": "view_graphs"
            },
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "source": [
                "# ============================================================================\n",
                "# 6️⃣ TEST PREDICTIONS ON IMAGES\n",
                "# ============================================================================\n",
                "%cd \"$PROJECT_DIR\"\n",
                "\n",
                "from pathlib import Path\n",
                "import requests\n",
                "\n",
                "pictures_dir = Path('Pictures')\n",
                "pictures_dir.mkdir(exist_ok=True)\n",
                "\n",
                "# URLs cu imagini de test (pisici și câini)\n",
                "test_images = [\n",
                "    ('https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800', 'cat_test1.jpg'),\n",
                "    ('https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800', 'dog_test1.jpg'),\n",
                "]\n",
                "\n",
                "for url, filename in test_images:\n",
                "    img_path = pictures_dir / filename\n",
                "    if not img_path.exists():\n",
                "        try:\n",
                "            print(f'📥 Downloading {filename}...')\n",
                "            r = requests.get(url, timeout=15)\n",
                "            if r.status_code == 200:\n",
                "                img_path.write_bytes(r.content)\n",
                "                print(f'   ✅ Saved: {img_path}')\n",
                "        except Exception as e:\n",
                "            print(f'   ⚠️ Download error: {e}')\n",
                "\n",
                "print('\\n🔍 Testing with trained model...')\n",
                "print('💡 For predictions, run: python ui_detector.py (on local PC)')\n",
                "print('💡 Or download best.pt and test locally with the graphical interface\\n')\n",
                "\n",
                "# Display results\n",
                "from IPython.display import Image, display\n",
                "output_dir = Path('output/test_results')\n",
                "if output_dir.exists():\n",
                "    result_images = sorted(output_dir.glob('result_*.jpg'))\n",
                "    if result_images:\n",
                "        print(f'\\n🖼️ PREDICTION RESULTS ({len(result_images)} images):\\n')\n",
                "        for img in result_images[:5]:\n",
                "            print(f'📷 {img.name}')\n",
                "            display(Image(filename=str(img), width=700))\n",
                "            print('\\n')"
            ],
            "metadata": {
                "id": "test_predictions"
            },
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "source": [
                "---\n",
                "\n",
                "## ✅ DONE! What to do now:\n",
                "\n",
                "1. **Check results in Google Drive:**\n",
                "   - `runs/train/` - all graphs and trained model (best.pt)\n",
                "   - `models/trained/` - final model copied\n",
                "   - `output/test_results/` - predictions on images\n",
                "\n",
                "2. **To test on your own images:**\n",
                "   - Upload images to `Pictures/` folder in Drive\n",
                "   - Run cell 6️⃣ again\n",
                "\n",
                "3. **For a new training:**\n",
                "   - Run cell 4️⃣ again (creates a new folder in runs/)\n",
                "\n",
                "4. **Download the trained model:**\n",
                "   - Right click on `runs/train/colab_*/weights/best.pt` in Drive\n",
                "   - Download and use it locally!\n",
                "\n",
                "---\n",
                "\n",
                "**💡 TIPS:**\n",
                "- Colab session disconnects after ~12 hours or inactivity - progress in Drive remains saved\n",
                "- For longer training, set more epochs in `train_colab.py`\n",
                "- Free T4 GPU in Colab may have time limitations - use Colab Pro for very long training"
            ],
            "metadata": {
                "id": "footer"
            }
        }
    ]
}

# Save the notebook
output_path = Path(__file__).parent / 'Train_and_Test_in_Colab.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f'✅ Notebook created successfully: {output_path}')
print(f'📊 Cells: {len(notebook["cells"])}')
print(f'📋 Format: nbformat {notebook["nbformat"]}.{notebook["nbformat_minor"]}')
print('\n🚀 Now you can upload the file to Google Colab!')

