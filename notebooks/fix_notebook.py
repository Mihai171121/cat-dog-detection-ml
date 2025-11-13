#!/usr/bin/env python3
"""Script pentru a crea un notebook Jupyter valid pentru Google Colab"""

import json
from pathlib import Path

# Creează notebook-ul cu structură validă
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
                "**IMPORTANT: Înainte de a rula, setează GPU:**\n",
                "1. Runtime → Change runtime type → Hardware accelerator: **GPU** (T4)\n",
                "2. Apoi rulează celulele în ordine (1 → 6)\n",
                "\n",
                "---\n",
                "\n",
                "## 📋 Ce face acest notebook:\n",
                "- ✅ Montează Google Drive\n",
                "- ✅ Instalează dependențe (PyTorch + YOLO)\n",
                "- ✅ Antrenează modelul YOLOv8 Medium (150 epoci, ~2 ore)\n",
                "- ✅ Afișează grafice și metrici\n",
                "- ✅ Testează predicții pe imagini\n",
                "\n",
                "**Rezultatele se salvează în Google Drive și rămân acolo chiar dacă sesiunea se închide!**"
            ],
            "metadata": {
                "id": "header"
            }
        },
        {
            "cell_type": "code",
            "source": [
                "# ============================================================================\n",
                "# 1️⃣ MONTARE GOOGLE DRIVE\n",
                "# ============================================================================\n",
                "from google.colab import drive\n",
                "import os\n",
                "\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# 🔧 AJUSTEAZĂ CALEA către proiectul tău din Drive:\n",
                "PROJECT_DIR = '/content/drive/MyDrive/ML Cats vs Dogs'\n",
                "\n",
                "# Verifică dacă proiectul există\n",
                "if os.path.exists(PROJECT_DIR):\n",
                "    print(f'✅ Proiect găsit: {PROJECT_DIR}')\n",
                "    !ls -lh \"$PROJECT_DIR\" | head -20\n",
                "else:\n",
                "    print(f'❌ EROARE: Proiectul nu există la: {PROJECT_DIR}')\n",
                "    print('\\n💡 SOLUȚII:')\n",
                "    print('   1. Încarcă folderul \"ML Cats vs Dogs\" în Google Drive (MyDrive)')\n",
                "    print('   2. SAU modifică PROJECT_DIR mai sus cu calea corectă')\n",
                "    print('\\nCăi disponibile în Drive:')\n",
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
                "# 2️⃣ VERIFICARE GPU & INSTALARE DEPENDENȚE\n",
                "# ============================================================================\n",
                "import torch\n",
                "\n",
                "print('=' * 70)\n",
                "print('🔍 VERIFICARE GPU')\n",
                "print('=' * 70)\n",
                "print(f'PyTorch versiune: {torch.__version__}')\n",
                "print(f'CUDA disponibil: {torch.cuda.is_available()}')\n",
                "\n",
                "if torch.cuda.is_available():\n",
                "    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')\n",
                "    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')\n",
                "    print(f'✅ CUDA versiune: {torch.version.cuda}')\n",
                "else:\n",
                "    print('❌ GPU NU ESTE ACTIV!')\n",
                "    print('\\n💡 Activează GPU: Runtime → Change runtime type → GPU')\n",
                "\n",
                "print('\\n' + '=' * 70)\n",
                "print('📦 INSTALARE DEPENDENȚE')\n",
                "print('=' * 70)\n",
                "\n",
                "# Instalare (quiet mode)\n",
                "!pip install -q --upgrade pip\n",
                "!pip install -q ultralytics opencv-python-headless matplotlib seaborn pandas PyYAML tqdm\n",
                "\n",
                "import ultralytics\n",
                "print(f'✅ Ultralytics YOLOv8: {ultralytics.__version__}')\n",
                "print('✅ Toate dependențele sunt instalate!')"
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
                "# 3️⃣ VERIFICARE FIȘIERE PROIECT\n",
                "# ============================================================================\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "project_dir = Path(PROJECT_DIR)\n",
                "assert project_dir.exists(), f'❌ Proiect nu găsit: {project_dir}'\n",
                "\n",
                "print('=' * 70)\n",
                "print('📂 VERIFICARE STRUCTURĂ PROIECT')\n",
                "print('=' * 70)\n",
                "\n",
                "# Fișiere esențiale\n",
                "required = {\n",
                "    'training/train_colab.py': 'Script antrenare automată (Colab)',\n",
                "    'training/train_local.py': 'Script antrenare interactivă (PC)',\n",
                "    'ui_detector.py': 'Interfață grafică pentru detectare',\n",
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
                "    print(f'\\n✅ Dataset YAML găsit: {data_yaml}')\n",
                "    !head -15 \"$data_yaml\"\n",
                "else:\n",
                "    print(f'\\n❌ Dataset YAML lipsă: {data_yaml}')\n",
                "\n",
                "# Creează directoare necesare\n",
                "for d in ['Pictures', 'models/trained', 'output/test_results']:\n",
                "    (project_dir / d).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print(f'\\n✅ Proiect verificat și gata de antrenare!')"
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
                "# 4️⃣ ANTRENARE MODEL - ALEGE MODUL\n",
                "# ============================================================================\n",
                "# Ai 2 opțiuni:\n",
                "# A) AUTOMATĂ - rulează cu setări default (YOLOv8m, 150 epoci, batch 8)\n",
                "# B) INTERACTIVĂ - alegi tu modelul, epocile, batch-ul, learning rate\n",
                "\n",
                "%cd \"$PROJECT_DIR\"\n",
                "\n",
                "print('=' * 70)\n",
                "print('🎯 ALEGE MODUL DE ANTRENARE')\n",
                "print('=' * 70)\n",
                "print('\\n1️⃣  AUTOMATĂ - Rapid, fără întrebări (YOLOv8m, 150 epoci, ~2h)')\n",
                "print('2️⃣  INTERACTIVĂ - Tu alegi toate setările (model, epoci, batch, lr)\\n')\n",
                "\n",
                "choice = input('Alege modul (1 sau 2): ').strip()\n",
                "\n",
                "if choice == '1':\n",
                "    print('\\n🚀 Pornește antrenarea AUTOMATĂ...')\n",
                "    print('💡 Setări: YOLOv8 Medium, 150 epoci, batch 8, lr 0.005')\n",
                "    print('💡 Progresul se salvează automat în Drive la fiecare 5-10 epoci')\n",
                "    print('💡 Poți închide tab-ul și reveni mai târziu - progresul rămâne salvat!\\n')\n",
                "    !python training/train_colab.py\n",
                "else:\n",
                "    print('\\n🚀 Pornește antrenarea INTERACTIVĂ...')\n",
                "    print('💡 Răspunde la întrebările de mai jos pentru a configura antrenarea')\n",
                "    print('💡 Apasă Enter pentru a folosi valoarea recomandată\\n')\n",
                "    !python training/train_local.py\n",
                "\n",
                "print('\\n✅ Antrenare finalizată!')\n",
                "print('📁 Rezultatele sunt în:', PROJECT_DIR + '/runs/train/')"
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
                "# 5️⃣ VIZUALIZARE GRAFICE & METRICI\n",
                "# ============================================================================\n",
                "%cd \"$PROJECT_DIR\"\n",
                "\n",
                "from pathlib import Path\n",
                "import matplotlib.pyplot as plt\n",
                "from IPython.display import Image, display\n",
                "import pandas as pd\n",
                "\n",
                "# Găsește ultima antrenare\n",
                "runs_dir = Path('runs/train')\n",
                "train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)\n",
                "\n",
                "if train_dirs:\n",
                "    latest = train_dirs[-1]\n",
                "    print(f'📊 Rezultate din: {latest.name}\\n')\n",
                "    \n",
                "    # Citește CSV cu rezultate\n",
                "    results_csv = latest / 'results.csv'\n",
                "    if results_csv.exists():\n",
                "        df = pd.read_csv(results_csv)\n",
                "        df.columns = df.columns.str.strip()\n",
                "        \n",
                "        print('=' * 70)\n",
                "        print('📈 REZULTATE FINALE (ultima epocă)')\n",
                "        print('=' * 70)\n",
                "        last = df.iloc[-1]\n",
                "        print(f\"  Epocă:     {int(last['epoch'])}\")\n",
                "        print(f\"  mAP50:     {last.get('metrics/mAP50(B)', 0):.4f} ({last.get('metrics/mAP50(B)', 0)*100:.2f}%)\")\n",
                "        print(f\"  mAP50-95:  {last.get('metrics/mAP50-95(B)', 0):.4f}\")\n",
                "        print(f\"  Precision: {last.get('metrics/precision(B)', 0):.4f}\")\n",
                "        print(f\"  Recall:    {last.get('metrics/recall(B)', 0):.4f}\")\n",
                "        print('=' * 70 + '\\n')\n",
                "    \n",
                "    # Afișează grafice\n",
                "    graphs = {\n",
                "        'results.png': '📈 Evoluție Training',\n",
                "        'confusion_matrix.png': '🎯 Matrice Confuzie',\n",
                "        'labels.jpg': '🏷️ Distribuție Labels',\n",
                "        'PR_curve.png': '📉 Curba Precision-Recall',\n",
                "        'F1_curve.png': '📊 Curba F1-Score',\n",
                "    }\n",
                "    \n",
                "    for img_file, title in graphs.items():\n",
                "        img_path = latest / img_file\n",
                "        if img_path.exists():\n",
                "            print(f'{title}')\n",
                "            display(Image(filename=str(img_path), width=900))\n",
                "            print('\\n')\n",
                "else:\n",
                "    print('❌ Nu există antrenări salvate!')"
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
                "# 6️⃣ TESTARE PREDICȚII PE IMAGINI\n",
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
                "            print(f'📥 Descărcare {filename}...')\n",
                "            r = requests.get(url, timeout=15)\n",
                "            if r.status_code == 200:\n",
                "                img_path.write_bytes(r.content)\n",
                "                print(f'   ✅ Salvat: {img_path}')\n",
                "        except Exception as e:\n",
                "            print(f'   ⚠️ Eroare la descărcare: {e}')\n",
                "\n",
                "print('\\n🔍 Testare cu modelul antrenat...')\n",
                "print('💡 Pentru predicții, rulează: python ui_detector.py (pe PC local)')\n",
                "print('💡 Sau descarcă best.pt și testează local cu interfața grafică\\n')\n",
                "\n",
                "# Afișează rezultatele\n",
                "from IPython.display import Image, display\n",
                "output_dir = Path('output/test_results')\n",
                "if output_dir.exists():\n",
                "    result_images = sorted(output_dir.glob('result_*.jpg'))\n",
                "    if result_images:\n",
                "        print(f'\\n🖼️ REZULTATE PREDICȚII ({len(result_images)} imagini):\\n')\n",
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
                "## ✅ GATA! Ce să faci acum:\n",
                "\n",
                "1. **Verifică rezultatele în Google Drive:**\n",
                "   - `runs/train/` - toate graficele și modelul antrenat (best.pt)\n",
                "   - `models/trained/` - modelul final copiat\n",
                "   - `output/test_results/` - predicții pe imagini\n",
                "\n",
                "2. **Pentru a testa pe propriile imagini:**\n",
                "   - Încarcă imagini în `Pictures/` folder în Drive\n",
                "   - Rulează din nou celula 6️⃣\n",
                "\n",
                "3. **Pentru o nouă antrenare:**\n",
                "   - Rulează din nou celula 4️⃣ (se creează un nou folder în runs/)\n",
                "\n",
                "4. **Descarcă modelul antrenat:**\n",
                "   - Click dreapta pe `runs/train/colab_*/weights/best.pt` în Drive\n",
                "   - Download și folosește-l local!\n",
                "\n",
                "---\n",
                "\n",
                "**💡 SFATURI:**\n",
                "- Sesiunea Colab se întrerupe după ~12 ore sau inactivitate - progresul în Drive rămâne salvat\n",
                "- Pentru antrenare mai lungă, setează epoci mai multe în `train_colab.py`\n",
                "- GPU T4 gratuit în Colab poate avea limitări de timp - folosește Colab Pro pentru antrenări foarte lungi"
            ],
            "metadata": {
                "id": "footer"
            }
        }
    ]
}

# Salvează notebook-ul
output_path = Path(__file__).parent / 'Train_and_Test_in_Colab.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f'✅ Notebook creat cu succes: {output_path}')
print(f'📊 Celule: {len(notebook["cells"])}')
print(f'📋 Format: nbformat {notebook["nbformat"]}.{notebook["nbformat_minor"]}')
print('\n🚀 Acum poți încărca fișierul în Google Colab!')

