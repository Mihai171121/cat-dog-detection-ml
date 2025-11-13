"""
Script for setting up the virtual environment and installing packages
"""

import subprocess
import sys
import os
from pathlib import Path


def create_virtual_environment():
    """Create the virtual environment"""
    print("=" * 60)
    print("CREATE VIRTUAL ENVIRONMENT")
    print("=" * 60)

    venv_path = Path(".venv1")

    if venv_path.exists():
        print("⚠️ Virtual environment already exists!")
        response = input("Do you want to delete it and create a new one? (yes/no): ")
        if response.lower() not in ['yes', 'y', 'da']:
            return venv_path

        import shutil
        shutil.rmtree(venv_path)
        print("✅ Old virtual environment deleted")

    print("\n📦 Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", ".venv1"], check=True)
    print("✅ Virtual environment created successfully!")

    return venv_path


def get_python_path(venv_path):
    """Get the path to python from the virtual environment"""
    if os.name == 'nt':  # Windows
        return venv_path / "Scripts" / "python.exe"
    else:  # Linux/Mac
        return venv_path / "bin" / "python"


def install_packages(venv_path):
    """Install packages from requirements.txt"""
    print("\n" + "=" * 60)
    print("INSTALL PACKAGES")
    print("=" * 60)

    python_path = get_python_path(venv_path)

    # Upgrade pip using python -m pip
    print("\n📦 Updating pip...")
    try:
        subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✅ Pip updated successfully!")
    except subprocess.CalledProcessError:
        print("⚠️ Pip could not be updated, but we continue with installation...")

    # Install PyTorch with CUDA first
    print("\n📦 Installing PyTorch with CUDA 11.8 support...")
    print("⏳ This process may take several minutes (downloading ~2.8 GB)...\n")

    subprocess.run([
        str(python_path), "-m", "pip", "install",
        "torch==2.7.1+cu118",
        "torchvision==0.22.1+cu118",
        "torchaudio==2.7.1+cu118",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ], check=True)

    print("\n✅ PyTorch installed successfully!")

    # Install remaining packages
    print("\n📦 Installing remaining packages from requirements.txt...")
    print("⏳ Installation in progress...\n")

    subprocess.run([
        str(python_path), "-m", "pip", "install",
        "ultralytics", "opencv-python", "opencv-contrib-python",
        "matplotlib", "seaborn", "pandas", "scipy", "requests",
        "psutil", "PyYAML", "tqdm", "jupyter", "jupyterlab",
        "notebook", "ipywidgets", "ipykernel"
    ], check=True)

    print("\n✅ All packages have been installed successfully!")


def verify_installation(venv_path):
    """Verify package installation"""
    print("\n" + "=" * 60)
    print("VERIFY INSTALLATION")
    print("=" * 60)

    python_path = get_python_path(venv_path)

    # Check PyTorch and CUDA
    check_script = """
import torch
import ultralytics

print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponibil: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA versiune: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Ultralytics: {ultralytics.__version__}")
"""

    result = subprocess.run(
        [str(python_path), "-c", check_script],
        capture_output=True,
        text=True
    )

    print("\n" + result.stdout)

    if "CUDA disponibil: True" in result.stdout:
        print("✅ GPU NVIDIA RTX 3060 detectat și funcțional!")
    else:
        print("⚠️ ATENȚIE: GPU nu a fost detectat!")


def main():
    """Funcția principală"""
    print("\n" + "=" * 60)
    print("SETUP MEDIU VIRTUAL - PYTHON 3.10 (.venv1)")
    print("Proiect: Detecție Pisici vs Câini - YOLOv8")
    print("GPU: NVIDIA RTX 3060 cu CUDA 11.8")
    print("=" * 60 + "\n")

    try:
        # Creează mediul virtual
        venv_path = create_virtual_environment()

        if venv_path is None:
            print("\n❌ Eroare: Nu s-a putut crea mediul virtual!")
            return

        # Instalează pachetele
        install_packages(venv_path)

        # Verifică instalarea
        verify_installation(venv_path)

        print("\n" + "=" * 60)
        print("✅ CONFIGURARE COMPLETĂ!")
        print("=" * 60)
        print("\n💡 Pentru a activa mediul virtual:")
        print("   Windows: .venv1\\Scripts\\activate")
        print("   Linux/Mac: source .venv1/bin/activate")
        print("\n🚀 Sau rulează: start.bat (Windows)")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Eroare: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
