"""
Script rapid pentru testarea GPU și CUDA
"""

import torch
import sys


def test_gpu():
    """Testează disponibilitatea și funcționalitatea GPU"""
    print("=" * 70)
    print(" " * 20 + "TEST GPU și CUDA")
    print("=" * 70)

    # Informații PyTorch
    print(f"\n📦 PyTorch versiune: {torch.__version__}")

    # Test CUDA
    print(f"\n🔧 CUDA disponibil: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✅ CUDA funcțional!")
        print(f"\n📊 Detalii GPU:")
        print(f"   • CUDA versiune: {torch.version.cuda}")
        print(f"   • Număr GPU-uri: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"   • Nume: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"   • Memorie totală: {props.total_memory / 1024**3:.2f} GB")
            print(f"   • Compute Capability: {props.major}.{props.minor}")
            print(f"   • Multi-processors: {props.multi_processor_count}")

        # Test calcul pe GPU
        print(f"\n🧪 Test calcul pe GPU...")
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"   ✅ Calculul pe GPU funcționează perfect!")

            # Benchmark simplu
            import time
            start = time.time()
            for _ in range(100):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            print(f"   ⚡ Timp pentru 100 înmulțiri matriciale: {gpu_time:.4f}s")

        except Exception as e:
            print(f"   ❌ Eroare la calculul pe GPU: {e}")

        print("\n" + "=" * 70)
        print("🎉 GPU-ul NVIDIA RTX 3060 este gata pentru antrenare!")
        print("=" * 70)

    else:
        print(f"❌ CUDA NU este disponibil!")
        print(f"\n⚠️ Posibile cauze:")
        print(f"   1. Driverele NVIDIA nu sunt instalate")
        print(f"   2. PyTorch nu este instalat cu suport CUDA")
        print(f"   3. GPU-ul nu este activat în sistem")
        print(f"\n💡 Soluție: Instalați PyTorch cu CUDA:")
        print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


if __name__ == '__main__':
    test_gpu()

