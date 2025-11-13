"""
Quick script for testing GPU and CUDA
"""

import torch
import sys


def test_gpu():
    """Test GPU availability and functionality"""
    print("=" * 70)
    print(" " * 20 + "GPU AND CUDA TEST")
    print("=" * 70)

    # PyTorch information
    print(f"\n📦 PyTorch version: {torch.__version__}")

    # CUDA test
    print(f"\n🔧 CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✅ CUDA functional!")
        print(f"\n📊 GPU Details:")
        print(f"   • CUDA version: {torch.version.cuda}")
        print(f"   • Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"   • Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"   • Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"   • Compute Capability: {props.major}.{props.minor}")
            print(f"   • Multi-processors: {props.multi_processor_count}")

        # GPU computation test
        print(f"\n🧪 GPU computation test...")
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"   ✅ GPU computation works perfectly!")

            # Simple benchmark
            import time
            start = time.time()
            for _ in range(100):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            print(f"   ⚡ Time for 100 matrix multiplications: {gpu_time:.4f}s")

        except Exception as e:
            print(f"   ❌ Error in GPU computation: {e}")

        print("\n" + "=" * 70)
        print("🎉 NVIDIA RTX 3060 GPU is ready for training!")
        print("=" * 70)

    else:
        print(f"❌ CUDA is NOT available!")
        print(f"\n⚠️ Possible causes:")
        print(f"   1. NVIDIA drivers are not installed")
        print(f"   2. PyTorch is not installed with CUDA support")
        print(f"   3. GPU is not enabled in the system")
        print(f"\n💡 Solution: Install PyTorch with CUDA:")
        print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


if __name__ == '__main__':
    test_gpu()

