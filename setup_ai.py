#!/usr/bin/env python3
"""
LuminaRetouch - AI Enhancement Setup
Installs PyTorch and GFPGAN for professional skin retouching.
"""

import subprocess
import sys
import platform


def check_cuda_available():
    """Check if CUDA is likely available."""
    try:
        # Try to detect NVIDIA GPU
        if platform.system() == "Windows":
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        else:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
    except Exception:
        return False


def install_pytorch(use_cuda=True):
    """Install PyTorch."""
    print("\n📦 Installing PyTorch...")

    if use_cuda:
        # CUDA 12.1 version
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ]
    else:
        # CPU only
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision"]

    subprocess.run(cmd, check=True)
    print("✓ PyTorch installed!")


def install_gfpgan():
    """Install GFPGAN."""
    print("\n📦 Installing GFPGAN...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "gfpgan"
    ], check=True)
    print("✓ GFPGAN installed!")


def verify_installation():
    """Verify AI components are working."""
    print("\n🔍 Verifying installation...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ CUDA not available (using CPU)")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    try:
        from gfpgan import GFPGANer
        print("✓ GFPGAN available")
    except ImportError:
        print("✗ GFPGAN not installed")
        return False

    print("\n✅ AI Enhancement ready!")
    print("   Model weights will download automatically on first use (~350MB)")
    return True


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           LuminaRetouch - AI Enhancement Setup                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  This script will install:                                       ║
║  • PyTorch (deep learning framework)                             ║
║  • GFPGAN (AI face enhancement model)                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Check for CUDA
    has_cuda = check_cuda_available()
    if has_cuda:
        print("🎮 NVIDIA GPU detected! Will install CUDA-enabled PyTorch.")
    else:
        print("💻 No NVIDIA GPU detected. Will install CPU-only PyTorch.")
        print("   (AI processing will be slower without GPU)")

    # Confirm
    print("\nProceed with installation? [Y/n]: ", end="")
    response = input().strip().lower()
    if response and response != 'y':
        print("Installation cancelled.")
        return

    try:
        install_pytorch(use_cuda=has_cuda)
        install_gfpgan()
        verify_installation()

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("\nTry installing manually:")
        print("  pip install torch torchvision")
        print("  pip install gfpgan")
        sys.exit(1)

    print("\n🎉 Setup complete! Restart LuminaRetouch to use AI enhancement.")


if __name__ == "__main__":
    main()
