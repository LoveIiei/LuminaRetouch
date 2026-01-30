"""
Setup models for LuminaRetouch.
Copies models from the local models/ directory to where facexlib expects them.

Run this once after downloading models:
    python setup_models.py
"""

import shutil
import sys
from pathlib import Path


def get_facexlib_weights_dir():
    """Find where facexlib stores its weights."""
    try:
        import facexlib
        facexlib_dir = Path(facexlib.__file__).parent / "weights"
        return facexlib_dir
    except ImportError:
        print("facexlib not installed. Run: pip install facexlib")
        return None


def setup_models():
    """Copy models from local models/ dir to facexlib weights dir."""
    local_models_dir = Path(__file__).parent / "models"
    facexlib_weights_dir = get_facexlib_weights_dir()

    if facexlib_weights_dir is None:
        return False

    # Ensure facexlib weights dir exists
    facexlib_weights_dir.mkdir(parents=True, exist_ok=True)

    # Models to copy
    models_to_copy = [
        ("detection_Resnet50_Final.pth", "Face detection (RetinaFace)"),
        ("parsing_parsenet.pth", "Face parsing"),
    ]

    print(f"Local models directory: {local_models_dir}")
    print(f"Facexlib weights directory: {facexlib_weights_dir}")
    print()

    # Check codeformer.pth (stays in local dir)
    codeformer_path = local_models_dir / "codeformer.pth"
    if codeformer_path.exists():
        print(f"[OK] codeformer.pth found ({codeformer_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"[MISSING] codeformer.pth - Download from:")
        print(f"    https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth")
        print(f"    Save to: {codeformer_path}")

    print()

    # Copy detection and parsing models to facexlib
    for model_name, description in models_to_copy:
        local_path = local_models_dir / model_name
        target_path = facexlib_weights_dir / model_name

        if target_path.exists():
            print(f"[OK] {model_name} already in facexlib ({target_path.stat().st_size / 1024 / 1024:.1f} MB)")
        elif local_path.exists():
            print(f"[COPY] {model_name}: {local_path} -> {target_path}")
            shutil.copy2(local_path, target_path)
            print(f"       Copied successfully!")
        else:
            print(f"[AUTO] {model_name} not found locally - facexlib will auto-download")
            print(f"       Or download manually from CodeFormer releases and place in {local_path}")

    print()
    print("Setup complete!")
    return True


def check_all_models():
    """Check status of all required models."""
    local_models_dir = Path(__file__).parent / "models"
    facexlib_weights_dir = get_facexlib_weights_dir()

    print("=" * 60)
    print("MODEL STATUS CHECK")
    print("=" * 60)

    required_models = {
        "codeformer.pth": {
            "location": local_models_dir / "codeformer.pth",
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            "purpose": "Face restoration neural network",
            "required": True,
        },
        "detection_Resnet50_Final.pth": {
            "location": facexlib_weights_dir / "detection_Resnet50_Final.pth" if facexlib_weights_dir else None,
            "alt_location": local_models_dir / "detection_Resnet50_Final.pth",
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "purpose": "Face detection (RetinaFace)",
            "required": True,
        },
        "parsing_parsenet.pth": {
            "location": facexlib_weights_dir / "parsing_parsenet.pth" if facexlib_weights_dir else None,
            "alt_location": local_models_dir / "parsing_parsenet.pth",
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
            "purpose": "Face parsing for smooth blending",
            "required": True,
        },
        "RealESRGAN_x4plus.pth": {
            "location": local_models_dir / "RealESRGAN_x4plus.pth",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "purpose": "4x image upscaling",
            "required": False,
        },
    }

    all_ok = True
    for name, info in required_models.items():
        exists = info["location"] and info["location"].exists()
        alt_exists = info.get("alt_location") and info["alt_location"].exists()

        if exists:
            size_mb = info["location"].stat().st_size / 1024 / 1024
            status = f"[OK] {size_mb:.1f} MB"
        elif alt_exists:
            size_mb = info["alt_location"].stat().st_size / 1024 / 1024
            status = f"[OK*] {size_mb:.1f} MB (in models/, run setup_models.py to copy)"
        elif info["required"]:
            status = "[MISSING]"
            all_ok = False
        else:
            status = "[OPTIONAL - not found]"

        print(f"\n{name}")
        print(f"  Status: {status}")
        print(f"  Purpose: {info['purpose']}")
        if not exists and not alt_exists and info["required"]:
            print(f"  Download: {info['url']}")

    print()
    print("=" * 60)
    if all_ok:
        print("All required models found!")
    else:
        print("Some models are missing. Download them from the URLs above.")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_all_models()
    else:
        setup_models()
        print()
        print("Run 'python setup_models.py --check' to verify all models.")
