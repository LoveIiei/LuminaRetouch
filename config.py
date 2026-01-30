"""
LuminaRetouch - Configuration
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class AppConfig:
    """Application configuration."""

    # Paths
    app_dir: Path = Path(__file__).parent
    models_dir: Path = Path(__file__).parent / "models"
    cache_dir: Path = Path(__file__).parent / "cache"

    # Default processing settings
    default_device: str = "cuda"
    default_face_restoration: float = 0.5
    default_skin_smoothness: float = 0.3
    default_eye_brightening: float = 0.2
    default_face_slimming: float = 0.0
    default_upscale: int = 1

    # UI settings
    window_width: int = 1280
    window_height: int = 800
    dark_mode: bool = True

    # Performance
    tile_size: int = 400  # For large image processing
    max_image_dimension: int = 8192

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load config from JSON file."""
        if path is None:
            path = Path(__file__).parent / "config.json"

        config = cls()

        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            except Exception as e:
                print(f"Error loading config: {e}")

        return config

    def save(self, path: Optional[Path] = None):
        """Save config to JSON file."""
        if path is None:
            path = Path(__file__).parent / "config.json"

        data = {
            "default_device": self.default_device,
            "default_face_restoration": self.default_face_restoration,
            "default_skin_smoothness": self.default_skin_smoothness,
            "default_eye_brightening": self.default_eye_brightening,
            "default_face_slimming": self.default_face_slimming,
            "default_upscale": self.default_upscale,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "dark_mode": self.dark_mode,
            "tile_size": self.tile_size,
            "max_image_dimension": self.max_image_dimension,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Model download URLs
# CodeFormer release: https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0
MODEL_URLS = {
    # Face Restoration - CodeFormer (main restoration network)
    "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",

    # Face Detection - RetinaFace (required for CodeFormer to locate faces)
    # This is downloaded automatically by facexlib, but can be placed manually
    "detection_resnet50": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",

    # Face Parsing - ParseNet (required for smooth face blending)
    # Also downloaded automatically by facexlib
    "parsing_parsenet": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",

    # Alternative: GFPGAN (includes its own detection)
    "gfpgan": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",

    # Upscaling - Real-ESRGAN
    "realesrgan_x4": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan_x2": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}

# Where facexlib expects to find models (it will auto-download if not present)
FACEXLIB_MODELS = {
    "detection": "~/.cache/facexlib/weights/detection_Resnet50_Final.pth",
    "parsing": "~/.cache/facexlib/weights/parsing_parsenet.pth",
}


def ensure_models_dir():
    """Ensure models directory exists."""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_required_models():
    """
    Returns information about required models and their status.

    Models needed for full functionality:
    1. codeformer.pth - Face restoration network
    2. detection_Resnet50_Final.pth - Face detection (auto-downloaded by facexlib)
    3. parsing_parsenet.pth - Face parsing for blending (auto-downloaded by facexlib)
    4. RealESRGAN_x4plus.pth - Upscaling (optional)
    """
    models_dir = ensure_models_dir()

    required = {
        "codeformer.pth": {
            "path": models_dir / "codeformer.pth",
            "url": MODEL_URLS["codeformer"],
            "purpose": "Face restoration (CodeFormer network)",
            "required": True,
        },
        "detection_Resnet50_Final.pth": {
            "path": Path.home() / ".cache/facexlib/weights/detection_Resnet50_Final.pth",
            "url": MODEL_URLS["detection_resnet50"],
            "purpose": "Face detection (RetinaFace) - auto-downloads",
            "required": True,
        },
        "parsing_parsenet.pth": {
            "path": Path.home() / ".cache/facexlib/weights/parsing_parsenet.pth",
            "url": MODEL_URLS["parsing_parsenet"],
            "purpose": "Face parsing for smooth blending - auto-downloads",
            "required": True,
        },
        "RealESRGAN_x4plus.pth": {
            "path": models_dir / "RealESRGAN_x4plus.pth",
            "url": MODEL_URLS["realesrgan_x4"],
            "purpose": "4x upscaling",
            "required": False,
        },
    }

    for name, info in required.items():
        info["exists"] = info["path"].exists()

    return required
