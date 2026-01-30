"""
LuminaRetouch - AI-Based Face Enhancement
Uses GFPGAN for professional-grade skin retouching and face enhancement.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path
import os

# Global state for AI models
_gfpgan_model = None
_gfpgan_available = None


def check_gfpgan_available() -> bool:
    """Check if GFPGAN is available."""
    global _gfpgan_available
    if _gfpgan_available is not None:
        return _gfpgan_available

    try:
        import torch
        from gfpgan import GFPGANer
        _gfpgan_available = True
    except ImportError:
        _gfpgan_available = False

    return _gfpgan_available


def get_gfpgan_model():
    """Get or initialize GFPGAN model."""
    global _gfpgan_model

    if _gfpgan_model is not None:
        return _gfpgan_model

    if not check_gfpgan_available():
        return None

    try:
        import torch
        from gfpgan import GFPGANer
        from basicsr.utils.download_util import load_file_from_url

        # Determine device
        device = None
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("⚠ WARNING: CUDA is not available. GFPGAN will run on CPU.")
            print("   For GPU acceleration, please ensure you have a compatible NVIDIA GPU,")
            print("   the latest drivers, and the CUDA-enabled version of PyTorch.")
            print("   You can run the setup_ai.py script for guided installation.")

        # Model info
        model_name = 'GFPGANv1.4.pth'
        model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'

        # Create models directory if needed
        model_dir = Path('models') / 'gfpgan'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / model_name

        # Check common locations first
        possible_paths = [
            model_path,
            Path.home() / '.cache' / 'gfpgan' / 'weights' / model_name,
            Path('gfpgan') / 'weights' / model_name,
        ]

        model_file = None
        for p in possible_paths:
            if p.exists():
                model_file = str(p)
                print(f"Found GFPGAN model at: {model_file}")
                break

        # Download if not found
        if model_file is None:
            print(f"Downloading GFPGAN model (~350MB)...")
            model_file = load_file_from_url(
                url=model_url,
                model_dir=str(model_dir),
                progress=True,
                file_name=model_name
            )
            print(f"Model downloaded to: {model_file}")

        # Initialize GFPGAN
        _gfpgan_model = GFPGANer(
            model_path=model_file,
            upscale=1,  # Don't upscale, just enhance
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,  # Don't upscale background
            device=device
        )

        print(f"GFPGAN loaded successfully on {device}")
        return _gfpgan_model

    except Exception as e:
        print(f"Failed to load GFPGAN: {e}")
        _gfpgan_model = None
        return None


def enhance_face_ai(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Enhance face using GFPGAN AI model.

    Args:
        image: RGB image (numpy array)
        strength: Enhancement strength (0.0 to 1.0)

    Returns:
        Enhanced image
    """
    if strength <= 0:
        return image

    model = get_gfpgan_model()
    if model is None:
        print("GFPGAN not available, using fallback")
        return enhance_face_traditional(image, strength)

    try:
        # GFPGAN expects BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Run enhancement
        _, _, enhanced_bgr = model.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        # Convert back to RGB
        enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        # Blend with original based on strength
        result = cv2.addWeighted(
            image.astype(np.float32),
            1 - strength,
            enhanced.astype(np.float32),
            strength,
            0
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"GFPGAN enhancement failed: {e}, using fallback")
        return enhance_face_traditional(image, strength)


def enhance_face_traditional(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Enhanced traditional skin smoothing using bilateral filtering.
    Fallback when AI is not available.

    Args:
        image: RGB image
        strength: Enhancement strength (0.0 to 1.0)

    Returns:
        Enhanced image
    """
    if strength <= 0:
        return image

    result = image.copy()

    # Multi-pass bilateral filtering for smooth skin
    # Bilateral preserves edges while smoothing
    d = 9  # Diameter
    sigma_color = 75 + strength * 50  # Color sigma increases with strength
    sigma_space = 75 + strength * 50  # Space sigma

    # Apply bilateral filter multiple times for stronger effect
    passes = max(1, int(strength * 3))
    for _ in range(passes):
        result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)

    # Blend with original to control strength
    blend_factor = min(strength * 0.8, 0.9)  # Cap at 90% to preserve some texture
    result = cv2.addWeighted(
        image.astype(np.float32),
        1 - blend_factor,
        result.astype(np.float32),
        blend_factor,
        0
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def remove_blemishes_ai(image: np.ndarray, face_mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Remove blemishes using AI-enhanced detection and inpainting.

    Args:
        image: RGB image
        face_mask: Mask of face region
        strength: Removal strength (0.0 to 1.0)

    Returns:
        Image with blemishes removed
    """
    if strength <= 0:
        return image

    result = image.copy()

    # Convert to LAB for better blemish detection
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # Detect blemishes using multiple methods
    blemish_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 1. Local contrast detection (dark spots)
    blur_size = max(15, int(min(image.shape[:2]) * 0.02))
    if blur_size % 2 == 0:
        blur_size += 1

    l_blur = cv2.GaussianBlur(l_channel, (blur_size, blur_size), 0)
    l_diff = l_blur - l_channel  # Positive = darker than surrounding

    dark_thresh = 8 - strength * 5  # More sensitive with higher strength
    dark_spots = (l_diff > dark_thresh).astype(np.uint8) * 255
    blemish_mask = cv2.bitwise_or(blemish_mask, dark_spots)

    # 2. Redness detection (a channel deviation)
    a_blur = cv2.GaussianBlur(a_channel, (blur_size, blur_size), 0)
    a_diff = a_channel - a_blur  # Positive = more red than surrounding

    red_thresh = 6 - strength * 3
    red_spots = (a_diff > red_thresh).astype(np.uint8) * 255
    blemish_mask = cv2.bitwise_or(blemish_mask, red_spots)

    # 3. Brown/yellow spots (b channel)
    b_blur = cv2.GaussianBlur(b_channel, (blur_size, blur_size), 0)
    b_diff = np.abs(b_channel - b_blur)

    brown_thresh = 8 - strength * 4
    brown_spots = (b_diff > brown_thresh).astype(np.uint8) * 255
    blemish_mask = cv2.bitwise_or(blemish_mask, brown_spots)

    # 4. High-frequency texture anomalies
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    texture_diff = cv2.absdiff(gray, gray_blur)

    texture_thresh = 15 - strength * 8
    texture_spots = cv2.threshold(texture_diff, texture_thresh, 255, cv2.THRESH_BINARY)[1]
    blemish_mask = cv2.bitwise_or(blemish_mask, texture_spots)

    # Only within face region
    blemish_mask = cv2.bitwise_and(blemish_mask, face_mask)

    # Clean up mask - remove noise, keep spot-sized regions
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    blemish_mask = cv2.morphologyEx(blemish_mask, cv2.MORPH_OPEN, kernel_open)

    # Remove very large regions (not blemishes)
    contours, _ = cv2.findContours(blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = (min(image.shape[:2]) * 0.03) ** 2  # Max blemish size
    for cnt in contours:
        if cv2.contourArea(cnt) > max_area:
            cv2.drawContours(blemish_mask, [cnt], -1, 0, -1)

    # Dilate for better inpainting coverage
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blemish_mask = cv2.dilate(blemish_mask, kernel_dilate, iterations=1)

    # Inpaint using NS method (better for skin)
    if np.sum(blemish_mask) > 0:
        result = cv2.inpaint(result, blemish_mask, 5, cv2.INPAINT_NS)

    return result


def get_skin_mask(image: np.ndarray, face_landmarks) -> np.ndarray:
    """
    Create a precise skin mask using face landmarks and color detection.

    Args:
        image: RGB image
        face_landmarks: FaceLandmarks object

    Returns:
        Skin mask (0-255)
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Start with face oval
    if len(face_landmarks.face_oval) >= 3:
        cv2.fillPoly(mask, [face_landmarks.face_oval.astype(np.int32)], 255)

    # Exclude eyes, eyebrows, lips
    exclude_regions = [
        face_landmarks.left_eye,
        face_landmarks.right_eye,
        face_landmarks.left_eyebrow,
        face_landmarks.right_eyebrow,
        face_landmarks.lips_outer,
    ]

    for region in exclude_regions:
        if len(region) >= 3:
            # Expand slightly for safety margin
            center = np.mean(region, axis=0)
            expanded = ((region - center) * 1.2 + center).astype(np.int32)
            cv2.fillPoly(mask, [expanded], 0)

    # Refine with skin color detection in YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # Typical skin color range
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)

    skin_color_mask = cv2.inRange(ycrcb, lower, upper)

    # Combine: must be in face region AND have skin color
    mask = cv2.bitwise_and(mask, skin_color_mask)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Feather edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    return mask


def install_ai_models():
    """Print instructions to install AI models."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           LuminaRetouch - AI Model Installation                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  To enable AI-powered skin retouching, install these packages:   ║
║                                                                  ║
║  1. Install PyTorch (with CUDA for GPU acceleration):            ║
║     pip install torch torchvision --index-url \\                  ║
║         https://download.pytorch.org/whl/cu121                   ║
║                                                                  ║
║     Or for CPU only:                                             ║
║     pip install torch torchvision                                ║
║                                                                  ║
║  2. Install GFPGAN:                                              ║
║     pip install gfpgan                                           ║
║                                                                  ║
║  3. The model weights will auto-download on first use (~350MB)   ║
║                                                                  ║
║  GPU recommended for real-time performance!                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    # Check availability and print status
    if check_gfpgan_available():
        print("✓ GFPGAN is available!")
        model = get_gfpgan_model()
        if model:
            print("✓ Model loaded successfully!")
    else:
        print("✗ GFPGAN not installed")
        install_ai_models()
