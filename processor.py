"""
LuminaRetouch - AI Processing Pipeline
Modular AI processors for portrait enhancement
"""

from PySide6.QtCore import QObject, Signal
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import cv2


@dataclass
class ProcessingSettings:
    """Settings for image processing."""
    face_enhancement: float = 0.3  # 0.0 to 1.0 - detail enhancement
    skin_smoothness: float = 0.3
    eye_brightening: float = 0.2
    teeth_whitening: float = 0.0
    face_slimming: float = 0.0
    upscale_factor: int = 1  # 1, 2, or 4
    hardware: str = "cuda"  # cuda, cpu, or vulkan


@dataclass
class ProcessingTask:
    """A single processing task."""
    image: np.ndarray
    settings: ProcessingSettings
    file_path: Optional[Path] = None
    result: Optional[np.ndarray] = None


class BaseProcessor(ABC):
    """Abstract base class for AI processors."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    @abstractmethod
    def load_model(self):
        """Load the AI model."""
        pass

    @abstractmethod
    def process(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Process an image."""
        pass

    def unload_model(self):
        """Unload the model to free memory."""
        self._model = None


class FaceEnhancementProcessor(BaseProcessor):
    """
    Face enhancement for portrait beautification.
    Uses OpenCV-based techniques that work without external AI models.

    Features:
    - Detail enhancement (sharpening eyes, eyebrows, lips)
    - Subtle contrast boost on facial features
    - Works with MediaPipe for face detection
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._face_mesh = None

    def load_model(self):
        """Load MediaPipe face mesh for landmark detection."""
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            print("Face enhancement processor loaded (MediaPipe)")
        except ImportError:
            print("MediaPipe not available for face enhancement")
            self._face_mesh = None

    def process(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Enhance facial features for portrait beautification.

        Args:
            image: Input image (RGB, uint8)
            strength: Enhancement strength (0.0 to 1.0)

        Returns:
            Enhanced image
        """
        if strength == 0:
            return image

        result = image.copy()

        # Detect face regions
        if self._face_mesh is not None:
            results = self._face_mesh.process(image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    result = self._enhance_face(result, face_landmarks, strength)
        else:
            # Fallback without landmarks
            result = self._basic_face_enhance(result, strength)

        return result

    def _enhance_face(self, image: np.ndarray, landmarks, strength: float) -> np.ndarray:
        """Enhance face using landmarks for targeted adjustments."""
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        # Create masks for different facial regions
        # Eyes - enhance detail and brightness
        eye_mask = self._create_region_mask(
            image, landmarks,
            indices=[33, 133, 160, 158, 144, 153, 362, 263, 385, 387, 373, 380],
            expand=1.2
        )

        # Eyebrows - subtle enhancement
        brow_mask = self._create_region_mask(
            image, landmarks,
            indices=[70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
            expand=1.1
        )

        # Lips - color and definition
        lip_mask = self._create_region_mask(
            image, landmarks,
            indices=[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
            expand=1.0
        )

        # Apply unsharp mask to eyes for detail
        if np.sum(eye_mask) > 0:
            blurred = cv2.GaussianBlur(result, (0, 0), 3)
            sharpened = cv2.addWeighted(result, 1.0 + 0.5 * strength, blurred, -0.5 * strength, 0)
            eye_mask_3ch = np.stack([eye_mask] * 3, axis=-1)
            result = result * (1 - eye_mask_3ch) + sharpened * eye_mask_3ch

        # Subtle enhancement to lips (slight saturation boost)
        if np.sum(lip_mask) > 0:
            hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + 0.15 * strength), 0, 255)
            enhanced_lips = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            lip_mask_3ch = np.stack([lip_mask] * 3, axis=-1)
            result = result * (1 - lip_mask_3ch) + enhanced_lips * lip_mask_3ch

        return np.clip(result, 0, 255).astype(np.uint8)

    def _create_region_mask(self, image: np.ndarray, landmarks,
                           indices: list, expand: float = 1.0) -> np.ndarray:
        """Create a soft mask for a facial region."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        points = []
        for idx in indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                points.append([int(lm.x * w), int(lm.y * h)])

        if len(points) < 3:
            return mask

        points = np.array(points, dtype=np.int32)

        # Expand region slightly
        if expand != 1.0:
            center = np.mean(points, axis=0)
            points = ((points - center) * expand + center).astype(np.int32)

        cv2.fillPoly(mask, [points], 1.0)

        # Feather edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return mask

    def _basic_face_enhance(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Fallback basic face enhancement using OpenCV."""
        # Detect faces
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        result = image.copy()

        for (x, y, w, h) in faces:
            # Extract face region with padding
            pad = int(w * 0.2)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image.shape[1], x + w + pad)
            y2 = min(image.shape[0], y + h + pad)

            face_region = result[y1:y2, x1:x2]

            # Apply bilateral filter for smoothing while preserving edges
            enhanced = cv2.bilateralFilter(face_region, 9, 75, 75)

            # Enhance details with unsharp mask
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

            # Blend based on strength
            result[y1:y2, x1:x2] = cv2.addWeighted(
                face_region, 1 - strength,
                enhanced, strength, 0
            )

        return result


class SkinSmoothingProcessor(BaseProcessor):
    """
    Skin smoothing using MediaPipe facial landmarks and frequency separation.
    Applies targeted smoothing only to skin areas.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._face_mesh = None

    def load_model(self):
        """Load MediaPipe face mesh model."""
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except ImportError:
            print("MediaPipe not available, using fallback skin detection")
            self._face_mesh = None

    def process(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply frequency separation skin smoothing.

        Args:
            image: Input image (RGB, uint8)
            strength: Smoothing strength (0.0 to 1.0)

        Returns:
            Processed image
        """
        if strength == 0:
            return image

        # Create skin mask
        skin_mask = self._create_skin_mask(image)

        if skin_mask is None or np.sum(skin_mask) < 100:
            return image

        # Apply frequency separation
        result = self._frequency_separation(image, skin_mask, strength)

        return result

    def _create_skin_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Create a mask for skin regions using MediaPipe or color-based detection."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self._face_mesh is not None:
            # Use MediaPipe for precise face landmark-based masking
            results = self._face_mesh.process(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get face oval points (indices for face outline)
                    face_oval_indices = [
                        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
                    ]

                    # Convert landmarks to points
                    h, w = image.shape[:2]
                    points = []
                    for idx in face_oval_indices:
                        lm = face_landmarks.landmark[idx]
                        points.append([int(lm.x * w), int(lm.y * h)])

                    points = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)

                    # Exclude eye regions
                    left_eye = [33, 160, 158, 133, 153, 144]
                    right_eye = [362, 385, 387, 263, 373, 380]
                    lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

                    for region_indices in [left_eye, right_eye, lips]:
                        region_points = []
                        for idx in region_indices:
                            lm = face_landmarks.landmark[idx]
                            region_points.append([int(lm.x * w), int(lm.y * h)])
                        region_points = np.array(region_points, dtype=np.int32)
                        cv2.fillPoly(mask, [region_points], 0)

        else:
            # Fallback: color-based skin detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

            # HSV skin color range
            hsv_mask = cv2.inRange(hsv, (0, 20, 70), (50, 255, 255))

            # YCrCb skin color range
            ycrcb_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

            # Combine masks
            mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Feather the mask edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        return mask

    def _frequency_separation(self, image: np.ndarray, mask: np.ndarray,
                             strength: float) -> np.ndarray:
        """
        Apply frequency separation technique.

        Separates the image into low frequency (color/tone) and high frequency
        (texture/detail) components, smooths only the low frequency, then
        recombines.
        """
        # Convert to float
        img_float = image.astype(np.float32)

        # Calculate blur radius based on image size
        blur_radius = max(3, int(min(image.shape[:2]) * 0.02))
        if blur_radius % 2 == 0:
            blur_radius += 1

        # Low frequency: Gaussian blur
        low_freq = cv2.GaussianBlur(img_float, (blur_radius, blur_radius), 0)

        # High frequency: Original - Low frequency + 128
        high_freq = img_float - low_freq + 128

        # Smooth the low frequency more aggressively
        smooth_radius = blur_radius * 2 + 1
        low_freq_smooth = cv2.GaussianBlur(low_freq, (smooth_radius, smooth_radius), 0)

        # Blend low frequencies based on strength
        low_freq_blended = cv2.addWeighted(
            low_freq, 1 - strength,
            low_freq_smooth, strength, 0
        )

        # Recombine: Low frequency + (High frequency - 128)
        result = low_freq_blended + high_freq - 128

        # Apply mask
        mask_float = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_float] * 3, axis=-1)

        result = img_float * (1 - mask_3ch) + result * mask_3ch

        # Clip and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


class EyeBrighteningProcessor(BaseProcessor):
    """Eye brightening using facial landmark detection."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._face_mesh = None

    def load_model(self):
        """Load MediaPipe face mesh."""
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except ImportError:
            self._face_mesh = None

    def process(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Brighten eyes in the image."""
        if strength == 0:
            return image

        eye_mask = self._create_eye_mask(image)

        if eye_mask is None or np.sum(eye_mask) < 10:
            return image

        # Create brightening effect
        result = image.copy()
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Increase value (brightness) and saturation slightly
        brightness_boost = 1.0 + (strength * 0.4)
        saturation_boost = 1.0 + (strength * 0.2)

        mask_float = eye_mask.astype(np.float32) / 255.0
        mask_float = np.stack([mask_float] * 3, axis=-1)

        hsv_boosted = hsv.copy()
        hsv_boosted[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
        hsv_boosted[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_boost, 0, 255)

        hsv_result = hsv * (1 - mask_float) + hsv_boosted * mask_float
        hsv_result = hsv_result.astype(np.uint8)

        result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2RGB)

        return result

    def _create_eye_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Create mask for eye regions."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self._face_mesh is None:
            return None

        results = self._face_mesh.process(image)

        if not results.multi_face_landmarks:
            return None

        h, w = image.shape[:2]

        # Eye landmark indices for iris region
        left_iris = [468, 469, 470, 471, 472]
        right_iris = [473, 474, 475, 476, 477]

        for face_landmarks in results.multi_face_landmarks:
            for iris_indices in [left_iris, right_iris]:
                # Get iris center and radius
                points = []
                for idx in iris_indices:
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        points.append([int(lm.x * w), int(lm.y * h)])

                if points:
                    center = np.mean(points, axis=0).astype(int)
                    radius = int(np.max([
                        np.linalg.norm(np.array(p) - center) for p in points
                    ]) * 1.5)

                    cv2.circle(mask, tuple(center), radius, 255, -1)

        # Feather mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return mask


class FaceSlimmingProcessor(BaseProcessor):
    """Face slimming using facial landmark-based warping."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._face_mesh = None

    def load_model(self):
        """Load MediaPipe face mesh."""
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except ImportError:
            self._face_mesh = None

    def process(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply face slimming effect."""
        if strength == 0 or self._face_mesh is None:
            return image

        results = self._face_mesh.process(image)

        if not results.multi_face_landmarks:
            return image

        h, w = image.shape[:2]
        result = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            # Get face contour points (cheeks)
            # Left cheek landmarks
            left_cheek_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
            # Right cheek landmarks
            right_cheek_indices = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

            # Face center (nose tip)
            nose_tip = face_landmarks.landmark[1]
            center_x = int(nose_tip.x * w)
            center_y = int(nose_tip.y * h)

            # Apply liquify-style warping toward center
            for indices, direction in [(left_cheek_indices, 1), (right_cheek_indices, -1)]:
                for idx in indices:
                    lm = face_landmarks.landmark[idx]
                    px, py = int(lm.x * w), int(lm.y * h)

                    # Warp toward center
                    warp_strength = strength * 0.1
                    result = self._local_warp(
                        result, (px, py), (center_x, center_y),
                        radius=int(w * 0.1),
                        strength=warp_strength
                    )

        return result

    def _local_warp(self, image: np.ndarray, src_point: Tuple[int, int],
                    dst_point: Tuple[int, int], radius: int,
                    strength: float) -> np.ndarray:
        """Apply local warping effect (simplified liquify)."""
        h, w = image.shape[:2]
        result = image.copy()

        # Create mesh grid
        y_indices, x_indices = np.mgrid[0:h, 0:w]

        # Calculate distance from source point
        dist = np.sqrt((x_indices - src_point[0])**2 + (y_indices - src_point[1])**2)

        # Create falloff
        falloff = np.maximum(0, 1 - dist / radius)
        falloff = falloff ** 2  # Quadratic falloff

        # Calculate displacement
        dx = (dst_point[0] - src_point[0]) * strength * falloff
        dy = (dst_point[1] - src_point[1]) * strength * falloff

        # New coordinates
        new_x = (x_indices - dx).astype(np.float32)
        new_y = (y_indices - dy).astype(np.float32)

        # Remap
        result = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

        return result


class TeethWhiteningProcessor(BaseProcessor):
    """Teeth whitening using facial landmark detection."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._face_mesh = None

    def load_model(self):
        """Load MediaPipe face mesh."""
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except ImportError:
            self._face_mesh = None

    def process(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Whiten teeth in the image."""
        if strength == 0 or self._face_mesh is None:
            return image

        results = self._face_mesh.process(image)

        if not results.multi_face_landmarks:
            return image

        result = image.copy()
        h, w = image.shape[:2]

        # Inner lip landmarks (mouth opening where teeth are visible)
        inner_lip_indices = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,  # Upper inner lip
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308   # Lower inner lip
        ]

        for face_landmarks in results.multi_face_landmarks:
            # Create teeth mask from inner lip region
            teeth_mask = np.zeros((h, w), dtype=np.uint8)

            points = []
            for idx in inner_lip_indices:
                lm = face_landmarks.landmark[idx]
                points.append([int(lm.x * w), int(lm.y * h)])

            if len(points) >= 3:
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(teeth_mask, [points], 255)

            # Detect teeth within mouth region using color
            # Teeth are typically brighter and less saturated than lips
            mouth_region = cv2.bitwise_and(result, result, mask=teeth_mask)

            # Convert to LAB for better whitening
            lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)

            # In the mouth region, look for bright pixels (likely teeth)
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            _, bright_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
            teeth_only = cv2.bitwise_and(teeth_mask, bright_mask)

            # Feather the mask
            teeth_only = cv2.GaussianBlur(teeth_only, (7, 7), 0)
            mask_float = teeth_only.astype(np.float32) / 255.0

            # Whiten: increase L (lightness), decrease b (yellow-blue, negative = less yellow)
            whitening_l = 20 * strength  # Lighten
            whitening_b = -15 * strength  # Remove yellow

            lab[:, :, 0] = np.clip(lab[:, :, 0] + whitening_l * mask_float, 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + whitening_b * mask_float, 0, 255)

            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        return result


class UpscalingProcessor(BaseProcessor):
    """
    Image upscaling using OpenCV (Real-ESRGAN optional).

    Default: High-quality Lanczos interpolation (no dependencies)
    Optional: Real-ESRGAN for AI-powered upscaling (requires basicsr, realesrgan)
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._upsampler = None
        self._use_ai = False

    def load_model(self):
        """Load Real-ESRGAN model if available, otherwise use OpenCV."""
        # Try to load Real-ESRGAN (optional)
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_path = Path(__file__).parent / "models" / "RealESRGAN_x4plus.pth"

            if not model_path.exists():
                print("Real-ESRGAN model not found, using OpenCV upscaling")
                self._use_ai = False
                return

            # Model configuration for RealESRGAN_x4plus
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )

            self._upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=400,  # Tile size for memory efficiency
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )
            self._use_ai = True
            print("Real-ESRGAN loaded for AI upscaling")

        except ImportError:
            # Real-ESRGAN not installed - use OpenCV (this is fine)
            print("Upscaling: Using OpenCV (install basicsr + realesrgan for AI upscaling)")
            self._use_ai = False
        except Exception as e:
            print(f"Real-ESRGAN loading failed: {e}, using OpenCV")
            self._upsampler = None
            self._use_ai = False

    def process(self, image: np.ndarray, scale: int) -> np.ndarray:
        """
        Upscale image.

        Args:
            image: Input image (RGB, uint8)
            scale: Scale factor (2 or 4)

        Returns:
            Upscaled image
        """
        if scale == 1:
            return image

        if self._upsampler is not None:
            try:
                # Real-ESRGAN expects BGR
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                output, _ = self._upsampler.enhance(bgr_image, outscale=scale)
                return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Real-ESRGAN error: {e}")

        # Fallback: OpenCV resize with Lanczos
        h, w = image.shape[:2]
        new_size = (w * scale, h * scale)
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)


class ProcessorWorker(QObject):
    """
    Worker class for running AI processing in a separate thread.
    Prevents UI freezing during inference.
    """

    # Signals
    progress = Signal(str, int)  # message, percentage (-1 for indeterminate)
    finished = Signal(list)  # list of processed images
    error = Signal(str)

    def __init__(self, tasks: List[ProcessingTask], parent=None):
        super().__init__(parent)
        self.tasks = tasks
        self._processors = {}
        self._is_cancelled = False

    def process(self):
        """Run the processing pipeline."""
        try:
            results = []

            # Initialize processors based on first task settings
            settings = self.tasks[0].settings
            self._init_processors(settings.hardware)

            total_tasks = len(self.tasks)

            for i, task in enumerate(self.tasks):
                if self._is_cancelled:
                    break

                self.progress.emit(
                    f"Processing image {i + 1}/{total_tasks}...",
                    int((i / total_tasks) * 100)
                )

                result = self._process_single(task)
                results.append(result)

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def _init_processors(self, device: str):
        """Initialize AI processors."""
        self.progress.emit("Loading AI models...", -1)

        # Map device names
        device_map = {
            "cuda": "cuda",
            "cpu": "cpu",
            "vulkan": "cpu"  # Vulkan not yet widely supported
        }
        device = device_map.get(device, "cpu")

        # Initialize processors
        self._processors["face_enhance"] = FaceEnhancementProcessor(device)
        self._processors["skin_smooth"] = SkinSmoothingProcessor(device)
        self._processors["eye_bright"] = EyeBrighteningProcessor(device)
        self._processors["teeth_white"] = TeethWhiteningProcessor(device)
        self._processors["face_slim"] = FaceSlimmingProcessor(device)
        self._processors["upscale"] = UpscalingProcessor(device)

        # Load models
        for name, processor in self._processors.items():
            self.progress.emit(f"Loading {name}...", -1)
            processor.load_model()

    def _process_single(self, task: ProcessingTask) -> np.ndarray:
        """Process a single image through the beauty pipeline."""
        image = task.image.copy()
        settings = task.settings

        # Step 1: Face Enhancement (detail sharpening)
        if settings.face_enhancement > 0:
            self.progress.emit("Enhancing facial details...", -1)
            image = self._processors["face_enhance"].process(
                image, settings.face_enhancement
            )

        # Step 2: Skin Smoothing
        if settings.skin_smoothness > 0:
            self.progress.emit("Smoothing skin...", -1)
            image = self._processors["skin_smooth"].process(
                image, settings.skin_smoothness
            )

        # Step 3: Eye Brightening
        if settings.eye_brightening > 0:
            self.progress.emit("Brightening eyes...", -1)
            image = self._processors["eye_bright"].process(
                image, settings.eye_brightening
            )

        # Step 4: Teeth Whitening
        if settings.teeth_whitening > 0:
            self.progress.emit("Whitening teeth...", -1)
            image = self._processors["teeth_white"].process(
                image, settings.teeth_whitening
            )

        # Step 5: Face Slimming
        if settings.face_slimming > 0:
            self.progress.emit("Applying face slimming...", -1)
            image = self._processors["face_slim"].process(
                image, settings.face_slimming
            )

        # Step 6: Upscaling
        if settings.upscale_factor > 1:
            self.progress.emit(
                f"Upscaling {settings.upscale_factor}x...", -1
            )
            image = self._processors["upscale"].process(
                image, settings.upscale_factor
            )

        return image

    def cancel(self):
        """Cancel processing."""
        self._is_cancelled = True
