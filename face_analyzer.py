"""
LuminaRetouch - Shared Face Analyzer
Centralized face detection and landmark caching for performance.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import mediapipe as mp


@dataclass
class FaceLandmarks:
    """Cached face landmarks for a single face."""
    # Raw landmarks (478 points)
    points: np.ndarray  # Shape: (478, 2) - x, y coordinates

    # Image dimensions
    image_width: int
    image_height: int

    # Pre-computed regions (pixel coordinates)
    face_oval: np.ndarray = field(default_factory=lambda: np.array([]))
    left_eye: np.ndarray = field(default_factory=lambda: np.array([]))
    right_eye: np.ndarray = field(default_factory=lambda: np.array([]))
    left_eyebrow: np.ndarray = field(default_factory=lambda: np.array([]))
    right_eyebrow: np.ndarray = field(default_factory=lambda: np.array([]))
    nose: np.ndarray = field(default_factory=lambda: np.array([]))
    lips_outer: np.ndarray = field(default_factory=lambda: np.array([]))
    lips_inner: np.ndarray = field(default_factory=lambda: np.array([]))
    left_iris: np.ndarray = field(default_factory=lambda: np.array([]))
    right_iris: np.ndarray = field(default_factory=lambda: np.array([]))

    # Computed centers and bounds
    left_eye_center: Tuple[int, int] = (0, 0)
    right_eye_center: Tuple[int, int] = (0, 0)
    nose_tip: Tuple[int, int] = (0, 0)
    chin: Tuple[int, int] = (0, 0)
    forehead: Tuple[int, int] = (0, 0)
    face_center: Tuple[int, int] = (0, 0)


# MediaPipe landmark indices for different face regions
LANDMARK_INDICES = {
    # Face oval (36 points)
    'face_oval': [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ],

    # Left eye contour
    'left_eye': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],

    # Right eye contour
    'right_eye': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],

    # Left eyebrow
    'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],

    # Right eyebrow
    'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],

    # Nose
    'nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18,
             200, 199, 175, 152],
    'nose_tip': [1],
    'nose_bridge': [6, 197, 195, 5],
    'nose_bottom': [2, 98, 327],

    # Lips outer contour
    'lips_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],

    # Lips inner contour
    'lips_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],

    # Iris (requires refine_landmarks=True)
    'left_iris': [468, 469, 470, 471, 472],
    'right_iris': [473, 474, 475, 476, 477],

    # Under-eye area (for dark circles)
    'left_under_eye': [111, 117, 118, 119, 120, 121, 128, 245, 193, 168],
    'right_under_eye': [340, 346, 347, 348, 349, 350, 357, 465, 417, 168],

    # Chin and jawline
    'chin': [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162],
    'jawline_left': [132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
    'jawline_right': [361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176],

    # Forehead approximation (top of face)
    'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 67, 109, 103, 54, 21, 162, 127],

    # Cheeks
    'left_cheek': [234, 93, 132, 58, 172, 136, 150, 149, 176, 148],
    'right_cheek': [454, 323, 361, 288, 397, 365, 379, 378, 400, 377],

    # Smile line / nasolabial fold
    'left_smile_line': [36, 205, 206, 207, 187],
    'right_smile_line': [266, 425, 426, 427, 411],
}


class FaceAnalyzer:
    """
    Centralized face analyzer that detects landmarks once and caches them.
    All processors share this analyzer for better performance.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern - only one instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._face_mesh = None
        self._current_image_hash = None
        self._cached_landmarks: List[FaceLandmarks] = []
        self._initialized = True
        self._load_model()

    def _load_model(self):
        """Load MediaPipe FaceMesh model."""
        try:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,  # Enables iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("FaceAnalyzer: MediaPipe FaceMesh loaded")
        except Exception as e:
            print(f"FaceAnalyzer: Failed to load MediaPipe: {e}")
            self._face_mesh = None

    def _compute_image_hash(self, image: np.ndarray) -> int:
        """Compute a hash of the image for cache invalidation."""
        # Use a fast hash based on image shape and sample pixels
        h, w = image.shape[:2]
        sample = image[::max(1, h//10), ::max(1, w//10)].flatten()
        return hash((h, w, sample.tobytes()[:1000]))

    def analyze(self, image: np.ndarray, force: bool = False) -> List[FaceLandmarks]:
        """
        Analyze image and return face landmarks.
        Results are cached - repeated calls with same image return cached results.

        Args:
            image: RGB image (numpy array)
            force: Force re-analysis even if cached

        Returns:
            List of FaceLandmarks for each detected face
        """
        if self._face_mesh is None:
            return []

        # Check cache
        image_hash = self._compute_image_hash(image)
        if not force and image_hash == self._current_image_hash:
            return self._cached_landmarks

        # Analyze image
        results = self._face_mesh.process(image)

        self._cached_landmarks = []
        self._current_image_hash = image_hash

        if not results.multi_face_landmarks:
            return []

        h, w = image.shape[:2]

        for face_lm in results.multi_face_landmarks:
            # Convert normalized landmarks to pixel coordinates
            points = np.array([
                [int(lm.x * w), int(lm.y * h)]
                for lm in face_lm.landmark
            ])

            # Create FaceLandmarks object
            landmarks = FaceLandmarks(
                points=points,
                image_width=w,
                image_height=h
            )

            # Extract regions
            for region_name, indices in LANDMARK_INDICES.items():
                region_points = points[indices] if all(i < len(points) for i in indices) else np.array([])
                if hasattr(landmarks, region_name):
                    setattr(landmarks, region_name, region_points)

            # Compute centers
            if len(landmarks.left_eye) > 0:
                landmarks.left_eye_center = tuple(np.mean(landmarks.left_eye, axis=0).astype(int))
            if len(landmarks.right_eye) > 0:
                landmarks.right_eye_center = tuple(np.mean(landmarks.right_eye, axis=0).astype(int))

            landmarks.nose_tip = tuple(points[1])  # Nose tip landmark
            landmarks.chin = tuple(points[152])  # Chin landmark
            landmarks.forehead = tuple(points[10])  # Top of head
            landmarks.face_center = tuple(points[1])  # Use nose as center

            self._cached_landmarks.append(landmarks)

        return self._cached_landmarks

    def get_region_mask(self, image: np.ndarray, face_idx: int,
                        region: str, expand: float = 1.0,
                        feather: int = 15) -> Optional[np.ndarray]:
        """
        Get a mask for a specific facial region.

        Args:
            image: The image
            face_idx: Index of the face (0 for first face)
            region: Region name (e.g., 'left_eye', 'nose', 'lips_outer')
            expand: Expansion factor (1.0 = no expansion)
            feather: Gaussian blur kernel size for soft edges

        Returns:
            Grayscale mask (0-255) or None if region not found
        """
        landmarks = self.analyze(image)

        if face_idx >= len(landmarks):
            return None

        face = landmarks[face_idx]
        h, w = image.shape[:2]

        # Get region points
        if not hasattr(face, region):
            return None

        points = getattr(face, region)
        if len(points) < 3:
            return None

        # Expand region if needed
        if expand != 1.0:
            center = np.mean(points, axis=0)
            points = ((points - center) * expand + center).astype(np.int32)

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)

        # Feather edges
        if feather > 0:
            if feather % 2 == 0:
                feather += 1
            mask = cv2.GaussianBlur(mask, (feather, feather), 0)

        return mask

    def clear_cache(self):
        """Clear the landmark cache."""
        self._current_image_hash = None
        self._cached_landmarks = []


# Global instance
_analyzer: Optional[FaceAnalyzer] = None


def get_face_analyzer() -> FaceAnalyzer:
    """Get the shared FaceAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FaceAnalyzer()
    return _analyzer
