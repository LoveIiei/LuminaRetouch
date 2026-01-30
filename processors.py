"""
LuminaRetouch - Image Processors
All portrait retouching processors using shared face analysis.
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
from dataclasses import dataclass

from face_analyzer import get_face_analyzer, FaceLandmarks, LANDMARK_INDICES


@dataclass
class ProcessingSettings:
    """All retouching settings."""
    # Face Enhancement
    face_enhancement: float = 0.0

    # Skin
    skin_smoothness: float = 0.0
    blemish_removal: float = 0.0

    # Eyes
    eye_brightness: float = 0.0
    eye_size: float = 0.0
    dark_circle_removal: float = 0.0

    # Mouth
    teeth_whitening: float = 0.0
    lip_saturation: float = 0.0
    smile_enhancement: float = 0.0

    # Face Shape
    face_slimming: float = 0.0
    nose_slimming: float = 0.0
    chin_adjustment: float = 0.0
    jawline_sharpen: float = 0.0

    # Output
    upscale_factor: int = 1

    def is_empty(self) -> bool:
        """Check if all settings are at zero."""
        return all(v == 0 for k, v in self.__dict__.items() if k != 'upscale_factor')


class BaseProcessor:
    """Base class for all processors."""

    def __init__(self):
        self.analyzer = get_face_analyzer()

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        """Process the image. Override in subclasses."""
        return image

    def _get_landmarks(self, image: np.ndarray,
                       landmarks: Optional[List[FaceLandmarks]] = None) -> List[FaceLandmarks]:
        """Get landmarks, using cache if available."""
        if landmarks is not None:
            return landmarks
        return self.analyzer.analyze(image)


class SkinSmoothingProcessor(BaseProcessor):
    """Frequency separation skin smoothing."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()
        h, w = image.shape[:2]

        for face in faces:
            # Create skin mask (face oval minus eyes, lips)
            mask = np.zeros((h, w), dtype=np.uint8)

            if len(face.face_oval) >= 3:
                cv2.fillPoly(mask, [face.face_oval.astype(np.int32)], 255)

            # Exclude eyes
            if len(face.left_eye) >= 3:
                cv2.fillPoly(mask, [face.left_eye.astype(np.int32)], 0)
            if len(face.right_eye) >= 3:
                cv2.fillPoly(mask, [face.right_eye.astype(np.int32)], 0)

            # Exclude eyebrows
            if len(face.left_eyebrow) >= 3:
                cv2.fillPoly(mask, [face.left_eyebrow.astype(np.int32)], 0)
            if len(face.right_eyebrow) >= 3:
                cv2.fillPoly(mask, [face.right_eyebrow.astype(np.int32)], 0)

            # Exclude lips
            if len(face.lips_outer) >= 3:
                cv2.fillPoly(mask, [face.lips_outer.astype(np.int32)], 0)

            # Feather mask
            mask = cv2.GaussianBlur(mask, (21, 21), 0)

            # Apply frequency separation
            result = self._frequency_separation(result, mask, strength)

        return result

    def _frequency_separation(self, image: np.ndarray, mask: np.ndarray,
                              strength: float) -> np.ndarray:
        """Apply frequency separation smoothing."""
        img_float = image.astype(np.float32)

        # Blur radius based on image size
        blur_radius = max(3, int(min(image.shape[:2]) * 0.015))
        if blur_radius % 2 == 0:
            blur_radius += 1

        # Low frequency (color/tone)
        low_freq = cv2.GaussianBlur(img_float, (blur_radius, blur_radius), 0)

        # High frequency (texture)
        high_freq = img_float - low_freq + 128

        # Smooth low frequency more
        smooth_radius = blur_radius * 2 + 1
        low_freq_smooth = cv2.GaussianBlur(low_freq, (smooth_radius, smooth_radius), 0)

        # Blend
        low_freq_blended = cv2.addWeighted(low_freq, 1 - strength, low_freq_smooth, strength, 0)

        # Recombine
        result = low_freq_blended + high_freq - 128

        # Apply mask
        mask_float = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        result = img_float * (1 - mask_float) + result * mask_float

        return np.clip(result, 0, 255).astype(np.uint8)


class BlemishRemovalProcessor(BaseProcessor):
    """Remove blemishes/acne using inpainting."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()
        h, w = image.shape[:2]

        for face in faces:
            # Create skin region mask
            skin_mask = np.zeros((h, w), dtype=np.uint8)
            if len(face.face_oval) >= 3:
                cv2.fillPoly(skin_mask, [face.face_oval.astype(np.int32)], 255)

            # Exclude eyes, lips, eyebrows
            for region in [face.left_eye, face.right_eye, face.lips_outer,
                          face.left_eyebrow, face.right_eyebrow]:
                if len(region) >= 3:
                    cv2.fillPoly(skin_mask, [region.astype(np.int32)], 0)

            # Detect blemishes (dark spots that contrast with surrounding skin)
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

            # Adaptive threshold to find spots
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            diff = cv2.absdiff(gray, blur)

            # Threshold based on strength
            thresh_val = int(20 - strength * 10)  # Lower = more sensitive
            _, blemish_mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

            # Only in skin region
            blemish_mask = cv2.bitwise_and(blemish_mask, skin_mask)

            # Dilate slightly for better inpainting
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            blemish_mask = cv2.dilate(blemish_mask, kernel, iterations=1)

            # Inpaint
            if np.sum(blemish_mask) > 0:
                result = cv2.inpaint(result, blemish_mask, 3, cv2.INPAINT_TELEA)

        return result


class EyeBrightnessProcessor(BaseProcessor):
    """Brighten eyes (iris and whites)."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()
        h, w = image.shape[:2]

        for face in faces:
            # Create eye mask
            eye_mask = np.zeros((h, w), dtype=np.uint8)

            # Use iris landmarks if available, otherwise full eye
            for iris in [face.left_iris, face.right_iris]:
                if len(iris) >= 3:
                    center = np.mean(iris, axis=0).astype(int)
                    radius = int(np.max([np.linalg.norm(p - center) for p in iris]) * 1.8)
                    cv2.circle(eye_mask, tuple(center), radius, 255, -1)

            if np.sum(eye_mask) < 10:
                # Fallback to eye contours
                for eye in [face.left_eye, face.right_eye]:
                    if len(eye) >= 3:
                        cv2.fillPoly(eye_mask, [eye.astype(np.int32)], 255)

            # Feather
            eye_mask = cv2.GaussianBlur(eye_mask, (15, 15), 0)
            mask_float = (eye_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

            # Brighten using HSV
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + 0.3 * strength), 0, 255)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + 0.15 * strength), 0, 255)
            brightened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            result = (result * (1 - mask_float) + brightened * mask_float).astype(np.uint8)

        return result


class EyeSizeProcessor(BaseProcessor):
    """Enlarge eyes using warping."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()

        for face in faces:
            # Enlarge each eye
            for eye_center in [face.left_eye_center, face.right_eye_center]:
                if eye_center != (0, 0):
                    result = self._enlarge_region(result, eye_center, strength * 0.15)

        return result

    def _enlarge_region(self, image: np.ndarray, center: Tuple[int, int],
                        strength: float) -> np.ndarray:
        """Enlarge a circular region (bulge effect)."""
        h, w = image.shape[:2]
        radius = int(min(h, w) * 0.08)

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        cx, cy = center

        # Distance from center
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Normalized distance (0 at center, 1 at radius)
        norm_dist = dist / radius
        norm_dist = np.clip(norm_dist, 0, 1)

        # Bulge factor (stronger at center)
        factor = 1 - (1 - norm_dist ** 2) * strength

        # Only affect pixels within radius
        mask = dist < radius

        # Calculate new coordinates
        new_x = cx + (x - cx) * np.where(mask, factor, 1)
        new_y = cy + (y - cy) * np.where(mask, factor, 1)

        # Remap
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class DarkCircleRemovalProcessor(BaseProcessor):
    """Remove dark circles under eyes."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()
        h, w = image.shape[:2]

        for face in faces:
            # Get under-eye regions
            under_eye_mask = np.zeros((h, w), dtype=np.uint8)

            # Create under-eye regions manually from eye landmarks
            for eye in [face.left_eye, face.right_eye]:
                if len(eye) >= 3:
                    # Get bottom half of eye region, extended down
                    eye_center = np.mean(eye, axis=0)
                    eye_bottom = eye[eye[:, 1] > eye_center[1]]

                    if len(eye_bottom) >= 2:
                        # Create under-eye polygon
                        offset = int(min(h, w) * 0.02)
                        under_eye = eye_bottom.copy()
                        under_eye[:, 1] += offset  # Shift down

                        # Add points to close polygon
                        points = np.vstack([eye_bottom, under_eye[::-1]])
                        cv2.fillPoly(under_eye_mask, [points.astype(np.int32)], 255)

            # Feather mask
            under_eye_mask = cv2.GaussianBlur(under_eye_mask, (21, 21), 0)
            mask_float = (under_eye_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

            # Color correct: reduce blue/purple, brighten
            lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)

            # Increase L (lightness)
            lab[:, :, 0] = np.clip(lab[:, :, 0] + 15 * strength, 0, 255)
            # Reduce blue (increase b)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + 8 * strength, 0, 255)
            # Reduce purple/red slightly (reduce a)
            lab[:, :, 1] = np.clip(lab[:, :, 1] - 3 * strength, 0, 255)

            corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            result = (result * (1 - mask_float) + corrected * mask_float).astype(np.uint8)

        return result


class TeethWhiteningProcessor(BaseProcessor):
    """Whiten teeth."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()
        h, w = image.shape[:2]

        for face in faces:
            if len(face.lips_inner) < 3:
                continue

            # Create mouth opening mask
            mouth_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mouth_mask, [face.lips_inner.astype(np.int32)], 255)

            # Detect bright areas (teeth) within mouth
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            _, bright_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            teeth_mask = cv2.bitwise_and(mouth_mask, bright_mask)

            # Feather
            teeth_mask = cv2.GaussianBlur(teeth_mask, (7, 7), 0)
            mask_float = teeth_mask.astype(np.float32) / 255.0

            # Whiten in LAB space
            lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:, :, 0] = np.clip(lab[:, :, 0] + 20 * strength * mask_float, 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] - 15 * strength * mask_float, 0, 255)

            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        return result


class LipSaturationProcessor(BaseProcessor):
    """Enhance lip color saturation."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()
        h, w = image.shape[:2]

        for face in faces:
            if len(face.lips_outer) < 3:
                continue

            # Create lip mask
            lip_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(lip_mask, [face.lips_outer.astype(np.int32)], 255)

            # Exclude inner mouth
            if len(face.lips_inner) >= 3:
                cv2.fillPoly(lip_mask, [face.lips_inner.astype(np.int32)], 0)

            # Feather
            lip_mask = cv2.GaussianBlur(lip_mask, (11, 11), 0)
            mask_float = (lip_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

            # Boost saturation
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + 0.4 * strength), 0, 255)
            enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            result = (result * (1 - mask_float) + enhanced * mask_float).astype(np.uint8)

        return result


class SmileEnhancementProcessor(BaseProcessor):
    """Enhance smile (lift mouth corners)."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()

        for face in faces:
            if len(face.lips_outer) < 10:
                continue

            # Mouth corners (approximate indices)
            left_corner = face.points[61]  # Left mouth corner
            right_corner = face.points[291]  # Right mouth corner

            # Lift corners upward
            lift = int(min(face.image_height, face.image_width) * 0.01 * strength)

            for corner in [left_corner, right_corner]:
                src = tuple(corner)
                dst = (corner[0], corner[1] - lift)
                result = self._local_warp(result, src, dst, radius=30, strength=strength * 0.5)

        return result

    def _local_warp(self, image: np.ndarray, src: Tuple[int, int],
                    dst: Tuple[int, int], radius: int, strength: float) -> np.ndarray:
        """Apply local warping."""
        h, w = image.shape[:2]
        y, x = np.mgrid[:h, :w]

        dist = np.sqrt((x - src[0]) ** 2 + (y - src[1]) ** 2)
        falloff = np.maximum(0, 1 - dist / radius) ** 2

        dx = (dst[0] - src[0]) * strength * falloff
        dy = (dst[1] - src[1]) * strength * falloff

        map_x = (x - dx).astype(np.float32)
        map_y = (y - dy).astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class FaceSlimmingProcessor(BaseProcessor):
    """Slim face by warping cheeks inward."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()

        for face in faces:
            # Warp cheeks toward center
            center = face.face_center

            # Left cheek points
            left_cheek = face.points[LANDMARK_INDICES['left_cheek']]
            for point in left_cheek[::2]:  # Every other point
                result = self._warp_toward(result, tuple(point), center, strength * 0.08)

            # Right cheek points
            right_cheek = face.points[LANDMARK_INDICES['right_cheek']]
            for point in right_cheek[::2]:
                result = self._warp_toward(result, tuple(point), center, strength * 0.08)

        return result

    def _warp_toward(self, image: np.ndarray, src: Tuple[int, int],
                     dst: Tuple[int, int], strength: float) -> np.ndarray:
        """Warp region toward destination."""
        h, w = image.shape[:2]
        radius = int(min(h, w) * 0.1)

        y, x = np.mgrid[:h, :w]
        dist = np.sqrt((x - src[0]) ** 2 + (y - src[1]) ** 2)
        falloff = np.maximum(0, 1 - dist / radius) ** 2

        dx = (dst[0] - src[0]) * strength * falloff
        dy = (dst[1] - src[1]) * strength * falloff

        map_x = (x - dx).astype(np.float32)
        map_y = (y - dy).astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class NoseSlimmingProcessor(BaseProcessor):
    """Slim nose by warping sides inward."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()

        for face in faces:
            nose_tip = face.nose_tip

            # Nose side points (nostrils area)
            left_nostril = face.points[219]  # Approximate
            right_nostril = face.points[439]

            # Warp nostrils toward nose center
            center_x = nose_tip[0]

            for nostril in [left_nostril, right_nostril]:
                src = tuple(nostril)
                dst = (center_x, nostril[1])
                result = self._local_warp(result, src, dst, strength * 0.3)

        return result

    def _local_warp(self, image: np.ndarray, src: Tuple[int, int],
                    dst: Tuple[int, int], strength: float) -> np.ndarray:
        """Apply local warping."""
        h, w = image.shape[:2]
        radius = int(min(h, w) * 0.05)

        y, x = np.mgrid[:h, :w]
        dist = np.sqrt((x - src[0]) ** 2 + (y - src[1]) ** 2)
        falloff = np.maximum(0, 1 - dist / radius) ** 2

        dx = (dst[0] - src[0]) * strength * falloff
        dy = (dst[1] - src[1]) * strength * falloff

        map_x = (x - dx).astype(np.float32)
        map_y = (y - dy).astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class ChinAdjustmentProcessor(BaseProcessor):
    """Adjust chin (reduce double chin)."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()

        for face in faces:
            chin = face.chin

            # Push chin area upward
            chin_point = face.points[152]  # Bottom of chin
            jaw_center = face.points[175]  # Higher jaw point

            src = tuple(chin_point)
            dst = (chin_point[0], chin_point[1] - int(strength * 15))

            result = self._local_warp(result, src, dst, strength * 0.5, radius_factor=0.12)

        return result

    def _local_warp(self, image: np.ndarray, src: Tuple[int, int],
                    dst: Tuple[int, int], strength: float,
                    radius_factor: float = 0.1) -> np.ndarray:
        """Apply local warping."""
        h, w = image.shape[:2]
        radius = int(min(h, w) * radius_factor)

        y, x = np.mgrid[:h, :w]
        dist = np.sqrt((x - src[0]) ** 2 + (y - src[1]) ** 2)
        falloff = np.maximum(0, 1 - dist / radius) ** 2

        dx = (dst[0] - src[0]) * strength * falloff
        dy = (dst[1] - src[1]) * strength * falloff

        map_x = (x - dx).astype(np.float32)
        map_y = (y - dy).astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class JawlineSharpenProcessor(BaseProcessor):
    """Sharpen jawline definition."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()

        for face in faces:
            # Warp jawline points inward/upward
            face_center = face.face_center

            # Left jawline
            left_jaw = face.points[LANDMARK_INDICES['jawline_left']]
            for point in left_jaw[::2]:
                # Move toward center and up
                dst = (
                    int(point[0] + (face_center[0] - point[0]) * 0.05),
                    int(point[1] - 3)
                )
                result = self._local_warp(result, tuple(point), dst, strength * 0.3)

            # Right jawline
            right_jaw = face.points[LANDMARK_INDICES['jawline_right']]
            for point in right_jaw[::2]:
                dst = (
                    int(point[0] + (face_center[0] - point[0]) * 0.05),
                    int(point[1] - 3)
                )
                result = self._local_warp(result, tuple(point), dst, strength * 0.3)

        return result

    def _local_warp(self, image: np.ndarray, src: Tuple[int, int],
                    dst: Tuple[int, int], strength: float) -> np.ndarray:
        """Apply local warping."""
        h, w = image.shape[:2]
        radius = int(min(h, w) * 0.06)

        y, x = np.mgrid[:h, :w]
        dist = np.sqrt((x - src[0]) ** 2 + (y - src[1]) ** 2)
        falloff = np.maximum(0, 1 - dist / radius) ** 2

        dx = (dst[0] - src[0]) * strength * falloff
        dy = (dst[1] - src[1]) * strength * falloff

        map_x = (x - dx).astype(np.float32)
        map_y = (y - dy).astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class FaceEnhancementProcessor(BaseProcessor):
    """Enhance facial details (sharpen eyes, lips, etc.)."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.astype(np.float32)
        h, w = image.shape[:2]

        for face in faces:
            # Create detail mask (eyes, eyebrows, lips)
            detail_mask = np.zeros((h, w), dtype=np.uint8)

            for region in [face.left_eye, face.right_eye,
                          face.left_eyebrow, face.right_eyebrow,
                          face.lips_outer]:
                if len(region) >= 3:
                    # Expand region slightly
                    center = np.mean(region, axis=0)
                    expanded = ((region - center) * 1.3 + center).astype(np.int32)
                    cv2.fillPoly(detail_mask, [expanded], 255)

            # Feather mask
            detail_mask = cv2.GaussianBlur(detail_mask, (15, 15), 0)
            mask_float = (detail_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

            # Unsharp mask for detail enhancement
            blurred = cv2.GaussianBlur(result, (0, 0), 3)
            sharpened = cv2.addWeighted(result, 1 + 0.5 * strength, blurred, -0.5 * strength, 0)

            result = result * (1 - mask_float) + sharpened * mask_float

        return np.clip(result, 0, 255).astype(np.uint8)


class UpscalingProcessor(BaseProcessor):
    """Upscale image using high-quality interpolation."""

    def process(self, image: np.ndarray, scale: int,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if scale <= 1:
            return image

        h, w = image.shape[:2]
        new_size = (w * scale, h * scale)

        # Use Lanczos for high-quality upscaling
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)


class RetouchPipeline:
    """
    Main processing pipeline that applies all retouching effects.
    Uses shared face analysis for performance.
    """

    def __init__(self):
        self.analyzer = get_face_analyzer()

        # Initialize all processors
        self.processors = {
            'face_enhancement': FaceEnhancementProcessor(),
            'skin_smoothness': SkinSmoothingProcessor(),
            'blemish_removal': BlemishRemovalProcessor(),
            'eye_brightness': EyeBrightnessProcessor(),
            'eye_size': EyeSizeProcessor(),
            'dark_circle_removal': DarkCircleRemovalProcessor(),
            'teeth_whitening': TeethWhiteningProcessor(),
            'lip_saturation': LipSaturationProcessor(),
            'smile_enhancement': SmileEnhancementProcessor(),
            'face_slimming': FaceSlimmingProcessor(),
            'nose_slimming': NoseSlimmingProcessor(),
            'chin_adjustment': ChinAdjustmentProcessor(),
            'jawline_sharpen': JawlineSharpenProcessor(),
            'upscale': UpscalingProcessor(),
        }

    def process(self, image: np.ndarray, settings: ProcessingSettings,
                progress_callback=None) -> np.ndarray:
        """
        Process image through the full retouching pipeline.

        Args:
            image: Input RGB image
            settings: Processing settings
            progress_callback: Optional callback(step_name, progress_percent)

        Returns:
            Processed image
        """
        result = image.copy()

        # Analyze face once (cached)
        landmarks = self.analyzer.analyze(result)

        if not landmarks:
            if progress_callback:
                progress_callback("No faces detected", 100)
            return result

        # Processing order (optimized for best results)
        steps = [
            ('face_enhancement', settings.face_enhancement),
            ('skin_smoothness', settings.skin_smoothness),
            ('blemish_removal', settings.blemish_removal),
            ('dark_circle_removal', settings.dark_circle_removal),
            ('eye_brightness', settings.eye_brightness),
            ('eye_size', settings.eye_size),
            ('teeth_whitening', settings.teeth_whitening),
            ('lip_saturation', settings.lip_saturation),
            ('smile_enhancement', settings.smile_enhancement),
            ('nose_slimming', settings.nose_slimming),
            ('face_slimming', settings.face_slimming),
            ('chin_adjustment', settings.chin_adjustment),
            ('jawline_sharpen', settings.jawline_sharpen),
        ]

        total_steps = len([s for s in steps if s[1] > 0]) + (1 if settings.upscale_factor > 1 else 0)
        current_step = 0

        for step_name, strength in steps:
            if strength > 0:
                if progress_callback:
                    progress_callback(step_name.replace('_', ' ').title(),
                                    int(current_step / total_steps * 100))

                processor = self.processors.get(step_name)
                if processor:
                    result = processor.process(result, strength, landmarks)

                # Re-analyze if face shape changed (for subsequent shape processors)
                if step_name in ['face_slimming', 'nose_slimming', 'chin_adjustment']:
                    landmarks = self.analyzer.analyze(result, force=True)

                current_step += 1

        # Upscaling (last step)
        if settings.upscale_factor > 1:
            if progress_callback:
                progress_callback("Upscaling", 95)
            result = self.processors['upscale'].process(result, settings.upscale_factor)

        if progress_callback:
            progress_callback("Complete", 100)

        return result
