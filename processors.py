"""
LuminaRetouch - Image Processors
All portrait retouching processors using shared face analysis.
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
from dataclasses import dataclass

from face_analyzer import get_face_analyzer, FaceLandmarks, LANDMARK_INDICES

# Try to import AI enhancer
try:
    from ai_enhancer import (
        check_gfpgan_available, enhance_face_ai, enhance_face_traditional,
        remove_blemishes_ai, get_skin_mask
    )
    AI_AVAILABLE = check_gfpgan_available()
except ImportError:
    AI_AVAILABLE = False

# Runtime flag for AI usage (set by pipeline)
USE_AI_MODE = False


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
    """
    Professional skin smoothing using AI (GFPGAN) or enhanced bilateral filtering.
    Achieves natural, soft skin while preserving texture and details.
    """

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()
        h, w = image.shape[:2]

        # Try AI enhancement if Quality mode is selected and available
        if USE_AI_MODE and AI_AVAILABLE and strength > 0.3:
            try:
                result = enhance_face_ai(result, strength * 0.7)
            except Exception:
                pass  # Fall through to traditional method

        # Apply additional smoothing with masking for precise control
        for face in faces:
            mask = self._create_skin_mask(face, h, w)
            result = self._smooth_skin(result, mask, strength)

        return result

    def _create_skin_mask(self, face: FaceLandmarks, h: int, w: int) -> np.ndarray:
        """Create precise skin mask excluding eyes, lips, eyebrows."""
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(face.face_oval) >= 3:
            cv2.fillPoly(mask, [face.face_oval.astype(np.int32)], 255)

        # Exclude facial features with expanded regions
        exclude_regions = [
            (face.left_eye, 1.4),
            (face.right_eye, 1.4),
            (face.left_eyebrow, 1.3),
            (face.right_eyebrow, 1.3),
            (face.lips_outer, 1.2),
        ]

        for region, expand in exclude_regions:
            if len(region) >= 3:
                center = np.mean(region, axis=0)
                expanded = ((region - center) * expand + center).astype(np.int32)
                cv2.fillPoly(mask, [expanded], 0)

        # Feather edges for smooth blending
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        return mask

    def _smooth_skin(self, image: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
        """Apply professional skin smoothing using bilateral filtering."""
        # Multi-pass bilateral filtering - preserves edges while smoothing
        smoothed = image.copy()

        # Parameters scale with strength
        d = 9  # Diameter of pixel neighborhood
        sigma_color = 50 + strength * 75  # Filter sigma in color space
        sigma_space = 50 + strength * 75  # Filter sigma in coordinate space

        # Multiple passes for stronger effect
        passes = max(1, int(strength * 4))
        for _ in range(passes):
            smoothed = cv2.bilateralFilter(smoothed, d, sigma_color, sigma_space)

        # Apply surface blur effect (additional smoothing while preserving edges)
        if strength > 0.5:
            # Edge-preserving filter
            smoothed = cv2.edgePreservingFilter(smoothed, flags=1, sigma_s=60, sigma_r=0.4 * strength)

        # Blend with original using mask
        mask_float = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

        # Control blend amount - never go 100% to preserve some natural texture
        blend_strength = min(strength * 0.85, 0.95)
        mask_float = mask_float * blend_strength

        result = image.astype(np.float32) * (1 - mask_float) + smoothed.astype(np.float32) * mask_float
        return np.clip(result, 0, 255).astype(np.uint8)


class BlemishRemovalProcessor(BaseProcessor):
    """
    Professional blemish removal using AI-enhanced detection and inpainting.
    Detects and removes acne, spots, redness, and skin discoloration.
    """

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
            # Create comprehensive skin mask
            skin_mask = self._create_skin_mask(face, h, w)

            # Detect blemishes using multiple methods
            blemish_mask = self._detect_blemishes(result, skin_mask, strength)

            # Remove blemishes using high-quality inpainting
            if np.sum(blemish_mask) > 0:
                result = self._remove_blemishes(result, blemish_mask)

        return result

    def _create_skin_mask(self, face: FaceLandmarks, h: int, w: int) -> np.ndarray:
        """Create skin region mask."""
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(face.face_oval) >= 3:
            cv2.fillPoly(mask, [face.face_oval.astype(np.int32)], 255)

        # Exclude eyes, lips, eyebrows with expansion
        for region in [face.left_eye, face.right_eye, face.lips_outer,
                       face.left_eyebrow, face.right_eyebrow]:
            if len(region) >= 3:
                center = np.mean(region, axis=0)
                expanded = ((region - center) * 1.3 + center).astype(np.int32)
                cv2.fillPoly(mask, [expanded], 0)

        return mask

    def _detect_blemishes(self, image: np.ndarray, skin_mask: np.ndarray,
                          strength: float) -> np.ndarray:
        """Multi-method blemish detection."""
        h, w = image.shape[:2]
        blemish_mask = np.zeros((h, w), dtype=np.uint8)

        # Convert to different color spaces for comprehensive detection
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        # Adaptive blur size based on image
        blur_size = max(15, int(min(h, w) * 0.02))
        if blur_size % 2 == 0:
            blur_size += 1

        # 1. Dark spots (lower L than surrounding)
        l_blur = cv2.GaussianBlur(l_channel, (blur_size, blur_size), 0)
        dark_diff = l_blur - l_channel
        dark_thresh = 10 - strength * 6
        dark_spots = (dark_diff > dark_thresh).astype(np.uint8) * 255
        blemish_mask = cv2.bitwise_or(blemish_mask, dark_spots)

        # 2. Bright spots (higher L than surrounding)
        bright_diff = l_channel - l_blur
        bright_thresh = 12 - strength * 6
        bright_spots = (bright_diff > bright_thresh).astype(np.uint8) * 255
        blemish_mask = cv2.bitwise_or(blemish_mask, bright_spots)

        # 3. Redness detection (high 'a' channel)
        a_blur = cv2.GaussianBlur(a_channel, (blur_size, blur_size), 0)
        red_diff = a_channel - a_blur
        red_thresh = 6 - strength * 3
        red_spots = (red_diff > red_thresh).astype(np.uint8) * 255
        blemish_mask = cv2.bitwise_or(blemish_mask, red_spots)

        # 4. Yellow/brown spots (b channel deviation)
        b_blur = cv2.GaussianBlur(b_channel, (blur_size, blur_size), 0)
        b_diff = np.abs(b_channel - b_blur)
        brown_thresh = 8 - strength * 4
        brown_spots = (b_diff > brown_thresh).astype(np.uint8) * 255
        blemish_mask = cv2.bitwise_or(blemish_mask, brown_spots)

        # 5. Texture anomalies (high local variance)
        gray_blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        texture_diff = cv2.absdiff(gray, gray_blur)
        texture_thresh = 18 - strength * 10
        texture_spots = cv2.threshold(texture_diff, texture_thresh, 255, cv2.THRESH_BINARY)[1]
        blemish_mask = cv2.bitwise_or(blemish_mask, texture_spots)

        # Only in skin region
        blemish_mask = cv2.bitwise_and(blemish_mask, skin_mask)

        # Clean up mask
        # Remove noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        blemish_mask = cv2.morphologyEx(blemish_mask, cv2.MORPH_OPEN, kernel_open)

        # Remove very large regions (not blemishes, might be features)
        contours, _ = cv2.findContours(blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = (min(h, w) * 0.025) ** 2
        for cnt in contours:
            if cv2.contourArea(cnt) > max_area:
                cv2.drawContours(blemish_mask, [cnt], -1, 0, -1)

        # Dilate for better inpainting coverage
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blemish_mask = cv2.dilate(blemish_mask, kernel_dilate, iterations=1)

        return blemish_mask

    def _remove_blemishes(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove blemishes using high-quality inpainting."""
        # Use Navier-Stokes based inpainting for better results
        result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
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
    """Enlarge eyes using warping with iris-aware algorithm."""

    def process(self, image: np.ndarray, strength: float,
                landmarks: Optional[List[FaceLandmarks]] = None) -> np.ndarray:
        if strength <= 0:
            return image

        faces = self._get_landmarks(image, landmarks)
        if not faces:
            return image

        result = image.copy()

        for face in faces:
            # Get eye regions for more natural warping
            eye_data = [
                (face.left_eye_center, face.left_eye, face.left_iris),
                (face.right_eye_center, face.right_eye, face.right_iris)
            ]

            for eye_center, eye_contour, iris in eye_data:
                if eye_center != (0, 0):
                    # Calculate eye-specific radius based on actual eye size
                    if len(eye_contour) > 0:
                        eye_width = np.max(eye_contour[:, 0]) - np.min(eye_contour[:, 0])
                        eye_height = np.max(eye_contour[:, 1]) - np.min(eye_contour[:, 1])
                        radius = int(max(eye_width, eye_height) * 0.8)
                    else:
                        radius = int(min(image.shape[:2]) * 0.05)

                    # Get iris center if available for more natural enlargement
                    if len(iris) >= 3:
                        iris_center = tuple(np.mean(iris, axis=0).astype(int))
                    else:
                        iris_center = eye_center

                    # Limit strength to avoid unrealistic distortion
                    effective_strength = min(strength * 0.12, 0.15)
                    result = self._enlarge_region(result, iris_center, effective_strength, radius)

        return result

    def _enlarge_region(self, image: np.ndarray, center: Tuple[int, int],
                        strength: float, radius: int) -> np.ndarray:
        """Enlarge a circular region with smooth falloff."""
        h, w = image.shape[:2]

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        cx, cy = center

        # Distance from center
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Normalized distance (0 at center, 1 at radius)
        norm_dist = np.clip(dist / radius, 0, 1)

        # Use smoother cubic falloff for more natural look
        # This preserves more detail at the center (iris) while expanding outer regions
        falloff = (1 - norm_dist ** 3) ** 2
        factor = 1 - falloff * strength

        # Smooth transition at edge to avoid artifacts
        edge_blend = np.clip((radius - dist) / (radius * 0.2), 0, 1)
        factor = factor * edge_blend + 1 * (1 - edge_blend)

        # Calculate new coordinates
        new_x = cx + (x - cx) * factor
        new_y = cy + (y - cy) * factor

        # Remap with cubic interpolation for better quality
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


class DarkCircleRemovalProcessor(BaseProcessor):
    """Remove dark circles under eyes with subtle, natural blending."""

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
            under_eye_mask = np.zeros((h, w), dtype=np.uint8)

            # Create under-eye regions from eye landmarks
            for eye in [face.left_eye, face.right_eye]:
                if len(eye) >= 3:
                    eye_center = np.mean(eye, axis=0)
                    eye_bottom = eye[eye[:, 1] > eye_center[1]]

                    if len(eye_bottom) >= 2:
                        # Smaller, more focused under-eye region
                        offset = int(min(h, w) * 0.015)
                        under_eye = eye_bottom.copy()
                        under_eye[:, 1] += offset

                        points = np.vstack([eye_bottom, under_eye[::-1]])
                        cv2.fillPoly(under_eye_mask, [points.astype(np.int32)], 255)

            if np.sum(under_eye_mask) == 0:
                continue

            # Heavy feathering for smooth blending (larger kernel)
            under_eye_mask = cv2.GaussianBlur(under_eye_mask, (31, 31), 0)
            mask_float = under_eye_mask.astype(np.float32) / 255.0

            # Convert to LAB for color correction
            lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)

            # Subtle, conservative corrections
            # Only slightly brighten and reduce blue/purple tint
            effective_strength = strength * 0.4  # Reduce overall effect

            # Gentle lightness boost (max +8 at full strength)
            lab[:, :, 0] = np.clip(
                lab[:, :, 0] + 8 * effective_strength * mask_float, 0, 255)

            # Very subtle color correction
            # Reduce blue tint (increase b slightly)
            lab[:, :, 2] = np.clip(
                lab[:, :, 2] + 4 * effective_strength * mask_float, 0, 255)

            # Minimal red adjustment
            lab[:, :, 1] = np.clip(
                lab[:, :, 1] - 2 * effective_strength * mask_float, 0, 255)

            corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

            # Blend with reduced opacity for more natural look
            blend_strength = mask_float * 0.7  # Never fully replace original
            mask_3d = blend_strength[:, :, np.newaxis]
            result = (result * (1 - mask_3d) + corrected * mask_3d).astype(np.uint8)

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
        h, w = image.shape[:2]

        for face in faces:
            # Collect all warp points first, then apply once
            warp_points = []
            center = face.face_center
            radius = int(min(h, w) * 0.1)

            # Left cheek points
            left_cheek = face.points[LANDMARK_INDICES['left_cheek']]
            for point in left_cheek[::2]:
                warp_points.append((tuple(point), center, strength * 0.08))

            # Right cheek points
            right_cheek = face.points[LANDMARK_INDICES['right_cheek']]
            for point in right_cheek[::2]:
                warp_points.append((tuple(point), center, strength * 0.08))

            # Apply all warps in single remap
            result = self._batch_warp(result, warp_points, radius)

        return result

    def _batch_warp(self, image: np.ndarray, warp_points: List[Tuple],
                    radius: int) -> np.ndarray:
        """Apply multiple warps in a single remap operation."""
        h, w = image.shape[:2]
        y, x = np.mgrid[:h, :w]

        # Accumulate displacements
        total_dx = np.zeros((h, w), dtype=np.float32)
        total_dy = np.zeros((h, w), dtype=np.float32)

        for src, dst, strength in warp_points:
            dist = np.sqrt((x - src[0]) ** 2 + (y - src[1]) ** 2)
            falloff = np.maximum(0, 1 - dist / radius) ** 2

            total_dx += (dst[0] - src[0]) * strength * falloff
            total_dy += (dst[1] - src[1]) * strength * falloff

        map_x = (x - total_dx).astype(np.float32)
        map_y = (y - total_dy).astype(np.float32)

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
        h, w = image.shape[:2]

        for face in faces:
            warp_points = []
            face_center = face.face_center
            radius = int(min(h, w) * 0.06)

            # Left jawline
            left_jaw = face.points[LANDMARK_INDICES['jawline_left']]
            for point in left_jaw[::2]:
                dst = (
                    int(point[0] + (face_center[0] - point[0]) * 0.05),
                    int(point[1] - 3)
                )
                warp_points.append((tuple(point), dst, strength * 0.3))

            # Right jawline
            right_jaw = face.points[LANDMARK_INDICES['jawline_right']]
            for point in right_jaw[::2]:
                dst = (
                    int(point[0] + (face_center[0] - point[0]) * 0.05),
                    int(point[1] - 3)
                )
                warp_points.append((tuple(point), dst, strength * 0.3))

            # Apply all warps in single remap
            result = self._batch_warp(result, warp_points, radius)

        return result

    def _batch_warp(self, image: np.ndarray, warp_points: List[Tuple],
                    radius: int) -> np.ndarray:
        """Apply multiple warps in a single remap operation."""
        h, w = image.shape[:2]
        y, x = np.mgrid[:h, :w]

        total_dx = np.zeros((h, w), dtype=np.float32)
        total_dy = np.zeros((h, w), dtype=np.float32)

        for src, dst, strength in warp_points:
            dist = np.sqrt((x - src[0]) ** 2 + (y - src[1]) ** 2)
            falloff = np.maximum(0, 1 - dist / radius) ** 2

            total_dx += (dst[0] - src[0]) * strength * falloff
            total_dy += (dst[1] - src[1]) * strength * falloff

        map_x = (x - total_dx).astype(np.float32)
        map_y = (y - total_dy).astype(np.float32)

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
                progress_callback=None, use_ai: bool = False) -> np.ndarray:
        """
        Process image through the full retouching pipeline.

        Args:
            image: Input RGB image
            settings: Processing settings
            progress_callback: Optional callback(step_name, progress_percent)
            use_ai: Whether to use AI enhancement (Quality mode)

        Returns:
            Processed image
        """
        result = image.copy()

        # Set global AI mode flag for processors
        global USE_AI_MODE
        USE_AI_MODE = use_ai

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

                current_step += 1

        # Upscaling (last step)
        if settings.upscale_factor > 1:
            if progress_callback:
                progress_callback("Upscaling", 95)
            result = self.processors['upscale'].process(result, settings.upscale_factor)

        if progress_callback:
            progress_callback("Complete", 100)

        return result
