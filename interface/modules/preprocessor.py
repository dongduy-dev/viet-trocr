# =============================================================================
# preprocessor.py — Step 1: Image Preprocessing
# CLAHE contrast enhancement + Hough-based deskew
# =============================================================================

import cv2
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Step 1 of the OCR pipeline: Image Preprocessing.

    Operations:
      1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
         Applied on the L channel of LAB color space.
      2. Deskew via Hough Line Transform
         Detects dominant line angle and corrects rotation.
    """

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size),
        )

    def enhance_clahe(self, image: Image.Image) -> Image.Image:
        """Apply CLAHE to enhance contrast."""
        img_np = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_ch)
        lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)

    def deskew(self, image: Image.Image, max_angle: float = 15.0):
        """
        Deskew image using Hough Line Transform.

        Returns:
            (deskewed_image, angle_degrees)
        """
        img_gray = np.array(image.convert("L"))
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=10,
        )

        if lines is None:
            return image, 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < max_angle:
                angles.append(angle)

        if not angles:
            return image, 0.0

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return image, 0.0

        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        bg = int(np.median(img_np))
        rotated = cv2.warpAffine(
            img_np, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderValue=(bg, bg, bg),
        )
        logger.info(f"[Preprocessor] Deskewed by {median_angle:.2f}°")
        return Image.fromarray(rotated), median_angle

    def preprocess(self, image: Image.Image):
        """
        Full preprocessing pipeline.

        Returns:
            (preprocessed_image, deskew_angle)
        """
        enhanced = self.enhance_clahe(image)
        deskewed, angle = self.deskew(enhanced)
        return deskewed, angle
