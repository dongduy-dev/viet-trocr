# =============================================================================
# text_detector.py — Step 2: Text Detection (DBNet via docTR)
# Uses docTR's db_resnet50 — same DBNet family as DBNet++
# Falls back to full-image mode if docTR is not installed
# =============================================================================

import logging
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class TextDetector:
    """
    Step 2: Text region detection using DBNet (db_resnet50) via docTR.

    docTR's db_resnet50 uses the same DBNet architecture family as DBNet++,
    with a ResNet-50 backbone. It provides accurate text detection for
    both printed and handwritten documents.

    If docTR is not installed, falls back to treating the entire
    image as a single text region (useful for pre-cropped inputs).
    """

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self._model = None
        self._available = False

        try:
            from doctr.models import detection_predictor
            import torch

            use_gpu = device == "cuda" and torch.cuda.is_available()
            self._model = detection_predictor(
                arch="db_resnet50",
                pretrained=True,
            )
            if use_gpu:
                self._model = self._model.cuda()

            self._available = True
            logger.info(f"[TextDetector] Loaded db_resnet50 (GPU={use_gpu})")
        except Exception as e:
            logger.warning(
                f"[TextDetector] docTR not available ({e}). "
                "Using full-image fallback mode."
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def detect(self, image: Image.Image) -> Tuple[list, list]:
        """
        Detect text regions in image.

        Returns:
            (polygons, scores) — polygons as [[x1,y1,x2,y2,...], ...],
            scores as list of float confidence values.
        """
        if not self._available:
            w, h = image.size
            polygon = [0, 0, w, 0, w, h, 0, h]
            return [polygon], [1.0]

        img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        h, w = img_np.shape[:2]

        result = self._model([img_np])

        # docTR 1.0.x returns list of dicts: [{"words": ndarray(N, 5)}]
        # Columns: rel_xmin, rel_ymin, rel_xmax, rel_ymax, confidence
        # Coordinates are relative (0-1 range)
        page = result[0]

        # Extract detections array from dict or use directly
        if isinstance(page, dict):
            detections = page.get("words", np.array([]))
        elif isinstance(page, np.ndarray):
            detections = page
        else:
            detections = np.array([])

        polygons = []
        scores = []

        if len(detections) > 0:
            for det in detections:
                rx1, ry1, rx2, ry2, conf = det[:5]

                if conf < 0.3:
                    continue

                # Convert relative → absolute pixel coordinates
                x1 = int(rx1 * w)
                y1 = int(ry1 * h)
                x2 = int(rx2 * w)
                y2 = int(ry2 * h)

                # Store as 4-point polygon for consistency
                polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
                polygons.append(polygon)
                scores.append(float(conf))

        # If no detections, fall back to full image
        if not polygons:
            polygon = [0, 0, w, 0, w, h, 0, h]
            polygons = [polygon]
            scores = [1.0]
            logger.info("[TextDetector] No regions found — using full image")
        else:
            logger.info(f"[TextDetector] Detected {len(polygons)} text regions")

        return polygons, scores

    @staticmethod
    def polygons_to_bboxes(polygons: list) -> List[Tuple[int, int, int, int]]:
        """Convert polygon coordinates to axis-aligned bounding boxes."""
        bboxes = []
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        return bboxes

    @staticmethod
    def draw_detections(
        image: Image.Image,
        bboxes: List[Tuple[int, int, int, int]],
        scores: Optional[List[float]] = None,
    ) -> Image.Image:
        """Draw bounding boxes on image for visualization."""
        img_draw = image.copy().convert("RGB")
        draw = ImageDraw.Draw(img_draw)

        colors = [
            "#00FF88", "#00CCFF", "#FF6B6B", "#FFD93D",
            "#6BCB77", "#4D96FF", "#FF6B9D", "#C0EEE4",
        ]

        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"#{i+1}"
            if scores and i < len(scores):
                label += f" ({scores[i]:.2f})"
            draw.text((x1 + 2, max(0, y1 - 14)), label, fill=color)

        return img_draw
