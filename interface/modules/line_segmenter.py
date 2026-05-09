# =============================================================================
# line_segmenter.py — Step 3: Line Segmentation (DBSCAN + reading order)
# =============================================================================

import logging
import numpy as np
from PIL import Image
from typing import List, Tuple

logger = logging.getLogger(__name__)


class LineSegmenter:
    """
    Step 3: Group detected text bboxes into reading-order lines.

    Algorithm:
      1. Cluster bboxes by y-center using DBSCAN (adaptive eps)
      2. Sort clusters top-to-bottom (reading order)
      3. Within each cluster, sort boxes left-to-right
      4. Merge boxes per line into single crop regions
    """

    def __init__(self, eps_ratio: float = 0.5, min_samples: int = 1, padding: int = 4):
        """
        Args:
            eps_ratio: DBSCAN eps = median_height × eps_ratio
            min_samples: Minimum boxes to form a cluster
            padding: Pixel padding when cropping line regions
        """
        self.eps_ratio = eps_ratio
        self.min_samples = min_samples
        self.padding = padding

    def segment(
        self, bboxes: List[Tuple[int, int, int, int]]
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Group bboxes into text lines via DBSCAN clustering.

        Args:
            bboxes: List of (x1, y1, x2, y2) bounding boxes.

        Returns:
            List of lines, each line is a list of bboxes sorted left-to-right.
            Lines are sorted top-to-bottom.
        """
        if not bboxes:
            return []

        if len(bboxes) == 1:
            return [list(bboxes)]

        from sklearn.cluster import DBSCAN

        y_centers = np.array([(b[1] + b[3]) / 2 for b in bboxes]).reshape(-1, 1)
        heights = np.array([b[3] - b[1] for b in bboxes])
        median_height = max(float(np.median(heights)), 10.0)

        eps = median_height * self.eps_ratio
        clustering = DBSCAN(eps=eps, min_samples=self.min_samples).fit(y_centers)
        labels = clustering.labels_

        # Group by cluster
        clusters = {}
        noise_counter = int(labels.max()) + 1 if len(labels) > 0 else 0
        for i, label in enumerate(labels):
            if label == -1:
                label = noise_counter
                noise_counter += 1
            clusters.setdefault(label, []).append(bboxes[i])

        # Sort lines top-to-bottom by mean y-center
        sorted_lines = sorted(
            clusters.values(),
            key=lambda boxes: np.mean([(b[1] + b[3]) / 2 for b in boxes]),
        )

        # Sort boxes within each line left-to-right
        for line in sorted_lines:
            line.sort(key=lambda b: b[0])

        logger.info(
            f"[LineSegmenter] Grouped {len(bboxes)} boxes into "
            f"{len(sorted_lines)} lines (eps={eps:.1f})"
        )
        return sorted_lines

    def merge_line_boxes(
        self, line_bboxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """Merge all bboxes in a line into one encompassing bbox."""
        x1 = min(b[0] for b in line_bboxes)
        y1 = min(b[1] for b in line_bboxes)
        x2 = max(b[2] for b in line_bboxes)
        y2 = max(b[3] for b in line_bboxes)
        return (x1, y1, x2, y2)

    def crop_lines(
        self,
        image: Image.Image,
        lines: List[List[Tuple[int, int, int, int]]],
        min_size: int = 8,
    ) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
        """
        Crop line regions from the original image.

        Returns:
            (line_images, merged_bboxes)
        """
        w_img, h_img = image.size
        crops = []
        merged = []

        for line_bboxes in lines:
            x1, y1, x2, y2 = self.merge_line_boxes(line_bboxes)

            # Apply padding
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w_img, x2 + self.padding)
            y2 = min(h_img, y2 + self.padding)

            if (y2 - y1) < min_size or (x2 - x1) < min_size:
                continue

            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
            merged.append((x1, y1, x2, y2))

        return crops, merged
