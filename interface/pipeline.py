# =============================================================================
# pipeline.py — E2E OCR Pipeline Orchestrator
# Chains: Preprocessing → Detection → Segmentation → Recognition → Correction
# =============================================================================

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from modules.preprocessor import Preprocessor
from modules.text_detector import TextDetector
from modules.line_segmenter import LineSegmenter
from modules.text_recognizer import TextRecognizer
from modules.post_processor import PostProcessor

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for all intermediate and final results."""

    # Step 1: Preprocessing
    original_image: Optional[Image.Image] = None
    preprocessed_image: Optional[Image.Image] = None
    deskew_angle: float = 0.0

    # Step 2: Detection
    polygons: list = field(default_factory=list)
    detection_scores: list = field(default_factory=list)
    bboxes: list = field(default_factory=list)
    annotated_image: Optional[Image.Image] = None

    # Step 3: Segmentation
    lines: list = field(default_factory=list)
    line_images: List[Image.Image] = field(default_factory=list)
    line_bboxes: list = field(default_factory=list)

    # Step 4: Recognition
    line_texts: List[str] = field(default_factory=list)
    full_text: str = ""

    # Step 5: Post-processing
    post_processed: Dict[str, str] = field(default_factory=dict)

    # Metadata
    timing: Dict[str, float] = field(default_factory=dict)
    num_regions: int = 0
    num_lines: int = 0


class OCRPipeline:
    """
    End-to-end Vietnamese OCR pipeline.

    Usage:
        pipeline = OCRPipeline(model_path="/path/to/final_model")
        result = pipeline.run(image)
        print(result.full_text)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 8,
        enable_phobert: bool = False,
        enable_address: bool = True,
        gazetteer_path: Optional[str] = None,
    ):
        self.batch_size = batch_size

        logger.info("=" * 60)
        logger.info("Initializing Vietnamese OCR Pipeline")
        logger.info("=" * 60)

        t0 = time.time()

        # Step 1: Preprocessor (lightweight, no GPU)
        self.preprocessor = Preprocessor()
        logger.info("[Init] ✓ Preprocessor")

        # Step 2: Text Detector (DBNet via docTR)
        self.detector = TextDetector(device=device)
        logger.info(f"[Init] ✓ TextDetector (available={self.detector.is_available})")

        # Step 3: Line Segmenter (no GPU)
        self.segmenter = LineSegmenter()
        logger.info("[Init] ✓ LineSegmenter")

        # Step 4: Text Recognizer (TrOCR)
        self.recognizer = TextRecognizer(model_path=model_path, device=device)
        logger.info("[Init] ✓ TextRecognizer")

        # Step 5: Post-processor
        self.post_processor = PostProcessor(
            enable_phobert=enable_phobert,
            enable_address=enable_address,
            gazetteer_path=gazetteer_path,
        )
        logger.info("[Init] ✓ PostProcessor")

        init_time = time.time() - t0
        logger.info(f"[Init] Pipeline ready in {init_time:.1f}s")

    def run(
        self,
        image: Image.Image,
        progress_fn: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Run the full 5-step OCR pipeline on an input image.

        Args:
            image: Input PIL image (document photo).
            progress_fn: Optional callback(step, total, message) for UI updates.

        Returns:
            PipelineResult with all intermediate and final outputs.
        """
        result = PipelineResult()
        result.original_image = image.copy()

        def _progress(step, msg):
            if progress_fn:
                progress_fn(step / 5, desc=msg)
            logger.info(f"[Pipeline] Step {step}/5: {msg}")

        # ── Step 1: Preprocessing ──────────────────────────────────────────
        _progress(1, "Preprocessing (CLAHE + Deskew)...")
        t = time.time()
        preprocessed, angle = self.preprocessor.preprocess(image)
        result.preprocessed_image = preprocessed
        result.deskew_angle = angle
        result.timing["step1_preprocess"] = time.time() - t

        # ── Step 2: Text Detection ─────────────────────────────────────────
        _progress(2, "Detecting text regions (DBNet++)...")
        t = time.time()
        polygons, scores = self.detector.detect(preprocessed)
        bboxes = TextDetector.polygons_to_bboxes(polygons)
        annotated = TextDetector.draw_detections(preprocessed, bboxes, scores)

        result.polygons = polygons
        result.detection_scores = scores
        result.bboxes = bboxes
        result.annotated_image = annotated
        result.num_regions = len(bboxes)
        result.timing["step2_detection"] = time.time() - t

        # ── Step 3: Line Segmentation ──────────────────────────────────────
        _progress(3, "Segmenting text lines (DBSCAN)...")
        t = time.time()
        lines = self.segmenter.segment(bboxes)
        line_images, line_bboxes = self.segmenter.crop_lines(preprocessed, lines)

        result.lines = lines
        result.line_images = line_images
        result.line_bboxes = line_bboxes
        result.num_lines = len(line_images)
        result.timing["step3_segmentation"] = time.time() - t

        if not line_images:
            _progress(5, "No text lines found.")
            result.full_text = ""
            return result

        # ── Step 4: Text Recognition ───────────────────────────────────────
        _progress(4, f"Recognizing {len(line_images)} lines (TrOCR)...")
        t = time.time()
        line_texts = self.recognizer.recognize_batch(
            line_images, batch_size=self.batch_size,
        )
        result.line_texts = line_texts
        result.full_text = "\n".join(line_texts)
        result.timing["step4_recognition"] = time.time() - t

        # ── Step 5: Post-processing ────────────────────────────────────────
        _progress(5, "Post-processing (Sanitize + Address Correction)...")
        t = time.time()
        post_results = self.post_processor.process(result.full_text)
        result.post_processed = post_results
        result.full_text = post_results.get("corrected", result.full_text)
        result.timing["step5_postprocess"] = time.time() - t

        total = sum(result.timing.values())
        result.timing["total"] = total
        logger.info(f"[Pipeline] Done — {result.num_lines} lines in {total:.2f}s")

        return result
