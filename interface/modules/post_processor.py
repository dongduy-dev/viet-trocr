# =============================================================================
# post_processor.py — Step 5: Language Correction & Post-processing
# RegexSanitizer + AddressCorrector + PhoBERT (optional)
# =============================================================================

import logging
import os
import sys
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Locate the Fine-tuning code directory for imports ───────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SEARCH_PATHS = [
    os.path.join(_THIS_DIR, "..", ".."),                           # Colab-style
    os.path.join(_THIS_DIR, "..", "..", "Fine tuning", "code"),    # Local dev
]
for _p in _SEARCH_PATHS:
    _p = os.path.abspath(_p)
    if os.path.isfile(os.path.join(_p, "eval_utils.py")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break


class PostProcessor:
    """
    Step 5: Language-aware post-processing.

    Components (loaded on demand):
      - RegexSanitizer: Deterministic hallucination collapse
      - AddressCorrector: Geographic name fuzzy-matching
      - PhoBERTCorrector: MLM rescoring (optional, off by default)
    """

    def __init__(
        self,
        enable_phobert: bool = False,
        enable_address: bool = True,
        gazetteer_path: Optional[str] = None,
        phobert_model: str = "vinai/phobert-base",
        phobert_threshold: float = 0.85,
    ):
        self.enable_phobert = enable_phobert
        self.enable_address = enable_address

        # RegexSanitizer (always on)
        self._sanitizer = None
        self._init_sanitizer()

        # AddressCorrector
        self._addr_corrector = None
        if enable_address:
            self._init_address_corrector(gazetteer_path)

        # PhoBERTCorrector
        self._phobert = None
        if enable_phobert:
            self._init_phobert(phobert_model, phobert_threshold)

    def _init_sanitizer(self):
        try:
            from eval_utils import RegexSanitizer
            self._sanitizer = RegexSanitizer()
            logger.info("[PostProcessor] RegexSanitizer loaded")
        except ImportError:
            logger.warning("[PostProcessor] eval_utils not found — RegexSanitizer disabled")

    def _init_address_corrector(self, gazetteer_path: Optional[str]):
        try:
            from address_corrector import AddressCorrector

            if gazetteer_path is None:
                # Search common locations
                candidates = [
                    os.path.join(_THIS_DIR, "..", "..", "data", "vietnam_gazetteer.json"),
                    os.path.join(_THIS_DIR, "..", "..", "Fine tuning", "code", "data", "vietnam_gazetteer.json"),
                ]
                for c in candidates:
                    if os.path.isfile(c):
                        gazetteer_path = os.path.abspath(c)
                        break

            if gazetteer_path and os.path.isfile(gazetteer_path):
                self._addr_corrector = AddressCorrector(gazetteer_path)
                logger.info(f"[PostProcessor] AddressCorrector loaded from {gazetteer_path}")
            else:
                logger.warning("[PostProcessor] Gazetteer not found — AddressCorrector disabled")
        except ImportError:
            logger.warning("[PostProcessor] address_corrector not found — disabled")

    def _init_phobert(self, model_name: str, threshold: float):
        try:
            from eval_utils import PhoBERTCorrector
            self._phobert = PhoBERTCorrector(
                model_name=model_name,
                confidence_threshold=threshold,
            )
            if self._phobert.is_loaded:
                logger.info("[PostProcessor] PhoBERTCorrector loaded")
            else:
                logger.warning("[PostProcessor] PhoBERT failed to load")
                self._phobert = None
        except ImportError:
            logger.warning("[PostProcessor] eval_utils not found — PhoBERT disabled")

    def process(self, text: str) -> Dict[str, str]:
        """
        Run full post-processing pipeline.

        Returns dict with keys:
            raw, sanitized, corrected, phobert_corrected
        """
        result = {
            "raw": text,
            "sanitized": text,
            "corrected": text,
            "phobert_corrected": text,
        }

        # Step 5a: Sanitize
        if self._sanitizer:
            result["sanitized"] = self._sanitizer.sanitize(text)

        # Step 5b: Address correction (on sanitized text)
        corrected = result["sanitized"]
        if self._addr_corrector:
            corrected = self._addr_corrector.correct(corrected)
            corrected = self._addr_corrector.correct_trailing_province(corrected)
        result["corrected"] = corrected

        # Step 5c: PhoBERT path (on sanitized text, independent)
        phobert_out = result["sanitized"]
        if self._phobert and self._phobert.is_loaded:
            phobert_out = self._phobert.correct(phobert_out)
            if self._addr_corrector:
                phobert_out = self._addr_corrector.correct(phobert_out)
                phobert_out = self._addr_corrector.correct_trailing_province(phobert_out)
        result["phobert_corrected"] = phobert_out

        return result
