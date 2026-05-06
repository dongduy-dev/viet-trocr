# =============================================================================
# eval_utils.py
# Core utilities for Evaluation & Error Analysis Pipeline
# RegexSanitizer, PhoBERTCorrector, ErrorCategorizer, safe_batch_decode
# =============================================================================

import re
import logging
import unicodedata
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# 1. RegexSanitizer — Deterministic Post-generation Cleanup
# =============================================================================

class RegexSanitizer:
    """
    Deterministic post-processing to collapse hallucinated repetitions
    from TrOCR greedy decoding.

    Rules (applied sequentially):
      1. Collapse 4+ consecutive identical characters → max 3
         (Vietnamese legitimately has "..." ellipsis = 3 dots)
      2. Collapse 3+ consecutive identical tokens (word-level) → max 2
         (e.g., "12 12 12 12" → "12 12")
      3. Collapse 3+ consecutive identical punctuation groups
         (e.g., "- - - - -" → "- -")
      4. Strip trailing incomplete repetitions at end of string
         (e.g., "text 12 - - -" → "text 12")
      5. Normalize excessive whitespace

    WARNING: This is a LOSSY operation. Always preserve raw_pred alongside
    sanitized_pred in outputs so no information is lost for debugging.
    """

    # 4+ identical characters → collapse to 3
    _RE_CHAR_REPEAT = re.compile(r"(.)\1{3,}")

    # 3+ identical whitespace-separated tokens → collapse to 2
    _RE_WORD_REPEAT = re.compile(r"\b(\S+)(?:\s+\1){2,}\b")

    # 3+ identical punctuation groups separated by spaces
    _RE_PUNCT_REPEAT = re.compile(r"([^\w\s]+)(?:\s+\1){2,}")

    # Trailing repeated short tokens (1-4 chars) at end of string
    _RE_TRAILING_REPEAT = re.compile(
        r"(\s+\S{1,4})\1{2,}\s*$"
    )

    # Multiple spaces → single space
    _RE_MULTI_SPACE = re.compile(r"\s{2,}")

    def sanitize(self, text: str) -> str:
        """Apply all sanitization rules sequentially."""
        if not text or not text.strip():
            return text

        result = text

        # Rule 1: Collapse character-level repeats (4+ → 3)
        result = self._RE_CHAR_REPEAT.sub(r"\1\1\1", result)

        # Rule 2: Collapse word-level repeats (3+ → 2)
        result = self._RE_WORD_REPEAT.sub(r"\1 \1", result)

        # Rule 3: Collapse punctuation group repeats
        result = self._RE_PUNCT_REPEAT.sub(r"\1 \1", result)

        # Rule 4: Strip trailing repetitions
        result = self._RE_TRAILING_REPEAT.sub("", result)

        # Rule 5: Normalize whitespace
        result = self._RE_MULTI_SPACE.sub(" ", result).strip()

        return result

    def has_hallucination_pattern(self, text: str) -> bool:
        """Check if text contains hallucination-like repetition patterns."""
        if not text:
            return False
        if self._RE_CHAR_REPEAT.search(text):
            return True
        if self._RE_WORD_REPEAT.search(text):
            return True
        if self._RE_PUNCT_REPEAT.search(text):
            return True
        return False


# =============================================================================
# 2. PhoBERTCorrector — Real MLM-based Language Correction (Step 5)
# =============================================================================

class PhoBERTCorrector:
    """
    PhoBERT-based language correction using Masked Language Model rescoring.

    Strategy:
      1. Word-segment the sanitized TrOCR prediction using pyvi.
      2. For each word, mask it and query PhoBERT MLM for top-k candidates.
      3. If the top-1 candidate has significantly higher probability than
         the original token AND the edit distance is ≤ 2, accept correction.
      4. De-segment (remove underscores) to produce final output.

    Requirements:
      - pip install pyvi transformers
      - Model: vinai/phobert-base (~540MB)

    VRAM: ~500MB in FP16. Loaded on same device as TrOCR.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        device: torch.device = None,
        confidence_threshold: float = 0.85,
        max_edit_distance: int = 2,
        top_k: int = 5,
    ):
        self.device = device or torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.max_edit_distance = max_edit_distance
        self.top_k = top_k
        self._model = None
        self._tokenizer = None
        self._segmenter = None
        self.model_name = model_name

    def load(self):
        """Lazy-load PhoBERT model, tokenizer, and Vietnamese word segmenter."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        logger.info(f"[PhoBERT] Loading {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self._model = self._model.half().to(self.device).eval()

        # Load pyvi for Vietnamese word segmentation
        try:
            from pyvi import ViTokenizer as ViSeg
            self._segmenter = ViSeg
            logger.info("[PhoBERT] pyvi ViTokenizer loaded for word segmentation.")
        except ImportError:
            logger.warning(
                "[PhoBERT] pyvi not installed. Word segmentation disabled. "
                "Install with: pip install pyvi"
            )
            self._segmenter = None

        param_count = sum(p.numel() for p in self._model.parameters())
        vram_mb = sum(
            p.numel() * p.element_size() for p in self._model.parameters()
        ) / 1e6
        logger.info(
            f"[PhoBERT] Loaded | params={param_count:,} | "
            f"VRAM≈{vram_mb:.0f}MB (FP16)"
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _segment(self, text: str) -> str:
        """Word-segment Vietnamese text. Falls back to raw text if pyvi unavailable."""
        if self._segmenter is None:
            return text
        try:
            return self._segmenter.tokenize(text)
        except Exception:
            return text

    def _desegment(self, text: str) -> str:
        """Remove word segmentation underscores."""
        return text.replace("_", " ")

    @staticmethod
    def _should_skip_token(token: str) -> bool:
        """
        Determine if a token should be skipped for PhoBERT correction.

        Skip:
          - Pure punctuation/symbols: , . ( ) [ ] / - : ; ! ? " '
          - Pure numbers: 123, 45.67, 199/46
          - Short all-uppercase abbreviations: TP, TX, TT, UBND (≤ 5 chars)
          - Single characters
        """
        clean = token.replace("_", "")
        if not clean:
            return True
        # Pure punctuation/symbols
        if not any(c.isalnum() for c in clean):
            return True
        # Single characters
        if len(clean) <= 1:
            return True
        # Pure numbers (possibly with . , / separators)
        if all(c.isdigit() or c in '.,-/' for c in clean):
            return True
        # Short all-uppercase (abbreviations like TP, TX, UBND)
        alpha_chars = [c for c in clean if c.isalpha()]
        if alpha_chars and len(alpha_chars) <= 5 and all(c.isupper() for c in alpha_chars):
            return True
        return False

    @staticmethod
    def _reattach_punctuation(original: str, corrected: str) -> str:
        """
        Re-attach punctuation that PhoBERT's word segmentation separated.

        PhoBERT (via pyvi) inserts spaces around punctuation:
          "đây)."  → "đây ) ."  (BAD)
          "Marx-Lenin" → "Marx - Lenin" (BAD)

        This method restores the original punctuation spacing by comparing
        against the pre-correction input.

        Rules:
          1. Remove space BEFORE closing punctuation: ) ] } . , ; : ! ?
          2. Remove space AFTER opening punctuation: ( [ {
          3. Remove spaces around hyphens/slashes when original had none
          4. General: if original had "X,Y" (no space), collapse "X , Y" → "X,Y"
        """
        if not original or not corrected:
            return corrected

        # Pattern: space + punctuation that should attach to preceding word
        # e.g., "word ." → "word." , "word ," → "word," , "12 %" → "12%"
        result = corrected

        # Closing punctuation: remove preceding space
        # Includes % which should attach to numbers (12% not 12 %)
        result = re.sub(r'\s+([)\]}>,.;:!?%])', r'\1', result)

        # Opening punctuation: remove following space
        result = re.sub(r'([([{<])\s+', r'\1', result)

        # Quotes: Skip handling entirely.
        # Vietnamese uses standard " for both opening and closing, so we
        # cannot reliably distinguish them. Removing spaces would hurt
        # opening quotes (e.g., 'nhận định " nguy' should keep the space
        # before the opening quote). The original TrOCR output spacing
        # is more reliable for quotes.

        # Hyphens between words: check if original had no spaces
        # e.g., "Marx-Lenin" should not become "Marx - Lenin"
        if '-' in original and ' - ' not in original:
            result = re.sub(r'\s+-\s+', '-', result)

        # Slashes: check if original had no spaces around them
        if '/' in original and ' / ' not in original:
            result = re.sub(r'\s+/\s+', '/', result)

        # En/em dashes: check original
        for dash in ['–', '—']:
            if dash in original and f' {dash} ' not in original:
                result = re.sub(rf'\s+{re.escape(dash)}\s+', dash, result)

        # Normalize multiple spaces
        result = re.sub(r'\s{2,}', ' ', result).strip()

        return result

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return PhoBERTCorrector._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev_row[j + 1] + 1
                dele = curr_row[j] + 1
                sub = prev_row[j] + (c1 != c2)
                curr_row.append(min(ins, dele, sub))
            prev_row = curr_row
        return prev_row[-1]

    @torch.no_grad()
    def _score_masked_position(
        self, segmented_tokens: List[str], mask_idx: int
    ) -> List[Tuple[str, float]]:
        """
        Mask one token and return top-k candidates with probabilities.

        Args:
            segmented_tokens: Word-segmented token list.
            mask_idx: Index of the token to mask.

        Returns:
            List of (candidate_string, probability) tuples.
        """
        masked = segmented_tokens.copy()
        masked[mask_idx] = self._tokenizer.mask_token
        masked_text = " ".join(masked)

        inputs = self._tokenizer(
            masked_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        # Find mask token position in input_ids
        mask_positions = (
            inputs["input_ids"] == self._tokenizer.mask_token_id
        ).nonzero(as_tuple=True)

        if len(mask_positions[1]) == 0:
            return []

        mask_pos = mask_positions[1][0].item()

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self._model(**inputs).logits

        mask_logits = logits[0, mask_pos]
        probs = F.softmax(mask_logits.float(), dim=-1)

        top_k_probs, top_k_ids = probs.topk(self.top_k)
        candidates = []
        for prob, tok_id in zip(top_k_probs.tolist(), top_k_ids.tolist()):
            token_str = self._tokenizer.decode([tok_id]).strip()
            candidates.append((token_str, prob))

        return candidates

    def correct(self, text: str) -> str:
        """
        Apply PhoBERT MLM rescoring to correct OCR errors.

        Pipeline:
          1. NFC normalize input
          2. Word-segment with pyvi
          3. For each word (skipping punct/numbers/abbreviations),
             mask and get MLM candidates
          4. Accept correction if: top-1 prob > threshold AND edit_dist ≤ max
          5. De-segment and re-attach punctuation to match original spacing

        Args:
            text: Sanitized TrOCR prediction string.

        Returns:
            Corrected string with original punctuation spacing preserved.
        """
        if not self.is_loaded or not text or not text.strip():
            return text

        original_text = text  # Preserve for punctuation re-attachment
        text = unicodedata.normalize("NFC", text)
        segmented = self._segment(text)
        tokens = segmented.split()

        if len(tokens) == 0:
            return text

        # Skip correction for very short texts (< 2 words) — not enough context
        if len(tokens) < 2:
            return text

        corrected_tokens = tokens.copy()
        corrections_made = 0

        for i, original_token in enumerate(tokens):
            # Skip tokens that shouldn't be corrected
            if self._should_skip_token(original_token):
                continue

            # Skip very long tokens (likely correct compound words)
            clean = original_token.replace("_", "")
            if len(clean) > 20:
                continue

            candidates = self._score_masked_position(tokens, i)
            if not candidates:
                continue

            top_candidate, top_prob = candidates[0]

            # Check if correction is warranted
            if top_prob < self.confidence_threshold:
                continue

            # Compute edit distance between original and candidate
            orig_clean = original_token.replace("_", "")
            cand_clean = top_candidate.replace("_", "")
            edit_dist = self._edit_distance(orig_clean.lower(), cand_clean.lower())

            if edit_dist == 0:
                # Same token — no correction needed
                continue

            if edit_dist > self.max_edit_distance:
                # Too different — likely a different word, not a correction
                continue

            # Accept correction
            corrected_tokens[i] = top_candidate
            corrections_made += 1

        result = " ".join(corrected_tokens)
        result = self._desegment(result)

        # Re-attach punctuation to match original input's spacing
        result = self._reattach_punctuation(original_text, result)

        if corrections_made > 0:
            logger.debug(
                f"[PhoBERT] Corrected {corrections_made} tokens: "
                f"'{original_text}' → '{result}'"
            )

        return result

    def correct_batch(self, texts: List[str]) -> List[str]:
        """Correct a batch of texts. Sequential — PhoBERT MLM is per-sample."""
        return [self.correct(t) for t in texts]


# =============================================================================
# 3. ErrorCategorizer — Programmatic Error Tagging
# =============================================================================

class ErrorCategorizer:
    """
    Categorize each prediction error into a diagnostic bucket.

    Categories:
      HALLUCINATION_LOOP — 4+ identical chars or 3+ repeated words detected
      TRUNCATION         — prediction much shorter than reference
      INSERTION          — prediction much longer than reference
      SUBSTITUTION       — significant CER but no structural anomaly
      MINOR_ERROR        — low CER, minor character-level mistakes
      PERFECT            — CER = 0
    """

    _sanitizer = RegexSanitizer()

    @classmethod
    def categorize(cls, ref: str, pred: str, cer: float) -> str:
        if cer == 0.0:
            return "PERFECT"

        if cls._sanitizer.has_hallucination_pattern(pred):
            return "HALLUCINATION_LOOP"

        ref_len = max(len(ref), 1)
        pred_len = len(pred)

        if pred_len < 0.3 * ref_len:
            return "TRUNCATION"

        if pred_len > 2.0 * ref_len and ref_len > 3:
            return "INSERTION"

        if cer > 0.3:
            return "SUBSTITUTION"

        return "MINOR_ERROR"


# =============================================================================
# 4. safe_batch_decode — From trainer.py (copied for independence)
# =============================================================================

def safe_batch_decode(tokenizer, token_ids_tensor) -> List[str]:
    """
    Decode generated token IDs to strings, bypassing HuggingFace byte
    truncation issues with AddedTokens (Vietnamese diacritics).

    Mirrors the exact logic from core/trainer.py to ensure consistency
    between training evaluation and standalone evaluation.
    """
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        -100,
    }
    results = []
    for seq in token_ids_tensor:
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()

        clean_ids = [idx for idx in seq if idx not in special_ids]
        tokens = tokenizer.convert_ids_to_tokens(clean_ids)

        # RoBERTa uses 'Ġ' for space, 'Ċ' for newline
        text = "".join(tokens).replace("Ġ", " ").replace("Ċ", "\n")
        results.append(text.strip())

    return results


# =============================================================================
# 5. Per-sample CER/WER computation
# =============================================================================

def compute_sample_cer(ref: str, pred: str) -> float:
    """Compute CER for a single sample. Returns 0.0 for empty ref."""
    import jiwer
    safe_ref = ref if ref.strip() else " "
    safe_pred = pred if pred.strip() else " "
    try:
        return float(jiwer.cer(safe_ref, safe_pred))
    except Exception:
        return 1.0


def compute_sample_wer(ref: str, pred: str) -> float:
    """Compute WER for a single sample."""
    import jiwer
    safe_ref = ref if ref.strip() else " "
    safe_pred = pred if pred.strip() else " "
    try:
        return float(jiwer.wer(safe_ref, safe_pred))
    except Exception:
        return 1.0
