# =============================================================================
# text_recognizer.py — Step 4: TrOCR Text Recognition (batched)
# =============================================================================

import logging
import unicodedata
from typing import List

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ── Inline: resize_for_vit (from data/dataset.py) ──────────────────────────
def resize_for_vit(
    image: Image.Image,
    target_h: int = 128,
    target_w: int = 1536,
    bg_color=None,
) -> Image.Image:
    """Scale-to-fit resize for ViT input. Never crops."""
    img_gray = np.array(image.convert("L"), dtype=np.uint8)
    fill = int(bg_color) if bg_color is not None else int(np.median(img_gray))

    w, h = image.size
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    image = image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), (fill, fill, fill))
    canvas.paste(image.convert("RGB"), (0, 0))
    return canvas


# ── Inline: safe_batch_decode (from eval_utils.py) ─────────────────────────
def safe_batch_decode(tokenizer, generated_ids: torch.Tensor) -> List[str]:
    """Robust decoding that handles Vietnamese AddedToken edge cases."""
    results = []
    for seq in generated_ids:
        token_ids = seq.tolist()
        # Remove special tokens
        special = set()
        for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special.add(tid)
        clean_ids = [t for t in token_ids if t not in special]

        pieces = []
        for tid in clean_ids:
            tok = tokenizer.convert_ids_to_tokens(tid)
            if tok is None:
                continue
            if isinstance(tok, list):
                tok = tok[0] if tok else ""
            tok = str(tok)
            # RoBERTa space marker
            tok = tok.replace("\u0120", " ")
            # Newline marker
            tok = tok.replace("\u010a", " ")
            pieces.append(tok)

        text = "".join(pieces).strip()
        text = unicodedata.normalize("NFC", text)
        results.append(text)
    return results


class TextRecognizer:
    """
    Step 4: TrOCR-based text recognition with batched inference.

    Loads the fine-tuned Vietnamese TrOCR model and runs batched
    inference on cropped line images.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        image_height: int = 128,
        image_width: int = 1536,
    ):
        self.device = torch.device(device)
        self.image_height = image_height
        self.image_width = image_width

        self.model, self.processor = self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load TrOCR model with all constraint fixes."""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info(f"[TextRecognizer] Loading from {model_path}")
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)

        # FP16 + eval
        model = model.half().to(self.device).eval()

        # Fix meta-device tensors (sinusoidal positional embeddings)
        for module in model.modules():
            if module.__class__.__name__ == "TrOCRSinusoidalPositionalEmbedding":
                if hasattr(module, "weights") and isinstance(module.weights, torch.Tensor):
                    if module.weights.device.type == "meta" or module.weights.device != self.device:
                        n_pos, d = module.weights.shape
                        pad_idx = getattr(module, "padding_idx", None)
                        module.weights = module.get_embedding(n_pos, d, pad_idx).to(self.device)
                if hasattr(module, "_float_tensor") and isinstance(module._float_tensor, torch.Tensor):
                    if module._float_tensor.device.type == "meta" or module._float_tensor.device != self.device:
                        module._float_tensor = torch.zeros(1, device=self.device)

        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.use_cache = True

        vram_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        logger.info(f"[TextRecognizer] Loaded — VRAM ≈ {vram_mb:.0f}MB")

        return model, processor

    @torch.no_grad()
    def recognize_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 8,
        repetition_penalty: float = 1.2,
        max_length: int = 256,
    ) -> List[str]:
        """
        Recognize text from a list of line images.

        Images are resized via resize_for_vit, then batched for inference.
        """
        if not images:
            return []

        # Preprocess all images
        resized = [
            resize_for_vit(img, self.image_height, self.image_width)
            for img in images
        ]

        all_texts = []

        for i in range(0, len(resized), batch_size):
            batch = resized[i : i + batch_size]

            pixel_values = self.processor(
                images=batch, return_tensors="pt"
            ).pixel_values.to(self.device, dtype=torch.float16)

            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=1,
                    do_sample=False,
                    repetition_penalty=repetition_penalty,
                )

            texts = safe_batch_decode(self.processor.tokenizer, generated_ids)
            all_texts.extend(texts)

        logger.info(
            f"[TextRecognizer] Recognized {len(all_texts)} lines "
            f"in {(len(images) + batch_size - 1) // batch_size} batches"
        )
        return all_texts
