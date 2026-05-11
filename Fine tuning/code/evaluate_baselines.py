#!/usr/bin/env python3
"""
evaluate_baselines.py — Multi-tier Baseline Benchmarking for Vietnamese OCR

Evaluates two baseline models against the exact same 4,477 test images
used in the TrOCR evaluation, with identical CER/WER computation logic.

Baseline 1 (Tier 2): VietOCR — VGG + Transformer (pbcquoc/vietocr)
Baseline 2 (Tier 1): CRNN + CTC (custom PyTorch implementation)

Usage (Colab):
    python evaluate_baselines.py \
        --test_printed /content/lmdb/line_printed/test \
        --test_handwritten /content/lmdb/line_handwritten/test \
        --output_dir /content/drive/MyDrive/OCR/checkpoints/baseline_eval \
        --batch_size 32

Fair Comparison Guarantees:
    ✓ Same LMDB test data (3,739 printed + 738 handwritten = 4,477)
    ✓ Same jiwer CER/WER functions (compute_sample_cer / compute_sample_wer)
    ✓ Same Unicode NFC normalization on labels
    ✓ Same error categorization (ErrorCategorizer)
    ✓ Hardware profiling: FPS + parameter count per model
"""

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Import shared evaluation utilities from the TrOCR codebase ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_utils import (
    RegexSanitizer,
    ErrorCategorizer,
    compute_sample_cer,
    compute_sample_wer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sanitizer = RegexSanitizer()


# =============================================================================
# 1. Lightweight LMDB Reader (no TrOCR-specific resize)
# =============================================================================

class RawLMDBReader:
    """
    Read raw PIL images + NFC-normalized labels from LMDB.
    No model-specific preprocessing — each baseline applies its own.
    """

    def __init__(self, lmdb_path: str, max_samples: Optional[int] = None):
        import lmdb as _lmdb

        self.env = _lmdb.open(
            lmdb_path, max_readers=8, readonly=True,
            lock=False, readahead=False, meminit=False,
        )

        with self.env.begin(write=False) as txn:
            raw = txn.get(b"num-samples")
            if raw is None:
                raise ValueError(f"Missing 'num-samples' in {lmdb_path}")
            total = int(raw.decode("utf-8"))

            # Auto-detect key format (mirrors dataset.py logic)
            self.key_fmt = "image-{:09d}"
            self.lbl_fmt = "label-{:09d}"
            self.start_idx = 1

            if txn.get(b"image-000000001") is not None:
                pass  # default 9-digit, 1-indexed
            elif txn.get(b"image-00000001") is not None:
                self.key_fmt = "image-{:08d}"
                self.lbl_fmt = "label-{:08d}"
            elif txn.get(b"image-000000000") is not None:
                self.start_idx = 0
            elif txn.get(b"image-00000000") is not None:
                self.key_fmt = "image-{:08d}"
                self.lbl_fmt = "label-{:08d}"
                self.start_idx = 0

        self.indices = list(range(self.start_idx, total + self.start_idx))
        if max_samples and max_samples < len(self.indices):
            import random
            random.seed(42)
            self.indices = random.sample(self.indices, max_samples)

        logger.info(f"  RawLMDBReader: {lmdb_path} → {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos: int) -> Tuple[Image.Image, str, int]:
        idx = self.indices[pos]
        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(self.key_fmt.format(idx).encode())
            lbl_bytes = txn.get(self.lbl_fmt.format(idx).encode())

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        label = unicodedata.normalize("NFC", lbl_bytes.decode("utf-8")) if lbl_bytes else ""
        return image, label, idx


# =============================================================================
# 2. VietOCR Baseline (VGG + Transformer)
# =============================================================================

class VietOCRBaseline:
    """
    Baseline using the official vietocr package (VGG-Transformer).
    Pre-trained on Vietnamese text — no fine-tuning on our data.
    """

    NAME = "VietOCR (VGG-Transformer)"

    def __init__(self, device: str = "cuda"):
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        config = Cfg.load_config_from_name("vgg_transformer")
        config["device"] = device
        # Use the default pre-trained weights (auto-downloaded)
        config["cnn"]["pretrained"] = True

        self.predictor = Predictor(config)
        self.device = device

        # Count parameters
        model = self.predictor.model
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        vram_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / 1e6

        logger.info(
            f"[VietOCR] Loaded | params={self.total_params:,} | "
            f"VRAM≈{vram_mb:.0f}MB"
        )

    def predict(self, image: Image.Image) -> str:
        """
        Run single-image inference. VietOCR handles its own preprocessing
        (resize to height=32, preserve aspect ratio).
        """
        text = self.predictor.predict(image)
        return unicodedata.normalize("NFC", text.strip())

    def predict_batch(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> List[str]:
        """
        Batch prediction. VietOCR's predictor.predict() is single-image,
        so we call it in a loop. Timing is measured externally.
        """
        results = []
        for img in images:
            results.append(self.predict(img))
        return results


# =============================================================================
# 3. CRNN + CTC Baseline
# =============================================================================

class CRNNBaseline:
    """
    Traditional CRNN + CTC baseline for Vietnamese OCR.

    Architecture: CNN (VGG-like) → BiLSTM → CTC decoder
    Uses the same vietocr framework but with 'vgg_seq2seq' config,
    which implements the classic CRNN+Attention (closest available
    pre-trained Vietnamese CRNN-family model).

    If vgg_seq2seq is unavailable, falls back to a custom lightweight
    CRNN with CTC decoding.
    """

    NAME = "CRNN + CTC (Seq2Seq)"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._use_vietocr = False

        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg

            # vgg_seq2seq is the CRNN-family model in vietocr
            config = Cfg.load_config_from_name("vgg_seq2seq")
            config["device"] = device
            config["cnn"]["pretrained"] = True

            self.predictor = Predictor(config)
            self._use_vietocr = True

            model = self.predictor.model
            self.total_params = sum(p.numel() for p in model.parameters())
            self.trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            vram_mb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / 1e6

            self.NAME = "CRNN-Seq2Seq (vgg_seq2seq)"
            logger.info(
                f"[CRNN] Loaded vgg_seq2seq | params={self.total_params:,} | "
                f"VRAM≈{vram_mb:.0f}MB"
            )

        except Exception as e:
            logger.warning(f"[CRNN] vgg_seq2seq failed ({e}). Using custom CRNN+CTC.")
            self._init_custom_crnn()

    def _init_custom_crnn(self):
        """
        Build a lightweight CRNN+CTC from scratch (no pre-trained weights).
        This establishes the architectural lower-bound baseline.
        """
        import torch.nn as nn

        # Vietnamese character set (simplified — ASCII + common Vietnamese)
        self.charset = list(
            " 0123456789abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
            "ìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
            "ùúủũụưứừửữựỳýỷỹỵđ"
            "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆ"
            "ÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
            "ÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
            ".,;:!?\"'()-/\\@#$%&*+=[]{}|<>~`^_"
        )
        self.char2idx = {c: i + 1 for i, c in enumerate(self.charset)}  # 0=blank
        self.idx2char = {i + 1: c for i, c in enumerate(self.charset)}
        num_classes = len(self.charset) + 1  # +1 for CTC blank

        class SimpleCRNN(nn.Module):
            def __init__(self, num_classes, hidden=256):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
                    nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
                    nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(),
                )
                self.rnn = nn.LSTM(512, hidden, num_layers=2,
                                   bidirectional=True, batch_first=False)
                self.fc = nn.Linear(hidden * 2, num_classes)

            def forward(self, x):
                conv = self.cnn(x)  # (B, 512, 1, W')
                conv = conv.squeeze(2).permute(2, 0, 1)  # (W', B, 512)
                rnn_out, _ = self.rnn(conv)
                output = self.fc(rnn_out)  # (W', B, num_classes)
                return output

        self.model = SimpleCRNN(num_classes).to(self.device).eval()
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = self.total_params
        self.NAME = "CRNN+CTC (untrained, lower-bound)"

        logger.info(
            f"[CRNN] Custom CRNN+CTC (NO pre-trained weights) | "
            f"params={self.total_params:,} | Lower-bound baseline"
        )

    def _preprocess_crnn(self, image: Image.Image) -> torch.Tensor:
        """Resize to height=32, preserve aspect ratio, grayscale, normalize."""
        img = image.convert("L")
        w, h = img.size
        new_h = 32
        new_w = max(1, int(w * new_h / max(h, 1)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)  # (1,1,32,W)
        return tensor.to(self.device)

    def _ctc_decode(self, output: torch.Tensor) -> str:
        """Greedy CTC decoding."""
        _, preds = output.max(2)  # (W', B)
        preds = preds.squeeze(1).cpu().numpy()  # (W',)

        chars = []
        prev = 0  # blank
        for p in preds:
            if p != 0 and p != prev:
                ch = self.idx2char.get(p, "")
                chars.append(ch)
            prev = p
        return "".join(chars)

    def predict(self, image: Image.Image) -> str:
        if self._use_vietocr:
            text = self.predictor.predict(image)
            return unicodedata.normalize("NFC", text.strip())

        # Custom CRNN path
        with torch.no_grad():
            inp = self._preprocess_crnn(image)
            output = self.model(inp)  # (W', 1, num_classes)
            text = self._ctc_decode(output)
        return unicodedata.normalize("NFC", text.strip())

    def predict_batch(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> List[str]:
        results = []
        for img in images:
            results.append(self.predict(img))
        return results


# =============================================================================
# 4. Evaluation Engine (shared logic)
# =============================================================================

def evaluate_model(
    model,
    reader: RawLMDBReader,
    domain: str,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Evaluate a baseline model on a single domain (printed or handwritten).
    Returns per-sample results with CER/WER metrics.
    """
    results = []
    total_samples = len(reader)

    logger.info(f"  Evaluating {model.NAME} on {domain} ({total_samples} samples)...")

    # Warm-up: 3 samples to stabilize GPU timings
    for i in range(min(3, total_samples)):
        img, _, _ = reader[i]
        _ = model.predict(img)

    # Timed inference
    start_time = time.time()

    for i in tqdm(range(total_samples), desc=f"  {model.NAME} [{domain}]"):
        img, ref, idx = reader[i]

        t0 = time.perf_counter()
        pred = model.predict(img)
        inference_ms = (time.perf_counter() - t0) * 1000

        # Apply sanitizer (same as TrOCR pipeline)
        sanitized = sanitizer.sanitize(pred)

        # Compute metrics on sanitized output
        cer = compute_sample_cer(ref, sanitized)
        wer = compute_sample_wer(ref, sanitized)

        # Error categorization (same logic as TrOCR)
        category = ErrorCategorizer.categorize(ref, sanitized, cer)

        results.append({
            "idx": idx,
            "domain": domain,
            "ref": ref,
            "raw_pred": pred,
            "sanitized_pred": sanitized,
            "cer": cer,
            "wer": wer,
            "error_category": category,
            "inference_ms": inference_ms,
        })

    total_time = time.time() - start_time
    fps = total_samples / total_time if total_time > 0 else 0

    avg_cer = np.mean([r["cer"] for r in results])
    avg_wer = np.mean([r["wer"] for r in results])

    logger.info(
        f"  ✓ {model.NAME} [{domain}]: CER={avg_cer:.4f} | WER={avg_wer:.4f} | "
        f"FPS={fps:.1f} | {total_samples} samples in {total_time:.1f}s"
    )

    return results


def aggregate_metrics(results: List[Dict], model_name: str) -> Dict:
    """Compute aggregate metrics matching TrOCR evaluation format."""
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    metrics = {
        "model_name": model_name,
        "per_domain": {},
    }

    all_cer = []
    all_wer = []
    total_samples = 0

    for domain, samples in sorted(by_domain.items()):
        cers = [s["cer"] for s in samples]
        wers = [s["wer"] for s in samples]
        infer_ms = [s["inference_ms"] for s in samples]

        categories = defaultdict(int)
        for s in samples:
            categories[s["error_category"]] += 1

        n = len(samples)
        avg_ms = np.mean(infer_ms)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0

        metrics["per_domain"][domain] = {
            "cer": float(np.mean(cers)),
            "wer": float(np.mean(wers)),
            "samples": n,
            "fps": round(fps, 1),
            "avg_inference_ms": round(avg_ms, 2),
            "categories": dict(categories),
            "perfect_count": categories.get("PERFECT", 0),
            "perfect_pct": round(100 * categories.get("PERFECT", 0) / max(n, 1), 1),
        }

        all_cer.extend(cers)
        all_wer.extend(wers)
        total_samples += n

    metrics["overall"] = {
        "cer": float(np.mean(all_cer)),
        "wer": float(np.mean(all_wer)),
        "total_samples": total_samples,
    }

    return metrics


# =============================================================================
# 5. Output Writers
# =============================================================================

def write_csv(results: List[Dict], filepath: str):
    """Write per-sample results to CSV."""
    if not results:
        return
    fieldnames = [
        "idx", "domain", "ref", "raw_pred", "sanitized_pred",
        "cer", "wer", "error_category", "inference_ms",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"  Saved: {filepath}")


def write_summary(
    all_metrics: Dict,
    param_info: Dict,
    output_dir: str,
):
    """Write combined JSON summary with all baselines + TrOCR reference."""
    summary = {
        "description": "Multi-tier Baseline Benchmarking — Vietnamese OCR",
        "test_data": {
            "printed_samples": 3739,
            "handwritten_samples": 738,
            "total": 4477,
        },
        "trocr_reference": {
            "model_name": "TrOCR (ViT + Decoder) — Fine-tuned",
            "overall": {"cer": 0.0286, "wer": 0.0792, "total_samples": 4477},
            "per_domain": {
                "printed": {"cer": 0.0117, "wer": 0.0396, "perfect_pct": 68.5},
                "handwritten": {"cer": 0.1057, "wer": 0.2628, "perfect_pct": 14.8},
            },
            "total_params": "~337M (ViT-base + RoBERTa decoder)",
        },
        "baselines": {},
    }

    for model_key, metrics in all_metrics.items():
        entry = metrics.copy()
        entry["parameter_count"] = param_info.get(model_key, {})
        summary["baselines"][model_key] = entry

    filepath = os.path.join(output_dir, "baseline_metrics_summary.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved: {filepath}")


def print_comparison_table(all_metrics: Dict, param_info: Dict):
    """Print a formatted comparison table to console."""
    print("\n" + "=" * 90)
    print("MULTI-TIER BASELINE COMPARISON — Vietnamese OCR")
    print("=" * 90)

    header = f"{'Model':<35} {'CER':>8} {'WER':>8} {'Pr.CER':>8} {'HW.CER':>8} {'Pr.PERF':>8} {'Params':>12}"
    print(header)
    print("-" * 90)

    # TrOCR reference
    print(
        f"{'TrOCR (ours, fine-tuned)':<35} "
        f"{'0.0286':>8} {'0.0792':>8} {'0.0117':>8} {'0.1057':>8} "
        f"{'68.5%':>8} {'~337M':>12}"
    )
    print("-" * 90)

    for model_key, metrics in all_metrics.items():
        ov = metrics["overall"]
        pr = metrics["per_domain"].get("printed", {})
        hw = metrics["per_domain"].get("handwritten", {})
        params = param_info.get(model_key, {}).get("total", 0)
        param_str = f"{params / 1e6:.1f}M" if params > 0 else "N/A"

        print(
            f"{metrics['model_name']:<35} "
            f"{ov['cer']:>8.4f} {ov['wer']:>8.4f} "
            f"{pr.get('cer', 0):>8.4f} {hw.get('cer', 0):>8.4f} "
            f"{pr.get('perfect_pct', 0):>7.1f}% "
            f"{param_str:>12}"
        )

    print("=" * 90)
    print()


# =============================================================================
# 6. Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Baseline Benchmarking for Vietnamese OCR")
    p.add_argument("--test_printed", required=True, help="LMDB path: line_printed/test")
    p.add_argument("--test_handwritten", required=True, help="LMDB path: line_handwritten/test")
    p.add_argument("--output_dir", default=None, help="Output directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=None, help="Limit samples per domain")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_vietocr", action="store_true", help="Skip VietOCR evaluation")
    p.add_argument("--skip_crnn", action="store_true", help="Skip CRNN evaluation")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.test_printed), "..", "..", "baseline_eval"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Multi-tier Baseline Benchmarking — Vietnamese OCR")
    logger.info("=" * 60)

    # ── Load test data ──────────────────────────────────────────────────────
    logger.info("\n[Data] Loading LMDB test sets...")
    reader_printed = RawLMDBReader(args.test_printed, args.max_samples)
    reader_hw = RawLMDBReader(args.test_handwritten, args.max_samples)
    logger.info(
        f"[Data] Total: {len(reader_printed)} printed + "
        f"{len(reader_hw)} handwritten = {len(reader_printed) + len(reader_hw)}"
    )

    all_metrics = {}
    param_info = {}

    # ── Baseline 1: VietOCR ─────────────────────────────────────────────────
    if not args.skip_vietocr:
        logger.info("\n" + "─" * 60)
        logger.info("[Baseline 1] VietOCR (VGG + Transformer)")
        logger.info("─" * 60)

        try:
            vietocr = VietOCRBaseline(device=args.device)

            results_pr = evaluate_model(vietocr, reader_printed, "printed", args.batch_size)
            results_hw = evaluate_model(vietocr, reader_hw, "handwritten", args.batch_size)
            all_results = results_pr + results_hw

            metrics = aggregate_metrics(all_results, vietocr.NAME)
            all_metrics["vietocr"] = metrics
            param_info["vietocr"] = {
                "total": vietocr.total_params,
                "trainable": vietocr.trainable_params,
            }

            write_csv(all_results, os.path.join(args.output_dir, "vietocr_predictions.csv"))

        except Exception as e:
            logger.error(f"[VietOCR] Failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Baseline 2: CRNN + CTC ──────────────────────────────────────────────
    if not args.skip_crnn:
        logger.info("\n" + "─" * 60)
        logger.info("[Baseline 2] CRNN + CTC")
        logger.info("─" * 60)

        try:
            crnn = CRNNBaseline(device=args.device)

            results_pr = evaluate_model(crnn, reader_printed, "printed", args.batch_size)
            results_hw = evaluate_model(crnn, reader_hw, "handwritten", args.batch_size)
            all_results = results_pr + results_hw

            metrics = aggregate_metrics(all_results, crnn.NAME)
            all_metrics["crnn"] = metrics
            param_info["crnn"] = {
                "total": crnn.total_params,
                "trainable": crnn.trainable_params,
            }

            write_csv(all_results, os.path.join(args.output_dir, "crnn_predictions.csv"))

        except Exception as e:
            logger.error(f"[CRNN] Failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Combined summary ────────────────────────────────────────────────────
    if all_metrics:
        write_summary(all_metrics, param_info, args.output_dir)
        print_comparison_table(all_metrics, param_info)

    logger.info(f"\n[Done] All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
