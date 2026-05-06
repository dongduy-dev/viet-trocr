#!/usr/bin/env python3
"""
evaluate_and_analyze.py — Vietnamese TrOCR Evaluation & Error Analysis Pipeline

Pipeline: Step 1 (resize_for_vit) → Step 4 (TrOCR) → Step 5 (PhoBERT)
Steps 2 (Detection) and 3 (Segmentation) are bypassed — test data is pre-cropped.

Usage (Colab):
    python evaluate_and_analyze.py \
        --model_path /content/drive/MyDrive/OCR/checkpoints/final_model \
        --test_printed /content/lmdb/line_printed/test \
        --test_handwritten /content/lmdb/line_handwritten/test \
        --output_dir /content/drive/MyDrive/OCR/checkpoints/evaluation \
        --batch_size 4 \
        --repetition_penalty 1.2
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent dir to path so we can import data.dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import resize_for_vit, LMDBDataset
from eval_utils import (
    RegexSanitizer,
    PhoBERTCorrector,
    ErrorCategorizer,
    safe_batch_decode,
    compute_sample_cer,
    compute_sample_wer,
)
from address_corrector import AddressCorrector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="TrOCR Vietnamese Evaluation Pipeline")
    p.add_argument("--model_path", required=True, help="Path to exported final_model/")
    p.add_argument("--test_printed", required=True, help="LMDB path for printed test set")
    p.add_argument("--test_handwritten", required=True, help="LMDB path for HW test set")
    p.add_argument("--output_dir", default=None, help="Output directory (default: model_path/../evaluation)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--image_height", type=int, default=128)
    p.add_argument("--image_width", type=int, default=1536)
    p.add_argument("--max_samples", type=int, default=None, help="Limit samples per domain (for testing)")
    p.add_argument("--worst_k", type=int, default=50, help="Number of worst cases to export")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_phobert", action="store_true", help="Skip PhoBERT correction")
    p.add_argument("--phobert_model", default="vinai/phobert-base")
    p.add_argument("--phobert_threshold", type=float, default=0.85)
    p.add_argument("--skip_address_correction", action="store_true", help="Skip gazetteer address correction")
    p.add_argument("--gazetteer_path", default=None, help="Path to vietnam_gazetteer.json")
    return p.parse_args()


# =============================================================================
# 2. Model Loading
# =============================================================================

def load_model_and_processor(model_path: str, device: torch.device):
    """Load exported TrOCR model with constraint verification."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    logger.info(f"[Model] Loading from {model_path}")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    # ── Constraint verification ──
    assert not processor.image_processor.do_resize, (
        "CRITICAL: processor.image_processor.do_resize must be False! "
        "The exported model expects external resize via resize_for_vit()."
    )
    logger.info("[Model] ✓ do_resize=False verified")

    eos_id = processor.tokenizer.eos_token_id
    assert model.config.decoder_start_token_id == eos_id, (
        f"decoder_start_token_id mismatch: {model.config.decoder_start_token_id} != {eos_id}"
    )
    logger.info("[Model] ✓ decoder_start_token_id verified")

    # Tokenizer round-trip check
    test_str = unicodedata.normalize("NFC", "Đại học Tôn Đức Thắng")
    test_ids = processor.tokenizer.encode(test_str, add_special_tokens=False)
    test_toks = processor.tokenizer.convert_ids_to_tokens(test_ids)
    logger.info(f"[Model] Tokenizer check: '{test_str}' → {test_toks[:10]}")

    # FP16 + eval mode
    model = model.half().to(device).eval()

    # ── Fix meta-device tensors (mirrors manual_export_model.py Step 6) ──
    # TrOCRSinusoidalPositionalEmbedding uses lazy init: its .weights tensor
    # stays on "meta" device after from_pretrained(). We must materialize it
    # on the real device before any forward pass.
    for module in model.modules():
        if module.__class__.__name__ == "TrOCRSinusoidalPositionalEmbedding":
            if hasattr(module, "weights") and isinstance(module.weights, torch.Tensor):
                if module.weights.device.type == "meta" or module.weights.device != device:
                    n_pos, d = module.weights.shape
                    pad_idx = getattr(module, "padding_idx", None)
                    module.weights = module.get_embedding(n_pos, d, pad_idx).to(device)
                    logger.info(f"[Model] Materialized SinusoidalPosEmbed: ({n_pos}, {d}) → {device}")
            if hasattr(module, "_float_tensor") and isinstance(module._float_tensor, torch.Tensor):
                if module._float_tensor.device.type == "meta" or module._float_tensor.device != device:
                    module._float_tensor = torch.zeros(1, device=device)

    # Verify use_cache
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'config'):
        model.decoder.config.use_cache = True

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    vram_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    logger.info(f"[Model] Encoder: {enc_params:,} | Decoder: {dec_params:,} | VRAM≈{vram_mb:.0f}MB")

    return model, processor


# =============================================================================
# 3. Inference Collate
# =============================================================================

def inference_collate(batch):
    """Minimal collate for inference. No label tokenization."""
    images = [s["image"] for s in batch]
    labels = [s["label"] for s in batch]
    indices = [s["idx"] for s in batch]
    raw_images = [s.get("raw_image") for s in batch]
    return images, labels, indices, raw_images


# =============================================================================
# 4. Single-domain Inference
# =============================================================================

@torch.no_grad()
def run_inference_on_domain(
    model,
    processor,
    lmdb_path: str,
    domain_name: str,
    device: torch.device,
    sanitizer: RegexSanitizer,
    phobert: Optional[PhoBERTCorrector],
    addr_corrector: Optional[AddressCorrector],
    args,
) -> List[Dict]:
    """
    Run full inference pipeline on one domain (printed or handwritten).

    Returns list of dicts with keys:
        idx, domain, ref, raw_pred, sanitized_pred, corrected_pred,
        cer_raw, cer_sanitized, cer_corrected, wer_corrected, error_category,
        raw_image_bytes
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"[Eval] Domain: {domain_name.upper()}")
    logger.info(f"[Eval] LMDB: {lmdb_path}")
    logger.info(f"{'='*60}")

    dataset = LMDBDataset(
        lmdb_path=lmdb_path,
        target_h=args.image_height,
        target_w=args.image_width,
        default_data_type=domain_name,
        max_samples=args.max_samples,
        keep_raw=True,  # Need raw images for error analysis
    )
    logger.info(f"[Eval] Samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=inference_collate,
        pin_memory=True,
    )

    results = []
    t0 = time.time()

    for images, labels, indices, raw_images in tqdm(loader, desc=f"Eval {domain_name}"):
        # Step 1 already applied by LMDBDataset (resize_for_vit)
        # Step 4: TrOCR inference
        pixel_values = processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device, dtype=torch.float16)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            generated_ids = model.generate(
                pixel_values,
                max_length=args.max_length,
                num_beams=1,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
            )

        raw_preds = safe_batch_decode(processor.tokenizer, generated_ids)

        for i, (idx, ref, raw_pred) in enumerate(zip(indices, labels, raw_preds)):
            # Sanitize
            sanitized = sanitizer.sanitize(raw_pred)

            # ── PRIMARY PATH: Sanitized → Address Correction (no PhoBERT) ──
            final_no_phobert = sanitized
            if addr_corrector is not None:
                final_no_phobert = addr_corrector.correct(final_no_phobert)
                final_no_phobert = addr_corrector.correct_trailing_province(final_no_phobert)

            # ── COMPARISON PATH: Sanitized → PhoBERT → Address Correction ──
            phobert_pred = phobert.correct(sanitized) if phobert and phobert.is_loaded else sanitized
            final_with_phobert = phobert_pred
            if addr_corrector is not None:
                final_with_phobert = addr_corrector.correct(final_with_phobert)
                final_with_phobert = addr_corrector.correct_trailing_province(final_with_phobert)

            # Metrics — PRIMARY (no PhoBERT)
            cer_raw = compute_sample_cer(ref, raw_pred)
            cer_san = compute_sample_cer(ref, sanitized)
            cer_cor = compute_sample_cer(ref, final_no_phobert)
            wer_cor = compute_sample_wer(ref, final_no_phobert)

            # Metrics — COMPARISON (with PhoBERT)
            cer_phobert = compute_sample_cer(ref, final_with_phobert)
            wer_phobert = compute_sample_wer(ref, final_with_phobert)

            # Error category (based on primary prediction)
            category = ErrorCategorizer.categorize(ref, raw_pred, cer_cor)

            # Serialize raw image for worst-case export
            raw_img_bytes = None
            if raw_images[i] is not None:
                buf = io.BytesIO()
                raw_images[i].save(buf, format="PNG")
                raw_img_bytes = buf.getvalue()

            results.append({
                "idx": idx,
                "domain": domain_name,
                "ref": ref,
                "raw_pred": raw_pred,
                "sanitized_pred": sanitized,
                "corrected_pred": final_no_phobert,        # PRIMARY
                "phobert_corrected_pred": final_with_phobert,  # COMPARISON
                "cer_raw": cer_raw,
                "cer_sanitized": cer_san,
                "cer_corrected": cer_cor,                   # PRIMARY
                "wer_corrected": wer_cor,                   # PRIMARY
                "cer_phobert": cer_phobert,                 # COMPARISON
                "wer_phobert": wer_phobert,                 # COMPARISON
                "error_category": category,
                "raw_image_bytes": raw_img_bytes,
            })

    elapsed = time.time() - t0
    fps = len(results) / max(elapsed, 0.001)

    # Aggregate metrics
    if results:
        import jiwer
        all_refs = [r["ref"] for r in results]
        all_cor = [r["corrected_pred"] for r in results]
        safe_refs = [r if r.strip() else " " for r in all_refs]
        safe_cors = [c if c.strip() else " " for c in all_cor]
        agg_cer = float(jiwer.cer(safe_refs, safe_cors))
        agg_wer = float(jiwer.wer(safe_refs, safe_cors))
    else:
        agg_cer = agg_wer = 1.0

    # VRAM
    if device.type == "cuda":
        vram_peak = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        vram_peak = 0.0

    logger.info(f"\n[Eval] {domain_name.upper()} Results:")
    logger.info(f"  Aggregate CER: {agg_cer:.4f}")
    logger.info(f"  Aggregate WER: {agg_wer:.4f}")
    logger.info(f"  Samples: {len(results)} | FPS: {fps:.1f} | Time: {elapsed:.1f}s")
    logger.info(f"  Peak VRAM: {vram_peak:.2f} GB")

    # Category distribution
    cats = {}
    for r in results:
        c = r["error_category"]
        cats[c] = cats.get(c, 0) + 1
    logger.info(f"  Error categories: {cats}")

    # Sample predictions
    n_show = min(5, len(results))
    logger.info(f"\n  Sample predictions ({n_show}):")
    for r in results[:n_show]:
        logger.info(f"    Ref:  {r['ref'][:80]}")
        logger.info(f"    Pred: {r['corrected_pred'][:80]}")
        logger.info(f"    CER:  {r['cer_corrected']:.4f} [{r['error_category']}]")
        logger.info("")

    return results, {"cer": agg_cer, "wer": agg_wer, "samples": len(results),
                      "fps": fps, "time_s": elapsed, "vram_peak_gb": vram_peak,
                      "categories": cats}


# =============================================================================
# 5. Error Analysis Export
# =============================================================================

def export_error_analysis(
    all_results: List[Dict],
    output_dir: str,
    worst_k: int = 50,
):
    """Export worst-K cases as CSV + Markdown + images."""
    ea_dir = os.path.join(output_dir, "error_analysis")
    img_dir = os.path.join(ea_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Sort by CER descending (worst first)
    sorted_results = sorted(all_results, key=lambda r: r["cer_corrected"], reverse=True)
    worst = sorted_results[:worst_k]

    # ── CSV ──
    csv_path = os.path.join(ea_dir, "worst_cases_report.csv")
    csv_fields = [
        "Rank", "Image_Name", "Domain", "Ground_Truth", "Raw_Pred",
        "Sanitized_Pred", "Corrected_Pred", "CER_Raw", "CER_Sanitized",
        "CER_Corrected", "WER_Corrected", "Error_Category",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rank, r in enumerate(worst, 1):
            img_name = f"{r['domain']}_{r['idx']:04d}.png"
            # Save image
            if r["raw_image_bytes"]:
                img_path = os.path.join(img_dir, img_name)
                with open(img_path, "wb") as imgf:
                    imgf.write(r["raw_image_bytes"])
            writer.writerow({
                "Rank": rank,
                "Image_Name": img_name,
                "Domain": r["domain"],
                "Ground_Truth": r["ref"],
                "Raw_Pred": r["raw_pred"],
                "Sanitized_Pred": r["sanitized_pred"],
                "Corrected_Pred": r["corrected_pred"],
                "CER_Raw": f"{r['cer_raw']:.4f}",
                "CER_Sanitized": f"{r['cer_sanitized']:.4f}",
                "CER_Corrected": f"{r['cer_corrected']:.4f}",
                "WER_Corrected": f"{r['wer_corrected']:.4f}",
                "Error_Category": r["error_category"],
            })
    logger.info(f"[Export] CSV: {csv_path}")

    # ── Markdown ──
    md_path = os.path.join(ea_dir, "worst_cases_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# TrOCR Error Analysis — Worst Cases Report\n\n")
        f.write(f"Total samples evaluated: {len(all_results)}\n")
        f.write(f"Showing top {len(worst)} worst cases by CER (corrected).\n\n")
        f.write("---\n\n")

        for rank, r in enumerate(worst, 1):
            img_name = f"{r['domain']}_{r['idx']:04d}.png"
            f.write(f"## #{rank} — CER={r['cer_corrected']:.4f} | {r['error_category']} | {r['domain']}\n\n")
            f.write(f"**Image:** `{img_name}`\n\n")
            f.write(f"| Field | Value |\n|---|---|\n")
            f.write(f"| Ground Truth | `{r['ref']}` |\n")
            f.write(f"| Raw Prediction | `{r['raw_pred']}` |\n")
            f.write(f"| Sanitized | `{r['sanitized_pred']}` |\n")
            f.write(f"| Final (no PhoBERT) | `{r['corrected_pred']}` |\n")
            f.write(f"| Final (with PhoBERT) | `{r.get('phobert_corrected_pred', r['corrected_pred'])}` |\n")
            f.write(f"| CER (raw→san→final) | {r['cer_raw']:.4f} → {r['cer_sanitized']:.4f} → {r['cer_corrected']:.4f} |\n")
            f.write(f"| CER (with PhoBERT) | {r.get('cer_phobert', r['cer_corrected']):.4f} |\n")
            f.write(f"| WER (final) | {r['wer_corrected']:.4f} |\n\n")
            f.write("---\n\n")

    logger.info(f"[Export] Markdown: {md_path}")

    # ── All predictions CSV (includes both pipelines for comparison) ──
    all_csv = os.path.join(output_dir, "all_predictions.csv")
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "domain", "ref", "raw_pred", "sanitized_pred",
            "corrected_pred", "phobert_corrected_pred",
            "cer_raw", "cer_sanitized", "cer_corrected",
            "cer_phobert", "wer_corrected", "wer_phobert",
            "error_category",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "idx": r["idx"],
                "domain": r["domain"],
                "ref": r["ref"],
                "raw_pred": r["raw_pred"],
                "sanitized_pred": r["sanitized_pred"],
                "corrected_pred": r["corrected_pred"],
                "phobert_corrected_pred": r.get("phobert_corrected_pred", r["corrected_pred"]),
                "cer_raw": f"{r['cer_raw']:.4f}",
                "cer_sanitized": f"{r['cer_sanitized']:.4f}",
                "cer_corrected": f"{r['cer_corrected']:.4f}",
                "cer_phobert": f"{r.get('cer_phobert', r['cer_corrected']):.4f}",
                "wer_corrected": f"{r['wer_corrected']:.4f}",
                "wer_phobert": f"{r.get('wer_phobert', r['wer_corrected']):.4f}",
                "error_category": r["error_category"],
            })
    logger.info(f"[Export] All predictions: {all_csv}")


# =============================================================================
# 6. Main
# =============================================================================

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model_path), "evaluation")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Vietnamese TrOCR — Evaluation & Error Analysis Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model:              {args.model_path}")
    logger.info(f"Printed test:       {args.test_printed}")
    logger.info(f"Handwritten test:   {args.test_handwritten}")
    logger.info(f"Output:             {args.output_dir}")
    logger.info(f"Device:             {device}")
    logger.info(f"Batch size:         {args.batch_size}")
    logger.info(f"Repetition penalty: {args.repetition_penalty}")
    logger.info(f"Image size:         {args.image_height}×{args.image_width}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f}GB")
        torch.cuda.reset_peak_memory_stats(device)

    # ── Load model ──
    model, processor = load_model_and_processor(args.model_path, device)

    # ── Initialize utilities ──
    sanitizer = RegexSanitizer()

    phobert = None
    if not args.skip_phobert:
        phobert = PhoBERTCorrector(
            model_name=args.phobert_model,
            device=device,
            confidence_threshold=args.phobert_threshold,
        )
        phobert.load()
    else:
        logger.info("[PhoBERT] Skipped (--skip_phobert)")

    # ── Initialize Address Corrector ──
    addr_corrector = None
    if not args.skip_address_correction:
        gaz_path = args.gazetteer_path
        if gaz_path is None:
            gaz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vietnam_gazetteer.json")
        if os.path.exists(gaz_path):
            addr_corrector = AddressCorrector(gaz_path)
        else:
            logger.warning(f"[AddressCorrector] Gazetteer not found: {gaz_path} — skipping")
    else:
        logger.info("[AddressCorrector] Skipped (--skip_address_correction)")

    # ── Run evaluation per domain ──
    all_results = []
    domain_metrics = {}

    for domain_name, lmdb_path in [
        ("printed", args.test_printed),
        ("handwritten", args.test_handwritten),
    ]:
        if not os.path.exists(lmdb_path):
            logger.warning(f"[Eval] LMDB not found: {lmdb_path} — skipping {domain_name}")
            continue

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        results, metrics = run_inference_on_domain(
            model, processor, lmdb_path, domain_name, device,
            sanitizer, phobert, addr_corrector, args,
        )
        all_results.extend(results)
        domain_metrics[domain_name] = metrics

    if not all_results:
        logger.error("[Eval] No results produced. Check LMDB paths.")
        return

    # ── Aggregate metrics ──
    import jiwer
    all_refs = [r["ref"] for r in all_results]
    all_cors = [r["corrected_pred"] for r in all_results]
    safe_refs = [r if r.strip() else " " for r in all_refs]
    safe_cors = [c if c.strip() else " " for c in all_cors]
    overall_cer = float(jiwer.cer(safe_refs, safe_cors))
    overall_wer = float(jiwer.wer(safe_refs, safe_cors))

    # ── PhoBERT comparison analysis ──
    # Compare PRIMARY (no PhoBERT) vs COMPARISON (with PhoBERT)
    all_phobert_cors = [r.get("phobert_corrected_pred", r["corrected_pred"]) for r in all_results]
    safe_phobert_cors = [c if c.strip() else " " for c in all_phobert_cors]
    overall_cer_phobert = float(jiwer.cer(safe_refs, safe_phobert_cors))
    overall_wer_phobert = float(jiwer.wer(safe_refs, safe_phobert_cors))

    phobert_helped = sum(1 for r in all_results if r.get("cer_phobert", r["cer_corrected"]) < r["cer_corrected"])
    phobert_hurt = sum(1 for r in all_results if r.get("cer_phobert", r["cer_corrected"]) > r["cer_corrected"])
    phobert_neutral = len(all_results) - phobert_helped - phobert_hurt

    logger.info(f"\n{'='*60}")
    logger.info("PhoBERT COMPARISON (no PhoBERT vs with PhoBERT)")
    logger.info(f"{'='*60}")
    logger.info(f"  Without PhoBERT — CER: {overall_cer:.4f} | WER: {overall_wer:.4f}  (PRIMARY)")
    logger.info(f"  With PhoBERT    — CER: {overall_cer_phobert:.4f} | WER: {overall_wer_phobert:.4f}")
    logger.info(f"  PhoBERT helped {phobert_helped} | hurt {phobert_hurt} | neutral {phobert_neutral}")

    # Per-domain PhoBERT comparison
    phobert_domain_metrics = {}
    for domain_name in domain_metrics:
        domain_results = [r for r in all_results if r["domain"] == domain_name]
        if domain_results:
            d_refs = [r["ref"] if r["ref"].strip() else " " for r in domain_results]
            d_phobert = [r.get("phobert_corrected_pred", r["corrected_pred"]) for r in domain_results]
            d_phobert = [c if c.strip() else " " for c in d_phobert]
            d_no_phobert = [r["corrected_pred"] if r["corrected_pred"].strip() else " " for r in domain_results]
            phobert_domain_metrics[domain_name] = {
                "cer_without_phobert": round(float(jiwer.cer(d_refs, d_no_phobert)), 6),
                "cer_with_phobert": round(float(jiwer.cer(d_refs, d_phobert)), 6),
                "wer_without_phobert": round(float(jiwer.wer(d_refs, d_no_phobert)), 6),
                "wer_with_phobert": round(float(jiwer.wer(d_refs, d_phobert)), 6),
            }
            logger.info(f"  {domain_name.upper():12s} — no PhoBERT CER: {phobert_domain_metrics[domain_name]['cer_without_phobert']:.4f} | "
                         f"with PhoBERT CER: {phobert_domain_metrics[domain_name]['cer_with_phobert']:.4f}")

    # ── Summary ──
    summary = {
        "overall": {"cer": overall_cer, "wer": overall_wer, "total_samples": len(all_results)},
        "per_domain": domain_metrics,
        "phobert_comparison": {
            "overall_cer_without_phobert": round(overall_cer, 6),
            "overall_cer_with_phobert": round(overall_cer_phobert, 6),
            "overall_wer_without_phobert": round(overall_wer, 6),
            "overall_wer_with_phobert": round(overall_wer_phobert, 6),
            "helped": phobert_helped,
            "hurt": phobert_hurt,
            "neutral": phobert_neutral,
            "per_domain": phobert_domain_metrics,
        },
        "config": {
            "model_path": args.model_path,
            "repetition_penalty": args.repetition_penalty,
            "max_length": args.max_length,
            "image_size": f"{args.image_height}x{args.image_width}",
            "batch_size": args.batch_size,
            "phobert": args.phobert_model if not args.skip_phobert else "disabled",
            "address_correction": not args.skip_address_correction,
        },
    }

    summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n[Export] Metrics summary: {summary_path}")

    # ── Error analysis ──
    export_error_analysis(all_results, args.output_dir, args.worst_k)

    # ── Final report ──
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS (PRIMARY = no PhoBERT)")
    logger.info("=" * 60)
    logger.info(f"  Overall CER: {overall_cer:.4f}")
    logger.info(f"  Overall WER: {overall_wer:.4f}")
    for domain, m in domain_metrics.items():
        logger.info(f"  {domain.upper():12s} — CER: {m['cer']:.4f} | WER: {m['wer']:.4f} | "
                     f"Samples: {m['samples']} | FPS: {m['fps']:.1f}")
    logger.info(f"\n  PhoBERT comparison CER: {overall_cer_phobert:.4f} (vs {overall_cer:.4f} without)")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
