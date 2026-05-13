# Multi-tier Baseline Benchmarking Suite

> **Location:** `Fine tuning/code/`
> **Entry point:** `evaluate_baselines.py`
> **Notebook:** `Viet_TrOCR.ipynb` → **"EXTERNAL BASELINE EVALUATION"** section
> **Dependencies:** `vietocr`, `jiwer`, `lmdb`, `Pillow`, `tqdm`

---

## Colab Quick Start

This evaluation is run from the **"EXTERNAL BASELINE EVALUATION"** section of `Viet_TrOCR.ipynb`.

### Prerequisites

Before running baseline evaluation, ensure:
1. The [Environment Setup](README.md#5-environment-setup) cells have been run (Drive mounted, LMDB copied to local SSD, code copied)
2. LMDB test sets exist at `/content/lmdb/line_printed/test` and `/content/lmdb/line_handwritten/test`

> **Note:** Unlike TrOCR evaluation, baseline evaluation does NOT require `final_model/`. VietOCR and CRNN weights are downloaded automatically.

### Step-by-Step

**Cell 1 — Install baseline dependencies:**

```python
!pip install -q vietocr jiwer lmdb Pillow tqdm
```

**Cell 2 — Run baselines:**

```python
%cd /content/trocr_viet
!python evaluate_baselines.py \
    --test_printed /content/lmdb/line_printed/test \
    --test_handwritten /content/lmdb/line_handwritten/test \
    --output_dir /content/drive/MyDrive/OCR/checkpoints/baseline_eval \
    --device cuda
```

**Cell 3 — View results:**

```python
import json
with open("/content/drive/MyDrive/OCR/checkpoints/baseline_eval/baseline_metrics_summary.json") as f:
    data = json.load(f)
print(json.dumps(data, indent=2, ensure_ascii=False))
```

**Expected runtime:** ~45 minutes on NVIDIA L4 (4,477 samples × 2 baselines, sequential inference).

---

## Overview

Evaluates two baseline models against the **exact same 4,477 test images** used in the TrOCR evaluation (`evaluation_v3`), using identical CER/WER computation logic. Establishes lower-bound and domain-specific baselines for the Multi-tier Benchmarking Strategy.

| Tier | Model | Architecture | Pre-trained | Purpose |
|---|---|---|---|---|
| **2** | VietOCR | VGG + Transformer | ✓ Vietnamese | Domain-specific baseline |
| **1** | CRNN + CTC | VGG + Seq2Seq / BiLSTM+CTC | ✓ / ✗ | Architectural lower-bound |
| **3** | TrOCR (ours) | ViT + Transformer Decoder | ✓ Fine-tuned | Proposed system (reference) |

---

## Fair Comparison Guarantees

| Constraint | Implementation |
|---|---|
| Same test data | Same LMDB paths: `line_printed/test` (3,739) + `line_handwritten/test` (738) |
| Same metrics | Reuses `compute_sample_cer` / `compute_sample_wer` from `eval_utils.py` |
| Same normalization | Unicode NFC on all labels + `RegexSanitizer` on all predictions |
| Same error taxonomy | Reuses `ErrorCategorizer` (PERFECT / MINOR / SUBSTITUTION / HALLUCINATION / TRUNCATION) |
| Model-appropriate preprocessing | Each baseline uses its own resize logic (height=32 for VietOCR/CRNN, not TrOCR's 128×1536) |
| Hardware profiling | Per-sample inference time (ms), FPS, total parameter count per model |

---

## Baselines

### Baseline 1 — VietOCR (VGG + Transformer)

- **Package:** [`pbcquoc/vietocr`](https://github.com/pbcquoc/vietocr)
- **Config:** `vgg_transformer` — VGG-based CNN encoder + Transformer decoder
- **Weights:** Pre-trained on Vietnamese text (auto-downloaded)
- **Preprocessing:** Internal (resize to height=32, preserve aspect ratio)
- **Inference:** Single-image via `Predictor.predict()`

### Baseline 2 — CRNN + CTC

- **Primary:** `vgg_seq2seq` config from vietocr — CRNN-family Seq2Seq model
- **Fallback:** Custom PyTorch CRNN+CTC (VGG-like CNN → BiLSTM → CTC greedy decode)
  - **No pre-trained weights** — establishes the pure architectural lower-bound
  - Vietnamese charset: 229 characters (ASCII + full Vietnamese diacritics + punctuation)
  - Input: grayscale, height=32, aspect-ratio preserved

---

## Usage

### CLI

```bash
python evaluate_baselines.py \
    --test_printed /content/lmdb/line_printed/test \
    --test_handwritten /content/lmdb/line_handwritten/test \
    --output_dir /content/drive/MyDrive/OCR/checkpoints/baseline_eval \
    --device cuda
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--test_printed` | (required) | LMDB path for printed test set |
| `--test_handwritten` | (required) | LMDB path for handwritten test set |
| `--output_dir` | `../baseline_eval` | Output directory |
| `--batch_size` | `32` | Not used for batching (sequential), reserved |
| `--max_samples` | `None` | Limit samples per domain (for quick testing) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--skip_vietocr` | `false` | Skip VietOCR evaluation |
| `--skip_crnn` | `false` | Skip CRNN evaluation |

### Colab Cells

**Cell 1 — Install:**
```python
!pip install -q vietocr jiwer lmdb Pillow tqdm
```

**Cell 2 — Run:**
```python
%cd /content/trocr_viet
!python evaluate_baselines.py \
    --test_printed /content/lmdb/line_printed/test \
    --test_handwritten /content/lmdb/line_handwritten/test \
    --output_dir /content/drive/MyDrive/OCR/checkpoints/baseline_eval
```

**Cell 3 — View:**
```python
import json
with open("/content/drive/MyDrive/OCR/checkpoints/baseline_eval/baseline_metrics_summary.json") as f:
    print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
```

---

## Output Files

| File | Description |
|---|---|
| `baseline_metrics_summary.json` | Combined metrics for all baselines + TrOCR reference |
| `vietocr_predictions.csv` | Per-sample predictions & metrics for VietOCR |
| `crnn_predictions.csv` | Per-sample predictions & metrics for CRNN |

### CSV Columns

| Column | Description |
|---|---|
| `idx` | LMDB sample index |
| `domain` | `printed` or `handwritten` |
| `ref` | Ground truth label (NFC-normalized) |
| `raw_pred` | Raw model output |
| `sanitized_pred` | After RegexSanitizer |
| `cer` | Character Error Rate (on sanitized) |
| `wer` | Word Error Rate (on sanitized) |
| `error_category` | Programmatic error classification |
| `inference_ms` | Per-sample inference time in milliseconds |

### JSON Structure (`baseline_metrics_summary.json`)

```json
{
  "test_data": { "printed_samples": 3739, "handwritten_samples": 738, "total": 4477 },
  "trocr_reference": {
    "overall": { "cer": 0.0286, "wer": 0.0792 },
    "per_domain": {
      "printed":     { "cer": 0.0117, "perfect_pct": 68.5 },
      "handwritten": { "cer": 0.1057, "perfect_pct": 14.8 }
    }
  },
  "baselines": {
    "vietocr": {
      "model_name": "VietOCR (VGG-Transformer)",
      "overall": { "cer": "...", "wer": "..." },
      "per_domain": { "printed": {...}, "handwritten": {...} },
      "parameter_count": { "total": "...", "trainable": "..." }
    },
    "crnn": { ... }
  }
}
```

---

## Pipeline Flow

```
LMDB test set (same 4,477 images)
  │
  ├─→ VietOCR (vgg_transformer)
  │     └─→ predict(PIL.Image) → NFC normalize → RegexSanitizer
  │           └─→ CER/WER (jiwer) + ErrorCategorizer + timing
  │
  ├─→ CRNN (vgg_seq2seq or custom CTC)
  │     └─→ predict(PIL.Image) → NFC normalize → RegexSanitizer
  │           └─→ CER/WER (jiwer) + ErrorCategorizer + timing
  │
  └─→ Aggregate → baseline_metrics_summary.json + comparison table
```

---

## Console Output Example

```
==========================================================================================
MULTI-TIER BASELINE COMPARISON — Vietnamese OCR
==========================================================================================
Model                                CER      WER   Pr.CER   HW.CER  Pr.PERF       Params
------------------------------------------------------------------------------------------
TrOCR (ours, fine-tuned)          0.0286   0.0792   0.0117   0.1057    68.5%        ~337M
------------------------------------------------------------------------------------------
VietOCR (VGG-Transformer)         0.XXXX   0.XXXX   0.XXXX   0.XXXX    XX.X%       XX.XM
CRNN-Seq2Seq (vgg_seq2seq)        0.XXXX   0.XXXX   0.XXXX   0.XXXX    XX.X%       XX.XM
==========================================================================================
```

---

## Notes

- **VietOCR inference is single-image** — the official `Predictor.predict()` does not support native batching. FPS reflects sequential processing.
- **CRNN fallback** — If `vgg_seq2seq` config fails to load, the script falls back to a custom untrained CRNN+CTC. This is intentional — it represents the pure architectural lower-bound without any learned Vietnamese features.
- **RegexSanitizer is applied to all baselines** — This ensures fair comparison since TrOCR's reported metrics include sanitization. Without it, hallucination patterns (which all autoregressive models can produce) would unfairly inflate CER.
- **No post-processing beyond Sanitizer** — AddressCorrector and PhoBERT are NOT applied to baselines. The comparison measures raw model capability + minimal sanitization only.
