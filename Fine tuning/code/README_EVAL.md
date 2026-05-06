# Post-Training Evaluation & Error Analysis Suite

> **Location:** `Fine tuning/code/`
> **Entry point:** `evaluate_and_analyze.py`
> **Dependencies:** `transformers`, `jiwer`, `Pillow`, `tqdm`, `pyvi`, `openpyxl`

---

## Overview

This suite implements a production-grade evaluation pipeline for the Vietnamese TrOCR (VisionEncoderDecoder) model. It performs end-to-end inference on LMDB test sets and applies a multi-stage post-processing pipeline with quantitative ablation at every stage.

The evaluation architecture mirrors Steps 4–5 of the proposed 5-step OCR pipeline:
- **Step 4**: TrOCR text recognition (greedy decoding with `repetition_penalty=1.2`)
- **Step 5**: Language-aware post-processing (sanitization + geographic correction)

### Pipeline Flow

```
TrOCR (FP16, greedy decode)
  → RegexSanitizer          [Hallucination collapse]
  → AddressCorrector v2     [Geographic name fuzzy-matching]
  → (PhoBERT comparison)    [MLM rescoring — comparison path only]
```

---

## Components

### 1. `evaluate_and_analyze.py` — Main Orchestrator

**CLI Usage:**
```bash
python evaluate_and_analyze.py \
    --model_path /path/to/final_model \
    --test_printed /path/to/lmdb/line_printed/test \
    --test_handwritten /path/to/lmdb/line_handwritten/test \
    --output_dir /path/to/output \
    --batch_size 4 \
    --repetition_penalty 1.2
```

**Key Features:**
- FP16 mixed-precision inference on CUDA
- Automatic materialization of TrOCR sinusoidal positional embeddings (meta-device fix)
- Dual-path execution: PRIMARY (no PhoBERT) and COMPARISON (with PhoBERT)
- Per-sample CER/WER computation with error categorization
- Exports: `metrics_summary.json`, `all_predictions.csv`, `worst_cases_report.csv/.md`

**CLI Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model_path` | (required) | Path to exported HuggingFace `final_model/` |
| `--test_printed` | (required) | LMDB path for printed test set |
| `--test_handwritten` | (required) | LMDB path for handwritten test set |
| `--output_dir` | `model_path/../evaluation` | Output directory |
| `--batch_size` | 4 | Inference batch size |
| `--repetition_penalty` | 1.2 | TrOCR repetition penalty |
| `--max_length` | 256 | Max generation length |
| `--image_height` | 128 | Input image height |
| `--image_width` | 1536 | Input image width |
| `--worst_k` | 50 | Number of worst cases to export |
| `--skip_phobert` | false | Skip PhoBERT loading entirely |
| `--phobert_threshold` | 0.85 | PhoBERT MLM confidence threshold |
| `--skip_address_correction` | false | Skip address gazetteer correction |
| `--gazetteer_path` | `data/vietnam_gazetteer.json` | Custom gazetteer path |
| `--max_samples` | None | Limit samples per domain (testing) |

---

### 2. `eval_utils.py` — Core Utilities

#### 2.1 RegexSanitizer

Deterministic post-generation cleanup that collapses hallucinated repetitions from TrOCR's greedy decoding.

**Rules (applied sequentially):**
1. 4+ consecutive identical characters → max 3 (preserves `...` ellipsis)
2. 3+ consecutive identical tokens → max 2
3. 3+ consecutive identical punctuation groups → max 2
4. Trailing repeated short tokens at end of string
5. Normalize excessive whitespace

**Impact:** Recovered 41 samples with CER improvements up to 11.7× per sample.

#### 2.2 PhoBERTCorrector

MLM-based language correction using `vinai/phobert-base` with `pyvi` word segmentation.

**Design:**
- Per-token masking with top-k candidate rescoring
- Accepts correction only if: confidence > threshold AND edit_distance ≤ 2
- Skips: pure punctuation, numbers, single characters, uppercase abbreviations (TP, TX)
- Punctuation re-attachment: restores original spacing around `,.;:!?%()[]/-`

**Research finding:** PhoBERT was empirically shown to be a net negative on this dataset (hurts 445 vs helps 67 samples). It is retained for comparison metrics only.

#### 2.3 ErrorCategorizer

Programmatic classification of prediction errors:

| Category | Condition |
|---|---|
| `PERFECT` | CER = 0 |
| `MINOR_ERROR` | CER ≤ 0.30, no structural anomaly |
| `SUBSTITUTION` | CER > 0.30, no structural anomaly |
| `HALLUCINATION_LOOP` | Detected 4+ identical chars or 3+ repeated words |
| `TRUNCATION` | Prediction length < 30% of reference |
| `INSERTION` | Prediction length > 200% of reference |

#### 2.4 safe_batch_decode

Robust token-to-text decoder that bypasses HuggingFace byte truncation issues with Vietnamese `AddedTokens`. Handles RoBERTa's `Ġ` (space) and `Ċ` (newline) markers.

---

### 3. `address_corrector.py` — Geographic Name Correction

Vietnamese address entity corrector using fuzzy-matching against official administrative division data.

**Data source:** [madnh/hanhchinhvn](https://github.com/madnh/hanhchinhvn) (General Statistics Office of Vietnam)

**Gazetteer:** `data/vietnam_gazetteer.json`
- 63 provinces/cities
- 705 districts
- 10,599 wards/communes

**Algorithm:**
1. Detect address-like text via administrative keywords (`Phường`, `Quận`, `Huyện`, `TP`, `Xã`, etc.)
2. Extract province context from trailing comma-separated segment
3. For each keyword-entity pair:
   - **Skip-if-valid**: If entity exactly matches any gazetteer entry → do not modify
   - **Province-scoped matching**: Restrict candidates to the detected province
   - **Fuzzy match**: Levenshtein distance ratio ≤ 0.35
4. Correct trailing province name (keyword-less final segment)

**Impact:** Modified 315 samples; helped 218 (69.2%), hurt 73 (23.2%).

---

## Output Files

| File | Description |
|---|---|
| `metrics_summary.json` | Aggregate CER/WER, per-domain metrics, PhoBERT comparison, config |
| `all_predictions.csv` | All 4,477 predictions with both pipeline paths and per-sample metrics |
| `error_analysis/worst_cases_report.csv` | Top-K worst predictions by CER |
| `error_analysis/worst_cases_report.md` | Formatted worst-case report with image references |
| `error_analysis/images/` | Extracted source images for worst cases |

### CSV Columns (`all_predictions.csv`)

| Column | Description |
|---|---|
| `idx` | Sample index within LMDB |
| `domain` | `printed` or `handwritten` |
| `ref` | Ground truth label |
| `raw_pred` | Raw TrOCR output |
| `sanitized_pred` | After RegexSanitizer |
| `corrected_pred` | **Primary** — After AddressCorrector (no PhoBERT) |
| `phobert_corrected_pred` | **Comparison** — After PhoBERT + AddressCorrector |
| `cer_raw` / `cer_sanitized` | CER at each stage |
| `cer_corrected` | **Primary CER** (reported metric) |
| `cer_phobert` | CER with PhoBERT path |
| `wer_corrected` / `wer_phobert` | WER for both paths |
| `error_category` | Programmatic error classification |

### metrics_summary.json Structure

```json
{
  "overall": { "cer": 0.0286, "wer": 0.0792, "total_samples": 4477 },
  "per_domain": {
    "printed": { "cer": ..., "wer": ..., "samples": ..., "fps": ..., "categories": {...} },
    "handwritten": { ... }
  },
  "phobert_comparison": {
    "overall_cer_without_phobert": ...,
    "overall_cer_with_phobert": ...,
    "helped": ..., "hurt": ..., "neutral": ...,
    "per_domain": { ... }
  },
  "config": { ... }
}
```

---

## Final Results (V3)

| Metric | Printed | Handwritten | Overall |
|---|---|---|---|
| **CER** | 0.0117 | 0.1057 | **0.0286** |
| **WER** | 0.0396 | 0.2628 | **0.0792** |
| PERFECT | 2,562 (68.5%) | 109 (14.8%) | 2,671 (59.7%) |
| FPS | 2.1 | 1.8 | — |
| VRAM Peak | 1.40 GB | 1.40 GB | — |

---

## Version History

| Version | Changes | CER Impact |
|---|---|---|
| V1 | Initial pipeline (RegexSanitizer + PhoBERT) | 0.0658 |
| V2 | + Punctuation re-attachment, + Address Corrector v1 | 0.0351 (−46.7%) |
| V3 | + Skip-if-valid, + Ward data, + Province context, + Dual-path PhoBERT | **0.0286** (−18.5%) |
