<div align="center">

# 🇻🇳 Vietnamese Printed & Handwritten Text Recognition Using TrOCR

**Research and Optimization of a Dual-Domain Vietnamese OCR System with Curriculum Learning, Elastic Weight Consolidation, and Deterministic Post-Processing**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.40%2B-FFD21E)](https://huggingface.co/docs/transformers)
[![Colab Ready](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)](https://colab.google/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br>

*Information Technology Project — Ton Duc Thang University*

</div>

---

## Abstract

Vietnamese Optical Character Recognition (OCR) presents a compounding set of challenges: a rich diacritical system where a single syllable can carry both a vowel modifier and a tone mark (e.g., `ộ` = `o` + circumflex + dot-below), two visually disjoint text domains (printed vs. handwritten), and the risk of catastrophic forgetting when adapting a model sequentially across these domains.

This project proposes a **multi-stage curriculum training pipeline** built on Microsoft's [TrOCR](https://arxiv.org/abs/2109.10282) architecture (ViT encoder + RoBERTa decoder, ~337M parameters), fine-tuned from `trocr-base-stage1` with a custom 260-token Vietnamese character-level tokenizer. The system employs a three-stage curriculum — progressing from word-level warm-up through printed line fine-tuning to handwritten domain adaptation — protected by **Elastic Weight Consolidation (EWC)** to preserve printed text accuracy during handwritten training.

The final system achieves an **overall CER of 2.86%** on a 4,477-sample dual-domain test set, representing a **55.5% improvement over VietOCR** and an **80.1% improvement over CRNN-Seq2Seq**. A deterministic post-processing pipeline (RegexSanitizer + AddressCorrector) contributes an additional **56.5% CER reduction** without model retraining. The complete system operates at ~2 FPS with 1.4 GB VRAM under FP16 inference on an NVIDIA L4 GPU.

---

## Key Innovations

| Innovation | Description |
|---|---|
| 🎓 **Curriculum Learning via Pseudo-Lines** | Three-phase Stage 1 (100% words → 50% pseudo-lines → 80% pseudo-lines) progressively calibrates the decoder's positional embeddings before real line-level data, achieving a **78.1% CER reduction** at the Stage 1→2a boundary. |
| 🛡️ **Elastic Weight Consolidation (EWC)** | Fisher Information-based regularization preserves printed CER to within **0.44 percentage points** during 17 epochs of handwritten adaptation — an asymmetric improvement-to-degradation ratio of **42:1**. |
| 🔤 **Vietnamese Character-Level Tokenizer** | Replaces RoBERTa's 50,265-token BPE tokenizer with a curated 260-token character vocabulary covering all 134 Vietnamese diacritical characters, reducing the embedding layer from ~38M to ~0.2M parameters. |
| 🖼️ **Aspect-Ratio-Aware ViT Input** | Bicubic interpolation of positional embeddings from 24×24 → 8×96 grid for 128×1536 input resolution, preserving spatial priors while supporting the natural aspect ratio of text lines. |
| ⚡ **FP16 Training with EWC Stability Fixes** | Custom FP16 pipeline for Fisher computation: FP32-cast gradients, inf-gradient batch skipping, dummy optimizer pattern, and post-batch scaler reset — enabling EWC on consumer GPUs. |
| 🏘️ **Province-Scoped Address Correction** | Fuzzy-matching geographic entity corrector with 63 provinces, 705 districts, and 10,599 wards — increasing handwritten perfect recognition by **65.2%**. |
| 🔬 **Dual-Domain Benchmarking** | Controlled evaluation against VietOCR (VGG-Transformer) and CRNN-Seq2Seq on an identical 4,477-sample test set with uniform CER/WER computation and Unicode NFC normalization. |

---

## System Architecture

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐     ┌───────────────────┐
 │  Step 1      │     │  Step 2      │     │  Step 3      │     │  Step 4              │     │  Step 5           │
 │  Image       │     │  Text        │     │  Line        │     │  Text Recognition    │     │  Post-Processing  │
Input ──▶│  Preprocessing│──▶│  Detection   │──▶│  Segmentation│──▶│                      │──▶│                   │──▶ Output
Image   │              │     │              │     │              │     │  TrOCR               │     │  RegexSanitizer   │   Text
 │  • CLAHE      │     │  • DBNet      │     │  • DBSCAN    │     │  (ViT + RoBERTa)     │     │  AddressCorrector │
 │  • Hough      │     │    (docTR)    │     │  • Adaptive  │     │  128×1536 FP16       │     │  (fuzzy-matching) │
 │    Deskew     │     │  • conf≥0.3  │     │    eps, L-R  │     │  Greedy decode       │     │                   │
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────────────┘     └───────────────────┘
       CPU                  GPU                  CPU                     GPU                        CPU
```

The end-to-end inference pipeline consists of five stages:

```
Input Image → [1] Preprocessing → [2] Text Detection → [3] Line Segmentation → [4] Text Recognition → [5] Post-Processing → Final Text
```

| Step | Module | Description | GPU |
|---|---|---|---|
| **1. Preprocessing** | CLAHE + Hough Deskew | Contrast enhancement (LAB L-channel) and rotation correction | ✗ |
| **2. Text Detection** | DBNet (`db_resnet50` via docTR) | Detects text regions with confidence filtering (≥ 0.3) | ✓ |
| **3. Line Segmentation** | DBSCAN Clustering | Adaptive-eps clustering on y-centers → reading-order sort | ✗ |
| **4. Text Recognition** | TrOCR (ViT + RoBERTa Decoder) | Batched FP16 inference at 128×1536, greedy decode with repetition penalty | ✓ |
| **5. Post-Processing** | RegexSanitizer + AddressCorrector | Hallucination collapse + geographic entity fuzzy-matching | ✗ |

**Total VRAM:** ~900 MB (without PhoBERT) — deployable on T4/L4 GPUs.

---

## Training Pipeline

```
┌─── Stage 1: Curriculum Learning (20 epochs, ~25.7 hours) ───────────────────────────┐
│  Phase 1A          Phase 1B             Phase 1C                                     │
│  Word-Level ──────▶ 50% Pseudo-Lines ──▶ 80% Pseudo-Lines                            │
│  5 epochs           10 epochs            5 epochs                                    │
└──────────────────────────────────────────────────────┬───────────────────────────────-┘
                                                       │ Positional embeddings calibrated
                                                       ▼
┌─── Stage 2a: Printed Fine-Tuning (12 epochs, ~9.5 hours) ──────────────────────────-┐
│  Line-level printed data │ Best Printed CER: 1.90% │ Fisher Information computed     │
└──────────────────────────────────────────────────────┬───────────────────────────────-┘
                                                       │ EWC protection activated
                                                       ▼
┌─── Stage 2b: Handwritten Adaptation (17 epochs, ~6.4 hours) ───────────────────────-┐
│  70% HW + 30% Printed Replay │ EWC + Mixed Replay + LLRD │ Final HW CER: 10.57%    │
└──────────────────────────────────────────────────────────────────────────────────────-┘

Grand Total: 49 epochs, ~41.6 hours on NVIDIA L4 GPU
```

---

## Repository Structure

```
Vietnamese-TrOCR/
│
├── 📄 README.md                          ← You are here
│
├── 📂 Data Processing/                   Data preprocessing & LMDB export
│   ├── scripts/                          4-step pipeline: parse → split → filter → export
│   ├── raw_data/                         7 source datasets (UIT-HWDB, Cinnamon, VinText, ...)
│   ├── processed/                        Normalized images + labels CSV
│   ├── lmdb/                             Output LMDB databases (word/line × printed/HW)
│   └── 📄 README.md                     ← Detailed data pipeline documentation
│
├── 📂 Generate Synthetic Printed Data/   Synthetic modern Vietnamese text generation
│   ├── 01_build_corpus.py                Wikipedia + domain-specific corpus (7 domains)
│   ├── 02_download_fonts.py              14 Google Font families
│   ├── 03_generate_images.py             30,000+ augmented text line images
│   ├── 04_integrate_pipeline.py          Auto-copy to Data Processing pipeline
│   └── 📄 README.md                     ← Synthetic data generation guide
│
├── 📂 Fine tuning/                       Model training, evaluation & analysis
│   ├── code/                             Training pipeline source code
│   │   ├── main.py                       Entry point: model setup & stage dispatch
│   │   ├── core/trainer.py               TrOCRTrainer: LLRD, AMP, dual validation
│   │   ├── core/ewc.py                   Elastic Weight Consolidation implementation
│   │   ├── data/dataset.py               LMDB dataset, pseudo-line builder, mixed sampling
│   │   ├── config.yaml                   Master hyperparameter configuration
│   │   ├── evaluate_and_analyze.py       Comprehensive evaluation suite
│   │   ├── evaluate_baselines.py         VietOCR & CRNN benchmark runner
│   │   └── 📄 README.md                 ← Full training pipeline documentation
│   ├── evaluation_v3/                    Final evaluation results & analysis
│   ├── baseline_eval/                    External baseline comparison data
│   ├── training-visualization/           Training dynamics charts & scripts
│   ├── checkpoints/                      Model checkpoints (Google Drive)
│   └── lmdb/                             Symlink/copy of training LMDB data
│
├── 📂 interface/                         End-to-end inference web application
│   ├── app.py                            Gradio web UI entry point
│   ├── pipeline.py                       OCRPipeline orchestrator
│   ├── modules/                          5-step pipeline modules
│   │   ├── preprocessor.py               CLAHE + Hough deskew
│   │   ├── text_detector.py              DBNet text detection (docTR)
│   │   ├── line_segmenter.py             DBSCAN line grouping
│   │   ├── text_recognizer.py            TrOCR batched recognition
│   │   └── post_processor.py             Sanitizer + Address + PhoBERT
│   └── 📄 README.md                     ← Interface & deployment guide
│
└── 📂 Report/                            Project document & figures
    └── REPORT.docx
```

> **Navigation tip:** Each subfolder contains its own detailed `README.md` with installation steps, usage instructions, and technical documentation. This root README focuses on the project's "what" and "why" — consult the sub-READMEs for the "how."

---

## Performance Summary

### Final Test Set Results (4,477 samples)

| Model | Params | Overall CER ↓ | Overall WER ↓ | Printed CER | HW CER | Printed Perfect | HW Perfect |
|---|---|---|---|---|---|---|---|
| CRNN-Seq2Seq | 22.4M | 14.34% | 27.95% | 5.47% | 59.27% | 31.9% | 0.0% |
| VietOCR | 37.7M | 6.42% | 18.17% | 3.04% | 23.55% | 45.2% | 0.3% |
| **TrOCR (Ours)** | **~337M** | **2.86%** | **7.92%** | **1.17%** | **10.57%** | **68.5%** | **14.8%** |

**Relative improvement over baselines:**

| Comparison | CER Reduction | WER Reduction |
|---|---|---|
| TrOCR vs. VietOCR | **55.5%** | **56.4%** |
| TrOCR vs. CRNN-Seq2Seq | **80.1%** | **71.7%** |

### Post-Processing Impact

| Pipeline Version | Overall CER | Δ CER |
|---|---|---|
| V1 (RegexSanitizer + PhoBERT) | 6.58% | — |
| V2 (+ AddressCorrector v1) | 3.51% | −46.7% |
| **V3 Final** (optimized, PhoBERT disabled) | **2.86%** | **−18.5%** |
| **Cumulative V1→V3** | | **−56.5%** |

### Computational Efficiency

| Model | Printed FPS | HW FPS | Avg Inference | VRAM (FP16) |
|---|---|---|---|---|
| CRNN-Seq2Seq | 13.3 | 9.5 | 75–105 ms | — |
| VietOCR | 2.5 | 2.3 | 397–440 ms | — |
| **TrOCR (Ours)** | **2.06** | **1.82** | **~490 ms** | **1.40 GB** |

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥4 GB VRAM (T4, L4, or better)
- CUDA 11.8+ with PyTorch 2.0+

### 1. Clone the Repository

```bash
git clone https://github.com/dongduy-dev/viet-trocr.git
cd viet-trocr
```
### 2. Download Pre-trained Model & Results

The fine-tuned model, training checkpoints, evaluation results, and source code are hosted on Google Drive:

📁 **[Google Drive — OCR Project Files](https://drive.google.com/drive/folders/1xjLg5rNRuZVtv7umXdsG0qRbFWvA7ToA)**

```
OCR/
├── code/                   ← Training pipeline source code
├── interface/              ← Inference web application
├── lmdb/                   ← LMDB training/test databases
├── checkpoints/
│   ├── final_model/        ← Exported HuggingFace model (ready for inference)
│   ├── evaluation/         ← TrOCR evaluation results (metrics_summary.json, etc.)
│   ├── baseline_eval/      ← VietOCR & CRNN benchmark results
│   ├── stage2b_best.pt     ← Best training checkpoint (~4.31 GB)
│   └── ewc_state.pt        ← Fisher Information state (~2.87 GB)
└── logs/                   ← Training logs
```

> Copy the entire `OCR/` folder to your Google Drive under `My Drive/OCR/` to use with the Colab notebook.

### 3. Open the Colab Notebook

The primary entry point for all operations is **`Viet_TrOCR.ipynb`** (at the repository root).
Upload it to Google Colab and use the appropriate section:

| Notebook Section | Task | Time on L4 |
|---|---|---|
| **FINE TUNING** | Multi-stage curriculum training (Stage 1 → 2a → 2b) | ~42 hours |
| **EVALUATE** | Full test set evaluation with post-processing ablation | ~30 min |
| **EXTERNAL BASELINE EVALUATION** | VietOCR + CRNN benchmark comparison | ~45 min |
| **MANUAL FINAL MODEL EXPORT** | Export best checkpoint to HuggingFace format | ~5 min |
| **INTERFACE** | Launch Gradio web UI for live OCR inference | Instant |

### 4. Component READMEs

For detailed documentation on each module:

| Task | Directory | README |
|---|---|---|
| Prepare training data | `Data Processing/` | [Data Processing README](Data%20Processing/README.md) |
| Generate synthetic printed data | `Generate Synthetic Printed Data/` | [Synthetic Data README](Generate%20Synthetic%20Printed%20Data/README.md) |
| Train the model | `Fine tuning/code/` | [Training Pipeline README](Fine%20tuning/code/README.md) |
| Evaluate the model | `Fine tuning/code/` | [Evaluation README](Fine%20tuning/code/README_EVAL.md) |
| Benchmark baselines | `Fine tuning/code/` | [Baseline Evaluation README](Fine%20tuning/code/README_EVAL_BASELINE.md) |
| Run inference | `interface/` | [Interface README](interface/README.md) |

> Each sub-README contains its own installation instructions, dependency lists, and step-by-step guides.

---

## Datasets

The training pipeline consolidates **7 Vietnamese text datasets** spanning both handwritten and printed domains:

| Dataset | Domain | Level | Samples | Split Strategy |
|---|---|---|---|---|
| [UIT-HWDB](https://uit-together.github.io/) | Handwritten | Word / Line / Paragraph | ~110k / ~7k / ~1k | Writer-independent (ID 1–229 train, 230–249 val, 250–255 test) |
| Cinnamon AI | Handwritten | Line | ~2,385 | Pre-defined (Data1=val, Data2=train, Private=test) |
| Viet-Wiki-Handwriting | Handwritten | Paragraph | ~5,796 | Random 80/10/10 |
| VinText | Printed | Word | ~43k | Pre-defined train/test/unseen |
| MC-OCR 2021 | Printed | Line | ~6,585 | Pre-defined train/val |
| Anyuuus (PaddleOCR) | Printed | Line | ~28k | Group-split by document ID 80/10/10 |
| **Synthetic Modern** | **Printed** | **Line** | **~30,000** | Random 90/5/5 |

All datasets are normalized to Unicode NFC, filtered for Vietnamese-only characters (10 Unicode ranges), dimension-filtered, and exported to LMDB format. See the [Data Processing README](Data%20Processing/README.md) for full details.

### Downloads

| Resource | Description | Link |
|---|---|---|
| 📁 **Pre-built LMDB** | Ready-to-use training/test databases (skip data processing) | [Google Drive](https://drive.google.com/drive/folders/1ejkt0MrcPWXWn5pDO4QnJ6NWrOzXM0YQ?usp=sharing) |
| 📦 Raw Printed Data | VinText, MC-OCR 2021, Anyuuus, Synthetic_Modern (~2.52 GB) | [Google Drive](https://drive.google.com/file/d/1Z-2pMTMPLuihWYU0pJYIVb1AvbWlJPha/view) |
| 📦 Raw Handwritten Data | UIT-HWDB, Cinnamon AI, Viet-Wiki-Handwriting (~3.09 GB) | [Google Drive](https://drive.google.com/file/d/10W3zPtEGnAXk4XhjHr4motOWxBbuJe3m/view) |

---

## Research Questions & Key Findings

| RQ | Question | Key Finding |
|---|---|---|
| **RQ1** | TrOCR vs. established Vietnamese OCR? | 2.86% CER — 55.5% better than VietOCR, 80.1% better than CRNN |
| **RQ2** | Effectiveness of curriculum training? | 78.1% CER reduction at Stage 1→2a boundary validates pseudo-line progression |
| **RQ3** | EWC against catastrophic forgetting? | Printed CER preserved within 0.44pp over 17 epochs (42:1 improvement ratio) |
| **RQ4** | Post-processing contribution? | 56.5% cumulative CER reduction; PhoBERT shown as net negative (6.6:1 hurt:help) |
| **RQ5** | Accuracy-efficiency trade-offs? | ~2 FPS, 1.4 GB VRAM — favorable for quality-first document digitization |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@thesis{huynh2026vietnamese_trocr,
  title   = {Research and Optimization of Vietnamese Printed and Handwritten 
             Text Recognition System Using TrOCR Architecture},
  author  = {Huynh, Kien Dong Duy and Nguyen, Chi Vy},
  year    = {2026},
  type    = {Information Technology Project},
  school  = {Ton Duc Thang University}
}
```

---

## Acknowledgments

- **[Microsoft TrOCR](https://github.com/microsoft/unilm/tree/master/trocr)** — Base architecture (`trocr-base-stage1`)
- **[VietOCR](https://github.com/pbcquoc/vietocr)** — Vietnamese OCR baseline
- **[docTR](https://github.com/mindee/doctr)** — Text detection engine (DBNet)
- **[PhoBERT](https://github.com/VinAIResearch/PhoBERT)** — Vietnamese language model (evaluated, disabled in production)
- **[UIT-HWDB](https://uit-together.github.io/)** — Vietnamese handwritten text database
- **Google Colab** — GPU compute infrastructure (NVIDIA L4)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.