# Vietnamese OCR — End-to-End Inference Interface

Interactive web interface for the Vietnamese OCR end-to-end system.

---

## Pipeline Architecture

```
           ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  Input ──▶│  Step 1      │ ──▶ │  Step 2      │ ──▶ │  Step 3      │
  Image    │  CLAHE +     │     │  DBNet       │     │  DBSCAN      │
           │  Hough Deskew│     │  (docTR)     │     │  Clustering  │
           └──────────────┘     └──────────────┘     └──────┬───────┘
                                                            │
           ┌──────────────┐     ┌──────────────┐           │
  Output◀──│  Step 5      │ ◀── │  Step 4      │ ◀─────────┘
  Text     │  Sanitizer + │     │  TrOCR       │   Cropped line images
           │  Address Corr│     │  (batched)   │
           └──────────────┘     └──────────────┘
```

| Step | Module | Function | GPU |
|---|---|---|---|
| 1 | `preprocessor.py` | CLAHE (LAB L-channel) + Hough line deskew | ✗ |
| 2 | `text_detector.py` | Text detection via docTR `db_resnet50` (DBNet family) | ✓ |
| 3 | `line_segmenter.py` | DBSCAN clustering on y-center → reading order sort | ✗ |
| 4 | `text_recognizer.py` | TrOCR (ViT+Decoder) FP16 batched inference | ✓ |
| 5 | `post_processor.py` | RegexSanitizer + AddressCorrector + PhoBERT (optional) | ✗/✓ |

---

## Directory Structure

```
interface/
├── app.py                    # Entry point — Gradio web UI
├── pipeline.py               # OCRPipeline orchestrator + PipelineResult dataclass
├── modules/
│   ├── __init__.py
│   ├── preprocessor.py       # Step 1: CLAHE contrast + Hough deskew
│   ├── text_detector.py      # Step 2: DBNet text detection (docTR)
│   ├── line_segmenter.py     # Step 3: DBSCAN line grouping
│   ├── text_recognizer.py    # Step 4: TrOCR recognition (batched)
│   └── post_processor.py     # Step 5: Sanitizer + Address + PhoBERT
├── requirements.txt
└── README.md
```

**External dependencies** (imported from parent directory via sys.path auto-detection):

| File | Source | Used by |
|---|---|---|
| `eval_utils.py` | `Fine tuning/code/` | `post_processor.py` → RegexSanitizer, PhoBERTCorrector |
| `address_corrector.py` | `Fine tuning/code/` | `post_processor.py` → AddressCorrector |
| `data/vietnam_gazetteer.json` | `Fine tuning/code/data/` | AddressCorrector (63 provinces, 705 districts, 10,599 wards) |

---

## Module Details

### Step 1 — Preprocessor (`preprocessor.py`)

**CLAHE** (Contrast Limited Adaptive Histogram Equalization):
- Converts image to LAB color space
- Applies CLAHE on L-channel (`clip_limit=2.0`, `tile_size=8×8`)
- Converts back to RGB

**Deskew** (Hough Line Transform):
- Edge detection via Canny → Hough Lines → median angle → `cv2.warpAffine`
- Only rotates if angle > 0.5° (avoids unnecessary rotation)
- Fills background with median pixel value

### Step 2 — Text Detector (`text_detector.py`)

- **Model**: docTR `db_resnet50` — same DBNet/DBNet++ architecture family
- **Output**: `{"words": ndarray(N, 5)}` — relative coords `[xmin, ymin, xmax, ymax, conf]`
- **Confidence filter**: Discards detections with confidence < 0.3
- **Fallback**: If docTR is unavailable → entire image treated as 1 text region (pre-cropped mode)

### Step 3 — Line Segmenter (`line_segmenter.py`)

- **Clustering**: DBSCAN on y-center of bounding boxes
- **Adaptive eps**: `median_bbox_height × 0.5` — self-adjusts to text size
- **Reading order**: Lines sorted top→bottom, boxes sorted left→right within each line
- **Crop**: Merge boxes in same line → crop + 4px padding from original image

### Step 4 — Text Recognizer (`text_recognizer.py`)

- **Model**: Fine-tuned Vietnamese TrOCR (`VisionEncoderDecoderModel`)
- **Resize**: `resize_for_vit()` — scale-to-fit 128×1536, median padding, never crops
- **Inference**: Batched (default 8), FP16, greedy decode (`repetition_penalty=1.2`)
- **Decode**: `safe_batch_decode()` — handles RoBERTa `Ġ` space marker + NFC normalization
- **Meta-device fix**: Materializes sinusoidal positional embeddings (TrOCR bug)

### Step 5 — Post Processor (`post_processor.py`)

Runs 3 stages sequentially:

| Stage | Component | Default | Function |
|---|---|---|---|
| 5a | RegexSanitizer | **Always on** | Collapse hallucinations: `....` ×80 → `...` |
| 5b | AddressCorrector v2 | On | Fuzzy-match geographic names (province-scoped, skip-if-valid) |
| 5c | PhoBERTCorrector | **Off** | MLM rescoring — empirically shown as net negative (6.6:1 hurt:help) |

**Import path auto-detection**: Automatically locates `eval_utils.py` at `../../` (Colab) or `../../Fine tuning/code/` (local).

---

## VRAM Usage

| Component | VRAM (FP16) |
|---|---|
| DBNet `db_resnet50` (docTR) | ~200 MB |
| TrOCR ViT + Decoder | ~600 MB |
| PhoBERT-base (if enabled) | ~500 MB |
| Batch of 8 images 128×1536 | ~100 MB |
| **Total (without PhoBERT)** | **~900 MB** |
| **Total (with PhoBERT)** | **~1.4 GB** |

Safe for T4 (16 GB) and L4 (24 GB).

---

## Colab Deployment

The interface can be launched using the **"INTERFACE"** section of `Viet_TrOCR.ipynb`
(located at the repository root). The cells below are pre-configured in the notebook.

### Google Drive Setup

```
My Drive/OCR/
├── checkpoints/final_model/    ← Exported TrOCR model
├── code/                       ← Training/evaluation code
│   ├── eval_utils.py
│   ├── address_corrector.py
│   └── data/vietnam_gazetteer.json
└── interface/                  ← This directory — upload to Drive
    ├── app.py
    ├── pipeline.py
    └── modules/...
```

### Cell 1 — Mount & Copy

```python
from google.colab import drive
drive.mount('/content/drive')

!cp -r "/content/drive/MyDrive/OCR/code" "/content/trocr_viet"
!cp -r "/content/drive/MyDrive/OCR/interface" "/content/trocr_viet/interface"
!touch /content/trocr_viet/data/__init__.py
!touch /content/trocr_viet/core/__init__.py
print("✅ Done"); !ls /content/trocr_viet/interface/modules/
```

### Cell 2 — Install Dependencies

```python
!pip install -q gradio transformers Pillow opencv-python-headless \
    scikit-learn pyvi jiwer
!pip install -q "python-doctr[torch]"
```

### Cell 3 — Launch

```python
%cd /content/trocr_viet/interface
!python app.py \
    --model_path /content/drive/MyDrive/OCR/checkpoints/final_model \
    --share
```

`--share` creates a public URL (72h) like `https://xxxxx.gradio.live`.

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `/content/.../final_model` | Path to HuggingFace model directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--port` | `7860` | Gradio server port |
| `--share` | `True` | Create public Gradio link |

---

## Web Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│  🔍 Vietnamese OCR System                               │
├──────────┬──────────────────────────────────────────────┤
│ Upload   │  Model Path / Batch Size / PhoBERT ☐         │
│ Image    │  Address Correction ☐                        │
│          │  [▶ Run OCR Pipeline]                        │
├──────────┴──────────────────────────────────────────────┤
│  📝 Recognized Text (copy button)                       │
│  ⏱ Timing: ① 0.3s │ ② 1.2s │ ③ 0.01s │ ④ 0.5s │ ⑤ 0s│
├─────────────────────────────────────────────────────────┤
│  ▸ ① Preprocessing   — Before/After + deskew angle     │
│  ▸ ② Text Detection  — Bounding boxes on image         │
│  ▸ ③ Line Segmentation — Gallery of cropped lines       │
│  ▸ ④ Recognition     — Table: Line# | Recognized Text  │
│  ▸ ⑤ Post-processing — Raw → Sanitized → Corrected     │
└─────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: doctr` | docTR not installed | `pip install "python-doctr[torch]"` |
| `Detected 1 text region` (whole image) | docTR failed to load or image too simple | Check `[TextDetector]` log, try multi-line image |
| `eval_utils not found` | `eval_utils.py` not in search path | Verify Drive structure matches setup section |
| `Gazetteer not found` | `vietnam_gazetteer.json` missing | Copy to `data/` alongside `eval_utils.py` |
| `CUDA out of memory` | Batch size too large | Reduce batch size in UI (8 → 4) |
| `Model path not found` | Wrong path or Drive not mounted | Use `os.path.isdir()` to verify (Drive shortcuts fail with `ls`) |

---

## Programmatic API

Use the pipeline directly without the Gradio UI:

```python
import sys, os
sys.path.insert(0, "/content/trocr_viet/interface")

from pipeline import OCRPipeline
from PIL import Image

pipeline = OCRPipeline(
    model_path="/content/drive/MyDrive/OCR/checkpoints/final_model",
    device="cuda",
    batch_size=8,
    enable_phobert=False,
    enable_address=True,
)

image = Image.open("document.jpg")
result = pipeline.run(image)

print(result.full_text)           # Final recognized text
print(result.num_lines)           # Number of detected lines
print(result.timing)              # Per-step timing breakdown
print(result.post_processed)      # {raw, sanitized, corrected, phobert_corrected}

# Access intermediate results
result.preprocessed_image.save("step1.png")
result.annotated_image.save("step2_detections.png")
for i, crop in enumerate(result.line_images):
    crop.save(f"line_{i}.png")
```
