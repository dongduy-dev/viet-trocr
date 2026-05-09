# Vietnamese OCR — Web Interface

Interactive demo of the End-to-End Vietnamese OCR pipeline for thesis defense.

## Architecture

```
Step 1: Preprocessor     → CLAHE + Hough Deskew
Step 2: TextDetector     → DBNet++ (mmocr)
Step 3: LineSegmenter    → DBSCAN clustering
Step 4: TextRecognizer   → TrOCR (fine-tuned)
Step 5: PostProcessor    → RegexSanitizer + AddressCorrector + PhoBERT(optional)
```

## Quick Start (Colab)

```python
# Cell 1: Install dependencies
!pip install gradio transformers torch Pillow opencv-python-headless scikit-learn pyvi jiwer

# Cell 2: Install mmocr for DBNet++ text detection
!pip install -U openmim
!mim install mmengine mmcv mmdet mmocr

# Cell 3: Upload code files to /content/trocr_viet/
# Make sure eval_utils.py, address_corrector.py, data/ are present

# Cell 4: Run the interface
%cd /content/trocr_viet/interface
!python app.py \
    --model_path /content/drive/MyDrive/OCR/checkpoints/final_model \
    --share
```

The `--share` flag creates a public URL you can share with your thesis committee.

## File Structure

```
interface/
├── app.py                    # Gradio UI entry point
├── pipeline.py               # E2E pipeline orchestrator
├── modules/
│   ├── __init__.py
│   ├── preprocessor.py       # Step 1: CLAHE + Deskew
│   ├── text_detector.py      # Step 2: DBNet++
│   ├── line_segmenter.py     # Step 3: DBSCAN
│   ├── text_recognizer.py    # Step 4: TrOCR
│   └── post_processor.py     # Step 5: Post-processing
├── requirements.txt
└── README.md
```

## UI Features

- **Upload** any document image (scanned or camera photo)
- **View intermediate steps**: preprocessing, bounding boxes, cropped lines
- **Per-line recognition results** in a table
- **Toggle PhoBERT** on/off for comparison
- **Copy** final text with one click
- **Timing breakdown** per pipeline step

## VRAM Usage

| Component | VRAM (FP16) |
|---|---|
| DBNet++ (ResNet50) | ~200MB |
| TrOCR (ViT + Decoder) | ~600MB |
| PhoBERT (optional) | ~500MB |
| **Total** | **~1.3GB** |

Safe for T4 (16GB) and L4 (24GB).
