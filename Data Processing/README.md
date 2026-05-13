# Data Processing Pipeline

Documentation for the data preprocessing component of the **Vietnamese Handwritten & Printed Text Recognition (HTR/OCR)** project using the TrOCR architecture.

---

## Overview

The pipeline consists of 4 scripts executed sequentially, converting raw data from 7 different sources into LMDB databases ready for TrOCR training.

```
raw_data/  →  [parse]  →  processed/  →  [split]  →  [filter]  →  [export]  →  lmdb/
```

| Script | Role | Input | Output |
|---|---|---|---|
| `01_parse_all_datasets.py` | Collect, normalize, copy images | `raw_data/` | `processed/`, `vocab/` |
| `02_split.py` | Split into train/val/test | `labels_master.csv` | `labels_split.csv` |
| `03_filter_outliers.py` | Filter outliers by dimensions | `labels_split.csv` | `labels_filtered.csv` |
| `04_export_lmdb.py` | Package LMDB for TrOCR | `labels_filtered.csv` + images | `lmdb/` |

**Auxiliary Scripts:**

| Script | Role |
|---|---|
| `05_verify_lmdb.py` | Quick verification of exported LMDB |
| `06_source_based_analysis.py` | Analyze image dimension distribution by source |

---

## Directory Structure

```
Data Processing/
├── raw_data/
│   ├── HandWritten/
│   │   ├── Cinnamon_AI_Dataset/
│   │   │   ├── Data1/              (~15 images – Val)
│   │   │   ├── Data2/              (~1,823 images – Train)
│   │   │   └── Private_Test/       (~549 images – Test)
│   │   ├── UIT_HWDB/
│   │   │   ├── UIT_HWDB_word/
│   │   │   │   ├── train_data/     (writer 1–249)
│   │   │   │   └── test_data/      (writer 250–255)
│   │   │   ├── UIT_HWDB_line/
│   │   │   │   ├── train_data/     (writer 1–249)
│   │   │   │   └── test_data/      (writer 250–255)
│   │   │   └── UIT_HWDB_paragraph/
│   │   │       ├── train_data/     (writer 1–249)
│   │   │       └── test_data/      (writer 250–255)
│   │   └── viet_wiki/
│   │       ├── images/             (~5,796 synthetic images)
│   │       ├── labels.csv
│   │       └── downloadscript.py
│   └── Printed/
│       ├── VinText_Cropped/
│       │   ├── train_images/ (~ 25,794 images)
│       │   ├── test_image/ (~ 7,220 images)
│       │   └── unseen_test_images/ (~ 10,086 images)
│       ├── Vietnamese Receipts MC_OCR 2021/
│       │   ├── text_recognition_mcocr_data/ (~ 6,585 images)
│       │   ├── text_recognition_train_data.txt
│       │   └── text_recognition_val_data.txt
│       ├── anyuuus - Vietnamese OCR with PaddleOCR/
│       │   ├── 23127151/
│       │   │   ├── final_crop/ (~ 6,155 images)
│       │   │   └── rec_gt.txt
│       │   ├── 23127215/
│       │   │   ├── final_crop/ (~ 8,615 images)
│       │   │   └── rec_gt.txt
│       │   ├── 23127263/
│       │   │   ├── final_crop/ (~ 5,653 images)
│       │   │   └── rec_gt.txt
│       │   └── 23127407/
│       │       ├── final_crop/ (~ 7,892 images)
│       │       └── rec_gt.txt
│       └── Synthetic_Modern/
│           ├── images/ (~ 30,000 images)
│           └── labels.txt
├── scripts/
│   ├── 01_parse_all_datasets.py
│   ├── 02_split.py
│   ├── 03_filter_outliers.py
│   ├── 04_export_lmdb.py
│   ├── 05_verify_lmdb.py
│   └── 06_source_based_analysis.py
├── processed/              ← Normalized RGB images + labels CSV
├── lmdb/                   ← Output LMDB databases for TrOCR
│   ├── word_printed/
│   │   ├── train/
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   ├── val/
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   └── test/
│   │       ├── data.mdb
│   │       └── lock.mdb
│   ├── word_handwritten/
│   │   ├── train/
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   ├── val/
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   └── test/
│   │       ├── data.mdb
│   │       └── lock.mdb
│   ├── line_handwritten/
│   │   ├── test/
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   ├── train/
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   └── val/
│   │       ├── data.mdb
│   │       └── lock.mdb
│   └── line_printed/
│       ├── test/
│       │   ├── data.mdb
│       │   └── lock.mdb
│       ├── train/
│       │   ├── data.mdb
│       │   └── lock.mdb
│       └── val/
│           ├── data.mdb
│           └── lock.mdb
└── vocab/
    └── vietnamese_vocab.txt
```

---

## Installation

```bash
pip install Pillow tqdm scikit-learn lmdb
```

---

## Dataset Download

Before running the pipeline, you need to download the raw datasets and place them in the correct directory structure.

### Download Links

| Dataset | Contents | Size | Link |
|---|---|---|---|
| **Printed** | VinText, MC-OCR 2021, Anyuuus, Synthetic_Modern | ~2.52 GB | [Google Drive](https://drive.google.com/file/d/1Z-2pMTMPLuihWYU0pJYIVb1AvbWlJPha/view) |
| **Handwritten** | UIT-HWDB, Cinnamon AI, Viet-Wiki-Handwriting | 3.09 GB | [Google Drive](https://drive.google.com/file/d/10W3zPtEGnAXk4XhjHr4motOWxBbuJe3m/view) |

### Download Instructions

**Option 1 — Manual download (browser):**

1. Click the Google Drive links above
2. Click **Download** (you may need to confirm for large files)
3. Extract the downloaded archives

**Option 2 — Command line (using `gdown`):**

```bash
pip install gdown

# Download Printed dataset
gdown --fuzzy "https://drive.google.com/file/d/1Z-2pMTMPLuihWYU0pJYIVb1AvbWlJPha/view" -O printed_data.zip

# Download Handwritten dataset
gdown --fuzzy "https://drive.google.com/file/d/10W3zPtEGnAXk4XhjHr4motOWxBbuJe3m/view" -O handwritten_data.zip
```

### Placement

After extracting, place the contents into the `raw_data/` directory so the structure matches:

```
Data Processing/
└── raw_data/
    ├── HandWritten/          ← from handwritten_data archive
    │   ├── Cinnamon_AI_Dataset/
    │   ├── UIT_HWDB/
    │   └── viet_wiki/
    └── Printed/              ← from printed_data archive
        ├── VinText_Cropped/
        ├── Vietnamese Receipts MC_OCR 2021/
        ├── anyuuus - Vietnamese OCR with PaddleOCR/
        └── Synthetic_Modern/
```

> **Important:** The folder names must match exactly as shown above — the parsing scripts use hardcoded paths relative to `raw_data/`. After placing the data, verify with:
>
> ```bash
> # Should show HandWritten/ and Printed/
> ls raw_data/
>
> # Should show subdatasets
> ls raw_data/HandWritten/
> ls raw_data/Printed/
> ```

---

## Usage Guide

### Step 1 — Parse all datasets

```bash
# Run all datasets
python scripts/01_parse_all_datasets.py all

# Or run individual datasets for debugging
python scripts/01_parse_all_datasets.py uit
python scripts/01_parse_all_datasets.py cinnamon
python scripts/01_parse_all_datasets.py wiki
python scripts/01_parse_all_datasets.py vintext
python scripts/01_parse_all_datasets.py mcocr
python scripts/01_parse_all_datasets.py anyuuus
python scripts/01_parse_all_datasets.py synthetic
```

> **WARNING**: The script automatically deletes all `labels_master.csv` files before running to prevent data duplication. `save_vocab()` only records characters from the parsers executed in the current invocation — always run with `all` to get the complete vocabulary.

### Step 2 — Split into train/val/test

```bash
python scripts/02_split.py
```

### Step 3 — Filter outliers + language

```bash
python scripts/03_filter_outliers.py
```

The script performs **2 filtering steps** for each level (word, line):

**Step 3a — Language filtering (runs first):**
Removes labels containing characters outside the valid Vietnamese character set:

| Script Removed | Examples | Primary Source |
|---|---|---|
| CJK (Chinese/Japanese/Korean) | 漢字, カタカナ | Synthetic_Modern, Anyuuus |
| Khmer/Cambodian | ក, ន, រ | Synthetic_Modern |
| Cyrillic (Russian) | Т, У | Synthetic_Modern |
| IPA Phonetic | ɑ, ə, ɛ | Synthetic_Modern |
| Fullwidth Forms | ，(U+FF0C) | Anyuuus |

Valid Vietnamese characters retained:
- ASCII printable (space – tilde)
- Latin-1 Supplement, Latin Extended-A/B (Ă, Đ, Ơ, Ư, etc.)
- Latin Extended Additional (Ạ–ỹ: 134 Vietnamese diacritical characters)
- General Punctuation (–, —, ', ', ", ", …)
- Superscripts (², ³), Currency (₫)

**Step 3b — Image dimension filtering:**

| Level | Filter | Value |
|---|---|---|
| Word | min_width, min_height | 10px |
| Word | max_height | 300px |
| Word | aspect ratio (W/H) | 0.3 – 25.0 |
| Line | min_width | 32px |
| Line | min_height | 16px |
| Line | max_height | 384px |
| Line | max_width | 3000px |
| Line | min aspect ratio (W/H) | 1.0 |

> **Note**: Language filtering runs **before** dimension filtering to avoid unnecessary image I/O operations.

### Step 4 — Export to LMDB

```bash
python scripts/04_export_lmdb.py
```

> **Note**: The script automatically prioritizes `labels_filtered.csv`. If Step 3 has not been run, it falls back to `labels_split.csv`.

---

## Datasets

### Handwritten

| Dataset | Level | Sample Count | Notes |
|---|---|---|---|
| UIT_HWDB | word / line / paragraph | ~110k / ~7k / ~1k | 249 writers, includes writer ID |
| Cinnamon AI | line | ~2,385 | Real-world addresses, difficult cursive handwriting |
| Viet-Wiki-Handwriting | paragraph | ~5,796 | Synthetic from Vietnamese Wikipedia |

### Printed

| Dataset | Level | Notes |
|---|---|---|
| VinText_Cropped | word | Printed text from real-world images |
| Vietnamese Receipts MC_OCR 2021 | line | Invoices and receipts |
| Anyuuus – PaddleOCR | line | Scanned documents (historical language) |
| **Synthetic_Modern** | **line** | **Synthetic printed, 46k modern corpus** |

---

## CSV Schema

Each file — `labels_master.csv` (output of Script 1), `labels_split.csv` (output of Script 2), and `labels_filtered.csv` (output of Script 3) — has the following structure:

| Column | Type | Description |
|---|---|---|
| `filename` | string | Image filename in `processed/{level}/images/` |
| `label` | string | NFC-normalized text label |
| `source` | string | Source identifier (`uit_word`, `cinnamon_d2`, `mcocr`, ...) |
| `level` | string | Level: `word` / `line` / `paragraph` |
| `data_type` | string | `handwritten` or `printed` |
| `writer_id` | string | Writer ID (if available), used for writer-independent split |
| `pre_split` | string | Split hint from original directory structure (see table below) |
| `final_split` | string | *(Present in labels_split.csv and labels_filtered.csv)* `train` / `val` / `test` |

### `pre_split` Values

| Value | Source | Handling in Script 2 |
|---|---|---|
| `train` | Cinnamon Data2, VinText train, MC_OCR train | Kept as-is |
| `val` | Cinnamon Data1, VinText test | Kept as-is |
| `test` | Cinnamon Private_Test, UIT test_data, VinText unseen | Kept as-is |
| `train_pool` | UIT_HWDB train_data | Writer 1–229 → train, Writer 230–249 → val |
| `unassigned` | Viet_Wiki | Random split 80/10/10 |
| `anyuuus_pool` | Anyuuus | Group-split by Document ID 80/10/10 |
| `synthetic_pool` | Synthetic_Modern | Random split 90/5/5 |

---

## Data Splitting Strategy

### Writer-Independent Split (UIT_HWDB)

Each writer appears in exactly one split (train, val, or test). This ensures the model is evaluated on handwriting from writers never seen during training.

```
Writer ID  1 – 229  →  train  (91.6%)
Writer ID 230 – 249  →  val   (8.0%)
Writer ID 250 – 255  →  test  (original author's test_data split)
```

### Group Split (Anyuuus)

All text lines cropped from the same source document are kept within the same split. This prevents the scenario where the model sees other lines from the same document during training but is tested on a remaining line from that same document (document-level data leakage).

### Random Split (Viet_Wiki)

Synthetic data has no writer identity information; `train_test_split(random_state=42)` is used to split 80/10/10. `random_state=42` ensures fully reproducible results.

### Random Split (Synthetic_Modern)

Clean synthetic data with no data leakage risk. A **90/5/5** ratio (instead of 80/10/10) is used to maximize the amount of modern linguistic data in training, diluting the historical language bias from the Anyuuus dataset.

---

## Data Filtering

Script `03_filter_outliers.py` applies **2 filters** before LMDB export:

### Language Filtering (non-Vietnamese character removal)

Removes the entire label+image pair if the label contains **any character** outside the valid Vietnamese character set. The filter uses a whitelist of 10 Unicode ranges (see `ALLOWED_RANGES` in the script), covering all 256 characters in `final_vietnamese_vocab.txt`.

Common scripts removed:
- **CJK** (~109 characters): Chinese Hán tự, Japanese Kanji — from Anyuuus (historical Chữ Nôm text) and Synthetic_Modern
- **Khmer** (~13 characters): from Synthetic_Modern corpus
- **IPA** (~8 characters): International Phonetic Alphabet symbols ɑ, ə, ɛ
- **Cyrillic** (2 characters): Т, У
- **Fullwidth** (1 character): ， (U+FF0C, fullwidth comma)

### Image Dimension Filtering

- **Word level**: Removes images < 10px, > 300px height, aspect ratio outside 0.3–25.0
- **Line level**: Removes images < 32px width, < 16px height, > 384px height, > 3000px width, square/portrait images (aspect < 1.0)

---

## LMDB Structure

Each LMDB database at `lmdb/{level}/{split}/` stores key-value pairs following the convention of [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark):

```
image-00000001  →  PNG image bytes (RGB, read directly from disk)
label-00000001  →  UTF-8 encoded text label
image-00000002  →  ...
label-00000002  →  ...
num-samples     →  total number of valid samples (string)
```

Both `word` and `line` levels are separated into two distinct LMDB databases by `data_type` (`printed` / `handwritten`). This allows flexible control over data ratios and fine-tuning schedules for each text type.

The `paragraph` level serves the DBNet text detector (uses pre-trained weights, no LMDB export needed).

### LMDB Level → Source Image Directory Mapping

| LMDB Level | Source CSV | Image Directory |
|---|---|---|
| `word_printed` | `processed/word/labels_filtered.csv` | `processed/word/images/` |
| `word_handwritten` | `processed/word/labels_filtered.csv` | `processed/word/images/` |
| `line_printed` | `processed/line/labels_filtered.csv` | `processed/line/images/` |
| `line_handwritten` | `processed/line/labels_filtered.csv` | `processed/line/images/` |

### Usage in TrOCR Training

```
Stage 1a - Pre-warm encoder (Printed)    : lmdb/word_printed/train/
Stage 1b - Pre-warm encoder (Handwritten): lmdb/word_handwritten/train/
Stage 2a - Printed fine-tune             : lmdb/line_printed/train/
Stage 2b - Handwritten adapt             : lmdb/line_handwritten/train/
```

---

## Vocabulary

There are 2 vocabulary files serving different purposes:

### `vocab/vietnamese_vocab.txt` (raw, ~409 characters)

**Automatically generated** by Script 01 — contains **all** characters appearing in the data, including foreign characters (CJK, Khmer, Cyrillic, IPA). Used for inspection, debugging, and data analysis.

> **WARNING**: `save_vocab()` only records characters from the parsers executed in the current invocation. Always run `01_parse_all_datasets.py all` to get the complete vocabulary.

### `final_vietnamese_vocab.txt` (curated, 256 characters) ← **Used for TrOCR tokenizer**

**Manually curated** file — contains only valid Vietnamese characters. This is the official file for configuring the character-level tokenizer for TrOCR.

**Statistics:**

| Group | Character Count | Details |
|---|---|---|
| ASCII printable | 95 | Space, digits 0–9, A–Z, a–z, punctuation |
| Vietnamese diacritics | 134 | 67 uppercase + 67 lowercase (Ạ–ỹ, Ă–ặ, Đ–đ, Ơ–ợ, Ư–ự) |
| Other symbols | 27 | °, ², ³, –, —, ', ', ", ", ₫, ⁰, ⁴–⁹, etc. |
| **Total** | **256** | |

**Validation checks performed:**
- ✅ 134/134 Vietnamese diacritics complete (all vowel + tone mark combinations)
- ✅ Digits 0–9, ASCII A–Z/a–z complete
- ✅ Currency symbol ₫ (Vietnamese đồng) present
- ✅ No CJK, Khmer, Thai, Cyrillic, or IPA characters
- ✅ All 256 characters fall within the `ALLOWED_RANGES` of `03_filter_outliers.py`

**Usage for TrOCR tokenizer:**
```
Vocab size: 256 characters + 4 special tokens = 260
Special tokens: <s> (BOS), </s> (EOS), <pad>, <unk>
```

The tokenizer replaces the RoBERTa BPE tokenizer (50,265 tokens) with a character-level tokenizer of 260 tokens — reducing the embedding layer from ~38M params to ~0.2M params.

---

## Technical Notes

**Unicode NFC** — All labels are normalized to Unicode NFC before saving. Vietnamese can represent tone marks in 2 ways (NFC: `ộ` = 1 codepoint; NFD: `ộ` = 3 codepoints). Without normalization, CER would be miscalculated even when the model recognizes text correctly from a visual standpoint.

**Reproducibility** — All random splits use `random_state=42` (sklearn). Re-running at any time produces identical split results.

**Duplicate Prevention** — Script `01_parse_all_datasets.py` automatically deletes all `labels_master.csv` files before running, preventing duplicates on re-execution.

**No Double Encoding** — `04_export_lmdb.py` reads PNG image bytes directly from disk instead of re-encoding through PIL, saving ~220k decode+encode cycles. A validation step (first 100 images) checks format consistency.

**Windows vs Linux** — LMDB on Linux automatically expands `map_size`. On Windows, `map_size` must be declared large enough upfront. The script calculates `map_size` based on actual file sizes on disk multiplied by `MAP_SIZE_SAFETY_FACTOR` (default 3.0x), combined with an automatic retry mechanism that doubles the size if an `MDB_MAP_FULL` error is encountered.
