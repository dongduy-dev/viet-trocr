# Synthetic Printed Vietnamese Data Generation Pipeline

Generate 30,000+ synthetic printed Vietnamese text line images for TrOCR training.

## Why This Exists

The Stage 2a training data is **77% historical Vietnamese text** (`anyuuus` dataset),
which biases the decoder toward generating feudal-era prose. This pipeline generates
**modern Vietnamese** printed text images to dilute that bias.

## Quick Start

```bash
cd Generate_Synthetic_Printed_Data

# Full pipeline (30k images, ~30-60 min)
python run_all.py

# Quick test (1k images, ~3 min)
python run_all.py --count 1000

# Skip font download if already done
python run_all.py --skip-fonts
```

## Pipeline Steps

| Step | Script | Time | Description |
|------|--------|------|-------------|
| 1 | `01_build_corpus.py` | ~5 sec | Build Vietnamese text corpus (~1500+ unique sentences) |
| 2 | `02_download_fonts.py` | ~2-5 min | Download 14 Google Fonts families |
| 3 | `03_generate_images.py` | ~20-40 min | Render text images with augmentation |
| 4 | `04_integrate_pipeline.py` | ~5-10 min | Copy to Data Processing pipeline |

**Total estimated time: ~30-60 minutes** (depending on machine speed)

## After Running

```bash
cd "../Data Processing"
python scripts/03_split.py          # Re-split with new data
python scripts/04_export_lmdb.py    # Re-export LMDB
```

## New Stage 2a Composition (Target)

| Source | Samples | Percentage |
|--------|---------|------------|
| anyuuus (historical) | ~18,116 | ~34% |
| mc_ocr (receipts) | ~5,285 | ~10% |
| **synthetic (modern)** | **~30,000** | **~56%** |
| **TOTAL** | **~53,401** | **100%** |

## Output Structure

```
Generate_Synthetic_Printed_Data/
├── config.yaml              # Settings
├── run_all.py               # Master runner
├── 01_build_corpus.py       # Corpus builder
├── 02_download_fonts.py     # Font downloader
├── 03_generate_images.py    # Image generator
├── 04_integrate_pipeline.py # Pipeline integrator
├── corpus/
│   └── modern_vietnamese.txt
├── fonts/
│   └── *.ttf
└── output/
    ├── images/
    │   └── synth_000000.png ...
    └── labels.txt
```

## Corpus Domains

The built-in corpus covers 7 domains:
- **Legal**: Laws, decrees, government documents
- **News**: Current events, economics, technology
- **Education**: Universities, degrees, academic terms
- **Business**: Contracts, financial reports, company info
- **Everyday**: Greetings, food, travel, culture
- **Sci/Tech**: AI, programming, engineering
- **Numeric**: Prices, dates, phone numbers, measurements

Plus generated Vietnamese addresses and name records.

## Requirements

```bash
pip install Pillow numpy
```
