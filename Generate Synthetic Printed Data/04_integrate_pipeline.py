# -*- coding: utf-8 -*-
"""
04_integrate_pipeline.py
========================
Copy generated synthetic images into the main Data Processing pipeline.

This script:
  1. Reads output/labels.txt
  2. Copies images to raw_data/Printed/Synthetic_Modern/
  3. Creates labels.txt in the target directory
  4. Provides instructions for running the main pipeline

After running this script, you need to:
  1. Add parse_synthetic() to parse_all_datasets.py
  2. Run 03_split.py
  3. Run 04_export_lmdb.py

Chay: python 04_integrate_pipeline.py
"""
import os
import sys
import csv
import shutil
import unicodedata
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT    = Path(__file__).resolve().parent
OUTPUT_DIR      = PROJECT_ROOT / "output"
IMAGE_DIR       = OUTPUT_DIR / "images"
LABELS_PATH     = OUTPUT_DIR / "labels.txt"

# Target in main pipeline
PIPELINE_ROOT   = PROJECT_ROOT.parent / "Data Processing"
TARGET_RAW_DIR  = PIPELINE_ROOT / "raw_data" / "Printed" / "Synthetic_Modern"
TARGET_IMG_DIR  = TARGET_RAW_DIR / "images"

# For direct CSV injection into processed/
PROCESSED_LINE  = PIPELINE_ROOT / "processed" / "line"

# Schema
FIELDNAMES = ["filename", "label", "source", "level", "data_type", "writer_id", "pre_split"]


def main():
    print("=" * 60)
    print("04_integrate_pipeline.py  -  Pipeline integration")
    print("=" * 60)

    # Validate source
    if not LABELS_PATH.exists():
        print(f"ERROR: Labels not found at {LABELS_PATH}")
        print("Run 03_generate_images.py first!")
        sys.exit(1)

    # Read labels
    entries = []
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if "\t" not in line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                entries.append((parts[0].strip(), parts[1].strip()))

    print(f"  Source images: {len(entries)}")
    print(f"  Source dir: {IMAGE_DIR}")

    # ── Option A: Copy to raw_data for full pipeline processing ──────────────
    print(f"\n  Copying to: {TARGET_RAW_DIR}")
    TARGET_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Copy images
    copied = 0
    skipped = 0
    for filename, label in entries:
        src = IMAGE_DIR / filename
        dst = TARGET_IMG_DIR / filename
        if src.exists():
            shutil.copy2(str(src), str(dst))
            copied += 1
        else:
            skipped += 1

    print(f"  Copied: {copied}, Skipped (missing): {skipped}")

    # Create labels.txt in target
    target_labels = TARGET_RAW_DIR / "labels.txt"
    with open(target_labels, "w", encoding="utf-8") as f:
        for filename, label in entries:
            f.write(f"{filename}\t{label}\n")
    print(f"  Labels saved: {target_labels}")

    # ── Option B: Write directly to processed/line/ ──────────────────────────
    print(f"\n  Also creating processed/line/ entries directly ...")

    line_img_dir = PROCESSED_LINE / "images"
    line_img_dir.mkdir(parents=True, exist_ok=True)

    # Copy images with source prefix
    rows = []
    direct_copied = 0
    for filename, label in entries:
        src = IMAGE_DIR / filename
        new_name = f"synthetic_modern_{filename}"
        dst = line_img_dir / new_name

        if src.exists():
            shutil.copy2(str(src), str(dst))
            direct_copied += 1

            label_nfc = unicodedata.normalize("NFC", label)
            rows.append({
                "filename":  new_name,
                "label":     label_nfc,
                "source":    "synthetic_modern",
                "level":     "line",
                "data_type": "printed",
                "writer_id": "synthetic",
                "pre_split": "unassigned",   # 03_split.py will assign 80/10/10
            })

    print(f"  Direct-copied to processed/line/images/: {direct_copied}")

    # Append to labels_master.csv
    master_csv = PROCESSED_LINE / "labels_master.csv"
    write_header = not master_csv.exists()

    with open(master_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"  Appended {len(rows)} rows to: {master_csv}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("INTEGRATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total synthetic samples: {len(rows)}")
    print(f"  Raw data location: {TARGET_RAW_DIR}")
    print(f"  Processed location: {PROCESSED_LINE}")

    print(f"\n  NEXT STEPS:")
    print(f"  1. Re-run data split:")
    print(f"     cd \"{PIPELINE_ROOT}\"")
    print(f"     python scripts/03_split.py")
    print(f"")
    print(f"  2. Re-export LMDB:")
    print(f"     python scripts/04_export_lmdb.py")
    print(f"")
    print(f"  3. New Stage 2a data composition (estimated):")

    # Estimate new composition
    anyuuus_est  = 18116
    mcocr_est    = 5285
    synth_count  = len(rows)
    total_new    = anyuuus_est + mcocr_est + synth_count

    print(f"     anyuuus (historical):  {anyuuus_est:6d}  ({100*anyuuus_est/total_new:.1f}%)")
    print(f"     mc_ocr (receipts):     {mcocr_est:6d}  ({100*mcocr_est/total_new:.1f}%)")
    print(f"     synthetic (modern):    {synth_count:6d}  ({100*synth_count/total_new:.1f}%)")
    print(f"     TOTAL:                 {total_new:6d}  (100%)")


if __name__ == "__main__":
    main()
