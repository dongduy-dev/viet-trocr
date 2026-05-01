# -*- coding: utf-8 -*-
"""
run_all.py
==========
Master script - Runs the entire synthetic data generation pipeline.

Usage:
  python run_all.py              # Full pipeline (30k images)
  python run_all.py --count 5000 # Quick test (5k images)
  python run_all.py --skip-fonts # Skip font download (if already done)

Estimated time (on a modern laptop):
  Step 1 (corpus):     ~5 seconds
  Step 2 (fonts):      ~2-5 minutes (download)
  Step 3 (generate):   ~20-40 minutes for 30k images
  Step 4 (integrate):  ~5-10 minutes (copy files)
  ─────────────────────────────────────
  TOTAL:               ~30-60 minutes
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(script: str, args: list[str] = None, step_name: str = ""):
    """Run a Python script and time it."""
    cmd = [sys.executable, str(PROJECT_ROOT / script)]
    if args:
        cmd.extend(args)

    print(f"\n{'#' * 60}")
    print(f"# STEP: {step_name}")
    print(f"# Script: {script}")
    print(f"{'#' * 60}\n")

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode}")
        print(f"Elapsed: {elapsed:.1f}s")
        sys.exit(1)

    minutes = elapsed / 60
    print(f"\n[OK] {step_name} completed in {elapsed:.1f}s ({minutes:.1f} min)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run the full synthetic data generation pipeline"
    )
    parser.add_argument(
        "--count", type=int, default=30000,
        help="Number of images to generate (default: 30000)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for generation (default: 4)"
    )
    parser.add_argument(
        "--skip-fonts", action="store_true",
        help="Skip font download (use existing fonts)"
    )
    parser.add_argument(
        "--skip-integrate", action="store_true",
        help="Skip integration step (generate images only)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SYNTHETIC PRINTED DATA GENERATION PIPELINE")
    print("=" * 60)
    print(f"  Target count: {args.count} images")
    print(f"  Workers: {args.workers}")
    print(f"  Skip fonts: {args.skip_fonts}")
    print(f"  Skip integrate: {args.skip_integrate}")
    print(f"  Project root: {PROJECT_ROOT}")

    total_start = time.time()
    timings = {}

    # Step 1: Build corpus
    t = run_step("01_build_corpus.py", step_name="Build Vietnamese corpus")
    timings["Corpus"] = t

    # Step 2: Download fonts (optional skip)
    if args.skip_fonts:
        font_dir = PROJECT_ROOT / "fonts"
        n_fonts = len(list(font_dir.glob("*.ttf"))) if font_dir.exists() else 0
        print(f"\n[SKIP] Font download (found {n_fonts} existing fonts)")
        timings["Fonts"] = 0
    else:
        t = run_step("02_download_fonts.py", step_name="Download Vietnamese fonts")
        timings["Fonts"] = t

    # Step 3: Generate images
    gen_args = ["--count", str(args.count), "--workers", str(args.workers)]
    t = run_step("03_generate_images.py", args=gen_args, step_name="Generate synthetic images")
    timings["Generation"] = t

    # Step 4: Integrate into pipeline
    if args.skip_integrate:
        print(f"\n[SKIP] Pipeline integration")
        timings["Integration"] = 0
    else:
        t = run_step("04_integrate_pipeline.py", step_name="Integrate into Data Processing pipeline")
        timings["Integration"] = t

    # Summary
    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Timing breakdown:")
    for step, t in timings.items():
        print(f"    {step:20s}: {t/60:6.1f} min")
    print(f"    {'─' * 32}")
    print(f"    {'TOTAL':20s}: {total_elapsed/60:6.1f} min")

    print(f"\n  Output directory: {PROJECT_ROOT / 'output'}")
    print(f"  Images: {PROJECT_ROOT / 'output' / 'images'}")
    print(f"  Labels: {PROJECT_ROOT / 'output' / 'labels.txt'}")

    if not args.skip_integrate:
        print(f"\n  REMAINING STEPS:")
        print(f"  1. cd \"{PROJECT_ROOT.parent / 'Data Processing'}\"")
        print(f"  2. python scripts/03_split.py")
        print(f"  3. python scripts/04_export_lmdb.py")
        print(f"  4. Upload new LMDB to Google Drive for Colab training")


if __name__ == "__main__":
    main()
