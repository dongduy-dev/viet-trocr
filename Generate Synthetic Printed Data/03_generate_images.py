# -*- coding: utf-8 -*-
"""
03_generate_images.py
=====================
Generate synthetic printed Vietnamese text line images using Pillow.

For each sentence from the corpus:
  1. Pick a random font
  2. Pick a random font size
  3. Render text on a background
  4. Apply augmentations (blur, noise, rotation)
  5. Save as PNG with tab-separated label file

Output:
  output/images/     - PNG images
  output/labels.txt  - "filename\\tlabel" for each image

Chay: python 03_generate_images.py [--count 30000]
"""
import os
import sys
import math
import random
import argparse
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_PATH  = PROJECT_ROOT / "corpus" / "modern_vietnamese.txt"
FONT_DIR     = PROJECT_ROOT / "fonts"
OUTPUT_DIR   = PROJECT_ROOT / "output"
IMAGE_DIR    = OUTPUT_DIR / "images"

# ── Default generation settings ──────────────────────────────────────────────
DEFAULT_COUNT      = 30000
IMAGE_HEIGHT       = 64
FONT_SIZE_MIN      = 28
FONT_SIZE_MAX      = 48
PADDING_X          = 12
PADDING_Y          = 8
BLUR_PROB          = 0.15
BLUR_RADIUS_MAX    = 1.5
NOISE_PROB         = 0.12
NOISE_STD          = 10
ROTATION_PROB      = 0.20
ROTATION_MAX_DEG   = 2.0
BG_COLOR_MIN       = 225
BG_COLOR_MAX       = 255
FG_COLOR_MIN       = 0
FG_COLOR_MAX       = 60

# ── Limit image width to keep aspect ratio reasonable ────────────────────────
MIN_WIDTH = 64
MAX_WIDTH = 2400


def load_corpus() -> list[str]:
    """Load the Vietnamese corpus."""
    if not CORPUS_PATH.exists():
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        print(f"Run 01_build_corpus.py first!")
        sys.exit(1)

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"  Loaded corpus: {len(lines)} sentences")
    return lines


def load_fonts() -> list[Path]:
    """Load all .ttf font paths."""
    if not FONT_DIR.exists():
        print(f"ERROR: Font directory not found at {FONT_DIR}")
        print(f"Run 02_download_fonts.py first!")
        sys.exit(1)

    fonts = sorted(FONT_DIR.glob("*.ttf"))
    if not fonts:
        print(f"ERROR: No .ttf files found in {FONT_DIR}")
        sys.exit(1)

    print(f"  Loaded fonts: {len(fonts)} .ttf files")
    return fonts


def render_one_image(
    text: str,
    font_path: str,
    font_size: int,
    bg_color: int,
    fg_color: int,
    apply_blur: bool,
    blur_radius: float,
    apply_noise: bool,
    noise_std: int,
    apply_rotation: bool,
    rotation_deg: float,
) -> Image.Image | None:
    """Render a single text line image. Returns PIL Image or None on error."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None

    # Measure text bounding box
    try:
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        return None

    if text_w <= 0 or text_h <= 0:
        return None

    # Calculate image size
    img_w = text_w + 2 * PADDING_X
    img_h = text_h + 2 * PADDING_Y

    # Clamp width
    if img_w > MAX_WIDTH:
        return None  # Text too long for this font size
    img_w = max(img_w, MIN_WIDTH)

    # Create image
    bg = (bg_color, bg_color, bg_color)
    fg = (fg_color, fg_color, fg_color)
    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    # Draw text centered vertically
    x = PADDING_X - bbox[0]
    y = PADDING_Y - bbox[1]
    draw.text((x, y), text, fill=fg, font=font)

    # ── Augmentations ────────────────────────────────────────────────────
    # Rotation
    if apply_rotation and abs(rotation_deg) > 0.1:
        img = img.rotate(
            rotation_deg,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=bg,
        )

    # Gaussian Blur
    if apply_blur and blur_radius > 0.1:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Gaussian Noise
    if apply_noise and noise_std > 0:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Resize to fixed height (maintaining aspect ratio)
    w, h = img.size
    if h != IMAGE_HEIGHT:
        new_w = max(MIN_WIDTH, int(w * IMAGE_HEIGHT / h))
        img = img.resize((new_w, IMAGE_HEIGHT), Image.BICUBIC)

    return img


def generate_batch(
    batch_args: list[tuple],
    batch_id: int,
) -> list[tuple[str, str]]:
    """
    Generate a batch of images. Called in a worker process.

    Args:
        batch_args: list of (index, text, font_path, seed) tuples
        batch_id: batch identifier for progress

    Returns:
        list of (filename, label) tuples for successfully generated images
    """
    results = []

    for idx, text, font_path, seed in batch_args:
        rng = random.Random(seed)

        font_size = rng.randint(FONT_SIZE_MIN, FONT_SIZE_MAX)
        bg_color  = rng.randint(BG_COLOR_MIN, BG_COLOR_MAX)
        fg_color  = rng.randint(FG_COLOR_MIN, FG_COLOR_MAX)

        apply_blur     = rng.random() < BLUR_PROB
        blur_radius    = rng.uniform(0.3, BLUR_RADIUS_MAX) if apply_blur else 0
        apply_noise    = rng.random() < NOISE_PROB
        noise_std_val  = rng.randint(3, NOISE_STD) if apply_noise else 0
        apply_rotation = rng.random() < ROTATION_PROB
        rotation_deg   = rng.uniform(-ROTATION_MAX_DEG, ROTATION_MAX_DEG) if apply_rotation else 0

        # Set numpy seed for this image's noise
        np.random.seed(seed)

        img = render_one_image(
            text=text,
            font_path=font_path,
            font_size=font_size,
            bg_color=bg_color,
            fg_color=fg_color,
            apply_blur=apply_blur,
            blur_radius=blur_radius,
            apply_noise=apply_noise,
            noise_std=noise_std_val,
            apply_rotation=apply_rotation,
            rotation_deg=rotation_deg,
        )

        if img is None:
            continue

        filename = f"synth_{idx:06d}.png"
        img_path = IMAGE_DIR / filename
        img.save(str(img_path), format="PNG")

        results.append((filename, text))

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Vietnamese printed text images")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of images to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("03_generate_images.py  -  Synthetic image generation")
    print("=" * 60)

    # Load resources
    corpus = load_corpus()
    font_paths = load_fonts()
    font_path_strs = [str(p) for p in font_paths]

    # Create output directory
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare generation plan
    total = args.count
    rng = random.Random(args.seed)

    print(f"\n  Generating {total} images ...")
    print(f"  Corpus size: {len(corpus)} sentences")
    print(f"  Fonts: {len(font_paths)}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {IMAGE_DIR}")

    # Build task list: (index, text, font_path, seed)
    # Strategy: maximize text diversity for decoder training.
    # Shuffle corpus, cycle through it so every sentence gets used
    # before any sentence repeats. This ensures the decoder sees
    # maximum unique text patterns per epoch.
    shuffled_corpus = corpus.copy()
    rng.shuffle(shuffled_corpus)
    repetitions = max(1, total // len(shuffled_corpus))
    remainder = total - (repetitions * len(shuffled_corpus))

    print(f"\n  Text diversity plan:")
    print(f"    Unique sentences: {len(shuffled_corpus)}")
    print(f"    Repetitions per sentence: ~{repetitions}")
    print(f"    (each repetition uses a DIFFERENT font + augmentation)")

    tasks = []
    for i in range(total):
        text = shuffled_corpus[i % len(shuffled_corpus)]
        font = rng.choice(font_path_strs)
        seed = rng.randint(0, 2**31)
        tasks.append((i, text, font, seed))

    # Split into batches
    batch_size = 500
    batches = []
    for start in range(0, len(tasks), batch_size):
        batches.append(tasks[start:start + batch_size])

    print(f"  Batches: {len(batches)} x ~{batch_size}")

    # Execute
    all_results: list[tuple[str, str]] = []
    completed = 0

    if args.workers <= 1:
        # Sequential
        for batch_id, batch in enumerate(batches):
            results = generate_batch(batch, batch_id)
            all_results.extend(results)
            completed += len(batch)
            pct = 100 * completed / total
            print(f"\r  Progress: {completed}/{total} ({pct:.1f}%)  "
                  f"[Generated: {len(all_results)}]", end="", flush=True)
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for batch_id, batch in enumerate(batches):
                fut = executor.submit(generate_batch, batch, batch_id)
                futures[fut] = len(batch)

            for fut in as_completed(futures):
                results = fut.result()
                all_results.extend(results)
                completed += futures[fut]
                pct = 100 * completed / total
                print(f"\r  Progress: {completed}/{total} ({pct:.1f}%)  "
                      f"[Generated: {len(all_results)}]", end="", flush=True)

    print(f"\n\n  Generation complete!")
    print(f"  Total images generated: {len(all_results)}")

    # Critical metric: unique text diversity
    unique_texts = len(set(label for _, label in all_results))
    print(f"  Unique text labels: {unique_texts}")
    print(f"  Avg repetitions per text: {len(all_results) / max(unique_texts, 1):.1f}")
    print(f"  (This is the number that matters for decoder diversity!)")

    # Sort by filename for deterministic output
    all_results.sort(key=lambda x: x[0])

    # Save labels
    labels_path = OUTPUT_DIR / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for filename, label in all_results:
            f.write(f"{filename}\t{label}\n")
    print(f"  Labels saved: {labels_path}")

    # Stats
    if all_results:
        # Check image sizes
        sample_paths = [IMAGE_DIR / r[0] for r in all_results[:100]]
        widths = []
        for p in sample_paths:
            if p.exists():
                with Image.open(p) as img:
                    widths.append(img.size[0])

        if widths:
            print(f"\n  Sample image stats (first 100):")
            print(f"    Width range: {min(widths)} - {max(widths)} px")
            print(f"    Height: {IMAGE_HEIGHT} px (fixed)")
            print(f"    Avg aspect ratio: {sum(widths)/len(widths)/IMAGE_HEIGHT:.1f}")

    # Estimate disk usage
    total_size = sum(f.stat().st_size for f in IMAGE_DIR.glob("*.png"))
    print(f"\n  Disk usage: {total_size / (1024**2):.1f} MB")
    print(f"  Avg image size: {total_size / max(len(all_results),1) / 1024:.1f} KB")

    failure_rate = 100 * (total - len(all_results)) / total
    if failure_rate > 5:
        print(f"\n  WARNING: {failure_rate:.1f}% of images failed to generate.")
        print(f"  This may happen if text is too long for the max width.")
        print(f"  Consider reducing font size or increasing MAX_WIDTH.")

    print(f"\n[DONE] Next: python 04_integrate_pipeline.py")


if __name__ == "__main__":
    main()
