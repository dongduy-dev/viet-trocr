"""
06_source_based_analysis.py
===========================
Objective:
  1. Read labels_filtered.csv / labels_split.csv / labels_master.csv for Word and Line levels.
  2. Group the data by 'source' (e.g., uit, cinnamon, vintext).
  3. Calculate size distributions (bins) for each source.
  4. Export the detailed list of outlier images to a .txt file for easy inspection.

Install : pip install Pillow numpy tqdm
Run     : python scripts/06_source_based_analysis.py
"""

import csv
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "processed"
REPORTS_DIR  = PROJECT_ROOT / "reports"

LEVELS = ["word", "line"]

# ── Bins Configuration ───────────────────────────────────────────────────────
WORD_W_BINS = [0, 10, 32, 64, 128, 256, 512, 1024, float('inf')]
WORD_H_BINS = [0, 10, 32, 64, 96, 128, 160, float('inf')]

LINE_W_BINS = [0, 50, 256, 512, 1024, 1536, 2048, 2500, float('inf')]
LINE_H_BINS = [0, 16, 32, 64, 96, 128, 160, 256, float('inf')]

def load_csv(level: str) -> list[dict]:
    """Prefers labels_filtered.csv, then labels_split.csv, falls back to labels_master.csv."""
    filtered_csv = PROCESSED / level / "labels_filtered.csv"
    split_csv = PROCESSED / level / "labels_split.csv"
    master_csv = PROCESSED / level / "labels_master.csv"

    if filtered_csv.exists():
        target_csv = filtered_csv
    elif split_csv.exists():
        target_csv = split_csv
    else:
        target_csv = master_csv

    if not target_csv.exists():
        return []
    with open(target_csv, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def format_distribution(name: str, values: list, bins: list) -> str:
    """Returns a formatted string of the distribution."""
    if not values:
        return f"    [No data for {name}]\n"
        
    total = len(values)
    counts = [0] * (len(bins) - 1)
    
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i+1]:
                counts[i] += 1
                break
                
    output = []
    output.append(f"  --- {name} Distribution ---")
    for i in range(len(counts)):
        if bins[i+1] == float('inf'):
            range_str = f">= {bins[i]:>4} px       "
        else:
            range_str = f"[{bins[i]:>4} -> {bins[i+1]-1:>4}] px"
            
        pct = (counts[i] / total) * 100 if total > 0 else 0
        output.append(f"    {range_str} : {counts[i]:>6} imgs ({pct:>5.2f}%)")
        
    return "\n".join(output)

def analyze_level(level: str):
    print(f"\n{'═' * 65}")
    print(f"DETAILED ANALYSIS BY SOURCE: {level.upper()} LEVEL")
    print(f"{'═' * 65}")
    
    rows = load_csv(level)
    if not rows:
        print(f"  [SKIP] No data found for {level}")
        return

    img_dir = PROCESSED / level / "images"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"outliers_report_{level}.txt"
    
    # Structure to hold data per source
    # source_data[source_name] = { "widths": [], "heights": [], "too_small": [], "too_large": [], "weird_ratio": [] }
    source_data = defaultdict(lambda: {
        "widths": [], 
        "heights": [], 
        "too_small": [], 
        "too_large": [], 
        "weird_ratio": []
    })

    # 1. Scan and process all images
    for row in tqdm(rows, desc=f"  Scanning {level}", unit="img"):
        img_path = img_dir / row["filename"]
        source = row.get("source", "unknown")
        
        if not img_path.exists():
            continue
            
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                source_data[source]["widths"].append(w)
                source_data[source]["heights"].append(h)
                
                # Outlier detection logic
                if w < 10 or h < 10:
                    source_data[source]["too_small"].append((row["filename"], w, h))
                elif h > 200:
                    source_data[source]["too_large"].append((row["filename"], w, h))
                elif h > 0 and (w/h < 0.5): 
                    # Aspect ratio < 0.5 means it's taller than it is wide (likely rotated or bad crop)
                    source_data[source]["weird_ratio"].append((row["filename"], w, h))
                    
        except Exception:
            pass

    if not source_data:
        print("  [WARN] No valid images found.")
        return

    # 2. Print distribution to console and write outliers to TXT
    w_bins = WORD_W_BINS if level == "word" else LINE_W_BINS
    h_bins = WORD_H_BINS if level == "word" else LINE_H_BINS

    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"OUTLIER REPORT - {level.upper()} LEVEL\n")
        f_out.write(f"{'=' * 50}\n\n")

        for source, data in sorted(source_data.items()):
            total_imgs = len(data["widths"])
            print(f"\n[SOURCE: {source.upper()}] - Total: {total_imgs} images")
            
            # Print Distribution to Console
            print(format_distribution("Width", data["widths"], w_bins))
            print(format_distribution("Height", data["heights"], h_bins))
            
            # Write Outliers to Text File
            f_out.write(f"[{source.upper()}]\n")
            f_out.write(f"Total processed: {total_imgs}\n")
            
            def write_outliers(title: str, items: list):
                f_out.write(f"  {title} ({len(items)} images):\n")
                if not items:
                    f_out.write("    - None\n")
                else:
                    for name, w, h in items:
                        f_out.write(f"    - {name} (W:{w}, H:{h})\n")
                f_out.write("\n")

            write_outliers("1. Too Small (W < 10 or H < 10)", data["too_small"])
            write_outliers("2. Too Tall (H > 200)", data["too_large"])
            write_outliers("3. Weird Aspect Ratio (W/H < 0.5)", data["weird_ratio"])
            
            f_out.write("-" * 50 + "\n\n")
            
    print(f"\n  → Outlier report saved to: {report_path}")

def main():
    print("06_source_based_analysis.py")
    print(f"Project root: {PROJECT_ROOT}\n")
    for level in LEVELS:
        analyze_level(level)
    print("\n[COMPLETE]")

if __name__ == "__main__":
    main()
