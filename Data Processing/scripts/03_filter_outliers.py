"""
03_filter_outliers.py
=====================
Lọc outlier images từ labels_split.csv dựa trên:
  1. Ngưỡng kích thước ảnh (dimension filters)
  2. Ký tự không phải tiếng Việt trong label (language filter)

Tạo labels_filtered.csv cho mỗi level, được 04_export_lmdb.py sử dụng.

Ngưỡng lọc kích thước:
  WORD: min_width=10, min_height=10, max_height=300, aspect 0.3–25.0
  LINE: min_width=32, min_height=16, max_height=384, max_width=3000, aspect ≥1.0

Lọc ngôn ngữ:
  Loại bỏ label chứa ký tự ngoài bộ ký tự tiếng Việt hợp lệ:
    - CJK (Chinese/Japanese/Korean)
    - Khmer/Cambodian
    - Thai
    - Cyrillic
    - IPA phonetic symbols
    - Fullwidth punctuation
    - Các script không phải Latin

Chạy: python scripts/03_filter_outliers.py
"""

import csv
import unicodedata
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "processed"

# ─── Dimension Thresholds ────────────────────────────────────────────────────
WORD_FILTERS = {
    "min_width":  10,
    "min_height": 10,
    "max_height": 300,
    "min_aspect": 0.3,   # width / height
    "max_aspect": 25.0,
}

LINE_FILTERS = {
    "min_width":  32,
    "min_height": 16,
    "max_height": 384,
    "max_width":  3000,
    "min_aspect": 1.0,
}


# ─── Vietnamese Character Validation ────────────────────────────────────────
# Allowed Unicode ranges for Vietnamese text labels:
#   - ASCII printable (space through tilde)
#   - Latin-1 Supplement (À-ÿ: accented chars used in Vietnamese/French loanwords)
#   - Latin Extended-A (Ă, ă, Đ, đ, etc.)
#   - Latin Extended-B (Ơ, ơ, Ư, ư)
#   - Latin Extended Additional (Ạ-ỹ: all Vietnamese-specific diacritics)
#   - Combining Diacritical Marks (stacking diacritics in NFC edge cases)
#   - General Punctuation (–, —, ', ', ", ", …, etc.)
#   - Superscripts/Subscripts (²,³ for chemistry/math in real-world text)
#   - Currency Symbols (₫ Vietnamese đồng)
#
# NOT allowed (will trigger removal):
#   - CJK Unified Ideographs (Chinese/Japanese Kanji): U+4E00–U+9FFF
#   - Khmer/Cambodian: U+1780–U+17FF
#   - Thai: U+0E00–U+0E7F
#   - Cyrillic: U+0400–U+04FF
#   - Arabic: U+0600–U+06FF
#   - Japanese Hiragana/Katakana: U+3040–U+30FF
#   - IPA Extensions: U+0250–U+02AF (phonetic, not normal text)
#   - Fullwidth Forms: U+FF00–U+FFEF
#   - Any other non-Latin script

ALLOWED_RANGES = [
    (0x0009, 0x000A),  # Tab, newline (edge case in labels)
    (0x0020, 0x007E),  # ASCII printable
    (0x00A0, 0x00FF),  # Latin-1 Supplement
    (0x0100, 0x017F),  # Latin Extended-A
    (0x0180, 0x024F),  # Latin Extended-B
    (0x0300, 0x036F),  # Combining Diacritical Marks
    (0x1E00, 0x1EFF),  # Latin Extended Additional (Vietnamese core)
    (0x2000, 0x206F),  # General Punctuation
    (0x2070, 0x209F),  # Superscripts and Subscripts
    (0x20A0, 0x20CF),  # Currency Symbols (₫)
]

# Pre-build a frozenset of all allowed codepoints for O(1) lookup
_allowed_codepoints: frozenset[int] = frozenset()
for _lo, _hi in ALLOWED_RANGES:
    _allowed_codepoints = _allowed_codepoints | frozenset(range(_lo, _hi + 1))


def is_vietnamese_label(label: str) -> tuple[bool, list[str]]:
    """
    Check if a label contains only Vietnamese-compatible characters.

    Returns:
        (True, [])             if all characters are allowed
        (False, [bad_chars])   if foreign characters found
    """
    bad_chars = []
    for ch in label:
        if ord(ch) not in _allowed_codepoints:
            bad_chars.append(ch)
    return (len(bad_chars) == 0), bad_chars


def passes_filter(w: int, h: int, filters: dict) -> bool:
    """Returns True if image dimensions pass all filters."""
    if w < filters.get("min_width", 0):
        return False
    if h < filters.get("min_height", 0):
        return False
    if h > filters.get("max_height", float("inf")):
        return False
    if w > filters.get("max_width", float("inf")):
        return False

    aspect = w / h if h > 0 else 0
    if aspect < filters.get("min_aspect", 0):
        return False
    if aspect > filters.get("max_aspect", float("inf")):
        return False

    return True


def categorize_foreign_chars(bad_chars: list[str]) -> str:
    """Categorize foreign characters into script families for reporting."""
    scripts = set()
    for ch in bad_chars:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            scripts.add("CJK")
        elif 0x1780 <= cp <= 0x17FF:
            scripts.add("Khmer")
        elif 0x0E00 <= cp <= 0x0E7F:
            scripts.add("Thai")
        elif 0x0400 <= cp <= 0x04FF:
            scripts.add("Cyrillic")
        elif 0x3040 <= cp <= 0x30FF:
            scripts.add("Japanese")
        elif 0x0600 <= cp <= 0x06FF:
            scripts.add("Arabic")
        elif 0x0250 <= cp <= 0x02AF:
            scripts.add("IPA")
        elif 0xFF00 <= cp <= 0xFFEF:
            scripts.add("Fullwidth")
        else:
            scripts.add(f"U+{cp:04X}")
    return "+".join(sorted(scripts))


def filter_level(level: str, filters: dict) -> None:
    csv_in  = PROCESSED / level / "labels_split.csv"
    csv_out = PROCESSED / level / "labels_filtered.csv"
    img_dir = PROCESSED / level / "images"

    if not csv_in.exists():
        print(f"  [SKIP] {csv_in}")
        return

    with open(csv_in, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"  [SKIP] {level}: no rows in CSV")
        return

    kept, removed = [], []
    reasons = Counter()
    foreign_source_counter = Counter()  # Track which sources have foreign chars

    for row in tqdm(rows, desc=f"  Filtering {level}", unit="img"):
        label = row.get("label", "")

        # ── Step 1: Language filter (check label for non-Vietnamese chars) ────
        is_viet, bad_chars = is_vietnamese_label(label)
        if not is_viet:
            removed.append(row)
            script_tag = categorize_foreign_chars(bad_chars)
            reasons[f"non_vietnamese ({script_tag})"] += 1
            foreign_source_counter[row.get("source", "?")] += 1
            continue

        # ── Step 2: Dimension filter (check image size) ──────────────────────
        img_path = img_dir / row["filename"]
        if not img_path.exists():
            removed.append(row)
            reasons["missing_file"] += 1
            continue

        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            removed.append(row)
            reasons["corrupt"] += 1
            continue

        if passes_filter(w, h, filters):
            kept.append(row)
        else:
            removed.append(row)
            # Categorize removal reason
            aspect = w / h if h > 0 else 0
            if w < filters.get("min_width", 0):
                reasons["dim:width_too_small"] += 1
            elif h < filters.get("min_height", 0):
                reasons["dim:height_too_small"] += 1
            elif h > filters.get("max_height", float("inf")):
                reasons["dim:height_too_large"] += 1
            elif w > filters.get("max_width", float("inf")):
                reasons["dim:width_too_large"] += 1
            elif aspect < filters.get("min_aspect", 0):
                reasons["dim:aspect_too_low"] += 1
            elif aspect > filters.get("max_aspect", float("inf")):
                reasons["dim:aspect_too_high"] += 1
            else:
                reasons["dim:other"] += 1

    # Write filtered CSV
    fieldnames = list(rows[0].keys())
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    # Print summary per split
    split_counter_kept = Counter(r.get("final_split", "?") for r in kept)
    split_counter_removed = Counter(r.get("final_split", "?") for r in removed)

    print(f"\n  → {level}: {len(kept)} kept / {len(removed)} removed")
    print(f"    Kept by split:    train={split_counter_kept.get('train',0)}  "
          f"val={split_counter_kept.get('val',0)}  "
          f"test={split_counter_kept.get('test',0)}")
    print(f"    Removed by split: train={split_counter_removed.get('train',0)}  "
          f"val={split_counter_removed.get('val',0)}  "
          f"test={split_counter_removed.get('test',0)}")
    print(f"    Removal reasons:")
    for reason, count in reasons.most_common():
        print(f"      {reason}: {count}")

    if foreign_source_counter:
        print(f"    Non-Vietnamese by source:")
        for source, count in foreign_source_counter.most_common():
            print(f"      {source}: {count} labels removed")

    print(f"    → Saved: {csv_out}")


if __name__ == "__main__":
    print("03_filter_outliers.py")
    print(f"Project root: {PROJECT_ROOT}\n")

    print("═" * 55)
    print("WORD LEVEL")
    print("═" * 55)
    print(f"  Dimension filters: {WORD_FILTERS}")
    print(f"  Language filter: non-Vietnamese character removal")
    filter_level("word", WORD_FILTERS)

    print("\n" + "═" * 55)
    print("LINE LEVEL")
    print("═" * 55)
    print(f"  Dimension filters: {LINE_FILTERS}")
    print(f"  Language filter: non-Vietnamese character removal")
    filter_level("line", LINE_FILTERS)

    print("\n[DONE]")
    print("  Bước tiếp theo: python scripts/04_export_lmdb.py")
    print("  (04_export_lmdb.py sẽ tự động dùng labels_filtered.csv nếu có)")
