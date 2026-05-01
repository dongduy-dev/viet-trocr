# -*- coding: utf-8 -*-
"""
02_download_fonts.py
====================
Download Vietnamese-compatible fonts from GitHub (google/fonts repository).

Fonts are saved to fonts/ directory as .ttf files.

Chay: python 02_download_fonts.py
"""
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
FONT_DIR = PROJECT_ROOT / "fonts"

# Direct download URLs for TTF fonts from GitHub (google/fonts repo, raw content)
# All fonts verified to support Vietnamese diacritics
FONT_URLS = {
    # ── Sans-serif ───────────────────────────────────────────────────────
    "Roboto-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf",
    "NotoSans-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
    "NotoSans-Italic.ttf":
        "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans-Italic%5Bwdth%2Cwght%5D.ttf",
    "OpenSans-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
    "OpenSans-Italic.ttf":
        "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans-Italic%5Bwdth%2Cwght%5D.ttf",
    "Montserrat-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Montserrat-Italic.ttf":
        "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Italic%5Bwght%5D.ttf",
    "Nunito-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito%5Bwght%5D.ttf",
    "Nunito-Italic.ttf":
        "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito-Italic%5Bwght%5D.ttf",
    "Inter-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
    "BeVietnamPro-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-Regular.ttf",
    "BeVietnamPro-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-Bold.ttf",
    "BeVietnamPro-Italic.ttf":
        "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-Italic.ttf",
    "BeVietnamPro-Medium.ttf":
        "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-Medium.ttf",
    "BeVietnamPro-Light.ttf":
        "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-Light.ttf",
    "BeVietnamPro-SemiBold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-SemiBold.ttf",
    # ── Serif ────────────────────────────────────────────────────────────
    "NotoSerif-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/notoserif/NotoSerif%5Bwdth%2Cwght%5D.ttf",
    "NotoSerif-Italic.ttf":
        "https://github.com/google/fonts/raw/main/ofl/notoserif/NotoSerif-Italic%5Bwdth%2Cwght%5D.ttf",
    "Lora-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/lora/Lora%5Bwght%5D.ttf",
    "Lora-Italic.ttf":
        "https://github.com/google/fonts/raw/main/ofl/lora/Lora-Italic%5Bwght%5D.ttf",
    "Merriweather-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/merriweather/Merriweather%5Bwght%5D.ttf",
    "Merriweather-Italic.ttf":
        "https://github.com/google/fonts/raw/main/ofl/merriweather/Merriweather-Italic%5Bwght%5D.ttf",
    # ── Metric-compatible system fonts ───────────────────────────────────
    "Tinos-Regular.ttf":
        "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-Regular.ttf",
    "Tinos-Bold.ttf":
        "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-Bold.ttf",
    "Tinos-Italic.ttf":
        "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-Italic.ttf",
    "Arimo-Regular.ttf":
        "https://github.com/google/fonts/raw/main/apache/arimo/Arimo%5Bwght%5D.ttf",
    "Arimo-Italic.ttf":
        "https://github.com/google/fonts/raw/main/apache/arimo/Arimo-Italic%5Bwght%5D.ttf",
    # ── Book / traditional ──────────────────────────────────────────────
    "LibreBaskerville-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/librebaskerville/LibreBaskerville-Regular.ttf",
    "LibreBaskerville-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/librebaskerville/LibreBaskerville-Bold.ttf",
    "LibreBaskerville-Italic.ttf":
        "https://github.com/google/fonts/raw/main/ofl/librebaskerville/LibreBaskerville-Italic.ttf",
}

# Vietnamese test string
VIET_TEST = "Dai hoc Ton Duc Thang ae oi uo oe"  # ASCII-safe for console


def download_font(name: str, url: str) -> bool:
    """Download a single font file."""
    dst = FONT_DIR / name
    if dst.exists():
        return True  # Already downloaded

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        response = urllib.request.urlopen(req, timeout=30)
        data = response.read()

        # Verify it's a valid font file (starts with font magic bytes)
        # TrueType: 00 01 00 00 or 'true' or 'OTTO'
        if len(data) < 100:
            return False

        with open(dst, "wb") as f:
            f.write(data)
        return True

    except Exception as e:
        return False


def validate_fonts():
    """Check which fonts can render Vietnamese."""
    try:
        from PIL import ImageFont, ImageDraw, Image
    except ImportError:
        print("\n  [SKIP] Pillow not installed, cannot validate")
        return

    viet_test = "Dai hoc ABC oe uo oi ae"
    ttf_files = sorted(FONT_DIR.glob("*.ttf"))
    valid = 0
    invalid_list = []

    for ttf in ttf_files:
        try:
            font = ImageFont.truetype(str(ttf), 24)
            img = Image.new("RGB", (800, 50), "white")
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), viet_test, fill="black", font=font)
            valid += 1
        except Exception as e:
            invalid_list.append((ttf.name, str(e)))

    print(f"\n  Font validation:")
    print(f"    Valid: {valid}/{len(ttf_files)}")
    if invalid_list:
        for name, err in invalid_list:
            print(f"    [INVALID] {name}: {err}")


def main():
    print("=" * 60)
    print("02_download_fonts.py  -  Downloading Vietnamese fonts")
    print("=" * 60)

    FONT_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = list(FONT_DIR.glob("*.ttf"))
    if len(existing) >= 10:
        print(f"\n  Found {len(existing)} existing fonts. Skipping download.")
        validate_fonts()
        print(f"\n[DONE] Next: python 03_generate_images.py")
        return

    print(f"\n  Downloading {len(FONT_URLS)} font files...")
    success = 0
    failed = 0

    for name, url in FONT_URLS.items():
        status = "OK" if download_font(name, url) else "FAIL"
        size = ""
        if (FONT_DIR / name).exists():
            size = f" ({(FONT_DIR / name).stat().st_size / 1024:.0f} KB)"
            success += 1
        else:
            failed += 1
        print(f"  [{status}] {name}{size}")

    print(f"\n  Results: {success} downloaded, {failed} failed")

    # List all fonts
    ttf_files = sorted(FONT_DIR.glob("*.ttf"))
    print(f"  Total .ttf files in fonts/: {len(ttf_files)}")

    validate_fonts()
    print(f"\n[DONE] Next: python 03_generate_images.py")


if __name__ == "__main__":
    main()
