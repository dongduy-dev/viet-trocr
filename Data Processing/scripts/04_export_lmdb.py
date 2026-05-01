"""
04_export_lmdb.py
=================
Mục tiêu:
  Đọc labels_filtered.csv (hoặc labels_split.csv nếu chưa filter) của các level
  word và line, đóng gói ảnh thành LMDB database tốc độ cao để train TrOCR.

Cấu trúc LMDB output:
  lmdb/{level}/{split}/
    image-00000001  <- PNG bytes của ảnh (đọc trực tiếp từ disk, không re-encode)
    label-00000001  <- label UTF-8 bytes
    ...
    num-samples     <- tổng số mẫu hợp lệ

Log thống kê in ra sau mỗi tập:
  - Tổng mẫu OK / lỗi
  - Breakdown: handwritten vs printed

Lưu ý:
  - Paragraph không export LMDB (dùng cho DBNet++ detector, pre-trained)
  - Level word_printed / word_handwritten: pre-warm TrOCR encoder
  - Level line_printed / line_handwritten: fine-tune chính TrOCR
  - Ảnh đã được parse bước 01 sang RGB PNG, nên chỉ cần đọc bytes trực tiếp
  - Bước validation (100 ảnh đầu tiên) kiểm tra PIL có mở được file không

Fix [MDB_MAP_FULL]:
  - calc_map_size() đo dung lượng file thực tế trên disk trước khi mở LMDB
  - export_one_lmdb() có retry loop: nếu vẫn đầy thì tự tăng map_size gấp đôi
    và ghi lại từ đầu cho đến khi thành công

Cài đặt : pip install lmdb Pillow tqdm
Chạy    : python scripts/04_export_lmdb.py
"""

import csv
import io
import shutil
from collections import Counter
from pathlib import Path

import lmdb
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ── Đường dẫn ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "processed"
LMDB_ROOT    = PROJECT_ROOT / "lmdb"

# ── Cấu hình export ──────────────────────────────────────────────────────────
# word và line đều được tách riêng printed / handwritten
# paragraph dùng cho DBNet++, không export LMDB
EXPORT_LEVELS = ["word_printed", "word_handwritten", "line_printed", "line_handwritten"]
SPLITS        = ["train", "val", "test"]

# Hệ số an toàn cho map_size:
#   - LMDB có overhead B-tree metadata (~5-10%)
#   - Windows không tự mở rộng file nên cần dư thoải mái
MAP_SIZE_SAFETY_FACTOR = 3.0   # nhân 3x so với tổng bytes thực trên disk
MAP_SIZE_MINIMUM_BYTES = 256 * 1024 * 1024   # tối thiểu 256 MB dù dataset nhỏ

# Số ảnh đầu tiên để validate format (PIL open check)
VALIDATION_SAMPLE_SIZE = 100


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_split_csv(level: str) -> list[dict]:
    """
    Đọc CSV cho một level thô (word / line).
    Ưu tiên labels_filtered.csv (output của 03_filter_outliers.py),
    fallback sang labels_split.csv nếu chưa chạy bước filter.
    """
    filtered_path = PROCESSED / level / "labels_filtered.csv"
    split_path    = PROCESSED / level / "labels_split.csv"

    if filtered_path.exists():
        target = filtered_path
        print(f"  [CSV] Dùng labels_filtered.csv cho {level}")
    elif split_path.exists():
        target = split_path
        print(f"  [CSV] Dùng labels_split.csv cho {level} (chưa chạy 03_filter_outliers.py)")
    else:
        print(f"  [SKIP] Không tìm thấy CSV cho {level} — chạy 02_split.py trước")
        return []

    with open(target, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_and_filter_csv(level: str) -> list[dict]:
    """
    Đọc CSV và lọc theo data_type cho cả word lẫn line.

    Quy tắc ánh xạ level → thư mục CSV gốc:
      word_printed  / word_handwritten  → processed/word/labels_filtered.csv (or split)
      line_printed  / line_handwritten  → processed/line/labels_filtered.csv (or split)
    Sau đó lọc theo field data_type == "printed" hoặc "handwritten".
    """
    if "word_" in level:
        raw_rows = load_split_csv("word")
    elif "line_" in level:
        raw_rows = load_split_csv("line")
    else:
        # Fallback: level không có hậu tố _printed/_handwritten (không dùng hiện tại)
        return load_split_csv(level)

    if level in ("word_printed", "line_printed"):
        return [r for r in raw_rows if r.get("data_type") == "printed"]
    elif level in ("word_handwritten", "line_handwritten"):
        return [r for r in raw_rows if r.get("data_type") == "handwritten"]

    return []


def read_image_bytes(img_path: Path) -> bytes | None:
    """
    Đọc bytes ảnh trực tiếp từ disk (không re-encode qua PIL).
    Ảnh đã được 01_parse_all_datasets.py chuyển sang RGB PNG,
    nên chỉ cần đọc raw bytes để ghi vào LMDB.
    Trả về None nếu file không tồn tại hoặc rỗng.
    """
    try:
        data = img_path.read_bytes()
        return data if data else None
    except (OSError, IOError):
        return None


def validate_image_format(rows: list[dict], img_dir: Path,
                          n: int = VALIDATION_SAMPLE_SIZE) -> None:
    """
    Kiểm tra format ảnh bằng PIL cho n ảnh đầu tiên.
    Đảm bảo file PNG hợp lệ và có thể decode được trước khi export toàn bộ.
    Raise RuntimeError nếu phát hiện file hỏng vượt ngưỡng (>5%).
    """
    sample = rows[:n]
    ok, bad = 0, 0
    bad_files = []

    for row in sample:
        img_path = img_dir / row["filename"]
        try:
            with Image.open(img_path) as img:
                img.load()  # Decode toàn bộ pixel — chắc chắn hơn verify() (chỉ check header)
            ok += 1
        except Exception as e:
            bad += 1
            bad_files.append((row["filename"], str(e)))

    total = ok + bad
    if total == 0:
        return

    bad_pct = (bad / total) * 100
    print(f"    [VALIDATE] {ok}/{total} ảnh OK  |  {bad} lỗi ({bad_pct:.1f}%)")

    if bad_files:
        for fname, err in bad_files[:5]:
            print(f"      ✗ {fname}: {err}")
        if len(bad_files) > 5:
            print(f"      ... và {len(bad_files) - 5} file khác")

    if bad_pct > 5:
        raise RuntimeError(
            f"Quá nhiều ảnh lỗi ({bad_pct:.1f}% > 5%). "
            f"Kiểm tra lại 01_parse_all_datasets.py hoặc dữ liệu gốc."
        )


def make_key(prefix: str, idx: int) -> bytes:
    """
    Key LMDB với zero-padding 8 chữ số.
    Convention chuẩn của deep-text-recognition-benchmark:
      image-00000001, label-00000001, ...
    """
    return f"{prefix}-{idx:08d}".encode("utf-8")


def calc_map_size(rows: list[dict], img_dir: Path,
                  safety: float = MAP_SIZE_SAFETY_FACTOR,
                  minimum: int  = MAP_SIZE_MINIMUM_BYTES) -> int:
    """
    Tính map_size LMDB dựa trên dung lượng file ảnh thực tế trên disk.

    Tại sao không dùng ước tính cố định (50KB/200KB)?
      - Ảnh scan/handwritten có thể nặng 500KB–2MB mỗi file
      - Một file outlier cũng đủ gây MDB_MAP_FULL

    Cách tính:
      total_disk_bytes × safety_factor  (mặc định 3.0x)
      → bao gồm overhead B-tree LMDB + buffer

    Tham số:
      rows    : danh sách dict từ CSV (cần field 'filename')
      img_dir : thư mục chứa ảnh
      safety  : hệ số nhân (mặc định 3.0)
      minimum : dung lượng tối thiểu (mặc định 256 MB)
    """
    total_disk_bytes = 0
    missing = 0

    for row in rows:
        p = img_dir / row["filename"]
        if p.exists():
            total_disk_bytes += p.stat().st_size
        else:
            missing += 1

    if missing:
        print(f"    [WARN] calc_map_size: {missing} file không tồn tại trên disk")

    computed = int(total_disk_bytes * safety)
    result   = max(computed, minimum)

    print(f"    [MAP_SIZE] disk={total_disk_bytes / (1024**2):.1f} MB  "
          f"× {safety}  =  {result / (1024**3):.2f} GB  "
          f"(min {minimum // (1024**2)} MB applied: {computed < minimum})")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CORE EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_one_lmdb(rows: list[dict], level: str, split: str) -> None:
    """
    Đóng gói một tập (level × split) vào LMDB.

    Quy trình:
      1. validate_image_format() kiểm tra 100 ảnh đầu tiên
      2. calc_map_size() đo dung lượng thực tế trên disk × safety factor
      3. Mở LMDB ở chế độ write
      4. Duyệt rows: đọc bytes trực tiếp từ disk -> ghi vào LMDB
      5. Ghi num-samples ở cuối transaction
      6. In log thống kê handwritten vs printed

    Định tuyến thư mục ảnh gốc:
      - word_printed / word_handwritten → processed/word/images/
      - line_printed / line_handwritten → processed/line/images/

    Retry loop (fix MDB_MAP_FULL):
      - Nếu lỗi MapFullError xảy ra, tự động tăng map_size gấp đôi,
        xóa thư mục LMDB cũ và ghi lại từ đầu.
      - Vòng lặp dừng khi ghi thành công hoặc sau MAX_RETRIES lần thất bại.
    """
    if not rows:
        print(f"  [SKIP] {level}/{split}: không có mẫu")
        return

    out_dir = LMDB_ROOT / level / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Định tuyến thư mục ảnh gốc theo prefix của level
    if "line_" in level:
        source_level = "line"
    elif "word_" in level:
        source_level = "word"
    else:
        source_level = level   # fallback cho level không có hậu tố

    img_dir = PROCESSED / source_level / "images"

    # Bước 0: Validate format consistency (chỉ cho split đầu tiên hoặc train)
    if split == "train":
        validate_image_format(rows, img_dir)

    # Bước 1: tính map_size dựa trên disk thực tế
    map_size = calc_map_size(rows, img_dir)

    MAX_RETRIES = 5   # tối đa 5 lần retry (tăng gấp đôi mỗi lần → 2^5 = 32x)

    for attempt in range(1, MAX_RETRIES + 1):
        count        = 0
        err_count    = 0
        type_counter: Counter = Counter()

        try:
            # writemap + map_async: tối ưu tốc độ ghi trên Windows
            env = lmdb.open(str(out_dir), map_size=map_size,
                            writemap=True, map_async=True)

            with env.begin(write=True) as txn:
                for row in tqdm(rows,
                                desc=f"    {level}/{split} (attempt {attempt})",
                                unit="img"):
                    img_path  = img_dir / row["filename"]
                    img_bytes = read_image_bytes(img_path)

                    if img_bytes is None:
                        err_count += 1
                        continue

                    count += 1
                    txn.put(make_key("image", count), img_bytes)
                    txn.put(make_key("label", count),
                            row["label"].encode("utf-8"))

                    type_counter[row.get("data_type", "unknown")] += 1

                # Lưu tổng số mẫu hợp lệ vào key đặc biệt
                txn.put(b"num-samples", str(count).encode("utf-8"))

            env.close()

            # ── Ghi thành công: in log và thoát retry loop ────────────────
            hw = type_counter.get("handwritten", 0)
            pt = type_counter.get("printed",     0)
            print(f"    ✓ {level:25s}/{split:5s} : "
                  f"{count:6d} OK  |  {err_count} lỗi  |  "
                  f"HW={hw}  PT={pt}  →  {out_dir}")
            return   # thành công, thoát hàm

        except lmdb.MapFullError:
            # Đóng env trước khi xóa thư mục (quan trọng trên Windows)
            try:
                env.close()
            except Exception:
                pass

            old_gb   = map_size / (1024 ** 3)
            map_size = int(map_size * 2)   # tăng gấp đôi
            new_gb   = map_size / (1024 ** 3)

            print(f"\n    [RETRY {attempt}/{MAX_RETRIES}] "
                  f"MDB_MAP_FULL: {old_gb:.2f} GB → tăng lên {new_gb:.2f} GB, "
                  f"ghi lại từ đầu...")

            # Xóa LMDB bị lỗi để tạo mới hoàn toàn
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

    # Thoát vòng lặp mà không thành công
    print(f"\n    [ERROR] {level}/{split}: vẫn lỗi MDB_MAP_FULL sau "
          f"{MAX_RETRIES} lần retry. "
          f"Kiểm tra dung lượng ổ cứng hoặc tăng MAP_SIZE_SAFETY_FACTOR.")


def verify_lmdb(level: str, split: str, n: int = 2) -> None:
    """
    Đọc thử n mẫu đầu tiên để xác nhận LMDB ghi đúng.
    Hiển thị kích thước ảnh và nhãn ngắn gọn.
    """
    lmdb_path = LMDB_ROOT / level / split
    if not lmdb_path.exists():
        return

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        total = int(txn.get(b"num-samples", b"0").decode())
        print(f"\n    [VERIFY] {level}/{split}  →  num-samples={total}")
        for i in range(1, min(n + 1, total + 1)):
            img_b   = txn.get(make_key("image", i))
            label_b = txn.get(make_key("label", i))
            if img_b and label_b:
                size  = Image.open(io.BytesIO(img_b)).size
                label = label_b.decode("utf-8")
                print(f"      [{i:08d}] size={size}  label='{label[:55]}'")
            else:
                print(f"      [{i:08d}] LỖI: key không tồn tại")
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("04_export_lmdb.py")
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Export levels : {EXPORT_LEVELS}")
    print(f"Splits        : {SPLITS}\n")

    grand_total = 0

    for level in EXPORT_LEVELS:
        print("═" * 60)
        print(f"Level: {level.upper()}")
        print("═" * 60)

        all_rows = load_and_filter_csv(level)
        if not all_rows:
            continue

        # Nhóm rows theo final_split một lần duy nhất
        split_groups: dict[str, list[dict]] = {s: [] for s in SPLITS}
        for row in all_rows:
            fs = row.get("final_split", "")
            if fs in split_groups:
                split_groups[fs].append(row)

        # In phân phối trước khi export
        dist = "  ".join(f"{s}={len(split_groups[s])}" for s in SPLITS)
        print(f"  Phân phối: {dist}")

        # Export từng split
        for split in SPLITS:
            export_one_lmdb(split_groups[split], level, split)
            grand_total += len(split_groups[split])

        # Verify nhanh tập train
        verify_lmdb(level, "train", n=2)

    # ── Tổng kết cuối ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("HOÀN THÀNH")
    print("═" * 60)
    print(f"  Tổng mẫu đã xử lý : {grand_total}")
    print(f"  LMDB root          : {LMDB_ROOT}")

    print("\n  Cấu trúc LMDB đã tạo:")
    for level in EXPORT_LEVELS:
        for split in SPLITS:
            p = LMDB_ROOT / level / split
            if p.exists():
                try:
                    env = lmdb.open(str(p), readonly=True, lock=False)
                    with env.begin() as txn:
                        n = txn.get(b"num-samples", b"0").decode()
                    env.close()
                    print(f"    lmdb/{level}/{split}/  ({n} samples)")
                except Exception:
                    print(f"    lmdb/{level}/{split}/")

    print("\n  TrOCR training guide:")
    print("    Stage 1a - Pre-warm encoder (Printed) : lmdb/word_printed/train/")
    print("    Stage 1b - Pre-warm encoder (Handwrit): lmdb/word_handwritten/train/")
    print("    Stage 2a - Printed fine-tune          : lmdb/line_printed/train/")
    print("    Stage 2b - Handwritten adapt          : lmdb/line_handwritten/train/")


if __name__ == "__main__":
    main()
