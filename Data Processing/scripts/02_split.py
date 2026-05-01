"""
02_split.py
===========
Mục tiêu:
  Đọc labels_master.csv của từng level, xử lý cột pre_split
  và tạo labels_split.csv với cột final_split ('train'|'val'|'test').

Logic chia theo pre_split:
  'train' | 'val' | 'test'
    -> Giữ nguyên (tôn trọng phân chia của tác giả dataset)

  'train_pool'  (UIT_HWDB train_data)
    -> writer_id 1–229   : final_split = 'train'
    -> writer_id 230–249 : final_split = 'val'
    Writer-independent: 1 người chỉ xuất hiện trong 1 tập.

  'unassigned'  (Viet_Wiki synthetic)
    -> sklearn train_test_split: 80% train / 10% val / 10% test
    -> random_state=42 đảm bảo tái lập

  'anyuuus_pool'  (Anyuuus printed)
    -> Group-split theo writer_id (Document ID)
    -> Lấy danh sách writer_id duy nhất, split 80/10/10
    -> Map ngược lại từng mẫu theo writer_id
    -> Chống data leakage: các dòng cùng 1 tài liệu không bị tách rời

  'synthetic_pool'  (Synthetic Modern printed)
    -> sklearn train_test_split: 90% train / 5% val / 5% test
    -> random_state=42 đảm bảo tái lập
    -> Tỷ lệ 90/5/5 (không phải 80/10/10) vì dữ liệu synthetic sạch,
      không có rủi ro data leakage, muốn tối đa lượng dữ liệu hiện đại cho train

Cài đặt : pip install scikit-learn tqdm
Chạy    : python scripts/02_split.py
"""

import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Seed cố định để tái lập ──────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Đường dẫn ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "processed"

LEVELS = ["word", "line", "paragraph"]

# ── Boundary writer_id cho UIT_HWDB train_pool ───────────────────────────────
UIT_VAL_START = 230   # writer_id >= 230 -> val
UIT_VAL_END   = 249   # writer_id <= 249 -> val

# ── Schema CSV đầu ra ────────────────────────────────────────────────────────
BASE_FIELDS = ["filename", "label", "source", "level",
               "data_type", "writer_id", "pre_split"]
OUT_FIELDS  = BASE_FIELDS + ["final_split"]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_master_csv(level: str) -> list[dict]:
    """Đọc labels_master.csv. Trả về [] nếu không tồn tại."""
    path = PROCESSED / level / "labels_master.csv"
    if not path.exists():
        print(f"  [SKIP] {path} — chạy 01_parse_all_datasets.py trước")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_split_csv(rows: list[dict], level: str) -> None:
    """Lưu labels_split.csv vào processed/{level}/."""
    if not rows:
        return
    out_path = PROCESSED / level / "labels_split.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Đã lưu: {out_path}  ({len(rows)} dòng)")


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def assign_determined(row: dict) -> str:
    """
    Gán final_split cho các row đã có pre_split xác định:
      - 'train' / 'val' / 'test' -> giữ nguyên
      - 'train_pool'              -> phân theo writer_id số nguyên
    """
    pre = row.get("pre_split", "")

    if pre in ("train", "val", "test"):
        return pre

    if pre == "train_pool":
        try:
            wid = int(row["writer_id"])
        except (ValueError, KeyError):
            return "train"   # fallback an toàn
        return "val" if UIT_VAL_START <= wid <= UIT_VAL_END else "train"

    # Không thuộc nhóm này -> caller xử lý
    return "unresolved"


def split_by_random(rows: list[dict]) -> list[dict]:
    """
    Chia ngẫu nhiên 80/10/10 (train/val/test) bằng sklearn.
    Dùng cho Viet_Wiki (pre_split == 'unassigned').
    """
    if not rows:
        return []

    indices = list(range(len(rows)))
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_SEED, shuffle=True
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
    )

    idx_train_set = set(idx_train)
    idx_val_set   = set(idx_val)

    result = []
    for i, row in enumerate(rows):
        if i in idx_train_set:
            final = "train"
        elif i in idx_val_set:
            final = "val"
        else:
            final = "test"
        result.append({**row, "final_split": final})

    return result


def split_by_group(rows: list[dict]) -> list[dict]:
    """
    Group-split theo writer_id: lấy danh sách writer_id duy nhất,
    chia 80/10/10, rồi map kết quả ngược lại vào từng mẫu.

    Dùng cho Anyuuus (pre_split == 'anyuuus_pool').
    Đảm bảo tất cả dòng cùng 1 tài liệu (writer_id) nằm trong 1 tập,
    tránh data leakage giữa các crop của cùng 1 văn bản.
    """
    if not rows:
        return []

    # Lấy danh sách writer_id duy nhất (sorted cho deterministic)
    unique_writers = sorted(set(r["writer_id"] for r in rows))

    # Chia writer list 80 / 10 / 10
    # train_test_split tự shuffle khi shuffle=True (mặc định),
    # random_state=42 đảm bảo reproducibility
    w_train, w_temp = train_test_split(
        unique_writers, test_size=0.20, random_state=RANDOM_SEED, shuffle=True
    )
    w_val, w_test = train_test_split(
        w_temp, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
    )

    train_set = set(w_train)
    val_set   = set(w_val)
    # test_set  = set(w_test)  # phần còn lại

    result = []
    for row in rows:
        wid = row["writer_id"]
        if wid in train_set:
            final = "train"
        elif wid in val_set:
            final = "val"
        else:
            final = "test"
        result.append({**row, "final_split": final})

    return result


def split_synthetic(rows: list[dict]) -> list[dict]:
    """
    Chia ngẫu nhiên 90/5/5 (train/val/test) bằng sklearn.
    Dùng cho Synthetic Modern (pre_split == 'synthetic_pool').

    Tỷ lệ 90/5/5 thay vì 80/10/10 vì:
      - Dữ liệu synthetic không có rủi ro data leakage
      - Là dữ liệu hiện đại cần tối đa hóa trong training để pha loãng
        ngôn ngữ lịch sử từ anyuuus
      - Vẫn cần val/test để monitor chất lượng dữ liệu tổng hợp
    """
    if not rows:
        return []

    indices = list(range(len(rows)))
    # 90% train, 10% temp
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.10, random_state=RANDOM_SEED, shuffle=True
    )
    # Split temp 50/50 -> 5% val, 5% test
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
    )

    idx_train_set = set(idx_train)
    idx_val_set   = set(idx_val)

    result = []
    for i, row in enumerate(rows):
        if i in idx_train_set:
            final = "train"
        elif i in idx_val_set:
            final = "val"
        else:
            final = "test"
        result.append({**row, "final_split": final})

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESS
# ─────────────────────────────────────────────────────────────────────────────

def process_level(level: str) -> None:
    """Chạy toàn bộ split pipeline cho một level."""
    print(f"\n{'─' * 55}")
    print(f"Level: {level.upper()}")
    print(f"{'─' * 55}")

    rows = load_master_csv(level)
    if not rows:
        return

    print(f"  Tổng mẫu: {len(rows)}")

    # Phân loại rows theo pre_split strategy
    determined_rows  : list[dict] = []
    unassigned_rows  : list[dict] = []   # Viet_Wiki
    anyuuus_pool_rows: list[dict] = []   # Anyuuus
    synthetic_pool_rows: list[dict] = [] # Synthetic Modern

    for row in tqdm(rows, desc="  Phân loại", unit="row"):
        pre = row.get("pre_split", "")
        if pre == "unassigned":
            unassigned_rows.append(row)
        elif pre == "anyuuus_pool":
            anyuuus_pool_rows.append(row)
        elif pre == "synthetic_pool":
            synthetic_pool_rows.append(row)
        else:
            # train / val / test / train_pool
            final = assign_determined(row)
            determined_rows.append({**row, "final_split": final})

    # Áp dụng các strategy tương ứng
    all_rows = (
        determined_rows
        + split_by_random(unassigned_rows)
        + split_by_group(anyuuus_pool_rows)
        + split_synthetic(synthetic_pool_rows)
    )

    # ── Thống kê tổng hợp ────────────────────────────────────────────────────
    counter = Counter(r["final_split"] for r in all_rows)
    print(f"\n  Kết quả split:")
    print(f"    train : {counter.get('train', 0)}")
    print(f"    val   : {counter.get('val',   0)}")
    print(f"    test  : {counter.get('test',  0)}")
    if counter.get("unresolved", 0):
        print(f"    [WARN] unresolved: {counter['unresolved']}")

    # ── Thống kê theo source ─────────────────────────────────────────────────
    source_split: dict[str, Counter] = {}
    for r in all_rows:
        src = r.get("source", "?")
        if src not in source_split:
            source_split[src] = Counter()
        source_split[src][r["final_split"]] += 1

    print("\n  Chi tiết theo source:")
    for src, cnt in sorted(source_split.items()):
        print(f"    {src:22s} "
              f"train={cnt.get('train',0):6d}  "
              f"val={cnt.get('val',0):5d}  "
              f"test={cnt.get('test',0):5d}")

    # ── Thống kê theo data_type ──────────────────────────────────────────────
    type_split: dict[str, Counter] = {}
    for r in all_rows:
        dt = r.get("data_type", "?")
        if dt not in type_split:
            type_split[dt] = Counter()
        type_split[dt][r["final_split"]] += 1

    print("\n  Chi tiết theo data_type:")
    for dt, cnt in sorted(type_split.items()):
        print(f"    {dt:15s} "
              f"train={cnt.get('train',0):6d}  "
              f"val={cnt.get('val',0):5d}  "
              f"test={cnt.get('test',0):5d}")

    save_split_csv(all_rows, level)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("02_split.py  |  random_seed =", RANDOM_SEED)
    print(f"Project root : {PROJECT_ROOT}\n")

    for level in LEVELS:
        process_level(level)

    print("\n[HOÀN THÀNH]")
    print("  Bước tiếp theo: python scripts/03_filter_outliers.py")
