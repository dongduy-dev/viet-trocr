# Data Processing Pipeline

Tài liệu hướng dẫn cho phần tiền xử lý dữ liệu của dự án **Nhận dạng Chữ viết tay & Chữ in Tiếng Việt (HTR/OCR)** sử dụng kiến trúc TrOCR.

---

## Tổng quan

Pipeline gồm 4 script chạy tuần tự, chuyển đổi dữ liệu thô từ 7 nguồn khác nhau thành các LMDB database sẵn sàng để huấn luyện TrOCR.

```
raw_data/  →  [parse]  →  processed/  →  [split]  →  [filter]  →  [export]  →  lmdb/
```

| Script | Vai trò | Input | Output |
|---|---|---|---|
| `01_parse_all_datasets.py` | Thu thập, chuẩn hóa, copy ảnh | `raw_data/` | `processed/`, `vocab/` |
| `02_split.py` | Phân chia train/val/test | `labels_master.csv` | `labels_split.csv` |
| `03_filter_outliers.py` | Lọc outlier theo kích thước | `labels_split.csv` | `labels_filtered.csv` |
| `04_export_lmdb.py` | Đóng gói LMDB cho TrOCR | `labels_filtered.csv` + ảnh | `lmdb/` |

**Scripts phụ trợ:**

| Script | Vai trò |
|---|---|
| `05_verify_lmdb.py` | Kiểm tra nhanh LMDB đã export |
| `06_source_based_analysis.py` | Phân tích phân bố kích thước ảnh theo nguồn |

---

## Cấu trúc thư mục

```
Data Processing/
├── raw_data/
│   ├── HandWritten/
│   │   ├── Cinnamon_AI_Dataset/
│   │   │   ├── Data1/              (~15 ảnh  – Val)
│   │   │   ├── Data2/              (~1.823 ảnh – Train)
│   │   │   └── Private_Test/       (~549 ảnh – Test)
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
│   │       ├── images/             (~5.796 ảnh tổng hợp)
│   │       ├── labels.csv
│   │       └── downloadscript.py
│   └── Printed/
│       ├── VinText_Cropped/
│       │   ├── train_images/ (~ 25794 ảnh)
│       │   ├── test_image/ (~ 7220 ảnh)
│       │   └── unseen_test_images/ (~ 10086 ảnh)
│       ├── Vietnamese Receipts MC_OCR 2021/
│       │   ├── text_recognition_mcocr_data/ (~ 6585 ảnh)
│       │   ├── text_recognition_train_data.txt
│       │   └── text_recognition_val_data.txt
│       ├── anyuuus - Vietnamese OCR with PaddleOCR/
│       │   ├── 23127151/
│       │   │   ├── final_crop/ (~ 6155 ảnh)
│       │   │   └── rec_gt.txt
│       │   ├── 23127215/
│       │   │   ├── final_crop/ (~ 8615 ảnh)
│       │   │   └── rec_gt.txt
│       │   ├── 23127263/
│       │   │   ├── final_crop/ (~ 5653 ảnh)
│       │   │   └── rec_gt.txt
│       │   └── 23127407/
│       │       ├── final_crop/ (~ 7892 ảnh)
│       │       └── rec_gt.txt
│       └── Synthetic_Modern/
│           ├── images/ (~ 30000 ảnh)
│           └── labels.txt
├── scripts/
│   ├── 01_parse_all_datasets.py
│   ├── 02_split.py
│   ├── 03_filter_outliers.py
│   ├── 04_export_lmdb.py
│   ├── 05_verify_lmdb.py
│   └── 06_source_based_analysis.py
├── processed/              ← ảnh đã chuẩn hóa RGB + labels CSV
├── lmdb/                   ← LMDB database đầu ra cho TrOCR
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

## Cài đặt

```bash
pip install Pillow tqdm scikit-learn lmdb
```

---

## Hướng dẫn chạy

### Bước 1 — Parse toàn bộ datasets

```bash
# Chạy tất cả
python scripts/01_parse_all_datasets.py all

# Hoặc chạy từng dataset riêng lẻ để debug
python scripts/01_parse_all_datasets.py uit
python scripts/01_parse_all_datasets.py cinnamon
python scripts/01_parse_all_datasets.py wiki
python scripts/01_parse_all_datasets.py vintext
python scripts/01_parse_all_datasets.py mcocr
python scripts/01_parse_all_datasets.py anyuuus
python scripts/01_parse_all_datasets.py synthetic
```

> **WARNING**: Script sẽ tự động xóa tất cả `labels_master.csv` trước khi chạy để tránh lặp dữ liệu. `save_vocab()` chỉ ghi nhận ký tự từ các parser được chạy trong lần gọi hiện tại — luôn chạy với `all` để có vocab đầy đủ.

### Bước 2 — Phân chia train/val/test

```bash
python scripts/02_split.py
```

### Bước 3 — Lọc outlier + ngôn ngữ

```bash
python scripts/03_filter_outliers.py
```

Script thực hiện **2 bước lọc** cho mỗi level (word, line):

**Bước 3a — Lọc ngôn ngữ (chạy trước):**
Loại bỏ label chứa ký tự ngoài bộ ký tự tiếng Việt hợp lệ:

| Script bị loại | Ví dụ | Nguồn chính |
|---|---|---|
| CJK (Chinese/Japanese/Korean) | 漢字, カタカナ | Synthetic_Modern, Anyuuus |
| Khmer/Cambodian | ក, ន, រ | Synthetic_Modern |
| Cyrillic (Russian) | Т, У | Synthetic_Modern |
| IPA Phonetic | ɑ, ə, ɛ | Synthetic_Modern |
| Fullwidth Forms | ，(U+FF0C) | Anyuuus |

Các ký tự Vietnamese hợp lệ được giữ lại:
- ASCII printable (space – tilde)
- Latin-1 Supplement, Latin Extended-A/B (Ă, Đ, Ơ, Ư, v.v.)
- Latin Extended Additional (Ạ–ỹ: 134 ký tự dấu tiếng Việt)
- General Punctuation (–, —, ', ', ", ", …)
- Superscripts (², ³), Currency (₫)

**Bước 3b — Lọc kích thước ảnh:**

| Level | Filter | Giá trị |
|---|---|---|
| Word | min_width, min_height | 10px |
| Word | max_height | 300px |
| Word | aspect ratio (W/H) | 0.3 – 25.0 |
| Line | min_width | 32px |
| Line | min_height | 16px |
| Line | max_height | 384px |
| Line | max_width | 3000px |
| Line | min aspect ratio (W/H) | 1.0 |

> **Note**: Lọc ngôn ngữ chạy **trước** lọc kích thước để tránh mở ảnh không cần thiết (tiết kiệm I/O).

### Bước 4 — Xuất LMDB

```bash
python scripts/04_export_lmdb.py
```

> **Note**: Script tự động ưu tiên `labels_filtered.csv`. Nếu chưa chạy bước 3, sẽ dùng `labels_split.csv`.

---

## Các bộ dữ liệu

### Chữ viết tay (Handwritten)

| Dataset | Level | Số mẫu | Ghi chú |
|---|---|---|---|
| UIT_HWDB | word / line / paragraph | ~110k / ~7k / ~1k | 249 người viết, có writer ID |
| Cinnamon AI | line | ~2.385 | Địa chỉ thực tế, chữ ngoáy khó |
| Viet-Wiki-Handwriting | paragraph | ~5.796 | Synthetic từ Wikipedia tiếng Việt |

### Chữ in (Printed)

| Dataset | Level | Ghi chú |
|---|---|---|
| VinText_Cropped | word | Chữ in từ ảnh thực tế |
| Vietnamese Receipts MC_OCR 2021 | line | Hóa đơn, biên lai |
| Anyuuus – PaddleOCR | line | Văn bản scan (ngôn ngữ lịch sử) |
| **Synthetic_Modern** | **line** | **Synthetic printed, 46k corpus hiện đại** |

---

## Schema CSV

Mỗi file `labels_master.csv` (output của Script 1), `labels_split.csv` (output của Script 2), và `labels_filtered.csv` (output của Script 3) có cấu trúc:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `filename` | string | Tên file ảnh trong `processed/{level}/images/` |
| `label` | string | Nhãn văn bản đã chuẩn hóa NFC |
| `source` | string | Định danh nguồn (`uit_word`, `cinnamon_d2`, `mcocr`, ...) |
| `level` | string | Cấp độ: `word` / `line` / `paragraph` |
| `data_type` | string | `handwritten` hoặc `printed` |
| `writer_id` | string | ID người viết (nếu có), dùng cho writer-independent split |
| `pre_split` | string | Gợi ý chia từ cấu trúc thư mục gốc (xem bảng bên dưới) |
| `final_split` | string | *(Có trong labels_split.csv và labels_filtered.csv)* `train` / `val` / `test` |

### Giá trị `pre_split`

| Giá trị | Nguồn | Xử lý trong Script 2 |
|---|---|---|
| `train` | Cinnamon Data2, VinText train, MC_OCR train | Giữ nguyên |
| `val` | Cinnamon Data1, VinText test | Giữ nguyên |
| `test` | Cinnamon Private_Test, UIT test_data, VinText unseen | Giữ nguyên |
| `train_pool` | UIT_HWDB train_data | Writer 1–229 → train, Writer 230–249 → val |
| `unassigned` | Viet_Wiki | Chia ngẫu nhiên 80/10/10 |
| `anyuuus_pool` | Anyuuus | Group-split theo Document ID 80/10/10 |
| `synthetic_pool` | Synthetic_Modern | Chia ngẫu nhiên 90/5/5 |

---

## Chiến lược phân chia dữ liệu

### Writer-Independent Split (UIT_HWDB)

Mỗi người viết chỉ xuất hiện trong đúng một tập (train hoặc val hoặc test). Điều này đảm bảo model được đánh giá trên chữ viết của những người chưa từng thấy trong quá trình huấn luyện.

```
Writer ID  1 – 229  →  train  (91.6%)
Writer ID 230 – 249  →  val   (8.0%)
Writer ID 250 – 255  →  test  (tập test_data của tác giả)
```

### Group Split (Anyuuus)

Tất cả dòng văn bản crop từ cùng một tài liệu được giữ trong cùng một tập. Tránh trường hợp model thấy các dòng khác của cùng một văn bản trong training, nhưng lại được test trên một dòng còn lại của chính văn bản đó (data leakage theo tài liệu).

### Random Split (Viet_Wiki)

Dữ liệu synthetic không có thông tin người viết, dùng `train_test_split(random_state=42)` chia 80/10/10. `random_state=42` đảm bảo kết quả tái lập hoàn toàn.

### Random Split (Synthetic_Modern)  

Dữ liệu synthetic sạch, không có rủi ro data leakage. Tỷ lệ **90/5/5** (thay vì 80/10/10) để tối đa lượng dữ liệu hiện đại trong training, pha loãng ngôn ngữ lịch sử từ Anyuuus.

---

## Data Filtering

Script `03_filter_outliers.py` áp dụng **2 bộ lọc** trước khi export LMDB:

### Lọc ngôn ngữ (non-Vietnamese character removal)

Loại bỏ toàn bộ label+image nếu label chứa **bất kỳ ký tự nào** ngoài bộ ký tự tiếng Việt hợp lệ. Bộ lọc sử dụng whitelist gồm 10 dải Unicode (xem `ALLOWED_RANGES` trong script), bao phủ toàn bộ 256 ký tự trong `final_vietnamese_vocab.txt`.

Các script phổ biến bị loại:
- **CJK** (~109 ký tự): Chinese Hán tự, Japanese Kanji — từ Anyuuus (văn bản lịch sử chữ Nôm) và Synthetic_Modern
- **Khmer** (~13 ký tự): từ Synthetic_Modern corpus
- **IPA** (~8 ký tự): ký hiệu phiên âm quốc tế ɑ, ə, ɛ
- **Cyrillic** (2 ký tự): Т, У
- **Fullwidth** (1 ký tự): ， (U+FF0C, fullwidth comma)

### Lọc kích thước ảnh (dimension filters)

- **Word level**: Loại ảnh < 10px, > 300px height, aspect ratio ngoài 0.3–25.0
- **Line level**: Loại ảnh < 32px width, < 16px height, > 384px height, > 3000px width, ảnh vuông/dọc (aspect < 1.0)

---

## Cấu trúc LMDB

Mỗi LMDB database tại `lmdb/{level}/{split}/` lưu các key-value theo convention của [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark):

```
image-00000001  →  bytes ảnh PNG (RGB, đọc trực tiếp từ disk)
label-00000001  →  nhãn text encode UTF-8
image-00000002  →  ...
label-00000002  →  ...
num-samples     →  tổng số mẫu hợp lệ (string)
```

Cả hai level `word` và `line` đều được tách thành hai LMDB riêng biệt theo `data_type` (`printed` / `handwritten`). Điều này cho phép kiểm soát tỷ lệ dữ liệu và lịch trình fine-tune linh hoạt hơn cho từng loại văn bản.

Level `paragraph` phục vụ cho DBNet++ text detector (dùng pre-trained, không cần export LMDB).

### Ánh xạ level LMDB → thư mục ảnh gốc

| Level LMDB | CSV nguồn | Thư mục ảnh |
|---|---|---|
| `word_printed` | `processed/word/labels_filtered.csv` | `processed/word/images/` |
| `word_handwritten` | `processed/word/labels_filtered.csv` | `processed/word/images/` |
| `line_printed` | `processed/line/labels_filtered.csv` | `processed/line/images/` |
| `line_handwritten` | `processed/line/labels_filtered.csv` | `processed/line/images/` |

### Sử dụng trong training TrOCR

```
Stage 1a - Pre-warm encoder (Printed)    : lmdb/word_printed/train/
Stage 1b - Pre-warm encoder (Handwritten): lmdb/word_handwritten/train/
Stage 2a - Printed fine-tune             : lmdb/line_printed/train/
Stage 2b - Handwritten adapt             : lmdb/line_handwritten/train/
```

---

## Vocabulary

Có 2 file vocab phục vụ các mục đích khác nhau:

### `vocab/vietnamese_vocab.txt` (raw, ~409 ký tự)

File **tự động tạo** bởi Script 01 — chứa **toàn bộ** ký tự xuất hiện trong data, bao gồm cả ký tự ngoại lai (CJK, Khmer, Cyrillic, IPA). Dùng để kiểm tra, debug, và phân tích dữ liệu.

> **WARNING**: `save_vocab()` chỉ ghi nhận ký tự từ các parser được chạy trong lần gọi hiện tại. Luôn chạy `01_parse_all_datasets.py all` để có vocab đầy đủ.

### `final_vietnamese_vocab.txt` (curated, 256 ký tự) ← **Dùng cho TrOCR tokenizer**

File **đã xử lý thủ công** — chỉ chứa các ký tự tiếng Việt hợp lệ. Đây là file chính thức để cấu hình character-level tokenizer cho TrOCR.

**Thống kê:**

| Nhóm | Số ký tự | Chi tiết |
|---|---|---|
| ASCII printable | 95 | Space, digits 0–9, A–Z, a–z, punctuation |
| Vietnamese diacritics | 134 | 67 uppercase + 67 lowercase (Ạ–ỹ, Ă–ặ, Đ–đ, Ơ–ợ, Ư–ự) |
| Other symbols | 27 | °, ², ³, –, —, ', ', ", ", ₫, ⁰, ⁴–⁹, v.v. |
| **Tổng** | **256** | |

**Validation đã kiểm tra:**
- ✅ 134/134 Vietnamese diacritics đầy đủ (tất cả tổ hợp nguyên âm + dấu thanh)
- ✅ Digits 0–9, ASCII A–Z/a–z đầy đủ
- ✅ Ký hiệu tiền tệ ₫ (Vietnamese đồng) có mặt
- ✅ Không chứa CJK, Khmer, Thai, Cyrillic, IPA
- ✅ Tất cả 256 ký tự đều nằm trong `ALLOWED_RANGES` của `03_filter_outliers.py`

**Sử dụng cho TrOCR tokenizer:**
```
Vocab size: 256 characters + 4 special tokens = 260
Special tokens: <s> (BOS), </s> (EOS), <pad>, <unk>
```

Tokenizer sẽ thay thế RoBERTa BPE tokenizer (50,265 tokens) bằng character-level tokenizer với 260 tokens — giảm embedding layer từ ~38M params xuống ~0.2M params.

---

## Lưu ý kỹ thuật

**Unicode NFC** — Tất cả label đều được chuẩn hóa về Unicode NFC trước khi lưu. Tiếng Việt có thể biểu diễn dấu thanh theo 2 cách (NFC: `ộ` = 1 codepoint; NFD: `ộ` = 3 codepoint). Không chuẩn hóa sẽ khiến CER bị tính sai ngay cả khi model nhận dạng đúng về mặt thị giác.

**Reproducibility** — Tất cả phép chia ngẫu nhiên đều dùng `random_state=42` (sklearn). Chạy lại bất kỳ lúc nào cũng cho kết quả split giống hệt nhau.

**Duplicate Prevention** — Script `01_parse_all_datasets.py` tự động xóa tất cả `labels_master.csv` trước khi chạy, ngăn chặn duplicate khi chạy lại.

**No Double Encoding** — `04_export_lmdb.py` đọc bytes ảnh PNG trực tiếp từ disk thay vì re-encode qua PIL, tiết kiệm ~220k decode+encode cycles. Bước validation (100 ảnh đầu tiên) kiểm tra format consistency.

**Windows vs Linux** — LMDB trên Linux tự động mở rộng `map_size`. Trên Windows, `map_size` phải được khai báo đủ lớn từ đầu. Script tính `map_size` dựa trên dung lượng file thực tế trên disk nhân với `MAP_SIZE_SAFETY_FACTOR` (mặc định 3.0x), kết hợp cơ chế retry tự động tăng gấp đôi nếu gặp lỗi `MDB_MAP_FULL`.
