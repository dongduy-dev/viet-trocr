# -*- coding: utf-8 -*-
"""
01_build_corpus.py
==================
Build a large, diverse modern Vietnamese text corpus for synthetic data generation.

Corpus sources (in priority order):
  1. Vietnamese Wikipedia articles (auto-scraped, ~8000-15000 sentences)
  2. Built-in sentences across 7 domains (~260 hand-crafted)
  3. Generated addresses and name records (~800)
  4. Niits/vietnamese-legal-ocr clean labels (~600)

The key insight: the DECODER needs TEXT diversity (unique sentences),
not just IMAGE diversity (fonts/augmentations). 1,500 unique sentences
is not enough to counter 10,000+ unique anyuuus lines.
Target: 8,000-15,000 unique sentences.

Output: corpus/modern_vietnamese.txt (one sentence per line, NFC normalized)

Chay: python 01_build_corpus.py [--skip-wiki] [--min-sentences 8000]
"""
import os
import sys
import re
import json
import random
import argparse
import unicodedata
import urllib.request
import urllib.parse
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = PROJECT_ROOT / "corpus"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: VIETNAMESE WIKIPEDIA SCRAPER
# ─────────────────────────────────────────────────────────────────────────────

# Curated seed categories covering diverse modern Vietnamese topics
WIKI_SEED_ARTICLES = [
    # Geography / Cities
    "Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Huế", "Nha Trang", "Đà Lạt", "Vũng Tàu", "Phú Quốc",
    "Vịnh Hạ Long", "Phố cổ Hội An", "Sapa", "Mũi Né", "Côn Đảo",
    # History
    "Lịch sử Việt Nam", "Chiến tranh Việt Nam", "Đổi Mới",
    "Triều Nguyễn", "Cách mạng tháng Tám",
    # Politics / Government
    "Quốc hội Việt Nam", "Chính phủ Việt Nam", "Hiến pháp Việt Nam",
    "Đảng Cộng sản Việt Nam",
    # Economy
    "Kinh tế Việt Nam", "Ngân hàng Nhà nước Việt Nam",
    "Sàn giao dịch chứng khoán Thành phố Hồ Chí Minh",
    "Tập đoàn Vingroup", "FPT (công ty)",
    # Education
    "Đại học Quốc gia Hà Nội", "Đại học Bách khoa Hà Nội",
    "Trường Đại học Công nghệ Thông tin",
    "Giáo dục Việt Nam", "Kỳ thi tốt nghiệp trung học phổ thông",
    # Science & Tech
    "Trí tuệ nhân tạo", "Học máy", "Internet", "Điện thoại thông minh",
    "Năng lượng tái tạo", "Biến đổi khí hậu", "COVID-19 tại Việt Nam",
    # Culture
    "Ẩm thực Việt Nam", "Phở", "Bánh mì", "Áo dài", "Tết Nguyên Đán",
    "Nhã nhạc cung đình Huế", "Văn học Việt Nam",
    "Múa rối nước", "Ca trù",
    # Sports
    "Đội tuyển bóng đá quốc gia Việt Nam", "SEA Games",
    "Thể thao Việt Nam",
    # Law
    "Bộ luật Dân sự Việt Nam", "Luật Doanh nghiệp",
    # Transport
    "Giao thông Việt Nam", "Cảng hàng không quốc tế Tân Sơn Nhất",
    "Đường sắt Việt Nam",
    # Nature
    "Sông Mê Kông", "Đồng bằng sông Cửu Long", "Vườn quốc gia Cát Bà",
    # People
    "Hồ Chí Minh", "Võ Nguyên Giáp", "Nguyễn Du",
    "Trịnh Công Sơn",
    # Health
    "Y tế Việt Nam", "Bệnh viện Bạch Mai",
    # Provinces (additional coverage)
    "Quảng Ninh", "Nghệ An", "Thanh Hóa", "Bình Dương", "Đồng Nai",
    "Lâm Đồng", "Khánh Hòa", "Bắc Ninh", "Hải Dương",
]


def fetch_wiki_article(title: str) -> str | None:
    """
    Fetch plain text of a Vietnamese Wikipedia article via the API.
    Returns the article text or None on failure.
    """
    url = (
        "https://vi.wikipedia.org/api/rest_v1/page/html/"
        + urllib.parse.quote(title.replace(" ", "_"))
    )

    # Fallback to MediaWiki API for plain text extraction
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "exlimit": "1",
        "format": "json",
    }
    api_url = "https://vi.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(api_url, headers={
            "User-Agent": "TrOCR-SyntheticData/1.0 (Vietnamese OCR research project)"
        })
        response = urllib.request.urlopen(req, timeout=15)
        data = json.loads(response.read().decode("utf-8"))

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None
            return page_data.get("extract", "")

    except Exception:
        return None


def extract_sentences(text: str) -> list[str]:
    """
    Extract clean sentences from Wikipedia article text.
    Filters out:
      - Section headers (== ... ==)
      - Reference markers
      - Very short/long lines
      - Lines that are mostly numbers or symbols
      - Empty lines
    """
    if not text:
        return []

    sentences = []
    # Split on common sentence boundaries
    # Vietnamese sentences typically end with . ! ? or are on separate lines
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        # Skip section headers
        if line.startswith("==") or line.startswith("{{") or line.startswith("|"):
            continue
        # Skip empty lines
        if not line:
            continue

        # Split line into sentences on . ? !
        # But be careful with abbreviations like "TP.", "GS.", "PGS.", etc.
        parts = re.split(r'(?<=[.!?])\s+', line)

        for part in parts:
            part = part.strip()
            # Clean up
            part = re.sub(r'\[\d+\]', '', part)  # Remove [1], [2], etc.
            part = re.sub(r'\s+', ' ', part).strip()

            # Quality filters
            if len(part) < 8:
                continue
            if len(part) > 200:
                # Try to split further on commas for very long sentences
                sub_parts = re.split(r',\s+', part)
                for sp in sub_parts:
                    sp = sp.strip()
                    if 8 <= len(sp) <= 200:
                        # Must contain at least some Vietnamese chars
                        viet_chars = sum(1 for c in sp if ord(c) > 127)
                        if viet_chars >= 2:
                            sentences.append(sp)
                continue

            # Must contain at least some Vietnamese-like characters
            viet_chars = sum(1 for c in part if ord(c) > 127)
            alpha_chars = sum(1 for c in part if c.isalpha())
            if viet_chars < 2 or alpha_chars < len(part) * 0.3:
                continue

            # Skip lines that are mostly parenthetical references
            if part.count('(') > 3 or part.count(')') > 3:
                continue

            sentences.append(part)

    return sentences


def scrape_wikipedia(max_articles: int = 80) -> list[str]:
    """Scrape Vietnamese Wikipedia articles and extract sentences."""
    all_sentences = []
    articles_fetched = 0

    articles = WIKI_SEED_ARTICLES[:max_articles]

    print(f"\n  Scraping {len(articles)} Vietnamese Wikipedia articles...")

    for i, title in enumerate(articles):
        text = fetch_wiki_article(title)
        if text:
            sents = extract_sentences(text)
            all_sentences.extend(sents)
            articles_fetched += 1
            status = f"{len(sents)} sentences"
        else:
            status = "FAILED"

        if (i + 1) % 10 == 0 or i == len(articles) - 1:
            print(f"    [{i+1}/{len(articles)}] Total: {len(all_sentences)} sentences "
                  f"from {articles_fetched} articles")

        # Polite delay
        time.sleep(0.3)

    print(f"  Wikipedia scraping complete: {len(all_sentences)} sentences "
          f"from {articles_fetched} articles")

    return all_sentences


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BUILT-IN CORPUS (hand-crafted, guaranteed quality)
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_SENTENCES = [
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "Độc lập - Tự do - Hạnh phúc",
    "Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam",
    "Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015",
    "Căn cứ Luật Doanh nghiệp ngày 17 tháng 6 năm 2020",
    "Căn cứ Bộ luật Lao động ngày 20 tháng 11 năm 2019",
    "Theo đề nghị của Bộ trưởng Bộ Tài chính",
    "Chính phủ ban hành Nghị định quy định chi tiết",
    "Điều 1. Phạm vi điều chỉnh",
    "Điều 2. Đối tượng áp dụng",
    "Điều 3. Giải thích từ ngữ",
    "Điều 4. Nguyên tắc hoạt động",
    "Nghị định này quy định chi tiết một số điều",
    "Thông tư này hướng dẫn thi hành Nghị định",
    "Quyết định này có hiệu lực thi hành kể từ ngày ký",
    "Bộ trưởng, Thủ trưởng cơ quan ngang bộ",
    "Chủ tịch Ủy ban nhân dân các cấp",
    "Các Bộ, cơ quan ngang Bộ, cơ quan thuộc Chính phủ",
    "Quy định về quản lý và sử dụng ngân sách nhà nước",
    "Hướng dẫn thực hiện chính sách bảo hiểm xã hội",
    "Quy định xử phạt vi phạm hành chính trong lĩnh vực",
    "Thủ tục đăng ký kinh doanh theo quy định của pháp luật",
    "Quyền và nghĩa vụ của công dân theo Hiến pháp",
    "Bảo vệ quyền lợi người tiêu dùng",
    "Luật Bảo vệ môi trường năm 2020",
    "Quy định về an toàn vệ sinh thực phẩm",
    "Nghị định về quản lý hoạt động xây dựng",
    "Thông tư hướng dẫn về thuế thu nhập doanh nghiệp",
    "Quy chế làm việc của Ủy ban nhân dân tỉnh",
    "Kế hoạch phát triển kinh tế xã hội năm 2025",
    "Báo cáo tổng kết công tác năm 2024",
    "Dự toán ngân sách nhà nước năm 2025",
    "Phê duyệt quy hoạch tổng thể phát triển",
    "Đề án đổi mới giáo dục và đào tạo",
    "Chương trình mục tiêu quốc gia xây dựng nông thôn mới",
    "Quy định về tiêu chuẩn chức danh nghề nghiệp",
    "Hướng dẫn đánh giá xếp loại cán bộ công chức",
    "Quy trình giải quyết thủ tục hành chính",
    "Cải cách hành chính nhà nước giai đoạn 2021 đến 2030",
    "Chiến lược phát triển bền vững Việt Nam",
    "Chương trình chuyển đổi số quốc gia đến năm 2025",
    "BỘ GIÁO DỤC VÀ ĐÀO TẠO",
    "BỘ KHOA HỌC VÀ CÔNG NGHỆ",
    "BỘ TÀI NGUYÊN VÀ MÔI TRƯỜNG",
    "BỘ NÔNG NGHIỆP VÀ PHÁT TRIỂN NÔNG THÔN",
    "BỘ GIAO THÔNG VẬN TẢI",
    "BỘ XÂY DỰNG",
    "BỘ Y TẾ",
    "BỘ CÔNG THƯƠNG",
    "BỘ VĂN HÓA, THỂ THAO VÀ DU LỊCH",
    "NGÂN HÀNG NHÀ NƯỚC VIỆT NAM",
    "ỦY BAN NHÂN DÂN THÀNH PHỐ HỒ CHÍ MINH",
    "ỦY BAN NHÂN DÂN THÀNH PHỐ HÀ NỘI",
    "Số: 01/2025/NĐ-CP Hà Nội, ngày 15 tháng 1 năm 2025",
    "Số: 25/2025/TT-BTC Hà Nội, ngày 20 tháng 3 năm 2025",
    "THÔNG TƯ", "NGHỊ ĐỊNH", "QUYẾT ĐỊNH", "CHỈ THỊ", "CÔNG VĂN",
]

NEWS_SENTENCES = [
    "Thủ tướng Chính phủ chủ trì phiên họp thường kỳ",
    "Quốc hội thông qua Luật Đất đai sửa đổi",
    "Thành phố Hồ Chí Minh đẩy mạnh chuyển đổi số",
    "Việt Nam tăng trưởng GDP đạt 6,5 phần trăm",
    "Xuất khẩu nông sản Việt Nam đạt kỷ lục mới",
    "Ngành du lịch phục hồi mạnh mẽ sau đại dịch",
    "Hà Nội triển khai dự án metro số 3",
    "Đà Nẵng phát triển thành trung tâm công nghệ",
    "Cần Thơ xây dựng đô thị thông minh",
    "Hải Phòng đón làn sóng đầu tư nước ngoài",
    "Tỷ lệ thất nghiệp giảm xuống mức thấp nhất",
    "Chỉ số giá tiêu dùng tăng nhẹ trong tháng qua",
    "Ngân hàng Nhà nước điều chỉnh lãi suất điều hành",
    "Thị trường chứng khoán Việt Nam tăng trưởng tích cực",
    "Doanh nghiệp nhỏ và vừa được hỗ trợ tiếp cận vốn",
    "Chính phủ ban hành gói hỗ trợ kinh tế mới",
    "Bộ Giáo dục công bố phương án thi tốt nghiệp",
    "Đại học Quốc gia Hà Nội lọt top 500 thế giới",
    "Học sinh Việt Nam đoạt huy chương vàng Olympic",
    "Chương trình giáo dục phổ thông mới được triển khai",
    "Bệnh viện Bạch Mai ứng dụng trí tuệ nhân tạo",
    "Việt Nam sản xuất thành công vaccine trong nước",
    "Dự án cao tốc Bắc Nam hoàn thành giai đoạn một",
    "Sân bay Long Thành khởi công xây dựng nhà ga",
    "Việt Nam phóng thành công vệ tinh quan sát Trái Đất",
    "Năng lượng tái tạo chiếm tỷ trọng ngày càng lớn",
    "Nông nghiệp công nghệ cao đóng góp lớn cho kinh tế",
    "Thương mại điện tử tăng trưởng nhanh nhất khu vực",
    "Bão số 3 đổ bộ vào vùng biển Quảng Ninh đến Hải Phòng",
    "Nhiệt độ tại Hà Nội dao động từ 25 đến 33 độ C",
    "Lũ quét gây thiệt hại nặng nề tại các tỉnh miền núi phía Bắc",
]

EDUCATION_SENTENCES = [
    "Trường Đại học Bách khoa Hà Nội",
    "Trường Đại học Công nghệ Thông tin",
    "Đại học Quốc gia Thành phố Hồ Chí Minh",
    "Học viện Kỹ thuật Quân sự",
    "Trường Đại học Kinh tế Quốc dân",
    "Luận văn tốt nghiệp đại học ngành Khoa học Máy tính",
    "Giảng viên hướng dẫn: Tiến sĩ Nguyễn Văn An",
    "Sinh viên thực hiện: Trần Thị Bích Ngọc",
    "Bảng điểm học tập toàn khóa",
    "Mã số sinh viên: 20210001",
    "Niên khóa 2021 đến 2025",
    "Hệ đào tạo: Chính quy",
    "Ngành đào tạo: Kỹ thuật Phần mềm",
    "Tổng số tín chỉ tích lũy: 150",
    "Điểm trung bình tích lũy: 3.45",
    "Xếp loại tốt nghiệp: Giỏi",
    "Chứng chỉ tiếng Anh IELTS 6.5",
    "Nghiên cứu ứng dụng học sâu trong nhận dạng hình ảnh",
    "Hội nghị khoa học sinh viên toàn quốc",
    "Cuộc thi Lập trình Sinh viên Quốc tế ACM ICPC",
]

BUSINESS_SENTENCES = [
    "Công ty Cổ phần Công nghệ Việt Nam",
    "Giấy chứng nhận đăng ký doanh nghiệp",
    "Mã số thuế: 0123456789",
    "Địa chỉ: Số 123, Đường Nguyễn Huệ, Quận 1, TP. HCM",
    "Điện thoại: (028) 3822 1234",
    "Email: contact@congty.com.vn",
    "Vốn điều lệ: 10.000.000.000 đồng",
    "Người đại diện theo pháp luật: Ông Nguyễn Văn Minh",
    "HỢP ĐỒNG LAO ĐỘNG",
    "HỢP ĐỒNG MUA BÁN HÀNG HÓA",
    "Bên A (Bên bán): Công ty TNHH ABC",
    "Giá trị hợp đồng: 500.000.000 đồng",
    "Phương thức thanh toán: Chuyển khoản ngân hàng",
    "BÁO CÁO TÀI CHÍNH NĂM 2024",
    "Bảng cân đối kế toán",
    "Tổng doanh thu: 125.678.900.000 đồng",
    "Lợi nhuận sau thuế: 12.187.600.000 đồng",
    "BIÊN BẢN HỌP ĐẠI HỘI ĐỒNG CỔ ĐÔNG",
    "Nghị quyết Đại hội đồng cổ đông thường niên",
]

EVERYDAY_SENTENCES = [
    "Xin chào, tôi tên là Nguyễn Văn An",
    "Rất vui được gặp anh chị hôm nay",
    "Cảm ơn bạn đã giúp đỡ tôi rất nhiều",
    "Hôm nay thời tiết rất đẹp phải không?",
    "Phở bò tái nạm gầu là món ăn truyền thống",
    "Bánh mì thịt nướng là bữa sáng phổ biến",
    "Bún chả Hà Nội nổi tiếng khắp thế giới",
    "Vịnh Hạ Long là di sản thiên nhiên thế giới",
    "Phố cổ Hội An lung linh ánh đèn lồng",
    "Đà Lạt được mệnh danh là thành phố ngàn hoa",
    "Tết Nguyên Đán là ngày lễ lớn nhất Việt Nam",
    "Vui lòng điền đầy đủ thông tin vào mẫu đơn",
    "Kết quả sẽ được công bố sau hai tuần làm việc",
]

SCIENCE_TECH_SENTENCES = [
    "Trí tuệ nhân tạo đang thay đổi mọi lĩnh vực",
    "Học máy và học sâu là nền tảng của AI hiện đại",
    "Mạng nơ-ron tích chập được sử dụng trong thị giác máy tính",
    "Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu tiếng Việt",
    "Dữ liệu lớn và phân tích dữ liệu đóng vai trò quan trọng",
    "Điện toán đám mây giúp doanh nghiệp tiết kiệm chi phí",
    "An ninh mạng là thách thức lớn trong kỷ nguyên số",
    "Phần mềm mã nguồn mở phổ biến trong cộng đồng phát triển",
    "Công nghệ 5G mang lại tốc độ truyền dữ liệu vượt trội",
    "Năng lượng mặt trời là giải pháp xanh cho tương lai",
]

NUMERIC_MIXED = [
    "Tổng cộng: 1.250.000 VNĐ",
    "Số lượng: 150 cái x 25.000 đồng/cái",
    "Thuế GTGT (10%): 125.000 đồng",
    "Thành tiền: 2.345.678 VNĐ",
    "Ngày: 15/04/2025",
    "Hóa đơn số: HD-2025-001234",
    "SĐT: 0901 234 567",
    "CMND/CCCD: 079123456789",
    "Mã số thuế: 0312345678",
    "Số tài khoản: 1234567890123",
    "Ngân hàng: Vietcombank - CN Sở Giao dịch",
    "Ngân hàng TMCP Ngoại thương Việt Nam",
    "Ngân hàng TMCP Công Thương Việt Nam",
    "Ngân hàng TMCP Đầu tư và Phát triển Việt Nam",
    "Diện tích: 125,5 m2",
    "Nhiệt độ: 37,5 độ C",
    "Tốc độ: 120 km/h",
    "Dân số: 100.000.000 người",
    "GDP bình quân đầu người: 4.200 USD",
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: GENERATED DATA (addresses, names)
# ─────────────────────────────────────────────────────────────────────────────

ADDRESS_TEMPLATES = [
    "Số {num}, Đường {street}, Phường {ward}, Quận {district}, {city}",
    "Số {num} {street}, {ward}, {district}, {city}",
    "{num}/{sub} {street}, Phường {ward}, Quận {district}",
    "Tầng {floor}, Tòa nhà {building}, Số {num} {street}, {city}",
]

STREETS = [
    "Nguyễn Huệ", "Lê Lợi", "Trần Hưng Đạo", "Lý Thường Kiệt",
    "Nguyễn Trãi", "Hai Bà Trưng", "Lê Duẩn", "Điện Biên Phủ",
    "Nguyễn Thị Minh Khai", "Võ Văn Tần", "Pasteur", "Tôn Đức Thắng",
    "Phạm Ngọc Thạch", "Cách Mạng Tháng 8", "Nam Kỳ Khởi Nghĩa",
    "Nguyễn Đình Chiểu", "Hoàng Văn Thụ", "Trường Chinh", "Quang Trung",
    "Lê Văn Sỹ", "Nguyễn Văn Cừ", "Phạm Văn Đồng", "Võ Nguyên Giáp",
    "Đại Cồ Việt", "Giải Phóng", "Trần Đại Nghĩa", "Tạ Quang Bửu",
    "Bạch Mai", "Phố Huế", "Hàng Bài", "Tràng Tiền", "Hàng Khay",
]

WARDS = [
    "Bến Nghé", "Bến Thành", "Phạm Ngũ Lão", "Tân Định",
    "Đa Kao", "Trung Hòa", "Nhân Chính", "Thanh Xuân",
    "Láng Hạ", "Thành Công", "Kim Mã", "Cống Vị",
    "Hàng Bông", "Tràng Tiền", "Phan Chu Trinh", "Quán Thánh",
]

DISTRICTS = [
    "1", "3", "5", "7", "10", "Bình Thạnh", "Phú Nhuận", "Tân Bình",
    "Ba Đình", "Hoàn Kiếm", "Đống Đa", "Hai Bà Trưng", "Cầu Giấy",
    "Thanh Xuân", "Hải Châu", "Sơn Trà", "Ninh Kiều", "Hồng Bàng",
]

CITIES = [
    "TP. Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ",
    "Hải Phòng", "Huế", "Nha Trang", "Vũng Tàu", "Biên Hòa",
]

BUILDINGS = [
    "Bitexco", "Landmark 81", "Vincom Center", "Saigon Trade Center",
    "Keangnam", "Lotte Center", "Diamond Plaza", "Times City",
    "Vinhomes Central Park", "Masteri Thảo Điền", "Sunrise City",
]

PERSON_NAMES = [
    "Nguyễn Văn An", "Trần Thị Bích Ngọc", "Lê Hoàng Minh",
    "Phạm Thị Hồng Nhung", "Hoàng Đức Tuấn", "Vũ Thị Mai Anh",
    "Đặng Quốc Hùng", "Bùi Thị Thanh Hằng", "Đỗ Minh Quang",
    "Ngô Thị Kim Oanh", "Dương Văn Thắng", "Lý Thị Phương Thảo",
    "Trịnh Xuân Hải", "Đinh Thị Ngọc Lan", "Phan Văn Đức",
    "Hồ Thị Mỹ Linh", "Tô Thanh Sơn", "Lương Thị Diệu Hiền",
    "Châu Minh Tâm", "Mai Thị Tuyết Nga", "Võ Hoàng Long",
]


def generate_addresses(n: int = 500) -> list[str]:
    """Generate random Vietnamese addresses."""
    random.seed(42)
    result = []
    for _ in range(n):
        tmpl = random.choice(ADDRESS_TEMPLATES)
        addr = tmpl.format(
            num=random.randint(1, 999),
            sub=random.randint(1, 50),
            street=random.choice(STREETS),
            ward=random.choice(WARDS),
            district=random.choice(DISTRICTS),
            city=random.choice(CITIES),
            floor=random.randint(1, 30),
            building=random.choice(BUILDINGS),
        )
        result.append(addr)
    return result


def generate_name_records(n: int = 200) -> list[str]:
    """Generate name + attribute combinations."""
    random.seed(43)
    titles = ["Ông", "Bà", "Anh", "Chị", "Tiến sĩ", "Thạc sĩ", "Kỹ sư", "Bác sĩ"]
    positions = [
        "Giám đốc", "Phó Giám đốc", "Trưởng phòng", "Phó phòng",
        "Chuyên viên", "Nhân viên", "Kế toán trưởng", "Thư ký",
        "Giảng viên", "Nghiên cứu sinh", "Sinh viên", "Giáo sư",
    ]
    result = []
    for _ in range(n):
        name = random.choice(PERSON_NAMES)
        fmt = random.choice([
            f"Họ và tên: {name}",
            f"{random.choice(titles)} {name}",
            f"{name} - {random.choice(positions)}",
            f"Người ký: {random.choice(titles)} {name}",
            f"{random.choice(positions)}: {name}",
        ])
        result.append(fmt)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: POST-PROCESSING (length control for TrOCR compatibility)
# ─────────────────────────────────────────────────────────────────────────────

# Vietnamese conjunctions / natural break points
SPLIT_PATTERNS = [
    r',\s+',                           # comma + space
    r'\s+và\s+',                       # "và" (and)
    r'\s+hoặc\s+',                     # "hoặc" (or)
    r'\s+nhưng\s+',                    # "nhưng" (but)
    r'\s+cũng\s+như\s+',              # "cũng như" (as well as)
    r'\s+trong\s+đó\s+',              # "trong đó" (in which)
    r'\s+bao\s+gồm\s+',              # "bao gồm" (including)
    r'\s+ngoài\s+ra\s+',             # "ngoài ra" (besides)
    r';\s+',                           # semicolon
    r'\s*-\s+',                        # dash
    r'\s+do\s+',                       # "do" (because/by)
    r'\s+với\s+',                      # "với" (with)
    r'\s+theo\s+',                     # "theo" (according to)
]


def split_long_sentence(text: str, max_chars: int) -> list[str]:
    """
    Split a long sentence into shorter segments at natural break points.
    Returns list of segments, each <= max_chars.
    Preserves diversity: a 150-char sentence becomes 2-3 shorter segments,
    not thrown away.
    """
    if len(text) <= max_chars:
        return [text]

    # Try each split pattern, starting with strongest breaks
    for pattern in SPLIT_PATTERNS:
        parts = re.split(pattern, text)
        if len(parts) > 1:
            # Recombine parts to fit within max_chars
            segments = []
            current = parts[0].strip()
            for part in parts[1:]:
                part = part.strip()
                if not part:
                    continue
                if len(current) + len(part) + 2 <= max_chars:
                    current = current + ", " + part
                else:
                    if current and len(current) >= 8:
                        segments.append(current)
                    current = part
            if current and len(current) >= 8:
                segments.append(current)

            # Check if we have valid segments
            if segments and all(len(s) <= max_chars for s in segments):
                return segments

    # Last resort: hard split at max_chars boundary on word boundaries
    words = text.split()
    segments = []
    current = ""
    for word in words:
        test = (current + " " + word).strip() if current else word
        if len(test) <= max_chars:
            current = test
        else:
            if current and len(current) >= 8:
                segments.append(current)
            current = word
    if current and len(current) >= 8:
        segments.append(current)

    return segments if segments else [text[:max_chars]]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MAIN BUILD LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(skip_wiki: bool = False, min_sentences: int = 5000,
                 max_chars: int = 70):
    """Build the complete Vietnamese corpus."""
    print("=" * 60)
    print("01_build_corpus.py  -  Building Vietnamese corpus")
    print("=" * 60)
    print(f"  Max chars per line: {max_chars} "
          f"(critical for TrOCR 384x384 input)")

    all_sentences = []

    # ── Wikipedia (the big one) ──────────────────────────────────────────
    if not skip_wiki:
        wiki_cache = CORPUS_DIR / "wiki_cache.txt"
        if wiki_cache.exists():
            print(f"\n  Found cached Wikipedia sentences at {wiki_cache}")
            with open(wiki_cache, "r", encoding="utf-8") as f:
                wiki_sents = [l.strip() for l in f if l.strip()]
            print(f"  Loaded {len(wiki_sents)} cached sentences")
        else:
            wiki_sents = scrape_wikipedia(max_articles=len(WIKI_SEED_ARTICLES))
            # Cache for re-runs
            CORPUS_DIR.mkdir(parents=True, exist_ok=True)
            with open(wiki_cache, "w", encoding="utf-8") as f:
                for s in wiki_sents:
                    f.write(s + "\n")
            print(f"  Cached {len(wiki_sents)} sentences to {wiki_cache}")

        all_sentences.extend(wiki_sents)
        print(f"  [Wikipedia] {len(wiki_sents)} sentences (raw)")
    else:
        print(f"\n  [Wikipedia] SKIPPED (--skip-wiki)")

    # ── Built-in sentences ───────────────────────────────────────────────
    domains = [
        ("Legal",       LEGAL_SENTENCES),
        ("News",        NEWS_SENTENCES),
        ("Education",   EDUCATION_SENTENCES),
        ("Business",    BUSINESS_SENTENCES),
        ("Everyday",    EVERYDAY_SENTENCES),
        ("Sci/Tech",    SCIENCE_TECH_SENTENCES),
        ("Numeric",     NUMERIC_MIXED),
    ]

    for name, sentences in domains:
        print(f"  [{name}] {len(sentences)} sentences")
        all_sentences.extend(sentences)

    # ── Generated addresses & names ──────────────────────────────────────
    addrs = generate_addresses(500)
    print(f"  [Addresses] {len(addrs)} generated")
    all_sentences.extend(addrs)

    names = generate_name_records(200)
    print(f"  [Names] {len(names)} generated")
    all_sentences.extend(names)

    # ── Niits labels: DISABLED ──────────────────────────────────────────
    # Removed: niits labels contain spelling errors, random mid-word
    # spacing, and duplicates that would corrupt the decoder's language
    # model. Our Wikipedia + built-in corpus is cleaner and larger.
    print(f"  [Niits] DISABLED (noisy labels — typos, random spacing)")

    # ── Deduplicate and normalize ────────────────────────────────────────
    raw_count = len(all_sentences)
    normalized = []
    seen = set()
    for s in all_sentences:
        s = unicodedata.normalize("NFC", s.strip())
        if s and s not in seen and len(s) >= 5:
            seen.add(s)
            normalized.append(s)

    print(f"\n  Before length filter: {len(normalized)} unique sentences")

    # ── LENGTH CONTROL: Split long sentences for TrOCR compatibility ─────
    # TrOCR input is 384x384. Long text rendered at readable font sizes
    # produces images far too wide, causing resize artifacts.
    # Split long sentences at natural Vietnamese break points.
    too_long = sum(1 for s in normalized if len(s) > max_chars)
    print(f"  Sentences > {max_chars} chars: {too_long} "
          f"({100*too_long/len(normalized):.1f}%) — will be SPLIT, not dropped")

    final_sentences = []
    split_count = 0
    drop_count = 0
    for s in normalized:
        if len(s) <= max_chars:
            final_sentences.append(s)
        else:
            segments = split_long_sentence(s, max_chars)
            for seg in segments:
                seg = seg.strip()
                if len(seg) >= 8 and len(seg) <= max_chars:
                    final_sentences.append(seg)
                    split_count += 1
                elif len(seg) > max_chars:
                    drop_count += 1

    # Re-deduplicate after splitting
    seen2 = set()
    deduped = []
    for s in final_sentences:
        if s not in seen2:
            seen2.add(s)
            deduped.append(s)
    final_sentences = deduped

    print(f"  After splitting: {len(final_sentences)} unique sentences")
    print(f"    Segments created from splits: {split_count}")
    print(f"    Segments still too long (dropped): {drop_count}")

    # ── Quality check ────────────────────────────────────────────────────
    if len(final_sentences) < min_sentences:
        print(f"\n  WARNING: Only {len(final_sentences)} sentences, "
              f"target was {min_sentences}.")

    # ── Save ─────────────────────────────────────────────────────────────
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPUS_DIR / "modern_vietnamese.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in final_sentences:
            f.write(s + "\n")

    print(f"\n  Saved: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")

    # ── Stats ────────────────────────────────────────────────────────────
    wc = [len(s.split()) for s in final_sentences]
    cc = [len(s) for s in final_sentences]
    print(f"\n  Word count stats:")
    print(f"    Min: {min(wc)}, Max: {max(wc)}, Avg: {sum(wc)/len(wc):.1f}")
    print(f"  Character count stats:")
    print(f"    Min: {min(cc)}, Max: {max(cc)}, Avg: {sum(cc)/len(cc):.1f}")

    # Length distribution
    bins = {"0-20": 0, "21-40": 0, "41-60": 0, "61-70": 0}
    for c in cc:
        if c <= 20: bins["0-20"] += 1
        elif c <= 40: bins["21-40"] += 1
        elif c <= 60: bins["41-60"] += 1
        else: bins["61-70"] += 1
    print(f"\n  Character length distribution:")
    for k, v in bins.items():
        pct = 100 * v / len(final_sentences)
        bar = "█" * int(pct / 2)
        print(f"    {k:>6s}: {v:6d} ({pct:5.1f}%) {bar}")

    print(f"\n  Effective images @ 3 fonts/sentence: {len(final_sentences) * 3}")
    print(f"  Effective images @ 5 fonts/sentence: {len(final_sentences) * 5}")

    return final_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Vietnamese corpus for synthetic data"
    )
    parser.add_argument("--skip-wiki", action="store_true",
                        help="Skip Wikipedia scraping (use only built-in corpus)")
    parser.add_argument("--min-sentences", type=int, default=5000,
                        help="Minimum target sentences (default: 5000)")
    parser.add_argument("--max-chars", type=int, default=70,
                        help="Max characters per line (default: 70, for TrOCR 384x384)")
    args = parser.parse_args()

    build_corpus(
        skip_wiki=args.skip_wiki,
        min_sentences=args.min_sentences,
        max_chars=args.max_chars,
    )
    print(f"\n[DONE] Next: python 02_download_fonts.py")

