# RESEARCH PROPOSAL

**TÊN ĐỀ TÀI:** Nghiên cứu và Tối ưu hóa Hệ thống Nhận dạng Chữ viết tay, chữ in Tiếng Việt sử dụng Kiến trúc TrOCR

**1. TÓM TẮT DỰ ÁN (ABSTRACT)**
Nhận dạng văn bản (OCR) cho tiếng Việt là một thách thức lớn do hệ thống dấu thanh phức tạp, sự đa dạng của nét chữ và sự phân mảnh của các bộ dữ liệu. Dự án này đề xuất một hệ thống OCR đầu-cuối thống nhất dựa trên kiến trúc Transformer (TrOCR), kết hợp cùng mạng phát hiện văn bản DBNet++ và mô hình ngôn ngữ PhoBERT để xử lý cả chữ viết tay và chữ in. Điểm đột phá của nghiên cứu nằm ở việc tối ưu hóa quy trình huấn luyện đa giai đoạn: áp dụng Curriculum Learning để tổng hợp các dòng văn bản giả (pseudo-lines) và cơ chế Elastic Weight Consolidation (EWC) nhằm ngăn chặn hiện tượng quên thảm họa giữa các miền dữ liệu. Hệ thống được cấu trúc lại bằng định dạng LMDB và tối ưu tính toán để hoạt động hiệu quả trên tài nguyên phần cứng bị giới hạn.

**2. Vấn Đề Nghiên Cứu Lõi (Problem Statement)**

- **Sự đa dạng của chữ Việt và Phân phối đa thể thức:** Hệ thống dấu thanh tiếng Việt rất nhạy cảm với các sai sót nhỏ. Các mô hình CNN-RNN truyền thống (như CRNN+CTC) hoặc các hệ thống thực tiễn thường xuyên gặp khó khăn trong việc nắm bắt ngữ cảnh dài và dễ bị giảm sút hiệu năng khi đối mặt với bố cục bất quy tắc của chữ viết tay.
- **Hiện tượng Quên thảm họa (Catastrophic Forgetting):** Việc huấn luyện mô hình luân phiên giữa miền dữ liệu chữ in sạch và chữ viết tay nhiễu khiến mô hình nhanh chóng đánh mất khả năng nhận dạng trên miền dữ liệu cũ.
- **Rủi ro từ tiền xử lý hình ảnh:** Các phương pháp thay đổi kích thước ảnh (resize) truyền thống thường phá vỡ tỷ lệ khung hình, gây biến dạng nét chữ, đặc biệt là các dấu nối và dấu thanh.

**3. Đóng Góp Và Mục Tiêu Của Đề Tài**

- **Kiến tạo Pipeline đồng nhất:** Kết hợp chuẩn hóa hình ảnh Aspect-Ratio-Aware, DBNet++ cho phát hiện vùng chữ, thuật toán gom cụm DBSCAN để phân mảnh dòng, nhận dạng bằng TrOCR và hiệu đính bằng PhoBERT.
- **Tối ưu hóa Huấn luyện:** Ứng dụng Học theo chương trình (Curriculum Learning) chuyển từ mức từ lên dòng, kết hợp lấy mẫu hỗn hợp và chuẩn hóa EWC. Tối ưu hóa việc sử dụng tài nguyên (AMP, Gradient Accumulation) để đào tạo mượt mà trên giới hạn 24GB VRAM của GPU NVIDIA L4.
- **Chuẩn hóa dữ liệu toàn diện:** Hợp nhất và chuyển đổi quy mô lớn các tập dữ liệu hỗn hợp sang định dạng LMDB để tăng tốc độ truy xuất ngẫu nhiên (I/O).

**4. DỮ LIỆU NGHIÊN CỨU (DATASETS)**
Tất cả dữ liệu được đồng bộ hóa định dạng nhãn (Unicode NFC) và lưu trữ dưới dạng LMDB.

- **UIT-HWDB:** ~110.000 từ, ~7.000 dòng, ~1.000 đoạn văn bản chữ viết tay thu thập từ 249 người viết (Writer-independent).
- **Cinnamon AI Dataset:** ~2.385 dòng chữ viết tay có độ khó cao, viết ngoáy, nhiễu nền.
- **Viet-Wiki-Handwriting:** ~5.796 đoạn văn bản chữ viết tay tổng hợp, cung cấp vốn từ vựng lớn.
- **VinText & MC-OCR 2021:** Lần lượt 2.000 ảnh tự nhiên và ~6.585 dòng chữ trên hóa đơn thực tế để củng cố năng lực xử lý chữ in biến dạng.
- **Anyuuus OCR:** ~28.000 dòng văn bản in sạch giúp mô hình học các phân phối định dạng cơ bản.
- **Synthetic Printed Vietnamese:** gồm khoảng 30.000 dòng, được tạo ra bằng 14 bộ phông chữ Google Fonts; thuộc 7 lĩnh vực (pháp luật, tin tức, giáo dục, kinh doanh, đời sống thường nhật, khoa học/công nghệ, số liệu).

**5. PIPELINE ĐỀ XUẤT (PROPOSED PIPELINE)**

- **Bước 1: Tiền xử lý (Image Preprocessing):** Áp dụng CLAHE để cân bằng độ tương phản, sử dụng Hough Line Transform để cân chỉnh góc nghiêng. Tối ưu thuật toán đệm ảnh (Adaptive Padding) bằng giá trị trung vị và giới hạn tỷ lệ khung hình (4:1) để chống biến dạng.
- **Bước 2: Phát hiện văn bản (Text Detection):** Sử dụng DBNet++ do khả năng bắt các dòng văn bản liền mạch, bị cong/nghiêng tốt hơn và cho tốc độ suy luận nhanh hơn CRAFT.
- **Bước 3: Phân mảnh dòng (Line Segmentation):** Gom cụm các hộp giới hạn (bounding boxes) bằng DBSCAN kết hợp phân tích bóng chiếu (Projection Profile) để xử lý chữ viết tay không thẳng hàng.
- **Bước 4: Nhận dạng văn bản (Text Recognition):** Sử dụng TrOCR (ViT Encoder + Transformer Decoder). Mã thông báo (Tokenizer) được mở rộng với các ký tự tiếng Việt đặc thù, và Positional Embedding được khởi tạo lại (Re-initialization) để xóa bỏ thiên kiến tiếng Anh.
- **Bước 5: Hiệu chỉnh ngôn ngữ (Language Correction):** Sử dụng PhoBERT (Masked Language Model) để rà soát và sửa lỗi dấu thanh dựa trên ngữ cảnh chuỗi song hướng.

**6. TIÊU CHÍ ĐÁNH GIÁ (EVALUATION METRICS)**

- Đánh giá qua hai chỉ số: Tỷ lệ Lỗi Ký tự (CER) và Tỷ lệ Lỗi Từ (WER).
- Sử dụng Chiến lược Đánh giá Miền Kép (Dual-Domain Evaluation Strategy) nhằm đo lường độc lập trên cả tập chữ in và chữ viết tay để giám sát tính ổn định của EWC.