# TrOCR Error Analysis — Worst Cases Report

Total samples evaluated: 4477
Showing top 50 worst cases by CER (corrected).

---

## #1 — CER=0.8261 | TRUNCATION | handwritten

**Image:** `handwritten_0235.png`

| Field | Value |
|---|---|
| Ground Truth | `Khu 5, Thị trấn Kỳ Sơn, Huyện Kỳ Sơn, Hòa Bình` |
| Raw Prediction | `Khuố, Thị` |
| Sanitized | `Khuố, Thị` |
| Final (no PhoBERT) | `Khuố, Thị` |
| Final (with PhoBERT) | `Khuố, Thị` |
| CER (raw→san→final) | 0.8261 → 0.8261 → 0.8261 |
| CER (with PhoBERT) | 0.8261 |
| WER (final) | 0.9091 |

---

## #2 — CER=0.7727 | SUBSTITUTION | handwritten

**Image:** `handwritten_0657.png`

| Field | Value |
|---|---|
| Ground Truth | `Số nhà 6, hẻm 124/22/53 âu Cơ, Phường Tứ Liên, Quận Tây Hồ, Hà Nội` |
| Raw Prediction | `Số Tiền Đà, F9, Quận 5, TP Hồ Chí Minh` |
| Sanitized | `Số Tiền Đà, F9, Quận 5, TP Hồ Chí Minh` |
| Final (no PhoBERT) | `Số Tiền Đà, F9, Quận 5, TP Hồ Chí Minh` |
| Final (with PhoBERT) | `Số Tiền Đà, F9, Quận 5, TP Hồ Chí Minh` |
| CER (raw→san→final) | 0.7727 → 0.7727 → 0.7727 |
| CER (with PhoBERT) | 0.7727 |
| WER (final) | 0.9333 |

---

## #3 — CER=0.7143 | HALLUCINATION_LOOP | printed

**Image:** `printed_0377.png`

| Field | Value |
|---|---|
| Ground Truth | `quân sang đây... Phỏng chừng nhà nước chịu được mấy phen` |
| Raw Prediction | `quân sang đây.................................................................................................................................................................................................................................................` |
| Sanitized | `quân sang đây...` |
| Final (no PhoBERT) | `quân sang đây...` |
| Final (with PhoBERT) | `quân sang đây...` |
| CER (raw→san→final) | 4.2500 → 0.7143 → 0.7143 |
| CER (with PhoBERT) | 0.7143 |
| WER (final) | 0.7273 |

---

## #4 — CER=0.6053 | SUBSTITUTION | handwritten

**Image:** `handwritten_0466.png`

| Field | Value |
|---|---|
| Ground Truth | `Hoá Thượng, Huyện Đồng Hỷ, Thái Nguyên` |
| Raw Prediction | `Hoà Thực vợi, ĐDND Y Cộn Đồng ty, Ứn Nguyễn` |
| Sanitized | `Hoà Thực vợi, ĐDND Y Cộn Đồng ty, Ứn Nguyễn` |
| Final (no PhoBERT) | `Hoà Thực vợi, ĐDND Y Cộn Đồng ty, Ứn Nguyễn` |
| Final (with PhoBERT) | `Hoà Thực vợi, ĐDND Y Cộn Đồng ty, Ứn Nguyễn` |
| CER (raw→san→final) | 0.6053 → 0.6053 → 0.6053 |
| CER (with PhoBERT) | 0.6053 |
| WER (final) | 1.2857 |

---

## #5 — CER=0.5517 | SUBSTITUTION | handwritten

**Image:** `handwritten_0386.png`

| Field | Value |
|---|---|
| Ground Truth | `Số 339, phố Huế, Phường Phố Huế, Quận Hai Bà Trưng, Hà Nội` |
| Raw Prediction | `Chong cố 739, phố thiếng Phường Phồ Đe Lê Quận đà Trương 15/002` |
| Sanitized | `Chong cố 739, phố thiếng Phường Phồ Đe Lê Quận đà Trương 15/002` |
| Final (no PhoBERT) | `Chong cố 739, phố thiếng Phường Phồ Đe Lê Quận đà Trương 15/002` |
| Final (with PhoBERT) | `Chong cố 739, phố thiếng Phường Phồ Đe Lê Quận đà Trương 15/002` |
| CER (raw→san→final) | 0.5517 → 0.5517 → 0.5517 |
| CER (with PhoBERT) | 0.5517 |
| WER (final) | 0.9231 |

---

## #6 — CER=0.5294 | SUBSTITUTION | handwritten

**Image:** `handwritten_0178.png`

| Field | Value |
|---|---|
| Ground Truth | `Một món quà ý nghĩa hơn cả những món quà quý giá, hạnh phúc ấy long lanh in trong mắt` |
| Raw Prediction | `một nói giảa làm " cả dũng má quà ýy góp. hoà phúc ấy bị doái - Lo chong đót` |
| Sanitized | `một nói giảa làm " cả dũng má quà ýy góp. hoà phúc ấy bị doái - Lo chong đót` |
| Final (no PhoBERT) | `một nói giảa làm " cả dũng má quà ýy góp. hoà phúc ấy bị doái - Lo chong đót` |
| Final (with PhoBERT) | `một nói giảa làm " cả dũng má quà ýy góp. hoà phúc ấy bị doái - Lo chong đót` |
| CER (raw→san→final) | 0.5294 → 0.5294 → 0.5294 |
| CER (with PhoBERT) | 0.5294 |
| WER (final) | 0.8500 |

---

## #7 — CER=0.5075 | SUBSTITUTION | handwritten

**Image:** `handwritten_0305.png`

| Field | Value |
|---|---|
| Ground Truth | `Số 49, ngõ 93, phố 8/3, Phường Quỳnh Mai, Quận Hai Bà Trưng, Hà Nội` |
| Raw Prediction | `Số 47, ngõ 5 93, Phố I7 / TP Uy Côn Muyệnh Ma, Quận và kim, Hời` |
| Sanitized | `Số 47, ngõ 5 93, Phố I7 / TP Uy Côn Muyệnh Ma, Quận và kim, Hời` |
| Final (no PhoBERT) | `Số 47, ngõ 5 93, Phố I7 / TP Uy Côn Muyệnh Ma, Quận và kim, Hời` |
| Final (with PhoBERT) | `Số 47, ngõ 5 93, Phố I7 / TP Uy Côn Muyệnh Ma, Quận và kim, Hời` |
| CER (raw→san→final) | 0.5075 → 0.5075 → 0.5075 |
| CER (with PhoBERT) | 0.5075 |
| WER (final) | 0.9333 |

---

## #8 — CER=0.5000 | SUBSTITUTION | printed

**Image:** `printed_0898.png`

| Field | Value |
|---|---|
| Ground Truth | `Quyển VI` |
| Raw Prediction | `QUYỂN VI` |
| Sanitized | `QUYỂN VI` |
| Final (no PhoBERT) | `QUYỂN VI` |
| Final (with PhoBERT) | `QUYỂN VI` |
| CER (raw→san→final) | 0.5000 → 0.5000 → 0.5000 |
| CER (with PhoBERT) | 0.5000 |
| WER (final) | 0.5000 |

---

## #9 — CER=0.5000 | SUBSTITUTION | handwritten

**Image:** `handwritten_0665.png`

| Field | Value |
|---|---|
| Ground Truth | `Số 39 ngõ 23 phố Đỗ Quang, Phường Trung Hoà, Quận Cầu Giấy, Hà Nội` |
| Raw Prediction | `ở Độ Người F Quan, Hoàng Trung Hò, Gòm Cầu Gốn, HồNG` |
| Sanitized | `ở Độ Người F Quan, Hoàng Trung Hò, Gòm Cầu Gốn, HồNG` |
| Final (no PhoBERT) | `ở Độ Người F Quan, Hoàng Trung Hò, Gòm Cầu Gốn, HồNG` |
| Final (with PhoBERT) | `ở Độ Người F Quan, Hoàng Trung Hò, Gòm Cầu Gốn, HồNG` |
| CER (raw→san→final) | 0.5000 → 0.5000 → 0.5000 |
| CER (with PhoBERT) | 0.5000 |
| WER (final) | 0.8667 |

---

## #10 — CER=0.4821 | SUBSTITUTION | handwritten

**Image:** `handwritten_0298.png`

| Field | Value |
|---|---|
| Ground Truth | `Số 127H Thụy Khuê, Phường Thuỵ Khuê, Quận Tây Hồ, Hà Nội` |
| Raw Prediction | `Số 127 Thực, Khuêng thôy Quy sao? Quận Một tây Hồ, Nội` |
| Sanitized | `Số 127 Thực, Khuêng thôy Quy sao? Quận Một tây Hồ, Nội` |
| Final (no PhoBERT) | `Số 127 Thực, Khuêng thôy Quy sao? Quận Một tây Hồ, Nội` |
| Final (with PhoBERT) | `Số 127 Thực, Khuêng thôy Quy sao? Quận Một tây Hồ, Nội` |
| CER (raw→san→final) | 0.4821 → 0.4821 → 0.4821 |
| CER (with PhoBERT) | 0.4821 |
| WER (final) | 0.7500 |

---

## #11 — CER=0.4655 | HALLUCINATION_LOOP | handwritten

**Image:** `handwritten_0625.png`

| Field | Value |
|---|---|
| Ground Truth | `Số 15 Đông Khê, Phường Đông Khê, Quận Ngô Quyền, Hải Phòng` |
| Raw Prediction | `Số I5ĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐĐường Khô, sực Tên quyền, Hải Phòn có` |
| Sanitized | `Số I5ĐĐĐường Khô, sực Tên quyền, Hải Phòn có` |
| Final (no PhoBERT) | `Số I5ĐĐĐường Khô, sực Tên quyền, Hải Phòng` |
| Final (with PhoBERT) | `Số I5ĐĐĐường Khô, sực Tên quyền, Hải Phòng` |
| CER (raw→san→final) | 1.5517 → 0.5172 → 0.4655 |
| CER (with PhoBERT) | 0.4655 |
| WER (final) | 0.7500 |

---

## #12 — CER=0.4595 | SUBSTITUTION | handwritten

**Image:** `handwritten_0125.png`

| Field | Value |
|---|---|
| Ground Truth | `tôn vinh thì trái lại những con người ấy đã bị đẩy xuống đáy cùng của cuộc` |
| Raw Prediction | `tỉ inh thì bạ không ba Người cậng phĩ bị dây xuống đây cùng mặi,` |
| Sanitized | `tỉ inh thì bạ không ba Người cậng phĩ bị dây xuống đây cùng mặi,` |
| Final (no PhoBERT) | `tỉ inh thì bạ không ba Người cậng phĩ bị dây xuống đây cùng mặi,` |
| Final (with PhoBERT) | `tỉ inh thì bạ không ba Người cậng phĩ bị dây xuống đây cùng mặi,` |
| CER (raw→san→final) | 0.4595 → 0.4595 → 0.4595 |
| CER (with PhoBERT) | 0.4595 |
| WER (final) | 0.7647 |

---

## #13 — CER=0.4571 | SUBSTITUTION | handwritten

**Image:** `handwritten_0150.png`

| Field | Value |
|---|---|
| Ground Truth | `câu hỏi đó tự đến với họ hoặc đến từ người thân, bạn bè đi kèm sự chua` |
| Raw Prediction | `cân hải do ám đều vàc kỵ loạn đến bị Nguyễ trong ban bè đe làm sự dụng` |
| Sanitized | `cân hải do ám đều vàc kỵ loạn đến bị Nguyễ trong ban bè đe làm sự dụng` |
| Final (no PhoBERT) | `cân hải do ám đều vàc kỵ loạn đến bị Nguyễ trong ban bè đe làm sự dụng` |
| Final (with PhoBERT) | `cân hải do ám đều vàc kỵ loạn đến bị Nguyễ trong ban bè đe làm sự dụng` |
| CER (raw→san→final) | 0.4571 → 0.4571 → 0.4571 |
| CER (with PhoBERT) | 0.4571 |
| WER (final) | 0.8333 |

---

## #14 — CER=0.4559 | SUBSTITUTION | handwritten

**Image:** `handwritten_0197.png`

| Field | Value |
|---|---|
| Ground Truth | `Số 33 Trần Cao Vân, Phường Ngọc Trạo, Thành phố Thanh Hoá, Thanh Hoá` |
| Raw Prediction | `" Số 77 Tân Cao Vân, Phường Yặc 5ạo, TP Chành về trinh đánh tước` |
| Sanitized | `" Số 77 Tân Cao Vân, Phường Yặc 5ạo, TP Chành về trinh đánh tước` |
| Final (no PhoBERT) | `" Số 77 Tân Cao Vân, Phường Yặc 5ạo, TP Chành về trinh đánh tước` |
| Final (with PhoBERT) | `" Số 77 Tân Cao Vân, Phường Yặc 5ạo, TP Chành về trinh đánh tước` |
| CER (raw→san→final) | 0.4559 → 0.4559 → 0.4559 |
| CER (with PhoBERT) | 0.4559 |
| WER (final) | 0.7857 |

---

## #15 — CER=0.4545 | SUBSTITUTION | handwritten

**Image:** `handwritten_0132.png`

| Field | Value |
|---|---|
| Ground Truth | `các nhà đầu tư trong Khu chế xuất Tân Thuận. Vậy thì khi không khắc phục ngay` |
| Raw Prediction | `các ghòa đầu bi tay khu Miếm Tân VN. Cập là không biết phục ngay` |
| Sanitized | `các ghòa đầu bi tay khu Miếm Tân VN. Cập là không biết phục ngay` |
| Final (no PhoBERT) | `các ghòa đầu bi tay khu Miếm Tân VN. Cập là không biết phục ngay` |
| Final (with PhoBERT) | `các ghòa đầu bi tay khu Miếm Tân VN. Cập là không biết phục ngay` |
| CER (raw→san→final) | 0.4545 → 0.4545 → 0.4545 |
| CER (with PhoBERT) | 0.4545 |
| WER (final) | 0.6471 |

---

## #16 — CER=0.4493 | SUBSTITUTION | handwritten

**Image:** `handwritten_0142.png`

| Field | Value |
|---|---|
| Ground Truth | `những sai phạm đó. Có như vậy niềm tin vào lẽ phải mới được khôi phục` |
| Raw Prediction | `nhưng Người làm đế. có như vậy mầm Gian vào Cê phả trác kháe dân,` |
| Sanitized | `nhưng Người làm đế. có như vậy mầm Gian vào Cê phả trác kháe dân,` |
| Final (no PhoBERT) | `nhưng Người làm đế. có như vậy mầm Gian vào Cê phả trác kháe dân,` |
| Final (with PhoBERT) | `nhưng Người làm đế. có như vậy mầm Gian vào Cê phả trác kháe dân,` |
| CER (raw→san→final) | 0.4493 → 0.4493 → 0.4493 |
| CER (with PhoBERT) | 0.4493 |
| WER (final) | 0.8125 |

---

## #17 — CER=0.4464 | HALLUCINATION_LOOP | handwritten

**Image:** `handwritten_0358.png`

| Field | Value |
|---|---|
| Ground Truth | `2/25 Lê Chí Dân, ấp 3, Thành Phố Thủ Dầu Một, Bình Dương` |
| Raw Prediction | `27/5 Lê Chi Dân, CM 7, Thành Nội Trư Dầu Nội, Km phơp, 20000n` |
| Sanitized | `27/5 Lê Chi Dân, CM 7, Thành Nội Trư Dầu Nội, Km phơp, 2000n` |
| Final (no PhoBERT) | `27/5 Lê Chi Dân, CM 7, Thành Nội Trư Dầu Nội, Km phơp, 2000n` |
| Final (with PhoBERT) | `27/5 Lê Chi Dân, CM 7, Thành Nội Trư Dầu Nội, Km phơp, 2000n` |
| CER (raw→san→final) | 0.4643 → 0.4464 → 0.4464 |
| CER (with PhoBERT) | 0.4464 |
| WER (final) | 0.7692 |

---

## #18 — CER=0.4400 | SUBSTITUTION | printed

**Image:** `printed_2242.png`

| Field | Value |
|---|---|
| Ground Truth | `Quang Chính Chí Đức Đại Công Thánh Văn Thần Vū Dat` |
| Raw Prediction | `Quảng Vận Cao Minh Quang Chính Chí Đức Đại Công Thánh Văn Thần Vũ Đạt` |
| Sanitized | `Quảng Vận Cao Minh Quang Chính Chí Đức Đại Công Thánh Văn Thần Vũ Đạt` |
| Final (no PhoBERT) | `Quảng Vận Cao Minh Quang Chính Chí Đức Đại Công Thánh Văn Thần Vũ Đạt` |
| Final (with PhoBERT) | `Quảng Vận Cao Minh Quang Chính Chí Đức Đại Công Thánh Văn Thần Vũ Đạt` |
| CER (raw→san→final) | 0.4400 → 0.4400 → 0.4400 |
| CER (with PhoBERT) | 0.4400 |
| WER (final) | 0.5455 |

---

## #19 — CER=0.4400 | SUBSTITUTION | handwritten

**Image:** `handwritten_0140.png`

| Field | Value |
|---|---|
| Ground Truth | `dương nào thiết thực hơn đối với anh Đại và những đồng nghiệp là xử lý ngay` |
| Raw Prediction | `doáng chiết khôc hơn vớn anh Đại và những dây Nguy kiểm lý ngường.` |
| Sanitized | `doáng chiết khôc hơn vớn anh Đại và những dây Nguy kiểm lý ngường.` |
| Final (no PhoBERT) | `doáng chiết khôc hơn vớn anh Đại và những dây Nguy kiểm lý ngường.` |
| Final (with PhoBERT) | `doáng chiết khôc hơn vớn anh Đại và những dây Nguy kiểm lý ngường.` |
| CER (raw→san→final) | 0.4400 → 0.4400 → 0.4400 |
| CER (with PhoBERT) | 0.4400 |
| WER (final) | 0.6471 |

---

## #20 — CER=0.4348 | SUBSTITUTION | printed

**Image:** `printed_0739.png`

| Field | Value |
|---|---|
| Ground Truth | `34 DAI VIETSUKY TOANTHU` |
| Raw Prediction | `34 ĐẠI VIỆT SỬ KÝ TOÀN THƯ` |
| Sanitized | `34 ĐẠI VIỆT SỬ KÝ TOÀN THƯ` |
| Final (no PhoBERT) | `34 ĐẠI VIỆT SỬ KÝ TOÀN THƯ` |
| Final (with PhoBERT) | `34 ĐẠI VIỆT SỬ KÝ TOÀN THƯ` |
| CER (raw→san→final) | 0.4348 → 0.4348 → 0.4348 |
| CER (with PhoBERT) | 0.4348 |
| WER (final) | 1.5000 |

---

## #21 — CER=0.4348 | SUBSTITUTION | printed

**Image:** `printed_0831.png`

| Field | Value |
|---|---|
| Ground Truth | `64DAI VIETSU KYTOAN THU` |
| Raw Prediction | `64 ĐẠI VIỆT SỬ KÝ TOÀN THƯ` |
| Sanitized | `64 ĐẠI VIỆT SỬ KÝ TOÀN THƯ` |
| Final (no PhoBERT) | `64 ĐẠI VIỆT SỬ KÝ TOÀN THƯ` |
| Final (with PhoBERT) | `64 ĐẠI VIỆT SỬ KÝ TOÀN THƯ` |
| CER (raw→san→final) | 0.4348 → 0.4348 → 0.4348 |
| CER (with PhoBERT) | 0.4348 |
| WER (final) | 1.7500 |

---

## #22 — CER=0.4310 | SUBSTITUTION | handwritten

**Image:** `handwritten_0632.png`

| Field | Value |
|---|---|
| Ground Truth | `Long Trung, Xã Long Thành Trung, Huyện Hoà Thành, Tây Ninh` |
| Raw Prediction | `Trong Truy, vào 1 tring Thành Hung, Huyện Hòa Qua, Từng Minh` |
| Sanitized | `Trong Truy, vào 1 tring Thành Hung, Huyện Hòa Qua, Từng Minh` |
| Final (no PhoBERT) | `Trong Truy, vào 1 tring Thành Hung, Huyện Hòa Qua, Từng Minh` |
| Final (with PhoBERT) | `Trong Truy, vào 1 tring Thành Hung, Huyện Hòa Qua, Từng Minh` |
| CER (raw→san→final) | 0.4310 → 0.4310 → 0.4310 |
| CER (with PhoBERT) | 0.4310 |
| WER (final) | 0.9091 |

---

## #23 — CER=0.4151 | SUBSTITUTION | handwritten

**Image:** `handwritten_0722.png`

| Field | Value |
|---|---|
| Ground Truth | `14 Nguyễn Thái Học, P Tân An, Quận Ninh Kiều, Cần Thơ` |
| Raw Prediction | `1 Là Nguyễn Thai cá đới, P.Cân Ang Krên Ninh Vĩn sử, Cần Thơ` |
| Sanitized | `1 Là Nguyễn Thai cá đới, P.Cân Ang Krên Ninh Vĩn sử, Cần Thơ` |
| Final (no PhoBERT) | `1 Là Nguyễn Thai cá đới, P.Cân Ang Krên Ninh Vĩn sử, Cần Thơ` |
| Final (with PhoBERT) | `1 Là Nguyễn Thai cá đới, P. Cân Ang Krên Ninh Vĩn sử, Cần Thơ` |
| CER (raw→san→final) | 0.4151 → 0.4151 → 0.4151 |
| CER (with PhoBERT) | 0.4151 |
| WER (final) | 0.8333 |

---

## #24 — CER=0.4133 | SUBSTITUTION | handwritten

**Image:** `handwritten_0135.png`

| Field | Value |
|---|---|
| Ground Truth | `đồng nghĩa với việc mất đi bao nhiêu cơ hội làm việc và cải thiện cuộc sống` |
| Raw Prediction | `dùng nhàn với mội đ1 bao nhiên có hật loài việc và cả khiếu sốc, Nay` |
| Sanitized | `dùng nhàn với mội đ1 bao nhiên có hật loài việc và cả khiếu sốc, Nay` |
| Final (no PhoBERT) | `dùng nhàn với mội đ1 bao nhiên có hật loài việc và cả khiếu sốc, Nay` |
| Final (with PhoBERT) | `dùng nhàn với mội @@ bao nhiên có hật loài việc và cả khiếu sốc, Nay` |
| CER (raw→san→final) | 0.4133 → 0.4133 → 0.4133 |
| CER (with PhoBERT) | 0.4267 |
| WER (final) | 0.7647 |

---

## #25 — CER=0.4026 | SUBSTITUTION | handwritten

**Image:** `handwritten_0147.png`

| Field | Value |
|---|---|
| Ground Truth | `như làm chạnh lòng tôi là có phải họ đã xả thân vì đại nghĩa để rồi gia đình,` |
| Raw Prediction | `như làm choai sáy bồng lành phả họ đã xã Nhai và đe nghĩa để " gò 1 trình` |
| Sanitized | `như làm choai sáy bồng lành phả họ đã xã Nhai và đe nghĩa để " gò 1 trình` |
| Final (no PhoBERT) | `như làm choai sáy bồng lành phả họ đã xã Nhai và đe nghĩa để " gò 1 trình` |
| Final (with PhoBERT) | `như làm choai sáy bồng lành phả họ đã xã Nhai và đe nghĩa để " gò 1 trình` |
| CER (raw→san→final) | 0.4026 → 0.4026 → 0.4026 |
| CER (with PhoBERT) | 0.4026 |
| WER (final) | 0.7368 |

---

## #26 — CER=0.4000 | SUBSTITUTION | printed

**Image:** `printed_1859.png`

| Field | Value |
|---|---|
| Ground Truth | `làm quan đài thôi!".` |
| Raw Prediction | `làm đài quan thôi!".` |
| Sanitized | `làm đài quan thôi!".` |
| Final (no PhoBERT) | `làm đài quan thôi!".` |
| Final (with PhoBERT) | `làm đài quan thôi! ".` |
| CER (raw→san→final) | 0.4000 → 0.4000 → 0.4000 |
| CER (with PhoBERT) | 0.4500 |
| WER (final) | 0.5000 |

---

## #27 — CER=0.4000 | SUBSTITUTION | handwritten

**Image:** `handwritten_0148.png`

| Field | Value |
|---|---|
| Ground Truth | `vợ con ly tán, thất nghiệp, lâm vào cảnh đời tan nát như thế sao! Vì sao họ` |
| Raw Prediction | `vợ conh lới, phốt Nghiệp, lần vào cảnh đời km bát khê Họ Vì 10 bị` |
| Sanitized | `vợ conh lới, phốt Nghiệp, lần vào cảnh đời km bát khê Họ Vì 10 bị` |
| Final (no PhoBERT) | `vợ conh lới, phốt Nghiệp, lần vào cảnh đời km bát khê Họ Vì 10 bị` |
| Final (with PhoBERT) | `vợ conh lới, phốt Nghiệp, lần vào cảnh đời km bát khê Họ Vì 10.` |
| CER (raw→san→final) | 0.4000 → 0.4000 → 0.4000 |
| CER (with PhoBERT) | 0.4133 |
| WER (final) | 0.7222 |

---

## #28 — CER=0.3881 | SUBSTITUTION | handwritten

**Image:** `handwritten_0141.png`

| Field | Value |
|---|---|
| Ground Truth | `những sai phạm (đã rõ mười mươi) và có biện pháp hữu hiệu khắc phục` |
| Raw Prediction | `những sai pham, (đã ào muàn của) và é lạm pháp biểu kiên khỏe phúc` |
| Sanitized | `những sai pham, (đã ào muàn của) và é lạm pháp biểu kiên khỏe phúc` |
| Final (no PhoBERT) | `những sai pham, (đã ào muàn của) và é lạm pháp biểu kiên khỏe phúc` |
| Final (with PhoBERT) | `những sai pham, (đã ào muàn của) và é lạm pháp biểu kiên khỏe phúc` |
| CER (raw→san→final) | 0.3881 → 0.3881 → 0.3881 |
| CER (with PhoBERT) | 0.3881 |
| WER (final) | 0.6667 |

---

## #29 — CER=0.3846 | SUBSTITUTION | handwritten

**Image:** `handwritten_0122.png`

| Field | Value |
|---|---|
| Ground Truth | `Thế nhưng những hiệp sĩ ấy giờ ra sao? Người thì lang thang, phiêu bạt, kẻ thì` |
| Raw Prediction | `Thế nhưng những luộp sẽ ấy giờ kmỏ. Ngoa tôi bay chong phảa dại, kẻ dàn` |
| Sanitized | `Thế nhưng những luộp sẽ ấy giờ kmỏ. Ngoa tôi bay chong phảa dại, kẻ dàn` |
| Final (no PhoBERT) | `Thế nhưng những luộp sẽ ấy giờ kmỏ. Ngoa tôi bay chong phảa dại, kẻ dàn` |
| Final (with PhoBERT) | `Thế nhưng những luộp sẽ ấy giờ kmỏ. Ngoa tôi bay chong phảa dại, kẻ dàn` |
| CER (raw→san→final) | 0.3846 → 0.3846 → 0.3846 |
| CER (with PhoBERT) | 0.3846 |
| WER (final) | 0.6471 |

---

## #30 — CER=0.3788 | SUBSTITUTION | handwritten

**Image:** `handwritten_0321.png`

| Field | Value |
|---|---|
| Ground Truth | `76/18 Lê Trọng Tấn, Phường Tây Thạnh, Quận Tân phú, TP Hồ Chí Minh` |
| Raw Prediction | `4 Lê Trọng Ba, Phường Tây Hoạt, Quận Tôn Đai, Chín Mộinh` |
| Sanitized | `4 Lê Trọng Ba, Phường Tây Hoạt, Quận Tôn Đai, Chín Mộinh` |
| Final (no PhoBERT) | `4 Lê Trọng Ba, Phường Tây Hoà, Quận Tôn Đai, Chín Mộinh` |
| Final (with PhoBERT) | `4 Lê Trọng Ba, Phường Tây Hoà, Quận Tôn Đai, Chín Mộinh` |
| CER (raw→san→final) | 0.3636 → 0.3636 → 0.3788 |
| CER (with PhoBERT) | 0.3788 |
| WER (final) | 0.6429 |

---

## #31 — CER=0.3770 | SUBSTITUTION | handwritten

**Image:** `handwritten_0134.png`

| Field | Value |
|---|---|
| Ground Truth | `làm xấu đi môi trường đầu tư của thành phố hay không, và cũng` |
| Raw Prediction | `làm xâc đi vận trường, Đầu từ anh Của phê (hong không, bà cùng` |
| Sanitized | `làm xâc đi vận trường, Đầu từ anh Của phê (hong không, bà cùng` |
| Final (no PhoBERT) | `làm xâc đi vận trường, Đầu từ anh Của phê (hong không, bà cùng` |
| Final (with PhoBERT) | `làm xâc đi vận trường, Đầu từ anh Của phê (hong không, bà cùng` |
| CER (raw→san→final) | 0.3770 → 0.3770 → 0.3770 |
| CER (with PhoBERT) | 0.3770 |
| WER (final) | 0.7857 |

---

## #32 — CER=0.3770 | SUBSTITUTION | handwritten

**Image:** `handwritten_0476.png`

| Field | Value |
|---|---|
| Ground Truth | `Số 13 Phạm Huy Thông, Thị Trấn Ân Thi, Huyện Ân Thi, Hưng Yên` |
| Raw Prediction | `Số 17 Thạm Huy Cháng, Thị cấm triện trời đai, Hàng Yên` |
| Sanitized | `Số 17 Thạm Huy Cháng, Thị cấm triện trời đai, Hàng Yên` |
| Final (no PhoBERT) | `Số 17 Thạm Huy Cháng, Thị cấm triện trời đai, Hưng Yên` |
| Final (with PhoBERT) | `Số 17 Thạm Huy Cháng, Thị cấm triện trời đai, Hưng Yên` |
| CER (raw→san→final) | 0.3934 → 0.3934 → 0.3770 |
| CER (with PhoBERT) | 0.3770 |
| WER (final) | 0.6429 |

---

## #33 — CER=0.3768 | SUBSTITUTION | handwritten

**Image:** `handwritten_0263.png`

| Field | Value |
|---|---|
| Ground Truth | `7 Lô D Lạc Long Quân, Phường 2, Thành Phố Vũng Tàu, Bà Rịa - Vũng Tàu` |
| Raw Prediction | `7 Lê D lại Cong Quân, Phương 2, Thành dế " Vũng Tầm, Vậy Trán` |
| Sanitized | `7 Lê D lại Cong Quân, Phương 2, Thành dế " Vũng Tầm, Vậy Trán` |
| Final (no PhoBERT) | `7 Lê D lại Cong Quân, Phương 2, Thành dế " Vũng Tầm, Vậy Trán` |
| Final (with PhoBERT) | `7 Lê D lại Cong Quân, Phương 2, Thành dế " Vũng Tầm, Vậy Trán` |
| CER (raw→san→final) | 0.3768 → 0.3768 → 0.3768 |
| CER (with PhoBERT) | 0.3768 |
| WER (final) | 0.7059 |

---

## #34 — CER=0.3699 | SUBSTITUTION | handwritten

**Image:** `handwritten_0130.png`

| Field | Value |
|---|---|
| Ground Truth | `lâu ngày đến vậy? Tôi không phải là người trong lĩnh vực chuyên môn nhưng` |
| Raw Prediction | `Làng ngờ đếi ủy? Trê không phải là ngoài tay tìm bạc chuyển mội. nhưng` |
| Sanitized | `Làng ngờ đếi ủy? Trê không phải là ngoài tay tìm bạc chuyển mội. nhưng` |
| Final (no PhoBERT) | `Làng ngờ đếi ủy? Trê không phải là ngoài tay tìm bạc chuyển mội. nhưng` |
| Final (with PhoBERT) | `Làng ngờ đếi ủy? Trê không phải là ngoài tay tìm bạc chuyển mội. nhưng` |
| CER (raw→san→final) | 0.3699 → 0.3699 → 0.3699 |
| CER (with PhoBERT) | 0.3699 |
| WER (final) | 0.7333 |

---

## #35 — CER=0.3676 | SUBSTITUTION | handwritten

**Image:** `handwritten_0729.png`

| Field | Value |
|---|---|
| Ground Truth | `02 - 04, KP 3, An bình, Phường An Bình, Thành phố Biên Hoà, Đồng Nai` |
| Raw Prediction | `05 - 04,KP3, Anh ởi ra, Quường A Định, Trương phố Bên the, Sồng khi` |
| Sanitized | `05 - 04,KP3, Anh ởi ra, Quường A Định, Trương phố Bên the, Sồng khi` |
| Final (no PhoBERT) | `05 - 04,KP3, Anh ởi ra, Quường A Định, Trương phố Bên the, Sồng khi` |
| Final (with PhoBERT) | `05 - 04, KP3, Anh ởi ra, Quường A Định, Trương phố Bên the, Sồng khi` |
| CER (raw→san→final) | 0.3676 → 0.3676 → 0.3676 |
| CER (with PhoBERT) | 0.3529 |
| WER (final) | 0.8750 |

---

## #36 — CER=0.3673 | SUBSTITUTION | handwritten

**Image:** `handwritten_0219.png`

| Field | Value |
|---|---|
| Ground Truth | `Tổ 19 P, Phường Mỹ An, Quận Ngũ Hành Sơn, Đà Nẵng` |
| Raw Prediction | `đồ 19/, Phường Nỹi Am, Quận Người tành, Sơn, Đà Vưỡng` |
| Sanitized | `đồ 19/, Phường Nỹi Am, Quận Người tành, Sơn, Đà Vưỡng` |
| Final (no PhoBERT) | `đồ 19/, Phường Núi Sam, Quận Người tành, Sơn, Đà Vưỡng` |
| Final (with PhoBERT) | `đồ 19 /, Phường Núi Sam, Quận Người tành, Sơn, Đà Vưỡng` |
| CER (raw→san→final) | 0.3061 → 0.3061 → 0.3673 |
| CER (with PhoBERT) | 0.3469 |
| WER (final) | 0.6667 |

---

## #37 — CER=0.3492 | SUBSTITUTION | handwritten

**Image:** `handwritten_0193.png`

| Field | Value |
|---|---|
| Ground Truth | `Số 187, tổ 8, ấp Tân Hòa, Xã Tân Tiến, Huyện Bù Đốp, Bình Phước` |
| Raw Prediction | `Số 187, Hờ &, án Hoài, Xã Tiên. Huyện Bù Đốp, Nh Phuốc` |
| Sanitized | `Số 187, Hờ &, án Hoài, Xã Tiên. Huyện Bù Đốp, Nh Phuốc` |
| Final (no PhoBERT) | `Số 187, Hờ &, án Hoài, Xã Tiên. Huyện Bù Đốp, Nh Phuốc` |
| Final (with PhoBERT) | `Số 187, Hờ &, án Hoài, Xã Tiên. Huyện Bù Đốp, Nh Phuốc` |
| CER (raw→san→final) | 0.3492 → 0.3492 → 0.3492 |
| CER (with PhoBERT) | 0.3492 |
| WER (final) | 0.6000 |

---

## #38 — CER=0.3478 | SUBSTITUTION | handwritten

**Image:** `handwritten_0578.png`

| Field | Value |
|---|---|
| Ground Truth | `13/5A Nguyễn Văn Quá, phường Đông Hưng Thuận, Quận 12, TP Hồ Chí Minh` |
| Raw Prediction | `17/5A Nguyễn Văn quá, phường Đồng [ Trưng TPuận, Thành sinh` |
| Sanitized | `17/5A Nguyễn Văn quá, phường Đồng [ Trưng TPuận, Thành sinh` |
| Final (no PhoBERT) | `17/5A Nguyễn Văn quá, phường Đồng [ Trưng TPuận, Thành sinh` |
| Final (with PhoBERT) | `17/5A Nguyễn Văn quá, phường Đồng [Trưng TPuận, Thành sinh` |
| CER (raw→san→final) | 0.3478 → 0.3478 → 0.3478 |
| CER (with PhoBERT) | 0.3478 |
| WER (final) | 0.7857 |

---

## #39 — CER=0.3469 | SUBSTITUTION | handwritten

**Image:** `handwritten_0653.png`

| Field | Value |
|---|---|
| Ground Truth | `Tổ 4, Thị trấn Kon Dơng, Huyện Mang Yang, Gia Lai` |
| Raw Prediction | `Số 4, Thị Đán Km Động, Ngận Mạng trang Gia Lai` |
| Sanitized | `Số 4, Thị Đán Km Động, Ngận Mạng trang Gia Lai` |
| Final (no PhoBERT) | `Số 4, Thị Đán Km Động, Ngận Mạng trang Gia Lai` |
| Final (with PhoBERT) | `Số 4, Thị Đán Km Động, Ngận Mạng trang Gia Lai` |
| CER (raw→san→final) | 0.3469 → 0.3469 → 0.3469 |
| CER (with PhoBERT) | 0.3469 |
| WER (final) | 0.6364 |

---

## #40 — CER=0.3448 | SUBSTITUTION | handwritten

**Image:** `handwritten_0472.png`

| Field | Value |
|---|---|
| Ground Truth | `Tổ Luộc 1, Thị Trấn Vĩnh Lộc, Huyện Chiêm Hoá, Tuyên Quang` |
| Raw Prediction | `Tố Lược 1, Thị Trấn, Huyện Chiêm Quan, Triên Xhoảng` |
| Sanitized | `Tố Lược 1, Thị Trấn, Huyện Chiêm Quan, Triên Xhoảng` |
| Final (no PhoBERT) | `Tố Lược 1, Thị Trấn, Huyện Chiêm Hóa, Triên Xhoảng` |
| Final (with PhoBERT) | `Tố Lược 1, Thị Trấn, Huyện Chiêm Hóa, Triên Xhoảng` |
| CER (raw→san→final) | 0.3793 → 0.3793 → 0.3448 |
| CER (with PhoBERT) | 0.3448 |
| WER (final) | 0.6667 |

---

## #41 — CER=0.3443 | SUBSTITUTION | handwritten

**Image:** `handwritten_0151.png`

| Field | Value |
|---|---|
| Ground Truth | `Và hôm nay, chúng ta cũng cần hỏi câu hỏi này với chính mình.` |
| Raw Prediction | `và bầm nay, chuống truông cồn hải cân bỏ củng và Chính mình.` |
| Sanitized | `và bầm nay, chuống truông cồn hải cân bỏ củng và Chính mình.` |
| Final (no PhoBERT) | `và bầm nay, chuống truông cồn hải cân bỏ củng và Chính mình.` |
| Final (with PhoBERT) | `và bầm nay, chuống truông cồn hải cân bỏ củng và Chính mình.` |
| CER (raw→san→final) | 0.3443 → 0.3443 → 0.3443 |
| CER (with PhoBERT) | 0.3443 |
| WER (final) | 0.8571 |

---

## #42 — CER=0.3438 | SUBSTITUTION | handwritten

**Image:** `handwritten_0133.png`

| Field | Value |
|---|---|
| Ground Truth | `mà còn để con đường xuống cấp thê thảm, có phải là góp thêm phần` |
| Raw Prediction | `mà còn đề con đường xuốp thê Ca, phải là quán Hậi phân` |
| Sanitized | `mà còn đề con đường xuốp thê Ca, phải là quán Hậi phân` |
| Final (no PhoBERT) | `mà còn đề con đường xuốp thê Ca, phải là quán Hậi phân` |
| Final (with PhoBERT) | `mà còn đề con đường xuốp thê Ca, phải là quán Hậi phân` |
| CER (raw→san→final) | 0.3438 → 0.3438 → 0.3438 |
| CER (with PhoBERT) | 0.3438 |
| WER (final) | 0.5333 |

---

## #43 — CER=0.3438 | SUBSTITUTION | handwritten

**Image:** `handwritten_0149.png`

| Field | Value |
|---|---|
| Ground Truth | `đã làm điều đó? Ba năm trôi qua, chắc hẳn đã hàng trăm, ngàn lần` |
| Raw Prediction | `tô bàn điều được Ba năm hơa quả, chắc hẳn đã hày lặm, ngoài kẻ` |
| Sanitized | `tô bàn điều được Ba năm hơa quả, chắc hẳn đã hày lặm, ngoài kẻ` |
| Final (no PhoBERT) | `tô bàn điều được Ba năm hơa quả, chắc hẳn đã hày lặm, ngoài kẻ` |
| Final (with PhoBERT) | `tô bàn điều được Ba năm hơa quả, chắc hẳn đã hày lặm, ngoài kẻ` |
| CER (raw→san→final) | 0.3438 → 0.3438 → 0.3438 |
| CER (with PhoBERT) | 0.3438 |
| WER (final) | 0.6000 |

---

## #44 — CER=0.3380 | SUBSTITUTION | handwritten

**Image:** `handwritten_0146.png`

| Field | Value |
|---|---|
| Ground Truth | `người khác. Thân phận, cuộc sống của họ là bi đát, và điều mỉa mai cũng` |
| Raw Prediction | `người Phíc. thôn phậm, bậi rấy của họ dị phí và điều mìa mai cùng` |
| Sanitized | `người Phíc. thôn phậm, bậi rấy của họ dị phí và điều mìa mai cùng` |
| Final (no PhoBERT) | `người Phíc. thôn phậm, bậi rấy của họ dị phí và điều mìa mai cùng` |
| Final (with PhoBERT) | `người Phíc. thôn phậm, bậi rấy của họ dị phí và điều mìa mai cùng` |
| CER (raw→san→final) | 0.3380 → 0.3380 → 0.3380 |
| CER (with PhoBERT) | 0.3380 |
| WER (final) | 0.6250 |

---

## #45 — CER=0.3368 | SUBSTITUTION | handwritten

**Image:** `handwritten_0177.png`

| Field | Value |
|---|---|
| Ground Truth | `công trên " chiến trường " bếp núc, nhưng lại thành công khi tặng mẹ " đoá hồng " của tình yêu.` |
| Raw Prediction | `ông trên " chiếu trường " lắp, mảng lại thành côp bé tặy ne " đó Tùng vềa tình nhâo.` |
| Sanitized | `ông trên " chiếu trường " lắp, mảng lại thành côp bé tặy ne " đó Tùng vềa tình nhâo.` |
| Final (no PhoBERT) | `ông trên " chiếu trường " lắp, mảng lại thành côp bé tặy ne " đó Tùng vềa tình nhâo.` |
| Final (with PhoBERT) | `ông trên " chiếu trường " lắp, mảng lại thành côp bé tặy ne " đó Tùng vềa tình nhâo.` |
| CER (raw→san→final) | 0.3368 → 0.3368 → 0.3368 |
| CER (with PhoBERT) | 0.3368 |
| WER (final) | 0.6364 |

---

## #46 — CER=0.3333 | SUBSTITUTION | printed

**Image:** `printed_0482.png`

| Field | Value |
|---|---|
| Ground Truth | `trấn Sơn Tây` |
| Raw Prediction | `trấn Sơn Tây(1).` |
| Sanitized | `trấn Sơn Tây(1).` |
| Final (no PhoBERT) | `trấn Sơn Tây(1).` |
| Final (with PhoBERT) | `trấn Sơn Tây (1).` |
| CER (raw→san→final) | 0.3333 → 0.3333 → 0.3333 |
| CER (with PhoBERT) | 0.4167 |
| WER (final) | 0.3333 |

---

## #47 — CER=0.3253 | SUBSTITUTION | handwritten

**Image:** `handwritten_0123.png`

| Field | Value |
|---|---|
| Ground Truth | `vợ bỏ, kẻ thì cày ải trên những con đường đô thị để lượm ve chai, phải tá túc trong` |
| Raw Prediction | `" thỏ, hề thì cày ai trên con đường, đô khiệm ve chia; phải tá búc trong` |
| Sanitized | `" thỏ, hề thì cày ai trên con đường, đô khiệm ve chia; phải tá búc trong` |
| Final (no PhoBERT) | `" thỏ, hề thì cày ai trên con đường, đô khiệm ve chia; phải tá búc trong` |
| Final (with PhoBERT) | `" thỏ, hề thì cày ai trên con đường, đô khiệm ve chia; phải tá búc trong` |
| CER (raw→san→final) | 0.3253 → 0.3253 → 0.3253 |
| CER (with PhoBERT) | 0.3253 |
| WER (final) | 0.5500 |

---

## #48 — CER=0.3243 | SUBSTITUTION | handwritten

**Image:** `handwritten_0120.png`

| Field | Value |
|---|---|
| Ground Truth | `mới biết việc làm dũng cảm và tốt đẹp đó đã mang đến bao hệ lụy phiền phức` |
| Raw Prediction | `mái biết " việc làm chúng cảm và đất đẹp đã xong vềi bao kỷ lụy quán phước` |
| Sanitized | `mái biết " việc làm chúng cảm và đất đẹp đã xong vềi bao kỷ lụy quán phước` |
| Final (no PhoBERT) | `mái biết " việc làm chúng cảm và đất đẹp đã xong vềi bao kỷ lụy quán phước` |
| Final (with PhoBERT) | `mái biết " việc làm chúng cảm và đất đẹp đã xong vềi bao kỷ lụy quán phước` |
| CER (raw→san→final) | 0.3243 → 0.3243 → 0.3243 |
| CER (with PhoBERT) | 0.3243 |
| WER (final) | 0.5556 |

---

## #49 — CER=0.3214 | SUBSTITUTION | handwritten

**Image:** `handwritten_0647.png`

| Field | Value |
|---|---|
| Ground Truth | `xã quảng thành, Xã Quảng Thành, Huyện Hải Hà, Quảng Ninh` |
| Raw Prediction | `Xã quan " Thành, xã Quảy Thành, Huyện Hà Nỹi, Nộnh` |
| Sanitized | `Xã quan " Thành, xã Quảy Thành, Huyện Hà Nỹi, Nộnh` |
| Final (no PhoBERT) | `Xã Quang Thành, xã Quảy Thành, Huyện Hà Nỹi, Nộnh` |
| Final (with PhoBERT) | `" quan " Thành, xã Quảy Thành, Huyện Hà Nỹi, Nộnh` |
| CER (raw→san→final) | 0.3393 → 0.3393 → 0.3214 |
| CER (with PhoBERT) | 0.3571 |
| WER (final) | 0.8182 |

---

## #50 — CER=0.3205 | SUBSTITUTION | handwritten

**Image:** `handwritten_0121.png`

| Field | Value |
|---|---|
| Ground Truth | `cho những người trong cuộc, những người mà theo tôi vô cùng xứng đáng với danh` |
| Raw Prediction | `cho người trang mội, Những người mà khôe bân vô cùng xuống đay và dánh` |
| Sanitized | `cho người trang mội, Những người mà khôe bân vô cùng xuống đay và dánh` |
| Final (no PhoBERT) | `cho người trang mội, Những người mà khôe bân vô cùng xuống đay và dánh` |
| Final (with PhoBERT) | `cho người trang mội, Những người mà khôe bân vô cùng xuống đay và dánh` |
| CER (raw→san→final) | 0.3205 → 0.3205 → 0.3205 |
| CER (with PhoBERT) | 0.3205 |
| WER (final) | 0.6250 |

---

