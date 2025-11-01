# Tổng quan các bộ dữ liệu nổi bật

| Bộ Dữ Liệu    | Năm  | Thời Lượng | Phương Ngữ  | Điểm Mạnh Chính        | Truy Cập  |
| ------------- | ---- | ---------- | ----------- | ---------------------- | --------- |
| **ViMD**      | 2024 | 102.56h    | 63 cấp tỉnh | Bao phủ toàn diện nhất | Công khai |
| **Bud500**    | 2024 | ~500h      | 3 vùng      | Quy mô lớn nhất        | Công khai |
| **LSVSC**     | 2024 | 100.5h     | 5 loại      | Đa dạng dân tộc        | Công khai |
| **VietMed**   | 2024 | 1,216h     | 3 vùng      | Chuyên ngành y tế      | Công khai |
| **VLSP 2020** | 2020 | 100h       | 3 vùng      | Cân bằng nhất          | Công khai |
**Hạn chế**:
- **Mất cân bằng giới tính**: ViMD có tỷ lệ nam/nữ 3:1
- **Mất cân bằng phương ngữ**: LSVSC thiên về phương ngữ Bắc (88.1%)
- **Cân bằng tốt nhất**: VLSP 2020 (50% Bắc, 40% Nam, 10% Trung)

**Đặc điểm dữ liệu**
- **Tần số lấy mẫu**: 16 kHz (tiêu chuẩn)
- **Định dạng**: WAV

**Gợi ý trường hợp sử dụng**

| Mục Đích Nghiên Cứu             | Bộ Dữ Liệu Được Khuyến Nghị | Lý Do                        |
| ------------------------------- | --------------------------- | ---------------------------- |
| Nghiên cứu phương ngữ cấp tỉnh  | **ViMD**                    | Duy nhất có 63 phương ngữ    |
| Huấn luyện ASR quy mô lớn       | **Bud500**                  | 500h đa dạng nhất            |
| Đánh giá đa phương ngữ cân bằng | **VLSP 2020**               | Cân bằng vùng miền tốt nhất  |
| Ứng dụng y tế                   | **VietMed**                 | Từ vựng y khoa chuyên biệt   |
| Đa dạng dân tộc thiểu số        | **LSVSC**                   | Bao gồm Tây Nguyên + dân tộc |

---

# 2. Nội dung chi tiết 

## 1. ViMD [(HugginFace)](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)

**Đặc điểm**: 
- Bộ dữ liệu đầu tiên bao phủ toàn diện đầu tiên tất cả 63 phương ngữ cấp tỉnh của Việt Nam (so với 3-5 biến thể vùng miền trên các bộ trước đây).

**Trạng thái Khả dụng:** Công khai cho mục đích nghiên cứu

**Đặc điểm Người tham gia:**
- **Phân bố Địa lý:** Tất cả 63 tỉnh thành Việt Nam với đại diện cân bằng (89-118 phút/tỉnh, trung bình 97.68 phút)
- **Phân bổ Vùng miền:** 25 tỉnh Bắc Bộ (40.59h), 19 tỉnh Trung Bộ (31.47h), 19 tỉnh Nam Bộ (30.5h)
- **Tiêu chí Lựa chọn:** Người dân địa phương được phỏng vấn bởi các Đài Phát thanh và Truyền hình chính thức ở mỗi tỉnh
- **Nhóm Tuổi:** Độ tuổi đa dạng (không xác định chính xác)
- **Điều kiện Ghi âm:** Môi trường phát sóng đa dạng từ các đài truyền hình tỉnh

**Thông số Độ dài:**
- **Tổng Thời lượng:** 102.56 giờ (81.43h huấn luyện, 10.26h xác thực, 10.87h kiểm tra)
- **Tổng số Bản ghi/Phát ngôn:** 18,949 phát ngôn
- **Độ dài Trung bình:** 19.5 giây/bản ghi (độ lệch chuẩn: 6.2 giây, phạm vi: 1-30 giây)

**Kích thước Từ vựng:** 5,155 từ duy nhất (1,239,167 tổng số từ trong tất cả bản phiên âm)

**Số lượng Người nói:** 12,955 người nói

**Số lượng Giới tính:** 2 (nam và nữ, mất cân bằng đáng kể: 75% nam, 25% nữ)

**Số lượng Phương ngữ:** **63 phương ngữ cấp tỉnh** (bao phủ đầy đủ tất cả các tỉnh thành Việt Nam)

**Phong cách Lời nói:** Lời nói tự phát/phỏng vấn từ các chương trình phát sóng truyền hình

**Năm Xuất bản:** Tháng 10/2024 

**Mục đích Chính:** 
- Nhận dạng phương ngữ (DI);
- ASR tổng quát với phân loại phương ngữ cấp tỉnh chi tiết

**Chi tiết Kỹ thuật Chính:** 
- Tần số lấy mẫu 16 kHz, 
- Định dạng WAV
- Bao gồm metadata ID người nói và giới tính cho mỗi mẫu (csv)

---

## 2. Bud500 ([hugginface](https://huggingface.co/datasets/linhtran92/viet_bud500 ))

**Đặc điểm**: 
- **Thời lượng dài**: ~ 500 giờ đọc & hội thoại tiếng việt
- **Đa dạng hình thức**: trải rộng từ podcast, nội dung du lịch, sách và chủ đề liên quan đến ẩm thực.

**Tác giả:** Anh Pham, Khanh Linh Tran, Linh Nguyen, Thanh Duy Cao, Phuc Phan, Duong A. Nguyen (VietAI Research)

**Trạng thái Khả dụng:** Công khai 

**Đặc điểm Người tham gia:**
- **Phân bố Địa lý:** Cả ba vùng miền chính của Việt Nam (Bắc, Trung, Nam)
- **Tiêu chí Lựa chọn:** Xuất phát từ các nguồn âm thanh công cộng miễn phí bao gồm YouTube và podcast
- **Nhóm Tuổi:** Không xác định (đa dạng do nguồn tài liệu công khai)
- **Điều kiện Ghi âm:** Đa dạng (nguồn âm thanh công cộng với môi trường ghi âm khác nhau)

**Thông số Độ dài:**
- **Tổng Thời lượng:** ~510.92 giờ (Huấn luyện: ~500h, Xác thực: ~5.46h, Kiểm tra: ~5.46h)
- **Tổng số Bản ghi/Phát ngôn:** 649,158 mẫu âm thanh (634,158 huấn luyện, 7,500 xác thực, 7,500 kiểm tra)

**Kích thước Từ vựng:** Không rõ 

**Số lượng Người nói:** Không xác định (số lượng lớn)

**Số lượng Giới tính:** 2 (nam và nữ, tỷ lệ không xác định)

**Số lượng Phương ngữ:** 3 phương ngữ vùng miền (Bắc, Trung, Nam) - bao phủ vùng miền rộng, không ở cấp tỉnh

**Phong cách Lời nói:** Lời nói tự phát hỗn hợp từ các nguồn công cộng đa dạng (podcast, sách nói, nội dung giáo dục)

**Năm Xuất bản:** 2024

**Mục đích Chính:** nghiên cứu nhận dạng tiếng nói tiếng Việt

**Chi tiết Kỹ thuật Chính:** 
- Tần số lấy mẫu: 16 kHz, 
- Định dạng âm thanh: WAV (~98GB)
- Định dạng metadata:  .parquet 

---

## 3. LSVSC ([HugginFace](https://huggingface.co/datasets/doof-ferb/LSVSC))

**Đặc điểm**:  gồm ba phương ngữ vùng miền chính mà còn có phương ngữ Tây Nguyên và các nhóm dân tộc thiểu số

**Trạng thái Khả dụng:** Công khai (mở tại [HugginFace](https://huggingface.co/datasets/doof-ferb/LSVSC))

**Đặc điểm Người tham gia:**
- **Phân bố Địa lý:** Năm vùng phương ngữ với thiên vị nặng về Bắc Bộ (Bắc 88.1%, Trung 7.65%, Nam 3.54%, Tây Nguyên 0.66%, Dân tộc thiểu số 0.05%)
- **Tiêu chí Lựa chọn:** Thu thập từ Internet 
- **Phạm vi Chủ đề:** Y tế (4.71%), thể thao, du lịch (~0.4%), tin tức (78.5%), đọc/kể chuyện (9.41%), sách nói, đời sống hàng ngày (3.66%)

**Thông số Độ dài:**
- **Tổng Thời lượng:** 100.5 giờ lời nói sạch (~ 80h tự phát, 20h đọc)
- **Độ dài Trung bình:** Phần lớn phát ngôn 3-10 giây

**Số lượng Người nói:** số lượng chính xác không xác định

**Số lượng Giới tính:** 2 (nam và nữ, khoảng 50% mỗi giới)

**Số lượng Phương ngữ:** 5 loại phương ngữ (Bắc, Trung, Nam, Tây Nguyên, Dân tộc thiểu số)

**Phong cách Lời nói:** Hỗn hợp - khoảng 80% lời nói tự phát, 20% lời nói đọc

**Năm Xuất bản:** Tháng 3/2024 

**Mục đích Chính:** ASR tổng quát cho các chủ đề và phương ngữ đa dạng, ứng dụng đầu tiên của mô hình LAS và Speech-Transformer cho tiếng Việt

---

## 4. VietMed [HugginFace](https://huggingface.co/datasets/leduckhai/VietMed): Bộ Dữ Liệu Nhận Dạng Giọng Nói Y Khoa Quy Mô Lớn

**Đặc điểm**: VietMed là bộ dữ liệu nhận dạng giọng nói y khoa công khai bằng tiếng Việt duy nhất (giá trị cao cho các ứng dụng y tế)

**Trạng thái Khả dụng:** Công khai với mã nguồn, dữ liệu và mô hình

**Đặc điểm Người tham gia:**
- **Phân bố Địa lý:** Tất cả các giọng trong Việt Nam (Bắc, Trung, Nam)
- **Tiêu chí Lựa chọn:** Các cuộc tư vấn và hội thoại y tế bao gồm nhiều vai trò (bệnh nhân, bác sĩ, chuyên viên y tế)
- **Nhóm Tuổi:** Không xác định 
- **Điều kiện Ghi âm:**  đa dạng
- **Phạm vi Bệnh tật:** Tất cả các nhóm bệnh ICD-10 

**Thông số Độ dài:**
- **Tổng Thời lượng:** 1,216 giờ tổng cộng (16h y khoa có gắn nhãn + 1,000h y khoa chưa gắn nhãn + 200h lĩnh vực tổng quát chưa gắn nhãn)

**Kích thước Từ vựng:** Bao gồm các thuật ngữ y khoa độc đáo (số lượng cụ thể không được cung cấp trong tài liệu có sẵn)

**Số lượng Người nói:** không thống kê

**Số lượng Giới tính:** 2 (nam và nữ, tỷ lệ không xác định)

**Số lượng Phương ngữ:** 3 phương ngữ/giọng vùng miền (Bắc, Trung, Nam - bao phủ tất cả các giọng trong Việt Nam)

**Phong cách Lời nói:** Tư vấn và hội thoại y tế (trong môi trường lâm sàng)

**Năm Xuất bản:** 2024

**Mục đích Chính:** Nhận diện giọng nói (ASR) cho y tế, với nhiều thuật ngữ y khoa

**Chi tiết Kỹ thuật Chính:** 
- Bao gồm các mô hình được huấn luyện trước WER tương đối 40%+ (từ 51.8% xuống 29.6%), 
- Bộ dữ liệu ASR đầu tiên bao phủ tất cả các nhóm bệnh ICD-10 và tất cả các giọng thuộc 3 miền

---

## 5. VLSP 2020-100h ([HugginFace](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h))

**Tiêu chuẩn điểm chuẩn**: Được tạo cho nhiệm vụ chia sẻ VLSP 2020, bộ dữ liệu này tự khẳng định là điểm chuẩn tiêu chuẩn hóa cho ASR tiếng Việt với đại diện cân bằng của ba phương ngữ vùng miền chính.

**Tác giả:** Viện VinBigdata, Hiệp hội Xử lý Ngôn ngữ và Tiếng nói Việt Nam (VLSP)

**Trạng thái Khả dụng:** có thể tải tại [HugginFace](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h)

**Đặc điểm Người tham gia:**
- **Phân bố Địa lý:** Ba phương ngữ với đại diện tỷ lệ (Bắc 50%, Nam 40%, Trung 10%)
- **Tiêu chí Lựa chọn:** Phương pháp hỗn hợp - lời nói đọc được ghi âm bởi người nói tình nguyện qua smartphone, lời nói tự phát thu thập từ nguồn mở
- **Nhóm Tuổi:** Không xác định (có khả năng đa dạng do phương pháp thu thập hỗn hợp)
- **Điều kiện Ghi âm:** Ghi âm smartphone trong các môi trường khác nhau (phần lời nói đọc), các nguồn Internet đa dạng (phần lời nói tự phát)
- **Phạm vi Chủ đề:** Tin tức, truyện, nội dung Wikipedia và các chủ đề tổng quát khác

**Thông số Độ dài:**
- **Tổng Thời lượng:** 100 giờ cho nhiệm vụ ASR-T1 (250 giờ tổng cộng trên các biến thể nhiệm vụ)
- **Tổng số Bản ghi/Phát ngôn:** Không xác định chính xác
- **Độ dài Trung bình:** Không xác định

**Kích thước Từ vựng:** Không được ghi chép trong các nguồn có sẵn

**Số lượng Người nói:** Không xác định

**Số lượng Giới tính:** 2 (nam và nữ, tỷ lệ không xác định)

**Số lượng Phương ngữ:** 3 phương ngữ vùng miền (Bắc, Trung, Nam) với đại diện cân bằng

**Phong cách Lời nói:** Hỗn hợp - khoảng 20% lời nói đọc (người nói đọc bản phiên âm chuẩn bị sẵn), 80% lời nói tự phát (từ nguồn mở)

**Năm Xuất bản:** 2020

**Mục đích Chính:** Điểm chuẩn cho nhiệm vụ chia sẻ ASR tiếng Việt, đánh giá nhận dạng giọng nói tổng quát

**Chi tiết Kỹ thuật Chính:** Tần số lấy mẫu 16 kHz, định dạng WAV với bản phiên âm văn bản, độ chính xác phiên âm 96% cho các phần lời nói tự phát, bao gồm hai nhiệm vụ đánh giá (ASR-T1 với 100h dữ liệu huấn luyện, ASR-T2 với dữ liệu huấn luyện không giới hạn), được sử dụng rộng rãi làm điểm chuẩn trong nghiên cứu ASR tiếng Việt

---

## Phân Tích So Sánh

### Tiến Hóa Bao Phủ Phương Ngữ
Phát triển quan trọng nhất trong các bộ dữ liệu ASR tiếng Việt gần đây là **bước đột phá của ViMD trong bao phủ cấp tỉnh**. Trong khi các bộ dữ liệu từ 2017-2023 chỉ nắm bắt 3-5 phương ngữ vùng miền, ViMD cung cấp bao phủ đầy đủ tất cả 63 phương ngữ cấp tỉnh.

### Quy Mô Bộ Dữ Liệu và Khả Năng Truy Cập
**Bud500 dẫn đầu về thời lượng** với 500 giờ, tiếp theo là 1,216 giờ của VietMed (mặc dù phần lớn chưa gắn nhãn), LSVSC và VLSP 2020 mỗi bộ 100 giờ, và ViMD với 102.56 giờ. 

### Thách Thức Tồn Tại
- **Mất cân bằng giới tính** vẫn là vấn đề trên các bộ dữ liệu. ViMD cho thấy tỷ lệ nam-nữ 3:1
- **Mất cân bằng phương ngữ** cũng tồn tại, với LSVSC nghiêng nặng về phương ngữ Bắc (88.1%). Chỉ có VLSP 2020 đạt tỷ lệ  đại diện vùng miền tương đối  (50% Bắc, 40% Nam, 10% Trung).

## Bảng Tổng Kết

| Bộ Dữ Liệu    | Năm  | Thời Lượng | Phương Ngữ  | Người Nói      | Công Khai | Bao Phủ Cấp Tỉnh |
| ------------- | ---- | ---------- | ----------- | -------------- | --------- | ---------------- |
| **ViMD**      | 2024 | 102.56h    | 63 cấp tỉnh | 12,955         | Có        | Tất cả 63 tỉnh   |
| **Bud500**    | 2024 | ~500h      | 3 vùng      | Không xác định | Có        | Chỉ cấp vùng     |
| **LSVSC**     | 2024 | 100.5h     | 5 loại      | Nhiều          | Có        | Vùng + thiểu số  |
| **VietMed**   | 2024 | 1,216h     | 3 vùng      | Số lượng lớn   | Có        | Vùng (y khoa)    |
| **VLSP 2020** | 2020 | 100h       | 3 vùng      | Không xác định | Có        | Vùng cân bằng    |

## Khuyến Nghị Cho Các Nhà Nghiên Cứu

**Cho nghiên cứu phương ngữ cấp tỉnh chi tiết:** ViMD là lựa chọn khả thi duy nhất với bao phủ 63 tỉnh độc đáo, làm cho nó không thể thiếu cho các nghiên cứu ngôn ngữ học và hệ thống ASR nhận biết phương ngữ.

**Cho huấn luyện ASR tổng quát quy mô lớn:** 500 giờ nội dung đa dạng của Bud500 cung cấp dữ liệu huấn luyện rộng lớn nhất để xây dựng các mô hình đa mục đích mạnh mẽ.

**Cho đánh giá đa phương ngữ cân bằng:** VLSP 2020 cung cấp đại diện vùng miền cân bằng nhất, làm cho nó lý tưởng cho so sánh điểm chuẩn mặc dù truy cập hạn chế.

**Cho ứng dụng y tế chuyên biệt:** VietMed cung cấp từ vựng y tế chuyên biệt và bối cảnh lâm sàng không có ở nơi khác.

**Cho đa dạng phương ngữ bao gồm thiểu số:** LSVSC độc đáo bao gồm phương ngữ Tây Nguyên và dân tộc thiểu số, mặc dù có sự mất cân bằng đáng kể thiên về phương ngữ Bắc.

Sự hội tụ của năm bộ dữ liệu này trong giai đoạn 2020-2024 đại diện cho tiến bộ đáng kể cho công nghệ giọng nói tiếng Việt, chuyển đổi cảnh quan từ các bộ dữ liệu 3 phương ngữ hạn chế sang bao phủ cấp tỉnh toàn diện, quy mô lớn và chuyên môn hóa theo lĩnh vực.
