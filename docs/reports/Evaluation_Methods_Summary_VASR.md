# Đánh giá Speech-to-text model
## 1. Word Error rate - Tỷ lệ lỗi từ vụng
$$ 
WER = \frac{S + D + I}{N} × 100 \% 
$$

- $S$ (Substitution): số từ bị thay thế sai
- $D$ (Deletion): số từ bị thiếu
- $I$ (Insertion): số từ thêm vào thừa
- $N$: tổng số từ trong văn bản 
### Ý nghĩa
- **Giá trị thấp** (gần 0%) = mô hình tốt
- **WER > 100%** là có thể (khi lỗi nhiều hơn số từ gốc)
- Phản ánh **tỷ lệ sai tổng thể** của mô hình
- Có thể phân tích **loại lỗi chủ yếu** (S, D, hay I chiếm đa số)
### Hạn chế
- **Không phân biệt mức độ nghiêm trọng**: từ quan trọng và từ không quan trọng được tính như nhau
- **Không xét đến ngữ nghĩa**: "không tốt" và "tồi" có nghĩa tương tự nhưng vẫn bị tính lỗi
- **Phụ thuộc vào chuẩn hóa văn bản**: cách viết số, ký hiệu ảnh hưởng kết quả

---

## 2. Character Error Rate (CER) - Tỷ Lệ Lỗi Ký Tự

### Công thức
$$
CER = \dfrac{S_c + D_c + I_c}{N_c} × 100%
$$
- $S_c, D_c, I_c$: lỗi thay thế, thiếu, thừa ở cấp độ ký tự
- $N_c$: tổng số ký tự trong văn bản chuẩn

### Ý nghĩa
- **Hữu ích cho ngôn ngữ không có khoảng trắng** (tiếng Trung, Nhật, Thái)
- **CER thường thấp hơn WER** vì một từ sai chỉ ảnh hưởng một số ký tự
- Phản ánh **độ chính xác chi tiết** hơn WER

### Hạn chế
- **Ít có ý nghĩa với ngôn ngữ có từ dài**: một từ sai có thể tạo nhiều lỗi ký tự
- **Không phản ánh lỗi ở cấp độ từ/câu**
- **Khó diễn giải**: CER = 5% không cho biết bao nhiêu từ bị sai

---

## 3. Word Information Lost (WIL) / Word Information Preserved (WIP)

### Công thức
$$ WIL = \dfrac{S + D + I}{N + I} $$
$$ WIP = 1 - WIL $$
- $S$ (Substitution): số từ bị thay thế sai
- $D$ (Deletion): số từ bị thiếu
- $I$ (Insertion): số từ thêm vào thừa
- $N_\text{ref}$: tổng số từ trong văn bản gốc (văn bản tham chiếu)
- $N_\text{ref}$: tổng số từ trong văn bản dự đoán (văn bản đầu ra)
### Ý nghĩa
- **Chuẩn hóa theo độ dài output** thay vì độ dài reference
- **WIP cao** = giữ được nhiều thông tin từ bản gốc
- Xem xét **cả việc thêm từ** ảnh hưởng đến thông tin
### Hạn chế
- **Ít phổ biến**: khó so sánh giữa các nghiên cứu
- **Vẫn không xét ngữ nghĩa**
- **Phức tạp hơn** trong việc giải thích so với WER

---

## 4. Sentence Error Rate (SER) - Tỷ Lệ Lỗi Câu

### Công thức
$$
SER = \dfrac{\text{Số câu có ít nhất 1 lỗi}}{\text{Tổng số câu}} × 100%
$$

### Ý nghĩa
- Đo **tỷ lệ câu hoàn hảo** vs câu có lỗi
- **Đánh giá khắt khe**: 1 lỗi nhỏ = cả câu sai
- Hữu ích cho ứng dụng cần **độ chính xác tuyệt đối**

### Hạn chế
- **Quá khắt khe**: không phân biệt câu có 1 lỗi vs 10 lỗi
- **Không cho biết mức độ nghiêm trọng** của lỗi
- **Phụ thuộc vào cách phân đoạn câu**

---

## 5. Match Error Rate (MER) - Tỷ lệ khớp/lỗi

### Công thức
$$ MER = \dfrac{S + D + I}{\max(N_\text{ref}, N_\text{hyp})} $$
- $S$ (Substitution): số từ bị thay thế sai
- $D$ (Deletion): số từ bị thiếu
- $I$ (Insertion): số từ thêm vào thừa
- $N_\text{ref}$: tổng số từ trong văn bản gốc (văn bản tham chiếu)
- $N_\text{ref}$: tổng số từ trong văn bản dự đoán (văn bản đầu ra)

### Ý nghĩa
- Chuẩn hoá tỉ lệ lỗi - tỷ lệ luôn < 0
- Bớt nhạy cảm với những khác biệt nhỏ ở văn bản đầu ra so với văn bản mẫu

---

## 5. Real-Time Factor (RTF) - Hệ Số Thời Gian Thực

### Công thức
```
RTF = Thời gian xử lý / Thời gian audio
```

### Ý nghĩa
- **RTF < 1**: xử lý nhanh hơn thời gian thực (tốt)
- **RTF = 1**: xử lý đúng bằng thời gian thực
- **RTF > 1**: chậm hơn thời gian thực (không phù hợp ứng dụng real-time)
- Phản ánh **hiệu năng tính toán** của mô hình

### Hạn chế
- **Phụ thuộc phần cứng**: GPU, CPU khác nhau cho kết quả khác nhau
- **Không đo chất lượng transcription**
- **Không tính latency khởi động**: chỉ tính thời gian xử lý thuần túy

---
## 6. Latency - Độ Trễ

### Công thức
```
Latency = Thời điểm nhận output - Thời điểm kết thúc input
```
### Ý nghĩa
- Đo **thời gian chờ đợi** của người dùng
- **Latency thấp** quan trọng cho live captioning, hội thoại
- Có thể đo **latency trung bình** hoặc **latency tối đa** (worst-case)

### Hạn chế
- **Khó so sánh giữa các hệ thống**: streaming vs batch processing
- **Phụ thuộc kiến trúc**: online vs offline models
- **Không phản ánh độ chính xác**
