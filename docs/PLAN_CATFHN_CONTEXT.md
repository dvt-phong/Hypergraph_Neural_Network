# Plan: vì sao CA-TFHN (103 epoch) cao hơn M trên OULAD

Ngày: 2026-10-20. Em chọn H1; đã code ngày 2026-10-20 (enhanced_context = 0 trong train.py, cả XuetangX và OULAD).

## 1. Kết quả server (OULAD, 5 seed, ngưỡng 0.5)

| Mô hình | test AUC | F1 |
|---|---|---|
| CA-TFHN, 103 epoch | 0.8100 ± 0.0015 | 0.6222 ± 0.0052 |
| M | 0.8059 ± 0.0008 | 0.6102 ± 0.0051 |

CA-TFHN − M = +0.0040 ± 0.0008, cao hơn ở cả 5 seed (+0.0030 đến +0.0049). Hết F1 = 0.

## 2. Nguyên nhân: "enhanced context" của CA-TFHN không chỉ là X55

Ngoài X55, CA-TFHN còn có `enhanced_context` (32 chiều). Nó là embedding của bước link prediction trên đồ thị người học – khóa học: mỗi người học và mỗi khóa có **một embedding riêng (theo ID)**, học từ việc ai đăng ký khóa nào, gồm **mọi kỳ, kể cả kỳ sau**. Plan X59-only (2026-10-18) đã đưa đặc trưng của node user / course về 0, nhưng embedding theo ID vẫn còn.

### 2.1 Context mang thông tin gì (XGBoost, seed 1, cùng cài đặt cho mọi dòng)

| Đầu vào | test AUC | người học 1 enrollment | người học nhiều enrollment |
|---|---|---|---|
| X55 | 0.7814 | 0.7916 | 0.7552 |
| X55 + enhanced context | **0.8253** | 0.8138 | 0.8113 |
| X55 + danh tính khóa học (one-hot 22 khóa) | 0.8041 | 0.8121 | 0.7610 |
| X55 + cấu trúc đăng ký (có học lại cùng module ở kỳ sau, có kỳ sau / kỳ trước, số enrollment) | 0.8184 | 0.7910 | 0.8624 |
| X55 + khóa học + cấu trúc đăng ký | 0.8361 | 0.8136 | 0.8708 |
| X55 + enhanced context + cấu trúc đăng ký | 0.8346 | 0.8126 | 0.8681 |

Chỉ dùng enhanced context, đoán được khóa học với độ chính xác 87.6%, đoán được "có học lại cùng module ở kỳ sau" với AUC 0.73.

Kết luận: enhanced context ≈ **danh tính khóa học** (tỉ lệ Withdrawn theo khóa từ 6.8% đến 47.6%) + **cấu trúc đăng ký của người học, kể cả tương lai** (406 enrollment test có học lại cùng module ở kỳ sau, 90.9% Withdrawn). Cả hai đều không nằm trong X55; M chỉ thấy chúng gián tiếp (Course hyperedge lấy trung bình X55 của khóa, User hyperedge lấy trung bình X55 của người học).

### 2.2 Xác nhận trên chính CA-TFHN (máy em, CPU, seed 1, 103 epoch)

| | test AUC |
|---|---|
| CA-TFHN gốc | 0.8057 |
| CA-TFHN, enhanced context = 0 (đồ thị giữ nguyên) | 0.8007 |
| M (server, seed 1) | 0.8067 |

Bỏ context làm mất 0.005, bằng khoảng chênh CA-TFHN − M. Lưu ý: chỉ 1 seed, và CPU cho 0.8057 trong khi GPU trên server cho 0.8106 (chênh 0.005), nên đây là bằng chứng phụ; bằng chứng chính là bảng 2.1.

Trên XuetangX, cũng là enhanced context này nhưng CA-TFHN (0.8592) vẫn thấp hơn M (0.8760); OULAD chỉ có 22 khóa với tỉ lệ bỏ học rất khác nhau và nhiều lần học lại, nên danh tính khóa và cấu trúc đăng ký có giá trị hơn nhiều.

## 3. Hướng xử lý (em chọn)

| | Hướng | Đổi gì | Hệ quả |
|---|---|---|---|
| **H1 (đề xuất)** | Đúng nguyên tắc "chỉ X, không thông tin khóa học / người học" (2026-10-18): `enhanced_context = 0` cho CA-TFHN, giữ số chiều 32, giữ đồ thị bạn cùng khóa | 1 dòng trong `train.py` (patch), cho **cả XuetangX và OULAD** | chạy lại CA-TFHN `full` trên 2 dataset (không cần dựng lại features); dự kiến OULAD khoảng 0.80, dưới M |
| H2 | Giữ nguyên, báo cáo trung thực: CA-TFHN > M trên OULAD, kèm bảng 2.1 giải thích lý do | không đổi code | bảng chính có một baseline thắng M nhờ thông tin ngoài X |
| H3 | Cho M cùng thông tin (thêm one-hot khóa + cấu trúc đăng ký vào X) | X của mọi mô hình | trái nguyên tắc thầy hướng dẫn; không đề xuất |

H1 nhất quán với cách đã làm cho CFIN (bỏ feat_augment) và MST-GCN (node course / object = 0): baseline chỉ được thấy X. Đồ thị bạn cùng khóa của CA-TFHN vẫn giữ, giống như M giữ hypergraph của mình.

## 4. Nếu chọn H1

1. `train.py` (CA-TFHN): sau khi đọc đồ thị, `graph.enhanced_context = 0` (cùng kích thước), ghi chú lý do; tạo lại patch, kiểm tra trên bản clone sạch.
2. Kiểm tra: seed 1 local OULAD ra lại 0.8007 (cùng code chẩn đoán); smoke XuetangX chạy được.
3. Server: `bash baseline/CA-TFHN/run_catfhn.sh setup`, rồi chạy `full` trên XuetangX (15 epoch) và OULAD (103 epoch); dời kết quả cũ sang `_CA-TFHN_with_context` trước khi chạy.
