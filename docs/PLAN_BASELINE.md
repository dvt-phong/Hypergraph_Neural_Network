# Kế hoạch so sánh baseline (XuetangX + OULAD)

Ngày lập: 2026-10-09 (sửa lần 1). **Chưa code.** Em duyệt và trả lời các câu ở mục 6 trước khi thầy làm.

**Thay đổi so với bản đầu (theo ý em):** bỏ nhóm C (hypergraph tổng quát) và nhóm D riêng, vì hypergraph chưa có trong bài toán dropout prediction nên chỉ so với các phương pháp dropout đã có. Giữ nhóm A + B, thêm **DP-SCL** vào nhóm B.

Mô hình của mình: **M** (Course + Object + User + self-loop, học W, feature `full`, HGNN 2 layer + MLP), giao thức transductive như hiện tại (commit d5c7771), 5 seed, chọn epoch theo val AUC.

---

## 1. Vì sao cần nhiều nhóm baseline

Mỗi nhóm trả lời một câu hỏi khác nhau mà reviewer sẽ hỏi:

| Nhóm | Câu hỏi reviewer | Nếu thiếu thì sao |
|---|---|---|
| **A. Tabular** (cùng X với M) | "Feature tốt là đủ, cần gì graph?" | Không chứng minh được graph có ích |
| **B. Bài dropout MOOC đã công bố** (em liệt kê) | "Có hơn state of the art của bài toán này không?" | Bài báo bị đánh giá thiếu so sánh với lĩnh vực |

## 2. Danh sách thầy gợi ý

Mức ưu tiên: **1** = bắt buộc, **2** = nên có, **3** = tùy thời gian.

### A. Tabular, cùng X (89 cột XuetangX, 108 cột OULAD), cùng split

| Mã | Mô hình | Ưu tiên | Ghi chú |
|---|---|---|---|
| LR | Logistic Regression | 1 | Mốc dưới, chạy vài giây |
| HGB | HistGradientBoosting (hoặc XGBoost) | 1 | Mô hình chính của Wu et al. 2026 trên OULAD; trên tabular thường rất mạnh, là đối thủ thật sự |
| MLP | Chỉ nhánh MLP của M, không HGNN | 1 | Cặp đối xứng với kịch bản H. Đây là phép so sánh "graph có ích không" sạch nhất (bộ nhớ cũ: graph +0.0044 so với MLP trên XuetangX) |

### B. Bài dropout MOOC đã công bố (code gốc, đã clone trong `baseline/`)

| Mã | Bài | Dữ liệu gốc | Ưu tiên | Tình trạng |
|---|---|---|---|---|
| CFIN | Feng et al., AAAI 2019 | XuetangX | 1 | XuetangX đã chạy: test AUC 0.8569 (1 seed, log1p, giao thức gốc). OULAD chưa có |
| MST-GCN | (src_xuet) | KDD Cup 2015, XuetangX | 1 | Cần DGL → chỉ chạy trên server Linux |
| SIG-Net | ACM SAC 2024 | KDD Cup 2015, NAVER | 1 | Cần DGL. Đồ thị sinh viên–object–khóa học, gần ý tưởng của M nhất → phải có |
| CA-TFHN | Classmates Augmented Time-Flow Hybrid Network | XuetangX | 1 | LSTM + self-attention + đồ thị bạn cùng lớp. Đã chạy được trên CPU (code cũ ở tag `full-hsl`) |
| DP-SCL | (em bổ sung: tên bài, nơi đăng, link code) | ? | 1 | **Chưa có thông tin.** Thư mục `D:/Project/PhD/DP-SCL` đang trống, tìm web không ra. Em gửi thầy bài báo / link code |

Code adapter XuetangX cho 4 bài này đã từng viết (tag `full-hsl`: `src/10_baseline_data.py`, `12_signet.py`, `13_mstgcn.py`, `14_catfhn.py`), nhưng theo giao thức inductive cũ → cần sửa lại cho split/giao thức hiện tại.

## 3. Đưa các bài nhóm B sang OULAD

Bốn bài dưới không có sẵn code cho OULAD, cần viết bộ chuyển dữ liệu (adapter) sang đúng định dạng đầu vào của từng code gốc. Mô hình giữ nguyên.

| Bài | Đầu vào gốc cần | Lấy từ OULAD | Rủi ro |
|---|---|---|---|
| CFIN | Số hoạt động theo loại × ngày + bối cảnh người học + bối cảnh khóa học | `studentVle` theo `activity_type` × ngày 0–34; studentInfo; module | Code TF1 (Python 3.7), viết thêm `feat_extract` cho OULAD |
| MST-GCN | Chuỗi hoạt động theo ngày theo loại + đồ thị | như CFIN | Cần DGL, chạy server |
| SIG-Net | Đồ thị sinh viên–object–khóa học | object = `id_site`, khóa học = `code_module` + `code_presentation` | Khớp tự nhiên nhất. Phải bỏ cạnh gắn nhãn (bản gốc dùng nhãn test làm loại cạnh → rò rỉ) |
| CA-TFHN | Bản ghi hoạt động theo thời gian + quan hệ người học–khóa học | `studentVle` + bảng đăng ký | OULAD ít người học nhiều khóa (User chỉ phủ 22.5%, XuetangX 94.5%) → đồ thị bạn cùng lớp yếu. Đây là đặc điểm dữ liệu, cứ báo cáo |
| DP-SCL | Chờ bài báo | | |

Trên OULAD, dòng HGB ở nhóm A chính là mô hình chính của Wu et al. 2026 (chạy trên X và split của mình). Kết quả pipeline gốc của Wu trên split gốc (0.7933) chỉ ghi ở chú thích, không đưa vào bảng.

## 4. Giao thức chung (để so sánh công bằng)

| Yếu tố | Đề xuất |
|---|---|
| Split | Đúng split của M: XuetangX test 67,699; OULAD split hiện tại trong `data/processed/oulad` |
| Chọn epoch | Train trên train, chọn epoch theo **val AUC**, test một lần (như M) |
| Seed | 5 seed giống M, báo mean ± std |
| Chỉ số | Giống cột của M: AUC, AUPRC, F1, … |
| Nhãn | Không baseline nào được thấy nhãn val/test (kể cả qua cấu trúc đồ thị như SIG-Net gốc) |
| Số liệu trong bài gốc | Không đưa vào cùng bảng (khác split, khác giao thức). Nếu muốn thì để ở chú thích |

## 5. Bảng kết quả cuối cùng (dự kiến)

| Nhóm | Mô hình | XuetangX AUC | XuetangX AUPRC | OULAD AUC | OULAD AUPRC |
|---|---|---|---|---|---|
| A | LR / HGB / MLP | | | | |
| B | CFIN / MST-GCN / SIG-Net / CA-TFHN / DP-SCL | | | | |
| | **M (của mình)** | | | | |

## 6. Em cần quyết định

- **Q0. DP-SCL:** em gửi tên bài đầy đủ (tác giả, năm, nơi đăng) và link code. Thầy cần đọc code trước mới biết nó cần đầu vào gì cho XuetangX và OULAD.
- **Q1. Đầu vào cho nhóm B:** (a) mỗi bài dùng cách trích feature của chính nó (thầy đề xuất, vì feature là một phần đóng góp của bài đó) hay (b) cho tất cả dùng X của mình?
- **Q2. Giao thức cho nhóm B:** (a) giao thức chung ở mục 4 (thầy đề xuất) hay (b) giữ giao thức gốc từng bài? Lưu ý CFIN hiện đang chạy giao thức gốc (train trên cả train + val, epoch 27 cố định, 1 seed). Nếu chọn (a) thì phải chạy lại CFIN XuetangX.
- **Q3. Thứ tự làm:** thầy đề xuất A trước (nhanh, có ngay mốc so sánh), rồi B trên XuetangX (đã có code cũ), cuối cùng B trên OULAD (tốn công nhất).
