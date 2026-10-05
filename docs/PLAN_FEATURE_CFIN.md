# Plan: áp dụng feature engineering của CFIN

Ngày lập: 2026-10-05. **Đã code ngày 2026-10-05** (age theo CFIN, bỏ các cột `_other`). Chưa chạy lại trên server.

> **Cập nhật cuối ngày 2026-10-05 (thay thế mục 1, 3, 4, 6, 8 bên dưới ở phần khối hành vi):** em chọn khối hành vi
> **giống MST-GCN**: 35 cột theo ngày để thô; tổng sự kiện, số object khác nhau và **22 action** (kể cả `close_forum`)
> chuẩn hóa z-score trên train; không dùng log1p. Phần ngữ cảnh (one-hot có missing, age theo CFIN) giữ như plan này.
> X = **89 cột**. Bố cục cột, lệnh chạy và kiểm tra: `docs/KE_HOACH_CHAY_LAI_X89.md`.

**Cập nhật sau khi rà soát cột `_other`:** ba cột `gender_other`, `education_other`, `category_other` bằng 0 ở mọi
dòng của train, val, test, vì dữ liệu không có giá trị nào ngoài từ điển. Vì vậy chúng đã bị bỏ, và code **báo lỗi**
nếu sau này gặp giá trị lạ. X hiện có **88 cột** (không phải 91 như bản đầu của plan này). Cột `close_info` (cũng luôn
bằng 0) chưa bỏ, chờ chốt Q4.

**Mục tiêu.** Phần đặc trưng đầu vào của mô hình chính M dựa theo một chuẩn đã công bố, để có dẫn chứng:
- Khối **ngữ cảnh** theo CFIN (Feng et al., AAAI 2019, Definition 4).
- Khối **hành vi** giữ như hiện tại, dẫn chứng theo MST-GCN.

So với X cũ (90 cột): **thêm cột age theo cách của CFIN** và **bỏ 3 cột `_other` luôn bằng 0**. X còn 88 cột.

**Nguồn.**
- CFIN: bài báo https://keg.cs.tsinghua.edu.cn/jietang/publications/AAAI19-Feng-dropout-moocs.pdf và code
  https://github.com/wzfhaha/dropout_prediction (`preprocess.py`, dòng 13–22).
- MST-GCN: https://doi.org/10.1038/s41598-026-40502-w và code `baseline/MST-GCN/MST-GCN/src_xuet`.

---

## 1. Lấy gì từ CFIN, không lấy gì

| Thành phần của CFIN | Quyết định | Lý do |
|---|---|---|
| Ngữ cảnh người học: gender, education ở dạng one-hot (Definition 4) | **Giữ như hiện tại** | Đã đúng Definition 4: *"categorical information is represented by a one-hot vector"* |
| Ngữ cảnh khóa học: category ở dạng one-hot | **Giữ như hiện tại** | Như trên |
| **Age**: *"continuous information (i.e. age) is represented as the value itself"*; trong code, thiếu hoặc ngoài 10–70 thì gán 0, sau đó chuẩn hóa | **Thêm mới** | Đây là phần duy nhất của Definition 4 mà X hiện tại còn thiếu |
| Location | Không lấy | Dữ liệu công khai không có trường này |
| Nhãn cụm learner (K-means) | Không lấy | Cần K-means trên toàn bộ dữ liệu cộng một file nạp sẵn, ngoài phạm vi project tối giản |
| Context-smoothing (mean/max theo user và course) | Không lấy | Hypergraph Course/User của mô hình đang học chính phần thông tin này. Đưa vào X thì không còn đo được đóng góp của graph |
| Số enrollment của learner và của khóa | Không lấy | Chính là kích thước hyperedge \|e_U\| và \|e_C\| |
| 21 action, số đếm thô, chuẩn hóa trên train + test | Không theo | Giữ 23 action và log1p. Thống kê chỉ fit trên train: chặt hơn CFIN, không dùng thông tin của test |

Khối hành vi (35 ngày + 23 action, log1p, chuẩn hóa trên train) **không đổi**. Phần này dẫn chứng theo MST-GCN, vì CFIN không có chuỗi theo ngày.

## 2. Định nghĩa cột age

Với enrollment v của learner u trong khóa c:

```
a_v = year(course_start_c) − birth_u                                 tuổi lúc khóa học bắt đầu
a_v = 0      nếu birth_u bị thiếu, hoặc a_v < 10, hoặc a_v > 70       như CFIN
x_v = (a_v − μ_train) / σ_train                                      μ, σ tính trên mọi node train, kể cả các giá trị 0
```

- Giá trị 0 không bao giờ là một tuổi thật, nên sau khi chuẩn hóa nó vẫn tách hẳn khỏi các tuổi thật. Mạng MLP/HGNN dùng nó như một cờ "thiếu".
- Theo Josse et al. (Statistical Papers, 2024), điền hằng số trước khi học vẫn nhất quán cho bài toán dự đoán.
- Không thêm cột `age_missing` riêng, vì CFIN không có cột này (xem câu hỏi Q3).

Đã đo trên `data/processed/simple` ngày 2026-10-05: 71.5% số enrollment sẽ nhận a_v = 0. Trong đó 71.2% là thiếu năm sinh và 0.4% có năm sinh nhưng tuổi ngoài 10–70.

## 3. Bố cục X mới (88 cột)

| Khối | Cột | Số cột |
|---|---|---|
| Hành vi: 35 ngày + 23 action | 0–57 | 58 |
| Người học: gender (female, male, missing) + education (7 mức + missing) + **age (1)** | 58–69 | 12 |
| Khóa học: category (17 lĩnh vực + missing) | 70–87 | 18 |

| Bộ feature | Trước | Sau |
|---|---|---|
| `feature` (F1) | 58 | 58 |
| `feature+user` (F2) | 71 | **70** |
| `feature+course` (F3) | 77 | **76** |
| `full` (M và các kịch bản còn lại) | 90 | **88** |

Cách mã hóa giá trị thiếu và giá trị lạ:

| Trường | Thiếu | Ngoài từ điển |
|---|---|---|
| gender, education, category | cột `_missing` = 1 (thiếu là một mức riêng, như mã 0 của CFIN) | báo lỗi, dừng `3_features.py` |
| age | a = 0, rồi chuẩn hóa: x = (0 − μ) / σ | tuổi < 10 hoặc > 70 cũng gán a = 0 |

## 4. Sửa code (4 file code + 1 file docs)

| File | Thay đổi |
|---|---|
| `src/0_config.py` | Thêm `AGE_MIN = 10`, `AGE_MAX = 70` (theo CFIN) và `AGE_FEATURE_INDEX` ngay sau khối education; one-hot chỉ còn cột `missing` (bỏ `other`). Thay dòng comment *"Age is left out"* bằng dòng dẫn nguồn CFIN |
| `src/2_preprocess.py` | Thêm cột `birth` vào CSV của từng split, ở cả hai chỗ `writerow` (enrollment có sự kiện và enrollment không có sự kiện). Giá trị lấy từ `user_info.csv` (dict `users` đã có sẵn) |
| `src/3_features.py` | `build_split_features`: tính a_v theo mục 2, ghi chú công thức trên dòng tính. `build_features`: tính μ, σ của age trên train rồi áp dụng cho cả ba split. `feature_metadata`: thêm dòng `("age", "course start year − birth, 0 if missing or outside [10, 70] (CFIN)")`. Thêm CFIN vào phần "Tham khảo" ở đầu file |
| `docs/KICH_BAN_THUC_NGHIEM.md` | Cập nhật bảng số cột (58 / 70 / 76 / 88) và mô tả M |

**Không sửa:** `4_hypergraph.py`, `5_graph_data.py`, `6_hgnn.py`, `7_mlp.py`, `8_model.py`, `9_train.py`, `10_summary.py`.
- `feature_columns()` lấy khoảng cột từ config, nên tự nhận 88 cột.
- H0 không phụ thuộc vào X.

## 5. Chạy

1. **Giữ lại kết quả cũ để so sánh:** đổi tên `outputs/results.csv` thành `outputs/results_x90.csv`, và chép `X.npy` của từng split thành `X_x90.npy`.
2. **Dựng lại dữ liệu:** `2_preprocess.py` → `3_features.py` → `4_hypergraph.py`. Bước 4 chỉ chạy lại cho đồng bộ; H0 phải giống hệt bản cũ.
3. **Chạy thử nhanh:** M với 10 epoch, 1 seed.
4. **Chạy đầy đủ:** `python src/9_train.py --scenario all --seeds 1 11 111 1111 11111`, tức 11 kịch bản × 5 seed = 55 lần, khoảng 4.6 giờ (khoảng 5 phút mỗi lần trên server).

## 6. Kiểm tra (chạy trước bước 5.3)

| # | Kiểm tra | Kết quả phải là |
|---|---|---|
| K1 | Số cột X của cả ba split; `feature_names.csv` | 88 cột; `age` ở cột 69 |
| K2 | X mới bỏ cột 69 (age) so với `X_x90.npy` bỏ 3 cột `_other` (61, 70, 89) | sai khác = 0 (chỉ thêm age, bỏ 3 cột luôn bằng 0) |
| K3 | Tỉ lệ a_v = 0 trước khi chuẩn hóa, trên toàn bộ ba split | khoảng 71.5% |
| K4 | Mean và std của cột age trên train sau chuẩn hóa | khoảng 0 và 1; val/test dùng μ, σ của train |
| K5 | `hypergraph.npz` mới so với bản cũ | giống hệt |

## 7. Đọc kết quả

Ghép cặp theo seed giữa `results.csv` (X 88 cột) và `results_x90.csv` (X 90 cột):
- **Δ(M)** = đóng góp của age khi có đủ ngữ cảnh.
- **Δ(F2)** = đóng góp của age trong khối người học.
- F1 và F3 không chứa age, nên Δ của chúng là **mức nhiễu khi chạy lại**. Đây là đối chứng có sẵn.

Quy tắc kết luận giữ như đợt 03/10: chỉ coi là khác biệt thật khi |Δ| lớn hơn khoảng 2 lần std theo seed (khoảng 0.0006) **và** cùng dấu ở 5/5 seed.

Bất kể kết quả thế nào, mô hình chính vẫn là M với bộ `full` (CFIN). Không chọn bộ feature dựa trên test AUC.

Câu gợi ý cho phần Method:

> *Each enrollment is described by its learning behavior and its context. Following MST-GCN, the behavior part
> holds the daily event counts of the 35-day history and the count of each action type, log-transformed and
> standardized with training statistics. Following CFIN (Feng et al., 2019), the context part one-hot encodes
> gender, education and course category, each with an explicit missing level, and keeps age as a value; age is
> computed at the course start and set to 0 when the birth year is missing or the age falls outside [10, 70].*

## 8. Câu hỏi cần em và giáo viên hướng dẫn chốt trước khi code

| # | Câu hỏi | Đề xuất của thầy |
|---|---|---|
| Q1 | Mốc tính tuổi: năm bắt đầu khóa học, hay năm cố định 2018 như code CFIN? | **Năm bắt đầu khóa** (đã code như vậy). Các khóa diễn ra năm 2015–2017, dùng mốc 2018 sẽ làm tuổi lệch 1–3 năm. Ghi rõ đây là điểm lệch nhỏ so với CFIN |
| Q2 | Khoảng tuổi hợp lệ: 10–70 (CFIN) hay 10–100 (code cũ của em)? | **10–70** (đã code), để khớp với tiền lệ trên cùng bộ dữ liệu |
| Q3 | Có thêm cột `age_missing` riêng không? | **Không** (đã code như vậy), để đúng như CFIN. Giá trị 0 đã đóng vai trò cờ thiếu, và trạng thái thiếu của age trùng với education ở 93% số enrollment (cột `education_missing` đã có) |
| Q4 | Giữ 23 action, hay rút về 21 action như CFIN (bỏ `close_forum`, `close_info`)? | **Giữ 23.** Khối hành vi dẫn chứng theo MST-GCN, mà MST-GCN dùng mọi action có trong log |
