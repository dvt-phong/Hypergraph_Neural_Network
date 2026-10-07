# Feature engineering và chuẩn hóa đặc trưng (trạng thái hiện tại)

Ngày viết: 2026-10-06. Mô tả đúng code ở commit `89f02c4` (nhánh `main`): `src/0_config.py`, `src/2_preprocess.py`,
`src/3_features.py`. Mục 8 ghi vấn đề đang mở về cách chuẩn hóa và đề xuất, **chưa áp dụng vào code**.

---

## 1. Tổng quan

- Mỗi node là một **enrollment** (một người học trong một khóa học) của bộ XuetangX công khai (bản phát hành của CFIN).
- Mỗi node có một vector đặc trưng **X gồm 89 cột**, chia thành 3 khối:

| Khối | Cột | Số cột | Theo |
|---|---|---|---|
| Hành vi | 0–58 | 59 | MST-GCN (đặc trưng của node enrollment) |
| Người học | 59–70 | 12 | CFIN (Definition 4) |
| Khóa học | 71–88 | 18 | CFIN (Definition 4) |

- Dữ liệu chia thành: train 126,354, validation 31,589 (80/20 train chính thức, xáo với seed 1), test 67,699
  (test chính thức, giữ nguyên).
- **Mọi thống kê chuẩn hóa (μ, σ) chỉ tính trên train**, rồi áp nguyên cho validation và test.
- Thông tin quan hệ (cùng khóa, cùng object, cùng người học) **không** làm thành đặc trưng thủ công; mô hình học qua
  hyperedge Course, Object, User của hypergraph.

## 2. Tiền xử lý sự kiện (`2_preprocess.py`)

| Bước | Cách làm |
|---|---|
| Ngày của sự kiện | d = ngày của sự kiện − ngày bắt đầu khóa trong `course_info.csv` |
| Cửa sổ quan sát | chỉ giữ 0 ≤ d < 35 (35 ngày đầu, như CFIN: D_h = 35) |
| Loại action | 22 loại action có trong log (log gốc: 42,110,402 sự kiện, đúng 22 loại). `close_info` không xuất hiện trong log nên không có |
| Enrollment không có sự kiện | vẫn giữ (3,305 enrollment), mọi số đếm bằng 0 |
| Thông tin ngữ cảnh ghi kèm | gender, education, **birth**, course_start, course_end, category |

## 3. Danh sách cột

### 3.1. Khối hành vi (59 cột)

| Cột | Tên | Ý nghĩa | Xử lý hiện tại |
|---|---|---|---|
| 0–34 | `activity_day_0` … `activity_day_34` | số sự kiện ở ngày d | **để thô**: x = c |
| 35 | `total_events` | tổng sự kiện trong 35 ngày (= tổng 35 cột trên) | z-score |
| 36 | `distinct_objects` | số object khác nhau đã dùng (video, bài tập, bài diễn đàn), mọi action | z-score |
| 37–41 | video: seek, play, pause, stop, load | số lần trong 35 ngày | z-score |
| 42–47 | bài tập: problem_get, problem_check, problem_save, reset_problem, problem_check_correct, problem_check_incorrect | | z-score |
| 48–52 | diễn đàn: create_thread, create_comment, delete_thread, delete_comment, close_forum | | z-score |
| 53–58 | trang web: click_info, click_courseware, click_about, click_forum, click_progress, close_courseware | | z-score |

### 3.2. Khối người học (12 cột)

| Cột | Tên | Xử lý |
|---|---|---|
| 59–61 | gender: female, male, **missing** | one-hot (0/1), không chuẩn hóa |
| 62–69 | education: Associate, Bachelor's, Doctorate, High, Master's, Middle, Primary, **missing** | one-hot (0/1), không chuẩn hóa |
| 70 | `age` | z-score (mục 4.3) |

### 3.3. Khối khóa học (18 cột)

| Cột | Tên | Xử lý |
|---|---|---|
| 71–88 | category: 17 lĩnh vực (art, biology, business, chemistry, computer, economics, education, electrical, engineering, foreign language, history, literature, math, medicine, philosophy, physics, social science) + **missing** | one-hot (0/1), không chuẩn hóa |

## 4. Chuẩn hóa (`3_features.py`, hàm `build_features`)

### 4.1. 35 cột theo ngày: để thô (như MST-GCN)

```
x = c
```

Thống kê trên train: 91.8% số ô bằng 0; giá trị lớn nhất 70,462 sự kiện trong một ngày.

### 4.2. Tổng sự kiện, số object, 22 cột action: z-score (như MST-GCN, nhưng chỉ fit trên train)

```
x = (c − μ_train) / σ_train          μ, σ tính riêng cho từng cột trên 126,354 node train
```

- Cột nào có σ = 0 thì đặt σ = 1.
- Trên train: mean ≈ 0, std = 1 cho mọi cột. Trên validation/test: mean −0.011 … 0.018.
- Giá trị lớn nhất sau chuẩn hóa: 292.6 (`problem_get`; một enrollment có 128,948 lần).

### 4.3. Age: theo CFIN

```
a = năm bắt đầu khóa − năm sinh
a = 0     nếu thiếu năm sinh, hoặc a < 10, hoặc a > 70
x = (a − μ_train) / σ_train          μ_train = 7.559, σ_train = 12.742 (tính cả các dòng a = 0)
```

- Tuổi không rõ → x = −0.593; tuổi thật 10…70 → x = 0.19…4.90. Hai nhóm tách hẳn nhau, nên giá trị 0 đóng vai trò cờ "thiếu".
- Khác CFIN ở một điểm: CFIN tính `2018 − năm sinh`; ở đây dùng năm bắt đầu khóa (các khóa diễn ra năm 2015–2017),
  để tuổi đúng với thời điểm học.
- Không có cột `age_missing` riêng, giống CFIN.

### 4.4. One-hot: không chuẩn hóa

Giữ 0/1. Chuẩn hóa one-hot sẽ biến các mức hiếm thành số rất lớn (ví dụ `education_primary` chiếm 0.1% → khoảng 31).
One-hot nhân với lớp Linear tương đương với tra embedding, đúng cách CFIN biểu diễn biến phân loại.

## 5. Giá trị thiếu

| Trường | Tỉ lệ thiếu (enrollment) | Cách xử lý |
|---|---|---|
| gender | 57.5% | cột `gender_missing` = 1 (thiếu là một mức riêng) |
| education | 67.6% | cột `education_missing` = 1 |
| age | 71.5% (71.2% thiếu năm sinh, 0.4% tuổi ngoài 10–70) | a = 0 rồi chuẩn hóa |
| category | 0.4% (2 khóa: `TsinghuaX+80590952X+2016_T2`, `TsinghuaX+20220214X+2017_T1`) | cột `category_missing` = 1 |
| course_start, hồ sơ learner | 0% | — |

- Thiếu theo cụm: 56.8% enrollment thiếu cả gender, education và age; 26.9% có đủ cả ba.
- **Không có cột `_other`**: dữ liệu không có giá trị nào ngoài từ điển (các cột `_other` trước đây luôn bằng 0).
  Nếu gặp giá trị lạ, `3_features.py` báo lỗi và dừng thay vì lặng lẽ xếp vào một cột.

## 6. Bộ đặc trưng theo kịch bản

| Bộ | Kịch bản | Gồm | Số cột |
|---|---|---|---|
| `feature` | F1 | hành vi | 59 |
| `feature+user` | F2 | hành vi + người học | 71 |
| `feature+course` | F3 | hành vi + khóa học | 77 |
| `full` | M, A1–A4, L1, B1, W1 | cả ba khối | 89 |

## 7. So sánh với các bài baseline (theo code công bố)

| Mục | CFIN | MST-GCN | CA-TFHN | Hiện tại |
|---|---|---|---|---|
| Chuỗi theo ngày | không có | 35 tổng/ngày, để thô | 35 × 22, để thô | 35 tổng/ngày, để thô |
| Số action | 21 | 22 | 22 | 22 |
| Cột tổng hợp | `all#count`, `session#count` | tổng sự kiện, số object | không | tổng sự kiện, số object |
| Biến đổi số đếm | z-score | ngày thô; còn lại z-score | thô | ngày thô; còn lại z-score |
| Gender | mã m/f, nhưng dữ liệu ghi `male`/`female` nên **luôn bằng 0** | không dùng | cùng lỗi với CFIN | one-hot + missing |
| Education | tính nhưng không đưa vào mô hình | không dùng | mã số → z-score | one-hot + missing |
| Age | 2018 − năm sinh, thiếu → 0 | không dùng | 2023 − năm sinh, thiếu → 0 | năm bắt đầu khóa − năm sinh, thiếu → 0 |
| Category | mã số → embedding | không dùng | mã số → z-score | one-hot + missing |
| Fit thống kê trên | train + test | train + test | train + test | **chỉ train** |

## 8. Vấn đề đang mở: chuẩn hóa số đếm theo ngày

**Hiện tượng.** Chạy M với cách chuẩn hóa hiện tại (ngày để thô) trên server, seed 1 và 11: test AUC khoảng 0.86.
Các lần trước đều dùng log(1 + c) rồi z-score (có trong code từ 18/09, commit `d1c0b5d`), kể cả lần 0.8749 (`ae90e29`)
và lần X 90 cột (`5c0b072`, M = 0.8729 ± 0.0003).

**Thử nhanh không dùng graph** (MLP giống nhánh MLP của mô hình, cùng protocol, 5 seed, khối hành vi 35 ngày + 21 action):

| Chuẩn hóa | Val AUC | Test AUC |
|---|---|---|
| log(1 + c) rồi z-score | 0.8649 ± 0.0004 | 0.8653 ± 0.0002 |
| Quy tắc CFIN: z-score trên số đếm thô | 0.8535 ± 0.0004 | 0.8511 ± 0.0004 |
| Quy tắc MST-GCN: ngày thô, action z-score | 0.8524 ± 0.0010 | 0.8519 ± 0.0011 |

**Bằng chứng cho log rồi z-score.**

- **Kloft et al. (2014), dự đoán dropout MOOC** (EMNLP 2014, tr. 60–65), dùng đặc trưng đếm từ clickstream. Nguyên văn mục 3:
  *"Box plots of these features showed that the distribution is highly skewed and non-normal, and furthermore all
  features are non-negative. We thus tried two standard features transformations: 1. logarithmic transformation
  2. box-cox transformation. […] The logarithmic transformation is however much faster and lead to better results in
  later pipeline steps, which is why it was taken for the remaining experiments. Subsequently, all features were
  centered and normalized to unit standard deviation."*
- **Zhuang et al. (SIGIR 2020)**: mạng nơ-ron bị ảnh hưởng bởi đặc trưng lệch; dùng sgn(x)·log(1 + |x|);
  NDCG@10 từ 47.26 (thô) lên 49.88 (log1p) trên MSLR-WEB30K, từ 67.09 lên 68.12 trên Istella, đều có ý nghĩa thống kê.
- **Bartlett (1947)**: log(x + 1) cho dữ liệu đếm có số 0. Không dùng được log(x) nguyên dạng vì 91.8% số ô bằng 0.
  Cơ số log không ảnh hưởng vì z-score loại bỏ hằng số nhân.

**Đề xuất (chờ duyệt).**

1. Khối hành vi: giữ nguyên bộ 59 cột (theo MST-GCN), đổi biến đổi thành `x = (log(1 + c) − μ_train) / σ_train`
   cho cả 59 cột, theo Kloft et al. (2014).
2. Thêm kịch bản ablation N1 = M nhưng giữ cách chuẩn hóa của MST-GCN, chạy 5 seed, để có bằng chứng trên chính dữ liệu XuetangX.
3. Dự kiến M quay về khoảng 0.8725–0.8735.

Câu gợi ý cho bài:

> *Activity counts are non-negative and heavily skewed (91.8% zeros; up to 70,462 events per day). Following
> Kloft et al. (2014), they are log-transformed and then standardized with training statistics; since most counts
> are zero, we use log(1 + c) (Bartlett, 1947).*

## 9. Kiểm tra đã làm (2026-10-05, bản dựng thử trên máy, không đụng `data/processed`)

| # | Kiểm tra | Kết quả |
|---|---|---|
| K1 | 89 cột cho cả 3 split; `age` ở cột 70; không có `_other`; không có NaN/inf | đạt |
| K2 | `total_events` bằng tổng 35 cột theo ngày (sau chuẩn hóa) | đạt |
| K3 | 24 cột z-score trên train: mean ≈ 0, std = 1 | đạt |
| K4 | 35 cột theo ngày là số nguyên thô, 0…70,462 | đạt |
| K5 | age: tuổi không rõ = −0.593, chiếm 71.5% | đạt |
| K6 | one-hot ngữ cảnh trùng X cũ (bỏ `_other`) | đạt |
| K7 | hypergraph giống hệt bản cũ | đạt |
| K8 | 1,879 node train không có sự kiện cùng một giá trị `distinct_objects` | đạt |
| — | chạy thử 11 kịch bản, seed 1, 10 epoch | chạy trọn, không lỗi, không NaN |

## 10. Tài liệu tham khảo

- Feng, W., Tang, J., Liu, T. X. (2019). Understanding Dropouts in MOOCs. AAAI 2019.
  Code: https://github.com/wzfhaha/dropout_prediction
- MST-GCN, Scientific Reports 2026. https://doi.org/10.1038/s41598-026-40502-w. Code: https://github.com/wudongze9/MST-GCN
- Liang, G. et al. (2023). MOOCs Dropout Prediction via Classmates Augmented Time-Flow Hybrid Network. ICONIP 2023.
  https://doi.org/10.1007/978-981-99-8184-7_31. Code: https://github.com/codeds27/CA-TFHN
- Kloft, M., Stiehler, F., Zheng, Z., Pinkwart, N. (2014). Predicting MOOC Dropout over Weeks Using Machine Learning
  Methods. EMNLP 2014, pp. 60–65. https://aclanthology.org/W14-4111.pdf
- Zhuang, H., Wang, X., Bendersky, M., Najork, M. (2020). Feature Transformation for Neural Ranking Models. SIGIR 2020.
  https://doi.org/10.1145/3397271.3401333
- Bartlett, M. S. (1947). The use of transformations. Biometrics, 3, 39–52.
- Josse, J., Chen, J. M., Prost, N., Scornet, E., Varoquaux, G. (2024). On the consistency of supervised learning with
  missing values. Statistical Papers. https://arxiv.org/abs/1902.06931
