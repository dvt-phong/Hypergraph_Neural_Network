# So sánh feature, thông tin người học và khóa học: XuetangX và OULAD

Ngày 2026-10-08. Mô tả X **sau** bước 2 (preprocess) và bước 3 (feature engineering).
- XuetangX: code `src/` (commit 601b5df). Số liệu đếm trên đủ 225.642 node của 3 file split local.
- OULAD: code `baseline/new_dataset/OULAD/`. Số liệu đếm trên đủ 32.593 node trong `data/processed/oulad`.

Mỗi node chỉ đếm một lần. Số trong ngoặc là số cột của X.

## 1. Nhìn tổng thể

| | XuetangX | OULAD |
|---|---|---|
| Node (enrollment) | 225.642 | 32.593 |
| Người học khác nhau | 77.083 (trung bình 2,9 enrollment / người) | 28.785 (1,13 enrollment / người) |
| Khóa học khác nhau | 247 course | 22 module-presentation (7 module × 4 kỳ) |
| Cửa sổ quan sát | Ngày 0–34 của khóa | Ngày 0–34 của module |
| **Tổng số cột X** | **89** | **103** |
| Khối hành vi (feature) | `[0, 59)` → **59** | `[0, 60)` → **60** |
| Khối người học (user) | `[59, 71)` → **12** | `[60, 92)` → **32** |
| Khối khóa học (course) | `[71, 89)` → **18** | `[92, 103)` → **11** |
| Kịch bản F1 / F2 / F3 / M | 59 / 71 / 77 / 89 | 60 / 92 / 71 / 103 |
| Nguồn của cách làm | Hành vi: MST-GCN; người học, khóa học: CFIN | Click theo ngày: như XuetangX; mọi cột còn lại: Wu et al. (2026) |

## 2. Khối hành vi (feature)

| Nhóm cột | XuetangX | OULAD |
|---|---|---|
| Hoạt động theo ngày | **35**: số **sự kiện** mỗi ngày 0–34 | **35**: tổng **click** (`sum_click`) mỗi ngày 0–34 |
| Tổng | **1**: total_events | **1**: total_clicks |
| Số đối tượng khác nhau | **1**: distinct_objects | — (Wu không có) |
| Theo loại hành động | **22** action, là **động từ**: video 5 (seek/play/pause/stop/load), assignment 6 (problem_get, problem_check, …), forum 5 (create_thread, …), web_page 6 (click_info, click_courseware, …) | **18** activity_type, là **loại tài liệu**: forumng 27,0%, oucontent 21,5%, homepage 18,6%, quiz 15,1%, subpage 9,6%, resource 3,1%, ouwiki 2,0%, url 1,8%, 10 loại còn lại < 0,5% (tỉ lệ trên tổng click) |
| Bài đánh giá | — (dataset không có điểm) | **6** (Wu): n_assess, submitted_any, avg_score, weighted_score, mean_lateness, max_lateness |
| **Tổng** | **59** | **60** |

Số liệu hoạt động trong 35 ngày (trước khi scale):

| | XuetangX (sự kiện) | OULAD (click) |
|---|---|---|
| Node không có hoạt động nào | 3.305 (1,5%) | 4.530 (13,9%) |
| Trung vị / trung bình / phân vị 90 | 28 / 179,7 / 459 | 165 / 284,9 / 707 |

Chi tiết 6 cột assessment của OULAD. Chỉ tính bài có hạn nộp trong ngày 0–34 **và** nộp trước ngày 35; Exam không có bài nào.

| Cột | Cách tính | Giá trị (trước scale) |
|---|---|---|
| n_assess | số bài đã nộp | 0–2; 23.045 node có ≥ 1 bài |
| submitted_any | 1 nếu n_assess > 0 | 0 / 1 |
| avg_score | trung bình score/100 (điểm trống = 0) | 0–1; trung vị 0,78 (trong số node có nộp) |
| weighted_score | Σ (score/100)·(weight/100) | 0–0,16; trung vị 0,072 |
| mean_lateness, max_lateness | ngày nộp − hạn nộp (âm = nộp sớm; bài banked ≈ −13 đến −34) | −35 đến 20; trung vị −1 |
| Node không nộp bài nào | cả 6 cột = 0 | 9.548 node |

## 3. Khối người học (user)

| Trường | XuetangX | OULAD |
|---|---|---|
| Giới tính | **Dùng**: one-hot female / male + missing (**3**). Thiếu 57,5%; male 66.416, female 29.582 | **Không dùng** (thuộc tính được bảo vệ, theo Wu). Không thiếu; M 17.875, F 14.718 |
| Học vấn | **Dùng**: one-hot 7 mức (Associate, Bachelor's, Doctorate, High, Master's, Middle, Primary) + missing (**8**). Thiếu 67,6% | **Dùng**: `highest_education` one-hot 5 mức (A Level or Equivalent, HE Qualification, Lower Than A Level, No Formal quals, Post Graduate Qualification), không có cột missing (**5**). Không thiếu |
| Tuổi | **Dùng**: số = năm bắt đầu khóa − năm sinh, ngoài [10, 70] hoặc thiếu → 0, rồi z-score (CFIN) (**1**). Năm sinh thiếu 71,2%; tuổi hợp lệ 28,5% (trung vị 24) | **Không dùng**: `age_band` (0-35 / 35-55 / 55<=) là thuộc tính được bảo vệ |
| Khu vực | — | **Dùng**: `region` one-hot 13 vùng (**13**). Không thiếu |
| Mức thiếu thốn khu vực sống | — | **Dùng**: `imd_band` one-hot 10 mức (0-10% … 90-100%) + missing (**11**). Thiếu 3,4% (1.111 node) |
| Khuyết tật | — | **Không dùng** (thuộc tính được bảo vệ). Y 3.164, N 29.429 |
| Tín chỉ đang học | — | **Dùng**: `studied_credits`, z-score (**1**). 30–655, trung vị 60 |
| Số lần đã học module này | — | **Dùng**: `num_of_prev_attempts`, z-score (**1**). 0–6, trung vị 0 |
| Ngày đăng ký | — | **Dùng**: `date_registration` (so với ngày bắt đầu), thiếu → 0, z-score (**1**). −322 đến 167, trung vị −57; thiếu 45 node |
| **Tổng** | **12** | **32** |

Điểm khác cần nhớ:
- **XuetangX thiếu rất nhiều** (57–71%), nên cột missing mang thông tin thật ("không khai báo").
- **OULAD gần như đủ**, chỉ thiếu imd_band. Bù lại, OULAD có thêm thông tin học vụ (tín chỉ, số lần học lại, ngày đăng ký) mà XuetangX không có.
- **Giới tính và tuổi ngược nhau:** XuetangX dùng; OULAD bỏ để có lý do fairness theo Wu et al.

## 4. Khối khóa học (course)

| Trường | XuetangX | OULAD |
|---|---|---|
| Lĩnh vực | **Dùng**: `category` one-hot 17 lĩnh vực + missing (**18**). Thiếu 0,4%; nhiều nhất computer 36.933, business 30.177, philosophy 29.263 | — (dataset không có) |
| Mã khóa | `course_id` **không vào X**, chỉ dùng làm Course hyperedge | **Dùng**: `code_module` one-hot 7 (AAA … GGG) (**7**). BBB 7.909 … AAA 748 |
| Kỳ học | — | **Dùng**: `code_presentation` one-hot 2013B, 2013J, 2014B, 2014J (**4**) |
| Độ dài khóa | `course_start`, `course_end` chỉ dùng để tính ngày, không vào X | `module_presentation_length` **không dùng** (Wu không dùng) |
| **Tổng** | **18** | **11** |

Điểm khác cần nhớ:
- XuetangX: one-hot lĩnh vực **gộp** 247 khóa vào 17 nhóm.
- OULAD: one-hot module × kỳ gần như **chỉ thẳng** ra từng khóa trong 22 khóa. Thông tin này trùng một phần với Course hyperedge.

## 5. Trường có trong dữ liệu gốc nhưng không vào X

| Dataset | Trường | Lý do |
|---|---|---|
| XuetangX | user_id, course_id, object_id | Chỉ dùng để dựng User / Course / Object hyperedge |
| XuetangX | course_start, course_end | Chỉ dùng để tính ngày (course_day) và tuổi |
| OULAD | id_student, code_module + code_presentation, id_site | Dùng để dựng User / Course / Object hyperedge (module và kỳ còn được one-hot) |
| OULAD | final_result | Nguồn của nhãn (Withdrawn = 1) |
| OULAD | date_unregistration | Có giá trị ⇔ Withdrawn, nên lộ nhãn |
| OULAD | gender, age_band, disability | Thuộc tính được bảo vệ, theo Wu et al. (2026); vẫn ghi trong CSV để kiểm tra fairness sau này |
| OULAD | week_from, week_to, module_presentation_length | Wu không dùng |
| OULAD | Bài đánh giá nộp từ ngày 35 trở đi, Exam | Ngoài cửa sổ quan sát |

## 6. Cách scale (mọi μ, σ chỉ tính trên train)

| Loại cột | XuetangX | OULAD |
|---|---|---|
| Số đếm hành vi | log1p + z (59 cột) | log1p + z (55 cột: 35 ngày, tổng, 18 activity, n_assess) |
| Số liên tục khác | age: z (1 cột) | avg_score, weighted_score, mean/max_lateness, studied_credits, num_of_prev_attempts, date_registration: z (7 cột) |
| Giữ nguyên 0/1 | 29 cột one-hot | 40 cột one-hot + submitted_any |
| Giá trị thiếu | One-hot: cột missing ở mọi trường; age: 0 | One-hot: cột missing chỉ ở imd_band; trường số: 0 |
| Giá trị lạ (không có trong danh sách) | Dừng chương trình (không có cột "other") | Như XuetangX |

Danh sách đầy đủ tên cột của X: `feature_names.csv` trong `data/processed/simple` (XuetangX) và
`data/processed/oulad` (OULAD). Lưu ý: file `feature_names.csv` local của XuetangX còn là bản cũ 90 cột; chạy lại
`python src/3_features.py` sẽ ra bản 89 cột.
