# Plan OULAD v4: dữ liệu, feature (đủ mọi trường) và kịch bản

Ngày 2026-10-08. Thay cho [PLAN_OULAD_v3.md](PLAN_OULAD_v3.md). Chưa sửa code.
Mọi số liệu đo trên bản zip UCI (trong scratchpad) theo đúng các luật ở mục 1.

**Khác v3:** có Exam; không bỏ trường nào (trừ 2 trường lộ nhãn); thêm weight, week_from/week_to, is_banked,
năm của presentation, module_presentation_length; luật thiếu dữ liệu thống nhất (mục 1, nguyên tắc chung).
Sửa lỗi câu hỏi trước: 11 bài Exam thiếu hạn thi thuộc **9** presentation, không phải 11.

## 1. Quyết định đã chốt

**Nguyên tắc chung:**
1. Mọi giá trị thiếu → 0, trừ trường one-hot. Trường one-hot nào có giá trị thiếu thì thêm cột missing.
2. Không bỏ trường nào. Chỉ `final_result` (nguồn của nhãn) và `date_unregistration` (có giá trị ⇔ Withdrawn)
   không vào X; cả hai vẫn được ghi trong CSV bước 2.
3. Mọi thứ chỉ lấy trong ngày 0–34, giống XuetangX.

| # | Nội dung | Chốt |
|---|---|---|
| D1 | Nhãn | Withdrawn + Fail = 1; Pass + Distinction = 0 |
| D2 | Sinh viên rút sớm | Giữ lại, đủ 32.593 enrollment |
| D3 | Cửa sổ | Ngày 0–34 cho click, assessment và Exam |
| D4 | Chia dữ liệu | Xáo `enroll_id` bằng seed 1: 80% đầu → train/val (80/20), 20% cuối → test |
| D5 | Hyperedge | Giống XuetangX: Course, Object, User "any", self-loop |
| D6 | Assessment | TMA, CMA **và Exam**, mỗi loại 6 cột; gộp vào khối hành vi |
| D7 | Trường phân loại | One-hot; chỉ imd_band có cột missing (trường duy nhất bị thiếu) |
| D8 | Trường số | z-score; thiếu → 0 trước khi z-score |
| D9 | Bài banked | Chỉ tính bài banked có hạn nộp < 35; thêm cột đếm banked |
| D10 | Scale | Cột đếm và tổng weight: log1p + z; mean_score: z |
| D11 | Hạn thi Exam thiếu (11 bài) | → 0 theo nguyên tắc 1, nên bài đó được tính là đến hạn trước ngày 35 |
| D12 | weight | Thêm cột tổng weight của các bài đã nộp, cho từng loại |
| D13 | week_from / week_to | 2 cột: click đúng lịch / click trước lịch |
| D14 | code_presentation | One-hot 4 giá trị (2013B, 2013J, 2014B, 2014J) |

## 2. Mọi trường của 7 bảng đi đâu

| Bảng | Trường | Dùng ở |
|---|---|---|
| courses | code_module | one-hot (7) + khóa của Course hyperedge |
| | code_presentation | one-hot (4) + khóa của Course hyperedge |
| | module_presentation_length | z-score (1) |
| studentInfo | id_student | khóa của User hyperedge |
| | gender, region, highest_education, imd_band, age_band, disability | one-hot (36, gồm 1 cột missing của imd_band) |
| | num_of_prev_attempts, studied_credits | z-score (2) |
| | final_result | **chỉ làm nhãn**, không vào X |
| studentRegistration | date_registration | z-score (1); 45 ô thiếu → 0 |
| | date_unregistration | **chỉ ghi trong CSV**, không vào X (lộ nhãn) |
| vle | id_site | Object hyperedge + distinct_sites |
| | activity_type | 20 cột click + nhóm của Object |
| | week_from, week_to | 2 cột click đúng / trước lịch |
| studentVle | date | 35 cột theo ngày + lọc cửa sổ |
| | sum_click | mọi cột click |
| assessments | assessment_type | tách TMA / CMA / Exam |
| | date (hạn nộp) | cột missed, late; thiếu → 0 |
| | weight | cột submitted_weight |
| | id_assessment | khóa nối bảng |
| studentAssessment | date_submitted | lọc cửa sổ + cột late |
| | is_banked | luật đếm + cột banked |
| | score | cột mean_score; điểm thiếu → 0 |

Các trường code_module, code_presentation, id_student còn xuất hiện ở các bảng khác chỉ để nối bảng.

## 3. Dữ liệu sau khi áp các luật

| Mục | Số |
|---|---|
| Node | 32.593; train / val / test = 20.859 / 5.215 / 6.519 |
| Tỉ lệ dropout train / val / test | 0,5243 / 0,5321 / 0,5363 |
| Click trong ngày 0–34 | 2.525.357 dòng, 9.286.737 click; 4.530 node không có click nào |
| Click theo lịch tài liệu | đúng lịch 1.121.265 · trước lịch 259.209 · tài liệu không có tuần 7.906.263 |
| Bài nộp được tính | TMA 21.936, CMA 7.025, Exam 0 (mọi bài thi nộp vào ngày 229–285) |
| Bài banked được tính | TMA 507, CMA 53, Exam 0 |
| Bài nộp có điểm trống | 22 → điểm 0 |
| Node có exam_missed > 0 (do D11) | 14.894 node (= 1: 10.460; = 2: 4.434), thuộc AAA, BBB, CCC và DDD 2014J |

## 4. Feature X (128 cột)

| Khối | Vị trí | Cột | Scale |
|---|---|---|---|
| Hành vi | `[0, 35)` | tổng click từng ngày 0–34 | log1p + z |
| | 35, 36 | total_clicks, distinct_sites | log1p + z |
| | `[37, 57)` | click theo 20 activity_type | log1p + z |
| | `[57, 59)` | clicks_on_schedule, clicks_ahead | log1p + z |
| | `[59, 77)` | Với từng loại TMA, CMA, Exam: submitted, missed, late, banked, submitted_weight, mean_score | mean_score: z; còn lại: log1p + z |
| Người học | `[77, 113)` | one-hot gender 2, region 13, highest_education 5, imd_band 10 + missing, age_band 3, disability 2 | 0/1 |
| | `[113, 116)` | num_of_prev_attempts, studied_credits, date_registration | z |
| Khóa học | `[116, 127)` | one-hot code_module 7, code_presentation 4 | 0/1 |
| | 127 | module_presentation_length | z |

Định nghĩa các cột mới (tính cho từng loại t ∈ {TMA, CMA, Exam}):
- **submitted_t**: số bài đã nộp, gồm bài thường có `date_submitted < 35` và bài banked có hạn nộp < 35.
- **missed_t**: số bài có hạn nộp < 35 (hạn thiếu = 0) mà chưa nộp.
- **late_t**: số bài thường đã nộp với `date_submitted >` hạn nộp.
- **banked_t**: số bài banked đã tính vào submitted_t.
- **submitted_weight_t**: tổng weight của các bài đã nộp.
- **mean_score_t**: điểm trung bình các bài đã nộp (điểm trống = 0); chưa nộp bài nào = 0.
- **clicks_on_schedule**: click vào tài liệu có tuần dự kiến nằm trong tuần 0–4 (`week_to ≤ 4`).
- **clicks_ahead**: click vào tài liệu dự kiến từ tuần 5 trở đi (`week_from ≥ 5`). Đã kiểm tra: không tài
  liệu nào kéo dài qua ranh tuần 4/5, nên hai cột không chồng nhau. Tài liệu không có tuần không tính vào cột nào.

Nhóm feature dùng trong kịch bản:

| Mã | Cột | Số cột |
|---|---|---|
| `feature` | hành vi (gồm click, lịch tài liệu, assessment, Exam) | 77 |
| `feature+user` | hành vi + người học | 116 |
| `feature+course` | hành vi + khóa học | 89 |
| `full` | tất cả | 128 |

## 5. Hypergraph

| Nhóm | OULAD | XuetangX (để so) |
|---|---|---|
| Course | 22 (365–2.498 node mỗi cái), phủ 100% | 247, phủ 100% |
| Object | 3.465 (trên 3.903 tài liệu có người dùng), phủ 86,1% | 22.421, phủ 88,4% |
| User "any" | 3.538, phủ 22,5% | 64.672, phủ 94,5% |
| Self-loop | 32.593 | 225.642 |

Rò rỉ do User "any" (để em báo thầy hướng dẫn): 1.309 node có lần học lại ở kỳ sau, 99,6% trong số đó là
dropout (các node còn lại 50,8%).

## 6. Kịch bản (11 cấu hình × 5 seed = 55 lần chạy)

Giống bộ kịch bản XuetangX ([KICH_BAN_THUC_NGHIEM.md](KICH_BAN_THUC_NGHIEM.md), bản 4). Siêu tham số giữ nguyên
`TRAIN`: tối đa 1000 epoch, early stopping theo val AUC (mỗi 5 epoch, patience 40), ngưỡng 0,5, seed 1, 11, 111, 1111, 11111.

| # | Mã | Course | Object | User | Self-loop | Học W | Feature | Layer HGNN | MLP |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **M** | ✓ | ✓ | ✓ | ✓ | ✓ | full (128) | 2 | ✓ |
| 2 | A1 | **✗** | ✓ | ✓ | ✓ | ✓ | full | 2 | ✓ |
| 3 | A2 | ✓ | **✗** | ✓ | ✓ | ✓ | full | 2 | ✓ |
| 4 | A3 | ✓ | ✓ | **✗** | ✓ | ✓ | full | 2 | ✓ |
| 5 | A4 | ✓ | ✓ | ✓ | **✗** | ✓ | full | 2 | ✓ |
| 6 | W1 | ✓ | ✓ | ✓ | ✓ | **✗ (W = I)** | full | 2 | ✓ |
| 7 | F1 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature (77)** | 2 | ✓ |
| 8 | F2 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature+user (116)** | 2 | ✓ |
| 9 | F3 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature+course (89)** | 2 | ✓ |
| 10 | L1 | ✓ | ✓ | ✓ | ✓ | ✓ | full | **1** | ✓ |
| 11 | H | ✓ | ✓ | ✓ | ✓ | ✓ | full | 2 | **✗** |

X1 (chỉ self-loop) vẫn là tùy chọn, không thuộc `all`.

Cần để ý khi đọc kết quả:
- 5 cột Exam (submitted, late, banked, submitted_weight, mean_score) bằng 0 ở mọi node, nên sau z-score
  vẫn bằng 0 và không ảnh hưởng mô hình. Chúng có mặt để đủ trường.
- exam_missed khác 0 ở 14.894 node, nhưng chỉ phụ thuộc vào presentation (D11), nên nó đóng vai như thêm
  một cột "khóa học" chứ không mô tả hành vi của sinh viên.
- Vì giữ cả các bạn rút sớm, AUC sẽ cao hơn so với các bài báo đã bỏ nhóm này.

## 7. Thay đổi code

Thêm `--dataset {xuetangx,oulad}`, mặc định `xuetangx`. Các lệnh và đường dẫn của XuetangX giữ nguyên.

| File | Thay đổi |
|---|---|
| `0_config.py` | Đường dẫn OULAD; URL UCI; 20 activity_type; 3 loại assessment; từ vựng one-hot; với mỗi trường one-hot ghi có cột missing hay không; `FEATURE_BLOCKS = {"xuetangx": (59, 71, 89), "oulad": (77, 116, 128)}` |
| `1_download.py` | `download_oulad`: tải zip về `data/raw/oulad/oulad.zip`, bỏ qua nếu đã có, kiểm tra đủ 7 CSV |
| `2_preprocess.py` | `preprocess_oulad`: đọc thẳng từ zip; nhãn D1; chia D4; ghi `{split}.csv` (dòng studentVle trong ngày 0–34 + `clicks`, `week_from`, `week_to` + mọi trường bối cảnh, kể cả `final_result`, `date_unregistration`) và `{split}_assessment.csv` (mỗi dòng = 1 bài của node: loại, hạn nộp, weight, ngày nộp, banked, điểm) |
| `3_features.py` | `build_split_features_oulad` theo mục 4; `set_one_hot` có tham số cột missing; `feature_columns(name, dataset)` đọc `FEATURE_BLOCKS` |
| `4_hypergraph.py` | OULAD: mọi activity_type là object; User "any" như XuetangX |
| `5_graph_data.py` | Truyền `dataset` vào `feature_columns` |
| `9_train.py`, `10_summary.py` | `--dataset` chọn thư mục dữ liệu và `outputs/oulad/` |
| `6_hgnn`, `7_mlp`, `8_model` | Không đổi |

## 8. Kiểm tra sau khi code

- O1. 32.593 node; 20.859 / 5.215 / 6.519; dropout 0,5243 / 0,5321 / 0,5363; không trùng enrollment.
- O2. Click: tổng 9.286.737; đúng lịch 1.121.265; trước lịch 259.209.
- O3. Assessment: submitted TMA 21.936, CMA 7.025, Exam 0; banked 507 / 53 / 0; 14.894 node có exam_missed > 0.
- O4. X có 128 cột, hữu hạn, không có giá trị thiếu; cột scale của train có mean 0 / std 1 (cột hằng số → 0); one-hot chỉ có 0/1.
- O5. Hypergraph: Course 22, Object 3.465, User 3.538; mỗi Object nằm trong đúng 1 Course.
- O6. Thử đảo nhãn val/test (C7): loss và trọng số khi train giống hệt (1 thread).
- O7. XuetangX: `9_train.py --scenario M --epochs 10 --seeds 1` cho cùng số trước và sau khi sửa.
- O8. Smoke: `9_train.py --dataset oulad --scenario all --epochs 10 --seeds 1` chạy hết 11 kịch bản.

## 9. Lệnh chạy trên server

```
python src/1_download.py   --dataset oulad
python src/2_preprocess.py --dataset oulad
python src/3_features.py   --dataset oulad
python src/4_hypergraph.py --dataset oulad
python src/9_train.py      --dataset oulad --scenario M --seeds 1 11 111 1111 11111
python src/9_train.py      --dataset oulad --scenario A1 A2 A3 A4 W1 F1 F2 F3 L1 H --seeds 1 11 111 1111 11111
python src/10_summary.py   --dataset oulad
```

## 10. Điểm thầy tự định nghĩa, em duyệt giúp

| # | Điểm | Thầy đề xuất |
|---|---|---|
| P1 | Nộp muộn | `date_submitted >` hạn nộp; bài banked không tính là muộn |
| P2 | Đúng lịch / trước lịch | Đúng lịch: `week_to ≤ 4`. Trước lịch: `week_from ≥ 5`. Tuần 0 = ngày 0–6, nên tuần 0–4 = ngày 0–34 |
| P3 | Tách khối khóa học | module_presentation_length vào khối khóa học (cùng bảng courses) |
