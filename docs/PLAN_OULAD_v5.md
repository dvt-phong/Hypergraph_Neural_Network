# Plan OULAD v5: kết hợp Wu et al. (2026) với cách làm của mình

Ngày 2026-10-08. Thay cho [PLAN_OULAD_v4.md](PLAN_OULAD_v4.md). Chưa sửa code.

Nguồn của Wu et al. (2026): code trên Zenodo `run_oulad_revision_pipeline.py`, bài "Calibrated early-warning models
with fairness auditing and selective prediction for course withdrawal risk: evidence from OULAD". Thầy đã đọc code
và chạy lại phần feature + mô hình chính của họ, nhưng chưa đọc toàn văn bài báo.
Trong plan này: **[Wu]** = làm đúng như code của Wu; **[Mình]** = theo cách mình đã làm với XuetangX hoặc em đã chốt.
Mọi số liệu đo trên bản zip UCI (trong scratchpad).

## 1. Quyết định đã chốt

| # | Nội dung | Chốt | Nguồn |
|---|---|---|---|
| D1 | Nhãn | Withdrawn + Fail = 1; Pass + Distinction = 0 | [Mình] (Wu: chỉ Withdrawn) |
| D2 | Ai được giữ | Đủ 32.593 enrollment, không loại ai | [Wu] |
| D3 | Cửa sổ | Ngày 0–34, bỏ hoạt động trước ngày 0 | [Mình] (Wu: 0–27) |
| D4 | Chia dữ liệu | Xáo `enroll_id` bằng seed 1: 80% đầu → train/val (80/20), 20% cuối → test | [Mình] (Wu: theo presentation) |
| D5 | Hyperedge | Course, Object, User "any", self-loop | [Mình] |
| D6 | Click | 35 cột theo ngày (thay cho cột theo tuần của Wu) | [Mình] |
| D7 | Các cột còn lại | Đúng các cột của Wu: tổng click, click theo activity_type, 6 cột assessment, 8 trường tĩnh | [Wu] |
| D8 | Lọc assessment | Tập bài theo Wu (hạn nộp trong ngày 0–34), **nhưng chỉ tính bài nộp trước ngày 35** | [Wu] + sửa lỗ hổng 1 |
| D9 | Độ trễ của bài banked | Để nguyên như Wu: ngày nộp −1 − hạn nộp | [Wu] |
| D10 | Thuộc tính được bảo vệ | Bỏ gender, age_band, disability khỏi X | [Wu] |
| D11 | Giá trị thiếu | One-hot: cột missing chỉ cho imd_band. Trường số: 0 | [Mình] (Wu: mode / trung vị) |
| D12 | Chuẩn hóa | Cột đếm: log1p + z; cột số khác: z; fit trên train | [Mình] (Wu: chỉ scale cho LR / SVM / KNN) |

**Khác v4:** bỏ distinct_sites, 2 cột đúng lịch / trước lịch (`week_from`, `week_to`), 18 cột assessment tách theo
loại, gender, age_band, disability, `module_presentation_length`. Thay vào đó là 6 cột assessment của Wu.

## 2. Mọi trường của 7 bảng đi đâu

| Bảng | Trường | Dùng ở |
|---|---|---|
| studentInfo | code_module, code_presentation | one-hot (khối khóa học) + khóa của Course hyperedge [Wu] |
| | region, highest_education, imd_band | one-hot (khối người học) [Wu] |
| | studied_credits, num_of_prev_attempts | số (khối người học) [Wu] |
| | gender, age_band, disability | **không vào X** (thuộc tính được bảo vệ) [Wu]; vẫn ghi trong CSV |
| | id_student | khóa của User hyperedge |
| | final_result | chỉ làm nhãn |
| studentRegistration | date_registration | số (khối người học) [Wu]; 45 ô thiếu → 0 |
| | date_unregistration | không vào X (lộ nhãn); vẫn ghi trong CSV |
| studentVle + vle | date, sum_click | 35 cột theo ngày + total_clicks |
| | activity_type | 18 cột click theo loại + nhóm của Object |
| | id_site | Object hyperedge |
| | week_from, week_to | không dùng (Wu không dùng) |
| assessments + studentAssessment | date, date_submitted, score, weight, is_banked | 6 cột assessment (mục 4) |
| courses | module_presentation_length | không dùng (Wu không dùng) |

## 3. Dữ liệu

| Mục | Số |
|---|---|
| Node | 32.593; train / val / test = 20.859 / 5.215 / 6.519 |
| Tỉ lệ dropout train / val / test | 0,5243 / 0,5321 / 0,5363 |
| Click trong ngày 0–34 | 2.525.357 dòng, 9.286.737 click; 4.530 node không có click nào |
| activity_type có click trong ngày 0–34 | 18 (không có `folder`, `repeatactivity`) |
| Bài có hạn nộp trong ngày 0–34 (có bản ghi nộp) | 26.696 bản ghi |
| Sau khi chỉ tính bài nộp trước ngày 35 (D8) | 25.930 (TMA 21.659, CMA 4.271); bỏ 766 bản ghi nộp sau ngày 35 |
| Trong đó bài banked | 560, độ trễ từ −34 đến −13 (do ngày nộp giả −1) |
| Bản ghi có điểm trống | 19 → điểm 0 (D11) |
| Node có `submitted_any = 1` | 23.045 (`n_assess` = 1: 20.160 node; = 2: 2.885 node) |

Exam: Wu bỏ những Exam thiếu hạn thi. Theo luật của mình, hạn thiếu → 0 thì 11 Exam này rơi vào cửa sổ. Nhưng vì
mọi cột assessment đều tính từ **bài đã nộp trước ngày 35**, và không ai nộp bài thi trước ngày 229, nên hai cách cho
**cùng một X**. Exam không ảnh hưởng gì.

## 4. Feature X (103 cột)

| Khối | Vị trí | Cột | Scale | Nguồn |
|---|---|---|---|---|
| Hành vi | `[0, 35)` | tổng `sum_click` từng ngày 0–34 | log1p + z | [Mình] |
| | 35 | total_clicks | log1p + z | [Wu] |
| | `[36, 54)` | click theo 18 activity_type | log1p + z | [Wu] |
| | 54 | n_assess | log1p + z | [Wu] |
| | 55 | submitted_any | xem P1 | [Wu] |
| | 56 | avg_score | z | [Wu] |
| | 57 | weighted_score | z | [Wu] |
| | 58, 59 | mean_lateness, max_lateness | z | [Wu] |
| Người học | `[60, 89)` | one-hot region 13, highest_education 5, imd_band 10 + missing | 0/1 | [Wu] + [Mình] |
| | `[89, 92)` | studied_credits, num_of_prev_attempts, date_registration | z (xem P2) | [Wu] |
| Khóa học | `[92, 103)` | one-hot code_module 7, code_presentation 4 | 0/1 | [Wu] |

Định nghĩa 6 cột assessment, theo code của Wu, chỉ áp lên bài có hạn nộp trong ngày 0–34 và nộp trước ngày 35:
- **n_assess**: số bài đã nộp. Lưu ý: đây không phải số bài đến hạn.
- **submitted_any**: 1 nếu n_assess > 0, ngược lại 0.
- **avg_score**: trung bình score / 100.
- **weighted_score**: Σ (score / 100) · (weight / 100).
- **mean_lateness, max_lateness**: trung bình và lớn nhất của (ngày nộp − hạn nộp). Giá trị âm = nộp sớm.
- Node không có bài nào: cả 6 cột = 0.

Nhóm feature dùng trong kịch bản:

| Mã | Cột | Số cột |
|---|---|---|
| `feature` | hành vi | 60 |
| `feature+user` | hành vi + người học | 92 |
| `feature+course` | hành vi + khóa học | 71 |
| `full` | tất cả | 103 |

## 5. Hypergraph (không đổi so với v4)

| Nhóm | OULAD | XuetangX |
|---|---|---|
| Course | 22, phủ 100% | 247, phủ 100% |
| Object | 3.465, phủ 86,1% | 22.421, phủ 88,4% |
| User "any" | 3.538, phủ 22,5% | 64.672, phủ 94,5% |
| Self-loop | 32.593 | 225.642 |

Rò rỉ do User "any": 1.309 node có lần học lại ở kỳ sau, 99,6% trong số đó là dropout.

## 6. Kịch bản (11 cấu hình × 5 seed = 55 lần chạy)

Giống bộ kịch bản của XuetangX. Siêu tham số giữ nguyên `TRAIN`.

| # | Mã | Course | Object | User | Self-loop | Học W | Feature | Layer HGNN | MLP |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **M** | ✓ | ✓ | ✓ | ✓ | ✓ | full (103) | 2 | ✓ |
| 2 | A1 | **✗** | ✓ | ✓ | ✓ | ✓ | full | 2 | ✓ |
| 3 | A2 | ✓ | **✗** | ✓ | ✓ | ✓ | full | 2 | ✓ |
| 4 | A3 | ✓ | ✓ | **✗** | ✓ | ✓ | full | 2 | ✓ |
| 5 | A4 | ✓ | ✓ | ✓ | **✗** | ✓ | full | 2 | ✓ |
| 6 | W1 | ✓ | ✓ | ✓ | ✓ | **✗ (W = I)** | full | 2 | ✓ |
| 7 | F1 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature (60)** | 2 | ✓ |
| 8 | F2 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature+user (92)** | 2 | ✓ |
| 9 | F3 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature+course (71)** | 2 | ✓ |
| 10 | L1 | ✓ | ✓ | ✓ | ✓ | ✓ | full | **1** | ✓ |
| 11 | H | ✓ | ✓ | ✓ | ✓ | ✓ | full | 2 | **✗** |

X1 (chỉ self-loop) vẫn là tùy chọn.

## 7. Thay đổi code

`--dataset {xuetangx,oulad}`, mặc định `xuetangx`; các lệnh và đường dẫn của XuetangX giữ nguyên.

| File | Thay đổi |
|---|---|
| `0_config.py` | Đường dẫn OULAD; URL UCI; 18 activity_type; từ vựng one-hot (region, highest_education, imd_band, code_module, code_presentation); trường nào có cột missing (chỉ imd_band); `FEATURE_BLOCKS = {"xuetangx": (59, 71, 89), "oulad": (60, 92, 103)}` |
| `1_download.py` | `download_oulad`: tải zip UCI về `data/raw/oulad/oulad.zip`, bỏ qua nếu đã có, kiểm tra đủ 7 CSV |
| `2_preprocess.py` | `preprocess_oulad`: đọc thẳng từ zip (`"?"` → thiếu); nhãn D1; chia D4; ghi `{split}.csv` (dòng studentVle trong ngày 0–34 + mọi trường bối cảnh, kể cả các trường không vào X) và `{split}_assessment.csv` (các bản ghi nộp bài theo D8: hạn nộp, weight, ngày nộp, banked, điểm) |
| `3_features.py` | `build_split_features_oulad` theo mục 4; `set_one_hot` có tham số cột missing; `feature_columns(name, dataset)` đọc `FEATURE_BLOCKS` |
| `4_hypergraph.py` | OULAD: mọi activity_type là object; User "any" như XuetangX |
| `5_graph_data.py` | Truyền `dataset` vào `feature_columns` |
| `9_train.py`, `10_summary.py` | `--dataset` chọn thư mục dữ liệu và `outputs/oulad/` |
| `6_hgnn`, `7_mlp`, `8_model` | Không đổi |

## 8. Kiểm tra sau khi code

- O1. 32.593 node; 20.859 / 5.215 / 6.519; dropout 0,5243 / 0,5321 / 0,5363; 3 split không trùng enrollment.
- O2. Tổng click = 9.286.737; có 18 cột activity_type.
- O3. Assessment: 25.930 bản ghi được tính, 560 banked; 23.045 node có submitted_any = 1.
- O4. Đối chiếu với Wu: chạy hàm `build_assessment_features` của Wu (chỉ đưa vào bản ghi nộp trước ngày 35, cửa sổ 35)
  → 6 cột assessment trước khi scale phải khớp từng giá trị với cột của mình (trừ 19 bản ghi điểm trống, do D11).
- O5. X có 103 cột, hữu hạn, không có giá trị thiếu; cột scale của train có mean 0 / std 1; one-hot chỉ có 0/1.
- O6. Hypergraph: Course 22, Object 3.465, User 3.538.
- O7. Thử đảo nhãn val/test (C7): loss và trọng số khi train giống hệt (1 thread).
- O8. XuetangX: `9_train.py --scenario M --epochs 10 --seeds 1` cho cùng số trước và sau khi sửa.
- O9. Smoke: `9_train.py --dataset oulad --scenario all --epochs 10 --seeds 1` chạy hết 11 kịch bản.

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

## 10. Điểm còn chưa rõ, em chốt giúp

| # | Điểm | Thầy đề xuất | Lựa chọn khác |
|---|---|---|---|
| P1 | `submitted_any` là cờ 0/1, không phải cột đếm | Giữ 0/1, không scale (như one-hot) | z-score (Wu coi nó là cột số, scale cho LR / SVM / KNN) |
| P2 | `num_of_prev_attempts` là số đếm (0–6) | z-score như v4 | log1p + z (theo luật "mọi cột đếm") |
| P3 | 19 bản ghi có điểm trống | → 0 rồi tính vào avg_score (theo D11) | Bỏ qua khi tính trung bình, như pandas trong code của Wu |
