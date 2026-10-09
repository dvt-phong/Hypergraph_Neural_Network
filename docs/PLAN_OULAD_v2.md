# Plan OULAD v2: chốt quyết định + so sánh XuetangX và OULAD

Ngày 2026-10-08. Thay cho [PLAN_OULAD.md](PLAN_OULAD.md). Chưa sửa code.
Số liệu OULAD đo trên bản zip UCI (trong scratchpad). Số liệu XuetangX đo trên `data/processed/simple/hypergraph.npz`
local (bản transductive, 2026-10-07).

## 1. Quyết định đã chốt

| # | Nội dung | Chốt |
|---|---|---|
| D1 | Nhãn | **dropout = Withdrawn + Fail = 1**, Pass + Distinction = 0 (em chốt) |
| D2 | Sinh viên rút trước ngày 35 | Bỏ 5.335 Withdrawn có `date_unregistration < 35` (nhãn đã lộ trong cửa sổ) — *em xác nhận* |
| D3 | Cửa sổ quan sát | Ngày 0–34 tính từ ngày bắt đầu module, giống XuetangX |
| D4 | Chia dữ liệu | Xáo các `enroll_id` bằng seed 1; 80% đầu → train/val (80/20 như XuetangX); 20% cuối → test |
| D5 | Hyperedge | **Giữ nguyên như XuetangX**: Course, Object, User "any", self-loop (em chốt) |
| D6 | Điểm assessments | Chưa dùng, để X cùng cấu trúc với XuetangX — *em xác nhận* |

## 2. Bảng 1 — So sánh hai dataset

| Mục | XuetangX | OULAD |
|---|---|---|
| Nguồn | moocdata.cn (Feng et al., AAAI 2019, CFIN) | Kuzilek et al., Scientific Data 2017 [6]; tải từ UCI |
| Loại khóa học | MOOC mở, tự do | Module đại học từ xa (Open University, UK), có học phí, có tín chỉ |
| Node = enrollment | (user, course) | (id_student, code_module, code_presentation) |
| Số node dùng | 225.642 | 27.258 (32.593 − 5.335 theo D2) |
| Train / val / test | 126.354 / 31.589 / 67.699 | 17.444 / 4.362 / 5.452 |
| Test lấy từ đâu | Test chính thức của dataset | Tự chia (D4); dataset không có test chính thức |
| Nhãn | Có sẵn trong `train_truth` / `test_truth` | Suy từ `final_result`: Withdrawn + Fail = 1 |
| Tỉ lệ dropout train / val / test | 0,7582 / 0,7601 / 0,7580 | 0,4375 / 0,4328 / 0,4318 |
| Khóa học | 247 course | 22 module-presentation (7 module × 2013B/2013J/2014B/2014J) |
| Độ dài khóa | — | 234–269 ngày → 35 ngày ≈ 13–15% khóa |
| Đơn vị log | 1 dòng = 1 sự kiện, có thời điểm chính xác | 1 dòng = tổng click của 1 sinh viên trên 1 tài liệu trong 1 ngày (`sum_click`) |
| Quy mô log thô | 42.110.402 sự kiện | 10.655.280 dòng, 39.605.099 click |
| Log trong cửa sổ 0–34 (các node giữ lại) | — | 2.454.902 dòng, 9.045.911 click |
| Loại hành động | 22 **động từ** (play_video, problem_check, …) | 20 **loại tài liệu** (resource, oucontent, quiz, forumng, …) |
| Đối tượng (object) | video / problem / forum post | `id_site`: 6.364 tài liệu, mỗi cái thuộc đúng 1 presentation |
| Node không có hoạt động trong cửa sổ | Có (giữ lại, hàng hành vi = 0) | 1.050 node, 83,6% là dropout (giữ lại như XuetangX) |
| Thông tin người học | gender, education, birth | gender, region, highest_education, imd_band, age_band, disability, num_of_prev_attempts, studied_credits, date_registration |
| Thiếu dữ liệu người học | age 71,5%, education 67,6%, gender 57,5% | chỉ imd_band 3,7%; các cột khác 0% |
| Thông tin khóa học | category (17 lĩnh vực), 0,4% thiếu | code_module (7), kỳ B (tháng 2) / J (tháng 10) |
| Điểm số | Không có (chỉ có hành động problem_check_correct/incorrect) | Có `studentAssessment` (chưa dùng, D6) |

## 3. Bảng 2 — Feature engineering, so cột với cột

Nguyên tắc giữ nguyên ở cả hai: hành vi = log1p rồi z-score (μ, σ chỉ từ train); one-hot luôn có thêm 1 cột
"missing", không có cột "other"; số liên tục dùng z-score; node không hoạt động có hàng hành vi = 0.

| Khối | XuetangX (89 cột, đang dùng) | OULAD (112 cột, đề xuất) | Ghi chú |
|---|---|---|---|
| Hoạt động theo ngày | 35 cột: **số sự kiện** ngày 0–34 | 35 cột: **tổng `sum_click`** ngày 0–34 | OULAD chỉ có tổng theo ngày, nên đếm click thay vì đếm dòng. Junejo et al. [2] cũng gộp click theo ngày |
| Tổng | total_events | total_clicks | |
| Số đối tượng khác nhau | distinct_objects | distinct_sites (số `id_site` khác nhau) | |
| Theo loại hành động | 22 cột, mỗi động từ 1 cột | 20 cột, tổng click theo `activity_type` | Ý nghĩa khác: XuetangX đếm "làm gì", OULAD đếm "vào loại tài liệu nào". Liu et al. [9] cũng dùng click theo loại tài liệu |
| **Tổng khối hành vi** | **59** `[0, 59)` | **57** `[0, 57)` | |
| Giới tính | one-hot 2 + missing | one-hot M/F + missing (3) | |
| Học vấn | one-hot 7 + missing | `highest_education` one-hot 5 + missing (6) | |
| Tuổi | 1 cột số: năm bắt đầu − năm sinh, z-score (CFIN) | `age_band` one-hot 3 + missing (4) | OULAD chỉ cho nhóm tuổi (0-35, 35-55, 55<=), không có năm sinh |
| Chỉ có ở OULAD | — | region 13 + missing (14), imd_band 10 + missing (11), disability Y/N + missing (3); số: num_of_prev_attempts, studied_credits, date_registration (z-score) | Đều biết lúc đăng ký, không chứa kết quả |
| **Tổng khối người học** | **12** `[59, 71)` | **44** `[57, 101)` | |
| Khóa học | category one-hot 17 + missing (18) | code_module one-hot 7 + missing (8), kỳ B/J + missing (3) | |
| **Tổng khối khóa học** | **18** `[71, 89)` | **11** `[101, 112)` | |
| **Tổng X** | **89** | **112** | |
| Không đưa vào X | label | `final_result`, `date_unregistration` (lộ nhãn), assessments (D6) | |

F1 / F2 / F3 giữ nguyên ý nghĩa: OULAD F1 = 57, F2 = 101, F3 = 68 cột.

## 4. Các bài báo trên OULAD xử lý feature thế nào

| Bài | Nhãn | Thời điểm dự đoán | Feature |
|---|---|---|---|
| Oğul et al., 2026 [5] | Fail + Withdrawn | Ngày 28; cắt VLE và assessment tại ngày 28 | Demographic + click VLE + assessment đến hạn trước ngày 28. 27.522 enrollment, 44,1% dương; bỏ điểm vẫn đạt AUC 0,762 (có điểm 0,790) |
| Alhazbi, 2026 [1] | Pass vs At-risk | Theo từng quý của khóa | Chỉ số hành vi tự thiết kế (độ đều đặn, nộp bài sớm/muộn) + demographic |
| Omarbekova et al., 2026 [8] | 4 lớp | 120 ngày đầu | 85 feature: demographic + assessment (bỏ điểm thi cuối) + hành vi |
| Junejo et al., 2024 [2] | 4 lớp | Nhiều mốc trong khóa | Tổng click theo ngày + demographic + assessment |
| Liu et al., 2022 [9] | Pass/Fail | Cả khóa | Click theo 12 loại tài liệu, gộp theo tuần / tháng |
| Aljohani et al., 2019 [3] | Pass/Fail | Theo tuần (10 tuần đầu) | Chuỗi click theo tuần đưa vào LSTM |

So với các bài trên, plan của mình:
- Giống [5] ở nhãn (Fail + Withdrawn), ở việc cắt dữ liệu tại thời điểm dự đoán, và ở quy mô
  (27.258 node, 43,6% so với 27.522 và 44,1% của họ). Đây là điểm đối chiếu tốt nhất để em double check.
- Hành vi theo ngày giống [2]; click theo loại tài liệu giống [9].
- Khác: phần lớn các bài dùng thêm điểm assessment, mình chưa dùng (D6). [5] cho thấy bỏ điểm làm AUC giảm
  khoảng 0,03 nhưng mô hình vẫn dùng được.
- Khác: [5] chia test **không trùng sinh viên** với train. Mình chia ngẫu nhiên theo enrollment như XuetangX.

## 5. Bảng 3 — Hypergraph (giữ nguyên các nhóm hyperedge của XuetangX)

| Nhóm | Luật | XuetangX | OULAD |
|---|---|---|---|
| Course | cùng course / module-presentation | 247 hyperedge; phủ 100% node | 22 hyperedge (345–1.967 node mỗi cái); phủ 100% |
| Object | cùng dùng 1 object trong ngày 0–34, \|e\| ≥ 2 | 22.421 hyperedge; 2.783.160 lần tham gia; phủ 88,4% | 3.461 hyperedge (trên 3.899 tài liệu có người dùng); 707.440 lần tham gia; phủ 96,1% |
| User | "any": mọi enrollment của cùng người, \|e\| ≥ 2 | 64.672 hyperedge; 213.231 lần tham gia; phủ 94,5% | 2.478 hyperedge; 5.069 lần tham gia; **phủ 18,6%** |
| Self-loop | mỗi node 1 cái | 225.642 | 27.258 |

Khác biệt cần nhớ:
- **Object:** XuetangX chỉ lấy hành động có object (video, problem, forum), bỏ nhóm web_page. Trên OULAD
  mọi click đều vào một `id_site`, nên mọi activity_type đều thành Object, kể cả `homepage` (22 tài liệu,
  mỗi presentation 1 cái). Hyperedge homepage gần trùng với Course.
- **User** trên OULAD phủ ít (18,6% so với 94,5%) vì đa số sinh viên chỉ học 1 module. Dự kiến A3 (bỏ User)
  sẽ ít ảnh hưởng hơn trên XuetangX.

**Lưu ý khi dùng User "any" với nhãn Withdrawn + Fail** (để em báo thầy hướng dẫn):
829 node có một lần học lại cùng module ở kỳ sau, và **100%** trong số đó là dropout (các node còn lại
41,8%). Có 167 node như vậy trong test. Nếu chỉ dùng một biến "có học lại sau không" thì AUC = 0,535,
vì nhóm này chỉ chiếm 3% số node. Vậy rò rỉ có thật nhưng nhỏ về độ lớn, và đi đúng qua User hyperedge.

## 6. Thay đổi code so với v1

Như mục 5 của [PLAN_OULAD.md](PLAN_OULAD.md), chỉ khác hai điểm:
- `2_preprocess.py`: `label = 1` nếu `final_result` ∈ {Withdrawn, Fail}.
- `4_hypergraph.py`: User dùng chung luật "any" của XuetangX (key `user|id_student`), không cần luật User
  riêng theo dataset trong config. Object: mọi activity_type đều là object (key `course|activity_type|id_site`).

## 7. Kiểm tra (cập nhật số)

- O1. 27.258 node; train/val/test = 17.444 / 4.362 / 5.452; dropout 0,4375 / 0,4328 / 0,4318; không trùng enrollment.
- O2. Tổng `clicks` trong 3 CSV = 9.045.911; số dòng có click = 2.454.902.
- O3. X có 112 cột, hữu hạn; các cột hành vi của train có mean 0 / std 1.
- O4. Hypergraph: Course 22, Object 3.461, User 2.478; mỗi Object nằm trong đúng 1 Course.
- O5–O7. Giống v1: thử đảo nhãn val/test (C7), XuetangX chạy ra số không đổi, smoke 10 epoch cả 11 kịch bản.

## Tài liệu tham khảo

[1] [Pedagogically‐Informed Behavioural Learning Analytics: An Expert Approach to Predicting at‐Risk Students](https://consensus.app/papers/details/cec6211669f150a7a982de4536082a0a/) (Alhazbi, 2026, Expert Systems)
[2] [Accurate multi-category student performance forecasting at early stages of online education using neural networks](https://consensus.app/papers/details/724acbc863aa581cbad5f01926041dd6/) (Junejo et al., 2024, Scientific Reports)
[3] [Predicting At-Risk Students Using Clickstream Data in the Virtual Learning Environment](https://consensus.app/papers/details/bb71406f8ab65a95b7ee0f680a5569be/) (Aljohani et al., 2019, Sustainability)
[5] [Early identification of at-risk students in online learning environments: A learning analytics approach using machine learning models](https://consensus.app/papers/details/c2581ab9821557358b3a16aa652e9ad8/) (Oğul et al., 2026, OPUS Journal of Society Research)
[6] [Open University Learning Analytics dataset](https://consensus.app/papers/details/8f0b0dadf5d35b7399dddbdf3fec5763/) (Kuzilek et al., 2017, Scientific Data)
[8] [Comparative Evaluation of Machine Learning Models for Early Prediction of Student Academic Performance Using the Open University Learning Analytics Dataset](https://consensus.app/papers/details/171a6da103ee58f6a16fe2597eb919ec/) (Omarbekova et al., 2026, iJIM)
[9] [Predicting Student Performance Using Clickstream Data and Machine Learning](https://consensus.app/papers/details/2081a03543cb510794a6b084e6fda98e/) (Liu et al., 2022, Education Sciences)

Thầy chỉ đọc phần tóm tắt (abstract) của các bài này, chưa đọc toàn văn.
