# Plan OULAD v3: dữ liệu, feature và kịch bản thực nghiệm

Ngày 2026-10-08. Thay cho [PLAN_OULAD_v2.md](PLAN_OULAD_v2.md). Chưa sửa code.
Mọi số liệu dưới đây được đo trên bản zip UCI (trong scratchpad) theo đúng các luật ở mục 1.

**Sửa lỗi của v2:** v2 ghi "22/22 presentation có bài đến hạn trước ngày 35". Số đúng là **22 bài, thuộc 19/22 presentation**.

## 1. Các quyết định em đã chốt

| # | Nội dung | Chốt |
|---|---|---|
| D1 | Nhãn | dropout = Withdrawn + Fail = 1; Pass + Distinction = 0 |
| D2 | Sinh viên rút trước ngày 35 | **Giữ lại**, dùng đủ 32.593 enrollment |
| D3 | Cửa sổ quan sát | Ngày 0–34, giống XuetangX (áp cho cả click và assessment) |
| D4 | Chia dữ liệu | Xáo `enroll_id` bằng seed 1: 80% đầu → train/val (80/20), 20% cuối → test |
| D5 | Hyperedge | Giống XuetangX: Course, Object, User "any", self-loop |
| D6 | Assessment | Có dùng, bỏ Exam; 8 cột tách TMA / CMA; **gộp vào khối hành vi** |
| D7 | Trường phân loại của user | One-hot; cột missing **chỉ cho imd_band** (trường duy nhất có giá trị thiếu) |
| D8 | Trường số của user | num_of_prev_attempts, studied_credits, date_registration: z-score; date_registration thiếu → 0 rồi mới z-score (như age của XuetangX) |
| D9 | Bài banked | Chỉ tính các bài banked có hạn nộp < 35 |
| D10 | Scale assessment | 6 cột đếm: log1p + z-score; 2 cột điểm: chỉ z-score |

## 2. Dữ liệu sau khi áp các luật

| Mục | Số |
|---|---|
| Node (enrollment) | 32.593 |
| Train / val / test | 20.859 / 5.215 / 6.519 |
| Tỉ lệ dropout train / val / test | 0,5243 / 0,5321 / 0,5363 (chung 0,528) |
| Log VLE trong ngày 0–34 | 2.525.357 dòng, 9.286.737 click |
| Node không có click nào trong ngày 0–34 | 4.530 (96,2% là dropout), giữ lại với hàng hành vi = 0 |
| Bài đến hạn trước ngày 35 (không tính Exam) | 22 bài (TMA 19, CMA 3), thuộc 19/22 presentation |
| Bài nộp được tính | 28.961 (TMA 21.936, CMA 7.025), gồm 560 bài banked; 23.142 node có ít nhất 1 bài |
| Bài nộp muộn | 5.882 |
| Cặp (node, bài đến hạn < 35) chưa nộp | 9.866 / 35.796 |

Lưu ý: vì giữ cả những bạn rút sớm (2.676 bạn rút trước ngày 0), tỉ lệ dropout lên 52,8% và có 4.530 node
không có click nào (96,2% trong số đó là dropout). AUC sẽ cao hơn so với khi bỏ nhóm này. Nếu so với các bài
đã bỏ nhóm này (ví dụ Oğul et al. 2026), cần nói rõ điểm khác này.

## 3. Feature X của OULAD (113 cột)

| Khối | Vị trí | Cột | Scale |
|---|---|---|---|
| Hành vi | `[0, 35)` | Tổng `sum_click` từng ngày 0–34 | log1p + z |
| | 35 | total_clicks | log1p + z |
| | 36 | distinct_sites | log1p + z |
| | `[37, 57)` | Click theo 20 activity_type | log1p + z |
| | `[57, 65)` | TMA: submitted, missed, late, mean_score · CMA: submitted, missed, late, mean_score | đếm: log1p + z; mean_score: z |
| Người học | `[65, 101)` | One-hot: gender 2, region 13, highest_education 5, imd_band 10 + missing, age_band 3, disability 2 (36 cột) | 0/1 |
| | `[101, 104)` | num_of_prev_attempts, studied_credits, date_registration | z |
| Khóa học | `[104, 113)` | One-hot: code_module 7, kỳ B/J 2 | 0/1 |

Định nghĩa 8 cột assessment, tính riêng cho TMA và cho CMA, chỉ trên các bài không phải Exam:
- **submitted**: số bài đã nộp. Gồm bài thường có `date_submitted < 35` và bài banked có hạn nộp < 35 (D9).
- **missed**: số bài có hạn nộp < 35 mà không nằm trong danh sách đã nộp ở trên.
- **late**: số bài thường đã nộp với `date_submitted >` hạn nộp. Bài banked không tính là muộn.
- **mean_score**: điểm trung bình các bài đã nộp có điểm; 0 nếu chưa nộp bài nào.

Không đưa vào X: `final_result` (nhãn), `date_unregistration` (lộ nhãn), bài Exam.

Các nhóm feature dùng trong kịch bản:

| Mã | Cột | Số cột |
|---|---|---|
| `feature` | hành vi (gồm cả assessment) | 65 |
| `feature+user` | hành vi + người học | 104 |
| `feature+course` | hành vi + khóa học | 74 |
| `full` | tất cả | 113 |

## 4. Hypergraph

| Nhóm | OULAD | XuetangX (để so) |
|---|---|---|
| Course | 22 hyperedge (365–2.498 node mỗi cái), phủ 100% | 247, phủ 100% |
| Object | 3.465 hyperedge (trên 3.903 tài liệu có người dùng), 734.714 lần tham gia, phủ 86,1% | 22.421, phủ 88,4% |
| User "any" | 3.538 hyperedge, 7.346 lần tham gia, phủ 22,5% | 64.672, phủ 94,5% |
| Self-loop | 32.593 | 225.642 |

Object: mọi activity_type đều là object (key `course|activity_type|id_site`), kể cả `homepage`.

Rò rỉ do User "any" (ghi lại để em báo thầy hướng dẫn): 1.309 node có lần học lại cùng module ở kỳ sau.
99,6% trong số đó là dropout, so với 50,8% ở các node còn lại. Riêng biến "có học lại sau không" cho AUC = 0,538.

## 5. Kịch bản thực nghiệm (11 cấu hình × 5 seed = 55 lần chạy)

Giống hệt bộ kịch bản của XuetangX ([KICH_BAN_THUC_NGHIEM.md](KICH_BAN_THUC_NGHIEM.md), bản 4). Chỉ khác X (113 cột).

M: Course + Object + User + self-loop, học W, feature `full`, HGNN 2 layer, có MLP; tối đa 1000 epoch,
early stopping theo val AUC (mỗi 5 epoch, patience 40); ngưỡng 0,5; seed 1, 11, 111, 1111, 11111.
Siêu tham số giữ nguyên `TRAIN` của XuetangX, không tune riêng.

✓ = giữ, ✗ = bỏ. Ô **in đậm** là yếu tố khác với M.

| # | Mã | Course | Object | User | Self-loop | Học W | Feature | Layer HGNN | MLP | Câu hỏi trả lời |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **M** | ✓ | ✓ | ✓ | ✓ | ✓ | full (113) | 2 | ✓ | Mô hình đầy đủ |
| 2 | A1 | **✗** | ✓ | ✓ | ✓ | ✓ | full | 2 | ✓ | Course đóng góp bao nhiêu |
| 3 | A2 | ✓ | **✗** | ✓ | ✓ | ✓ | full | 2 | ✓ | Object đóng góp bao nhiêu |
| 4 | A3 | ✓ | ✓ | **✗** | ✓ | ✓ | full | 2 | ✓ | User đóng góp bao nhiêu |
| 5 | A4 | ✓ | ✓ | ✓ | **✗** | ✓ | full | 2 | ✓ | Self-loop đóng góp bao nhiêu |
| 6 | W1 | ✓ | ✓ | ✓ | ✓ | **✗ (W = I)** | full | 2 | ✓ | Lợi ích của việc học W |
| 7 | F1 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature (65)** | 2 | ✓ | Chỉ có hành vi + assessment |
| 8 | F2 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature+user (104)** | 2 | ✓ | Hành vi + người học |
| 9 | F3 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature+course (74)** | 2 | ✓ | Hành vi + khóa học |
| 10 | L1 | ✓ | ✓ | ✓ | ✓ | ✓ | full | **1** | ✓ | HGNN 1 layer so với 2 layer |
| 11 | H | ✓ | ✓ | ✓ | ✓ | ✓ | full | 2 | **✗** | Mô hình đầy đủ khi thiếu MLP |

X1 (chỉ self-loop) vẫn là kịch bản tùy chọn, không nằm trong `all`, như XuetangX.

Dự kiến cần để ý khi đọc kết quả:
- A3: User chỉ phủ 22,5% node (XuetangX 94,5%), nên ảnh hưởng của User trên OULAD có thể nhỏ hơn.
- A1: chỉ có 22 Course hyperedge, mỗi cái rất lớn (365–2.498 node).
- Vì assessment nằm trong khối hành vi, bộ kịch bản này **không đo riêng** được đóng góp của assessment (em đã chọn như vậy).

## 6. Thay đổi code

Thêm tham số `--dataset {xuetangx,oulad}`, mặc định `xuetangx`. Các lệnh và đường dẫn của XuetangX giữ nguyên.

| File | Thay đổi |
|---|---|
| `0_config.py` | Đường dẫn OULAD (`data/raw/oulad`, `data/processed/oulad`, `outputs/oulad`); URL UCI; 20 activity_type; từ vựng one-hot; với mỗi trường ghi có cột missing hay không (chỉ imd_band có); `FEATURE_BLOCKS = {"xuetangx": (59, 71, 89), "oulad": (65, 104, 113)}` |
| `1_download.py` | `download_oulad`: tải zip UCI về `data/raw/oulad/oulad.zip`, bỏ qua nếu đã có, kiểm tra đủ 7 CSV |
| `2_preprocess.py` | `preprocess_oulad`: đọc thẳng từ zip; `enroll_id` = số thứ tự dòng trong studentInfo; nhãn D1; chia D4; ghi `{split}.csv` (mỗi dòng là 1 dòng studentVle trong ngày 0–34, có cột `clicks`, cộng các cột bối cảnh) và `{split}_assessment.csv` (mỗi dòng = 1 bài không phải Exam của node: loại, hạn nộp, ngày nộp, điểm, banked) |
| `3_features.py` | `build_split_features_oulad`: 57 cột click + 8 cột assessment + user + course; `set_one_hot` có thêm tham số cột missing; scale theo D8, D10; `feature_columns(name, dataset)` đọc điểm cắt từ `FEATURE_BLOCKS` |
| `4_hypergraph.py` | OULAD: mọi activity_type là object; User "any" dùng chung luật với XuetangX |
| `5_graph_data.py` | Truyền `dataset` vào `feature_columns` |
| `9_train.py`, `10_summary.py` | `--dataset` chọn thư mục dữ liệu và `outputs/oulad/` |
| `6_hgnn`, `7_mlp`, `8_model` | Không đổi |

## 7. Kiểm tra sau khi code

- O1. 32.593 node; train/val/test = 20.859 / 5.215 / 6.519; dropout 0,5243 / 0,5321 / 0,5363; 3 split không trùng enrollment.
- O2. Tổng `clicks` = 9.286.737 trên 2.525.357 dòng.
- O3. Assessment: 28.961 bài được tính (TMA 21.936, CMA 7.025, banked 560); missed tổng 9.866; late tổng 5.882.
- O4. X có 113 cột, hữu hạn; 63 cột đếm và 2 cột điểm của train có mean 0 / std 1; cột one-hot chỉ có 0/1.
- O5. Hypergraph: Course 22, Object 3.465, User 3.538; mỗi Object nằm trong đúng 1 Course.
- O6. Thử đảo nhãn val/test (C7): loss và trọng số khi train giống hệt (1 thread).
- O7. XuetangX: `9_train.py --scenario M --epochs 10 --seeds 1` cho cùng số trước và sau khi sửa.
- O8. Smoke: `9_train.py --dataset oulad --scenario all --epochs 10 --seeds 1` chạy hết 11 kịch bản.

## 8. Lệnh chạy trên server

```
python src/1_download.py   --dataset oulad
python src/2_preprocess.py --dataset oulad
python src/3_features.py   --dataset oulad
python src/4_hypergraph.py --dataset oulad
python src/9_train.py      --dataset oulad --scenario M --seeds 1 11 111 1111 11111
python src/9_train.py      --dataset oulad --scenario A1 A2 A3 A4 W1 F1 F2 F3 L1 H --seeds 1 11 111 1111 11111
python src/10_summary.py   --dataset oulad
```

## 9. Các điểm thầy tự đề xuất, em duyệt giúp

| # | Điểm | Thầy đề xuất | Lý do |
|---|---|---|---|
| P1 | Cột missing cho khối khóa học | Không có (code_module và kỳ B/J không thiếu ô nào) | Áp cùng luật D7 |
| P2 | 22 bài nộp có điểm trống | Vẫn tính là đã nộp, không đưa vào điểm trung bình | Đã nộp là sự kiện có thật; điểm trống không phải điểm 0 |
| P3 | Định nghĩa "nộp muộn" | `date_submitted >` hạn nộp | Theo đúng cột `date` của bảng assessments |
| P4 | mean_score khi chưa nộp bài nào | 0, rồi mới z-score | Đã ghi trong lựa chọn assessment em chọn |
