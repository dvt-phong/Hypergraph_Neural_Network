# Kế hoạch chi tiết từng baseline (XuetangX)

Ngày lập: 2026-10-10. **Chưa code.** Bản này nối tiếp `PLAN_BASELINE_v2.md` (7 quy tắc R1–R7).

**Đã chốt với em:** mọi baseline nhận **cùng một bộ dữ liệu thô**. Từ dữ liệu thô đó, mỗi bài **tự xử lý bằng code gốc của nó** (trích feature, dựng đồ thị, chuẩn hóa). Mình chỉ can thiệp ở 3 chỗ: **split, cách chọn epoch, và lỗi/rò rỉ nhãn**.

Mọi số dòng trong bài này thầy đọc trực tiếp từ code đang có trong `baseline/` hôm nay.

---

## 0. Mặc định thầy chọn cho các câu hỏi còn mở (em sửa nếu không đồng ý)

| Câu | Mặc định | Lý do |
|---|---|---|
| Q1. CFIN log1p? | **Không.** Dùng z-score gốc | Em chọn nguyên tắc "cách xử lý là của nó". log1p là cách xử lý của mình, không phải của CFIN. Bản log1p cũ (0.8569) chỉ để tham khảo |
| Q2. Sửa lỗi gender/category trong CFIN, CA-TFHN? | **Sửa** | Lỗi đọc dữ liệu, không phải phương pháp (R7). Chi tiết ở mục 4 |
| Q3. SVM | **LinearSVC** | SVM kernel RBF trên 126k dòng không chạy nổi |
| Q4. HGB | **Không thêm** | Theo danh sách của em. Rủi ro reviewer hỏi vẫn còn (PLAN_BASELINE_v2 mục 7) |
| Q5. Cửa sổ thời gian của SIG-Net | **3 cửa sổ phủ đủ 35 ngày** (12/12/11) | Xem mục 7, chỗ sửa S2 |
| Q6. File `cluster` của CFIN | **Giữ** | Tác giả CFIN phát hành sẵn file này nhưng không có code tạo ra nó. Ghi chú trong bài |

---

## 1. Toàn bộ luồng chạy

```
data/raw/xuetangx (prediction_data.tar.gz, user_info.csv, course_info.csv)
        │
        ▼
 [B0] prepare_raw.py ── lọc giống hệt M: 22 action, ngày 0–34 tính từ ngày mở khóa
        │                chia 3 split giống hệt M (SPLIT_SEED 1)
        ▼
data/processed/baseline_raw/   ← đúng định dạng gốc XuetangX mà 4 code gốc đang đọc
        │
        ├──► [B1–B2] LR, SVM ──────────── đọc X 89 cột của M
        ├──► [B3] CFIN     ─┐
        ├──► [B4] CA-TFHN   │  code gốc tự trích feature, dựng đồ thị
        ├──► [B5] MST-GCN   │  + patch nhỏ: split, chọn epoch theo val, lỗi, rò rỉ
        └──► [B6] SIG-Net  ─┘
                    │
                    ▼  mỗi mô hình, mỗi seed ghi 2 file dự đoán: enroll_id, label, prob
outputs/baselines/<tên>/seed<s>_{validation,test}.csv
                    │
                    ▼
 [B7] evaluate_baselines.py ── tính chỉ số bằng ĐÚNG hàm metrics() của src/9_train.py
                    │
                    ▼
outputs/baseline_results.csv  →  Bảng 1 trong bài
```

**Luật chung cho mọi baseline học sâu** (để khỏi lặp lại ở từng mục):

| Luật | Cụ thể |
|---|---|
| Split | train 126,354 / val 31,589 / test 67,699, đọc từ file của B0 |
| Nhãn | Loss chỉ dùng nhãn train. Nhãn val/test không bao giờ vào feature, đồ thị hay loại cạnh |
| Transductive | Feature (không có nhãn) của enrollment val/test **được** vào đồ thị, như M. Trong code gốc các bài cũng làm vậy |
| Chuẩn hóa | Scaler ở mức **enrollment** chỉ fit trên train rồi áp cho cả 3 split. Bảng ở mức **node chung** (user, course, object) không thuộc riêng split nào nên giữ như code gốc |
| Chọn epoch | Mỗi epoch tính val AUC. Dừng khi 5 epoch liên tiếp không vượt mức tốt nhất. Lấy lại trọng số tốt nhất rồi dự đoán val và test **một lần** |
| Số epoch tối đa | 3 × số epoch cố định của bài gốc. Nếu epoch tốt nhất chạm trần thì tăng trần và ghi lại |
| Siêu tham số | Giữ nguyên mặc định của code gốc |
| Seed | (1, 11, 111, 1111, 11111), giống M. Seed đổi phần khởi tạo và thứ tự batch, split giữ cố định |

---

## 2. B0 — Bộ dữ liệu thô chung

**File mới:** `baseline/common/prepare_raw.py`

**Đầu ra** `data/processed/baseline_raw/`, cùng định dạng với `prediction_data.tar.gz` gốc:

| File | Nội dung |
|---|---|
| `prediction_log/{train,validation,test}_log.csv` | Các cột gốc `enroll_id, username, course_id, session_id, action, object, time`. Chỉ giữ dòng thỏa **đúng luật của M** (`src/2_preprocess.py:97-100`): action thuộc 22 action, `0 ≤ ngày(time) − ngày(course start) < 35` |
| `prediction_log/{train,validation,test}_truth.csv` | `enroll_id, truth` |
| `prediction_log/enrollments.csv` | `enroll_id, username, course_id, split` của **cả 225,642** enrollment, kể cả 3,305 enrollment không có sự kiện nào trong ngày 0–34 |
| `user_info.csv`, `course_info.csv` | Chép nguyên bản |

**Cách làm:** import `split_train_enrollments` và các hằng số từ `src/2_preprocess.py` và `0_config.py`, để split và luật lọc **dùng chung code** với M, không viết lại.

**Vì sao cần `enrollments.csv`:** cả 4 code gốc lấy danh sách enrollment từ file log. Enrollment không có log trong 35 ngày sẽ bị rơi mất, và test sẽ còn 66,745 thay vì 67,699. Mỗi patch sẽ lấy danh sách từ file truth và gán 0 cho các đếm còn thiếu.

**Kiểm tra C1:**
- Tập enroll_id từng split trùng với `data/processed/simple/{split}.csv`.
- Số sự kiện từng split bằng số dòng có action trong file của M.
- In số sự kiện bị loại theo từng lý do (ngoài 22 action, ngoài ngày 0–34).

---

## 3. B1–B2 — LR và SVM

**File mới:** `baseline/common/run_tabular.py` (sklearn, chạy local được)

| | LR | SVM |
|---|---|---|
| Đầu vào | X 89 cột của M (`data/processed/simple/{split}/X.npy`) | như LR |
| Mô hình | `LogisticRegression(max_iter=5000)` | `LinearSVC(dual=False)` |
| Siêu tham số | C ∈ {0.01, 0.1, 1, 10, 100}, chọn theo val AUC | như LR |
| Huấn luyện | Fit trên train, chấm val để chọn C, rồi dự đoán test với C tốt nhất | như LR |
| prob ghi ra | `predict_proba[:, 1]` | `σ(decision_function)`. Hàm này đơn điệu nên AUC/AUPRC không đổi, và p ≥ 0.5 ⇔ score ≥ 0. Đây không phải xác suất đã hiệu chỉnh, chỉ dùng để đặt ngưỡng |
| Seed | Chạy đủ 5 seed cho đồng bộ. Std gần như bằng 0 vì hai mô hình gần như tất định | như LR |

**Lưu ý:** `X.npy` ở máy local đang cũ (90 cột). Phải chạy lại `3_features.py` trước (đã có trong bộ nhớ ngày 2026-10-08).

---

## 4. B3 — CFIN (Feng et al., AAAI 2019)

**Code gốc:** `wzfhaha/dropout_prediction` commit a99db2e (đã có trong `baseline/CFIN/`). Python 3.7, TensorFlow 1.15. Môi trường `.venv` local và server đã có.

**CFIN tự xây gì từ dữ liệu thô (giữ nguyên):**

| Nhóm | Cách CFIN tự làm |
|---|---|
| Hoạt động | `feat_extract.py`: đếm 21 action + `all#count` + `session#count` cho mỗi enrollment |
| Context smoothing | `main.py feat_augment`: với mỗi đếm, thêm mean/max theo user và theo course, tính trên mọi enrollment (không dùng nhãn) |
| Người học | age = 2018 − birth (0 nếu ngoài 10–70), gender, `cluster_label`, `user_enroll_num` |
| Khóa học | `course_category`, `course_enroll_num` |
| Chuẩn hóa | z-score cho các đếm, age và enroll_num |
| Mô hình | Embedding + conv + attention giữa context và hoạt động, DNN |

**Chỗ phải sửa** (thay `cfin_changes.patch` bản cũ, bỏ phần log1p):

| # | Chỗ | Gốc | Sửa thành | Lý do |
|---|---|---|---|---|
| C-1 | `feat_extract.py` (toàn file) | Đọc `train_log`, `test_log`, danh sách enrollment lấy từ log | Đọc 3 log + 3 truth; danh sách lấy từ truth; enrollment không có log thì đếm = 0; ghi 3 file feature. Vẫn đọc theo chunk như patch cũ (đã kiểm tra cho ra file giống hệt bản gốc) | Split; giữ đủ 67,699 test |
| C-2 | `preprocess.py:26` `gender_convert` | So với `'m'`/`'f'`, nhưng dữ liệu ghi `'male'`/`'female'` → gender = 0 với mọi người | `'male'`→1, `'female'`→2 | Lỗi đọc dữ liệu (R7) |
| C-3 | `preprocess.py:66` | `course_info` index theo cột `id` (số), nhưng tra bằng chuỗi `course_id` → category = 0 với mọi khóa | `index_col='course_id'` | Lỗi đọc dữ liệu (R7) |
| C-4 | `preprocess.py:84-89` | Bỏ log1p (của patch cũ); `StandardScaler` fit trên train + test | Không log1p; fit trên train, transform cả 3 split | Q1; R3 |
| C-5 | `main.py:135` | Train trên toàn bộ train chính thức (= train + val của M), lấy test làm "valid", `early_stopping=False`, epoch 27 cố định | `fit(train, valid = val của M)`, dừng theo luật chung (patience 5 trùng với `early_stopping_round=5` có sẵn của CFIN), lưu checkpoint tốt nhất bằng `saver` có sẵn, max 81 epoch | R4 |
| C-6 | `main.py` cuối file | `random_seed` 12345, chỉ in AUC | Nhận `--seed`, ghi 2 file dự đoán | R6, B7 |

**Giữ nguyên:** mọi siêu tham số (`embedding 32, conv 512, lr 1e-4, batch 32, dropout 0.9, l2 1e-5…`), `feat_augment`, `dataparse` (vocab tính trên mọi split, không dùng nhãn), file `cluster`.

**Kiểm tra:**
- C-1 cho ra đúng 67,699 dòng test.
- Sau C-2, C-3: gender và category không còn bằng 0 với mọi người (in phân bố).
- Kiểm tra cô lập nhãn (C3 ở mục 9).

**Thời gian:** khoảng 105 s/epoch trên CPU server. Ước tính 30–80 epoch × 5 seed ≈ 5–12 giờ.

---

## 5. B4 — CA-TFHN (Liang et al., ICONIP 2023)

**Code gốc:** `codeds27/CA-TFHN` commit f8ef659 (`baseline/CA-TFHN/`). torch 1.11, torch-geometric 2.1.

**CA-TFHN tự xây gì (giữ nguyên):**

| Nhóm | Cách CA-TFHN tự làm |
|---|---|
| Chuỗi hoạt động | `0_process…py`: đếm 22 action × 35 ngày (tính từ ngày mở khóa), đưa vào TFHN (LSTM + self-attention) dưới dạng đếm thô |
| Context | `1_process…py`: age (2023 − birth), gender, education, `user_enroll_num`, `course_enroll_num`, `course_category` |
| Dự đoán liên kết | `linkprediction_models.py`: GNN dị thể user–course học embedding, dùng cạnh "có đăng ký" (không dùng nhãn dropout) |
| Đồ thị bạn cùng lớp | `0_create_strong…py`: trong mỗi khóa, nối hai enrollment khi cosine của embedding ≥ 0.95 |
| Huấn luyện | GraphSAGE sampling (8, 4), BCE |

**Chỗ phải sửa** (file mới `baseline/CA-TFHN/catfhn_changes.patch`):

| # | Chỗ | Gốc | Sửa thành | Lý do |
|---|---|---|---|---|
| T-1 | `0_process…py:36-50` | 770 lần `groupby().sum()` trên toàn bộ ~30 triệu dòng log (rất chậm) | Một lần `pivot` cho ra cùng bảng đếm. Kiểm tra bằng cách so với code gốc trên mẫu nhỏ: hai bảng phải giống hệt | Chỉ để chạy được, kết quả không đổi |
| T-2 | `0_process…py:8-17, 58-61` | 2 log, danh sách enrollment lấy từ log | 3 log + 3 truth; danh sách lấy từ truth, gán 0 khi thiếu | Split; đủ 67,699 |
| T-3 | `1_process…py:31` | Lỗi gender giống CFIN | Sửa giống C-2 | R7 |
| T-4 | `1_process…py:60` | Lỗi category giống CFIN | Sửa giống C-3 | R7 |
| T-5 | `1_process…py` | **Không tạo cột `cluster_label`**, nhưng `3_mapping.py:9`, `linkprediction…py:23` và `1_graph…py:28` đều đọc cột này → code gốc chạy là lỗi | Thêm `cluster_label` từ đúng file `cluster` của CFIN (CA-TFHN dựa trên context của CFIN) | Code gốc thiếu bước |
| T-6 | `1_process…py:81-82` | Scaler cho age, enroll_num fit trên train + test | Fit trên train | R3 |
| T-7 | `3_mapping.py`, `0_create…py` | `tt_label` chỉ có train/test; chỉ có `train_mask`, `test_mask` | Thêm `validation` và `val_mask` | Split |
| T-8 | `0_create…py` dòng cuối, `1_graph…py` dòng cuối | `open(os.path.join(dir, 'X.pkl', 'wb'))`: đặt nhầm `'wb'` vào trong `join` → không ghi được file | `open(os.path.join(dir, 'X.pkl'), 'wb')` | Lỗi code |
| T-9 | `train.py:42-118` | Mỗi epoch in chỉ số **test**, không có val, 15 epoch, `-r` (seed) nhận vào nhưng không dùng | Thêm `val_loader`; dừng theo luật chung, max 45 epoch; đặt seed cho torch, numpy, random; dự đoán val và test bằng trọng số tốt nhất, ghi theo `n_id` | R4, R6 |

**Bỏ qua:** `2_table_2_numpy.py`, vì `train.py` không đọc file mà nó tạo ra.

**Giữ nguyên:** siêu tham số `param_dict`, lr 1e-4, ngưỡng cosine 0.95, NeighborLoader (8, 4), batch 256, link prediction (có split nội bộ 10% riêng để chọn epoch cho chính nó, không dính nhãn dropout).

**Lưu ý:** NeighborLoader lấy mẫu hàng xóm ngẫu nhiên, nên lúc dự đoán cũng có ngẫu nhiên. Cố định seed lúc dự đoán để chạy lại ra đúng kết quả cũ.

**Thời gian:** trước đây đã chạy được trên CPU (tag `full-hsl`). Sẽ đo lại bằng smoke test.

---

## 6. B5 — MST-GCN

**Code gốc:** `baseline/MST-GCN/MST-GCN/src_xuet` (commit 7350a7e), đã có sẵn bản XuetangX. Cần DGL → chạy trên server Linux.

**MST-GCN tự xây gì (giữ nguyên):**

| Nhóm | Cách MST-GCN tự làm (`xuetangx_graph_builder.py`) |
|---|---|
| Ngày | Ngày mở khóa **suy từ log** (thời điểm sớm nhất của khóa), cửa sổ 35 ngày |
| Node enrollment | 35 đếm theo ngày (thô) + tổng hành động, số object khác nhau, phân bố action (z-score) |
| Node object | Phân bố action trên object (z-score) |
| Node course | Tổng hành động, số enrollment, số user, mật độ (z-score) |
| Đồ thị | Đồ thị dị thể enrollment–object–course; mỗi target lấy một subgraph gồm object của nó, khóa học của nó, và các enrollment khác của cùng người học |
| Mô hình | `Multi_MST_GCN` (2 layer, hidden 128, BiLSTM, gate dài hạn/ngắn hạn) |

**Chỗ phải sửa** (file mới `mstgcn_changes.patch`):

| # | Chỗ | Gốc | Sửa thành | Lý do |
|---|---|---|---|---|
| M-1 | `xuetangx_graph_builder.py:243-260` | Gộp train + test; lấy username/course_id từ log; `dropna` làm rơi enrollment không có log | Đọc 3 log + 3 truth; username/course_id lấy từ `enrollments.csv`; giữ cột split | Split; đủ 67,699 |
| M-2 | `xuetangx_graph_builder.py:179` | Scaler feature enrollment fit trên mọi enrollment | Fit trên enrollment train | R3 |
| M-3 | `trainer.py:209-212` | Chia 80/20 **theo thứ tự dòng** của train + test gộp → 22,570 enrollment của test chính thức nằm trong train | Index train/val/test lấy theo cột split | R2 |
| M-4 | `trainer.py:216-240` | 8 epoch cố định, chấm 1 lần ở epoch cuối | Mỗi epoch tính val AUC, dừng theo luật chung, max 24 epoch, lưu trọng số tốt nhất, ghi dự đoán | R4 |
| M-5 | `trainer.py:242-244` | Vẽ t-SNE, gate, tương quan trên test | Tắt (không ảnh hưởng kết quả) | Đỡ tốn thời gian |

**Giữ nguyên:** siêu tham số mặc định trong `main.py` (hidden 128, 2 layer, dropout 0.3, lr 1e-4, batch 256, weight decay 1e-5), cách suy ngày mở khóa, đếm theo ngày thô, scaler ở mức object/course. Không có `friends.csv` nên không có cạnh bạn bè, giống bản gốc trên XuetangX.

**Rủi ro:** mỗi target gọi `dgl.node_subgraph` riêng, có thể chậm với 225k enrollment. Chạy smoke test với `--subset_ratio 0.01` trước để đo thời gian mỗi epoch.

---

## 7. B6 — SIG-Net (Kim et al., ACM SAC 2024)

**Code gốc:** `Noverse0/SIG-Net` commit 87bc320. Bản gốc **chỉ có KDD Cup 2015 và NAVER**, nên đây là baseline duy nhất phải viết bộ dựng đồ thị mới cho XuetangX. Cần DGL → chạy trên server.

**Nguyên tắc:** dịch `kddcup_graph.process` + `__getitem__` (`graph_builder.py:32-480`) sang XuetangX **theo từng bước**. Chỉ đổi chỗ nào XuetangX khác KDD Cup. Model (`model.py`, `Multi_RGCN`) giữ nguyên, không sửa dòng nào.

**SIG-Net tự xây gì:**

| Nhóm | Bản gốc KDD Cup | Trên XuetangX |
|---|---|---|
| Ngày | Ngày (log) − ngày mở khóa (`date.csv`) | Ngày (log) − `course_info.start`. Trùng với B0 |
| Loại sự kiện | problem 0, video 1, discussion 2, còn lại 3 | Nhóm action của M: assignment 0, video 1, forum 2, web_page 3 |
| Node | enrollment, object, course | như gốc |
| Cạnh object–course | loại 0–3 theo loại object | như gốc |
| Cạnh enrollment–object trong cửa sổ w | loại 4–7 | như gốc |
| Cạnh enrollment–enrollment cùng người học | loại 8/9 theo trạng thái của enrollment gửi | Xem S3 |
| Feature enrollment | Đếm theo ngày, z-score **theo từng khóa học** | như gốc. Scaler fit trên train của khóa (S4) |
| Feature course | Số module theo loại (`object.csv`) + tổng enrollment | XuetangX không có bảng module → số object theo 4 loại thấy trong log + tổng enrollment (S5) |
| Mô hình | RGCN 2 layer trên 3 cửa sổ + BiLSTM + MLP | như gốc |

**Chỗ phải sửa / phải chuyển đổi** (file mới `signet_changes.patch` = thêm class `xuetangx_graph` + sửa `trainer.py`, `main.py`):

| # | Chỗ | Gốc | Trên XuetangX | Lý do |
|---|---|---|---|---|
| S1 | Đọc dữ liệu | `kddcup_load()` | `xuetangx_load()` đọc B0 | Dữ liệu khác |
| S2 | Cửa sổ | 30 ngày = 3 × 10 ngày | **3 cửa sổ phủ đủ 35 ngày:** 0–11, 12–23, 24–34 (Q5). Cách khác: giữ 3 × 10 và bỏ ngày 30–34, nhưng như vậy SIG-Net thấy ít thông tin hơn M (trái R1) | R1 |
| S3 | Loại cạnh cửa sổ 3 (`graph_builder.py:280, 296`) | Lấy **nhãn dropout thật** của enrollment gửi, kể cả nhãn test → **rò rỉ nhãn** | Enrollment gửi thuộc train → 8/9 theo nhãn thật; thuộc val/test → **loại 10 "chưa biết"**. `model.py:37,41` đã khai báo `num_rels=11`, trong khi bản gốc chỉ dùng 0–9, nên không phải sửa model. Bước che cạnh của chính target trong `__getitem__` giữ nguyên. Cửa sổ 1, 2 dùng "không hoạt động ở cửa sổ sau" (lấy từ log, không phải nhãn) nên giữ nguyên | R3, rò rỉ |
| S4 | Scaler theo khóa (`graph_builder.py:171`) | Fit trên mọi enrollment của khóa | Fit trên enrollment train của khóa, áp cho tất cả | R3 |
| S5 | Feature course | Đếm module theo 6 loại từ `object.csv` | Đếm object theo 4 loại từ log + tổng enrollment. Ghi rõ trong bài | Dữ liệu không có |
| S6 | Tốc độ | `iterrows` và lọc cả bảng cho từng enrollment: với 225k enrollment / 30 triệu dòng log thì không chạy xong | Viết lại bằng thao tác vector. **Kiểm tra:** chạy bản dịch thẳng (vòng lặp như gốc) và bản vector trên 2,000 enrollment; danh sách cạnh, loại cạnh và feature phải giống hệt | Chỉ để chạy được |
| S7 | `trainer.py:39, 51-55, 73-118` | Sampler theo split KDD, 5 lần chạy cùng seed 0, 4 epoch, chấm test ở epoch cuối | Sampler theo split B0; 1 lần chạy cho mỗi seed của mình; dừng theo luật chung, max 12 epoch; ghi dự đoán | R2, R4, R6 |

**Giữ nguyên:** RGCN, 2 layer, hidden 64, dropout 0.3, lr 1e-3, batch 256, BiLSTM. Subgraph của target gồm mọi object của khóa, các enrollment cùng người học, khóa học và chính target (`__getitem__`).

**Không dùng lại** adapter cũ `src/12_signet.py` (tag `full-hsl`). Bản đó dùng cache và cách xử lý của mình, lại theo giao thức inductive cũ, trái nguyên tắc vừa chốt.

**Rủi ro:**
1. Siêu tham số gốc được chọn cho KDD Cup, chưa từng chạy trên XuetangX. Ghi rõ trong bài.
2. Subgraph gồm mọi object của khóa nên có thể rất lớn (khóa đông object) → chậm, tốn RAM. Smoke test sẽ cho biết.

---

## 8. B7 — Đánh giá chung

**File mới:** `baseline/common/evaluate_baselines.py`

1. Đọc `outputs/baselines/<tên>/seed<s>_{validation,test}.csv`.
2. Kiểm tra: tập enroll_id trùng đúng split của B0 (val 31,589 / test 67,699), không thiếu, không trùng, prob hữu hạn.
3. Tính chỉ số bằng `metrics()` import từ `src/9_train.py`, **cùng một hàm với M**: AUC, AUPRC, Acc, P, R, F1, ngưỡng 0.5.
4. Ghi `outputs/baseline_results.csv` (mỗi dòng một mô hình × seed) và in bảng mean ± std.

---

## 9. Kiểm tra tổng

| Mã | Kiểm tra | Áp dụng |
|---|---|---|
| C1 | B0: tập enroll_id và số sự kiện từng split trùng với dữ liệu của M | B0 |
| C2 | Số dòng dự đoán: val 31,589, test 67,699; enroll_id trùng split | Mọi baseline |
| C3 | **Cô lập nhãn:** đổi ngẫu nhiên nhãn val/test trong B0, chạy lại phần tiền xử lý của từng bài → mọi file feature/đồ thị (trừ cột nhãn) phải giống hệt từng byte. Cùng ý với kiểm tra K3 của M | CFIN, CA-TFHN, MST-GCN, SIG-Net |
| C4 | Phần viết lại chỉ để tăng tốc cho kết quả giống hệt code gốc trên mẫu | CFIN C-1, CA-TFHN T-1, SIG-Net S6 |
| C5 | Smoke test 1 epoch trên tập con, đo thời gian mỗi epoch | Mọi baseline học sâu |
| C6 | Epoch tốt nhất không chạm trần | Mọi baseline học sâu |

---

## 10. Cấu trúc file (giống cách CFIN đang làm)

```
baseline/
  common/            (mới, track)  prepare_raw.py, run_tabular.py, evaluate_baselines.py
  CFIN/              (đã track)    cfin_changes.patch (bản 2), run_cfin.sh, requirements.txt
  CA-TFHN/           (mới track)   catfhn_changes.patch, run_catfhn.sh, requirements.txt
  MST-GCN/           (mới track)   mstgcn_changes.patch, run_mstgcn.sh, requirements.txt
  SIG-Net/           (mới track)   signet_changes.patch, run_signet.sh, requirements.txt
```

Mỗi `run_*.sh` làm 4 việc: clone đúng commit vào `baseline/<TÊN>/code/` (không track), apply patch, trỏ tới `data/processed/baseline_raw/`, rồi chạy 5 seed.

Ba clone hiện đang nằm trực tiếp ở `baseline/CA-TFHN`, `baseline/MST-GCN`, `baseline/SIG-Net` sẽ chuyển vào `code/`. Thư mục `code/` không track, nên việc chuyển không làm thay đổi git.

`.gitignore` cần thêm ngoại lệ cho 4 thư mục trên.

## 11. Thứ tự làm

| Bước | Việc | Chạy ở đâu | Ghi chú |
|---|---|---|---|
| 1 | B0 + C1 | local | Đọc 313 MB tar.gz một lượt |
| 2 | B1, B2, B7 | local | Có ngay 2 dòng đầu của bảng |
| 3 | B3 CFIN | local smoke → server | Môi trường đã có |
| 4 | B4 CA-TFHN | local smoke → server | |
| 5 | B5 MST-GCN | server (DGL) | Smoke để đo thời gian trước |
| 6 | B6 SIG-Net | server (DGL) | Tốn công nhất (S6) |

Mỗi bước thầy làm xong, chạy kiểm tra, rồi dừng để em xem trước khi sang bước sau.

## 12. Em cần trả lời

1. Đồng ý các mặc định ở mục 0 (Q1–Q6)?
2. Đồng ý luật "max epoch = 3 × epoch gốc, patience 5"?
3. Đồng ý cấu trúc file ở mục 10 (chuyển 3 clone vào `code/`)?

Trả lời xong em bảo **"tiến hành code"** thì thầy bắt đầu từ bước 1.
