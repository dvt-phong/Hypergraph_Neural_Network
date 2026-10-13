# Kịch bản chạy từng baseline (XuetangX)

Ngày lập: 2026-10-10. **Chưa code.** Bản này viết lại `PLAN_BASELINE_CHI_TIET.md` thành từng bước cụ thể. Nội dung kỹ thuật giữ nguyên, các mặc định Q1–Q6 cũng giữ nguyên (em vẫn chưa trả lời).

---

## Cách đọc

Mỗi baseline là **một kịch bản riêng**. Kịch bản nào cũng có cùng 7 bước:

| Bước | Tên | Câu hỏi bước này trả lời |
|---|---|---|
| 1 | Môi trường | Cài gì để code gốc chạy được? |
| 2 | Lấy code gốc | Lấy đúng code tác giả công bố, ở commit nào? |
| 3 | Đưa dữ liệu thô vào | Code gốc đọc dữ liệu từ đâu? |
| 4 | Sửa (patch) | Mình đụng vào chỗ nào, vì sao? Ngoài các chỗ này không đụng gì |
| 5 | Tiền xử lý | Code gốc tự trích feature hoặc dựng đồ thị. Mình kiểm tra nó không thấy nhãn val/test |
| 6 | Chạy thử nhỏ | Chạy 1 epoch trên tập con: có lỗi không, mỗi epoch mất bao lâu? |
| 7 | Chạy thật | 5 seed, dừng theo val, ghi file dự đoán |

Mỗi bước ghi 4 thứ: **Làm gì**, **Lệnh**, **Ra file gì**, **Đạt khi**.

Mọi baseline dùng chung một kiểu lệnh:

```
bash baseline/<TÊN>/run_<tên>.sh setup      # bước 1–4
bash baseline/<TÊN>/run_<tên>.sh features   # bước 5
bash baseline/<TÊN>/run_<tên>.sh smoke      # bước 6
bash baseline/<TÊN>/run_<tên>.sh full       # bước 7 (5 seed)
```

Thứ tự làm: **Bước chung A → LR, SVM → CFIN → CA-TFHN → MST-GCN → SIG-Net → Bước chung B.** Xong mỗi kịch bản thầy dừng lại để em xem.

---

## Bước chung A — Tạo bộ dữ liệu thô chung (làm một lần)

**Làm gì:** từ file tải về, tạo một bản dữ liệu thô **giống hệt định dạng gốc** của XuetangX, nhưng:
- chỉ giữ đúng những sự kiện M đang dùng: 22 action, trong ngày 0–34 tính từ ngày mở khóa;
- chia sẵn 3 phần train / validation / test, giống hệt split của M.

**Lệnh:** `python baseline/common/prepare_raw.py`

**Ra file:** `data/processed/baseline_raw/`
```
prediction_log/train_log.csv        validation_log.csv        test_log.csv
prediction_log/train_truth.csv      validation_truth.csv      test_truth.csv
prediction_log/enrollments.csv      (225,642 enrollment: enroll_id, username, course_id, split)
user_info.csv, course_info.csv      (chép nguyên bản)
```

**Đạt khi:**
- [ ] Số enrollment train / val / test = 126,354 / 31,589 / 67,699, và tập enroll_id trùng với dữ liệu của M.
- [ ] Số sự kiện mỗi split bằng số dòng sự kiện trong `data/processed/simple/{split}.csv`.
- [ ] In ra số sự kiện bị loại và lý do.

---

## Kịch bản 1 — LR (Logistic Regression)

**Tóm tắt:** mô hình tuyến tính đơn giản, làm mốc dưới. Không có cách trích feature riêng, nên dùng X 89 cột của M.

| Bước | Làm gì | Đạt khi |
|---|---|---|
| 1. Môi trường | Dùng `.venv` của project (đã có sklearn) | — |
| 2. Code | Không có code gốc; viết `baseline/ML/LR.py` (độc lập, không dùng chung file nào) | — |
| 3. Dữ liệu | Đọc `data/processed/simple/{train,validation,test}/X.npy` và nhãn. **Chạy lại `3_features.py` trước**, vì `X.npy` ở máy local đang cũ (90 cột) | X có 89 cột |
| 4. Sửa | Không có | — |
| 5. Tiền xử lý | Không có (X đã chuẩn hóa bằng scaler fit trên train) | — |
| 6. Chạy thử | Bỏ qua (chạy thật chỉ mất vài phút) | — |
| 7. Chạy thật | Train bằng SGD (`SGDClassifier(loss="log_loss")`), giống cách M train: mỗi epoch xáo thứ tự train theo seed, chấm val AUC, dừng khi 10 epoch liên tiếp không cải thiện (tối đa 200), lấy epoch tốt nhất. Thử alpha (L2) ∈ {1e-6, …, 1}, chọn theo val AUC; lặp 5 seed. Seed đổi thứ tự dữ liệu nên có std thật, giống M | `outputs/baselines/LR/results.csv` (5 dòng) + 10 file `seed<s>_{validation,test}.csv` |

**Lệnh:** `python baseline/ML/LR.py`

---

## Kịch bản 2 — SVM

**Tóm tắt:** giống LR nhưng dùng hinge loss. Dùng SVM tuyến tính, vì SVM kernel RBF trên 126k dòng không chạy nổi.

Giống hệt kịch bản 1, chỉ khác 2 chỗ:
- Loss là hinge (`SGDClassifier(loss="hinge")` = SVM tuyến tính).
- SVM không trả xác suất, nên ghi `prob = σ(score)`. Hàm này đơn điệu nên AUC không đổi, và ngưỡng 0.5 tương ứng score ≥ 0.

**Lệnh:** `python baseline/ML/SVM.py` (file riêng `baseline/ML/SVM.py`, kết quả ở `outputs/baselines/SVM/`)

---

## Kịch bản 3 — CFIN (Feng et al., AAAI 2019)

**Tóm tắt:** đếm hoạt động của mỗi enrollment, "làm mượt" bằng mức trung bình/cao nhất của cùng người học và cùng khóa học, rồi dùng attention để kết hợp với thông tin người học và khóa học.

**Nó tự làm (mình không đụng):** đếm 21 action; context smoothing (mean/max theo user và course); tuổi, giới tính, nhóm người học (cluster); loại khóa học; số khóa mỗi người đăng ký; số người mỗi khóa; z-score; toàn bộ kiến trúc và siêu tham số.

**Mình đụng 6 chỗ:**

| Sửa | Ở đâu | Hiện tại | Sửa thành | Vì sao |
|---|---|---|---|---|
| 1 | `feat_extract.py` | Đọc 2 file train/test; enrollment không có log bị rơi | Đọc 3 split; enrollment không có log thì đếm = 0 | Đúng split, đủ 67,699 test |
| 2 | `preprocess.py:26` | So giới tính với `'m'`/`'f'`, nhưng dữ liệu ghi `'male'`/`'female'` → mọi người đều 0 | So với `'male'`/`'female'` | Lỗi đọc dữ liệu |
| 3 | `preprocess.py:66` | Tra loại khóa học sai cột → mọi khóa đều 0 | Tra đúng cột `course_id` | Lỗi đọc dữ liệu |
| 4 | `preprocess.py:84-89` | Có log1p (patch cũ của mình); scaler fit trên train + test | Bỏ log1p; scaler fit trên train | Đúng cách xử lý gốc; không dùng thống kê của test |
| 5 | `main.py:135` | Train trên train + val, chấm test mỗi epoch, cố định epoch 27 | Train trên train, chấm val mỗi epoch, dừng sau 5 epoch không cải thiện (tối đa 81), lấy trọng số tốt nhất | Chọn epoch theo val |
| 6 | `main.py` cuối | Seed cố định 12345, chỉ in AUC | Nhận `--seed`, ghi file dự đoán | 5 seed, đánh giá chung |

**Các bước:**

| Bước | Làm gì | Lệnh | Đạt khi |
|---|---|---|---|
| 1. Môi trường | Python 3.7 + TF 1.15 (local `baseline/CFIN/.venv`, server conda `cfin`) — **đã có** | — | `python -c "import tensorflow"` chạy được |
| 2. Code gốc | Clone `wzfhaha/dropout_prediction` ở commit a99db2e (đã có) | `run_cfin.sh setup` | — |
| 3. Dữ liệu | Trỏ code gốc tới `data/processed/baseline_raw/` | `run_cfin.sh setup` | — |
| 4. Sửa | Apply `cfin_changes.patch` bản 2 (6 chỗ trên) | `run_cfin.sh setup` | `git diff --stat` chỉ đụng 4 file |
| 5. Tiền xử lý | `feat_extract.py` → `preprocess.py` | `run_cfin.sh features` | 3 file feature: 126,354 / 31,589 / 67,699 dòng; giới tính và loại khóa học **không còn** toàn 0; kiểm tra cô lập nhãn đạt |
| 6. Chạy thử | 1 epoch trên 4,000 dòng | `run_cfin.sh smoke` | Không lỗi; biết thời gian/epoch (trước đây ~105 s trên server) |
| 7. Chạy thật | 5 seed trên server | `run_cfin.sh full` | 10 file dự đoán; epoch tốt nhất < 81 |

**Thời gian dự kiến:** 5–12 giờ trên server.

---

## Kịch bản 4 — CA-TFHN (Liang et al., ICONIP 2023)

**Tóm tắt:** đưa chuỗi 22 action × 35 ngày qua LSTM + self-attention; thêm thông tin từ "bạn cùng lớp" (những người cùng khóa có sở thích giống nhau, tìm bằng một mô hình dự đoán liên kết user–khóa học).

**Nó tự làm (mình không đụng):** chuỗi 22 × 35 đếm thô; context (tuổi, giới tính, học vấn, số đăng ký, loại khóa học); mô hình dự đoán liên kết user–course; đồ thị bạn cùng lớp (cosine ≥ 0.95); GraphSAGE lấy mẫu (8, 4); toàn bộ siêu tham số.

**Mình đụng 9 chỗ:**

| Sửa | Ở đâu | Hiện tại | Sửa thành | Vì sao |
|---|---|---|---|---|
| 1 | `0_process…py:36-50` | 770 lần `groupby` trên ~30 triệu dòng (chạy rất lâu) | Một lần `pivot`, ra cùng một bảng | Chỉ để chạy được; kiểm tra bằng cách so với code gốc trên mẫu nhỏ |
| 2 | `0_process…py` | 2 split; enrollment không có log bị rơi | 3 split; đếm = 0 khi thiếu | Đúng split |
| 3 | `1_process…py:31` | Lỗi giới tính giống CFIN | Sửa giống CFIN | Lỗi đọc dữ liệu |
| 4 | `1_process…py:60` | Lỗi loại khóa học giống CFIN | Sửa giống CFIN | Lỗi đọc dữ liệu |
| 5 | `1_process…py` | **Không tạo cột `cluster_label`**, nhưng 3 file sau đều đọc cột này → code gốc lỗi | Thêm từ file cluster của CFIN | Code gốc thiếu bước |
| 6 | `1_process…py:81` | Scaler fit trên train + test | Fit trên train | Không dùng thống kê của test |
| 7 | `3_mapping.py`, `0_create…py` | Chỉ có train/test | Thêm validation | Đúng split |
| 8 | Dòng cuối `0_create…py`, `1_graph…py` | `'wb'` đặt nhầm trong `os.path.join` → không lưu được file | Sửa lại | Lỗi code |
| 9 | `train.py` | Mỗi epoch in kết quả test, 15 epoch, seed nhận vào nhưng không dùng | Chấm val mỗi epoch, dừng sau 5 epoch không cải thiện (tối đa 45), đặt seed, ghi file dự đoán | Chọn epoch theo val; 5 seed |

**Các bước:**

| Bước | Làm gì | Lệnh | Đạt khi |
|---|---|---|---|
| 1. Môi trường | Python 3.9, torch 1.11, torch-geometric 2.1 (theo README). Viết `requirements.txt` | — | `import torch_geometric` chạy được |
| 2. Code gốc | Clone `codeds27/CA-TFHN` commit f8ef659 vào `baseline/CA-TFHN/code/` | `run_catfhn.sh setup` | — |
| 3. Dữ liệu | Trỏ `datastore` tới `baseline_raw` | `run_catfhn.sh setup` | — |
| 4. Sửa | Apply `catfhn_changes.patch` (9 chỗ trên) | `run_catfhn.sh setup` | — |
| 5. Tiền xử lý | Chạy 6 bước của `feat_extract.sh` (bỏ `2_table_2_numpy.py`, vì `train.py` không dùng file nó tạo ra) | `run_catfhn.sh features` | Đồ thị có 225,642 node; có đủ `train_mask` / `val_mask` / `test_mask`; kiểm tra cô lập nhãn đạt |
| 6. Chạy thử | 1 epoch | `run_catfhn.sh smoke` | Không lỗi; biết thời gian/epoch |
| 7. Chạy thật | 5 seed | `run_catfhn.sh full` | 10 file dự đoán; epoch tốt nhất < 45 |

---

## Kịch bản 5 — MST-GCN

**Tóm tắt:** dựng đồ thị enrollment – object – khóa học; với mỗi enrollment lấy một đồ thị con, rồi học cả thay đổi theo thời gian lẫn quan hệ trong đồ thị.

**Nó tự làm (mình không đụng):** suy ngày mở khóa từ log; 35 đếm theo ngày; feature của object và khóa học; cách lấy đồ thị con; kiến trúc và siêu tham số (hidden 128, 2 layer, lr 1e-4, batch 256).

**Mình đụng 5 chỗ:**

| Sửa | Ở đâu | Hiện tại | Sửa thành | Vì sao |
|---|---|---|---|---|
| 1 | `xuetangx_graph_builder.py:243-260` | Gộp train + test; enrollment không có log bị bỏ | Đọc 3 split; giữ mọi enrollment | Đúng split |
| 2 | `xuetangx_graph_builder.py:179` | Scaler fit trên mọi enrollment | Fit trên train | Không dùng thống kê của test |
| 3 | `trainer.py:209-212` | Lấy 80/20 **theo thứ tự dòng** → 22,570 enrollment của test chính thức bị đưa vào train | Chia theo cột split | Test phải tách riêng |
| 4 | `trainer.py:216-240` | 8 epoch cố định, chấm test ở epoch cuối | Chấm val mỗi epoch, dừng sau 5 epoch không cải thiện (tối đa 24), ghi file dự đoán | Chọn epoch theo val |
| 5 | `trainer.py:242-244` | Vẽ hình t-SNE, gate | Tắt | Không ảnh hưởng kết quả, đỡ tốn thời gian |

**Các bước:**

| Bước | Làm gì | Lệnh | Đạt khi |
|---|---|---|---|
| 1. Môi trường | **Server Linux**: torch + DGL (cùng phiên bản CUDA với server). Viết `requirements.txt` | — | `import dgl` chạy được |
| 2. Code gốc | Clone repo MST-GCN commit 7350a7e vào `baseline/MST-GCN/code/` | `run_mstgcn.sh setup` | — |
| 3. Dữ liệu | Trỏ `data/xuetangx` tới `baseline_raw` | `run_mstgcn.sh setup` | — |
| 4. Sửa | Apply `mstgcn_changes.patch` (5 chỗ trên) | `run_mstgcn.sh setup` | — |
| 5. Tiền xử lý | Dựng đồ thị, lưu cache | `run_mstgcn.sh features` | 225,642 node enrollment; kiểm tra cô lập nhãn đạt |
| 6. Chạy thử | 1 epoch với `--subset_ratio 0.01` | `run_mstgcn.sh smoke` | Không lỗi; **ước được thời gian 1 epoch đầy đủ**. Nếu quá 3 giờ/epoch thì dừng, báo em |
| 7. Chạy thật | 5 seed | `run_mstgcn.sh full` | 10 file dự đoán; epoch tốt nhất < 24 |

---

## Kịch bản 6 — SIG-Net (Kim et al., ACM SAC 2024)

**Tóm tắt:** dựng đồ thị sinh viên – object – khóa học cho 3 khoảng thời gian; mỗi enrollment lấy một đồ thị con gồm các khóa khác cùng người học, các object của khóa và khóa học, rồi đưa qua RGCN + BiLSTM.

**Khác 3 bài trên:** code gốc **chỉ có bản KDD Cup và NAVER**, không có XuetangX. Vì vậy phải **dịch bộ dựng đồ thị** từ bản KDD Cup sang XuetangX, từng bước một. Model giữ nguyên, không sửa dòng nào.

**Bảng dịch KDD Cup → XuetangX:**

| Thành phần | KDD Cup (gốc) | XuetangX (mình làm) |
|---|---|---|
| Ngày | tính từ ngày mở khóa | như gốc, ngày mở khóa lấy từ `course_info` |
| Loại sự kiện | problem / video / discussion / khác | assignment / video / forum / web_page (nhóm action của M) |
| 3 khoảng thời gian | 3 × 10 ngày (30 ngày) | 0–11, 12–23, 24–34 (phủ đủ 35 ngày, Q5) |
| Feature khóa học | số module theo loại (file `object.csv`) | XuetangX không có file này → số object theo 4 loại đếm từ log + tổng số enrollment |

**Mình đụng 7 chỗ:**

| Sửa | Ở đâu | Hiện tại | Sửa thành | Vì sao |
|---|---|---|---|---|
| 1 | Đọc dữ liệu | `kddcup_load()` | Thêm `xuetangx_load()` đọc `baseline_raw` | Dữ liệu khác |
| 2 | Khoảng thời gian | 3 × 10 ngày | 12 / 12 / 11 ngày | Thấy đủ 35 ngày như M |
| 3 | `graph_builder.py:280, 296` | Cạnh giữa các khóa cùng người học ở khoảng 3 được gán loại theo **nhãn dropout thật, kể cả nhãn test** | Enrollment train: giữ theo nhãn thật; enrollment val/test: gán loại "chưa biết" (loại 10, model đã có sẵn chỗ) | **Rò rỉ nhãn test** |
| 4 | `graph_builder.py:171` | Chuẩn hóa theo từng khóa, fit trên mọi enrollment | Fit trên enrollment train của khóa | Không dùng thống kê của test |
| 5 | Feature khóa học | Đếm module từ `object.csv` | Đếm object từ log | Dữ liệu không có file đó |
| 6 | Tốc độ | Dùng `iterrows` và lọc cả bảng cho từng enrollment → với 225k enrollment không chạy xong | Viết lại bằng thao tác vector, cho ra **cùng** đồ thị | Chỉ để chạy được |
| 7 | `trainer.py` | Split của KDD, 5 lần chạy cùng seed 0, 4 epoch, chấm test ở epoch cuối | Split của mình, 5 seed, dừng sau 5 epoch không cải thiện (tối đa 12), ghi file dự đoán | Chọn epoch theo val |

**Các bước:**

| Bước | Làm gì | Lệnh | Đạt khi |
|---|---|---|---|
| 1. Môi trường | **Server Linux**: torch 1.12.1 + DGL (theo `requirements.txt` và Dockerfile gốc). Thử dùng chung môi trường với MST-GCN | — | `import dgl` chạy được |
| 2. Code gốc | Clone `Noverse0/SIG-Net` commit 87bc320 vào `baseline/SIG-Net/code/` | `run_signet.sh setup` | — |
| 3. Dữ liệu | Thêm `xuetangx_load()` đọc `baseline_raw` | `run_signet.sh setup` | — |
| 4. Sửa | Apply `signet_changes.patch` (7 chỗ trên) | `run_signet.sh setup` | — |
| 5a. Kiểm tra bản dịch | Trên 2,000 enrollment, chạy **bản dịch thẳng** (vòng lặp y như gốc) và **bản nhanh** (sửa 6) | `run_signet.sh check` | Cạnh, loại cạnh, feature giống hệt nhau |
| 5b. Tiền xử lý | Dựng 3 đồ thị cho toàn bộ dữ liệu | `run_signet.sh features` | Không cạnh nào xuất phát từ val/test mang loại 8/9; kiểm tra cô lập nhãn đạt |
| 6. Chạy thử | 1 epoch trên tập con | `run_signet.sh smoke` | Không lỗi; ước được thời gian/epoch và RAM (đồ thị con chứa **mọi object của khóa**, có thể rất lớn) |
| 7. Chạy thật | 5 seed | `run_signet.sh full` | 10 file dự đoán; epoch tốt nhất < 12 |

**Lưu ý khi viết bài:** siêu tham số gốc được chọn cho KDD Cup, chưa từng chạy trên XuetangX. Ghi rõ điều này.

---

## Bước chung B — Tổng hợp kết quả

**Làm gì:** đọc mọi file dự đoán, tính chỉ số bằng **đúng hàm `metrics()` của M** (`src/9_train.py`), ra bảng mean ± std.

**Lệnh:** `python baseline/common/evaluate_baselines.py`

**Ra file:** `outputs/baseline_results.csv`, và in bảng:

| Mô hình | AUC | AUPRC | F1 | Acc | P | R |
|---|---|---|---|---|---|---|
| LR / SVM / CFIN / CA-TFHN / MST-GCN / SIG-Net / **M** | mean ± std | | | | | |

**Đạt khi:**
- [ ] Mỗi mô hình có đủ 5 seed.
- [ ] Mỗi file test có đúng 67,699 enroll_id trùng với split.
- [ ] Không có prob NaN.

---

## Kiểm tra cô lập nhãn (dùng ở bước 5 của 4 bài học sâu)

1. Tạo một bản `baseline_raw` thứ hai, trong đó nhãn val/test bị **đổi ngẫu nhiên**.
2. Chạy tiền xử lý của baseline trên cả hai bản.
3. So mọi file feature/đồ thị (trừ cột nhãn): **phải giống hệt từng byte.**

Nếu giống hệt thì baseline đó không thể dùng nhãn val/test, dù vô tình. Nếu khác thì dừng lại, tìm chỗ rò rỉ.

---

## Tóm tắt khối lượng

| Baseline | Số chỗ sửa | Chạy ở đâu | Độ khó | Rủi ro chính |
|---|---|---|---|---|
| LR, SVM | 0 (viết mới ~60 dòng) | local | Dễ | — |
| CFIN | 6 | local thử → server | Dễ (đã chạy được) | Thời gian 5–12 giờ |
| CA-TFHN | 9 | local thử → server | Trung bình | Code gốc có lỗi, cần sửa mới chạy được |
| MST-GCN | 5 | server | Trung bình | Lấy đồ thị con từng enrollment có thể chậm |
| SIG-Net | 7 + dịch bộ dựng đồ thị | server | Khó | Phải dịch từ KDD Cup; đồ thị con lớn |

## Em cần trả lời

1. Các mặc định Q1–Q6 (`PLAN_BASELINE_CHI_TIET.md` mục 0).
2. Luật dừng: tối đa 3 × số epoch gốc, dừng sau 5 epoch không cải thiện.
3. Chuyển 3 clone vào `code/` (mục 10 của plan chi tiết).

Em bảo **"tiến hành code"** thì thầy bắt đầu từ **Bước chung A**.
