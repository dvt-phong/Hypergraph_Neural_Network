# So sánh baseline v3: cùng t_obs, cùng split theo learner, cùng metric

Lập 2026-10-13. **Đã code 2026-10-13**: LR, SVM đã chạy xong ở local; CFIN, CA-TFHN, MST-GCN, SIG-Net đã sửa xong, kiểm tra xong phần chạy được ở local, chờ chạy trên server. Bản này thay `PLAN_BASELINE_v2.md`, `PLAN_BASELINE_CHI_TIET.md`, `KICH_BAN_BASELINE.md` và `PLAN_CFIN.md`.

Baseline: **LR, SVM, CFIN, CA-TFHN, MST-GCN, SIG-Net**. Mọi thứ nằm trong `baseline/`. **Mô hình chính (src/) không đổi một dòng nào.**

---

## 0. Phương án, và WorkflowView thật ra làm gì

> Mỗi baseline giữ tiền xử lý gốc, nhưng chạy lại trên cùng t_obs, cùng split theo learner, cùng metric (AUPR + AUC).

WorkflowView (Verma & Counts, arXiv 2606.14654, phụ lục A.1.2) chỉ làm **một phần** của phương án này:

| Điểm | WorkflowView | Ở đây |
|---|---|---|
| Giữ code/tiền xử lý gốc | Có: "use the authors' public implementation … re-train CFIN … otherwise following the paper/code defaults" | Có |
| Cùng cửa sổ quan sát | Có: dựng lại feature "under our 6-day input/24-hour evaluation window" | Có: ngày 0–34 của khóa (t_obs = 35 ngày, như M) |
| Split theo learner | **Không**: dùng test chính thức của Feng et al. (67,699 enrollment) | **Có** |
| Metric | **Weighted F1** | **AUC + AUPRC** |

Khi viết bài: trích WorkflowView cho ý "giữ code gốc, chỉ dựng lại input theo cùng cửa sổ"; split theo learner và AUC/AUPRC là lựa chọn của mình, cần nêu lý do riêng (mục 1).

---

## 1. Giao thức chung

| Mã | Quy tắc |
|---|---|
| P1 | **t_obs**: chỉ dùng sự kiện có `0 ≤ ngày(sự kiện) − ngày(mở khóa) < 35`, 22 action (như M) |
| P2 | **Split theo learner**: mỗi `username` nằm trọn trong một phần train / val / test |
| P3 | **Metric**: AUC và AUPRC, lớp dương = dropout (1), `roc_auc_score`, `average_precision_score` |
| P4 | Chọn epoch / siêu tham số theo **val AUC**; test chỉ chấm một lần, với trọng số tốt nhất |
| P5 | Siêu tham số = mặc định của code gốc. Dừng sau 5 epoch không cải thiện, tối đa 3 × số epoch gốc |
| P6 | Scaler / thống kê chuẩn hóa theo enrollment hay người học fit trên train; nhãn val/test không vào tiền xử lý |
| P7 | 5 split theo learner (seed 1, 11, 111, 1111, 11111), mỗi split 1 lần chạy, seed huấn luyện = seed split; báo mean ± std (ddof = 1) |

**Lý do chia theo learner** (để viết bài): ở split chính thức, 76.1% enrollment test có cùng learner với một enrollment train. Chia theo learner kiểm tra mô hình trên **người học mới hoàn toàn**.

---

## 2. Dữ liệu chung: `baseline/prepare_raw.py`

```
python baseline/prepare_raw.py          # ~8 phút, một lần
```

Ghi `data/processed/baseline_raw/`:

| File | Nội dung |
|---|---|
| `prediction_log/log.csv` | 40,558,640 sự kiện của 225,642 enrollment (train + test chính thức gộp lại), cột gốc, đã lọc theo P1 (bỏ 1,551,762 sự kiện sau ngày 34) |
| `prediction_log/truth.csv`, `enrollments.csv` | nhãn; `username`, `course_id` của mọi enrollment, kể cả 3,305 enrollment không có sự kiện |
| `splits/learner_s{seed}.csv` | `enroll_id, split`. Xáo danh sách learner đã sắp xếp bằng `numpy.random.default_rng(seed)`, chia learner 60 / 20 / 20 |
| `user_info.csv`, `course_info.csv` | chép nguyên |

Chỉ một file log (5.9 GB) cho mọi split; split nằm ở file nhỏ riêng. Mỗi baseline tự đọc split theo seed. Script import (chỉ đọc) `src/0_config.py` và các hàm đọc file của `src/2_preprocess.py`.

| Seed | train | val | test | dropout train / val / test |
|---|---|---|---|---|
| 1 | 135,304 | 45,539 | 44,799 | 0.757 / 0.763 / 0.758 |
| 11 | 135,347 | 45,375 | 44,920 | 0.758 / 0.757 / 0.760 |
| 111 | 135,514 | 45,423 | 44,705 | 0.757 / 0.760 / 0.762 |
| 1111 | 136,376 | 44,345 | 44,921 | 0.758 / 0.756 / 0.762 |
| 11111 | 135,528 | 45,071 | 45,043 | 0.760 / 0.757 / 0.755 |

77,083 learner; không learner nào nằm ở 2 phần (script tự kiểm tra); mọi khóa đều có enrollment train ở cả 5 split.

---

## 3. Mô hình chính M

Theo quyết định của em (2026-10-13): **không đụng vào M**. Lưu ý khi viết bài: M hiện báo trên split chính thức (theo enrollment), còn bảng baseline dùng split theo learner. Muốn đặt M vào cùng bảng thì phải chạy M trên split learner (việc này chưa làm và cần em quyết định riêng).

---

## 4. Từng baseline: giữ gì, đổi gì

Mỗi baseline học sâu có một thư mục `baseline/<TÊN>/` gồm `run_<tên>.sh`, `<tên>_changes.patch`, `requirements.txt` (git theo dõi); code gốc nằm trong `code/` (CFIN: `dropout_prediction/`), không đưa vào git. `run_<tên>.sh setup` clone đúng commit gốc rồi áp patch. Cả 4 patch đã thử áp lên bản clone sạch: đều áp được và compile được.

### 4.1 LR, SVM (`baseline/ML/LR.py`, `SVM.py`)

- Không có tiền xử lý gốc, nên dùng X 89 cột của M (`data/processed/simple/{split}/X.npy`, chỉ đọc). Các dòng được sắp lại theo `enroll_id`; 60 cột đã z-score (59 cột hành vi + tuổi) được z-score lại bằng train của split learner. z-score bất biến với biến đổi affine, nên kết quả giống hệt log1p + z-score fit trên train learner. Cột one-hot giữ 0/1.
- SGD (`log_loss` / `hinge`), alpha ∈ {1e-6 … 1} chọn theo val AUC, dừng sau 10 epoch không cải thiện (tối đa 200).
- Lần đầu chạy tạo cache `baseline_raw/x89_enroll_ids.csv` (enroll_id của từng dòng X).

### 4.2 CFIN (wzfhaha/dropout_prediction @ a99db2e)

| Đổi | Vì sao |
|---|---|
| `feat_extract.py`: đọc `log.csv` theo chunk, ghi một `all_features.csv` cho mọi enrollment (đếm = 0 nếu không có sự kiện) | Phần đếm không phụ thuộc split; bản chunk ra cùng số đếm |
| `preprocess.py --split-seed s`: chọn split; scaler fit trên train; sửa gender (`'male'/'female'`) và category (tra theo `course_id`); **bỏ log1p** (Q3); ghi `feat/s<s>/` | P6, Q4 |
| `main.py --seed s`: train / val / test của split; mỗi epoch chấm val, dừng theo đúng luật của tác giả `training_termination(…, 5)`, tối đa 81; khôi phục trọng số tốt nhất; seed cho numpy/random/TF; ghi dự đoán + AUC/AUPRC | P4, P7 |
| **`model.py` `batch_norm_layer` (lỗi mới tìm thấy)**: bản gốc tạo nhánh batch norm lúc train **bên ngoài** `tf.cond`, nên lần `sess.run` nào cũng cập nhật moving mean/var, kể cả lúc dự đoán. Hai lần dự đoán liền nhau trên cùng dữ liệu lệch nhau tới 0.026, và việc chấm val/test làm thay đổi mô hình. Sửa: tạo cả hai lệnh `batch_norm` bên trong hai nhánh của `tf.cond` | Sau khi sửa: số tham số không đổi (241,729); dự đoán giống hệt giữa các lần; chạy lại cùng seed ra cùng kết quả |

### 4.3 CA-TFHN (codeds27/CA-TFHN @ f8ef659)

| Đổi | Vì sao |
|---|---|
| Bước 0: đếm `<action>_<day>#num` bằng một lần groupby theo chunk, thay cho 770 lần groupby; ghi `all_features_35.csv` | Chỉ để chạy được; so với vòng lặp gốc trên 1,000 enrollment: **giống hệt** |
| Bước 1 `--split-seed`: sửa gender/category; thêm `cluster_label` (code gốc không tạo nhưng 3 bước sau đọc cột này; lấy từ file cluster như CFIN); scaler fit trên train; bỏ lệnh `print` từng khóa | Q4, P6, lỗi code gốc |
| Bước 3, link prediction, 2 bước dựng đồ thị: thêm validation; scaler người học / khóa học fit trên train; seed; sửa `'wb'` nằm trong `os.path.join`; thêm `enroll_id`, `val_mask` vào đồ thị | P6, lỗi code gốc |
| **`models.py:185` (lỗi mới)**: `if i == 1` trong vòng `for i in range(5)` → tuần 0 gọi `torch.cat((None, x))`, code gốc không chạy được. Sửa thành `i == 0` | Lỗi code gốc |
| `train.py`: mask train/val/test; dừng theo val AUC (patience 5, tối đa 45); seed; ghi dự đoán | P4, P7 |

Lưu ý môi trường: trên Windows, Git đổi xuống dòng trong file pickle `datastore/cluster/user_dict` (lỗi `numpy.core.multiarray\r`). Trên server Linux không bị; file sau khi bỏ CR giống hệt bản của CFIN.

### 4.4 MST-GCN (wudongze9/MST-GCN @ 7350a7e, `src_xuet`)

| Đổi | Vì sao |
|---|---|
| Đọc `log.csv` / `truth.csv`; lấy `username`, `course_id` từ `enrollments.csv` (bản gốc `dropna` làm rơi enrollment không có log) | Đủ enrollment |
| Cache đồ thị một lần với feature tổng hợp **chưa chuẩn hóa**; `apply_split()` fit scaler trên train của từng split | P6; đồ thị không phụ thuộc split |
| `trainer.py`: bỏ chia 80/20 theo thứ tự dòng; val AUC mỗi epoch, patience 5, tối đa 24; khôi phục trọng số tốt nhất; ghi dự đoán; bỏ các hình t-SNE / gate | P4 |

Giữ như gốc: ngày mở khóa suy từ sự kiện đầu tiên của khóa; 35 đếm theo ngày để thô; feature của khóa học và object được chuẩn hóa trên mọi khóa / object (thống kê theo khóa, không theo enrollment).

### 4.5 SIG-Net (Noverse0/SIG-Net @ 87bc320)

Code gốc chỉ có KDD Cup và NAVER. File mới `code/src/xuetangx_graph_builder.py` dịch `kddcup_graph` sang XuetangX theo từng bước; model RGCN + BiLSTM giữ nguyên, không sửa dòng nào.

| Thành phần | KDD (gốc) | XuetangX |
|---|---|---|
| Ngày | từ `date.csv` | từ `course_info.start` |
| Loại sự kiện | problem / video / discussion / khác | assignment / video / forum / web_page (nhóm action của M) |
| 3 khoảng | 3 × 10 ngày | 0–11, 12–23, 24–35 (ngày 35 luôn trống; model dùng chung lớp cho 3 đồ thị nên 3 khoảng phải cùng độ rộng) |
| Feature khóa học | số module theo loại (`object.csv`) | số object khác nhau theo loại, đếm từ log, + số enrollment |

Đổi có chủ đích: chuẩn hóa theo khóa fit trên train; cạnh sinh viên ở đồ thị 3 mang loại theo nhãn dropout (8/9) chỉ khi enrollment nguồn thuộc train, val/test mang loại 10 "chưa biết" (bản gốc gán theo nhãn của cả train lẫn test = rò rỉ nhãn).

Giữ như gốc (để em biết khi viết bài):
- `feature[ngày]` = click_sum của **dòng (loại sự kiện, object) cuối cùng** trong ngày đó theo thứ tự groupby. Vòng lặp gốc **ghi đè**, không cộng dồn cả ngày.
- Các dòng không có object bị groupby bỏ. Ở XuetangX, mọi action web_page và close_forum không có object, nên loại cạnh 3 và 7 trống.
- Cạnh từ chính enrollment đích được che trong `__getitem__`.

Kiểm tra: `run_signet.sh check` so bản vector hóa với bản dịch nguyên văn các vòng lặp (iterrows) trên mẫu 600 learner (1,774 enrollment): cạnh, loại cạnh, id cạnh sinh viên, feature, nhãn đều **giống hệt**. Bản nạp chỉ giữ dòng có object (bớt RAM) cũng cho đồ thị giống hệt. Trên toàn bộ dữ liệu: nạp 76 s (5.8 GB RAM), dựng 43 s; 298,775 node; khoảng 3.3 triệu cạnh mỗi đồ thị; một khóa có tới 17,257 object, nên đồ thị con của khóa này lớn (cần xem thời gian mỗi epoch ở bước smoke).

---

## 5. Cách chạy

```
python baseline/prepare_raw.py                      # một lần
python baseline/ML/LR.py ; python baseline/ML/SVM.py # local, ~2 phút mỗi mô hình

bash baseline/CFIN/run_cfin.sh      setup | features | smoke | full     # conda cfin (Python 3.7, TF 1.15)
bash baseline/CA-TFHN/run_catfhn.sh setup | features | smoke | full     # conda catfhn (torch 1.11 + PyG 2.1)
bash baseline/MST-GCN/run_mstgcn.sh setup | features | smoke | full     # conda graph (torch + DGL)
bash baseline/SIG-Net/run_signet.sh setup | check | smoke | full        # conda graph

python baseline/summary.py                          # bảng AUC / AUPRC mean ± std + kiểm tra file dự đoán
```

Kết quả ở `outputs/baselines/<TÊN>/`: `results.csv` (1 dòng / split) + `seed<s>_{validation,test}.csv` (`split, enroll_id, label, prob`). Thư mục `smoke/` riêng. Chạy `smoke` trước, nếu một epoch quá lâu thì dừng lại báo em.

Trên server: chép `data/processed/baseline_raw/` sang (hoặc chạy lại `prepare_raw.py`); LR/SVM cần thêm `data/processed/simple/{split}/X.npy` và `{split}.csv` của M.

---

## 6. Kiểm tra đã làm

| Mã | Kiểm tra | Kết quả |
|---|---|---|
| K1 | Không learner nào ở 2 phần | Đạt, 5 split |
| K3 | Tổng 225,642; mỗi file dự đoán test đúng tập test của split | Đạt cho LR, SVM (`summary.py` kiểm tra tự động) |
| K4 | t_obs | 40,558,640 sự kiện, bằng số của lần chạy trước theo luật của M |
| K5 | Cô lập nhãn: đổi ngẫu nhiên 842 nhãn val/test trên bộ mẫu | Feature CFIN, feature + mapping CA-TFHN (bước 0–3), đồ thị SIG-Net: **giống hệt**, chỉ cột nhãn val/test khác. MST-GCN: nhãn chỉ nằm ở `labels`, không vào feature |
| K6 | Chạy thử | LR/SVM chạy đủ. CFIN chạy thông trên bộ mẫu, tái lập được. Phần dữ liệu của MST-GCN chạy trên bộ mẫu (dùng DGL giả lập). CA-TFHN: link prediction / train cần `pyg-lib` hoặc `torch-sparse`, không có bản Windows, nên phải smoke trên server. MST-GCN và SIG-Net cũng phải train trên server (cần DGL) |

## 7. Kết quả

| Mô hình | Test AUC | Test AUPRC |
|---|---|---|
| LR | 0.8453 ± 0.0011 | 0.9245 ± 0.0015 |
| SVM | 0.8503 ± 0.0016 | 0.9298 ± 0.0019 |
| CFIN, CA-TFHN, MST-GCN, SIG-Net | chờ chạy | |

## 8. Quyết định của em (2026-10-13)

Q1 5 split theo learner · Q2 60/20/20 · Q3 giữ tiền xử lý gốc (bỏ log1p) · Q4 sửa lỗi đọc gender/category · Q5 giữ CFIN · Q6 chưa làm OULAD · M không đổi · mọi thứ trong `baseline/`.
