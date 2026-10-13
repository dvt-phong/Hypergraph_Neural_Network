# Kế hoạch CFIN (XuetangX)

Ngày lập: 2026-10-11. **Chưa code.** Em duyệt và trả lời mục 3 trước khi thầy làm.

CFIN = Feng et al., AAAI 2019, code gốc `wzfhaha/dropout_prediction` commit a99db2e (có sẵn trong `baseline/CFIN/dropout_prediction`, kèm `cfin_changes.patch` đã commit ở 8db47ef).

Nguyên tắc đã chốt: CFIN nhận **dữ liệu thô** (cùng phạm vi với M), **tự trích feature bằng code của nó**. Mình chỉ sửa 3 thứ: split, cách chọn epoch, và lỗi.

---

## 1. Trạng thái hiện tại (thầy đã đọc lại toàn bộ code)

**Đã ổn:** code gốc chạy được (Python 3.7, TF 1.15, có `.venv` local); `feat_extract.py` đọc theo chunk và ra file giống hệt bản gốc; tập test 67,699 trùng test của M.

**Chưa đạt:**

| # | Vấn đề | Ở đâu |
|---|---|---|
| 1 | Train trên 157,943 enrollment = train + val của M | `feat_extract.py` dùng nguyên file train chính thức |
| 2 | Không có val: chấm test mỗi epoch, epoch 27 cố định | `main.py:135` |
| 3 | Chỉ 1 seed (12345) | `main.py:161` |
| 4 | Scaler fit trên train + test | `preprocess.py:88-89` |
| 5 | Đọc mọi ngày trong log (kể cả trước ngày mở khóa và ngày 35–36); M chỉ đọc ngày 0–34 | `feat_extract.py` không lọc |
| 6 | gender = 0 và course_category = 0 với **mọi** enrollment | `preprocess.py:26` (so `'m'/'f'`, dữ liệu ghi `'male'/'female'`); `preprocess.py:66` (tra `course_info` theo cột `id` thay vì `course_id`) |
| **7 (mới)** | **Không đặt seed cho numpy.** Hầu hết trọng số khởi tạo bằng `np.random.normal` (`model.py:164-226`) và thứ tự batch xáo bằng `np.random.shuffle` (`model.py:239-252`), nhưng code chỉ đặt `tf.set_random_seed` (`model.py:62`). Hai lần chạy cùng seed 12345 vẫn ra kết quả khác nhau → số 0.8569 cũ không chạy lại đúng được | `model.py`, `main.py` |
| **8 (mới)** | **Không lưu trọng số tốt nhất.** `model.fit` luôn lưu trọng số của epoch **cuối** (`model.py:330`). Nhánh early stopping có sẵn (`model.py:297-300`) chỉ trả về số thứ tự epoch tốt nhất, không khôi phục lại trọng số của epoch đó | `model.py` |

---

## 2. Cách làm

### Bước 1 — Dữ liệu thô chung (B0)

**File mới:** `baseline/prepare_raw.py` (dùng chung cho cả 4 baseline học sâu sau này).

- Đọc `data/raw/xuetangx/prediction_data.tar.gz`. Lọc **đúng luật của M**: 22 action, `0 ≤ ngày(time) − ngày(course start) < 35`. Chia 3 split **bằng chính hàm** `split_train_enrollments` của `src/2_preprocess.py` (import, không viết lại).
- Ghi `data/processed/baseline_raw/`:
  - `prediction_log/{train,validation,test}_log.csv`: cột gốc `enroll_id, username, course_id, session_id, action, object, time`.
  - `prediction_log/{train,validation,test}_truth.csv`.
  - `prediction_log/enrollments.csv`: `enroll_id, username, course_id, split` của cả 225,642 enrollment, kể cả 3,305 enrollment không có sự kiện nào trong ngày 0–34.
  - `user_info.csv`, `course_info.csv`: chép nguyên bản.

**Kiểm tra:**
- Số enrollment 126,354 / 31,589 / 67,699, tập enroll_id trùng `data/processed/simple/{split}.csv`.
- Số sự kiện từng split bằng số dòng sự kiện trong file của M.
- In số sự kiện bị loại theo lý do.

### Bước 2 — Sửa CFIN (`cfin_changes.patch` bản 2, thay bản log1p cũ)

| File | Sửa | Giải quyết |
|---|---|---|
| `feat_extract.py` | Đọc 3 log + 3 truth từ `baseline_raw` (đường dẫn qua biến môi trường `CFIN_RAW`). Danh sách enrollment lấy từ truth; enrollment không có log thì đếm = 0, username/course_id lấy từ `enrollments.csv`. Ghi 3 file feature. Phần đếm giữ nguyên | #1, #5 |
| `preprocess.py` | Đọc 3 file; gender `'male'`→1, `'female'`→2; tra category theo `course_id`; **bỏ log1p**; `StandardScaler` fit trên train, transform cả 3 split. Mọi thứ khác giữ nguyên (age = 2018 − birth, enroll_num, cluster) | #4, #6 |
| `main.py` | (a) Đọc 3 split; `feat_augment` và `dataparse` gộp cả 3 (không dùng nhãn, như gốc). (b) Bỏ đoạn chia ngẫu nhiên 80/20 trong train; val = val của M. (c) Vòng huấn luyện: mỗi epoch gọi `model.fit(..., max_epoch=1)` trên train → chấm val AUC → nếu tốt hơn thì lưu bằng `model.saver` → dừng theo **chính luật của tác giả** `model.training_termination(valid_result, 5)`, tối đa 81 epoch → khôi phục trọng số tốt nhất → dự đoán val và test **một lần**. (d) `--seed`: đặt `np.random.seed(seed)`, `random.seed(seed)`, và `random_seed=seed` cho TF. (e) Ghi file dự đoán + `results.csv`, cùng cột và cùng hàm tính chỉ số với LR/SVM (ngưỡng 0.5) | #2, #3, #7, #8 |
| `model.py` | **Không sửa thêm** (chỉ giữ chỗ sửa lỗi `sess` → `self.sess` đã có) | — |

**Giữ nguyên:** mọi siêu tham số (`embedding 32, attn 16, conv 512, context 32, deep [256], dropout 0.9, lr 1e-4, batch 32, l2 1e-5, batch_norm`), `feat_augment` (mean/max theo user và course), file `cluster`, kiến trúc.

### Bước 3 — `run_cfin.sh`

Đổi thành 3 lệnh con:

```
bash baseline/CFIN/run_cfin.sh features   # feat_extract + preprocess
bash baseline/CFIN/run_cfin.sh smoke      # 1 epoch trên 4,000 dòng
bash baseline/CFIN/run_cfin.sh full       # 5 seed: 1 11 111 1111 11111
```

Kết quả ghi vào `outputs/baselines/CFIN/`: `results.csv` + `seed<s>_{validation,test}.csv` + log.

### Bước 4 — Kiểm tra

| Mã | Kiểm tra | Đạt khi |
|---|---|---|
| K1 | Số dòng 3 file feature | 126,354 / 31,589 / 67,699 |
| K2 | Phân bố gender, course_category | Không còn toàn 0 (gender khoảng 42% có giá trị, theo số liệu thiếu đã đếm) |
| K3 | **Cô lập nhãn:** đổi ngẫu nhiên nhãn val/test trong `baseline_raw`, chạy lại `feat_extract` + `preprocess` | Mọi cột feature giống hệt, chỉ cột `truth` khác |
| K4 | Seed chạy lại được: chạy smoke 2 lần cùng seed | Hai lần ra cùng val AUC |
| K5 | Smoke 1 epoch | Không lỗi, biết thời gian mỗi epoch |
| K6 | Epoch tốt nhất | Nhỏ hơn 81 |

### Bước 5 — Chạy thật

- **Thời gian ước tính:** lần trước khoảng 105 s/epoch trên CPU server với 157,943 dòng train. Giờ train còn 126,354 dòng nên khoảng 85–100 s/epoch. Nếu dừng ở epoch 15–40 thì mỗi seed mất 0.5–1.2 giờ, **5 seed mất 3–6 giờ**.
- Chạy được ở máy em (`.venv` Python 3.7 đã có) hoặc trên server.

---

## 3. Em cần trả lời

1. **Log1p:** bỏ log1p, dùng z-score như code gốc (thầy đề xuất, vì đúng nguyên tắc "xử lý là của nó"), hay giữ log1p như bản đã chạy?
2. **Sửa lỗi gender/category:** sửa (thầy đề xuất), hay giữ nguyên như bản gốc?
3. **File `cluster`** do tác giả phát hành sẵn: giữ (thầy đề xuất) hay bỏ?
4. **Chạy ở đâu:** máy em hay server?

Trả lời xong, em bảo **"tiến hành code"** thì thầy làm Bước 1.
