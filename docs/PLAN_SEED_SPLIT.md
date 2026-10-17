# Plan: mỗi seed chia lại cả train / val / test, 60 / 10 / 30, phân tầng (XuetangX và OULAD)

Ngày: 2026-10-17 (bản 3, thay bản 2 "test = official test"). Em đã chốt: seed chia lại cả test; tỉ lệ 60 / 10 / 30; phân tầng theo nhãn; OULAD giống XuetangX. Đã code XuetangX (OULAD làm theo PLAN_OULAD_X60).

## 1. Quy tắc (như `src/dpscl/splits.py` của DP-SCL)

```
mọi enrollment có nhãn (XuetangX: official train + official test gộp lại)
  ── seed s, phân tầng theo y ──► test 30%
  70% còn lại ── seed s, phân tầng ──► val 10% (= 1/7 của phần còn lại) | train 60%
```

```python
rest, test = train_test_split(ids, test_size=TEST_RATIO, stratify=y, random_state=seed)
train, validation = train_test_split(rest, test_size=VALIDATION_RATIO / (TRAIN_RATIO + VALIDATION_RATIO),
                                     stratify=y[rest], random_state=seed)
```

| Dataset | Train (60%) | Val (10%) | Test (30%) |
|---|---|---|---|
| XuetangX (225,642) | 135,384 | 22,565 | 67,693 |
| OULAD (32,593) | 19,555 | 3,260 | 9,778 |

Phân tầng: chia riêng nhóm y = 1 và y = 0 theo cùng tỉ lệ, nên tỉ lệ dropout bằng nhau ở 3 tập.

Seed s quyết định: (1) enrollment nào vào train / val / **test**; (2) μ, σ của X (train của seed); (3) khởi tạo mô hình. H không đổi (dựng trên mọi enrollment).

Official split của XuetangX không còn dùng để chia; chỉ dùng để lấy nhãn (train_truth.csv + test_truth.csv).

## 2. Code

| File | Việc |
|---|---|
| `src/0_config.py` | `TRAIN_RATIO = 0.80` → `TRAIN_RATIO = 0.60`, `VALIDATION_RATIO = 0.10`, `TEST_RATIO = 0.30` |
| `src/2_preprocess.py` | `split_enrollments(node_ids, labels, seed)` theo mục 1 (gộp hai file truth); giữ nguyên phần em đã xoá ở đầu file |
| `src/3_features.py` → `src/10_summary.py` | không sửa (đọc split theo seed) |
| Baseline, `baseline/summary.py` | không sửa: LR, SVM, CFIN, MST-GCN, CA-TFHN đều đọc `split` từ `x59_s{seed}.npz`; `summary.py` kiểm test theo npz của từng seed |
| `README.md` | dòng "Test", "Lặp", kích thước split |
| `docs/PLAN_OULAD_X60.md` | mục split theo quy tắc này (đã sửa) |

## 3. Chạy lại

| Bước | Làm gì |
|---|---|
| Dữ liệu | xoá `data/processed/simple/splits/` và `x59_s*.npz` (giữ `events.csv`, `hypergraph.npz`); chạy `src/2_preprocess.py`, `src/3_features.py` |
| Kết quả cũ | xoá `outputs/` |
| M, LR, SVM, CFIN, MST-GCN | chạy lại (patch không đổi, MST-GCN giữ cache) |
| CA-TFHN | xoá `datastore/s*/`, chạy lại `features` rồi `full` |

## 4. Kiểm tra

1. Mỗi seed: 3 tập rời nhau, hợp lại đủ 225,642; kích thước đúng bảng mục 1.
2. Tỉ lệ dropout ở 3 tập của mỗi seed chênh nhau < 0.0001.
3. Test của hai seed khác nhau; phần chung kỳ vọng ≈ 30% của test (≈ 20,300).
4. `events.csv`, `hypergraph.npz` không đổi; số đếm X59 trước khi scale không đổi.
5. Smoke M 10 epoch seed 1; `baseline/summary.py` kiểm file test của baseline khớp npz.

## 5. Hệ quả cần biết

- Không còn dùng official test của XuetangX; kết quả không so trực tiếp với số trong bài gốc (vốn đã không so được vì baseline chạy trên X59).
- Std lớn hơn hiện nay (ước lượng 0.001–0.0015 thay vì 0.0003) vì có thêm dao động của tập test.
- Cấu hình `TRAIN` của M (chọn trên split cũ) vẫn dùng; kích thước train / val gần như cũ.

## 6. Ghi trong bài

*"For each of five seeds (1, 11, 111, 1111, 11111), all labeled enrollments are split into training, validation and test sets (60/10/30, stratified by label). The seed also sets the feature scaling statistics (training rows only) and the model initialization. All models use the same split of each seed; results are mean ± std over the five seeds. OULAD is split in the same way."*
