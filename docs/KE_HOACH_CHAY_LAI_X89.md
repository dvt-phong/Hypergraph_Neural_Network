# Kế hoạch chạy lại toàn bộ kịch bản với X 89 cột

Ngày lập: 2026-10-05. Code đã sửa, **chưa commit, chưa chạy trên server**. Thay cho `KE_HOACH_CHAY_LAI_X88.md` (đã bỏ).

**Đã kiểm tra trên máy (05/10, bản dựng trong thư mục tạm, không đụng `data/processed`):** K1–K5 ở mục 3 đều đạt;
cột one-hot trùng X cũ; hypergraph giống hệt bản cũ. Chạy thử `--scenario all --seeds 1 --epochs 10 --eval-limit 2000`:
cả 11 kịch bản chạy trọn, không lỗi, không NaN (test AUC 0.54–0.73 sau 10 epoch, chỉ để kiểm tra pipeline).

## 1. Đặc trưng mới (X = 89 cột)

| Khối | Cột | Nội dung | Cách xử lý | Nguồn |
|---|---|---|---|---|
| Hành vi | 0–34 | số sự kiện ở ngày d = 0…34 (d = ngày sự kiện − ngày bắt đầu khóa trong `course_info.csv`) | **để thô** | MST-GCN |
| | 35 | `total_events`: tổng sự kiện trong 35 ngày | z-score | MST-GCN (`total_actions`) |
| | 36 | `distinct_objects`: số object khác nhau đã dùng | z-score | MST-GCN (`unique_objects`) |
| | 37–58 | số lần của từng action trong **22 action** có trong log | z-score | MST-GCN (`pd.crosstab`) |
| Người học | 59–61 | gender: female, male, missing | one-hot | CFIN Definition 4 |
| | 62–69 | education: 7 mức + missing | one-hot | CFIN Definition 4 |
| | 70 | age = năm bắt đầu khóa − năm sinh; thiếu hoặc ngoài 10–70 → 0 | z-score | CFIN (`preprocess.py`) |
| Khóa học | 71–88 | category: 17 lĩnh vực + missing | one-hot | CFIN Definition 4 |

z-score: x = (c − μ_train) / σ_train, với μ, σ **chỉ tính trên train** (MST-GCN và CFIN fit trên train + test).
Không còn cột `_other` (luôn bằng 0 trong dữ liệu); giá trị ngoài từ điển làm `3_features.py` báo lỗi.

| Bộ feature | Kịch bản | Số cột |
|---|---|---|
| `feature` | F1 | 59 |
| `feature+user` | F2 | 71 |
| `feature+course` | F3 | 77 |
| `full` | M và các kịch bản còn lại | 89 |

**Không đổi:** cách chia dữ liệu (seed 1, 80/20, test chính thức 67,699), hypergraph H0, siêu tham số trong
`TRAIN`, danh sách 11 kịch bản, 5 seed, ngưỡng 0.5.

File code đã sửa: `src/0_config.py`, `src/2_preprocess.py` (thêm cột `birth`), `src/3_features.py`.
Chỉ đổi comment hoặc số cột: `src/5_graph_data.py`, `README.md`, `docs/KICH_BAN_THUC_NGHIEM.md`.

## 2. Kịch bản: 11 × 5 seed = 55 lần

| Mã | Đổi gì so với M | Số cột X |
|---|---|---|
| **M** | Course + Object + User + self-loop, `full`, HGNN 2 layer, có MLP, học W | 89 |
| A1 / A2 / A3 / A4 | bỏ Course / Object / User / self-loop | 89 |
| F1 / F2 / F3 | chỉ hành vi / hành vi + người học / hành vi + khóa học | 59 / 71 / 77 |
| L1 | HGNN 1 layer | 89 |
| B1 | bỏ nhánh MLP | 89 |
| W1 | W = I cố định | 89 |

Seed 1, 11, 111, 1111, 11111. X1 không chạy. Ước tính khoảng 4.7 phút mỗi lần, tổng **khoảng 4.3 giờ**.

## 3. Các bước trên server

**Bước 0.** Commit và push từ máy này (em quyết định lúc nào), rồi `git pull` trên server.

**Bước 1. Giữ kết quả cũ (X 90 cột).**

```bash
cd outputs && mv results.csv results_x90.csv && mv summary.csv summary_x90.csv && mv ket_qua.xlsx ket_qua_x90.xlsx && cd ..
```

**Bước 2. Dựng lại dữ liệu.** Phải xóa 3 file CSV cũ trước, vì `2_preprocess.py` bỏ qua bước này nếu chúng đã tồn tại.

```bash
rm data/processed/simple/{train,validation,test}.csv
.venv/bin/python src/2_preprocess.py
.venv/bin/python src/3_features.py      # in ra: Age: train mean≈7.559, std≈12.742, unknown≈71.6%
.venv/bin/python src/4_hypergraph.py    # course 247, object 21,747, user 32,797
```

**Bước 3. Kiểm tra.**

```bash
.venv/bin/python - <<'EOF'
import numpy as np, pandas as pd
P = "data/processed/simple/"
n = pd.read_csv(P + "feature_names.csv").feature_name
print("K1", len(n), "age at", list(n[n == "age"].index), "| _other:", n.str.endswith("_other").any())
x = np.load(P + "train/X.npy"); days = x[:, :35].sum(1)
z = (days - days.mean()) / days.std()
print("K2 total_events = Σ ngày (đã chuẩn hóa):", np.abs(z - x[:, 35]).max() < 1e-3)
print("K3 cột chuẩn hóa trên train: mean", np.abs(x[:, 35:59].mean(0)).max().round(4), "std", x[:, 35:59].std(0).round(3).min(), "-", x[:, 35:59].std(0).round(3).max())
print("K4 ngày để thô: min", x[:, :35].min(), "max", x[:, :35].max())
print("K5 age: tuổi không rõ", x[:, 70].min().round(3), "| tỉ lệ", np.mean(np.isclose(x[:, 70], x[:, 70].min())).round(3))
EOF
```

| # | Kết quả phải là |
|---|---|
| K1 | 89 cột; `age` ở cột 70; không có `_other` |
| K2 | `True` |
| K3 | mean ≈ 0; std = 1 (riêng `close_forum` có thể lệch vì chỉ vài enrollment khác 0) |
| K4 | min 0, max là số nguyên lớn (đo trên máy: 70,462) |
| K5 | tuổi không rõ ≈ −0.593, tỉ lệ ≈ 0.716 |

**Bước 4. Chạy thử nhanh, rồi xóa dòng chạy thử.**

```bash
bash scripts/run_tmux.sh --scenario all --seeds 1 --epochs 10 --eval-limit 2000
# xem log trong outputs/logs/, rồi:
rm outputs/results.csv outputs/summary.csv outputs/ket_qua.xlsx
```

**Bước 5a. Chạy M trước để xem kết quả sớm** (5 seed, khoảng 25 phút).

```bash
bash scripts/run_tmux.sh --scenario M --seeds 1 11 111 1111 11111
```

So `outputs/summary.csv` với M của lần X 90 cột (test AUC 0.8729 ± 0.0003, F1 0.9090). Nếu M thấp hơn rõ
(quá 0.0006) thì dừng lại xem log trước, chưa chạy tiếp.

**Bước 5b. Chạy 10 kịch bản còn lại** (khoảng 4 giờ). Dòng của M ở bước 5a vẫn được giữ trong `results.csv`.

```bash
bash scripts/run_tmux.sh --scenario A1 A2 A3 A4 F1 F2 F3 L1 B1 W1 --seeds 1 11 111 1111 11111
```

Muốn chạy một lần tất cả thì dùng `--scenario all` thay cho 5a + 5b.

Khi xong, tmux tự chạy `10_summary.py`, sinh ra `outputs/results.csv`, `summary.csv`, `ket_qua.xlsx`.
Chép 3 file này cùng `results_x90.csv` vào `docs/ket_qua/<ngày chạy>/`.

## 4. Đọc kết quả

- Bảng chính dùng kết quả X 89 cột lần này. Mọi Δ so với M của cùng lần chạy.
- So với lần X 90 cột: khác nhau ở nhiều chỗ cùng lúc (chuẩn hóa hành vi, thêm 2 cột tổng hợp, thêm age, bỏ `_other`),
  nên chỉ đọc được tổng ảnh hưởng của cả gói đặc trưng mới, không tách được từng phần.
- Quy tắc kết luận: |Δ| lớn hơn khoảng 2 lần std theo seed **và** cùng dấu ở 5/5 seed. Mô hình chính vẫn là M (`full`).

## 5. Lưu ý trước khi chạy

- Thử nhanh không dùng graph (MLP, 05/10) cho thấy chuẩn hóa kiểu MST-GCN có val AUC thấp hơn log1p + z-score khoảng 0.0125.
  Em chọn cách MST-GCN là lựa chọn tạm thời; nếu M giảm rõ so với lần X 90 thì nên xem lại điểm này đầu tiên.
- Số đếm theo ngày để thô có giá trị tới 70,462. Nếu loss bị NaN hoặc dao động mạnh trong log, đó là nguyên nhân đầu tiên cần kiểm tra.
