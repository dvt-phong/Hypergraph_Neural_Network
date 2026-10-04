# Plan rút gọn project (bản để em kiểm tra, CHƯA thực hiện)

Ngày lập: 2026-10-04. Chưa có file code nào bị sửa và chưa có lệnh nào được chạy.

---

## 1. Vì sao cần rút gọn

Project hiện có nhiều thứ nằm ngoài yêu cầu ban đầu:

- Checkpoint `.pt`, chọn epoch tốt nhất trên validation, early stopping (patience).
- Ngưỡng t\* chọn trên validation.
- DeLong, phân tích đóng góp (analyze_contribution).
- 12 script, 2 workbook Excel khác nhau, report JSON, file xác suất `.npz`, hình vẽ.
- Phần Behavioral/kNN và mảng ứng viên HSL còn sót lại trong `4_hypergraph.py`.

Mục tiêu là chỉ còn một đường chạy:

```
tải → tiền xử lý → feature → hypergraph → train 1000 epoch
    → đánh giá val + test ở ngưỡng 0.5 → ghi 1 dòng CSV → tổng hợp mean ± std (CSV + Excel)
```

## 2. Các quyết định em đã chọn

| Nội dung | Quyết định |
|---|---|
| Chia dữ liệu | train/validation = 80/20, lấy từ train chính thức (giữ code cũ); test = test chính thức (67,699) |
| Số epoch | cố định 1000, dùng model ở epoch cuối |
| Validation | chỉ đánh giá 1 lần ở cuối để báo cáo, không dùng để chọn gì |
| Ngưỡng phân lớp | 0.5 cố định |
| Checkpoint `.pt` | bỏ hẳn |
| DeLong, vẽ hình, check_leakage | bỏ |
| Giữ lại | file Excel tổng hợp, script tmux |
| Danh mục kịch bản | **quyết sau**; tạm giữ nguyên danh sách hiện tại |

## 3. So sánh giao thức với 3 bài tham khảo (đọc từ code gốc trong `baseline/`)

| | SIG-Net | MST-GCN (src_xuet) | CA-TFHN | Project (sau khi rút gọn) |
|---|---|---|---|---|
| Test | KDDCup chính thức; Naver chia ngẫu nhiên 60/40 | gộp train + test, cắt 80/20 theo thứ tự dòng | test chính thức | test chính thức 67,699 |
| Validation | không | không | không | 20% của train, chỉ để báo cáo |
| Epoch | cố định, lấy epoch cuối | cố định, lấy epoch cuối | in kết quả test mỗi epoch | cố định 1000, lấy epoch cuối |
| Ngưỡng | 0.5 | 0.5 | — | 0.5 |
| Checkpoint | không | không | không | không |

## 4. Cấu trúc trước và sau

### Trước (hiện tại)

| File | Dòng | Số phận |
|---|---|---|
| src/0_config.py | 111 | **Sửa** |
| src/1_download.py | 32 | Giữ nguyên |
| src/2_preprocess.py | 270 | Giữ logic, chỉ sửa comment đầu file |
| src/3_features.py | 300 | **Sửa nhỏ** |
| src/4_hypergraph.py | 527 | **Sửa lớn** (bỏ khoảng một nửa) |
| src/5_model.py | 126 | **Sửa nhỏ** |
| src/6_train.py | 410 | **Viết lại** (khoảng 150 dòng) |
| src/7_summary.py | — | **Tạo mới** (khoảng 40 dòng) |
| scripts/run_tmux.sh | ~60 | **Sửa** (gọi thẳng 6_train + 7_summary) |
| scripts/analyze_contribution.py | 129 | Xóa |
| scripts/check_leakage.py | 166 | Xóa |
| scripts/collect_results.py | 130 | Xóa |
| scripts/delong.py | 97 | Xóa |
| scripts/export_excel.py | 422 | Xóa |
| scripts/plot_results.py | 152 | Xóa |
| scripts/summarize_results.py | 511 | Xóa |
| scripts/scenarios.py | 135 | Xóa (danh mục kịch bản chuyển vào 0_config.py) |
| scripts/run_all.sh | ~200 | Xóa |
| scripts/run_integrity.sh | ~90 | Xóa |
| scripts/run_scenarios.sh | ~80 | Xóa |
| docs/TONG_QUAN_PROJECT.md | — | Xóa (lỗi thời, còn mô tả HSL / 8_train) |
| README.md | — | **Viết lại ngắn** |
| requirements.txt | — | Bỏ matplotlib |
| environment-gpu.yml | — | Thêm openpyxl |
| .gitignore | — | Gộp `outputs/runs`, `outputs/reports`, `result/` thành `/outputs/*` |
| docs/assets/, baseline/ | — | Không đụng |

Mọi file bị xóa vẫn lấy lại được qua git tag `before-simple` (mục 7, bước 0).

### Sau

```
src/0_config.py      đường dẫn, seed, cột feature, loại hyperedge, SCENARIOS, THRESHOLD
src/1_download.py    tải dữ liệu
src/2_preprocess.py  chia 80/20 + test chính thức, ghi 3 file CSV
src/3_features.py    ma trận X (fit trên train)
src/4_hypergraph.py  hyperedge Course / Object / User + local graph cho val/test
src/5_model.py       HGNN 2 lớp + skip MLP + W theo loại hyperedge
src/6_train.py       train → val/test ở 0.5 → ghi 1 dòng vào outputs/results.csv
src/7_summary.py     results.csv → outputs/summary.csv + outputs/ket_qua.xlsx
scripts/run_tmux.sh  chạy 6_train + 7_summary trong tmux trên server
```

## 5. Chi tiết từng file

### src/0_config.py
- Xóa `RUNS` và `REPORTS`.
- Thêm `OUTPUTS = ROOT / "outputs"`, `RESULTS_CSV = OUTPUTS / "results.csv"` và `THRESHOLD = 0.5`.
- Đổi `EDGE_FAMILIES = ("course", "object", "user", "self_loop")`: bỏ `behavioral`, vì không kịch bản nào dùng.
- Thêm `SCENARIOS`, chuyển từ `scripts/scenarios.py` sang dạng dict settings. Nội dung giữ nguyên:

| Mã | Mô tả | Settings |
|---|---|---|
| M-any | HGNN + skip + W; Course, Object, User any | course,object,user; hypergraph.npz; skip; W |
| M0 | Như M-any, User temporal + causal | course,object,user; hypergraph_temporal.npz; causal; skip; W |
| A1 | M0 bỏ User | course,object; skip; W |
| A2 | M0 bỏ Object | course,user; temporal; causal; skip; W |
| A3 | M0 bỏ Course | object,user; temporal; causal; skip; W |
| A4 | M0 bỏ W | course,object,user; temporal; causal; skip |
| A5 | M0 bỏ skip MLP | course,object,user; temporal; causal; W |
| C1 | A1 trên graph xáo trộn | course,object; shuffle_graph 7; skip; W |
| C2 | MLP (chỉ self-loop) | self_loop |
| B-HGNN | HGNN gốc, User any | course,object,user; hypergraph.npz |
| B-HGNN-T | HGNN gốc, User temporal + causal | course,object,user; temporal; causal |

Muốn bỏ kịch bản nào sau này thì chỉ cần xóa dòng tương ứng.

### src/2_preprocess.py
- Logic chia dữ liệu giữ nguyên:
  - xáo ngẫu nhiên các id train chính thức (seed 1), 80% thành train, 20% thành validation;
  - test chính thức giữ nguyên;
  - enrollment không có sự kiện nào vẫn được giữ lại.
- Chỉ sửa comment đầu file.

### src/3_features.py
- Xóa `feature_columns()`, vì luôn dùng toàn bộ X. Các hàm khác giữ nguyên.

### src/4_hypergraph.py
- **Xóa:**
  - `nearest_train_neighbors` (kNN);
  - `resolve_device` (chuyển sang 6_train);
  - hyperedge Behavioral, tham số `k` / `k_max`;
  - CLI `--k`, `--k-max`, `--device`, `--neighbor-batch-size`, `--reuse-neighbors`;
  - toàn bộ `candidate_*` (HSL) và phần ứng viên trong `select_families`;
  - các mảng `*_neighbors` trong bundle;
  - import torch.
- **Giữ:**
  - `user_hyperedges` (luật `any` / `temporal`);
  - `build_train_hyperedges` (Course, Object, User);
  - `add_self_loops`, `select_families`;
  - `load_train_graph`, `load_evaluation_split`, `build_local_graph`, `merge_local_graphs`;
  - `shuffle_seed` (cho C1).
- CLI chỉ còn `--user-rule {any,temporal}` và `--hypergraph-file`.
- Build bundle không cần GPU nữa. Bundle cũ không dùng được, phải build lại.

### src/5_model.py
- `forward` chỉ trả về tensor `logits`, bỏ `logit_graph` / `logit_self`.
- Giữ `prepare_graph` (có causal), `skip_connection`, `family_weights` và `weight_summary` (để ghi `w_*`).

### src/6_train.py (viết lại)

`DEFAULT_SETTINGS`:

| Tham số | Giá trị |
|---|---|
| families | course,object,user |
| hypergraph | hypergraph.npz |
| causal | False |
| shuffle_graph | None |
| hidden_dim | 128 |
| dropout | 0.5 |
| skip_connection | False |
| family_weights | False |
| learning_rate | 1e-3 |
| weight_decay | 5e-4 |
| family_weight_lr | 0.05 |
| epochs | 1000 |
| eval_batch_size | 8 |

Hàm `run(settings, scenario, seed)`:
1. Đặt seed, load train graph.
2. Train full-batch đủ `epochs`, cứ 50 epoch in loss và train AUC.
3. Load validation và test, dự đoán từng target trên local graph của nó.
4. Tính `auc, auprc, accuracy, precision, recall, f1` tại ngưỡng 0.5.
5. Ghi thêm 1 dòng vào `outputs/results.csv`.

Bỏ hẳn:
- checkpoint, early stopping, patience, `validation_limit`;
- `best_threshold`, report JSON, file xác suất `.npz`, branch metrics;
- `--mode`, `--checkpoint`, guard HSL.

Lệnh:
```
python src/6_train.py --scenario M0 C2 --seeds 1 11 111 1111 11111
python src/6_train.py --scenario M0 --seeds 1 --epochs 10 --eval-limit 2000          # chạy thử
python src/6_train.py --tag thu-nghiem --families course,object --skip-connection    # không theo kịch bản
```

### Cột của outputs/results.csv (mỗi lần chạy 1 seed = 1 dòng)

| Nhóm | Cột |
|---|---|
| Nhận dạng | time, scenario, seed |
| Tham số quan trọng | families, hypergraph, causal, skip_connection, family_weights, shuffle_graph, hidden_dim, dropout, learning_rate, weight_decay, epochs |
| Validation | val_auc, val_auprc, val_f1 |
| Test | test_auc, test_auprc, test_accuracy, test_precision, test_recall, test_f1 |
| Trọng số học được | w_course, w_object, w_user, w_self_loop |
| Thời gian | minutes |

### src/7_summary.py (mới)
- Đọc `results.csv`. Nếu cùng một (scenario, seed) có nhiều dòng thì lấy dòng mới nhất.
- Gom theo kịch bản: số seed, mean ± std của các chỉ số val và test, mean của `w_*`, mô tả lấy từ `SCENARIOS`.
- Ghi ra:
  - `outputs/summary.csv`;
  - `outputs/ket_qua.xlsx`, gồm sheet `Tong_hop` (mỗi kịch bản 1 dòng) và sheet `Tung_seed` (nguyên `results.csv`).

### scripts/run_tmux.sh
- Chạy `python -u src/6_train.py "$@" && python src/7_summary.py` trong tmux, ghi log vào `outputs/logs/<ngày_giờ>.log`.
- Ví dụ: `bash scripts/run_tmux.sh --scenario M0 C2 --seeds 1 11 111 1111 11111`

### README.md
Viết lại ngắn gọn, gồm:
- mô hình;
- 8 bước;
- lệnh chạy;
- giao thức (bảng ở mục 3);
- bảng kịch bản;
- ý nghĩa các cột kết quả.

## 6. Dữ liệu phải build lại

**Local:** CSV và X.npy đã đúng (test 67,699), chỉ cần build lại 2 bundle:
```
python src/4_hypergraph.py --user-rule any
python src/4_hypergraph.py --user-rule temporal --hypergraph-file hypergraph_temporal.npz
```

**Server:** dữ liệu cũ thiếu 3,305 enrollment (train thiếu 1,879, val thiếu 472, test thiếu 954), nên test chỉ có 66,745:
```
mv data/processed/simple data/processed/simple_old
mv result result_old;  rm -rf outputs/runs outputs/reports      # bỏ kết quả và .pt cũ
python src/2_preprocess.py && python src/3_features.py
python src/4_hypergraph.py --user-rule any
python src/4_hypergraph.py --user-rule temporal --hypergraph-file hypergraph_temporal.npz
```
Kết quả cũ (M0 0.8736, …) đo trên tập test thiếu, nên không đưa vào bảng mới.

## 7. Thứ tự thực hiện (chỉ khi em đồng ý)

0. Commit trạng thái hiện tại và gắn tag `before-simple`, để có thể lấy lại mọi thứ.
1. Sửa `0_config.py`, `3_features.py`, `5_model.py`.
2. Sửa `4_hypergraph.py`.
3. Viết lại `6_train.py`, tạo `7_summary.py`.
4. Sửa `run_tmux.sh`; xóa các script và tài liệu cũ.
5. Sửa `README.md`, `requirements.txt`, `environment-gpu.yml`, `.gitignore`.
6. Kiểm tra theo mục 8.

## 8. Kiểm tra sau khi sửa (local, CPU)

1. Chạy `python -m py_compile src/*.py`.
2. Build lại 2 bundle. Số hyperedge Course (247), Object (21,747) và User phải giữ nguyên như bản cũ.
3. Chạy `python src/6_train.py --scenario M0 C2 --seeds 1 --epochs 10 --eval-limit 2000`. Kết quả mong đợi: `results.csv` có 2 dòng đủ cột; không có file `.pt`, `.json` hay `.npz` nào được tạo.
4. Chạy `python src/7_summary.py`. Mở được `summary.csv` và `ket_qua.xlsx`.
5. Chạy thử C1 (graph xáo trộn) và A5 (không skip) với 10 epoch.
6. Tìm tham chiếu còn sót: `checkpoint|patience|validation_limit|best_threshold|behavioral|k_max|candidate|delong|REPORTS|RUNS`.

## 9. Điểm em cần xác nhận khi đọc plan

- [ ] Đồng ý xóa toàn bộ script ghi "Xóa" ở mục 4 (lấy lại được qua tag `before-simple`).
- [ ] Đồng ý xóa `docs/TONG_QUAN_PROJECT.md`.
- [ ] Đồng ý bỏ Behavioral/kNN (không kịch bản nào dùng).
- [ ] Đồng ý chuyển danh mục kịch bản vào `0_config.py`.
- [ ] Danh mục kịch bản: tạm giữ cả 11, quyết sau.
- [ ] Cho phép thầy commit và gắn tag `before-simple` trước khi sửa.
