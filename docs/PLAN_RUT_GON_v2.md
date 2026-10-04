# Plan rút gọn project, bản 2 (để em kiểm tra, CHƯA thực hiện)

Ngày lập: 2026-10-04. Chưa có file code nào bị sửa và chưa có lệnh nào được chạy.

So với bản 1 (`PLAN_RUT_GON.md`), bản 2 **bỏ hẳn khái niệm kịch bản**. Project chỉ còn **một mô hình chính**, chạy theo **một giao thức**. Phần kịch bản (ablation, MLP, …) sẽ tính sau, khi project đã sạch.

---

## 1. Mục tiêu

Một project mà em đọc từ đầu đến cuối vẫn kiểm soát được từng dòng:

```
1_download → 2_preprocess → 3_features → 4_hypergraph → 6_train → 7_summary
                                                          ↑
                                                       5_model
```

- **Một mô hình:** HGNN 2 lớp + nhánh skip MLP + trọng số W theo loại hyperedge. Hyperedge gồm Course, Object, User (temporal) và self-loop. Có causal mask.
- **Một giao thức:**
  - train/validation = 80/20 từ train chính thức, test = test chính thức (67,699);
  - train cố định 1000 epoch, dùng model ở epoch cuối;
  - đánh giá validation và test **một lần** ở ngưỡng 0.5.
- **Một chỗ ghi kết quả:** `outputs/results.csv`, mỗi seed một dòng. Thêm `7_summary.py` để tổng hợp mean ± std ra CSV và Excel.
- **Bỏ hẳn:** checkpoint, early stopping, ngưỡng t\*, DeLong, vẽ hình, check_leakage, kịch bản, luật User "any", graph xáo trộn, Behavioral/kNN, phần HSL còn sót.

## 2. Cấu trúc sau khi rút gọn

```
src/
  0_config.py       đường dẫn, seed, bố cục cột X, loại hyperedge, siêu tham số train
  1_download.py     tải 3 file raw                                        (không đổi)
  2_preprocess.py   chia 80/20 + test chính thức → train/validation/test.csv  (không đổi logic)
  3_features.py     ma trận X mỗi split, fit trên train                   (sửa nhỏ)
  4_hypergraph.py   hypergraph train H0 + local graph cho val/test        (rút gọn mạnh)
  5_model.py        DropoutModel cố định: HGNN + skip + W, causal         (rút gọn)
  6_train.py        train 1000 epoch → val/test ở 0.5 → 1 dòng results.csv (viết lại)
  7_summary.py      results.csv → summary.csv + ket_qua.xlsx              (mới)
scripts/
  run_tmux.sh       chạy 6_train + 7_summary trong tmux trên server       (sửa)
docs/
  assets/           sơ đồ mô hình                                          (không đổi)
README.md           viết lại ngắn
requirements.txt    numpy, scikit-learn, torch, pandas, openpyxl
```

## 3. Bảng giữ / sửa / xóa

| File | Hiện tại (dòng) | Sau | Việc |
|---|---|---|---|
| src/0_config.py | 111 | ~90 | Sửa |
| src/1_download.py | 32 | 32 | Giữ |
| src/2_preprocess.py | 270 | 270 | Giữ (chỉ sửa comment đầu file) |
| src/3_features.py | 300 | ~280 | Sửa nhỏ |
| src/4_hypergraph.py | 527 | ~220 | Rút gọn |
| src/5_model.py | 126 | ~90 | Rút gọn |
| src/6_train.py | 410 | ~140 | Viết lại |
| src/7_summary.py | — | ~40 | Mới |
| scripts/run_tmux.sh | ~60 | ~30 | Sửa |
| scripts/analyze_contribution.py, check_leakage.py, collect_results.py, delong.py, export_excel.py, plot_results.py, summarize_results.py, scenarios.py, run_all.sh, run_integrity.sh, run_scenarios.sh | ~2,100 | 0 | **Xóa** |
| docs/TONG_QUAN_PROJECT.md | — | — | **Xóa** (lỗi thời) |
| docs/PLAN_RUT_GON.md (bản 1) | — | — | **Xóa** sau khi em duyệt bản 2 |
| README.md | — | ngắn | Viết lại |
| requirements.txt | — | — | Bỏ matplotlib |
| environment-gpu.yml | — | — | Thêm openpyxl |
| .gitignore | — | — | `outputs/runs`, `outputs/reports`, `result/` → `/outputs/*` |
| baseline/ (gitignored) | — | — | Không đụng |

Tổng code giảm từ khoảng 4,900 dòng (src + scripts) xuống khoảng 1,200 dòng. Mọi thứ bị xóa vẫn lấy lại được qua git tag `before-simple` (mục 7, bước 0).

## 4. Chi tiết từng file

### src/0_config.py
- Giữ: đường dẫn `ROOT/RAW/PROCESSED`, `SEEDS`, cấu hình tải và tiền xử lý, bố cục cột X (behavior, user, course).
- Đổi `EDGE_FAMILIES = ("course", "object", "user", "self_loop")`.
- Xóa: `RUNS`, `REPORTS`, `GRAPH_FAMILIES`, `USER_RULES`, `OBJECT`/`behavioral` ở các chỗ liên quan, và các hằng `*_CLI_DESCRIPTION` (đưa thẳng vào argparse).
- Thêm:
  ```python
  OUTPUTS = ROOT / "outputs"
  RESULTS_CSV = OUTPUTS / "results.csv"
  THRESHOLD = 0.5
  # Siêu tham số train (HGNN, Feng et al. 2019)
  TRAIN = {
      "hidden_dim": 128, "dropout": 0.5,
      "learning_rate": 1e-3, "weight_decay": 5e-4, "family_weight_lr": 0.05,
      "epochs": 1000, "eval_batch_size": 8,
  }
  ```

### src/2_preprocess.py
- Giữ nguyên logic:
  - xáo ngẫu nhiên các id train chính thức (seed 1), 80% làm train và 20% làm validation;
  - giữ nguyên test chính thức;
  - giữ cả enrollment không có sự kiện nào (feature hành vi bằng 0).

### src/3_features.py
- Xóa `feature_columns()`, vì luôn dùng toàn bộ X. Các hàm khác giữ nguyên.

### src/4_hypergraph.py
- **Xóa:**
  - kNN (`nearest_train_neighbors`), `resolve_device`, Behavioral, `k`/`k_max`, mọi `candidate_*` (HSL);
  - luật User "any" (chỉ còn temporal);
  - graph xáo trộn (`node_permutation`, `shuffle_seed`);
  - `select_families`, `family_ids` (luôn dùng đủ 3 loại);
  - các cờ CLI `--k`, `--k-max`, `--device`, `--neighbor-batch-size`, `--user-rule`, `--hypergraph-file`, `--reuse-neighbors`;
  - import torch.
- **Giữ** (đã rút gọn):
  - `user_hyperedges(train_nodes)`: một hyperedge cho mỗi enrollment, gồm enrollment đó và các enrollment của cùng người học ở khóa bắt đầu không muộn hơn;
  - `build_train_hyperedges`: Course, Object, User;
  - `add_self_loops`;
  - `load_train_graph(output_dir)` và `load_evaluation_split(output_dir, split_name)`;
  - `build_local_graph`: target nối với các thành viên train trong hyperedge Course, Object và User temporal của nó;
  - `merge_local_graphs`.
- Chỉ còn **một** bundle `hypergraph.npz` (bỏ `hypergraph_temporal.npz`). Lệnh build: `python src/4_hypergraph.py`, không cần GPU.

### src/5_model.py
- `prepare_graph(graph, device)`: **luôn** áp causal mask, không còn tham số `causal`.
- `DropoutModel(input_dim, hidden_dim, dropout)` có cấu trúc cố định:
  - nhánh graph: HGNN 2 lớp với W theo loại hyperedge;
  - nhánh skip: MLP(X);
  - classifier nhận `[z_graph ‖ z_self]`.
- Bỏ các công tắc `skip_connection` và `family_weights`, bỏ `logit_graph`/`logit_self`. `forward` trả về `logits`.
- Giữ `weight_summary()` để ghi `w_*` vào kết quả.

### src/6_train.py (viết lại)
```
python src/6_train.py --seeds 1 11 111 1111 11111            # chạy thật
python src/6_train.py --seeds 1 --epochs 10 --eval-limit 2000  # chạy thử nhanh
```
- CLI chỉ có `--seeds`, `--epochs` (ghi đè), `--eval-limit` (chỉ để thử), `--device` và `--tag` (nhãn ghi vào CSV, mặc định `main`).
- Mỗi seed đi qua 5 bước:
  1. đặt seed, load train graph;
  2. train full-batch BCE đủ `epochs`, cứ 50 epoch in loss và train AUC;
  3. dự đoán validation và test bằng local graph;
  4. tính `auc, auprc, accuracy, precision, recall, f1` tại 0.5;
  5. ghi thêm 1 dòng vào `outputs/results.csv`.
- Không lưu `.pt`, `.json` hay `.npz`.

### Cột của outputs/results.csv
| Nhóm | Cột |
|---|---|
| Nhận dạng | time, tag, seed |
| Siêu tham số | epochs, hidden_dim, dropout, learning_rate, weight_decay |
| Validation | val_auc, val_auprc, val_f1 |
| Test | test_auc, test_auprc, test_accuracy, test_precision, test_recall, test_f1 |
| W học được | w_course, w_object, w_user, w_self_loop |
| Thời gian | minutes |

### src/7_summary.py (mới)
- Đọc `results.csv`. Nếu cùng một (tag, seed) có nhiều dòng thì lấy dòng mới nhất.
- Gom theo `tag` để tính số seed, mean ± std của các chỉ số val và test, và mean của `w_*`.
- Ghi ra:
  - `outputs/summary.csv`;
  - `outputs/ket_qua.xlsx`, gồm sheet `Tong_hop` và sheet `Tung_seed`.

### scripts/run_tmux.sh
- Mở một phiên tmux chạy `python -u src/6_train.py "$@" && python src/7_summary.py`, ghi log vào `outputs/logs/<ngày_giờ>.log`.
- Ví dụ: `bash scripts/run_tmux.sh --seeds 1 11 111 1111 11111`

### README.md
Viết lại ngắn gọn, gồm:
- mô hình (kèm sơ đồ `docs/assets`);
- 7 bước;
- 6 lệnh chạy;
- giao thức;
- ý nghĩa các cột kết quả.

## 5. Giao thức (để ghi vào README và bài báo)

| Mục | Cách làm |
|---|---|
| Dữ liệu | XuetangX (CFIN), node = enrollment, feature = 35 ngày đầu của khóa |
| Chia | train chính thức → 80% train / 20% validation (seed 1); test chính thức giữ nguyên (67,699) |
| Fit tiền xử lý | chỉ trên train |
| Graph | train: H0 toàn bộ; val/test: mỗi target nối với các node train qua Course, Object, User temporal |
| Train | full-batch, BCE, Adam, 1000 epoch cố định, dùng model ở epoch cuối |
| Đánh giá | validation và test 1 lần, ngưỡng 0.5; AUC và AUPRC không phụ thuộc ngưỡng |
| Lặp | 5 seed (1, 11, 111, 1111, 11111), báo cáo mean ± std |

## 6. Dữ liệu phải build lại

**Local:** CSV và X.npy đã đúng. Chỉ cần build lại bundle:
```
python src/4_hypergraph.py
```
Sau đó xóa `data/processed/simple/hypergraph_temporal.npz` và `data/processed/simple/baselines/`.

**Server:** dữ liệu cũ thiếu 3,305 enrollment, nên test chỉ có 66,745. Làm như sau:
```
mv data/processed/simple data/processed/simple_old
mv result result_old;  rm -rf outputs/runs outputs/reports
python src/2_preprocess.py && python src/3_features.py && python src/4_hypergraph.py
```
Kết quả cũ không đưa vào bảng mới.

## 7. Thứ tự thực hiện (chỉ khi em nói "làm")

0. Commit trạng thái hiện tại và gắn tag `before-simple`.
1. Sửa `0_config.py`, `3_features.py`.
2. Rút gọn `4_hypergraph.py`, `5_model.py`.
3. Viết lại `6_train.py`, tạo `7_summary.py`.
4. Sửa `run_tmux.sh`; xóa các script, tài liệu và plan bản 1.
5. Sửa `README.md`, `requirements.txt`, `environment-gpu.yml`, `.gitignore`.
6. Kiểm tra theo mục 8.

## 8. Kiểm tra sau khi sửa (local, CPU)

1. Chạy `python -m py_compile src/*.py`.
2. Chạy `python src/4_hypergraph.py`. Số hyperedge phải giữ nguyên: Course 247, Object 21,747, User giống bundle temporal cũ.
3. Chạy `python src/6_train.py --seeds 1 --epochs 10 --eval-limit 2000`. `results.csv` phải có 1 dòng đủ cột, và không có file `.pt`, `.json` hay `.npz` nào được tạo.
4. So sánh với code cũ trên cùng seed và cùng 10 epoch (dùng tag `before-simple`): loss và AUC phải trùng nhau, để chắc việc rút gọn không làm đổi kết quả.
5. Chạy `python src/7_summary.py`. Mở được `summary.csv` và `ket_qua.xlsx`.
6. Tìm tham chiếu còn sót: `checkpoint|patience|threshold_|behavioral|k_max|candidate|delong|shuffle|user_rule|scenario|REPORTS|RUNS`.

## 9. Điểm em cần xác nhận

- [ ] Project chỉ còn 1 mô hình chính (HGNN + skip + W, Course + Object + User temporal, causal), không có công tắc bật/tắt.
- [ ] Bỏ luật User "any" và bundle `hypergraph_temporal.npz`; chỉ còn `hypergraph.npz` (temporal).
- [ ] Xóa 11 script trong `scripts/` (giữ `run_tmux.sh`), `docs/TONG_QUAN_PROJECT.md` và plan bản 1.
- [ ] Siêu tham số nằm trong `config.TRAIN`; CLI chỉ có seeds / epochs / eval-limit / device / tag.
- [ ] Cho phép commit và gắn tag `before-simple` trước khi sửa.
- [ ] Khi làm kịch bản sau này, các công tắc cần thiết sẽ được thêm lại từng cái một, theo yêu cầu của em.
