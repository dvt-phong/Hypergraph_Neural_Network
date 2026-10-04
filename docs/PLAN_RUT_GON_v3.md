# Plan rút gọn project, bản 3 (để em kiểm tra, CHƯA thực hiện)

Ngày lập: 2026-10-04. Chưa có file code nào bị sửa và chưa chạy lệnh nào.

**Thay đổi so với bản 2:**
1. Mỗi file chỉ làm **một việc**. HGNN và MLP được tách thành hai file riêng. Phần dựng hypergraph (chạy một lần) tách khỏi phần đọc graph lúc train và đánh giá.
2. Có **quy ước comment công thức**: mỗi dòng tính toán chính đều có công thức toán ngay bên cạnh (mục 5).

Bản 3 thay thế bản 1 và bản 2. Hai bản cũ sẽ bị xóa khi em duyệt bản này.

---

## 1. Mục tiêu

- **Một mô hình:** HGNN 2 lớp (nhánh graph) + MLP (nhánh riêng của node) + lớp phân loại. Hyperedge gồm Course, Object, User (temporal) và self-loop, kèm W theo loại hyperedge và causal mask.
- **Một giao thức:** chia 80/20 từ train chính thức; test chính thức (67,699); 1000 epoch cố định; đánh giá validation và test một lần ở ngưỡng 0.5.
- **Một chỗ ghi kết quả:** `outputs/results.csv`. Bước tổng hợp xuất mean ± std ra CSV và Excel.
- **Không có:** kịch bản, công tắc bật/tắt, checkpoint, early stopping, ngưỡng t\*, DeLong, vẽ hình, check_leakage, luật User "any", graph xáo trộn, Behavioral/kNN, phần HSL còn sót.

## 2. Cấu trúc file mới: mỗi file một việc

```
src/
  0_config.py       Hằng số: đường dẫn, seed, bố cục cột X, loại hyperedge, siêu tham số
  1_download.py     Tải 3 file raw
  2_preprocess.py   Chia train/validation/test, cắt sự kiện ngày 0–34, ghi 3 CSV
  3_features.py     Ma trận đặc trưng X của mỗi split (fit trên train)
  4_hypergraph.py   DỰNG hypergraph train H0 (Course, Object, User) → hypergraph.npz  [chạy 1 lần]
  5_graph_data.py   ĐỌC H0 cho train; dựng local graph cho từng target val/test       [dùng lúc train]
  6_hgnn.py         Nhánh graph: ma trận liên thuộc, causal mask, lớp HGNN, encoder 2 lớp + W
  7_mlp.py          Nhánh MLP: encoder 2 lớp trên đặc trưng riêng của node
  8_model.py        Ghép 2 nhánh: [z_graph ‖ z_self] → logit
  9_train.py        Train 1000 epoch, đánh giá val/test ở 0.5, ghi 1 dòng results.csv
  10_summary.py     results.csv → summary.csv + ket_qua.xlsx (mean ± std)
scripts/
  run_tmux.sh       Chạy 9_train + 10_summary trong tmux trên server
docs/
  assets/           Sơ đồ mô hình
README.md
requirements.txt    numpy, scikit-learn, torch, pandas, openpyxl
```

Luồng phụ thuộc:

```
0_config ─┬─ 1_download
          ├─ 2_preprocess ─ 3_features ─ 4_hypergraph            (chuẩn bị dữ liệu, chạy 1 lần)
          └─ 5_graph_data ─┐
             6_hgnn ───────┼─ 8_model ─ 9_train ─ 10_summary     (mỗi lần train)
             7_mlp ────────┘
```

## 3. Bảng giữ / sửa / tách / xóa

| File hiện tại | Dòng | Thành | Việc |
|---|---|---|---|
| src/0_config.py | 111 | 0_config.py (~90) | Sửa |
| src/1_download.py | 32 | 1_download.py | Giữ |
| src/2_preprocess.py | 270 | 2_preprocess.py | Giữ logic, thêm comment công thức |
| src/3_features.py | 300 | 3_features.py (~280) | Bỏ `feature_columns`, thêm comment công thức |
| src/4_hypergraph.py | 527 | **4_hypergraph.py** (~110) + **5_graph_data.py** (~150) | Tách đôi và rút gọn |
| src/5_model.py | 126 | **6_hgnn.py** (~90) + **7_mlp.py** (~30) + **8_model.py** (~40) | Tách ba |
| src/6_train.py | 410 | **9_train.py** (~140) | Viết lại |
| — | — | **10_summary.py** (~40) | Mới |
| scripts/run_tmux.sh | ~60 | run_tmux.sh (~30) | Sửa |
| 11 script còn lại trong scripts/ | ~2,100 | — | **Xóa** |
| docs/TONG_QUAN_PROJECT.md, PLAN_RUT_GON.md, PLAN_RUT_GON_v2.md | — | — | **Xóa** |
| README.md | — | ngắn | Viết lại |
| requirements.txt / environment-gpu.yml | — | — | Bỏ matplotlib / thêm openpyxl |
| .gitignore | — | — | Gộp thành `/outputs/*` |

Các script bị xóa: analyze_contribution, check_leakage, collect_results, delong, export_excel, plot_results, summarize_results, scenarios, run_all, run_integrity, run_scenarios.

Mọi thứ bị xóa vẫn lấy lại được qua git tag `before-simple`.

## 4. Nội dung từng file

### 0_config.py
- Gồm: đường dẫn (`ROOT`, `RAW`, `PROCESSED`, `OUTPUTS`, `RESULTS_CSV`), `SEEDS`, cấu hình tải, `SPLIT_SEED = 1`, `TRAIN_RATIO = 0.8`, `OBSERVATION_DAYS = 35`, danh sách action, bố cục cột X, `EDGE_FAMILIES = ("course", "object", "user", "self_loop")`, `THRESHOLD = 0.5`.
- Siêu tham số:
  ```python
  TRAIN = {"hidden_dim": 128, "dropout": 0.5, "learning_rate": 1e-3,
           "weight_decay": 5e-4, "family_weight_lr": 0.05, "epochs": 1000, "eval_batch_size": 8}
  ```
- Xóa: `RUNS`, `REPORTS`, `GRAPH_FAMILIES`, `USER_RULES`, `behavioral`, các hằng `*_CLI_DESCRIPTION`.

### 1_download.py
Giữ nguyên.

### 2_preprocess.py
- Logic giữ nguyên:
  - xáo ngẫu nhiên các id train chính thức (seed 1): 80% làm train, 20% làm validation;
  - test chính thức giữ nguyên;
  - chỉ giữ sự kiện ngày 0–34;
  - enrollment không có sự kiện nào vẫn được giữ.

### 3_features.py
- Bỏ `feature_columns()`.
- Giữ: đếm hành vi, one-hot user/course, tuổi, chuẩn hóa fit trên train, ghi `X.npy` và `feature_names.csv`.

### 4_hypergraph.py: dựng H0 (chạy 1 lần)
- `user_hyperedges(train_nodes)`: luật temporal.
- `build_train_hyperedges(...)`: tạo Course, Object, User; bỏ các hyperedge chỉ có 1 thành viên.
- `build_hypergraph(output_dir)`: ghi `hypergraph.npz` gồm `node_ids`, `edge_ids`, `edge_family` và `edge_keys`.
- Lệnh: `python src/4_hypergraph.py`. Không có cờ, không cần GPU.
- Bỏ: kNN, Behavioral, `k`/`k_max`, ứng viên HSL, luật "any", các cờ CLI, import torch.

### 5_graph_data.py: đọc graph lúc train và đánh giá
- `load_train_graph(output_dir)`: trả về X train, nhãn train, và H0 đã thêm self-loop.
- `load_evaluation_split(output_dir, split_name)`: chuẩn bị mọi thứ để dựng local graph cho validation/test.
- `build_local_graph(split_data, target_id)`: target (node 0) nối với các node train cùng Course, cùng Object và cùng User (temporal).
- `merge_local_graphs(local_graphs)`: ghép nhiều local graph thành một batch, các graph không nối với nhau.
- `add_self_loops`, `start_day(s)`.
- Bỏ: `select_families`, `family_ids`, `node_permutation`, ứng viên HSL.

### 6_hgnn.py: nhánh graph (HGNN)
- `prepare_graph(graph, device)`: dựng ma trận thưa `H_T` (node → hyperedge) và `H_recv` (hyperedge → node, **luôn** có causal mask), cùng `edge_size` (De) và `edge_family`.
- `class HGNNEncoder(nn.Module)`:
  - `family_logits` (θ_f), `family_weights()` → w_f = softplus(θ_f);
  - `propagate(x, graph, w)`: một lần lan truyền node → hyperedge → node;
  - `forward(x, graph)`: 2 lớp HGNN → z_graph.
- Tham khảo: HGNN (Feng et al., AAAI 2019); HGNN+ (Gao et al., TPAMI 2023) cho W theo nhóm hyperedge.

### 7_mlp.py: nhánh MLP
- `class MLPEncoder(nn.Module)`: Linear → ReLU → Dropout → Linear → ReLU, chạy trên đặc trưng riêng của node. Có cùng kích thước ẩn với HGNN nhưng **không lan truyền** qua graph.
- Tham khảo: ý tưởng initial residual / skip trong UniGNN (Huang & Yang, IJCAI 2021).

### 8_model.py: ghép hai nhánh
- `class DropoutModel(nn.Module)`, gồm `hgnn = HGNNEncoder`, `mlp = MLPEncoder` và `classifier = Linear(2·hidden, 1)`.
- `forward(x, graph)` trả về logit:
  - z_graph = HGNN(X, H);
  - z_self = MLP(X);
  - logit = wᵀ [z_graph ‖ z_self] + b.
- `weight_summary()` trả về `w_course`, `w_object`, `w_user`, `w_self_loop`.

### 9_train.py: train và đánh giá
```
python src/9_train.py --seeds 1 11 111 1111 11111              # chạy thật
python src/9_train.py --seeds 1 --epochs 10 --eval-limit 2000  # chạy thử
```
- CLI: `--seeds`, `--epochs`, `--eval-limit` (chỉ để thử), `--device`, `--tag` (mặc định `main`).
- Các hàm:
  - `set_seed`;
  - `metrics(labels, probs)`: tính AUC, AUPRC, accuracy, precision, recall, F1 tại 0.5;
  - `predict(model, split_data)`: dự đoán bằng local graph, theo batch;
  - `append_result(row)`: ghi thêm 1 dòng vào CSV;
  - `run(seed)`.
- `run(seed)` làm lần lượt:
  1. đặt seed và load H0;
  2. train full-batch BCE với Adam đủ `epochs` (W có lr riêng, không có weight decay); cứ 50 epoch in loss và train AUC;
  3. dự đoán validation và test;
  4. ghi 1 dòng kết quả.
- Không lưu `.pt`, `.json` hay `.npz`.

Cột của `outputs/results.csv`:

| Nhóm | Cột |
|---|---|
| Nhận dạng | time, tag, seed |
| Siêu tham số | epochs, hidden_dim, dropout, learning_rate, weight_decay |
| Validation | val_auc, val_auprc, val_f1 |
| Test | test_auc, test_auprc, test_accuracy, test_precision, test_recall, test_f1 |
| W học được | w_course, w_object, w_user, w_self_loop |
| Thời gian | minutes |

### 10_summary.py: tổng hợp
- Đọc `results.csv`. Với mỗi cặp (tag, seed), chỉ lấy dòng mới nhất.
- Gom theo tag: số seed, mean ± std các chỉ số val và test, mean của `w_*`.
- Ghi ra `outputs/summary.csv` và `outputs/ket_qua.xlsx` (2 sheet: `Tong_hop`, `Tung_seed`).

### scripts/run_tmux.sh
- Chạy `python -u src/9_train.py "$@" && python src/10_summary.py` trong tmux, ghi log vào `outputs/logs/<ngày_giờ>.log`.

## 5. Quy ước comment công thức

**Quy tắc:**
- Đầu mỗi file: 1–3 dòng nói file làm gì, kèm tên bài báo và link tham khảo.
- Đầu mỗi hàm: Input / Output, như hiện tại.
- **Mỗi dòng tính toán chính:** một comment ghi công thức toán ở cuối dòng hoặc ngay trên dòng. Ký hiệu dùng thống nhất trong mọi file, theo bảng dưới. Không comment những dòng hiển nhiên (đọc file, ghi file, log).

**Bảng ký hiệu dùng chung** (ghi ở đầu `6_hgnn.py` và trong README):

| Ký hiệu | Nghĩa |
|---|---|
| N, E | số node (enrollment), số hyperedge |
| X ∈ ℝ^{N×D} | ma trận đặc trưng |
| H ∈ {0,1}^{N×E} | ma trận liên thuộc: h(v,e) = 1 nếu node v thuộc hyperedge e |
| W = diag(w_e) | trọng số hyperedge; w_e = w_{f(e)} = softplus(θ_{f(e)}), với f(e) là loại của e |
| De = diag(δ(e)) | δ(e) = \|e\|, số thành viên của e |
| Dv = diag(d(v)) | d(v) = Σ_e h_recv(v,e) · w_e |
| s(v) | ngày bắt đầu khóa học của enrollment v |
| Θ^{(l)} | trọng số lớp l |

**Các công thức sẽ được comment trong code:**

| File | Dòng code | Comment công thức |
|---|---|---|
| 2_preprocess | tính ngày trong khóa | `d = date(event) − date(course_start)`; giữ nếu `0 ≤ d < 35` |
| 2_preprocess | chia train/val | `n_train = ⌊0.8 · n⌋` sau khi xáo với seed 1 |
| 3_features | đếm hành vi | `c_v[d] = #sự kiện của v ở ngày d`; `c_v[a] = #lần action a` |
| 3_features | chuẩn hóa hành vi | `x = (log(1 + c) − μ_train) / σ_train` |
| 3_features | tuổi | `age = năm(course_start) − birth`, chỉ hợp lệ khi `10 ≤ age ≤ 100`; thiếu thì điền median_train và bật cờ `age_missing = 1`; rồi `(age − μ_train) / σ_train` |
| 4_hypergraph | User temporal | `e_U(a) = {a} ∪ {m : user(m) = user(a), s(m) ≤ s(a)}` |
| 4_hypergraph | Course / Object | `e_C(c) = {v : course(v) = c}`; `e_O(o) = {v : v dùng object o}` |
| 5_graph_data | local graph | `e ∩ V_train ∪ {target}` cho mỗi hyperedge e mà target thuộc |
| 6_hgnn | causal mask | `h_recv(v,e) = h(v,e) · 1[s(v) ≥ max_{u∈e} s(u)]` |
| 6_hgnn | W theo loại | `w_e = softplus(θ_{f(e)})`, khởi tạo θ sao cho `w = 1` |
| 6_hgnn | bậc node | `d(v) = Σ_e h_recv(v,e) · w_e` |
| 6_hgnn | node → hyperedge | `m_e = (w_e / δ(e)) · Σ_{u∈e} d(u)^{-1/2} · x_u` |
| 6_hgnn | hyperedge → node | `x'_v = d(v)^{-1/2} · Σ_e h_recv(v,e) · m_e` |
| 6_hgnn | dạng ma trận | `X' = Dv^{-1/2} H_recv W De^{-1} Hᵀ Dv^{-1/2} X Θ` (Feng et al., 2019, Eq. 10) |
| 6_hgnn | 2 lớp | `Z1 = ReLU(P(X Θ1))`; `Z_graph = ReLU(P(Dropout(Z1) Θ2))` |
| 7_mlp | MLP | `z_self = ReLU(W2 · Dropout(ReLU(W1 x + b1)) + b2)` |
| 8_model | phân loại | `logit = wᵀ [z_graph ‖ z_self] + b`; `p = σ(logit)` |
| 9_train | loss | `L = −(1/N) Σ [y log p + (1 − y) log(1 − p)]` (BCE) |
| 9_train | tối ưu | Adam; `L2 = 5e-4` cho Θ; W dùng lr 0.05, không L2; clip `‖g‖ ≤ 5` |
| 9_train | dự đoán | `ŷ = 1[p ≥ 0.5]` |
| 9_train | chỉ số | `Precision = TP/(TP+FP)`; `Recall = TP/(TP+FN)`; `F1 = 2PR/(P+R)`; `Acc = (TP+TN)/N`; `AUC = P(p_pos > p_neg)`; `AUPRC = Σ_n (R_n − R_{n−1}) P_n` |
| 10_summary | tổng hợp | `mean = (1/n) Σ x_i`; `std = √(Σ (x_i − mean)² / (n − 1))` |

**Ví dụ comment trong `6_hgnn.py`:**
```python
def propagate(self, x, graph, w):
    node_degree = torch.sparse.mm(graph["H_recv"], w[:, None])      # d(v) = Σ_e h_recv(v,e)·w_e
    node_scale = node_degree.clamp_min(1e-6).rsqrt()                  # d(v)^(-1/2)
    edge_x = torch.sparse.mm(graph["H_T"], x * node_scale)            # Σ_{u∈e} d(u)^(-1/2)·x_u
    edge_x = edge_x * (w / graph["edge_size"])[:, None]               # m_e = (w_e/δ(e))·Σ ...
    return torch.sparse.mm(graph["H_recv"], edge_x) * node_scale      # x'_v = d(v)^(-1/2)·Σ_e h_recv(v,e)·m_e
```

## 6. Giao thức (để ghi vào README và bài báo)

| Mục | Cách làm |
|---|---|
| Dữ liệu | XuetangX (CFIN, Feng et al., 2019); node = enrollment; đặc trưng = 35 ngày đầu của khóa |
| Chia | train chính thức → 80% train / 20% validation (seed 1); test chính thức giữ nguyên (67,699) |
| Fit tiền xử lý | chỉ trên train |
| Graph | train: toàn bộ H0; val/test: mỗi target nối với node train qua Course, Object, User temporal |
| Train | full-batch, BCE, Adam, 1000 epoch cố định, dùng model ở epoch cuối |
| Đánh giá | validation và test một lần, ngưỡng 0.5; AUC và AUPRC không phụ thuộc ngưỡng |
| Lặp | 5 seed (1, 11, 111, 1111, 11111), báo cáo mean ± std |

## 7. Dữ liệu phải build lại

**Local:** CSV và X.npy đã đúng. Chỉ cần chạy `python src/4_hypergraph.py`, rồi xóa `hypergraph_temporal.npz` và `data/processed/simple/baselines/`.

**Server:** dữ liệu cũ thiếu 3,305 enrollment, nên test chỉ có 66,745. Chạy:
```
mv data/processed/simple data/processed/simple_old
mv result result_old;  rm -rf outputs/runs outputs/reports
python src/2_preprocess.py && python src/3_features.py && python src/4_hypergraph.py
```

## 8. Thứ tự thực hiện (chỉ khi em nói "làm")

0. Commit trạng thái hiện tại và gắn tag `before-simple`.
1. `0_config.py`.
2. `2_preprocess.py`, `3_features.py`: chỉ thêm comment công thức (và bỏ `feature_columns`).
3. Tách `4_hypergraph.py` thành `4_hypergraph.py` + `5_graph_data.py`.
4. Tách `5_model.py` thành `6_hgnn.py` + `7_mlp.py` + `8_model.py`.
5. Viết `9_train.py` và `10_summary.py`; xóa `src/5_model.py` và `src/6_train.py` cũ.
6. Sửa `run_tmux.sh`; xóa các script, tài liệu và plan cũ.
7. Sửa `README.md` (thêm bảng ký hiệu), `requirements.txt`, `environment-gpu.yml`, `.gitignore`.
8. Kiểm tra theo mục 9.

## 9. Kiểm tra sau khi sửa (local, CPU)

1. `python -m py_compile src/*.py`.
2. `python src/4_hypergraph.py`: số hyperedge Course (247), Object (21,747) và User phải trùng với bundle temporal cũ.
3. **Đối chiếu với code cũ:** cùng seed 1, cùng 10 epoch, cùng 2,000 target. Loss, train AUC và test AUC phải trùng với code ở tag `before-simple`. Điều này chứng minh việc tách file không làm đổi phép tính.
4. `python src/9_train.py --seeds 1 --epochs 10 --eval-limit 2000`: `results.csv` có 1 dòng đủ cột; không có file `.pt`, `.json` hay `.npz` nào được tạo.
5. `python src/10_summary.py`: mở được `summary.csv` và `ket_qua.xlsx`.
6. Tìm tham chiếu còn sót: `checkpoint|patience|behavioral|k_max|candidate|delong|shuffle|user_rule|scenario|skip_connection|REPORTS|RUNS`.

## 10. Điểm em cần xác nhận

- [ ] Cấu trúc 11 file trong `src/` như mục 2, với HGNN (`6_hgnn.py`) và MLP (`7_mlp.py`) tách riêng.
- [ ] Tách `4_hypergraph.py` (dựng, chạy 1 lần) khỏi `5_graph_data.py` (đọc lúc train).
- [ ] Quy ước comment công thức và bảng ký hiệu như mục 5.
- [ ] Một mô hình chính, không có công tắc; chỉ còn bundle `hypergraph.npz` (User temporal).
- [ ] Xóa 11 script trong `scripts/` (giữ `run_tmux.sh`), `docs/TONG_QUAN_PROJECT.md` và plan bản 1, bản 2.
- [ ] Cho phép commit và gắn tag `before-simple` trước khi sửa.
