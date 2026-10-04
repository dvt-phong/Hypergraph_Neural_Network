# Plan rút gọn project, bản 5 (để em kiểm tra, CHƯA thực hiện)

Ngày lập: 2026-10-04. Chưa có file code nào bị sửa và chưa train lần nào.

**Bản 5 = bản 4 + các sửa theo nhận xét phản biện.** Thầy đã kiểm tra từng điểm trên code gốc iMoonLab/HGNN (bản clone ở `baseline/HGNN`) và trên code của em:

| # | Nhận xét | Đã kiểm tra | Sửa ở |
|---|---|---|---|
| 1 | Luật chọn hyperedge của target chưa thống nhất: User cần ≥ 1 node train, Object cần ≥ 2 | Đúng. `build_local_graph` chỉ lấy Object có trong H0 | 7.3, 7.4 |
| 2 | Câu "lớp 2 có ReLU là khác code gốc" sai | Đúng. `models/layers.py` có `HGNN_embedding` (conv1 → ReLU → dropout → conv2 → **ReLU**) và `HGNN_classifier` (một lớp Linear) | 6.2 |
| 3 | Trong code gốc W toàn số 1, cố định | Đúng. `_generate_G_from_H`: `W = np.ones(n_edge)` | 6.2 |
| 4 | Code gốc gọi `F.dropout` không có `training=` nên dropout chạy cả lúc đánh giá | Đúng. Code của em dùng `self.training`, nên không chép chỗ này | 6.2, 5 |
| 5 | Lớp bảo vệ causal cho hyperedge của target | Nên thêm. Một dòng code, không đổi kết quả nếu Course/Object cùng ngày bắt đầu | 7.3 |
| 6 | Phép thử 12.4 phải chạy ở chế độ eval, E(v) lấy theo H_recv, không cộng lại số hạng của v; thêm kiểm tra ~50 target bằng vòng lặp | Đúng | 12 |
| 7 | Viết d(u) trong bài là "causal degree", không dùng chữ "rò rỉ" | Đồng ý | 6.3 |
| 8 | `family_logits` phải là `nn.Parameter` thì mới nằm trong `state_dict` | Đã xem: `nn.Parameter(torch.full(...))` trong `src/5_model.py` ✓ | 5 |
| 9 | Đánh giá trong lúc train phải chạy trong `torch.no_grad()`, sau đó gọi lại `model.train()` | Đúng | 5 |
| 10 | Chọn theo AUC hay AUPRC là quyết định của em, chỉ cần nhất quán | Giữ val AUC. Test AUC là chỉ số chính của bài | 5 |
| 11 | `pos_rate` | Dropout là lớp **đa số**: train 0.758, val 0.760, test 0.758. AUPRC của mô hình ngẫu nhiên vì vậy ≈ 0.758. Cần ghi khi báo cáo AUPRC | 9 |

Bản 5 thay thế các bản 1–4. Các bản cũ sẽ bị xóa khi em duyệt.

---

## 1. Mục tiêu

- **Một mô hình:** nhánh graph HGNN 2 lớp, theo `HGNN_embedding` của code gốc, có W học theo loại hyperedge và causal mask. Thêm nhánh MLP trên đặc trưng riêng của node. Classifier tuyến tính nhận cả hai nhánh. Hyperedge gồm Course, Object, User (temporal) và self-loop.
- **Một giao thức:** chia 80/20 từ train chính thức; giữ nguyên test chính thức (67,699); train tối đa 1000 epoch với early stopping theo val AUC; chấm test **một lần** bằng model tốt nhất, ở ngưỡng 0.5.
- **Một chỗ ghi kết quả:** `outputs/results.csv`, rồi tổng hợp mean ± std ra CSV và Excel.
- **Không có:** kịch bản, công tắc bật/tắt, file checkpoint, ngưỡng t\*, DeLong, vẽ hình, check_leakage, luật User "any", graph xáo trộn, Behavioral/kNN, phần HSL còn sót.

## 2. Cấu trúc file: mỗi file một việc

```
src/
  0_config.py       Hằng số: đường dẫn, seed, bố cục cột X, loại hyperedge, siêu tham số
  1_download.py     Tải 3 file raw
  2_preprocess.py   Chia train/validation/test, giữ sự kiện ngày 0–34, ghi 3 CSV
  3_features.py     Ma trận đặc trưng X của mỗi split (fit trên train)
  4_hypergraph.py   DỰNG H0 (Course, Object, User) + bảng Object 1 thành viên → hypergraph.npz   [1 lần]
  5_graph_data.py   ĐỌC H0 cho train; tìm hyperedge của từng target val/test                    [lúc train]
  6_hgnn.py         Nhánh graph: liên thuộc, causal mask, lan truyền, encoder 2 lớp,
                    đánh giá quy nạp cho target (mục 7)
  7_mlp.py          Nhánh MLP: encoder 2 lớp trên đặc trưng riêng của node
  8_model.py        Ghép 2 nhánh: [z_graph ‖ z_self] → logit
  9_train.py        Train + early stopping theo val AUC; chấm val/test ở 0.5; ghi results.csv
  10_summary.py     results.csv → summary.csv + ket_qua.xlsx (mean ± std)
scripts/
  run_tmux.sh       Chạy 9_train + 10_summary trong tmux trên server
docs/assets/        Sơ đồ mô hình
README.md, requirements.txt (numpy, scikit-learn, torch, pandas, openpyxl)
```

```
0_config ─┬─ 1_download
          ├─ 2_preprocess ─ 3_features ─ 4_hypergraph             (chuẩn bị dữ liệu, chạy 1 lần)
          └─ 5_graph_data ─┐
             6_hgnn ───────┼─ 8_model ─ 9_train ─ 10_summary      (mỗi lần train)
             7_mlp ────────┘
```

## 3. Bảng giữ / sửa / tách / xóa

| File hiện tại | Dòng | Thành | Việc |
|---|---|---|---|
| src/0_config.py | 111 | 0_config.py (~95) | Sửa |
| src/1_download.py | 32 | 1_download.py | Giữ |
| src/2_preprocess.py | 270 | 2_preprocess.py | Giữ logic, thêm comment công thức |
| src/3_features.py | 300 | 3_features.py (~280) | Bỏ `feature_columns`, thêm comment |
| src/4_hypergraph.py | 527 | **4_hypergraph.py** (~120) + **5_graph_data.py** (~130) | Tách, rút gọn, bỏ local star graph |
| src/5_model.py | 126 | **6_hgnn.py** (~140) + **7_mlp.py** (~30) + **8_model.py** (~50) | Tách ba |
| src/6_train.py | 410 | **9_train.py** (~170) | Viết lại, có early stopping |
| — | — | **10_summary.py** (~40) | Mới |
| scripts/run_tmux.sh | ~60 | run_tmux.sh (~30) | Sửa |
| 11 script khác trong scripts/ | ~2,100 | — | **Xóa** |
| docs/TONG_QUAN_PROJECT.md, PLAN_RUT_GON.md, _v2, _v3, _v4 | — | — | **Xóa** |
| README.md / requirements / environment-gpu.yml / .gitignore | — | — | Sửa |

Các script bị xóa: analyze_contribution, check_leakage, collect_results, delong, export_excel, plot_results, summarize_results, scenarios, run_all, run_integrity, run_scenarios. Tất cả vẫn lấy lại được qua git tag `before-simple`.

## 4. Nội dung từng file (tóm tắt)

| File | Hàm / lớp chính |
|---|---|
| 0_config | đường dẫn, `SEEDS`, `SPLIT_SEED=1`, `TRAIN_RATIO=0.8`, `OBSERVATION_DAYS=35`, bố cục X, `EDGE_FAMILIES=("course","object","user","self_loop")`, `THRESHOLD=0.5`, `TRAIN={...}` (mục 5) |
| 2_preprocess | giữ nguyên logic chia; thêm comment công thức |
| 3_features | đếm hành vi, one-hot, tuổi, chuẩn hóa fit trên train |
| 4_hypergraph | `user_hyperedges` (temporal, so theo **ngày**), `build_train_hyperedges`, `build_hypergraph` → `hypergraph.npz`, gồm H0 và bảng `singleton_object_keys` / `singleton_object_nodes` (Object chỉ có 1 node train, xem 7.4) |
| 5_graph_data | `load_train_graph`; `load_split_targets` lấy cho mỗi target: X, nhãn, ngày bắt đầu, các hyperedge H0 chung (Course, Object), các Object 1 thành viên, và thành viên User temporal |
| 6_hgnn | `prepare_graph`, `HGNNEncoder`: `family_weights`, `propagate`, `forward` (train), `cache_train_states` (Bước A) và `forward_targets` (Bước B) |
| 7_mlp | `MLPEncoder` |
| 8_model | `DropoutModel` = HGNN + MLP + classifier; `forward`, `predict_targets`, `weight_summary` |
| 9_train | `set_seed`, `metrics`, `evaluate`, `train_one_seed` (early stopping), `append_result`, CLI |
| 10_summary | gom theo tag; mean ± std; tỉ lệ w; ghi CSV và Excel |

## 5. Early stopping theo validation

| Tham số (`config.TRAIN`) | Giá trị | Ghi chú |
|---|---|---|
| `epochs` | 1000 | số epoch tối đa |
| `eval_every` | 5 | cứ 5 epoch, tính AUC trên **toàn bộ** validation (31,589 mẫu) |
| `patience` | 40 | dừng nếu 40 lần đánh giá liên tiếp (200 epoch) không cải thiện |
| `select_metric` | `val_auc` | em chọn; test AUC là chỉ số chính trong bài |

Mỗi seed chạy như sau:
1. Train full-batch ở `model.train()`.
2. Cứ `eval_every` epoch thì đánh giá:
   - gọi `model.eval()`;
   - **trong `torch.no_grad()`** chạy Bước A (tính lại trạng thái train trên H0) và Bước B (validation), rồi tính val AUC;
   - gọi lại `model.train()`.
3. Nếu val AUC tốt hơn mức tốt nhất hiện có: lưu `best_state = copy.deepcopy(model.state_dict())` **trong RAM** và ghi `best_epoch`. `state_dict` đã gồm `family_logits`, vì nó là `nn.Parameter`.
4. Nếu `patience` lần liên tiếp không cải thiện thì dừng.
5. Nạp `best_state`, đặt `model.eval()`, chấm val và test **một lần** ở ngưỡng 0.5, rồi ghi 1 dòng vào `results.csv`.

Không ghi file `.pt`, `.json` hay `.npz`. Tập test không được dùng để chọn bất cứ thứ gì. Val AUC tại epoch tốt nhất hơi lạc quan vì là giá trị được chọn, nhưng mức lạc quan rất nhỏ. Bài báo lấy test làm chỉ số chính.

Các cột của `results.csv`:

| Nhóm | Cột |
|---|---|
| Nhận dạng | time, tag, seed |
| Siêu tham số | epochs, eval_every, patience, hidden_dim, dropout, learning_rate, weight_decay |
| Dừng | best_epoch, epochs_run |
| Validation | val_auc, val_auprc, val_f1 |
| Test | test_auc, test_auprc, test_accuracy, test_precision, test_recall, test_f1 |
| W học được | w_course, w_object, w_user, w_self_loop |
| Thời gian | minutes |

## 6. Rà soát công thức HGNN 2 lớp

### 6.1 Công thức chuẩn và code gốc

Bài báo (Feng et al., AAAI 2019, Eq. 10):
```
X^(l+1) = σ( Dv^-1/2 · H · W · De^-1 · Hᵀ · Dv^-1/2 · X^(l) · Θ^(l) )
d(v) = Σ_e w(e)·h(v,e)        δ(e) = Σ_v h(v,e)
```

Code gốc iMoonLab/HGNN (đã đọc trong `baseline/HGNN`):

| File | Nội dung |
|---|---|
| `utils/hypergraph_utils.py::_generate_G_from_H` | `W = np.ones(n_edge)` (cố định, không học); `DV = Σ H·W`; `DE = Σ H`; `G = DV2·H·W·invDE·Hᵀ·DV2` |
| `models/layers.py::HGNN_conv` | `x = x·Θ`; `x = x + b`; `x = G·x`: bias cộng **trước** khi lan truyền |
| `models/HGNN.py::HGNN` | conv1 → ReLU → dropout → conv2, **không** ReLU cuối (mô hình phân loại node) |
| `models/layers.py::HGNN_embedding` | conv1 → ReLU → dropout → conv2 → **ReLU** (nhánh embedding) |
| `models/layers.py::HGNN_classifier` | một lớp `Linear(n_hid, n_class)` |
| Mọi chỗ dùng dropout | `F.dropout(x, self.dropout)` **không** truyền `training`, nên dropout chạy cả lúc đánh giá |

### 6.2 Code của em, đối chiếu từng dòng

| Dòng code | Phép tính | So với code gốc |
|---|---|---|
| `receive = 1[s(v) ≥ max_{u∈e} s(u)]` | causal mask h_recv(v,e) | Phần mở rộng của em. Khi không node nào bị chặn thì H_recv = H, trùng code gốc |
| `H_T` có giá trị 1 cho mọi membership | Hᵀ (bên gửi, không chặn) | ✓ |
| `edge_size = bincount(edge_ids)` | δ(e) = \|e\| | ✓ = `DE = Σ H` |
| `w = softplus(θ)[edge_family]` | W = diag(w_e), một trọng số cho mỗi loại hyperedge, khởi tạo w = 1 | Phần của em. Code gốc W = 1 cố định. Ý tưởng nhóm hyperedge có trọng số học được theo HGNN+ (Gao et al., TPAMI 2023) |
| `node_degree = H_recv @ w` | d(v) = Σ_e h_recv(v,e)·w_e | ✓ = `DV = Σ H·W`, với H thay bằng H_recv (causal degree, 6.3) |
| `node_scale = d^-1/2` | Dv^-1/2 | ✓; d(v) ≥ w_self > 0 nhờ self-loop |
| `edge_x = H_T @ (x · d^-1/2)` | Σ_{u∈e} d(u)^-1/2·x_u | ✓ Hᵀ·Dv^-1/2·X |
| `edge_x *= w / edge_size` | m_e = (w_e/δ(e))·Σ … | ✓ W·De^-1 |
| `H_recv @ edge_x · d^-1/2` | x'_v = d(v)^-1/2·Σ_e h_recv(v,e)·m_e | ✓ Dv^-1/2·H |
| `propagate(layer1(x))` | G·(XΘ1 + b1) | ✓ giống `HGNN_conv` |
| `relu → dropout → propagate(layer2(h)) → relu` | Z_g = ReLU(G·(Dropout(Z1)Θ2 + b2)) | ✓ **giống `HGNN_embedding`** |
| `classifier(cat[z_g, z_self])` | logit = [Z_g ‖ Z_s]·u + b | Theo `HGNN_classifier`; điểm khác là có thêm nhánh MLP |
| `F.dropout(h, p, self.training)` | dropout chỉ chạy khi train | Đúng. Cố ý **không** chép cách gọi thiếu `training` của code gốc |

Cách viết trong bài: "Nhánh graph theo `HGNN_embedding` của Feng et al. (2019), gồm 2 lớp HGNN, mỗi lớp có ReLU. Ma trận lan truyền dùng trọng số hyperedge học theo loại (HGNN+) và causal degree. Classifier tuyến tính nhận biểu diễn graph nối với biểu diễn MLP của chính node."

Công thức đầy đủ của mô hình (ghi trong README và trong `6_hgnn.py`):
```
G      = Dv^-1/2 · H_recv · W · De^-1 · Hᵀ · Dv^-1/2,   d(v) = Σ_e h_recv(v,e)·w_e,   w_e = softplus(θ_{f(e)})
Z1     = ReLU( G · (X·Θ1 + b1) )
Z_g    = ReLU( G · (Dropout(Z1)·Θ2 + b2) )
Z_s    = ReLU( Dropout(ReLU(X·A1 + c1))·A2 + c2 )                      (MLP)
logit  = [Z_g ‖ Z_s]·u + b,    p = σ(logit)
```

### 6.3 Ba điểm đã kiểm tra

1. **Causal degree.** Bậc của node chỉ tính trên các hyperedge mà node được phép nhận. Lý do theo thứ tự thời gian: một enrollment không nên mang thông tin về các enrollment bắt đầu sau nó. Nếu tính trên H đầy đủ, d(u) sẽ phụ thuộc vào số khóa mà người học đăng ký về sau. Trong bài, trình bày đây là một lựa chọn thiết kế theo thứ tự thời gian, không dùng chữ "rò rỉ".
2. **δ(e) đếm mọi thành viên.** Thành viên của User hyperedge đều bắt đầu không muộn hơn anchor. Thành viên của Course/Object có cùng ngày bắt đầu (khóa học là một phần của key). ✓
3. **Chỉ tỉ lệ giữa các w có nghĩa.** Nhân mọi w với cùng hằng số c thì G không đổi. Vì vậy `10_summary` báo thêm tỉ lệ w_f / Σ w.

### 6.4 Dùng ngày cho cả luật User và causal

Luật User hiện so sánh chuỗi `course_start` có cả giờ (dữ liệu có 41 mốc giờ khác nhau), còn causal so sánh theo ngày. Bản mới dùng `start_day` (ngày) cho cả hai.

## 7. Đánh giá val/test: thay local star graph bằng quy nạp 2 bước

### 7.1 Vấn đề của cách hiện tại

Lớp 2 của target dùng trạng thái lớp 1 Z1_u của hàng xóm. Nhưng Z1_u đang được tính trên graph nhỏ bị cắt:
- u mất các Object riêng và User hyperedge;
- bậc d(u) bị thay đổi.

Vì vậy cách tính lúc đánh giá không khớp với lúc train và không đúng định nghĩa HGNN 2 lớp.

```
Lúc train (H0 đầy đủ)                        Local star graph hiện tại
u thuộc: Course c, 30 Object, User(u), self  u thuộc: Course c, 3 Object chung với t, self
d(u) = w_c + 30·w_o + w_u + w_s              d_loc(u) = w_c + 3·w_o + w_s
```

### 7.2 Những gì giữ nguyên vì đúng

- Target chỉ nối tới node train và không thấy nhãn.
- Các target không nối với nhau.

### 7.3 Cách mới

**Luật chọn hyperedge của target (thống nhất cho mọi loại).** Một hyperedge của t được dùng khi **cỡ tính cả t ≥ 2**, tức có ≥ 1 node train. Đây cũng là luật đang áp cho node train trong H0: cỡ tính cả node đó ≥ 2, nếu không thì bỏ vì self-loop đã bao. Cụ thể:

| Loại | Thành viên train của hyperedge của t | Lấy S_e từ |
|---|---|---|
| Course | các node train cùng khóa | S_e đã tính sẵn trên H0 |
| Object (≥ 2 node train) | hyperedge Object trong H0 | S_e đã tính sẵn trên H0 |
| Object (đúng 1 node train) | node u trong bảng `singleton_object_*` | tính trực tiếp từ u |
| User | các enrollment train của cùng người học có `s(m) ≤ s(t)` | tính trực tiếp từ các thành viên |
| Self-loop | không có | S_e = 0 |

**Lớp bảo vệ causal (1 dòng).** Bỏ hyperedge e khỏi E(t) nếu có thành viên u với s(u) > s(t). Theo 6.3, Course và Object cùng ngày bắt đầu, còn User đã lọc sẵn. Vì vậy dòng này không đổi kết quả, chỉ để chắc chắn.

**Bước A.** Mỗi lần đánh giá chạy một lần, ở chế độ eval, trong `no_grad`, trên H0:
```
d(u)  = Σ_e h_recv(u,e)·w_e                       bậc causal của node train
S1_e  = Σ_{u∈e} d(u)^-1/2 · (x_u·Θ1 + b1)          tổng gửi lớp 1 của mỗi hyperedge H0
Z1_u  = ReLU(G·(XΘ1 + b1))_u                       trạng thái lớp 1 của mọi node train
S2_e  = Σ_{u∈e} d(u)^-1/2 · (Z1_u·Θ2 + b2)         tổng gửi lớp 2 của mỗi hyperedge H0
```

**Bước B.** Cho mỗi target t (theo batch):
```
E(t)  = các hyperedge của t theo luật trên (+ self-loop)
d(t)  = Σ_{e∈E(t)} w_e
z1_t  = ReLU( d(t)^-1/2 · Σ_{e∈E(t)} w_e/(δ(e)+1) · ( S1_e + d(t)^-1/2·(x_t·Θ1 + b1) ) )
z_t   = ReLU( d(t)^-1/2 · Σ_{e∈E(t)} w_e/(δ(e)+1) · ( S2_e + d(t)^-1/2·(z1_t·Θ2 + b2) ) )
```
Ở đây δ(e) là số thành viên train của e, cộng 1 vì có thêm t. Với self-loop, δ(e) = 0.

**Vì sao đúng.** Đây chính là HGNN 2 lớp khi thêm một node mới vào H0, với quy ước rằng node mới không làm đổi trạng thái của node train. Quy ước này cũng đúng với cách mọi node train "thấy" hàng xóm lúc train. Thêm vào đó, toàn bộ validation chỉ tốn khoảng một lần forward.

### 7.4 Thay đổi ở `4_hypergraph.py` để hỗ trợ luật trên

- H0 vẫn bỏ hyperedge có cỡ < 2. Luật này không đổi với node train.
- Ghi thêm vào `hypergraph.npz` bảng Object chỉ có 1 node train: `singleton_object_keys` và `singleton_object_nodes`. Nhờ đó `5_graph_data` không phải đọc lại sự kiện của train.
- Node train u trong bảng này gửi vào hyperedge mới của target với bậc d(u) trên H0, giống cách các thành viên của User hyperedge mới của target gửi.

## 8. Quy ước comment công thức

**Bảng ký hiệu dùng chung** (đặt ở đầu `6_hgnn.py` và trong README):

| Ký hiệu | Nghĩa |
|---|---|
| N, E | số node train, số hyperedge của H0 |
| X ∈ ℝ^{N×D} | ma trận đặc trưng |
| H, H_recv | liên thuộc đầy đủ / liên thuộc nhận (đã áp causal mask) |
| W = diag(w_e) | w_e = softplus(θ_{f(e)}), f(e) là loại của e |
| De: δ(e) = \|e\| | số thành viên |
| Dv: d(v) = Σ_e h_recv(v,e)·w_e | causal degree |
| s(v) | ngày bắt đầu khóa học của v (số ngày) |
| S1_e, S2_e | tổng gửi của hyperedge e ở lớp 1, lớp 2 (7.3) |
| E(t) | tập hyperedge của target t |

**Các dòng sẽ được comment công thức:**

| File | Dòng code | Comment |
|---|---|---|
| 2_preprocess | ngày trong khóa | `d = date(event) − date(course_start)`; giữ nếu `0 ≤ d < 35` |
| 2_preprocess | chia train/val | `n_train = ⌊0.8·n⌋` sau khi xáo với seed 1 |
| 3_features | chuẩn hóa hành vi | `x = (log(1+c) − μ_train)/σ_train` |
| 3_features | tuổi | `age = năm(start) − birth`, hợp lệ nếu `10 ≤ age ≤ 100`; nếu thiếu thì điền median_train và đặt `age_missing = 1` |
| 4_hypergraph | User temporal | `e_U(a) = {a} ∪ {m : user(m) = user(a), s(m) ≤ s(a)}` |
| 4_hypergraph | Course / Object | `e_C(c) = {v : course(v) = c}`, `e_O(o) = {v : v dùng o}`; bỏ nếu \|e\| < 2 |
| 6_hgnn | causal | `h_recv(v,e) = h(v,e)·1[s(v) ≥ max_{u∈e} s(u)]` |
| 6_hgnn | W | `w_e = softplus(θ_{f(e)})`, khởi tạo w = 1 |
| 6_hgnn | lan truyền | 5 dòng như ví dụ dưới, cộng dạng ma trận Eq. 10 |
| 6_hgnn | 2 lớp | `Z1 = ReLU(G(XΘ1+b1))`; `Z_g = ReLU(G(Dropout(Z1)Θ2+b2))` (HGNN_embedding) |
| 6_hgnn | target mới | công thức Bước A, Bước B (7.3) |
| 7_mlp | MLP | `z_s = ReLU(Dropout(ReLU(xA1+c1))A2+c2)` |
| 8_model | phân loại | `logit = [z_g ‖ z_s]·u + b`; `p = σ(logit)` |
| 9_train | loss | `L = −(1/N) Σ [y log p + (1−y) log(1−p)]` |
| 9_train | tối ưu | Adam; L2 = 5e-4 cho Θ; W có lr 0.05 và không L2; clip `‖g‖ ≤ 5` |
| 9_train | early stopping | `best = argmax_epoch AUC_val`; dừng sau 40 lần đánh giá không tăng |
| 9_train | chỉ số | `ŷ = 1[p ≥ 0.5]`; Precision, Recall, F1, Acc; `AUC = P(p_pos > p_neg)`; `AUPRC = Σ (R_n − R_{n−1})P_n` |
| 10_summary | tổng hợp | mean, `std = √(Σ(x−mean)²/(n−1))`, tỉ lệ `w_f/Σw` |

Ví dụ trong `6_hgnn.py`:
```python
def propagate(self, x, graph, w):
    node_degree = torch.sparse.mm(graph["H_recv"], w[:, None])      # d(v) = Σ_e h_recv(v,e)·w_e  (causal degree, 6.3)
    node_scale = node_degree.clamp_min(1e-6).rsqrt()                  # d(v)^(-1/2)
    edge_x = torch.sparse.mm(graph["H_T"], x * node_scale)            # Σ_{u∈e} d(u)^(-1/2)·x_u
    edge_x = edge_x * (w / graph["edge_size"])[:, None]               # m_e = (w_e/δ(e))·Σ_{u∈e} d(u)^(-1/2)·x_u
    return torch.sparse.mm(graph["H_recv"], edge_x) * node_scale      # x'_v = d(v)^(-1/2)·Σ_e h_recv(v,e)·m_e
```

## 9. Giao thức (cho README và bài báo)

| Mục | Cách làm |
|---|---|
| Dữ liệu | XuetangX (CFIN, Feng et al., 2019); node = enrollment; đặc trưng = 35 ngày đầu khóa |
| Chia | train chính thức → 80% train / 20% validation (seed 1); giữ nguyên test chính thức (67,699) |
| Tỉ lệ dropout | train 0.758, val 0.760, test 0.758 (dropout là lớp đa số; AUPRC ngẫu nhiên ≈ 0.758) |
| Fit tiền xử lý | chỉ trên train |
| Graph | train: H0; val/test: quy nạp 2 bước, target đọc từ trạng thái node train (7.3) |
| Train | full-batch, BCE, Adam, tối đa 1000 epoch, early stopping theo val AUC (mỗi 5 epoch, patience 40) |
| Đánh giá | model tốt nhất theo val; val và test ở ngưỡng 0.5; test chạy một lần |
| Lặp | 5 seed (1, 11, 111, 1111, 11111), báo cáo mean ± std |

## 10. Dữ liệu phải build lại

**Local:** CSV và X.npy đã đúng. Chạy `python src/4_hypergraph.py`, rồi xóa `hypergraph_temporal.npz` và `data/processed/simple/baselines/`.

**Server:** dữ liệu cũ thiếu 3,305 enrollment, nên test chỉ có 66,745. Chạy lại toàn bộ:
```
mv data/processed/simple data/processed/simple_old
mv result result_old;  rm -rf outputs/runs outputs/reports
python src/2_preprocess.py && python src/3_features.py && python src/4_hypergraph.py
```

## 11. Thứ tự thực hiện (chỉ khi em nói "làm")

0. Commit trạng thái hiện tại và gắn tag `before-simple`.
1. `0_config.py`.
2. `2_preprocess.py`, `3_features.py`: thêm comment công thức, bỏ `feature_columns`.
3. `4_hypergraph.py` (H0, so theo ngày, bảng Object 1 thành viên) và `5_graph_data.py` (hyperedge của target theo luật 7.3).
4. `6_hgnn.py` (train + Bước A/B), `7_mlp.py`, `8_model.py`.
5. `9_train.py` (early stopping trong RAM, `no_grad` khi đánh giá), `10_summary.py`.
6. `run_tmux.sh`; xóa script, tài liệu và plan cũ.
7. README (công thức 6.2, bảng ký hiệu, giao thức), requirements, environment-gpu.yml, .gitignore.
8. Kiểm tra theo mục 12.

## 12. Kiểm tra sau khi sửa (local, CPU)

1. `python -m py_compile src/*.py`.
2. `python src/4_hypergraph.py`: Course (247) và Object (21,747) giữ nguyên. User có thể chênh rất ít do so theo ngày; ghi lại con số. Ghi thêm số Object chỉ có 1 thành viên.
3. **Forward lúc train trùng code cũ:** cùng seed 1, 10 epoch, so loss và train AUC với tag `before-simple`. Nếu User thay đổi theo 6.4 thì so trên Course + Object.
4. **Phép thử nhất quán trên node train.** Lấy 1,000 node train v, cho chạy qua đường đánh giá Bước A/B với 3 điều kiện:
   - (a) cả hai phía đều ở `model.eval()`;
   - (b) E(v) là các hyperedge H0 mà v **được phép nhận** theo H_recv, cộng self-loop của v;
   - (c) δ(e) **không** cộng 1 và **không** cộng thêm số hạng của v, vì v đã có trong S_e.

   Logit phải trùng forward đầy đủ trên H0, sai lệch < 1e-5.
5. **Kiểm tra theo định nghĩa trên ~50 target.** Viết thẳng vòng lặp từng hyperedge, từng thành viên theo công thức 7.3, không dùng S_e đã tính sẵn. So với Bước B, sai lệch phải < 1e-5. Phép thử này đi qua phần δ+1, phần tính S_e từ đầu cho User và Object 1 thành viên, và phần số hạng của t, là những phần mục 4 không đi qua.
6. **Thứ tự thời gian:** với 2,000 target ngẫu nhiên, mọi thành viên trong E(t) có s(u) ≤ s(t), và không có node val/test nào nằm trong H0.
7. `python src/9_train.py --seeds 1 --epochs 30 --eval-every 5 --patience 2`: phải dừng sớm đúng quy tắc; `results.csv` có `best_epoch`; không có file `.pt`, `.json` hay `.npz` nào được tạo.
8. `python src/10_summary.py`: mở được `summary.csv` và `ket_qua.xlsx`.
9. Tìm tham chiếu còn sót: `checkpoint|behavioral|k_max|candidate|delong|shuffle|user_rule|scenario|skip_connection|build_local_graph|REPORTS|RUNS`.

## 13. Điểm em cần xác nhận

- [ ] Early stopping: theo **val AUC**, mỗi 5 epoch, patience 40, tối đa 1000 epoch, `best_state` giữ trong RAM, đánh giá trong `no_grad`.
- [ ] Mô tả nhánh graph trong bài: "theo `HGNN_embedding` + W học theo loại (HGNN+) + causal degree" (6.2).
- [ ] Dùng ngày cho cả luật User và causal (6.4).
- [ ] **Luật hyperedge của target: cỡ tính cả t ≥ 2 cho mọi loại**, kèm bảng Object 1 thành viên trong `hypergraph.npz` (7.3, 7.4).
- [ ] Thay local star graph bằng quy nạp 2 bước, kèm lớp bảo vệ causal (7.3).
- [ ] Cấu trúc 11 file ở mục 2; xóa 11 script, `TONG_QUAN_PROJECT.md` và các plan bản 1–4.
- [ ] Cho phép commit và gắn tag `before-simple` trước khi sửa.
