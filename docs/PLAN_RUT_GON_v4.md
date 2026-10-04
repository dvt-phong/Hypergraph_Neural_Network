# Plan rút gọn project, bản 4 (để em kiểm tra, CHƯA thực hiện)

Ngày lập: 2026-10-04. Chưa có file code nào bị sửa, chưa chạy train.

**Thay đổi so với bản 3:**
1. **Early stopping theo validation** (mục 5). Trọng số tốt nhất được giữ trong RAM, không ghi file `.pt`.
2. **Rà soát công thức HGNN 2 lớp** (mục 6). Thầy đối chiếu từng dòng code với bài báo HGNN và code gốc iMoonLab.
3. **Rà soát local graph** (mục 7). Có một chỗ train và đánh giá không khớp nhau. Thầy đề xuất cách đánh giá mới, khớp chính xác với lúc train.

Bản 4 thay thế các bản 1–3. Các bản cũ sẽ bị xóa khi em duyệt.

---

## 1. Mục tiêu

- **Một mô hình:** HGNN 2 lớp (nhánh graph) + MLP (nhánh riêng của node) + lớp phân loại. Hyperedge gồm Course, Object, User (temporal) và self-loop, kèm W theo loại hyperedge và causal mask.
- **Một giao thức:**
  - chia 80/20 từ train chính thức; test chính thức (67,699);
  - train tối đa 1000 epoch, **early stopping theo AUC validation**;
  - đánh giá test **một lần** bằng model tốt nhất trên validation, ở ngưỡng 0.5.
- **Một chỗ ghi kết quả:** `outputs/results.csv`. Tổng hợp mean ± std ra CSV và Excel.
- **Không có:** kịch bản, công tắc bật/tắt, file checkpoint, ngưỡng t\*, DeLong, vẽ hình, check_leakage, luật User "any", graph xáo trộn, Behavioral/kNN, phần HSL còn sót.

## 2. Cấu trúc file: mỗi file một việc

```
src/
  0_config.py       Hằng số: đường dẫn, seed, bố cục cột X, loại hyperedge, siêu tham số
  1_download.py     Tải 3 file raw
  2_preprocess.py   Chia train/validation/test, giữ sự kiện ngày 0–34, ghi 3 CSV
  3_features.py     Ma trận đặc trưng X của mỗi split (fit trên train)
  4_hypergraph.py   DỰNG hypergraph train H0 (Course, Object, User) → hypergraph.npz   [chạy 1 lần]
  5_graph_data.py   ĐỌC H0 cho train; tìm các hyperedge của từng target val/test       [lúc train]
  6_hgnn.py         Nhánh graph: liên thuộc, causal mask, lan truyền, encoder 2 lớp,
                    và cách tính cho target mới (mục 7)
  7_mlp.py          Nhánh MLP: encoder 2 lớp trên đặc trưng riêng của node
  8_model.py        Ghép 2 nhánh: [z_graph ‖ z_self] → logit
  9_train.py        Train + early stopping theo val AUC; đánh giá val/test ở 0.5; ghi results.csv
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
| src/4_hypergraph.py | 527 | **4_hypergraph.py** (~110) + **5_graph_data.py** (~120) | Tách và rút gọn; bỏ local star graph (mục 7) |
| src/5_model.py | 126 | **6_hgnn.py** (~130) + **7_mlp.py** (~30) + **8_model.py** (~50) | Tách ba |
| src/6_train.py | 410 | **9_train.py** (~170) | Viết lại, có early stopping |
| — | — | **10_summary.py** (~40) | Mới |
| scripts/run_tmux.sh | ~60 | run_tmux.sh (~30) | Sửa |
| 11 script khác trong scripts/ | ~2,100 | — | **Xóa** |
| docs/TONG_QUAN_PROJECT.md, PLAN_RUT_GON.md, _v2, _v3 | — | — | **Xóa** |
| README.md / requirements / environment-gpu.yml / .gitignore | — | — | Sửa |

Các script bị xóa: analyze_contribution, check_leakage, collect_results, delong, export_excel, plot_results, summarize_results, scenarios, run_all, run_integrity, run_scenarios.

Mọi thứ bị xóa vẫn lấy lại được qua git tag `before-simple`.

## 4. Nội dung từng file (tóm tắt)

| File | Hàm / lớp chính |
|---|---|
| 0_config | đường dẫn, `SEEDS`, `SPLIT_SEED=1`, `TRAIN_RATIO=0.8`, `OBSERVATION_DAYS=35`, bố cục X, `EDGE_FAMILIES=("course","object","user","self_loop")`, `THRESHOLD=0.5`, `TRAIN={...}` (mục 5) |
| 2_preprocess | giữ nguyên logic chia; thêm comment công thức |
| 3_features | đếm hành vi, one-hot, tuổi, chuẩn hóa fit trên train |
| 4_hypergraph | `user_hyperedges` (temporal), `build_train_hyperedges`, `build_hypergraph` → `hypergraph.npz`; không còn cờ CLI |
| 5_graph_data | `load_train_graph`, `load_split_targets` (với mỗi target: X, nhãn, danh sách hyperedge H0 cùng Course/Object, và thành viên User temporal) |
| 6_hgnn | `prepare_graph`, `HGNNEncoder`: `family_weights`, `propagate`, `forward` (train), `cache_train_states` và `forward_targets` (val/test, mục 7) |
| 7_mlp | `MLPEncoder` |
| 8_model | `DropoutModel` = HGNN + MLP + classifier; `forward`, `predict_targets`, `weight_summary` |
| 9_train | `set_seed`, `metrics`, `evaluate`, `train_one_seed` (early stopping), `append_result`, CLI |
| 10_summary | gom theo tag, mean ± std, ghi CSV + Excel |

## 5. Early stopping theo validation

| Tham số (`config.TRAIN`) | Giá trị | Ghi chú |
|---|---|---|
| `epochs` | 1000 | số epoch tối đa |
| `eval_every` | 5 | cứ 5 epoch thì tính AUC trên **toàn bộ** validation (31,589) |
| `patience` | 40 | dừng sau 40 lần đánh giá (tức 200 epoch) không cải thiện |
| `select_metric` | `val_auc` | AUC là chỉ số chính của bài; trước đây dùng AUPRC |

Quy trình mỗi seed:
1. Train full-batch. Cứ `eval_every` epoch thì tính val AUC.
2. Nếu val AUC tốt hơn giá trị tốt nhất: lưu `best_state = copy.deepcopy(model.state_dict())` **trong RAM** và ghi lại `best_epoch`.
3. Nếu `patience` lần liên tiếp không tốt hơn thì dừng.
4. Nạp lại `best_state`, tính chỉ số val và test **một lần** ở ngưỡng 0.5, rồi ghi 1 dòng vào `results.csv`.

Không ghi file `.pt`, `.json` hay `.npz`. Test không bao giờ được dùng để chọn gì.

Với cách đánh giá mới ở mục 7, tính toán toàn bộ validation gần như chỉ bằng một lần forward, nên đánh giá mỗi 5 epoch vẫn nhanh. Không cần tập con 5,000 mẫu như trước.

Cột của `results.csv`:

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

### 6.1 Công thức chuẩn (Feng et al., AAAI 2019, Eq. 10 và code gốc iMoonLab/HGNN)

```
X^(l+1) = σ( Dv^-1/2 · H · W · De^-1 · Hᵀ · Dv^-1/2 · X^(l) · Θ^(l) )
d(v) = Σ_e w(e)·h(v,e)        δ(e) = Σ_v h(v,e)
```

Code gốc (`hypergraph_utils.generate_G_from_H` và `HGNN_conv`):
- `G = DV2 · H · W · invDE · Hᵀ · DV2`;
- mỗi lớp tính `x = x·Θ + b`, rồi `x = G·x`. Như vậy bias được cộng **trước** khi lan truyền;
- model 2 lớp: `relu(hgc1) → dropout → hgc2`.

### 6.2 Code hiện tại (`src/5_model.py`) đối chiếu từng dòng

| Dòng code | Phép tính | So với chuẩn |
|---|---|---|
| `receive = 1[start(v) ≥ max_{u∈e} start(u)]` | causal mask: h_recv(v,e) | Mở rộng thêm. Khi không có node nào bị chặn thì H_recv = H, trùng chuẩn |
| `H_T` có giá trị 1 cho mọi membership | Hᵀ (bên gửi, không chặn) | ✓ |
| `edge_size = bincount(edge_ids)` | δ(e) = \|e\|, đếm mọi thành viên | ✓ đúng định nghĩa De |
| `w = softplus(θ)[edge_family]` | W = diag(w_e), một trọng số cho mỗi loại hyperedge | ✓ W là đường chéo như trong bài; khởi tạo w = 1 |
| `node_degree = H_recv @ w` | d(v) = Σ_e h_recv(v,e)·w_e | ✓ đúng định nghĩa Dv (dùng H_recv, xem 6.3) |
| `node_scale = d^-1/2` | Dv^-1/2 | ✓ d(v) ≥ w_self > 0 nhờ self-loop, không chia cho 0 |
| `edge_x = H_T @ (x · d^-1/2)` | Σ_{u∈e} d(u)^-1/2 · x_u | ✓ Hᵀ·Dv^-1/2·X |
| `edge_x *= w / edge_size` | m_e = (w_e/δ(e)) · Σ … | ✓ W·De^-1 |
| `H_recv @ edge_x · d^-1/2` | x'_v = d(v)^-1/2 · Σ_e h_recv(v,e)·m_e | ✓ Dv^-1/2·H |
| `propagate(layer1(x))` | G·(XΘ1 + b1) | ✓ bias trước khi lan truyền, giống code gốc |
| `relu → dropout → propagate(layer2(h)) → relu` | Z = ReLU(G·(Dropout(Z1)Θ2 + b2)) | Khác code gốc: lớp 2 ở đây có ReLU và ra vector ẩn 128 chiều, không ra logit. Lý do: lớp phân loại nằm sau, ghép với nhánh MLP. Cần ghi rõ trong bài: "2 lớp HGNN + 1 lớp tuyến tính" |

**Kết luận:** phép lan truyền khớp đúng công thức HGNN, mỗi nhân tử ứng với một dòng code. Có một khác biệt có chủ ý (ReLU ở lớp 2, classifier riêng) và một mở rộng (causal mask). Cả hai cần ghi trong bài.

Công thức chính xác của mô hình sẽ ghi trong README và trong comment của `6_hgnn.py`:

```
G      = Dv^-1/2 · H_recv · W · De^-1 · Hᵀ · Dv^-1/2,   d(v) = Σ_e h_recv(v,e)·w_e
Z1     = ReLU( G · (X·Θ1 + b1) )
Z_g    = ReLU( G · (Dropout(Z1)·Θ2 + b2) )
Z_s    = ReLU( Dropout(ReLU(X·A1 + c1))·A2 + c2 )                      (MLP)
logit  = [Z_g ‖ Z_s]·u + b,    p = σ(logit)
```

### 6.3 Ba điểm thầy đã kiểm tra kỹ

1. **Vì sao bên gửi cũng dùng d(u) tính từ H_recv mà không dùng H đầy đủ.** Một enrollment cũ u là thành viên của User hyperedge của các enrollment *sau* cùng người học. Nếu tính d(u) bằng H đầy đủ, d(u) sẽ phụ thuộc vào số khóa học mà người học này đăng ký **trong tương lai**, và thông tin đó sẽ đi vào mọi thông điệp u gửi cho bạn cùng khóa. Đó là rò rỉ. Code hiện tại dùng d(u) từ H_recv, nên **đúng và không rò rỉ**. Thầy giữ nguyên và ghi lý do vào comment.
2. **De đếm mọi thành viên.** User hyperedge chỉ có thành viên bắt đầu không muộn hơn anchor; Course và Object có cùng ngày bắt đầu. Vì vậy δ(e) không mang thông tin tương lai. ✓
3. **Chỉ tỉ lệ giữa các w có ý nghĩa.** Nếu nhân mọi w với cùng một hằng số c thì d(v) nhân c, Dv^-1/2 hai bên cho 1/c, W cho c, nên G không đổi. Vì vậy khi báo cáo, dùng tỉ lệ w_f / Σ w; `10_summary` sẽ thêm cột này.

### 6.4 Một điểm không nhất quán nhỏ (đề xuất sửa)

- Luật User temporal so sánh **chuỗi** `course_start` gồm cả giờ (ví dụ `2016-11-16 08:00:00`; dữ liệu có 41 giờ khác nhau). Trong khi đó causal mask so sánh **ngày** (`toordinal`).
- Hai khóa bắt đầu cùng ngày nhưng khác giờ sẽ bị luật User coi là "muộn hơn", còn causal coi là "cùng lúc". Điều này không gây rò rỉ, chỉ là thiếu nhất quán.
- **Đề xuất:** dùng ngày (`start_day`) cho cả hai.

## 7. Rà soát local graph

### 7.1 Cách làm hiện tại (local star graph)

Với mỗi target t thuộc val/test, code dựng một graph nhỏ:
- t là node 0;
- mỗi hyperedge của t gồm t và các thành viên train của nó: Course, các Object t dùng, User temporal;
- mỗi node có thêm một self-loop.

Sau đó code chạy 2 lớp HGNN trên graph nhỏ này.

```
Lúc train (H0 đầy đủ)                        Lúc đánh giá (local star graph)
u thuộc: Course c, 30 Object, User(u),       u thuộc: Course c, 3 Object chung với t,
         self-loop                                    self-loop
d(u) = w_c + 30·w_o + w_u + w_s              d_loc(u) = w_c + 3·w_o + w_s
Z1_u = trung bình trên TẤT CẢ hyperedge      Z1_u = trung bình CHỈ trên các hyperedge chung với t
```

### 7.2 Vấn đề: train và đánh giá không khớp ở bước thứ 2

- **Lớp 1 của target:** đúng, vì t thấy đủ các hyperedge của nó.
- **Lớp 2 của target:** dùng Z1 của các hàng xóm u. Nhưng Z1_u lúc đánh giá được tính trên graph bị cắt:
  - u mất các Object riêng, User hyperedge và self-information từ khóa khác;
  - d(u) cũng khác, nên hệ số chuẩn hóa d(u)^-1/2 trong thông điệp u gửi cho t cũng khác.
- **Hệ quả:** lúc train, một node nhận từ hàng xóm "đầy đủ"; lúc đánh giá, target nhận từ hàng xóm "bị cắt". Đây không phải rò rỉ, nhưng là **sai lệch phân phối giữa train và đánh giá**. Nó làm thấp phần đóng góp của graph và không đúng định nghĩa HGNN 2 lớp. Memory ngày 2026-10-04 đã ghi điểm này là "cần nêu trong bài". Thầy đề xuất sửa luôn.
- Hai điểm khác của local graph thì **đúng**:
  - target chỉ nối với node train, không thấy nhãn;
  - các target không nối với nhau;
  - causal: target nhận từ mọi hyperedge của nó, vì các thành viên đều bắt đầu không muộn hơn.

### 7.3 Đề xuất: đánh giá quy nạp 2 bước, khớp chính xác với lúc train

**Ý tưởng:** các node train giữ nguyên biểu diễn đã tính trên H0, giống hệt lúc train. Target mới chỉ "đọc" từ chúng.

**Bước A.** Một lần cho mỗi lần đánh giá: chạy HGNN trên H0 ở chế độ eval (không dropout), lưu lại:

```
d(u)                       bậc của node train trên H0
S1_e = Σ_{u∈e} d(u)^-1/2 · (x_u·Θ1 + b1)      tổng gửi lớp 1 của mỗi hyperedge H0
Z1_u                       trạng thái lớp 1 của mọi node train
S2_e = Σ_{u∈e} d(u)^-1/2 · (Z1_u·Θ2 + b2)     tổng gửi lớp 2 của mỗi hyperedge H0
```

**Bước B.** Cho mỗi target t (theo batch, chỉ là phép cộng vector):

```
E(t)  = {Course của t, các Object của t (có trong H0), User temporal của t, self-loop của t}
d(t)  = Σ_{e∈E(t)} w_e
Lớp 1: z1_t = ReLU( d(t)^-1/2 · Σ_{e∈E(t)} w_e/(δ(e)+1) · ( S1_e + d(t)^-1/2·(x_t·Θ1 + b1) ) )
Lớp 2: z_t  = ReLU( d(t)^-1/2 · Σ_{e∈E(t)} w_e/(δ(e)+1) · ( S2_e + d(t)^-1/2·(z1_t·Θ2 + b2) ) )
```

- δ(e)+1 vì target được thêm vào hyperedge.
- Với User hyperedge của t (không có sẵn trong H0), S_e được tính trực tiếp từ các thành viên train của nó.
- Self-loop của t: S_e = 0, δ = 0, nên số hạng còn lại là w_s · d(t)^-1/2 · (…).

**Vì sao cách này đúng:**
- Đây chính là công thức HGNN 2 lớp khi thêm một node mới vào H0, với quy ước rằng node mới không làm thay đổi biểu diễn của node train. Đó cũng là cách mọi node train nhìn thấy hàng xóm lúc train.
- Phép thử kiểm chứng: lấy một node **train** v cho chạy qua đường đánh giá (với δ(e) không cộng 1 vì v đã có trong e). Kết quả phải **trùng với forward đầy đủ trên H0** đến sai số khoảng 1e-6. Local star graph hiện tại không qua được phép thử này.
- Không rò rỉ: S_e và Z1_u chỉ dùng node train và causal mask của H0; target chỉ nhận từ các hyperedge mà mọi thành viên đều bắt đầu không muộn hơn nó.
- Nhanh hơn nhiều: không phải dựng hàng trăm node cho mỗi target. Toàn bộ 31,589 target validation chỉ tốn gần bằng một lần forward. Nhờ vậy early stopping có thể đánh giá trên toàn bộ validation mỗi 5 epoch.

**Hệ quả:** kết quả val/test sẽ khác các số cũ. Phải chạy lại, nhưng lần nào cũng phải chạy lại vì dữ liệu server sai.

### 7.4 Hai chi tiết giữ nguyên (ghi trong bài)

- Hyperedge chỉ có 1 thành viên train bị bỏ khỏi H0. Vì vậy nếu target chỉ dùng chung Object với đúng 1 node train thì Object đó không được tính. Điều này nhất quán với lúc train, vì lúc train hyperedge đó cũng không tồn tại.
- Target có thể không có User hyperedge (người học không có khóa nào trước đó trong train). Khi đó E(t) chỉ còn Course, Object và self-loop.

## 8. Quy ước comment công thức (như bản 3, bổ sung)

**Bảng ký hiệu dùng chung** (đặt ở đầu `6_hgnn.py` và trong README):

| Ký hiệu | Nghĩa |
|---|---|
| N, E | số node (enrollment train), số hyperedge của H0 |
| X ∈ ℝ^{N×D} | ma trận đặc trưng |
| H, H_recv | liên thuộc đầy đủ / liên thuộc nhận (đã áp causal mask) |
| W = diag(w_e) | w_e = softplus(θ_{f(e)}), với f(e) là loại của e |
| De: δ(e) = \|e\| | số thành viên |
| Dv: d(v) = Σ_e h_recv(v,e)·w_e | bậc nhận |
| s(v) | ngày bắt đầu khóa học của v |
| S1_e, S2_e | tổng gửi của hyperedge e ở lớp 1, lớp 2 (mục 7.3) |

**Các dòng sẽ được comment công thức:**

| File | Dòng code | Comment |
|---|---|---|
| 2_preprocess | ngày trong khóa | `d = date(event) − date(course_start)`; giữ nếu `0 ≤ d < 35` |
| 2_preprocess | chia train/val | `n_train = ⌊0.8·n⌋` sau khi xáo với seed 1 |
| 3_features | chuẩn hóa hành vi | `x = (log(1+c) − μ_train)/σ_train` |
| 3_features | tuổi | `age = năm(start) − birth`, hợp lệ nếu `10 ≤ age ≤ 100`; thiếu thì điền median_train và đặt `age_missing = 1` |
| 4_hypergraph | User temporal | `e_U(a) = {a} ∪ {m : user(m) = user(a), s(m) ≤ s(a)}` |
| 4_hypergraph | Course / Object | `e_C(c) = {v : course(v) = c}`, `e_O(o) = {v : v dùng o}`; bỏ nếu \|e\| < 2 |
| 6_hgnn | causal | `h_recv(v,e) = h(v,e)·1[s(v) ≥ max_{u∈e} s(u)]` |
| 6_hgnn | W | `w_e = softplus(θ_{f(e)})`, khởi tạo `w = 1` |
| 6_hgnn | lan truyền | 5 dòng như ví dụ dưới, cộng dạng ma trận Eq. 10 |
| 6_hgnn | 2 lớp | `Z1 = ReLU(G(XΘ1+b1))`; `Z_g = ReLU(G(Dropout(Z1)Θ2+b2))` |
| 6_hgnn | target mới | các công thức Bước A và Bước B ở mục 7.3 |
| 7_mlp | MLP | `z_s = ReLU(Dropout(ReLU(xA1+c1))A2+c2)` |
| 8_model | phân loại | `logit = [z_g ‖ z_s]·u + b`; `p = σ(logit)` |
| 9_train | loss | `L = −(1/N) Σ [y log p + (1−y) log(1−p)]` |
| 9_train | tối ưu | Adam; L2 = 5e-4 cho Θ; W dùng lr 0.05, không L2; clip `‖g‖ ≤ 5` |
| 9_train | early stopping | `best = argmax_epoch AUC_val`; dừng khi 40 lần đánh giá liên tiếp không tăng |
| 9_train | chỉ số | `ŷ = 1[p ≥ 0.5]`; Precision, Recall, F1, Acc, `AUC = P(p_pos > p_neg)`, `AUPRC = Σ (R_n − R_{n−1})P_n` |
| 10_summary | tổng hợp | `mean`, `std = √(Σ(x−mean)²/(n−1))`, tỉ lệ `w_f/Σw` |

Ví dụ trong `6_hgnn.py`:

```python
def propagate(self, x, graph, w):
    node_degree = torch.sparse.mm(graph["H_recv"], w[:, None])      # d(v) = Σ_e h_recv(v,e)·w_e  (không dùng H đầy đủ: tránh rò rỉ, mục 6.3)
    node_scale = node_degree.clamp_min(1e-6).rsqrt()                  # d(v)^(-1/2)
    edge_x = torch.sparse.mm(graph["H_T"], x * node_scale)            # Σ_{u∈e} d(u)^(-1/2)·x_u
    edge_x = edge_x * (w / graph["edge_size"])[:, None]               # m_e = (w_e/δ(e))·Σ_{u∈e} d(u)^(-1/2)·x_u
    return torch.sparse.mm(graph["H_recv"], edge_x) * node_scale      # x'_v = d(v)^(-1/2)·Σ_e h_recv(v,e)·m_e
```

## 9. Giao thức (cho README và bài báo)

| Mục | Cách làm |
|---|---|
| Dữ liệu | XuetangX (CFIN, Feng et al., 2019); node = enrollment; đặc trưng = 35 ngày đầu của khóa |
| Chia | train chính thức → 80% train / 20% validation (seed 1); test chính thức giữ nguyên (67,699) |
| Fit tiền xử lý | chỉ trên train |
| Graph | train: H0; val/test: quy nạp 2 bước, target đọc từ trạng thái node train (mục 7.3) |
| Train | full-batch, BCE, Adam, tối đa 1000 epoch, early stopping theo val AUC (đánh giá mỗi 5 epoch, patience 40) |
| Đánh giá | model tốt nhất trên val; val và test ở ngưỡng 0.5; test chỉ chạy một lần |
| Lặp | 5 seed (1, 11, 111, 1111, 11111), báo cáo mean ± std |

## 10. Dữ liệu phải build lại

**Local:** CSV và X.npy đã đúng. Chỉ cần chạy `python src/4_hypergraph.py`, rồi xóa `hypergraph_temporal.npz` và `data/processed/simple/baselines/`.

**Server:** dữ liệu cũ thiếu 3,305 enrollment, nên test chỉ có 66,745. Làm lại như sau:
```
mv data/processed/simple data/processed/simple_old
mv result result_old;  rm -rf outputs/runs outputs/reports
python src/2_preprocess.py && python src/3_features.py && python src/4_hypergraph.py
```

## 11. Thứ tự thực hiện (chỉ khi em nói "làm")

0. Commit trạng thái hiện tại và gắn tag `before-simple`.
1. `0_config.py`.
2. `2_preprocess.py`, `3_features.py`: thêm comment công thức, bỏ `feature_columns`.
3. `4_hypergraph.py` (dựng H0, so sánh theo ngày) và `5_graph_data.py` (hyperedge của target).
4. `6_hgnn.py` (train + đánh giá quy nạp 2 bước), `7_mlp.py`, `8_model.py`.
5. `9_train.py` (early stopping trong RAM), `10_summary.py`.
6. `run_tmux.sh`; xóa script, tài liệu và plan cũ.
7. README (công thức mục 6.2, bảng ký hiệu, giao thức), requirements, environment-gpu.yml, .gitignore.
8. Kiểm tra theo mục 12.

## 12. Kiểm tra sau khi sửa (local, CPU)

1. `python -m py_compile src/*.py`.
2. `python src/4_hypergraph.py`: số hyperedge Course (247) và Object (21,747) giữ nguyên. User có thể chênh rất ít do chuyển sang so sánh theo ngày; ghi lại con số.
3. **Forward lúc train trùng code cũ:** cùng seed 1, 10 epoch, so loss và train AUC với tag `before-simple`. Nếu dữ liệu User thay đổi theo mục 6.4 thì so trên Course + Object.
4. **Phép thử nhất quán của đánh giá mới:** 1,000 node train cho chạy qua đường đánh giá phải ra logit trùng forward đầy đủ trên H0 (sai lệch < 1e-5).
5. **Không rò rỉ:** với 2,000 target ngẫu nhiên, mọi thành viên trong E(t) có s(u) ≤ s(t), và không có node val/test nào xuất hiện trong H0.
6. `python src/9_train.py --seeds 1 --epochs 30 --eval-every 5 --patience 2`: phải dừng sớm đúng quy tắc; `results.csv` có `best_epoch`; không có file `.pt`, `.json` hay `.npz` nào được tạo.
7. `python src/10_summary.py`: mở được `summary.csv` và `ket_qua.xlsx`.
8. Tìm tham chiếu còn sót: `checkpoint|behavioral|k_max|candidate|delong|shuffle|user_rule|scenario|skip_connection|build_local_graph|REPORTS|RUNS`.

## 13. Điểm em cần xác nhận

- [ ] Early stopping: chọn theo **val AUC**, đánh giá mỗi 5 epoch, patience 40, tối đa 1000 epoch, giữ trọng số tốt nhất trong RAM (không có file).
- [ ] Đồng ý với kết luận rà soát HGNN ở mục 6. Giữ d(u) tính từ H_recv ở bên gửi, vì đây là lựa chọn không rò rỉ.
- [ ] Sửa so sánh ngày bắt đầu: dùng ngày cho cả luật User và causal (mục 6.4).
- [ ] **Thay local star graph bằng đánh giá quy nạp 2 bước** (mục 7.3).
- [ ] Cấu trúc 11 file ở mục 2; xóa 11 script, `TONG_QUAN_PROJECT.md` và các plan bản 1–3.
- [ ] Cho phép commit và gắn tag `before-simple` trước khi sửa.
