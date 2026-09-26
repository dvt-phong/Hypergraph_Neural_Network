# Hướng dẫn đọc code file 4–8: từ `H0` đến huấn luyện HSL

Tài liệu này đi cùng code trong `src/4_hypergraph.py` → `src/8_train.py`. Nên đọc
theo thứ tự các mục; mỗi mục chỉ ra hàm nào trong code tương ứng với bước nào của
mô hình.

## 1. Bức tranh chung

```text
X (feature)  +  H0 (Course, Object, Behavioral + self-loop)
                    │
          Z0 = HGNN(X, H0)                         file 5
                    │
   H* = Me ⊙ Mv ⊙ (H0 + ΔH) + I                    file 6  (HSL)
                    │
          Z* = HGNN(X, H*)     ← cùng trọng số     file 5
                    │
          logits = Linear(Z*)
                    │
   L = weighted BCE + λ · L_CL(Z0, Z*)             file 7
```

| Ký hiệu | Shape | Ý nghĩa |
|---|---:|---|
| `X` | `[N, F]` | feature của `N` enrollment (`F = 92` với `full`) |
| `H0`, `H*` | `[N, E]` | incidence matrix: `H[v, e] = 1` nếu node `v` thuộc hyperedge `e` |
| `Z0`, `Z*` | `[N, 64]` | embedding trên cấu trúc gốc / cấu trúc đã học |
| `logits` | `[N]` | điểm dropout trước sigmoid |

## 2. Hypergraph được lưu như thế nào

Thay vì lưu cả ma trận `H` (rất thưa), code chỉ lưu **danh sách các ô bằng 1**,
gọi là *membership*:

```text
node_ids = [0, 1, 2,   0, 2,   1, 3]
edge_ids = [0, 0, 0,   1, 1,   2, 2]
```

đọc là: node 0, 1, 2 thuộc hyperedge 0; node 0, 2 thuộc hyperedge 1; node 1, 3
thuộc hyperedge 2. Thêm một mảng `edge_family[e]` cho biết hyperedge `e` thuộc
loại nào (`0_config.EDGE_FAMILIES`):

| id | family | Ý nghĩa |
|---:|---|---|
| 0 | `course` | các enrollment cùng khóa học |
| 1 | `object` | các enrollment cùng dùng một video / bài tập / forum |
| 2 | `behavioral` | một anchor + `k` enrollment có hành vi giống nhất |
| 3 | `self_loop` | mỗi node một hyperedge chỉ chứa chính nó (HSL Eq. 9) |

`H*` dùng đúng dạng này, cộng thêm một mảng `weights` (một số cho mỗi
membership): `1` với `H0`, còn với `H*` là mask 0/1 do HSL học.

Cả graph là một `dict`:

```python
graph = {
    "num_nodes": N,
    "node_ids": ..., "edge_ids": ...,        # memberships
    "edge_family": ...,                      # [E]
    "candidate_edge_ids": ...,               # [B]      Behavioral hyperedge
    "candidate_node_ids": ...,               # [B, c]   ứng viên ΔH của hyperedge đó
}
```

## 3. File 4 — `4_hypergraph.py`

### Phần A: dựng `H0` (chạy một lần)

| Hàm | Làm gì |
|---|---|
| `nearest_train_neighbors` | kNN cosine chính xác trên 58 cột hành vi, chạy theo batch |
| `read_object_events` | đọc CSV, trả `(node_id, "course\|type\|object_id")` |
| `build_train_hyperedges` | gom train node thành Course / Object / Behavioral hyperedge, bỏ hyperedge chỉ có 1 node |
| `build_hypergraph` | điều phối và ghi `hypergraph.npz` |

`hypergraph.npz` chứa:

| Key | Nội dung |
|---|---|
| `node_ids`, `edge_ids` | memberships của `H0` (đã sắp theo hyperedge) |
| `edge_family` | loại của từng hyperedge |
| `edge_keys` | `course_id`, khóa object, hoặc ID của anchor (Behavioral) |
| `train_neighbors` | `k_max` train neighbor của mỗi train node |
| `validation_neighbors`, `test_neighbors` | `k_max` train neighbor của mỗi target |
| `k` | số neighbor thật sự nằm trong Behavioral hyperedge |

Neighbor thứ `k .. k_max-1` **không** nằm trong `H0`; chúng là ứng viên để HSL
thêm vào (ΔH).

### Phần B: nạp graph khi train và đánh giá

| Hàm | Làm gì |
|---|---|
| `add_self_loops` | thêm một self-loop cho mỗi node |
| `load_train_graph` | trả `features`, `labels`, `graph` của tập train |
| `load_evaluation_split` | nạp những gì cần để dựng local graph của validation/test |
| `build_local_graph` | local graph của **một** target |
| `merge_local_graphs` | ghép nhiều local graph thành một batch |

**Quy tắc chống leakage.** Validation/test không bao giờ được thêm vào `H0`. Mỗi
target có một local graph riêng: node 0 là target, các node còn lại là train
enrollment mà target nối tới:

```text
target ──┬── Course hyperedge  (target + train cùng course)
         ├── Object hyperedges (target + train cùng dùng object đó)
         └── Behavioral hyperedge (target + k train neighbor gần nhất)
```

Khi ghép batch, các local graph nằm cạnh nhau và không có hyperedge nào nối hai
graph, nên các target không nhìn thấy nhau.

## 4. File 5 — `5_model.py`

### Một lớp HGNN

```text
X' = Dv^-1/2 · H · De^-1 · Hᵀ · Dv^-1/2 · (X Θ + b)
```

Code tương ứng (`hgnn_propagate`):

1. Tính bậc node `Dv` và bậc hyperedge `De` từ `weights`.
2. Node → hyperedge: `weighted_sum(x · Dv^-1/2, ..., node_ids → edge_ids)`, rồi chia `De`.
3. Hyperedge → node: `weighted_sum(..., edge_ids → node_ids)`, rồi nhân `Dv^-1/2`.

`weighted_sum` chỉ là "cộng các message theo membership". Nó chạy theo từng chunk
1 triệu membership và dùng `checkpoint` để backward tính lại message thay vì giữ
tất cả trong bộ nhớ, nhờ vậy vừa GPU với toàn bộ XuetangX. Kết quả đã được kiểm
tra khớp với công thức ma trận dày (sai số ~1e-7, cả giá trị lẫn gradient).

Bậc `Dv`, `De` được tính từ `weights.detach()`: hyperedge bị HSL xóa hết thành
viên có `De = 0`, code thay bằng 1 để không chia cho 0 (hyperedge đó chỉ gửi
vector 0).

### `HSLModel`

```python
z0 = encode(X, H0)                       # lượt 1
h_e = hyperedge_means(z0, H0)            # biểu diễn hyperedge = trung bình Z0 của thành viên
H*, weights = structure_learner(z0, h_e, H0)
z_star = encode(X, H*, weights)          # lượt 2, cùng layer1/layer2
logits = classifier(z_star)
```

`encode` = `Linear → propagate → ReLU → Dropout → Linear → propagate → ReLU`,
giống HGNN gốc (Feng et al., 2019). Khi chạy `--no-hsl`, model bỏ qua bước HSL và
`Z* = Z0`, tức HGNN baseline.

## 5. File 6 — `6_hsl.py`

Đối chiếu paper HSL (Cai et al., IJCAI 2022) với code:

| Paper | Code | Ý nghĩa |
|---|---|---|
| Eq. 4–5, `ΔH` | `implicit_connections` | mỗi Behavioral hyperedge thêm `add_per_edge` ứng viên có `cos(z_v, h_e)` cao nhất |
| Eq. 2–3, `Me` | `edge_scorer` + `keep_mask` | giữ/bỏ cả hyperedge, xác suất `σ(MLP([h_e ‖ family]))` |
| Eq. 6–7, `Mv` | `membership_logits` + `keep_mask` | giữ/bỏ từng membership, xác suất `σ(MLP([z_v ‖ h_e]))` |
| Eq. 9, `+ I` | `torch.where(is_self_loop, 1.0, ...)` | self-loop luôn được giữ |
| Eq. 8 | `weights = edge_keep[edge_ids] * membership_keep` | `H* = Me ⊙ Mv ⊙ (H0 + ΔH) + I` |

### `keep_mask`: Gumbel straight-through

```text
train:  soft = σ((logit + noise) / τ),  hard = 1[soft > 0.5]
        giá trị dùng = hard (đúng 0/1), gradient lấy từ soft
eval:   hard = 1[logit > 0]   (không ngẫu nhiên → kết quả lặp lại được)
```

Bias cuối của hai scorer khởi tạo bằng 3 (`σ(3) ≈ 0.95`), nên lúc đầu mô hình giữ
gần hết cấu trúc rồi mới học cách cắt, giống cách HSL khởi tạo.

### Hai điều chỉnh cho bài toán MOOC (cần nêu trong luận án)

1. **`Me` là MLP của biểu diễn hyperedge**, không phải một tham số riêng cho mỗi
   hyperedge như code HSL. Validation/test dùng local graph mới, nên cần một hàm
   áp dụng được cho hyperedge chưa thấy.
2. **ΔH chỉ thêm vào Behavioral hyperedge**, với ứng viên là neighbor `k..k_max-1`.
   Thêm learner khóa khác vào Course hyperedge sẽ sai ý nghĩa "cùng khóa học".

`summary` in ra tỉ lệ membership được giữ theo từng family (`kept_course`,
`kept_object`, `kept_behavioral`) và số membership ΔH còn lại (`added`). Đây là
bản MOOC của pruning rate `r_v` trong Table 1 của paper.

## 6. File 7 — `7_losses.py`

```text
L = BCE(logits, y; pos_weight = #không dropout / #dropout) + λ · L_CL
```

**Intra-hyperedge contrastive (HSL Eq. 10).** Với anchor `i`:

| Loại | Cặp | Ý nghĩa |
|---|---|---|
| positive | `Z0[i] ↔ Z*[i]` | cùng một enrollment ở hai cấu trúc phải giống nhau |
| negative | `Z0[i] ↔ Z0[j]`, `Z0[i] ↔ Z*[j]`, với `j ∈ T_i` | learner cùng hyperedge phải phân biệt được (chống over-smoothing) |

`T_i` là các node chung hyperedge với `i`. `sample_hyperedge_neighbors` lấy
`contrastive_neighbors` node như sau: chọn ngẫu nhiên một hyperedge của `i`, rồi
chọn ngẫu nhiên một thành viên của nó. Loss được tính hai chiều (anchor ở `Z0` và
anchor ở `Z*`) rồi lấy trung bình.

## 7. File 8 — `8_train.py`

Mọi siêu tham số nằm trong `DEFAULT_SETTINGS` ở đầu file; dòng lệnh chỉ ghi đè.

```text
mỗi epoch:
    forward trên toàn bộ train graph → loss → backward → clip → Adam step
mỗi eval_every epoch:
    validation (local graph) → lưu checkpoint nếu AUC tốt hơn
    dừng sớm sau `patience` lần validation không cải thiện
sau cùng (--mode test / both):
    nạp checkpoint tốt nhất → chấm test đúng một lần
```

### Ablation theo Fig. 3 của paper HSL

| Biến thể | Lệnh | Tên run |
|---|---|---|
| Base (HGNN) | `--no-hsl` | `hgnn_...` |
| + edge | `--no-node-sampling --add-per-edge 0` | `hsl_no-node_no-add_...` |
| + node | `--no-edge-sampling` | `hsl_no-edge_...` |
| − CL | `--lambda-cl 0` | `hsl_no-cl_...` |
| HSL đầy đủ | (mặc định) | `hsl_...` |

Checkpoint ở `outputs/runs/<tên run>.pt`, report ở
`outputs/reports/<tên run>_train.json` và `_test.json`.

### Đọc log

```text
[train] epoch 5: loss=0.85, bce=0.46, contrastive=3.94, train_auc=0.99,
        kept_course=0.98, kept_object=0.94, kept_behavioral=0.96, added=763
[validation] 31,588 targets in 250s: auc=..., auprc=..., f1=...
[checkpoint] new best val_auc=... at epoch 5
```

- `kept_*` giảm dần: HSL đang cắt bớt cấu trúc. Toàn bộ về `1.0`: mô hình thấy
  không cần cắt.
- `contrastive` giảm: `Z0` và `Z*` đang được kéo về gần nhau.

## 8. Checklist khi thay đổi mô hình

1. Node vẫn là enrollment?
2. Graph vẫn có `node_ids`, `edge_ids`, `edge_family` thẳng hàng?
3. Validation/test chỉ nối tới train node?
4. `encode` vẫn trả `[N, D]`?
5. Positive của contrastive vẫn là cùng một enrollment ở `Z0` và `Z*`?
6. Checkpoint chỉ được chọn bằng validation, test chỉ chạy một lần?
