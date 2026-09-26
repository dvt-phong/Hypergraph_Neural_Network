# Code walkthrough — XuetangX dropout với HGNN + HSL

Tài liệu này là bản đồ nhanh của toàn project. Phần giải thích sâu file 4–8 nằm ở
[`FILES_4_TO_8_GUIDE.md`](FILES_4_TO_8_GUIDE.md).

## 1. Pipeline

```text
raw XuetangX
  ↓ 1_download.py
raw files
  ↓ 2_preprocess.py
train.csv | validation.csv | test.csv
  ↓ 3_features.py
train/validation/test X.npy
  ↓ 4_hypergraph.py
H0 (memberships) + train-reference neighbors
  ↓ 5_model.py + 6_hsl.py
Z0 → H* → Z* → logits
  ↓ 7_losses.py
weighted BCE + intra-hyperedge contrastive
  ↓ 8_train.py
train → validation checkpoint selection → test
```

Một node là một enrollment. Một user học hai course tạo hai node khác nhau.

## 2. Trách nhiệm từng file

| File | Trách nhiệm | Artifact chính |
|---|---|---|
| `0_config.py` | Đường dẫn, split seed, action vocabulary, feature layout | constants |
| `1_download.py` | Tải raw files nếu chưa tồn tại | raw files |
| `2_preprocess.py` | Giữ official test, chia raw train và không làm mất enrollment có zero observed events | ba split CSV |
| `3_features.py` | Fit transform trên train, transform cả ba split | `X.npy` |
| `4_hypergraph.py` | Course → Object → Behavioral hyperedges, local graph cho validation/test | `hypergraph.npz` |
| `5_model.py` | HGNN encoder dùng chung cho hai lượt và classifier | model outputs |
| `6_hsl.py` | HSL: `H* = Me ⊙ Mv ⊙ (H0 + ΔH) + I` | `H*` + mask weights |
| `7_losses.py` | Weighted BCE + intra-hyperedge contrastive (HSL Eq. 10) | scalar loss |
| `8_train.py` | Train, validation, checkpoint, test | `.pt`, report JSON |

## 3. API của file 4–8

### `4_hypergraph.py`

| Function | Vai trò |
|---|---|
| `nearest_train_neighbors` | Exact cosine kNN trên behavior columns, theo batch. |
| `read_object_events` | Đọc `(node_id, course\|type\|object_id)` từ split CSV. |
| `build_train_hyperedges` | Gom Course, Object, Behavioral hyperedge thành memberships. |
| `build_hypergraph` | Điều phối và ghi `hypergraph.npz`. |
| `add_self_loops` | Thêm một self-loop cho mỗi node. |
| `load_train_graph` | `features`, `labels`, `graph` của train (kèm ứng viên ΔH). |
| `load_evaluation_split` | Dữ liệu để dựng local graph cho validation/test. |
| `build_local_graph` | Graph `1 target + train references`. |
| `merge_local_graphs` | Ghép nhiều local graph thành một batch không nối nhau. |

### `5_model.py`

| Function/class | Vai trò |
|---|---|
| `graph_to_device` | NumPy graph → PyTorch tensors. |
| `weighted_sum` | Cộng message theo membership, chia chunk để tiết kiệm bộ nhớ. |
| `hgnn_propagate` | `Dv^-1/2 H De^-1 Hᵀ Dv^-1/2 x` với membership weights. |
| `hyperedge_means` | Biểu diễn hyperedge `h_e` = trung bình `Z0` của thành viên. |
| `HSLModel.encode` | Hai layer HGNN. |
| `HSLModel.forward` | `H0 → Z0 → H* → Z* → logits`. |

### `6_hsl.py`

| Function/class | Vai trò |
|---|---|
| `keep_mask` | Mask 0/1 bằng Gumbel straight-through (train) hoặc ngưỡng 0.5 (eval). |
| `StructureLearner.forward` | Ghép `ΔH`, `Me`, `Mv`, self-loop thành `H*`. |
| `StructureLearner.implicit_connections` | ΔH: thêm ứng viên gần nhất vào Behavioral hyperedge. |
| `StructureLearner.membership_logits` | Logit của `Mv` cho mọi membership. |
| `StructureLearner.summary` | Tỉ lệ membership được giữ theo family. |

### `7_losses.py`

| Function | Vai trò |
|---|---|
| `positive_class_weight` | `negative / positive` cho weighted BCE. |
| `build_neighbor_sampler` | Chỉ mục node ↔ hyperedge để sample `T_i`. |
| `sample_hyperedge_neighbors` | Lấy ngẫu nhiên node chung hyperedge với anchor. |
| `contrastive_loss` | Intra-hyperedge InfoNCE hai chiều giữa `Z0` và `Z*`. |
| `total_loss` | `BCE + lambda_cl × contrastive`. |

### `8_train.py`

| Function | Vai trò |
|---|---|
| `DEFAULT_SETTINGS` | Toàn bộ siêu tham số của thí nghiệm. |
| `classification_metrics` | AUC, AUPRC, F1, precision, recall (scikit-learn). |
| `make_run_name`, `make_model` | Tên run theo biến thể ablation và tạo model. |
| `evaluate` | Chấm validation/test trên local graph. |
| `train` | Train + validation + best checkpoint + report. |
| `test` | Nạp checkpoint đã chọn rồi chạy official test. |

## 4. Artifact và contract

### `hypergraph.npz`

| Key | Nội dung |
|---|---|
| `node_ids`, `edge_ids` | memberships của `H0`, sắp theo hyperedge |
| `edge_family`, `edge_keys` | loại và khóa của từng hyperedge |
| `train_neighbors` | `k_max` train neighbor của train node |
| `validation_neighbors`, `test_neighbors` | `k_max` train reference của target |
| `k` | số neighbor nằm trong Behavioral hyperedge; `k..k_max-1` là ứng viên ΔH |

### Model output

```python
{
    "logits": logits,        # [N]
    "z0": z0,                # [N, D]
    "z_star": z_star,        # [N, D]
    "structure": {...},      # kept_course, kept_object, kept_behavioral, added
}
```

### Leakage-free evaluation

Train dùng toàn bộ train hypergraph. Mỗi validation/test target có local graph
riêng với train nodes làm reference; khi batch, các local graph nằm cạnh nhau và
không có hyperedge nào nối hai target.

### Model selection

Validation AUC quyết định best checkpoint và early stopping. `test()` chỉ nạp
checkpoint đó sau khi training kết thúc.

## 5. Những điểm cần giữ khi mở rộng

- Graph luôn gồm `node_ids`, `edge_ids`, `edge_family` thẳng hàng; self-loop ở cuối.
- Validation/test chỉ tham chiếu train nodes.
- Contrastive positive là `z0[i] ↔ z_star[i]`; negative là node chung hyperedge.
- `--no-hsl` cho HGNN baseline với cùng encoder (`H* = H0`, `Z* = Z0`).
- Checkpoint chỉ được chọn bằng validation.

Nếu thêm Temporal hyperedge: thêm tên vào `0_config.EDGE_FAMILIES` (trước
`self_loop`), sinh hyperedge trong `build_train_hyperedges`, và cho target tham
gia trong `build_local_graph`. HSL tự học mask cho family mới.
