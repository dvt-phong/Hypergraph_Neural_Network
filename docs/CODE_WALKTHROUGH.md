# Code walkthrough — XuetangX dropout với HGNN + HSL-inspired refinement

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
sparse H0 + train-reference neighbors
  ↓ 5_model.py + 6_hsl.py
Z0 → H* → Z* → logits
  ↓ 7_losses.py
weighted BCE + contrastive consistency
  ↓ 8_train.py
train → validation checkpoint selection → test
```

Một node là một enrollment. Một user học hai course tạo hai node khác nhau.

## 2. Trách nhiệm từng file

| File | Trách nhiệm | Artifact chính |
|---|---|---|
| `0_config.py` | Đường dẫn, split seed, action vocabulary, feature layout | constants |
| `1_download.py` | Tải raw files nếu chưa tồn tại | raw files |
| `2_preprocess.py` | Giữ official test, chia raw train thành train/validation | ba split CSV |
| `3_features.py` | Fit transform trên train, transform cả ba split | `X.npy` |
| `4_hypergraph.py` | Course → Object → Behavioral edges, sparse `H0` | `hypergraph.npz` |
| `5_model.py` | Hai lượt HGNN và classifier | model outputs |
| `6_hsl.py` | HSL-inspired top-r membership refinement | sparse `H*` |
| `7_losses.py` | Weighted BCE + contrastive consistency | scalar loss |
| `8_train.py` | Train, validation, checkpoint, test | `.pt`, report JSON |

## 3. API hiện tại của file 4–8

### `4_hypergraph.py`

| Function | Vai trò |
|---|---|
| `log_event` | In log có timestamp và flush ngay. |
| `resolve_device` | Chọn CPU/CUDA. |
| `behavioral_neighbors` | Exact cosine kNN theo batch. |
| `build_course_hyperedges` | Nhóm train enrollment cùng course. |
| `build_object_hyperedges` | Nhóm theo course, object type và object ID. |
| `build_behavioral_hyperedges` | Tạo edge `anchor + k neighbors`. |
| `build_h0` | Ghép ba family theo đúng thứ tự thành SciPy CSR. |
| `build_hypergraph` | Điều phối và ghi `hypergraph.npz`. |
| `load_train_graph` | Load `X`, `H0`, label và edge metadata. |
| `load_evaluation_data` | Load target split cùng train-only references. |
| `build_local_graph` | Tạo graph `1 target + train references`. |

### `5_model.py`

| Function/class | Vai trò |
|---|---|
| `to_torch_sparse` | SciPy sparse → PyTorch sparse COO. |
| `hypergraph_propagation` | Tính HGNN propagation chuẩn hóa. |
| `HGSLModel.encode` | Hai layer HGNN. |
| `HGSLModel.forward` | `H0 → Z0 → H* → Z* → logits`. |
| `WeightedIndexAdd` | Backward an toàn bộ nhớ cho learned sparse `H*`. |
| `weighted_index_add` | Wrapper gọi custom autograd operation. |

### `6_hsl.py`

| Function | Vai trò |
|---|---|
| `select_hyperedges` | Sample edge cân bằng theo family/size. |
| `sample_candidate_nodes` | Sample positive member và negative non-member. |
| `membership_scores` | Tính score node–edge đã học. |
| `build_refined_incidence` | Lắp sparse `H*`, tránh isolated node. |
| `refine_hypergraph` | Score → sigmoid → top-r → `H*`. |

Đây là HSL-inspired structure refinement cho project, không phải bản sao nguyên
vẹn HSL IJCAI 2022.

### `7_losses.py`

| Function | Vai trò |
|---|---|
| `train_pos_weight` | Tính `negative / positive`. |
| `contrastive_loss` | Positive pair là cùng enrollment ở `Z0` và `Z*`. |
| `total_loss` | `BCE + lambda_cl × contrastive`. |

### `8_train.py`

| Function | Vai trò |
|---|---|
| `classification_metrics` | AUC, AUPRC, F1, precision, recall. |
| `batch_local_graphs` | Ghép local graphs thành block-diagonal batch. |
| `evaluate` | Dự đoán target row của từng leakage-free local graph. |
| `train_one_epoch` | Forward, loss, backward, clip, optimizer step. |
| `train` | Train + validation + best checkpoint + report. |
| `test` | Load checkpoint đã chọn rồi chạy official test. |

Các helper `log_event`, `memory_summary`, `set_seed`, `resolve_device` phục vụ trực
tiếp cho các function trên.

## 4. Artifact và contract

### `hypergraph.npz`

| Key | Nội dung |
|---|---|
| `h0_data`, `h0_indices`, `h0_indptr`, `h0_shape` | sparse CSR `H0` |
| `edge_families`, `edge_sizes` | metadata thẳng hàng với cột `H0` |
| `train_neighbors` | `k_max` train neighbor của train node |
| `validation_neighbors`, `test_neighbors` | `k_max` train reference của target |
| `k`, `k_max` | số neighbor dùng và số neighbor lưu |
| `neighbor_backend` | backend exact cosine hiện tại |

### Model output

```python
{
    "logits": logits,     # [N]
    "z0": z0,             # [N, D]
    "z_star": z_star,     # [N, D]
    "h_star": h_star,     # [N, E], sparse
}
```

### Leakage-free evaluation

Train dùng toàn bộ train hypergraph. Validation/test không được dựng thành một
graph chung. Mỗi target có local graph riêng với train nodes làm reference; khi
batch, các graph được ghép block-diagonal nên không có cạnh chéo giữa target.

### Model selection

Validation AUC quyết định best checkpoint và early stopping. `test()` chỉ load
checkpoint đó sau khi training kết thúc. Official test không quyết định epoch hay
hyperparameter.

## 5. Những điểm cần giữ khi mở rộng

- Thứ tự family trong `H0`: Course → Object → Behavioral.
- `H0` và `H*` tiếp tục là sparse incidence matrix `[N, E]`.
- Edge metadata phải thẳng hàng với cột incidence matrix.
- Validation/test chỉ tham chiếu train nodes.
- Contrastive positive là `z0[i] ↔ z_star[i]`.
- HSL ablation dùng `H*=H0`, `Z*=Z0`.
- Checkpoint chỉ được chọn bằng validation.

Nếu thêm Temporal hyperedge, hãy bắt đầu từ contract của `build_h0`, bổ sung edge
family/metadata, rồi kiểm tra `select_hyperedges` và local evaluation trước khi đổi
encoder hoặc loss.
