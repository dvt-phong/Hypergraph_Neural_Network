# Làm chủ pipeline từ `H0` đến huấn luyện

Tài liệu này bám theo code hiện tại trong `src/4_hypergraph.py` đến
`src/8_train.py`. Các file 0–3 là đầu vào đã chốt.

## 1. Bản đồ chung

```text
X
├─ Course hyperedges
├─ Object hyperedges
└─ Behavioral similarity hyperedges
          ↓
     sparse H0
          ↓
     HGNN(X, H0) = Z0
          ↓
 HSL-inspired refinement(Z0, H0) = H*
          ↓
     HGNN(X, H*) = Z*
          ↓
 classifier(Z*) = logits
          ↓
 weighted BCE + lambda × contrastive(Z0, Z*)
```

Quy ước shape:

| Biến | Shape | Ý nghĩa |
|---|---:|---|
| `X` | `[N, F]` | feature của `N` enrollment |
| `H0`, `H*` | `[N, E]` | incidence matrix node–hyperedge |
| `Z0`, `Z*` | `[N, D]` | embedding trước/sau refinement |
| `logits` | `[N]` | điểm dropout trước sigmoid |

## 2. File 4 — tạo `H0`

### `behavioral_neighbors`

Hàm chuẩn hóa các behavior feature rồi tính cosine similarity theo batch bằng
PyTorch. Đây là exact kNN, không cần FAISS:

```text
normalized_query @ normalized_train.T = cosine similarity
```

- `k_max`: số neighbor được tìm và lưu cho mỗi node.
- `k`: số neighbor đầu tiên dùng khi dựng Behavioral hyperedge.
- Với train, chính node đang truy vấn bị loại khỏi kết quả.
- Với validation/test, mọi neighbor đều là train node.

Exact search dễ kiểm chứng và không có sai số approximate. Đổi sang FAISS chỉ nên
làm khi profiling cho thấy phép nhân cosine là nút thắt trên dữ liệu thực.

### `build_course_hyperedges`

Nhóm các node train theo `course_id`. Mỗi course có từ hai enrollment trở lên tạo
một Course hyperedge.

### `build_object_hyperedges`

Đọc event CSV và nhóm theo:

```python
(course_id, object_type, object_id)
```

Thêm `course_id` vào key để hai object trùng ID ở hai course không bị gộp nhầm.
Chỉ action có trong `OBJECT_ACTIONS` và object hợp lệ mới tham gia.

### `build_behavioral_hyperedges`

Mỗi train node là một anchor. Edge của anchor gồm:

```text
anchor + k neighbor gần nhất
```

Vì vậy `N` train node tạo tối đa `N` Behavioral hyperedge.

### `build_h0`

Hàm ghép edge đúng thứ tự Course → Object → Behavioral. Bản chất incidence matrix
được tạo trực tiếp bằng hai list:

```python
for edge_id, members in enumerate(hyperedges):
    for node_id in members:
        rows.append(node_id)
        columns.append(edge_id)
```

Sau đó `H0[node_id, edge_id] = 1` được lưu dưới dạng SciPy CSR. Không có file
membership trung gian.

### `build_hypergraph`

Luồng điều phối:

1. Load `train/validation/test X.npy`.
2. Tính train neighbors với `k_max`.
3. Tính validation/test neighbors, chỉ tìm trong train.
4. Dựng Course, Object, Behavioral hyperedge.
5. Dựng sparse `H0`.
6. Ghi một `hypergraph.npz`.

Bundle chứa CSR arrays của `H0`, `edge_families`, `edge_sizes`, ba ma trận
neighbors, `k`, `k_max` và tên backend.

### `load_train_graph`

Khôi phục `H0` từ bundle, chọn feature-set và trả dictionary:

```text
features, incidence_matrix, labels, families, sizes
```

### `load_evaluation_data` và `build_local_graph`

Evaluation luôn theo nguyên tắc:

```text
1 target validation/test + các train reference → một local graph
```

Target luôn ở local row 0. Các target validation/test không bao giờ nhìn thấy
nhau. Đây là điểm chống leakage quan trọng nhất khi thay graph builder.

## 3. File 5 — HGNN và forward pass

### `to_torch_sparse`

Đổi SciPy sparse matrix thành coalesced PyTorch COO tensor trên đúng device.

### `hypergraph_propagation`

Thực hiện:

```text
Dv^(-1/2) H De^(-1) H^T Dv^(-1/2) X
```

Trình tự đọc code:

1. Tính degree của node và edge.
2. Chuẩn hóa node feature bằng `Dv^(-1/2)`.
3. Gom node → edge bằng `H^T`.
4. Chia cho edge degree `De`.
5. Phát edge → node bằng `H`.
6. Chuẩn hóa node lần nữa.

`WeightedIndexAdd` ở cuối file là chi tiết kỹ thuật cho `H*` có gradient. Nó dùng
indexed reduction để backward không tạo ma trận gần-dense rất lớn trên CUDA. Khi
học flow mô hình, đọc `HGSLModel` trước và xem block này sau.

### `HGSLModel.encode`

Hai layer HGNN:

```text
Linear → propagation → ReLU → Dropout
Linear → propagation → ReLU
```

### `HGSLModel.forward`

Đây là flow chính cần nhớ:

```python
z0 = self.encode(node_features, h0)
h_star = refine_hypergraph(z0, h0, ...) if hsl else h0
z_star = self.encode(node_features, h_star) if hsl else z0
logits = self.classifier(z_star).squeeze(-1)
```

Output là dictionary gồm `logits`, `z0`, `z_star`, `h_star`. Khi `hsl=False`,
`H*=H0` và `Z*=Z0`, tạo HGNN baseline cùng encoder.

## 4. File 6 — HSL-inspired structure refinement

Đây là refinement lấy cảm hứng từ HSL, không phải bản tái hiện nguyên gốc thuật
toán IJCAI 2022.

### `select_hyperedges`

Chia edge theo family và bucket kích thước, shuffle có seed, rồi lấy round-robin.
Cách này giúp sample không bị một family lớn chiếm hết budget.

### `sample_candidate_nodes`

Với một edge:

- positive candidate được sample từ member hiện tại;
- negative candidate được sample từ non-member;
- local evaluation có thể giới hạn candidate theo node/edge group để không trộn
  các local graph trong cùng batch.

### `membership_scores`

Edge embedding là trung bình positive embedding:

```text
z_e = mean(z_v), v thuộc positive candidates
```

Điểm membership:

```text
s(v,e) = (Wn z_v)^T (We z_e) / sqrt(D) + b
p(v,e) = sigmoid(s(v,e))
```

### `build_refined_incidence`

Hàm thay membership của các edge được chọn, giữ nguyên edge không được chọn và
khôi phục một membership cũ nếu refinement làm node bị cô lập. Kết quả là sparse
`H*` có gradient ở learned membership values.

### `refine_hypergraph`

Flow đọc từ trên xuống:

```text
select edge
→ sample positive/negative candidates
→ membership score
→ sigmoid
→ giữ top_r
→ build H*
```

Chỉ có cơ chế `top_r`; không còn nhánh threshold.

## 5. File 7 — loss

File chỉ có ba hàm:

- `train_pos_weight`: `N_negative / N_positive`.
- `contrastive_loss`: đối chiếu hai view `z0` và `z_star`.
- `total_loss`: weighted BCE + `lambda_cl × contrastive`.

Positive pair của contrastive loss là:

```text
z0[i] ↔ z_star[i]
```

Đó là cùng một enrollment trước và sau refinement, không phải hai learner cùng
nhãn dropout.

## 6. File 8 — train, validation và test

### `classification_metrics`

Tính ROC-AUC, AUPRC và precision/recall/F1 tại probability 0.5. Metric được gom
trong một function để không che luồng train.

### `batch_local_graphs`

Ghép nhiều local graph thành block-diagonal sparse graph. Hàm đồng thời lưu vị
trí row 0 của từng graph, vì chỉ target row mới được đưa vào metric.

### `evaluate`

Với từng target:

1. Gọi `build_local_graph`.
2. Batch các local graph độc lập.
3. Chạy model deterministic.
4. Lấy logit ở target positions.
5. Tính metric sau khi xử lý toàn bộ target.

### `train_one_epoch`

Một epoch gồm đúng các bước: forward → total loss → backward → kiểm gradient HSL
→ gradient clipping → optimizer step.

### `train`

Hàm này cố ý giữ flow thí nghiệm ở cùng một nơi:

```text
load train graph và validation data
→ tạo model + Adam
→ train từng epoch
→ validation
→ lưu checkpoint khi validation AUC tốt hơn
→ early stopping
→ ghi train report
```

### `test`

`test` chỉ nhận checkpoint đã được validation chọn, nạp lại settings/state rồi
đánh giá official test. Test không tham gia chọn checkpoint.

## 7. Checklist khi thay mô hình

Khi nâng cấp sang Temporal Hypergraph hoặc encoder khác, kiểm tra lần lượt:

1. Node vẫn là enrollment hay đã đổi định nghĩa?
2. `H0` có còn `[N, E]` và sparse không?
3. Edge metadata có còn thẳng hàng với cột của `H0` không?
4. Validation/test target có chỉ nối tới train reference không?
5. `encode` có trả `[N, D]` không?
6. Positive pair của contrastive loss có còn cùng enrollment không?
7. Checkpoint có chỉ được chọn bằng validation không?
8. Test có được chạy đúng một lần sau model selection không?

Nếu tám contract này còn đúng, ta có thể thay từng module mà không vô tình đổi
protocol thí nghiệm.
