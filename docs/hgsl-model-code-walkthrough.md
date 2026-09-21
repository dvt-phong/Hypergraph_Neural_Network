# Đối chiếu từng khối mô hình HGSL với code

Tài liệu này giải thích trực tiếp sơ đồ
`docs/assets/hypergraph-neural-network-v3.png` bằng code hiện tại của project.
Mục tiêu là giúp kiểm tra ba câu hỏi:

1. Mỗi khối trong hình được cài đặt ở đâu?
2. Dữ liệu đi vào và đi ra khỏi khối đó là gì?
3. Khối nào đã hoàn thiện, khối nào đang là lựa chọn thiết kế hoặc còn phải chốt?

![Sơ đồ HGSL](assets/hypergraph-neural-network-v3.png)

Cách đọc tài liệu này: bắt đầu từ luồng ở `src/main.py::run_pipeline()`, rồi
theo từng hàng trong bảng dưới đây. Trong code, mỗi hàm/class có comment `#`
ngay trước định nghĩa: `Mục đích` giải thích vai trò, `Đầu vào` và `Đầu ra`
cho biết dữ liệu đi qua khối, `Lưu ý` ghi invariant hoặc nguy cơ leakage khi cần.
Các chuỗi SQL ba nháy là câu truy vấn thực thi, không phải comment khối.

| Thứ tự trên sơ đồ | Nên mở hàm nào trước | Đầu vào → đầu ra chính |
|---|---|---|
| Raw Data, Preprocessing | [`prepare_dataset()`](../src/data/preprocess.py), [`build_splits()`](../src/data/split.py) | CSV → canonical Parquet và user-disjoint split |
| Feature Engineering, `X` | [`build_features()`](../src/features/transform.py), [`NodeFeatureStore.rows()`](../src/features/io.py) | Event/context + train IDs → `X_base`, `X_context`, `X` |
| Build Hyperedges, `H0` | [`build_hyperedges()`](../src/hypergraph/construction.py), [`build_initial_hypergraph()`](../src/hypergraph/construction.py) | Node/event/features/split → Course, Object, Behavioral incidence và sparse `H0` |
| Initial HGNN, `Z0` | [`HGSLModel.forward()`](../src/hsl.py), [`HGNNBaseline.encode()`](../src/model.py) | `(X, H0)` → `Z0` |
| Hyperedge/Incident Node Sampling | [`sample_balanced_hyperedges()`](../src/hsl.py), [`sample_incident_nodes()`](../src/hsl.py) | `H0`, family, size, RNG → sampled edge IDs và positive/negative candidates |
| Hyperedge Representation, Scorer, `H*` | [`HypergraphRefiner.forward()`](../src/hsl.py), [`MembershipScorer.forward()`](../src/hsl.py) | `Z0` + candidates → weighted sparse `H*` |
| Refined HGNN, `Z*`, Prediction | [`HGSLModel.forward()`](../src/hsl.py), [`HGNNBaseline.classify()`](../src/model.py) | `(X, H*)` → `Z*` → logits; sigmoid → xác suất |
| BCE, Contrastive, Total Loss | [`classification_loss()`](../src/hsl.py), [`contrastive_alignment_loss()`](../src/hsl.py), [`hgsl_objective()`](../src/hsl.py) | Logits, labels, `Z0`, `Z*` → scalar loss |
| Train, Validation, Test | [`run_hgsl_training()`](../src/train.py), [`evaluate_hgsl_checkpoint()`](../src/train.py) | Artifacts + config → checkpoint, metrics, report |

`User hyperedge` có trong hình nhưng không có trong `H0` chính; xem mục 6.5 và
22.1. Khi train, `H*` chỉ refine sampled edges; khi validation/test, toàn bộ
edge của từng local graph được refine theo cách deterministic.

## 1. Bản đồ nhanh từ sơ đồ sang code

| Khối trong hình | Code chính | Trạng thái |
|---|---|---|
| Dataset XuetangX | `data/preprocess.py`, `data/split.py` | Đã triển khai |
| Feature Engineering | `features/engineering.py`, `features/context.py`, `features/transform.py` | Đã triển khai |
| Nodes: Enrollment | `nodes.parquet` | Đã triển khai |
| Course hyperedge | `hypergraph/course.py` | Đã triển khai |
| Object hyperedge | `hypergraph/object.py` | Đã triển khai |
| Behavioral hyperedge | `hypergraph/behavioral.py` | Đã triển khai |
| User hyperedge | Không đưa vào `H0` | Chủ ý không dùng vì nguy cơ temporal leakage |
| Initial hypergraph `H0` | `hypergraph/construction.py` | Đã triển khai |
| Node features `X` | `features/transform.py`, `features/io.py` | Đã triển khai |
| HGNN đầu tiên | `model.py` | Đã triển khai |
| Initial embedding `Z0` | `HGNNBaseline.encode()` | Đã triển khai |
| Hyperedge Sampling | `hsl.py::sample_balanced_hyperedges()` | Đã triển khai |
| Incident Node Sampling | `hsl.py::sample_incident_nodes()` | Đã triển khai |
| Hyperedge representation | `hsl.py::HypergraphRefiner.forward()` | Đã triển khai |
| Refine Hypergraph `H*` | `hsl.py::HypergraphRefiner` | Train sampling; validation/test deterministic trên toàn bộ local edge |
| HGNN thứ hai | `hsl.py::HGSLModel.forward()` | Đã triển khai, dùng chung trọng số |
| Refined embedding `Z*` | `hsl.py::HGSLModel.forward()` | Đã triển khai |
| Classifier | `model.py::HGNNBaseline.classify()` | Đã triển khai |
| Prediction | `sigmoid(logits)` trong `train.py` | Đã triển khai |
| BCE Loss | `hsl.py::classification_loss()` | Đã triển khai |
| Contrastive Loss | `hsl.py::contrastive_alignment_loss()` | Đã triển khai |
| Total Loss | `hsl.py::hgsl_objective()` | Đã triển khai |
| Train/validation/checkpoint | `train.py::run_hgsl_training()` | Đã triển khai |
| Test | `train.py::evaluate_hgsl_checkpoint()` | Đã triển khai; deterministic và batch-invariant |

## 2. Ký hiệu và kích thước

| Ký hiệu | Ý nghĩa |
|---|---|
| `N` | Số enrollment node trong graph đang xét |
| `E` | Số hyperedge |
| `F` | Số feature đầu vào |
| `D` | Kích thước embedding, mặc định 64 |
| `H0 ∈ {0,1}^{N×E}` | Incidence matrix ban đầu |
| `H* ∈ R^{N×E}` | Incidence matrix sau refinement, giá trị membership dương |
| `X ∈ R^{N×F}` | Node feature matrix |
| `Z0 ∈ R^{N×D}` | Embedding từ `H0` |
| `Z* ∈ R^{N×D}` | Embedding từ `H*` |
| `y ∈ {0,1}^N` | Nhãn dropout |

Artifact hiện tại có các kích thước:

```text
X_base:    [5 seed, 225642 node, 60 feature]
X_context: [5 seed, 225642 node, 36 feature]

behavior:        60 chiều
behavior_user:   75 chiều
behavior_course: 81 chiều
full:            96 chiều
```

Với seed 1 và `behavioral_k=10`, train graph hiện tại là:

```text
N = 144543 train nodes
E = 153290 hyperedges
nnz(H0) = 3369506 memberships

Course hyperedges:     247
Object hyperedges:   21891
Behavioral hyperedges: 131152
```

## 3. Luồng end-to-end trong code

Luồng dễ đọc nhất nằm ở `src/main.py::run_pipeline()`:

```python
prepared_data = prepare_dataset()
data_splits = build_splits()
node_features = build_features()
hyperedges = build_hyperedges()
initial_hypergraph = build_initial_hypergraph()

training_report = run_hgsl_training(training_config)
test_report = evaluate_hgsl_checkpoint(...)
```

Phần forward của mô hình nằm ở `src/hsl.py::HGSLModel.forward()`:

```python
z0 = backbone.encode(node_features, initial_operator)

refinement = refiner(
    z0,
    initial_incidence,
    families,
    sizes,
    generator,
)

z_star = backbone.encode(node_features, refinement.operator)
logits = backbone.classify(z_star)
```

Đây chính là:

```text
(X, H0) → Z0 → H* → Z* → logits
```

## 4. Khối Dataset XuetangX

### Mục tiêu

Chuyển dữ liệu CSV gốc thành bốn bảng chuẩn, có thứ tự node ổn định và cửa sổ
quan sát 35 ngày.

### Code

```text
src/data/preprocess.py
  validate_source_files()
  prepare_dataset()

src/data/split.py
  assign_user_groups()
  build_splits()
```

### Đầu vào

```text
train_log.csv
test_log.csv
train_truth.csv
test_truth.csv
user_info.csv
course_info.csv
```

### Đầu ra

```text
nodes.parquet
events_35d.parquet
users.parquet
courses.parquet
splits.parquet
```

### Quyết định quan trọng

- Một node là một enrollment.
- `node_id` là chỉ số liên tục dùng trong NumPy/SciPy/PyTorch.
- `enroll_id` là identity của enrollment trong dataset.
- Split theo user, không split độc lập từng enrollment.
- Một user không xuất hiện đồng thời ở train, validation và test.
- Nhãn `1` là dropout, nhãn `0` là non-dropout.

### Cách kiểm tra

```powershell
python run.py prepare-data
python run.py split-data
python -m unittest tests.test_phase1 tests.test_phase2 -v
```

## 5. Khối Feature Engineering

### Mục tiêu

Biến log hành vi, thông tin user và course thành vector số cho từng enrollment.

### Behavioral features

`X_base` có 60 chiều:

```text
35 daily activity counts
+ 23 action counts
+ session_count
+ distinct_observed_objects
= 60
```

Raw count được biến đổi:

```text
x_logged = log(1 + x)
x_scaled = (x_logged - mean_train) / std_train
```

`mean_train` và `std_train` được tính riêng cho từng seed, chỉ trên node thuộc
train split.

### Context features

```text
User demographic: 15 chiều
Course context:    21 chiều
Tổng context:      36 chiều
```

Numeric context dùng train median để điền missing và train mean/std để chuẩn hóa.
Categorical context dùng fixed vocabulary, không học vocabulary từ test.

### Code

```text
src/features/engineering.py  behavioral aggregation
src/features/context.py      user/course context
src/features/transform.py    train-only transform và ghi array
src/features/io.py           chọn feature block và load row
src/config.py                thứ tự/tên feature
```

### Đầu vào và đầu ra

```text
nodes + events + users + courses + splits
→ X_base.npy + X_context.npy
```

### Điểm chống leakage

Toàn bộ node đều được transform, nhưng thống kê transform chỉ được fit trên
train IDs của seed tương ứng.

## 6. Khối Hypergraph Construction

### 6.1 Nodes

Mỗi enrollment là một node:

```text
v_i = enrollment_i
```

Một user học hai course tạo hai enrollment và vì vậy tạo hai node khác nhau.

### 6.2 Course hyperedge

Một course hyperedge nối các train enrollment cùng course:

```text
e_course(c) = {v_i | course_id(v_i) = c}
```

Code:

```text
src/hypergraph/course.py::course_membership_query()
```

### 6.3 Object hyperedge

Một object hyperedge nối các enrollment tương tác với cùng object trong cùng
course. Object được nhóm thành video, assignment và forum.

```text
e_object(c, o) = {v_i | v_i tương tác object o trong course c}
```

Code:

```text
src/hypergraph/object.py::object_membership_query()
```

Chỉ giữ structural hyperedge có ít nhất hai node.

### 6.4 Behavioral hyperedge

Mỗi anchor node được nối với `k` train node gần nhất theo cosine similarity trên
behavioral feature 60 chiều:

```text
e_behavior(i) = {i} ∪ kNN_train(i)
```

Code:

```text
src/hypergraph/behavioral.py::build_seed_neighbors()
```

FAISS chỉ index train nodes. Validation/test node được dùng làm query nhưng mọi
neighbor trả về đều thuộc train.

### 6.5 User hyperedge trong hình

Sơ đồ có vẽ User hyperedge, nhưng cấu hình nghiên cứu hiện tại không đưa family
này vào `H0`.

Lý do: hai enrollment của cùng user có thể thuộc những thời điểm/course khác
nhau. Nối trực tiếp theo user có nguy cơ truyền thông tin ngoài observation
window hoặc làm mất tính user-disjoint của evaluation.

Vì vậy code hiện chỉ dùng:

```text
Course + Object + Behavioral
```

Nếu luận án vẫn giữ User hyperedge trong hình, cần ghi chú “candidate/ablation,
không dùng trong main configuration”. Cách rõ hơn là xóa User hyperedge khỏi hình
main model và đưa nó sang hình ablation.

## 7. Khối Initial Hypergraph `H0`

### Incidence matrix

```text
H0[i, e] = 1 nếu node i thuộc hyperedge e
H0[i, e] = 0 nếu không thuộc
```

Code materialize graph:

```text
src/hypergraph/construction.py::build_initial_hypergraph()
```

Định dạng:

```text
SciPy CSR uint8
H0_train_seed_{seed}.npz
```

Các invariant được kiểm tra:

- incidence là binary;
- không có duplicate membership;
- không có edge rỗng hoặc singleton;
- mọi train edge chỉ chứa train node;
- metadata có cùng thứ tự với column của `H0`.

## 8. Khối Node Features `X`

`src/graph_data.py::load_train_hypergraph()` đảm bảo ba thành phần có cùng thứ tự:

```text
node_ids[row]
X[row, :]
H0[row, :]
```

Đây là invariant rất quan trọng. Nếu thứ tự này lệch, mô hình vẫn chạy nhưng
feature sẽ thuộc sai enrollment.

Tùy `feature_set`, `X` có kích thước:

```text
behavior        [N, 60]
behavior_user   [N, 75]
behavior_course [N, 81]
full            [N, 96]
```

## 9. Khối HGNN đầu tiên

### Mục tiêu

Truyền thông tin giữa các node thuộc chung hyperedge và tạo initial embedding
`Z0`.

### Toán tử propagation

Với incidence matrix `H`, edge weight `W`, node degree `Dv` và edge degree `De`:

```text
G(H) = Dv^(-1/2) H W De^(-1) Hᵀ Dv^(-1/2)
```

Một HGNN layer thực hiện:

```text
Y = G(H) X Θ + b
```

Code:

```text
src/model.py::HypergraphOperator
src/model.py::HGNNLayer
src/model.py::HGNNBaseline
```

Code không materialize ma trận dense `G(H)`. Phép nhân được tách thành:

```text
node → edge → node
```

nhờ sparse matrix multiplication.

### Hai layer encoder

```text
hidden = ReLU(HGNNLayer1(X, H0))
hidden = Dropout(hidden)
Z0     = ReLU(HGNNLayer2(hidden, H0))
```

Đầu vào:

```text
X  [N, F]
H0 [N, E]
```

Đầu ra:

```text
Z0 [N, D]
```

## 10. Khối Hyperedge Sampling

### Tại sao phải sample?

Seed 1 hiện có hơn 153 nghìn hyperedge. Scoring mọi node-edge pair trong mỗi
epoch sẽ rất tốn bộ nhớ và thời gian.

Code chia edge theo hai chiều:

```text
family: course / object / behavioral
size:   small / medium / large
```

Size bucket:

```text
small:  size ≤ 10
medium: 10 < size ≤ 100
large:  size > 100
```

Sau đó sample round-robin giữa các strata để edge lớn hoặc behavioral edge
không chiếm toàn bộ budget.

Code:

```text
src/hsl.py::cardinality_buckets()
src/hsl.py::sample_balanced_hyperedges()
```

Đầu vào:

```text
families [E]
sizes    [E]
sampled_hyperedges
random generator
```

Đầu ra:

```text
sampled_edge_ids [S]
```

## 11. Khối Incident Node Sampling

Với mỗi sampled hyperedge, code lấy:

- positive nodes: node đang incident với edge;
- negative nodes: node không incident với edge.

```text
candidates(e) = positives(e) ∪ negatives(e)
```

Code:

```text
src/hsl.py::sample_incident_nodes()
```

Trong train graph, negative có thể lấy từ toàn bộ train node. Trong batched local
evaluation graph, negative bị giới hạn trong đúng local graph bằng
`node_groups` và `edge_groups`, nên hai target graph không trao đổi node.

## 12. Hyperedge Representations

Với positive node embedding được sample từ edge `e`, hyperedge representation là
mean pooling:

```text
r_e = mean({z_i^0 | i thuộc sampled positives của e})
```

Code nằm trong:

```text
src/hsl.py::HypergraphRefiner.forward()
```

Điểm cần hiểu đúng: code hiện dùng `Z0` để tạo cả node representation và edge
representation trong bước refinement. Đường nét đứt trong hình không phải một
vòng lặp nhiều lần giữa `Z*` và `H*`.

## 13. Khối Membership Scorer

Mỗi candidate node-edge pair được tính compatibility score:

```text
s(i,e) = ((Wv z_i) ⊙ (We r_e)).sum / sqrt(D) + b
p(i,e) = sigmoid(s(i,e))
```

Code:

```text
src/hsl.py::MembershipScorer
```

Parameter học được:

```text
Wv: node projection
We: edge projection
b:  scalar bias
```

Gradient từ classification và contrastive loss đi qua `H*` về scorer.

## 14. Khối Refine Hypergraph `H*`

### Selection

Hai chế độ được hỗ trợ:

```text
top_r:     giữ r candidate có score cao nhất
threshold: giữ candidate có sigmoid(score) ≥ threshold
```

Code luôn giữ ít nhất một positive membership cho mỗi sampled edge để tránh edge
rỗng.

### Cách ghép `H*`

```text
unsampled edge → giữ nguyên membership với weight 1
sampled edge   → thay bằng selected membership với weight sigmoid(score)
isolated node  → phục hồi một membership cũ với weight 1
```

`H*` là sparse PyTorch COO tensor. Degree normalization được tính lại từ
membership weight mới.

### Gradient sparse

`src/model.py::_SparseValuesMM` cung cấp gradient theo sparse incidence values mà
không tạo dense `[N,E]` matrix.

Đã kiểm tra trên graph nhỏ:

```text
forward sparse và dense: sai khác 0
gradient feature:         sai khác 0
gradient membership:      sai khác khoảng 7e-15
```

### Train và inference

Khi train, `H*` chỉ refine những edge được sample trong epoch hiện tại. Khi
validation/test, toàn bộ edge của từng local graph được refine deterministic.
Project không lưu một global `H*` cố định vì evaluation theo thiết lập inductive:
mỗi target có một local graph riêng với train-reference nodes.

## 15. Khối HGNN thứ hai và `Z*`

Sau khi dựng `H*`:

```text
Z* = HGNN(X, H*)
```

Code:

```text
src/hsl.py::HGSLModel.forward()
```

Điểm quan trọng: hai lần HGNN đang dùng chung `layer1`, `layer2` và classifier
backbone. Nghĩa là:

```text
encoder_before_HSL.parameters
== encoder_after_HSL.parameters
```

Nếu luận án muốn hai encoder độc lập, code hiện chưa khớp. Nếu muốn shared-weight
encoder, cần ghi rõ trong mô tả mô hình.

## 16. Classifier và Prediction

Classifier là một linear layer:

```text
logit_i = wᵀ z_i* + b
probability_i = sigmoid(logit_i)
```

Code:

```text
src/model.py::HGNNBaseline.classify()
```

Trong train, loss dùng logits trực tiếp. Sigmoid chỉ dùng khi cần probability để
tính metric.

## 17. BCE Loss

Do dropout/non-dropout mất cân bằng, BCE dùng train-only positive weight:

```text
pos_weight = number_of_negative_train_labels
             / number_of_positive_train_labels
```

```text
L_BCE = BCEWithLogits(logits, labels, pos_weight)
```

Code:

```text
src/hsl.py::train_pos_weight()
src/hsl.py::classification_loss()
```

Validation/test labels không tham gia tính `pos_weight`.

## 18. Contrastive Loss

Mục tiêu là giữ representation của cùng một node gần nhau giữa hai graph view:

```text
view 1: Z0 từ H0
view 2: Z* từ H*
```

Sau L2 normalization:

```text
similarity(i,j) = z_i^0 · z_j* / temperature
```

Positive pair là cùng node ở hai view. Các node khác trong sampled contrastive
batch là negative.

Code dùng symmetric InfoNCE:

```text
L_CL = 0.5 × [CE(Z0 → Z*) + CE(Z* → Z0)]
```

Code:

```text
src/hsl.py::contrastive_alignment_loss()
```

Để tránh ma trận similarity của toàn bộ train node, mỗi epoch chỉ sample tối đa
`contrastive_nodes` node.

## 19. Total Loss

```text
L_total = L_BCE + λ L_CL
```

Code:

```text
src/hsl.py::hgsl_objective()
```

Khi `contrastive_weight=0`, total loss khớp đúng BCE baseline.

## 20. Training Protocol

### Một epoch

`src/train.py::run_hgsl_training()` thực hiện:

```text
1. Tạo random generator từ seed và epoch
2. Sample contrastive nodes
3. Tính Z0 từ H0
4. Sample hyperedge và incident candidates
5. Tạo H*
6. Tính Z* và logits
7. Tính BCE + contrastive loss
8. Backward
9. Clip gradient
10. Optimizer step
11. Evaluate validation
12. Save checkpoint nếu validation AUC tốt hơn
```

Optimizer hiện tại:

```text
Adam
```

Checkpoint chứa:

```text
model state
optimizer state
epoch
config
validation metrics
graph manifest signature
feature manifest signature
torch RNG state
```

## 21. Validation và Test Graph

### Train

Train dùng full train `H0`:

```text
train nodes ↔ train hyperedges
```

### Validation/Test

Mỗi target được dựng thành local graph:

```text
1 target node
+ các train reference nodes
+ course/object/behavioral local hyperedges
```

Target không kết nối tới validation/test target khác. Điều này giữ inductive
evaluation và tránh trao đổi thông tin giữa evaluation samples.

Code:

```text
src/graph_data.py::LocalGraphStore
src/graph_data.py::batch_local_hypergraphs()
```

### Model selection

```text
validation AUC → chọn checkpoint
test → chỉ đánh giá checkpoint đã chọn
```

Code:

```text
src/train.py::run_hgsl_training()
src/train.py::evaluate_hgsl_checkpoint()
```

## 22. Những điểm sơ đồ và code chưa hoàn toàn trùng nhau

### 22.1 User hyperedge

Hình có, main code không dùng. Cần sửa hình hoặc ghi chú rõ đây là ablation.

### 22.2 Đường feedback bằng nét đứt

Hình có thể khiến người đọc hiểu rằng `Z*` được lặp lại nhiều lần để refine
hypergraph. Code hiện chỉ thực hiện một lần:

```text
Z0 → H* → Z*
```

Không có iterative refinement `Z*(t) → H*(t+1)`.

### 22.3 Hai khối HGNN

Hình vẽ hai khối HGNN, nhưng code dùng chung parameter. Cần thêm chú thích
“shared weights” nếu đây là thiết kế mong muốn.

### 22.4 Final `H*`

Khi train, code tạo stochastic partial `H*` theo sampled edge trong từng epoch.
Khi validation/test, code refine toàn bộ edge của từng local graph và chọn node
candidate theo thứ tự cố định. Vì vậy inference có `H*` deterministic, nhưng không
lưu một ma trận global `H*` riêng ra đĩa.

## 23. Các điểm protocol đã hoàn thiện

### 23.1 Full validation

`HGSLTrainingConfig` và CLI hiện mặc định:

```text
validation_limit = 0
```

Smoke run vẫn có thể đặt một giới hạn nhỏ bằng tham số `--validation-limit`.

### 23.2 Evaluation không còn phụ thuộc batch size

Validation/test dùng protocol deterministic:

1. Refine toàn bộ hyperedge của mỗi local graph.
2. Chọn positive/negative candidate theo thứ tự cố định.
3. Giới hạn negative trong chính local graph đó.

Kiểm tra lại cùng checkpoint mới và cùng 8 test target:

| Batch size | AUC | AUPRC | F1 |
|---:|---:|---:|---:|
| 1 | 0.80 | 0.925 | 0.75 |
| 8 | 0.80 | 0.925 | 0.75 |

Audit refinement cũng giống nhau ở hai batch size. `batch_size` giờ chỉ ảnh hưởng
cách gom graph để tính nhanh hơn, không thay đổi prediction.

### 23.3 Kiểm tra artifact khi load checkpoint

Evaluator tính lại SHA-256 của graph/feature manifest và so sánh với checkpoint.
Checkpoint cũ không khớp artifact hiện tại sẽ bị từ chối thay vì âm thầm tạo kết
quả sai phiên bản.

### 23.4 Tên checkpoint cho tuning

Tên checkpoint chứa `experiment_id`, được ghép từ `configuration_id`, graph hash
và feature hash. Vì vậy hai training config, `behavioral_k`, hoặc feature artifact
khác nhau không ghi đè checkpoint của nhau. Report chi tiết cũng chứa ID này;
report không hậu tố luôn trỏ tới run mới nhất.

### 23.5 Chốt inference protocol cho `H*`

Protocol đã chọn là phương án refine toàn bộ edge bằng scorer đã học trong từng
local validation/test graph. Cách này trực quan, lặp lại được và phù hợp để giải
thích trong luận án. Train vẫn dùng sampling để giữ chi phí trong giới hạn.

## 24. Checklist double-check code

### Data và feature

- [ ] Một user chỉ thuộc một experiment split.
- [ ] Observation window đúng ngày 0–34.
- [ ] Transform chỉ fit trên train.
- [ ] Thứ tự `node_ids`, `X` và row của `H0` trùng nhau.

### Hypergraph

- [ ] `H0` chỉ gồm Course, Object và Behavioral.
- [ ] Mọi train membership đều nối train node.
- [ ] Behavioral neighbor chỉ lấy train reference.
- [ ] Không có edge rỗng hoặc singleton.

### HSL

- [ ] Hyperedge sampling cân bằng family/size.
- [ ] Positive và negative candidate không overlap.
- [ ] Mỗi sampled edge giữ ít nhất một positive.
- [ ] Không có isolated node sau refinement.
- [ ] Gradient tới `MembershipScorer` hữu hạn và khác 0.

### Training

- [ ] `pos_weight` chỉ từ train labels.
- [ ] Validation AUC chọn checkpoint.
- [ ] Test không được dùng để chọn epoch/hyperparameter.
- [x] Full experiment mặc định dùng `validation_limit=0` và `test_limit=0`.
- [x] Checkpoint không bị ghi đè giữa các config.
- [x] Artifact hash được kiểm tra khi load checkpoint.
- [x] Inference `H*` deterministic và không phụ thuộc batch size.

### Báo cáo

- [ ] Chạy đủ năm seed `1, 11, 111, 1111, 11111`.
- [ ] Báo cáo mean và standard deviation.
- [x] Ghi rõ shared-weight HGNN.
- [x] Ghi rõ User hyperedge không thuộc main configuration.
- [x] Ghi rõ stochastic train và deterministic inference cho `H*`.

## 25. Lệnh đọc và kiểm tra theo từng tầng

```powershell
# Data và split
python -m unittest tests.test_phase1 tests.test_phase2 -v

# Feature
python -m unittest tests.test_phase3 -v

# Hyperedge và H0
python -m unittest tests.test_phase4 tests.test_phase5 -v

# HGNN
python -m unittest tests.test_phase6 -v

# HSL và sparse gradient
python -m unittest tests.test_phase7 tests.test_behavior_regression -v

# Training và checkpoint
python -m unittest tests.test_phase8 -v

# Test protocol
python -m unittest tests.test_phase9 -v

# Toàn bộ
python -m unittest discover -s tests -v
```

## 26. Kết luận kỹ thuật

Code hiện đã triển khai đường tính toán cốt lõi:

```text
Train:           X + H0 → Z0 → sampled H* → Z* → logits → BCE
                            Z0 ─────────── Z* → contrastive loss
Validation/test: X + H0 → Z0 → deterministic local H* → Z* → prediction
Total loss:      BCE + λ × contrastive loss
```

Các lỗi protocol đã biết về full validation, batch-invariant evaluation,
checkpoint identity, artifact hash và deterministic inference đã được xử lý.
Việc còn lại trước khi lấy số liệu luận án là chọn hyperparameter bằng validation,
chạy đủ năm seed, chạy ablation và tổng hợp bảng kết quả; smoke report hiện tại
không phải kết quả nghiên cứu cuối.
