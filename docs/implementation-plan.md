# Kế hoạch triển khai mô hình HGSL trên XuetangX

> Trạng thái: Phase 0–6 đã hoàn thành. Phase tiếp theo là Phase 7 — Hypergraph
> Structure Learning.

## 1. Phạm vi đã chốt

- Chỉ dùng phần XuetangX có nhãn: **225.642 enrollment, 77.083 user, 247 course**.
- Node dự đoán là enrollment; khóa chuẩn là `enroll_id`.
- Nhãn lấy từ `train_truth.csv` và `test_truth.csv`.
- Cửa sổ quan sát chuẩn là ngày 0–34. Pipeline cho phép tạo thêm cửa sổ
  7, 14, 21 và 28 ngày để chạy early-prediction ablation.
- Chạy năm experiment seed cố định: `1, 11, 111, 1111, 11111`; mỗi seed điều khiển
  cả user split và random state của training/sampling tương ứng.
- Không dùng 351 triệu activity event cho thí nghiệm chính vòng đầu.
- Không tạo package `mooc_hgsl`. Toàn bộ code đặt trực tiếp dưới `src/`.
- `baseline/` chỉ để tham khảo và chạy đối chiếu; code trong đó không được import
  vào mô hình đề xuất.

## 2. Cấu trúc code mục tiêu

```text
src/
  paths.py
  cli.py
  data/
    __init__.py
    schema.py
    audit.py
    preprocess.py
    split.py
    cache.py
  features/
    __init__.py
    engineering.py
    transform.py
  hypergraph/
    __init__.py
    course.py
    object.py
    behavioral.py
    user.py
    construction.py
    audit.py
    io.py
  models/
    __init__.py
    hgnn.py
    hyperedge_sampling.py
    incident_node_sampling.py
    refinement.py
    classifier.py
    model.py
  losses/
    __init__.py
    contrastive.py
    objective.py
  training/
    __init__.py
    trainer.py
    evaluator.py
```

`run.py` là entry point duy nhất ở project root. Các lệnh dự kiến:

```text
python run.py audit-data
python run.py prepare-data
python run.py split-data
python run.py build-features
python run.py build-hyperedges
python run.py build-hypergraph
python run.py train-baseline --seed 1 --epochs 1
python run.py train
python run.py evaluate
```

## 3. Cách tổ chức artifact và cache

Không tạo nhiều tầng thư mục. Toàn bộ artifact của dataset 247 course đặt tại:

```text
data/processed/xuetangx_247/
  nodes.parquet
  events_35d.parquet
  users.parquet
  courses.parquet
  splits.parquet
  features_raw.parquet
  X_base.npy
  feature_transform.joblib
  structural_memberships.parquet
  behavioral_neighbors.npz
  hyperedge_audit.json
  hyperedge_manifest.json
  H0_train_seed_1.npz
  H0_train_seed_11.npz
  H0_train_seed_111.npz
  H0_train_seed_1111.npz
  H0_train_seed_11111.npz
  train_node_index.parquet
  validation_memberships.parquet
  test_memberships.parquet
  hyperedges.parquet
  hypergraph_audit.json
  hypergraph_manifest.json
  audit.json
  manifest.json
```

Nguyên tắc cache:

- `manifest.json` lưu hash nguồn, schema version, feature version, experiment seed và
  tham số tạo hypergraph.
- Chỉ rebuild artifact chịu ảnh hưởng khi input hoặc tham số của nó thay đổi.
- File tạm dùng hậu tố `.part`; chỉ đổi thành tên chính thức sau khi kiểm tra xong.
- Không dùng pickle cho graph/data lớn. Feature dense dùng `.npy`, incidence sparse
  dùng `.npz`, bảng dùng Parquet; `joblib` chỉ lưu transformer nhỏ do project tự tạo.

## Phase 0 — Khóa data contract

### Việc thực hiện

- Tạo schema cố định cho raw log, truth, user và course.
- Phân biệt hai field:
  - `source_partition`: train/test do nguồn cung cấp.
  - `experiment_split`: train/validation/test của thí nghiệm.
- Định nghĩa `object_key=(course_id, object)`.
- Đổi tên feature thành `distinct_observed_objects` để phản ánh object missing.
- Khóa action vocabulary 23 chiều; `close_info` được giữ dù bằng 0 trong phần có nhãn.

### Điều kiện hoàn thành

- Schema, action vocabulary và quy ước tên được khai báo một lần trong
  `src/data/schema.py`.
- Không có module khác tự định nghĩa lại schema.

## Phase 1 — Data audit và canonical events

### Việc thực hiện

- Gộp train/test log và hai file truth theo `enroll_id`.
- Chuẩn hóa `username` thành `user_id` và parse timestamp.
- Join `course_info.start`, tính `course_day`, giữ ngày 0–34.
- Giữ event thiếu object cho behavioral features; không tạo object incidence cho
  các event này.
- Audit:
  - `enroll_id` duy nhất và `(user_id, course_id)` ánh xạ một-một trong dữ liệu hiện tại;
  - label chỉ thuộc `{0,1}`;
  - unknown action bằng 0;
  - timestamp lỗi, event trước course start, sau course end;
  - course duration dưới 35 ngày;
  - exact duplicate và missing theo field.

### Artifact

```text
data/processed/xuetangx_247/
  nodes.parquet
  events_35d.parquet
  users.parquet
  courses.parquet
  audit.json
```

### Điều kiện hoàn thành

- Có đúng 225.642 node và 247 course.
- Mỗi node có đúng một label.
- Audit lỗi nghiêm trọng phải làm pipeline dừng, không chỉ ghi warning.

## Phase 2 — User-disjoint split

### Việc thực hiện

- Group toàn bộ enrollment theo `user_id`.
- Chia user theo tỷ lệ mục tiêu 64/16/20 cho từng seed trong
  `{1, 11, 111, 1111, 11111}`.
- Dùng thuật toán gán group có ràng buộc để đồng thời giảm sai lệch:
  - số user;
  - số enrollment;
  - tỷ lệ dropout.
- Khóa split ngay sau khi tạo; không đổi seed dựa trên kết quả mô hình.

### Artifact

```text
splits.parquet
split_manifest.json
```

`splits.parquet` gồm:

```text
seed, node_id, enroll_id, user_id, source_partition, experiment_split
```

### Điều kiện hoàn thành

- Trong từng seed, user overlap giữa mọi cặp split bằng 0.
- Trong từng seed, tổng node của ba split bằng 225.642.
- Báo cáo số user, enrollment và label của từng split.

### Kết quả đã khóa cho năm seed

| Split | User | Enrollment | Dropout | Dropout rate |
|---|---:|---:|---:|---:|
| Train | 49.333 | 144.543 | 109.625 | 75,84% |
| Validation | 12.333 | 36.028 | 27.374 | 75,98% |
| Test | 15.417 | 45.071 | 34.134 | 75,73% |

User overlap bằng 0; tổng cộng 77.083 user và 225.642 enrollment.
Các seed có cùng số lượng và tỷ lệ nhãn nhưng khác thành viên trong từng tập.

## Phase 3 — Feature engineering

### 3.1 Feature chính `X_base`

| Khối | Số chiều | Cách tạo |
|---|---:|---|
| `day_00..day_34` | 35 | Count event theo `course_day` |
| `action_*` | 23 | Tổng one-hot action của event |
| `session_count` | 1 | Số session khác nhau |
| `distinct_observed_objects` | 1 | Số `object_key` quan sát được |
| Tổng | 60 | |

### 3.2 Feature ablation

```text
event_count, active_days, active_span_days, first_active_day,
last_active_day, days_since_last_activity, active_day_ratio, has_activity
```

Metadata user/course chỉ dùng trong context/fairness ablation, không đưa vào main
`X_base`.

### 3.3 Transform

- Count dùng `log1p` rồi standardize.
- Median, scaler và categorical vocabulary chỉ fit trên train.
- Validation/test chỉ gọi `transform`.
- Node không activity được điền count bằng 0 và có `has_activity=0`.
- Kiểm tra:

```text
event_count = sum(day_*)
event_count = sum(action_*)
unknown_action_count = 0
```

### Artifact

```text
features_raw.parquet
X_base.npy
feature_transform.joblib
feature_manifest.json
```

`X_base.npy` có layout `[seed_index, node_id, feature_index]`. Không lưu thêm ba
bản sao train/validation/test; dùng `splits.parquet` để lấy node index tương ứng.

Canonical event 35 ngày được dùng lại để tạo `X^(7)`, `X^(14)`, `X^(21)` và
`X^(28)`; không đọc lại raw CSV.

## Phase 4 — Xác định các nhóm hyperedge

### 4.1 Course hyperedge

Mỗi course tạo một hyperedge:

```text
e_course(c) = {node i | course_id(i) = c}
```

- Có tối đa 247 course hyperedge.
- Giữ tất cả hyperedge có ít nhất hai node.
- Audit size, median, P90, P95, P99 và max trước khi quyết định weighting/capping.
- Trong train graph, hyperedge chỉ chứa train nodes. Với một validation/test target,
  local hyperedge chỉ gồm target đó và các train-reference node cùng course; không
  nối nhiều target nodes vào cùng một hyperedge.

### 4.2 Object hyperedge

Mỗi `object_key=(course_id, object)` tạo một hyperedge:

```text
e_object(o) = {node i | i tương tác với object_key o}
```

- Event thiếu object không tạo incidence.
- Dùng chung luật `cardinality >= 2` cho video, bài tập và diễn đàn.
- Không loại toàn bộ forum; các forum object không phải singleton vẫn được giữ.
- `object_type` được lưu thành metadata hyperedge để chạy ablation theo loại.
- Với validation/test, một object edge chỉ được tạo giữa target và train-reference
  nodes có cùng `object_key`; object chỉ xuất hiện ở target không được dùng để tạo
  hyperedge mới.

### 4.3 Behavioral hyperedge

- Dùng cosine similarity trên `X_base` đã transform.
- Với mỗi anchor node, tạo:

```text
e_behavior(i) = {i} ∪ k-nearest-neighbors(i)
```

- Train anchor tìm neighbor trong train.
- Validation/test anchor chỉ tìm neighbor trong **train reference set**. Cách này
  giữ thiết lập inductive và không phụ thuộc các node test khác.
- Loại hyperedge trùng hoàn toàn.
- Chọn `k` trên validation, dự kiến thử `{5, 10, 20}`.
- Không dùng label trong tìm kiếm neighbor.

### 4.4 User hyperedge

Không đưa User hyperedge vào main `H0` vòng đầu vì nhiều enrollment của cùng user
có course start khác nhau; nối trực tiếp có thể đưa hành vi tương lai vào enrollment
trước đó.

Module `src/hypergraph/user.py` chỉ phục vụ ablation sau này. Khi bật, một node chỉ
được nối với enrollment của cùng user đã quan sát được trước prediction cutoff của
node đó.

### Điều kiện hoàn thành

Mỗi family phải xuất cùng một audit format:

```text
family, hyperedges, median_size, p90, p95, p99, max_size, singletons
```

Audit global dùng mô tả dataset; mọi threshold và quyết định model chỉ dựa trên train.

### Kết quả triển khai Phase 4

- `structural_memberships.parquet` lưu Course/Object incidence ứng viên; object
  dùng khóa ghép `(course_id, object_id)` và metadata `object_type`.
- Có 247 course; 80.745 object key gồm 12.239 video, 11.119 bài tập và 57.387
  diễn đàn. Có 58.324 object singleton toàn cục.
- Sau khi lọc `cardinality >= 2` trên train, mỗi seed còn khoảng 21,9 nghìn
  Object hyperedge; cả 247 Course hyperedge đều được giữ.
- `behavioral_neighbors.npz` lưu 20 train-neighbor gần nhất cho mọi anchor và cả
  năm seed. Các cấu hình `k={5,10,20}` dùng prefix của cùng kết quả này.
- Validation/test chỉ nhận train node làm neighbor. Label không được đọc khi tạo
  Course, Object hoặc Behavioral candidates.
- Hyperedge Behavioral trùng hoàn toàn đã được audit; việc khử trùng khi tạo ma
  trận incidence thực hiện ở Phase 5.

## Phase 5 — Materialize node features và initial hypergraph

### Node features

```text
X ∈ R^(N×60)
```

Thứ tự hàng của `X` phải trùng tuyệt đối với `node_id` trong node index.

### Initial hypergraph

Main configuration:

```text
H0 = [H_course | H_object | H_behavior]
```

`H0` là sparse binary incidence matrix:

```text
H0[i,e] = 1 nếu node i thuộc hyperedge e, ngược lại 0
```

Không tạo dense `H`. Lưu trực tiếp trong cùng thư mục processed:

```text
data/processed/xuetangx_247/
  H0_train_seed_<seed>.npz
  train_node_index.parquet
  validation_memberships.parquet
  test_memberships.parquet
  hyperedges.parquet
  hypergraph_audit.json
  hypergraph_manifest.json
```

`hyperedges.parquet` chứa `hyperedge_id`, `family`, `source_key`, `object_type`,
`size` và `weight`.

Graph được dùng theo hai hình thức:

- Train graph: chỉ có train nodes.
- Validation/test: mỗi target có một local hypergraph độc lập gồm target và các
  train-reference node liên quan theo course, object hoặc behavioral similarity.
- Các local graph được batch như những graph rời nhau; target này không nhận message
  từ target khác.

Không sử dụng validation/test label khi xây graph.

### Điều kiện hoàn thành

- Số hàng của train `H0` khớp train node index.
- Mỗi local evaluation graph có đúng một target mask; thứ tự feature lấy từ
  `X_base.npy` theo `seed_index` và global `node_id`.
- Không có hyperedge rỗng hoặc singleton trong main `H0`.
- Mọi incidence và feature đều có manifest/hash để cache có thể tái sử dụng.

### Kết quả triển khai Phase 5

- Mỗi seed có 144.543 train node và một ma trận CSR `uint8`; số hyperedge nằm
  trong khoảng 152.572–154.592, số incidence khoảng 3,36–3,38 triệu.
- `behavioral_k=10` là candidate mặc định để materialize, chưa phải kết quả tuning.
  CLI cho phép dựng lại với `k=5` hoặc `k=20`; quyết định cuối dựa trên validation.
- `hyperedges.parquet` lưu metadata và weight `1.0`; `train_node_index.parquet`
  ánh xạ chính xác hàng local của `H0` về global `node_id` trong `X_base`.
- Local membership không nhân bản toàn bộ train incidence. Course/Object trỏ tới
  train hyperedge; object chỉ có một train reference lưu trực tiếp node đó;
  Behavioral trỏ tới anchor trong `behavioral_neighbors.npz`.
- `src/hypergraph/io.py` đọc trực tiếp `(X_train, H0_train)` và dựng local graph
  có đúng một target mask. Mọi reference trong local graph đều thuộc train.

## Phase 6 — Baseline HGNN trên `H0`

Triển khai HGNN chuẩn trước khi thêm structure learning:

```text
Z0 = HGNN(X, H0)
prediction = Classifier(Z0)
loss = BCE
```

Mục tiêu phase này là xác nhận data, incidence normalization, forward/backward và
metric hoạt động đúng. Đây cũng là baseline trực tiếp để đo contribution của HSL.

### Kết quả triển khai Phase 6

- Cài sparse propagation đúng công thức
  `Dv^-1/2 H W De^-1 H^T Dv^-1/2`; không materialize ma trận truyền dense.
- Baseline gồm hai HGNN layer, ReLU/dropout và linear dropout classifier.
- `pos_weight=negative/positive` chỉ tính từ train label.
- Validation local graph được batch block-diagonal; các target trong cùng batch
  không truyền message cho nhau.
- Có AUC, AUPRC, F1, precision, recall; early stopping theo validation AUC và
  checkpoint chứa hash `hypergraph_manifest.json`.
- Smoke run một epoch trên toàn bộ train graph seed 1 đã đạt loss/gradient hữu
  hạn. Validation 128 target chỉ dùng kiểm tra kỹ thuật, không phải kết quả nghiên cứu.
- Full validation dùng `--validation-limit 0`; thí nghiệm đủ năm seed và tuning
  hyperparameter vẫn thực hiện ở Phase 9.

## Phase 7 — Hypergraph Structure Learning

Theo flowchart:

```text
H0 + X
→ HGNN tạo Z0
→ Hyperedge Sampling
→ Incident Node Sampling
→ Refine Hypergraph H*
→ HGNN tạo Z*
```

Các bước triển khai:

1. Sample hyperedge theo family và size để tránh chỉ chọn hyperedge lớn.
2. Sample incident nodes trong từng hyperedge.
3. Tạo hyperedge representation từ embedding của incident nodes.
4. Tính membership score giữa node và hyperedge.
5. Giữ/refine incidence theo threshold hoặc top-r được chọn trên validation.
6. Sinh sparse `H*`, không materialize ma trận dense.

Phase đầu chỉ refine membership của hyperedge đã có; chưa tự sinh family mới.

## Phase 8 — Loss và classifier

```text
L_total = L_BCE + λ L_CL
```

- `L_BCE`: phân loại dropout từ `Z*`.
- `L_CL`: contrastive alignment giữa hai view `Z0` và `Z*` của cùng node.
- `λ` chọn trên validation.
- Class imbalance được xử lý bằng `pos_weight` tính từ train; không dùng phân bố
  validation/test.

## Phase 9 — Training, evaluation và ablation

### Training

- Early stopping theo validation AUC.
- Lưu checkpoint tốt nhất và đầy đủ seed/hyperparameter/hash artifact.
- Chạy đúng năm seed `{1, 11, 111, 1111, 11111}` và báo cáo mean ± std.

### Metric

```text
AUC, AUPRC, F1, precision, recall
```

### Ablation bắt buộc

```text
MLP(X)
HGNN(H0) không HSL
HGNN + HSL
Course only
Course + Object
Course + Object + Behavioral
Object: video / video+problem / all non-singleton
X_base / X_augmented
7 / 14 / 21 / 28 / 35 observation days
```

SIG-Net và MST-GCN phải chạy lại trên cùng 247 course và cùng split. Kết quả được
gọi là reimplementation trên XuetangX-247, không so trực tiếp với số công bố trên
XuetangX 1.213 course.

## Phase 10 — Kiểm thử và tiêu chí hoàn thành

### Unit test

- Feature formula và invariant.
- Split không trùng user.
- Composite `object_key`.
- Sparse incidence shape/cardinality.
- HGNN normalization.
- Sampling và refinement không tạo index sai.
- Total loss và gradient.

### Integration test

- Chạy toàn pipeline trên một sample nhỏ.
- Chạy một epoch CPU.
- Tạo lại artifact từ cùng seed phải cho cùng split và manifest.

### Hoàn thành dự án vòng đầu khi

1. Pipeline từ raw data đến `X` và `H0` chạy lại được bằng CLI.
2. Không có leakage theo user, transform hoặc graph construction.
3. Baseline HGNN và HGSL train/evaluate thành công.
4. Có bảng main result, ablation và audit hypergraph.
5. Mọi artifact lớn đọc lại từ cache, không preprocess 40 triệu event ở mỗi lần chạy.

## Thứ tự thực hiện ngay

```text
Phase 0–1: schema + audit + canonical events
→ Phase 2: khóa split
→ Phase 3: tạo X_base
→ Phase 4–5: tạo từng family và H0
→ Phase 6: HGNN baseline
→ Phase 7–8: HSL + loss
→ Phase 9–10: experiments + tests
```

## 11. Danh sách công việc triển khai

Các task dưới đây là thứ tự thực hiện thực tế. Không bắt đầu phase sau khi điều kiện
hoàn thành của phase trước chưa đạt.

### Milestone A — Data và feature sẵn sàng

| ID | File chính | Công việc | Đầu ra/kiểm tra |
|---|---|---|---|
| A01 | `src/paths.py` | Khai báo đường dẫn raw, processed, output; không hard-code ở module khác | Test mọi path nằm trong project root |
| A02 | `src/data/schema.py` | Khai báo raw columns, 23 actions, observation days và schema version | Action không trùng; đủ 23 action |
| A03 | `src/data/audit.py` | Kiểm tra file, hash, row count, schema, missing, duplicate và timestamp | `audit.json`; lỗi nghiêm trọng trả exit code khác 0 |
| A04 | `src/data/preprocess.py` | Tạo node index, join truth/user/course, tạo canonical event ngày 0–34 | 225.642 node, 247 course, 40.558.640 event |
| A05 | `src/data/split.py` | Tạo user summary và chia group 64/16/20 cho năm seed đã khóa | User overlap bằng 0 trong từng seed |
| A06 | `src/data/split.py` | Ghi riêng `seed`, `source_partition` và `experiment_split` | Mỗi seed đủ node; split manifest cố định |
| A07 | `src/features/engineering.py` | Aggregate 35 daily count, 23 action count, session và observed object | `features_raw.parquet`; invariant đúng |
| A08 | `src/features/transform.py` | Fit `log1p`/standardization trên train; transform các split còn lại | Transformer không đọc val/test khi fit |
| A09 | `src/features/engineering.py` | Hỗ trợ cắt cửa sổ 7/14/21/28/35 từ canonical event | Không đọc lại raw CSV |
| A10 | `src/data/cache.py` | Ghi manifest, hash và kiểm tra khả năng reuse | Chạy lại không đổi input phải cache hit |

Milestone A hoàn thành khi `python run.py build-features` tạo được `X_base.npy`,
split cố định và báo cáo audit mà không cần code mô hình.

### Milestone B — Initial hypergraph `H0`

| ID | File chính | Công việc | Đầu ra/kiểm tra |
|---|---|---|---|
| B01 | `src/hypergraph/course.py` | Tạo course incidence trên train | Không có singleton; key duy nhất |
| B02 | `src/hypergraph/object.py` | Tạo composite object key và incidence train | Không nối object khác course |
| B03 | `src/hypergraph/object.py` | Giữ video/problem/forum có cardinality ≥ 2 | Forum non-singleton không bị loại nhầm |
| B04 | `src/hypergraph/behavioral.py` | Fit cosine neighbor index bằng train `X_base` | Index không nhận label |
| B05 | `src/hypergraph/behavioral.py` | Tạo anchor-plus-k-neighbor edge; loại edge trùng | Thử `k={5,10,20}` bằng validation |
| B06 | `src/hypergraph/user.py` | Cài causal user-edge builder nhưng để mặc định tắt | Main configuration không chứa user edge |
| B07 | `src/hypergraph/audit.py` | Thống kê số edge, size, P90/P95/P99/max/singleton theo family | Có bảng global và train riêng |
| B08 | `src/hypergraph/construction.py` | Ghép `H_course`, `H_object`, `H_behavior` thành sparse `H0_train` | Không tạo dense matrix |
| B09 | `src/hypergraph/construction.py` | Tạo local memberships cho từng val/test target với train reference | Mỗi local graph đúng một target; không target-target edge |
| B10 | `src/data/cache.py` | Ghi `.npz`, membership table và hyperedge metadata | Shape/hash khớp node index |

Milestone B hoàn thành khi `python run.py build-hypergraph` tạo được `H0_train.npz`
và local memberships, đồng thời toàn bộ audit hyperedge đạt invariant.

### Milestone C — HGNN baseline

| ID | File chính | Công việc | Đầu ra/kiểm tra |
|---|---|---|---|
| C01 | `src/models/hgnn.py` | Cài sparse hypergraph convolution và degree normalization | So khớp phép tính trên graph đồ chơi |
| C02 | `src/models/classifier.py` | Linear head tạo dropout logit | Output shape `[batch]` |
| C03 | `src/models/model.py` | Ghép `X`, `H0`, HGNN và classifier | Forward không dùng label |
| C04 | `src/losses/objective.py` | BCEWithLogitsLoss; `pos_weight` chỉ tính từ train | Loss hữu hạn và có gradient |
| C05 | `src/training/trainer.py` | Train/validation loop, early stopping và checkpoint | Chạy được một epoch CPU/GPU |
| C06 | `src/training/evaluator.py` | AUC, AUPRC, F1, precision, recall | So khớp sklearn trên dữ liệu nhỏ |

Milestone C hoàn thành khi HGNN không HSL train ổn định và tạo được kết quả baseline.
Nếu baseline này chưa chạy đúng thì chưa triển khai structure learning.

### Milestone D — Hypergraph Structure Learning

| ID | File chính | Công việc | Đầu ra/kiểm tra |
|---|---|---|---|
| D01 | `src/models/hyperedge_sampling.py` | Sample cân bằng theo family và bucket kích thước | Edge lớn không chiếm toàn bộ batch |
| D02 | `src/models/incident_node_sampling.py` | Sample positive incident nodes và negative nodes | Không lẫn positive vào negative |
| D03 | `src/models/refinement.py` | Tạo edge embedding và tính node-edge membership score | Score đúng shape, không dense toàn graph |
| D04 | `src/models/refinement.py` | Refine incidence bằng top-r/threshold chọn trên validation | `H*` sparse; không edge rỗng |
| D05 | `src/models/model.py` | Chạy HGNN lần hai trên `H*` để tạo `Z*` | Forward trả `Z0`, `Z*`, logits và audit stats |
| D06 | `src/losses/contrastive.py` | Contrastive loss giữa hai view của cùng node | Positive/negative mask đúng |
| D07 | `src/losses/objective.py` | Tính `L_total=L_BCE+λL_CL` | `λ=0` khớp HGNN baseline |

Milestone D hoàn thành khi HGSL chạy end-to-end trên sample nhỏ, gradient đi qua
refinement và không materialize ma trận node-hyperedge dense.

### Milestone E — Thí nghiệm và bàn giao

| ID | File/đầu ra | Công việc | Điều kiện hoàn thành |
|---|---|---|---|
| E01 | `src/cli.py`, `run.py` | Hoàn thiện sáu lệnh CLI; validate tham số và log rõ cache hit/miss | Mỗi phase chạy độc lập được |
| E02 | `outputs/runs/` | Lưu checkpoint, seed, metric, hash data/graph và tham số | Có thể truy lại đúng input của mỗi run |
| E03 | Main experiment | Chạy HGSL với cấu hình được chọn trên validation | Test chỉ đánh giá một lần sau khi khóa cấu hình |
| E04 | Ablation | Chạy feature, hyperedge, HSL và observation-window ablation | Cùng split, metric và seed |
| E05 | Baseline | Chạy SIG-Net/MST-GCN trên cùng XuetangX-247 | Không so trực tiếp với published XuetangX-1213 |
| E06 | `tests/` | Unit, integration, leakage và determinism tests | Toàn bộ test pass |
| E07 | Documentation | Cập nhật README, lệnh chạy và bảng artifact | Người khác chạy lại từ raw data được |

## 12. Thứ tự thí nghiệm

Chỉ mở rộng mô hình sau khi cấu hình trước đó chạy ổn định:

1. Logistic/MLP trên `X_base` để kiểm tra feature và split.
2. HGNN với Course hyperedge.
3. HGNN với Course + Object.
4. HGNN với Course + Object + Behavioral.
5. Thêm HSL và contrastive loss.
6. Chạy ablation object type, feature set và observation window.
7. Chạy năm seed và baseline đối chiếu.

Việc chọn feature, family, `k`, top-r/threshold, `λ`, hidden dimension và số layer
chỉ dùng validation. Test không được dùng để thay đổi thiết kế.

## 13. Phụ thuộc phần mềm cần bổ sung

| Nhóm | Package | Mục đích |
|---|---|---|
| Data | `duckdb`, `ijson` | Đọc/aggregate dữ liệu lớn |
| Numeric | `numpy`, `scipy` | Dense feature và sparse incidence |
| ML utility | `scikit-learn`, `joblib` | kNN, scaler và metric |
| Model | `torch` | HGNN, HSL và training |
| Test | built-in `unittest` | Unit/integration test |

Không thêm DGL hoặc PyTorch Geometric ở vòng đầu. Hypergraph operation được cài
bằng PyTorch sparse để giảm phụ thuộc và giữ đúng cấu trúc `H` của mô hình.
