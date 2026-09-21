# Diễn giải kỹ thuật pipeline HGSL trên XuetangX-247

> Tài liệu này mô tả đúng implementation hiện tại của project: dữ liệu đi qua từng
> khối như thế nào, tensor thay đổi ra sao, công thức nào được dùng và file/hàm nào
> hiện thực mỗi bước. Các kết quả smoke test chỉ xác nhận pipeline chạy được, không
> được xem là kết quả thực nghiệm chính thức.

## Mục lục

1. [Preprocessing](#1-preprocessing)
2. [Feature engineering](#2-feature-engineering)
3. [Hypergraph construction](#3-hypergraph-construction)
4. [Hyperedge: biểu diễn, lọc và sampling](#4-hyperedge-biểu-diễn-lọc-và-sampling)
5. [Initial hypergraph H0 và node features X](#5-initial-hypergraph-h0-và-node-features-x)
6. [Hypergraph Structure Learning](#6-hypergraph-structure-learning)
7. [Contrastive loss](#7-contrastive-loss)
8. [BCE loss](#8-bce-loss)
9. [Total loss](#9-total-loss)
10. [Quá trình train](#10-quá-trình-train)
11. [Bảng tra nhanh code](#11-bảng-tra-nhanh-code-theo-10-khối)
12. [Điểm cần ghi rõ trong luận án](#12-các-điểm-phải-ghi-rõ-khi-mô-tả-trong-luận-án)

## 0. Bức tranh tổng thể

```text
Raw XuetangX CSV
  │
  ├─ preprocessing + audit + user-disjoint split
  │      ├─ nodes.parquet
  │      ├─ events_35d.parquet
  │      ├─ users.parquet / courses.parquet
  │      └─ splits.parquet
  │
  ├─ feature engineering
  │      ├─ X_base.npy       : behavioral, 60 chiều
  │      └─ X_context.npy    : user + course context, 36 chiều
  │
  ├─ hyperedge construction
  │      ├─ Course hyperedges
  │      ├─ Object hyperedges
  │      └─ Behavioral hyperedges, kNN chỉ trên X_base
  │
  ├─ materialize H0 + chọn node feature X
  │
  ├─ Z0 = HGNN(X, H0)
  │
  ├─ hypergraph structure learning
  │      └─ H0 → sampled memberships → H*
  │
  ├─ Z* = HGNN(X, H*)
  │
  ├─ dropout logits = Linear(Z*)
  │
  └─ Ltotal = LBCE + λLCL
```

### Ký hiệu dùng trong tài liệu

| Ký hiệu | Ý nghĩa |
|---|---|
| `N` | Số enrollment node trong graph đang xét |
| `E` | Số hyperedge |
| `D` | Số chiều node feature: 60, 75, 81 hoặc 96 |
| `d` | Hidden/embedding dimension, mặc định 64 |
| `H0 ∈ R^(N×E)` | Sparse incidence ban đầu |
| `H* ∈ R^(N×E)` | Sparse weighted incidence sau refinement |
| `X ∈ R^(N×D)` | Node-feature matrix |
| `Z0, Z* ∈ R^(N×d)` | Node embedding trước/sau refinement |

Nhãn được khóa như sau:

```text
truth = 1: dropout
truth = 0: non-dropout
```

---

## 1. Preprocessing

### 1.1 Mục tiêu

Preprocessing chuyển các CSV gốc thành bốn bảng canonical có kiểu dữ liệu ổn định,
khóa identity của node và tạo cửa sổ quan sát 35 ngày. Một node tương ứng với một
enrollment, không phải một user hay một course.

### 1.2 Đầu vào

```text
train_log.csv, test_log.csv
train_truth.csv, test_truth.csv
user_info.csv
course_info.csv
```

Raw train/test do nguồn cung cấp được lưu trong `source_partition`. Đây không phải
train/test của thí nghiệm. Split thí nghiệm được tạo lại và lưu riêng trong
`experiment_split`.

#### 1.2.1 `source_partition`: bản ghi đến từ file gốc nào?

`source_partition` là thông tin **nguồn gốc dữ liệu** (data provenance). Biến này
được tạo ngay khi đọc các CSV do bộ dữ liệu cung cấp và chỉ có hai giá trị:

| CSV gốc | `source_partition` được gán |
|---|---|
| `train_log.csv`, `train_truth.csv` | `train` |
| `test_log.csv`, `test_truth.csv` | `test` |

Ví dụ, nếu enrollment `773` được đọc từ `train_log.csv` thì mọi event và node của
enrollment đó mang `source_partition = 'train'`. Giá trị này không đổi khi chạy
thí nghiệm với seed khác.

Trong code, `source_partition` có ba nhiệm vụ chính:

1. Ghép log với đúng file nhãn. Điều kiện join là cặp
   `(enroll_id, source_partition)`, không chỉ dùng riêng `enroll_id`:

   ```sql
   FROM enrollment_meta e
   JOIN labels l USING(enroll_id, source_partition)
   ```

2. Giữ lineage để có thể truy ngược một node/event về file CSV ban đầu.
3. Audit số mẫu từ raw train và raw test sau mỗi bước xử lý, giúp phát hiện mất
   hoặc trộn sai dữ liệu.

`source_partition` **không được dùng** để quyết định node nào được đưa vào train,
validation hoặc test của mô hình. Nó cũng không phải node feature và không được
đưa vào ma trận `X`.

> Lưu ý: từ “partition” trong tên biến này chỉ mang nghĩa nhóm nguồn `train/test`.
> Đây là một cột chuỗi trong bảng, không có nghĩa mỗi giá trị bắt buộc được lưu
> thành một thư mục Parquet vật lý riêng.

#### 1.2.2 `experiment_split`: node được dùng ở bước nào của thí nghiệm?

`experiment_split` là **vai trò của node trong một lần chạy thí nghiệm**. Biến này
được sinh sau preprocessing bởi `src/data/split.py`, có ba giá trị:

| `experiment_split` | Dùng để làm gì? |
|---|---|
| `train` | Fit scaler/encoder, cập nhật trọng số mô hình |
| `validation` | Chọn checkpoint, hyperparameter và early stopping |
| `test` | Đánh giá cuối cùng; không dùng để fit mô hình |

Việc chia tập được thực hiện theo `user_id`, không chia độc lập từng enrollment.
Vì vậy, trong cùng một seed, toàn bộ enrollment của một user luôn thuộc cùng một
`experiment_split`. Quy tắc này ngăn hành vi của cùng một người xuất hiện ở cả
train và test.

Mỗi seed tạo một phép chia mới. Do đó một node có thể thuộc `validation` ở seed 1
nhưng thuộc `train` ở seed 11. Ví dụ từ artifact hiện tại:

| `node_id` | `enroll_id` | `user_id` | `source_partition` | seed 1 | seed 11 |
|---:|---:|---:|---|---|---|
| 1 | 773 | 1544995 | `train` | `validation` | `train` |
| 3 | 775 | 1520977 | `test` | `train` | `train` |
| 4 | 776 | 561867 | `train` | `validation` | `train` |

Dòng thứ hai cho thấy rõ: một node có `source_partition = 'test'` vẫn có thể có
`experiment_split = 'train'`. Điều này không phải lỗi. Bộ dữ liệu cục bộ có nhãn
ở cả `train_truth.csv` và `test_truth.csv`, nên pipeline hợp nhất toàn bộ node có
nhãn rồi tạo lại phép chia user-disjoint phù hợp với protocol nghiên cứu.

```text
Nguồn dữ liệu gốc
├── train_log + train_truth ──> source_partition = train ─┐
└── test_log  + test_truth  ──> source_partition = test  ─┤
                                                          │
                                  hợp nhất toàn bộ node có nhãn
                                                          │
                                      split theo user + seed
                                                          │
                      ┌───────────────────┬───────────────┴─────┐
                      v                   v                     v
            experiment_split=train   validation              test
```

Hai biến có thể nhớ ngắn gọn như sau:

| Câu hỏi | Biến trả lời | Có đổi theo seed? | Có điều khiển train/evaluate? |
|---|---|---:|---:|
| “Bản ghi này ban đầu đến từ file nào?” | `source_partition` | Không | Không |
| “Trong lần chạy này, node có vai trò gì?” | `experiment_split` | Có | Có |

Quy tắc sử dụng trong toàn pipeline:

```text
truy vết và kiểm tra dữ liệu gốc  -> source_partition
chọn dữ liệu train/val/test       -> experiment_split + seed
```

Không đưa một trong hai biến vào `X`: `source_partition` có thể mang dấu vết cách
bộ dữ liệu được phát hành, còn `experiment_split` tiết lộ trực tiếp vai trò của
mẫu. Dùng chúng làm feature sẽ tạo shortcut hoặc data leakage.

#### 1.2.3 Các file `.parquet` là gì?

Parquet là định dạng file bảng dạng cột (columnar), tương tự CSV ở chỗ đều lưu dữ
liệu dạng hàng/cột, nhưng Parquet là định dạng nhị phân có schema và nén. Nó không
phải tensor đầu vào mô hình, cũng không phải một loại hypergraph.

| Đặc điểm | CSV | Parquet |
|---|---|---|
| Cách lưu | Văn bản theo hàng | Nhị phân theo cột |
| Kiểu dữ liệu | Thường phải suy luận lại khi đọc | Lưu sẵn kiểu `INTEGER`, `TIMESTAMP`, `VARCHAR`, ... |
| Kích thước | Thường lớn hơn | Có nén, thường nhỏ hơn |
| Đọc vài cột trong bảng lớn | Vẫn phải xử lý nhiều dữ liệu | Có thể chỉ đọc các cột cần thiết |
| Phù hợp với pipeline này | Dữ liệu nguồn | Bảng canonical sau preprocessing |

Lợi ích quan trọng nhất ở đây là `events_35d.parquet` có hơn 40 triệu dòng. DuckDB
có thể query trực tiếp file, chỉ đọc các cột và row group cần thiết mà không phải
nạp toàn bộ bảng vào RAM.

Mỗi file Parquet trong preprocessing tương đương một bảng có mục đích riêng:

| File | Một dòng biểu diễn | Số dòng hiện tại | Nội dung chính |
|---|---|---:|---|
| `nodes.parquet` | Một enrollment/node | 225.642 | `node_id`, khóa user/course, nhãn, nguồn gốc |
| `events_35d.parquet` | Một event trong ngày 0–34 | 40.558.640 | node, thời gian, action, object, `course_day` |
| `users.parquet` | Một user có liên quan | 77.083 | gender, education, birth year |
| `courses.parquet` | Một course có liên quan | 247 | thời gian, loại và category của course |
| `splits.parquet` | Một phép gán `(seed, node)` | 1.128.210 | node thuộc train/validation/test ở seed nào |

`splits.parquet` có số dòng bằng `225.642 × 5 = 1.128.210`, vì mỗi node xuất hiện
một lần cho từng seed. Ngược lại, node chỉ xuất hiện một lần trong `nodes.parquet`.

Ví dụ đọc train split của seed 1 mà không cần chuyển Parquet thành CSV:

```sql
SELECT node_id, user_id, source_partition, experiment_split
FROM read_parquet('data/processed/splits.parquet')
WHERE seed = 1 AND experiment_split = 'train';
```

Luồng thay đổi định dạng dữ liệu ở giai đoạn này là:

```text
CSV gốc (log, truth, user, course)
    -> chuẩn hóa tên cột và kiểu dữ liệu
    -> join/kiểm tra khóa
    -> lọc observation window 35 ngày
    -> các bảng canonical .parquet
    -> splits.parquet được tạo riêng cho 5 seed
```

### 1.3 Tạo node table

Log được group theo `enroll_id` để lấy `user_id`, `course_id` và source partition;
sau đó join với truth. `node_id` là số nguyên liên tục từ 0 đến 225.641, sắp theo
`enroll_id`.

```text
nodes.parquet
  node_id
  enroll_id
  user_id
  course_id
  source_partition
  label
```

Invariant chính:

```text
rows = distinct(node_id) = distinct(enroll_id) = 225642
node_id ∈ [0, 225641]
label ∈ {0,1}
```

### 1.4 Chuẩn hóa metadata

`users.parquet` chỉ giữ 77.083 user tham gia tập có nhãn:

```text
user_id, gender, education, birth_year
```

`courses.parquet` chỉ giữ 247 course liên quan:

```text
metadata_id, course_id, course_start, course_end, course_type, category
```

Chuỗi rỗng được đổi thành null; ID, timestamp, birth year và course type được cast
sang kiểu rõ ràng.

### 1.5 Tạo canonical event và observation window

Mỗi event được join với node và course, sau đó tính:

```text
course_day = date(event_time) - date(course_start)
```

Chỉ giữ:

```text
0 ≤ course_day ≤ 34
```

Kết quả là 40.558.640 event trong `events_35d.parquet`, từ 42.110.402 event có
nhãn ban đầu.

### 1.6 User-disjoint experiment split

Mỗi user được gán nguyên khối vào một trong ba tập:

```text
train / validation / test = 64% / 16% / 20% user
```

Thuật toán cân bằng đồng thời số user, số enrollment và số dropout. Năm seed cố
định là `1, 11, 111, 1111, 11111`. Trong mỗi seed, một user không thể xuất hiện ở
hai split.

```text
splits.parquet
  seed
  node_id
  enroll_id
  user_id
  source_partition
  experiment_split
```

### 1.7 File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/config.py` | `DatasetContract`, `SOURCE_SCHEMAS` | Khóa tên file, schema, số lượng và protocol thí nghiệm |
| `src/data/preprocess.py` | `validate_source_files()`, `prepare_dataset()` | Kiểm tra đầu vào và tạo nodes/users/courses/events canonical |
| `src/data/split.py` | `assign_user_groups()`, `build_splits()` | Tạo và kiểm tra năm user-disjoint split |
| `src/main.py` | `prepare-data`, `split-data` | Entry point dòng lệnh |

Phase preprocessing không dùng cache hay manifest. Mỗi lệnh luôn tạo lại artifact
của chính nó. `split-data` yêu cầu `nodes.parquet` đã tồn tại và không tự chạy lại
`prepare-data`.

### 1.8 Artifact đầu ra

```text
nodes.parquet
users.parquet
courses.parquet
events_35d.parquet
splits.parquet
```

---

## 2. Feature engineering

### 2.1 Behavioral feature `X_base`

Mỗi enrollment được aggregate trong 35 ngày:

| Khối | Chiều | Cách tính |
|---|---:|---|
| `day_00..day_34` | 35 | Số event từng ngày |
| `action_*` | 23 | Số lần từng action xuất hiện |
| `session_count` | 1 | Số session khác nhau |
| `distinct_observed_objects` | 1 | Số `(course_id, object_id)` khác nhau đã quan sát |
| Tổng | 60 | |

Hai invariant quan trọng:

```text
event_count = Σ day_*
event_count = Σ action_*
```

Enrollment không có event trong cửa sổ được điền tất cả count bằng 0. Count được
biến đổi:

```text
x_log = log(1 + x_raw)
x_scaled = (x_log - μtrain) / σtrain
```

`μtrain` và `σtrain` được fit riêng cho train của từng seed. Validation/test chỉ
dùng lại transformer của seed tương ứng.

### 2.2 User demographic

Metadata được broadcast xuống enrollment qua `nodes.user_id`:

| Feature | Encoding | Chiều |
|---|---|---:|
| Gender | female, male, missing, other | 4 |
| Education | 7 giá trị + missing + other | 9 |
| Age | standardized age + missing indicator | 2 |
| Tổng | | 15 |

Tuổi được tính tại thời điểm course bắt đầu:

```text
age_at_course_start = year(course_start) - birth_year
```

Age ngoài `[10,100]` được chuyển thành missing. Missing age được train-median
imputation; sau đó numeric value được standardize bằng train statistics. Missing
indicator vẫn giữ dạng nhị phân.

### 2.3 Course context

Metadata được broadcast xuống enrollment qua `nodes.course_id`:

| Feature | Encoding | Chiều |
|---|---|---:|
| Category | 17 category + missing + other | 19 |
| Duration | standardized duration + missing indicator | 2 |
| Tổng | | 21 |

```text
course_duration_days = date(course_end) - date(course_start)
```

Duration âm hoặc null được xem là missing. `course_type` bị loại vì toàn bộ 247
course hiện có cùng giá trị 0.

### 2.4 Bốn node-feature configuration

```text
behavior        = X_behavior                         ∈ R^(N×60)
behavior_user   = [X_behavior | X_user]              ∈ R^(N×75)
behavior_course = [X_behavior | X_course]            ∈ R^(N×81)
full            = [X_behavior | X_user | X_course]   ∈ R^(N×96)
```

Trên đĩa chỉ lưu `X_base.npy` và `X_context.npy`. Loader lấy các hàng cần thiết rồi
concatenate, tránh lưu thêm một bản `X_full.npy` lớn.

### 2.5 File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/config.py` | `base_feature_columns()` | Schema behavioral 60 chiều |
| `src/config.py` | `user_feature_columns()` | Schema demographic 15 chiều |
| `src/config.py` | `course_feature_columns()` | Schema course 21 chiều |
| `src/config.py` | `feature_columns()` | Ánh xạ bốn feature set |
| `src/features/engineering.py` | `_aggregate_query()`, `build_raw_features()` | Aggregate event thành feature enrollment |
| `src/features/context.py` | `build_raw_context()` | Join user/course context xuống node |
| `src/features/transform.py` | `_write_x_base()` | `log1p` và train-only standardization |
| `src/features/transform.py` | `_write_x_context()` | One-hot, median imputation và numeric scaling |
| `src/features/io.py` | `NodeFeatureStore` | Memory-map và ghép feature theo cấu hình |

### 2.6 Artifact đầu ra

```text
features_raw.parquet
context_raw.parquet
X_base.npy       [5, 225642, 60], float32
X_context.npy    [5, 225642, 36], float32
feature_transform.joblib
context_transform.joblib
feature_manifest.json
```

---

## 3. Hypergraph construction

### 3.1 Khái niệm

Graph thường nối từng cặp node. Hypergraph cho phép một hyperedge nối đồng thời
nhiều enrollment có chung quan hệ. Pipeline tạo ba family: Course, Object và
Behavioral.

### 3.2 Course hyperedge

Với course `c`:

```text
e_course(c) = {i | course_id(i) = c}
```

Course edge biểu diễn context cấu trúc: các enrollment học cùng course có thể trao
đổi message qua một hyperedge. Train graph chỉ chứa train nodes của seed đang xét.

### 3.3 Object hyperedge

Object dùng composite key:

```text
object_key = (course_id, object_id)
e_object(o) = {i | enrollment i tương tác với object_key o}
```

Không dùng `object_id` độc lập vì cùng một ID có thể xuất hiện trong nhiều course.
Event thiếu object không tạo incidence. Action được ánh xạ sang ba object type:

```text
video / assignment / forum
```

### 3.4 Behavioral hyperedge

Mỗi anchor tạo một tập:

```text
e_behavior(i) = {i} ∪ kNN_train(i)
```

Cosine similarity được tính trên `X_base` 60 chiều đã normalize. FAISS HNSW chỉ
index train nodes; mọi train, validation và test anchor đều chỉ nhận train node làm
neighbor. `k` hỗ trợ `{5,10,20}`, mặc định hiện tại là 10.

Context không tham gia behavioral kNN. Nhờ đó feature ablation chỉ thay đổi node
input, không thay đổi cấu trúc graph.

### 3.5 Chống leakage

```text
Course/Object train edge: chỉ group train nodes
Behavioral reference set: chỉ train nodes
Validation/test target: không nối với target validation/test khác
Label: không được đọc khi tạo hyperedge
```

### 3.6 File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/hypergraph/course.py` | `course_membership_query()` | Sinh candidate Course incidence |
| `src/hypergraph/object.py` | `object_membership_query()` | Sinh candidate Object incidence |
| `src/hypergraph/behavioral.py` | `build_seed_neighbors()` | FAISS cosine kNN với train references |
| `src/hypergraph/construction.py` | `_build_structural()` | Gộp Course và Object candidate |
| `src/hypergraph/construction.py` | `build_hyperedges()` | Điều phối ba family, cache và audit |
| `src/hypergraph/audit.py` | `audit_structural()`, `audit_behavioral()` | Kiểm tra cardinality và train-only contract |

### 3.7 Artifact đầu ra

```text
structural_memberships.parquet
behavioral_neighbors.npz
hyperedge_audit.json
hyperedge_manifest.json
```

---

## 4. Hyperedge: biểu diễn, lọc và sampling

### 4.1 Từ candidate đến hyperedge thật

Một structural group chỉ trở thành hyperedge trong train graph khi có ít nhất hai
train node:

```text
cardinality(e) ≥ 2
```

Behavioral edge gồm anchor và `k` neighbor. Các behavioral edge có tập node giống
hệt nhau được gộp bằng `np.unique`.

Mỗi hyperedge có metadata:

```text
seed, hyperedge_id, family, source_key, object_type,
course_id, object_id, anchor_node_id, size, weight
```

Initial edge weight luôn bằng 1.

### 4.2 Hyperedge sampling trong HSL

Không refine toàn bộ hơn 150 nghìn hyperedge trong mỗi forward. Edge được chia theo:

```text
family × size_bucket
```

Size bucket:

```text
small  : size ≤ 10
medium : 11 ≤ size ≤ 100
large  : size > 100
```

Sampler đi round-robin qua các strata không rỗng để Course/Object/Behavioral và
large/small edge đều có cơ hội được chọn. Mặc định lấy 96 edge mỗi epoch.

### 4.3 Incident-node sampling

Với mỗi sampled edge:

```text
positive = node đang incident với edge
negative = node không incident với edge
```

Mặc định sample tối đa 16 positive và 16 negative. Khi validation/test local graph
được batch block-diagonal, negative chỉ được chọn trong cùng local graph nhờ
`node_groups` và `edge_groups`.

### 4.4 File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/hypergraph/construction.py` | `_structural_groups()` | Group và lọc structural edge theo train cardinality |
| `src/hypergraph/construction.py` | `_materialize_seed()` | Gộp edge trùng, gán hyperedge ID và metadata |
| `src/hsl.py` | `cardinality_buckets()` | Chia size bucket |
| `src/hsl.py` | `sample_balanced_hyperedges()` | Round-robin family/size sampling |
| `src/hsl.py` | `sample_incident_nodes()` | Sample positive/negative membership candidates |

---

## 5. Initial hypergraph `H0` và node features `X`

### 5.1 Incidence matrix

Với node `i` và hyperedge `e`:

```text
H0[i,e] = 1 nếu i thuộc e
H0[i,e] = 0 nếu ngược lại
```

Main configuration:

```text
H0 = [H_course | H_object | H_behavior]
```

`H0` được lưu dạng SciPy CSR `uint8`, không bao giờ materialize dense. Với seed 1:

```text
Ntrain       = 144543
E            = 153290
incidences   = 3369506
Course edge  = 247
Object edge  = 21891
Behavior edge= 131152
```

### 5.2 Ghép `H0` với `X`

`H0` dùng local train-row index; `X_base`/`X_context` dùng global `node_id`.
`train_node_index.parquet` ánh xạ:

```text
(seed, local_node_id) → global node_id
```

Loader lấy đúng hàng feature theo global node ID rồi trả:

```text
TrainHypergraph(node_ids, features, incidence)
```

Invariant bắt buộc:

```text
H0.shape[0] = len(node_ids) = X.shape[0]
X.shape[1] ∈ {60,75,81,96}
```

### 5.3 Inductive local graph cho validation/test

Mỗi target được dựng một graph riêng:

```text
target
+ train references cùng course
+ train references cùng object
+ k behavioral train neighbors
```

Không có target–target message passing. Khi đánh giá nhiều target, các local graph
được ghép block-diagonal nên vẫn độc lập.

### 5.4 HGNN propagation operator

Với edge-weight matrix `W`, node degree `Dv` và edge degree `De`:

```text
G(H) = Dv^(-1/2) H W De^(-1) Hᵀ Dv^(-1/2)
```

Một HGNN layer:

```text
Y = G(H) X Θ + b
```

Code thực hiện lần lượt node → edge → node bằng sparse matrix multiplication, không
tạo dense `G(H)`.

### 5.5 File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/hypergraph/construction.py` | `_materialize_seed()` | Tạo sparse binary `H0` |
| `src/hypergraph/construction.py` | `build_initial_hypergraph()` | Tạo H0 cho năm seed và local-membership metadata |
| `src/graph_data.py` | `load_train_hypergraph()` | Load H0 và X theo cùng row order |
| `src/graph_data.py` | `LocalGraphStore` | Dựng inductive local graph |
| `src/graph_data.py` | `batch_local_hypergraphs()` | Ghép local graph block-diagonal |
| `src/features/io.py` | `NodeFeatureStore` | Chọn và ghép node feature set |
| `src/model.py` | `HypergraphOperator` | Tính sparse normalized propagation |
| `src/model.py` | `HGNNLayer` | Linear projection + hypergraph propagation |

### 5.6 Artifact đầu ra

```text
H0_train_seed_<seed>.npz
train_node_index.parquet
validation_memberships.parquet
test_memberships.parquet
hyperedges.parquet
hypergraph_audit.json
hypergraph_manifest.json
```

---

## 6. Hypergraph Structure Learning

### 6.1 Initial embedding

HGNN backbone có hai layer:

```text
U  = Dropout(ReLU(G(H0) X Θ1 + b1))
Z0 = ReLU(G(H0) U Θ2 + b2)
```

Hidden dimension mặc định là 64, dropout mặc định 0,5.

### 6.2 Hyperedge representation

Với sampled edge `e`, lấy mean embedding của sampled positive nodes:

```text
q_e = mean({z0_i | i thuộc sampled positive của e})
```

Lưu ý: đây là mean của sampled positives, không nhất thiết là mean của toàn bộ node
trong edge.

### 6.3 Membership scorer

Với node embedding `z_i`, edge representation `q_e` và hidden dimension `d`:

```text
s(i,e) = (Wn z_i)ᵀ(We q_e) / sqrt(d) + b
p(i,e) = sigmoid(s(i,e))
```

`Wn`, `We` và `b` là tham số học được.

### 6.4 Chọn membership

Hai mode được hỗ trợ:

```text
top_r     : giữ r candidate có logit cao nhất
threshold : giữ candidate có sigmoid(logit) ≥ threshold
```

Mặc định dùng `top_r=8`. Code ép mỗi sampled edge giữ ít nhất một sampled positive,
tránh edge được tái tạo hoàn toàn từ negative candidates.

### 6.5 Tạo `H*`

Hành vi hiện tại cần hiểu chính xác:

1. Edge không được sample giữ nguyên toàn bộ incidence với value 1.
2. Edge được sample bị bỏ toàn bộ incidence cũ.
3. Chỉ candidate được chọn được đưa trở lại với value `p(i,e)`.
4. Node bị cô lập sau bước trên được khôi phục một original incidence với value 1.
5. `H*` có cùng shape với `H0`; chỉ sparse values/locations thay đổi.

```text
H0 ∈ {0,1}^(N×E)
H* ∈ [0,1]^(N×E), sparse
```

Custom sparse autograd chỉ tính gradient cho những sparse values đang tồn tại, tránh
gradient dense kích thước `N×E`.

### 6.6 Refined embedding

```text
Z* = HGNN(X, H*)
logit = Linear(Z*)
```

Hai lần HGNN hiện dùng chung một backbone và chung trọng số:

```text
Z0 = Fθ(X, H0)
Z* = Fθ(X, H*)
```

Refinement chỉ chạy một lần trong mỗi forward; không có vòng lặp nhiều bước
`Z* → H** → Z**`.

### 6.7 File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/model.py` | `HGNNBaseline.encode()` | Tạo embedding bằng HGNN hai layer |
| `src/hsl.py` | `HGSLModel.forward()` | Điều phối `H0 → Z0 → H* → Z* → logits` |
| `src/hsl.py` | `MembershipScorer` | Bilinear node-edge score |
| `src/hsl.py` | `HypergraphRefiner` | Sampling, selection và dựng sparse `H*` |
| `src/model.py` | `_SparseValuesMM` | O(nnz) gradient cho learnable sparse values |
| `src/model.py` | `HGNNBaseline.classify()` | Linear classifier từ `Z*` sang một logit |

---

## 7. Contrastive loss

### 7.1 Mục tiêu

`Z0` và `Z*` là hai biểu diễn của cùng node dưới hai cấu trúc `H0` và `H*`.
Contrastive loss giữ embedding của cùng node gần nhau, đồng thời phân biệt node khác.

### 7.2 Chuẩn hóa và similarity

Với một sample gồm `M` node:

```text
v_i  = normalize(z0_i)
v*_j = normalize(z*_j)
S_ij = v_iᵀv*_j / τ
```

Đường chéo `S_ii` là positive pair. Các phần tử khác cùng batch là negatives.

### 7.3 Symmetric InfoNCE

```text
L0→* = CrossEntropy(S, target=[0,1,...,M-1])
L*→0 = CrossEntropy(Sᵀ, target=[0,1,...,M-1])

LCL = 0.5 × (L0→* + L*→0)
```

Mặc định:

```text
temperature τ = 0.2
contrastive nodes M = min(512, Ntrain)
```

Chỉ tạo similarity matrix `M×M`, không tạo `N×N` cho toàn graph.

### 7.4 File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/hsl.py` | `contrastive_alignment_loss()` | Symmetric cross-view InfoNCE |
| `src/train.py` | `run_hgsl_training()` | Sample node indices cho contrastive loss |

---

## 8. BCE loss

### 8.1 Classifier output

Classifier trả một raw logit cho mỗi node:

```text
logit_i = wᵀz*_i + b
probability_i = sigmoid(logit_i)
```

Training truyền raw logits trực tiếp vào `binary_cross_entropy_with_logits` để có
độ ổn định số tốt hơn sigmoid rồi log thủ công.

### 8.2 Train-only class weight

```text
pos_weight = Ntrain(label=0) / Ntrain(label=1)
```

Vì `truth=1` là dropout và dropout là lớp đa số, seed 1 hiện có
`pos_weight ≈ 0.3185`. Giá trị nhỏ hơn 1 làm giảm đóng góp tương đối của lớp positive
đang chiếm đa số.

Với một node, dạng khái niệm của weighted BCE là:

```text
ℓ_i = -[w+ y_i log σ(l_i) + (1-y_i) log(1-σ(l_i))]
```

`pos_weight` chỉ được tính từ train labels; validation/test label không tham gia.

### 8.3 File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/model.py` | `HGNNBaseline.classify()` | Sinh dropout logit |
| `src/hsl.py` | `train_pos_weight()` | Tính train-only class weight |
| `src/hsl.py` | `classification_loss()` | Weighted BCE-with-logits |

---

## 9. Total loss

Objective cuối cùng:

```text
Ltotal = LBCE + λLCL
```

Mặc định kỹ thuật hiện tại:

```text
λ = 0.1
τ = 0.2
```

Ý nghĩa:

- `LBCE` tối ưu trực tiếp mục tiêu dự đoán dropout.
- `LCL` regularize để hai view `H0` và `H*` không làm embedding cùng node lệch quá
  xa.
- `λ` điều khiển mức đánh đổi giữa classification và cross-view alignment.

Nếu `λ=0`, code bỏ qua contrastive computation và `Ltotal=LBCE`.

### File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/hsl.py` | `hgsl_objective()` | Tính BCE, InfoNCE và tổng loss |
| `src/train.py` | `run_hgsl_training()` | Truyền λ, τ và contrastive node indices |

---

## 10. Quá trình train

### 10.1 Khởi tạo một run

Với một seed và `feature_set`:

1. Khóa Python/NumPy/PyTorch random seed.
2. Load train `H0`, node IDs và feature matrix `X`.
3. Load family/size metadata của hyperedge.
4. Load train labels theo đúng node order.
5. Tính `pos_weight` chỉ từ train labels.
6. Tạo sparse `HypergraphOperator(H0)`.
7. Khởi tạo `HGSLModel` và Adam optimizer.

### 10.2 Một training epoch

```text
generator = RNG(seed, epoch)
sample contrastive node indices

optimizer.zero_grad()

Z0       = HGNN(X, H0)
H*       = Refiner(Z0, H0)
Z*       = HGNN(X, H*)
logits   = Classifier(Z*)

LBCE     = WeightedBCE(logits, labels)
LCL      = SymmetricInfoNCE(Z0, Z*)
Ltotal   = LBCE + λLCL

Ltotal.backward()
clip_grad_norm(max_norm=5.0)
optimizer.step()
```

Code còn kiểm tra riêng gradient norm của membership scorer. Run bị dừng bằng lỗi
nếu scorer không nhận gradient hữu hạn và dương.

### 10.3 Validation

Sau mỗi epoch:

1. Dựng local graph cho từng validation target.
2. Có thể batch nhiều graph bằng block-diagonal incidence.
3. Chạy cùng HGSL forward nhưng không tính gradient.
4. Chỉ lấy prediction tại target index của mỗi local graph.
5. Tính AUC, AUPRC, F1, precision và recall.

Checkpoint tốt nhất được chọn bằng validation AUC. Early stopping dừng khi AUC
không cải thiện trong `patience` epoch.

### 10.4 Checkpoint và khả năng tái lập

Checkpoint lưu:

```text
model state
optimizer state
epoch
training config + feature_set
validation metrics
refinement audit
PyTorch RNG state
graph manifest signature
feature manifest signature
ordered feature names
```

Tên file chứa feature configuration và seed, ví dụ:

```text
hgsl_full_seed_1.pt
hgsl_behavior_user_seed_11.pt
```

### 10.5 Train graph và evaluation graph khác nhau

```text
Train      : full-batch trên toàn bộ train H0
Validation : inductive local graph, target + train references
Test       : phải dùng cùng inductive local-graph protocol
```

`run_hgsl_training()` hiện chỉ train và chọn checkpoint bằng validation; report ghi
`test_split_used=false`. Final test evaluator là bước riêng cần hoàn thiện trước khi
chạy bảng kết quả nghiên cứu chính thức.

### 10.6 Cấu hình mặc định đáng chú ý

| Tham số | Giá trị hiện tại |
|---|---:|
| Hidden dimension | 64 |
| Dropout | 0,5 |
| Learning rate | 0,001 |
| Weight decay | 0,0005 |
| Gradient max norm | 5,0 |
| Sampled hyperedges | 96 |
| Positive/negative candidates | 16 / 16 |
| Membership selection | `top_r=8` |
| Contrastive weight `λ` | 0,1 |
| Temperature `τ` | 0,2 |
| Contrastive nodes | 512 |

Đây là cấu hình kỹ thuật, chưa phải hyperparameter tối ưu cuối cùng.

### 10.7 File code

| File | Hàm/lớp | Trách nhiệm |
|---|---|---|
| `src/train.py` | `HGSLTrainingConfig` | Khóa training/refinement configuration |
| `src/train.py` | `run_hgsl_training()` | Training loop, validation, early stopping, report |
| `src/train.py` | `_write_hgsl_checkpoint()` | Atomic checkpoint write |
| `src/train.py` | `set_seed()`, `resolve_device()` | Reproducibility và device selection |
| `src/train.py` | `select_evaluation_targets()` | Lấy validation/test target, hỗ trợ smoke limit |
| `src/metrics.py` | `binary_metrics()` | AUC, AUPRC, F1, precision, recall |
| `src/train.py` | `run_hgsl_smoke()` | One-step end-to-end gradient smoke test |
| `src/main.py` | `train-hgsl`, `check-hgsl`, `evaluate` | Entry point dòng lệnh |

### 10.8 Lệnh chạy

Build toàn bộ data/graph artifact:

```powershell
.\.venv\Scripts\python.exe run.py prepare-data
.\.venv\Scripts\python.exe run.py split-data
.\.venv\Scripts\python.exe run.py build-features
.\.venv\Scripts\python.exe run.py build-hyperedges
.\.venv\Scripts\python.exe run.py build-hypergraph --behavioral-k 10
```

Smoke test cấu hình đầy đủ 96 chiều:

```powershell
.\.venv\Scripts\python.exe run.py check-hgsl --seed 1 --feature-set full
```

Train với toàn bộ validation targets:

```powershell
.\.venv\Scripts\python.exe run.py train-hgsl `
  --seed 1 `
  --feature-set full `
  --epochs 100 `
  --patience 10 `
  --validation-limit 0
```

---

## 11. Bảng tra nhanh code theo 10 khối

| Khối | File chính |
|---|---|
| 1. Preprocessing | `src/config.py`, `preprocess.py`, `split.py` |
| 2. Feature engineering | `src/features/engineering.py`, `context.py`, `transform.py`, `io.py` |
| 3. Hypergraph construction | `src/hypergraph/course.py`, `object.py`, `behavioral.py`, `construction.py` |
| 4. Hyperedge | `src/hsl.py` |
| 5. Initial `H0` và `X` | `src/hypergraph/construction.py`, `src/graph_data.py`, `src/model.py` |
| 6. Hypergraph structure learning | `src/model.py`, `src/hsl.py` |
| 7. Contrastive loss | `src/hsl.py` |
| 8. BCE loss | `src/hsl.py`, `src/model.py` |
| 9. Total loss | `src/hsl.py` |
| 10. Training | `src/train.py`, `src/metrics.py` |

## 12. Các điểm phải ghi rõ khi mô tả trong luận án

1. Node là enrollment; user/course metadata được broadcast thành node attributes.
2. User-disjoint split khác với raw `source_partition`.
3. Behavioral kNN luôn dùng `X_base` 60 chiều, kể cả khi HGNN dùng `X_full`.
4. `H0` là binary sparse incidence; `H*` là weighted sparse incidence.
5. Hai lần HGNN trước/sau refinement dùng chung trọng số.
6. Sampled edge hiện bị thay toàn bộ membership, không chỉ update sampled entries.
7. Refinement chạy một lượt trong mỗi forward.
8. Contrastive positive pair là cùng node ở `Z0` và `Z*`.
9. `pos_weight` được tính trên train và nhỏ hơn 1 vì dropout là lớp đa số.
10. Validation chọn checkpoint; test không được dùng cho training hoặc tuning.
