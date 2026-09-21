# Đặc tả kỹ thuật mô hình HGSL trên XuetangX-247

> Mục đích: dùng sơ đồ mô hình làm chuẩn để đối chiếu với code tại commit
> `12d0fe2`. Tài liệu mô tả trạng thái đã triển khai đến hết Phase 8; các giá trị
> hyperparameter hiện tại là cấu hình kỹ thuật, chưa phải kết quả tối ưu.

![Sơ đồ mô hình HGSL](assets/hypergraph-neural-network-v3.png)

## 1. Luồng xử lý tổng thể

```text
XuetangX-247
→ tạo node enrollment và X ∈ R^(N×D), D ∈ {60, 75, 81, 96}
→ tạo Course/Object/Behavioral hyperedge
→ sparse initial incidence H0
→ HGNN(H0, X) sinh Z0
→ sample hyperedge và candidate node
→ học membership score, tạo sparse weighted H*
→ HGNN(H*, X) sinh Z*
→ linear classifier sinh dropout logit
→ Ltotal = weighted BCE + λ × symmetric InfoNCE(Z0, Z*)
```

Nhãn được khóa là `truth=1: dropout`, `truth=0: non-dropout`.

## 2. Đối chiếu từng khối trong hình

| Khối trong hình | Đặc tả đang triển khai | Trạng thái |
|---|---|---|
| Dataset XUETANGX | Một dataset XuetangX-247 có nhãn: 225.642 enrollment, 77.083 user, 247 course | Khớp |
| Features Engineering | 60 behavioral + 15 user demographic + 21 course context; cấu hình đầy đủ 96 chiều | Khớp |
| Nodes | Một node là một enrollment; khóa nội bộ `node_id`, khóa nguồn `enroll_id` | Khớp |
| Course hyperedge | Nối các train enrollment cùng `course_id`; loại edge có dưới 2 node | Khớp |
| Object hyperedge | Nối theo khóa `(course_id, object_id)` cho video, assignment, forum; loại edge dưới 2 node | Khớp |
| Behavioral hyperedge | Anchor + k train-neighbor gần nhất theo cosine trên `X`; hiện `k=10` | Khớp, k chưa tune |
| User hyperedge | Không được tạo và không nằm trong `H0` | **Không khớp hình** |
| Initial hypergraph `H0` | Sparse binary CSR, kích thước `N×E`, edge weight ban đầu bằng 1 | Khớp |
| Node features `X` | Chọn `behavior`/`behavior_user`/`behavior_course`/`full`; tương ứng 60/75/81/96 chiều | Khớp |
| HGNN → `Z0` | HGNN 2 lớp, hidden 64, ReLU và dropout 0,5 | Khớp |
| Hyperedge Sampling | Round-robin theo `(family, size bucket)`; bucket `≤10`, `11–100`, `>100` | Khớp |
| Incident Node Sampling | Mỗi sampled edge lấy tối đa 16 positive và 16 non-incident negative | Khớp |
| Refine Hypergraph `H*` | Bilinear membership score; chọn `top-r` hoặc threshold; sinh sparse weighted incidence | Khớp một phần |
| HGNN → `Z*` | Chạy HGNN lần hai trên `H*`; hiện dùng chung tham số với HGNN tạo `Z0` | Hình chưa nói rõ shared weight |
| Contrastive Loss | Symmetric InfoNCE giữa cùng node ở `Z0` và `Z*`, mặc định sample 512 node | Khớp |
| Classifier | Một linear layer từ hidden embedding tới một dropout logit | Khớp |
| BCE Loss | Weighted BCE; `pos_weight = số label 0 / số label 1`, chỉ tính trên train | Khớp |
| Total Loss | `LBCE + λLCL`, hiện `λ=0,1` | Khớp, λ chưa tune |

**Quyết định hiện tại:** User hyperedge được loại khỏi cấu hình chính vì một user có
thể có nhiều enrollment và dễ tạo đường truyền thông tin không phù hợp với user-disjoint
split. Nếu giữ quyết định này thì cần xóa User hyperedge khỏi hình chính hoặc ghi rõ
“optional/ablation only”.

## 3. Dữ liệu, split và feature

### 3.1 Dataset contract

| Thành phần | Giá trị |
|---|---:|
| Enrollment/node | 225.642 |
| User | 77.083 |
| Course | 247 |
| Event gốc có nhãn | 42.110.402 |
| Event giữ trong ngày 0–34 | 40.558.640 |
| Observation window chính | 35 ngày |
| Seed | `1, 11, 111, 1111, 11111` |

Split được thực hiện theo **user-disjoint**, mục tiêu 64/16/20. Mỗi seed hiện có
144.543 train node, 36.028 validation node và 45.071 test node; không có user xuất
hiện ở hai split.

### 3.2 Behavioral features `X_base`

```text
35 day counts
+ 23 action counts
+ 1 session_count
+ 1 distinct_observed_objects
= 60 features
```

- Enrollment không có event được điền count bằng 0; không còn NA trong `X_base`.
- Không có categorical one-hot trong cấu hình 60 chiều hiện tại.
- Biến đếm được biến đổi `log1p`.
- Mean và standard deviation chỉ fit trên train của từng seed, sau đó dùng để transform
  train/validation/test của cùng seed.
- `X_base.npy` có layout `[seed_index, global_node_id, feature_index]` và shape
  `[5, 225642, 60]`.

### 3.3 User demographic và course context

Node dự đoán vẫn là enrollment. Metadata được gắn xuống từng node bằng:

```text
nodes.user_id   → users.(gender, education, birth_year)
nodes.course_id → courses.(category, course_start, course_end)
```

Khối user có 15 chiều: gender one-hot 4 chiều; education one-hot 9 chiều;
`age_at_course_start` và `age_missing`. Khối course có 21 chiều: category one-hot
19 chiều; `course_duration_days` và `course_duration_missing`. `course_type` bị loại
vì cả 247 course đều bằng 0.

```text
age_at_course_start = year(course_start) - birth_year
course_duration_days = date(course_end) - date(course_start)
```

Age ngoài `[10,100]` và duration âm được chuyển thành missing. Numeric context dùng
train-median imputation rồi standardize chỉ theo train của từng seed. Categorical
dùng vocabulary cố định, có cột `missing` và `other`, không standardize.

`X_context.npy` có layout `[seed_index, global_node_id, context_feature_index]` và
shape `[5, 225642, 36]`. `X_full` không được lưu trùng trên đĩa mà được ghép khi load:

| `feature_set` | Thành phần | Số chiều |
|---|---|---:|
| `behavior` | `X_base` | 60 |
| `behavior_user` | `X_base + X_user` | 75 |
| `behavior_course` | `X_base + X_course` | 81 |
| `full` | `X_base + X_user + X_course` | 96 |

CLI giữ mặc định `behavior` để tái lập kết quả cũ; thí nghiệm dùng context phải ghi
rõ `--feature-set full` hoặc cấu hình ablation tương ứng.

## 4. Initial hypergraph `H0`

Với seed 1:

| Thuộc tính | Giá trị |
|---|---:|
| Train nodes `N` | 144.543 |
| Hyperedges `E` | 153.290 |
| Course edges | 247 |
| Object edges | 21.891 |
| Behavioral edges | 131.152 |
| Incidences | 3.369.506 |
| Min/max edge size | 2 / 2.761 |

Quy tắc tạo edge:

1. Course edge chứa toàn bộ train enrollment của cùng course.
2. Object edge dùng composite key `(course_id, object_id)`, không dùng `object_id`
   riêng vì ID có thể chỉ duy nhất trong phạm vi course.
3. Behavioral edge của train gồm anchor và `k=10` train neighbor; cosine search luôn
   dùng normalized `X_base` 60 chiều và FAISS HNSW. Context chỉ thay đổi HGNN node
   input, không làm thay đổi cấu trúc graph. Edge trùng nhau được gộp.
4. `H0` chỉ chứa edge có ít nhất hai node và mọi incidence ban đầu bằng 1.

Validation/test dùng **local inductive graph** cho từng target:

- target + các train node cùng Course/Object edge;
- target + `k` behavioral train-neighbor;
- target validation/test không được đưa vào train graph;
- khi batch nhiều local graph, incidence được ghép block-diagonal và negative candidate
  chỉ được lấy trong cùng local graph.

## 5. HGNN encoder

Với incidence `H`, edge weight `W`, node degree `Dv` và edge degree `De`, toán tử
truyền tin là:

```text
G(H) = Dv^(-1/2) H W De^(-1) Hᵀ Dv^(-1/2)
```

Encoder hai lớp:

```text
U  = Dropout(ReLU(G(H) X Θ1 + b1))
Z  = ReLU(G(H) U Θ2 + b2)
```

Code thực hiện sparse propagation, không tạo ma trận `G(H)` dense. Với `H*`, gradient
chỉ được tính trên các sparse values đang tồn tại để tránh gradient dense `N×E`.

Hiện hai lần tính embedding dùng **chung một backbone**:

```text
Z0 = Fθ(X, H0)
Z* = Fθ(X, H*)
```

Đây là weight sharing, cần được ghi rõ trong hình hoặc đổi code nếu ý tưởng nghiên cứu
là hai encoder độc lập.

## 6. Hypergraph Structure Learning

### 6.1 Sampling

- Mặc định sample 96 hyperedge mỗi epoch.
- Sampling cân bằng round-robin theo family và size bucket để Behavioral/large edge
  không chiếm toàn bộ sample.
- Với mỗi edge `e`, lấy tối đa 16 incident node làm positive và 16 non-incident node
  làm negative.
- Hyperedge representation được tính **từ `Z0` của sampled positive nodes**:

```text
qe = mean({z0_i | i thuộc sampled positive của e})
```

### 6.2 Membership score

Với node embedding `z_i`, edge embedding `q_e`, hidden dimension `d`:

```text
s(i,e) = (Wn z_i)ᵀ(We q_e) / sqrt(d) + b
p(i,e) = sigmoid(s(i,e))
```

Mặc định chọn `top_r=8`; lựa chọn threshold cũng đã được hỗ trợ. Code luôn ép sampled
edge giữ ít nhất một sampled positive node.

### 6.3 Cách tạo `H*` hiện tại

- Edge không được sample: giữ nguyên toàn bộ incidence, weight bằng 1.
- Edge được sample: **bỏ toàn bộ incidence cũ**, sau đó chỉ đưa các positive/negative
  candidate được chọn trở lại với weight `p(i,e)`.
- Nếu một node bị cô lập sau refinement, khôi phục một incidence gốc với weight 1.
- Không sinh hyperedge family mới; `H*` có cùng shape `N×E` với `H0`.
- Refinement hiện chạy **một lần trong mỗi forward**, không lặp `Z* → sampling → H*`
  nhiều vòng.

Hai ý “thay toàn bộ membership của sampled edge” và “refine một lượt” cần được xác nhận
vì các đường nét đứt trong hình có thể được hiểu là một vòng refinement lặp.

## 7. Prediction và objective

Classifier:

```text
logit_i = wᵀz*_i + b
probability_i = sigmoid(logit_i)
```

Weighted BCE dùng `truth=1` là positive class:

```text
w+ = N(label=0) / N(label=1)
LBCE = BCEWithLogits(logit, label, pos_weight=w+)
```

Seed 1 hiện có `w+=0,3185` vì dropout là lớp đa số. Đây là giá trị phù hợp với định
nghĩa nhãn hiện tại, nhưng cần giữ nguyên định nghĩa này khi báo cáo precision/recall/F1.

Contrastive objective dùng cùng node làm positive pair giữa hai view; các node khác
trong sample làm negative. Loss được tính đối xứng `Z0→Z*` và `Z*→Z0`:

```text
Ltotal = LBCE + λLCL
```

Mặc định kỹ thuật: `λ=0,1`, temperature `τ=0,2`, 512 contrastive nodes. Các giá trị này
phải được chọn bằng validation ở Phase 9.

## 8. Protocol huấn luyện và đánh giá

| Thành phần | Cấu hình hiện tại |
|---|---|
| Optimizer | Adam |
| Learning rate | 0,001 |
| Weight decay | 0,0005 |
| Hidden dimension | 64 |
| Dropout | 0,5 |
| Gradient clipping | max norm 5,0 |
| Early stopping | validation AUC, patience 5 |
| Checkpoint | model, optimizer, config, validation metrics, graph-manifest hash |
| Metrics | AUC, AUPRC, F1, precision, recall |
| F1 threshold | 0,5 |

Train chạy full-batch trên train graph. Validation chạy từng local graph và có thể ghép
block-diagonal để tiết kiệm thời gian. Test không được dùng trong training, tuning hay
chọn checkpoint; `evaluate_hgsl_checkpoint()` load checkpoint tốt nhất rồi mới
đánh giá test.

Smoke run Phase 8 chỉ chạy seed 1, một epoch và 32 validation target. Các metric của
smoke run chỉ chứng minh pipeline hoạt động, **không phải kết quả nghiên cứu**.

## 9. Điểm cần chốt trước Phase 9

### Mức bắt buộc

- [ ] **User hyperedge:** xóa khỏi hình chính, hay bổ sung lại code với cơ chế chống leakage?
- [ ] **Refinement:** một lượt như code, hay lặp theo hai feedback arrow nét đứt trong hình?
- [ ] **Hai HGNN:** dùng chung trọng số như code, hay dùng hai encoder độc lập?
- [ ] **Membership replacement:** sampled edge có thật sự phải mất mọi incidence không
  được sample, hay chỉ cập nhật candidate và giữ phần incidence còn lại?
- [ ] **Validation command:** thí nghiệm thật phải dùng `--validation-limit 0`, không dùng
  mặc định smoke 32 target.

### Cần theo dõi bằng thực nghiệm

- [ ] Tune `behavioral_k ∈ {5,10,20}` trên validation.
- [ ] Tune `λ`, `τ`, learning rate, dropout, sampling budget, `top_r`/threshold.
- [ ] Smoke run có scorer gradient norm khoảng `9,83×10⁻⁷`; kiểm tra gradient có đủ mạnh
  qua nhiều epoch hay không.
- [ ] Smoke validation khôi phục 28.262 isolated-node incidence trên 32 target; kiểm tra
  đây là hành vi mong muốn hay hệ quả của việc thay toàn bộ sampled-edge membership.
- [ ] Chạy đủ năm seed và báo cáo mean ± standard deviation.
- [ ] Thực hiện ablation HGNN không HSL, từng hyperedge family và observation window
  7/14/21/28/35 ngày.

## 10. File dùng để kiểm chứng

| Nội dung | File |
|---|---|
| Dataset contract và feature schema | `src/config.py` |
| User-disjoint split | `src/data/split.py` |
| Feature engineering/transform | `src/features/engineering.py`, `src/features/transform.py` |
| Course/Object/Behavioral construction | `src/hypergraph/` |
| `H0` và local graph | `src/hypergraph/construction.py`, `src/graph_data.py` |
| Sparse HGNN | `src/model.py` |
| Sampling và refinement | `src/hsl.py` |
| End-to-end model | `src/model.py`, `src/hsl.py` |
| BCE + InfoNCE | `src/hsl.py` |
| Training protocol | `src/train.py` |
| Phase 8 tests | `tests/test_phase8.py` |
