# Tài liệu project: từ XuetangX đến dự đoán dropout

![Sơ đồ mô hình HGSL](assets/hypergraph-neural-network-v3.png)

Mục tiêu của code là phục vụ nghiên cứu và học thuật toán. Mỗi bước được giữ đủ
trực tiếp để có thể lần theo dữ liệu từ raw archive đến prediction, đồng thời giữ
các thành phần ảnh hưởng tới tính đúng của thí nghiệm.

## 1. Flow tổng thể

```text
Raw XuetangX
    ↓ download.py
prediction_data.tar.gz + user_info.csv + course_info.csv
    ↓ preprocess.py
nodes.csv + users.csv + courses.csv + events_35d.csv.gz
    ↓ features.py
feature_base.npz + node_objects.csv.gz + X_seed_*.npy
    ↓ hypergraph.py
Course/Object/Behavioral hyperedges + H0_seed_*.npz
    ↓ model.py
H0 → Z0
    ↓ hsl.py
H0 + Z0 → H*
    ↓ model.py
H* → Z* → classifier → logits
    ↓ losses.py
weighted BCE + λ × contrastive loss
    ↓ train.py
backpropagation → validation checkpoint → test
```

## 2. Các khái niệm chính

| Khái niệm | Ý nghĩa |
|---|---|
| Node | Một enrollment của một user trong một course |
| `node_id` | Số hàng của node trong `nodes.csv` và `X_seed_*.npy` |
| Hyperedge | Một nhóm node có quan hệ chung |
| `H0` | Sparse incidence matrix ban đầu, hàng là train node và cột là hyperedge |
| `H*` | Incidence matrix có membership được HSL học lại |
| `Z0` | Node embedding sinh từ `X` và `H0` |
| `Z*` | Node embedding sinh từ `X` và `H*` |

## 3. Download

`src/download.py` tải đúng ba file vào `data/raw/xuetangx/`:

```text
prediction_data.tar.gz
user_info.csv
course_info.csv
```

File đã tồn tại được bỏ qua. Script không checksum, không dùng `.part`, không
giải nén và không chuyển JSON. Nếu một lần tải bị gián đoạn, người chạy xóa file
đó rồi tải lại.

## 4. Preprocessing

`src/preprocess.py` đọc archive theo hai lượt tuần tự:

1. Đọc hai truth table để có label.
2. Stream `train_log.csv` và `test_log.csv` từng dòng để tạo event sạch.

Hai lượt đọc giữ mức sử dụng RAM nhỏ và tránh giải nén gần 6 GB log ra đĩa.
Event ngoài khoảng ngày 0–34 tính từ ngày bắt đầu course bị loại. Event được ghi
trực tiếp vào `events_35d.csv.gz`.

Preprocessing chỉ giữ các kiểm tra có thể ngăn chương trình âm thầm sinh dữ liệu
sai: raw/member bị thiếu, label/action sai, course không tồn tại, metadata của một
enrollment mâu thuẫn hoặc user metadata bị thiếu. Không kiểm exact count cố định.

Nếu cả bốn output đã tồn tại, preprocessing dừng ngay mà không đọc raw:

- `nodes.csv`
- `users.csv`
- `courses.csv`
- `events_35d.csv.gz`

## 5. Split thí nghiệm

Split 64/16/20 được thực hiện theo user. Tất cả enrollment của cùng user luôn nằm
trong cùng train, validation hoặc test split. Cùng seed và cùng `nodes.csv` sẽ cho
cùng split.

Split không được lưu thành một lớp cache riêng. Các bước phụ thuộc seed gọi cùng
hàm `split_nodes()`. `train_ids_seed_*.npy` vẫn được lưu vì nó là ánh xạ hàng của
`H0`, không phải cache state.

## 6. Feature engineering

### Behavioral features

Mỗi node có 60 behavioral feature:

- 35 số đếm activity theo `course_day`.
- 23 số đếm theo action.
- 1 số session khác nhau.
- 1 số object khác nhau đã quan sát.

Hai log có hơn 40 triệu event. Để đếm distinct session/object chính xác,
`features.py` phân event theo `node_id % buckets`. Mỗi bucket được tổng hợp riêng,
sau đó thư mục tạm được xóa tự động.

### Context features

User context có 15 chiều: gender, education, age và missing indicators. Course
context có 21 chiều: category, duration và missing indicators. Tổng cộng `X` có
96 chiều.

Các count được biến đổi bằng `log1p`. Mean/std của behavioral feature, median cho
giá trị thiếu và mean/std của feature số chỉ được fit trên train node. Sau đó
cùng transform được áp dụng cho validation và test.

### Output

- `feature_base.npz`: count và context chưa phụ thuộc seed.
- `node_objects.csv.gz`: object membership cần để dựng hypergraph.
- `X_seed_*.npy`: feature đã chuẩn hóa cho một seed.
- `feature_stats_seed_*.json`: tên feature và tham số transform.

Nếu `feature_base.npz` và `node_objects.csv.gz` cùng tồn tại, seed mới không cần
đọc lại toàn bộ event.

## 7. Hypergraph construction

`src/hypergraph.py` tạo ba family:

| Family | Membership |
|---|---|
| Course | Train node cùng course |
| Object | Train node cùng course, object và object type |
| Behavioral | Anchor cộng `k` train neighbor gần nhất |

Behavioral feature được chuẩn hóa L2; FAISS HNSW tìm neighbor bằng inner product,
tương đương cosine similarity sau chuẩn hóa. Index chỉ chứa train node, nên
neighbor của validation/test không làm rò rỉ node ngoài train.

Các artifact gồm:

- `neighbors_seed_*.npy`
- `edge_memberships_seed_*.csv.gz`
- `edge_meta_seed_*.csv`
- `H0_seed_*.npz`
- `train_ids_seed_*.npy`
- `graph_config_seed_*.json`

`graph_config` chỉ ghi `k`, `k_max` và số lượng graph; nó không chứa fingerprint
và không được dùng để tự động quyết định rebuild.

## 8. HGNN và HSL

### HGNN propagation

`src/model.py` thực hiện phép truyền node → hyperedge → node bằng sparse
`torch.sparse.mm()`. Hai lượt HGNN trước và sau HSL dùng chung trọng số:

```text
X + H0 → shared HGNN → Z0
X + H* → shared HGNN → Z*
```

### HSL

`src/hsl.py` thực hiện tuần tự:

```text
chọn hyperedge
→ lấy positive node
→ lấy negative node
→ tạo edge embedding từ positive node
→ score candidate membership
→ chọn top-r hoặc threshold
→ ghép H*
```

Train lấy mẫu hyperedge theo family và kích thước. Validation/test xử lý toàn bộ
edge trong local graph theo thứ tự xác định. Nếu refinement làm một node mất toàn
bộ membership, code khôi phục một membership ban đầu để HGNN không có node cô
lập.

`H*` là sparse tensor có value phụ thuộc membership scorer. PyTorch autograd qua
`torch.sparse.mm()` truyền gradient về `node_projection`, `edge_projection` và
`membership_bias`; test suite kiểm tra trực tiếp bất biến này.

## 9. Loss

`src/losses.py` tính:

```text
Total Loss = Weighted BCE + λ × Symmetric InfoNCE(Z0, Z*)
```

Positive class weight chỉ được tính từ label của train split. Với HGNN ablation
`--no-hsl`, refinement bị bỏ và contrastive weight bằng 0.

## 10. Train, validation và test

`src/train.py` chạy flow:

```text
set seed
→ load X, H0, labels
→ create model
→ mỗi epoch: forward → loss → backward → optimizer step → validation
→ lưu checkpoint khi validation AUC tốt hơn
→ early stopping
→ test checkpoint đã chọn
```

Validation/test dùng local graph gồm một target và các train reference. Target
validation/test luôn ở vị trí đầu, các target khác không xuất hiện trong cùng
local graph trừ khi batching ghép các graph thành block diagonal. `node_groups`
và `edge_groups` ngăn negative sampling đi qua block khác.

Checkpoint có tên dễ đọc, ví dụ `simple_hgsl_full_seed_1.pt`, và chứa trọng số
cùng cấu hình cần để load model. Chạy lại cùng experiment sẽ ghi đè checkpoint
cũ. Code không hash dữ liệu và không tạo run ID.

## 11. Chạy nhiều seed

Mỗi seed cần feature và graph riêng:

```powershell
$seeds = 1, 11, 111, 1111, 11111
foreach ($seed in $seeds) {
    .\.venv\Scripts\python.exe src/features.py --seed $seed
    .\.venv\Scripts\python.exe src/hypergraph.py --seed $seed --k 10 --k-max 20
}
.\.venv\Scripts\python.exe src/train.py --seeds 1 11 111 1111 11111 --feature-set full
```

Hyperparameter được chọn bằng validation. Chỉ sau khi chốt cấu hình mới chạy test
cho checkpoint của các seed dùng để báo cáo kết quả.

## 12. Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test suite dùng archive nhỏ tự tạo để kiểm tra download skip, preprocessing trực
tiếp từ tar, train-only split/normalization, ba hyperedge family, local graph
không leakage, sparse gradient, checkpoint selection và test evaluation.
