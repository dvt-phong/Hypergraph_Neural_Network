# Chi tiết 3 file: 4_hypergraph.py, 5_graph_data.py, 6_hgnn.py (để em kiểm tra, CHƯA thực hiện)

Ngày lập: 2026-10-04. File này bổ sung cho [PLAN_RUT_GON_v5.md](PLAN_RUT_GON_v5.md). Nếu em duyệt, nó thay cho mục 4 (các dòng của file 4, 5, 6) và mục 7.3–7.4 của bản 5.

Mọi con số dưới đây đo trên dữ liệu local hiện tại (train 126,354 / val 31,589 / test 67,699). Thầy chỉ đọc dữ liệu, không sửa gì.

---

## 0. Trả lời nhanh 3 câu hỏi của em

| Câu hỏi | Trả lời |
|---|---|
| Có đảm bảo đủ 3 hyperedge chính (Course, Object, User) + self-loop không? | **Self-loop và Course: luôn có (100%)**. **Object: 88%** số node; node không dùng video/bài tập/forum nào thì không có. **User: 53%** số node; chỉ có khi người học có khóa trước đó trong train. Hai loại sau không thể "đảm bảo", vì phụ thuộc dữ liệu. Code xử lý đúng cả khi thiếu (mục 1.4). |
| Bảng Object 1 thành viên để làm gì? | Để target dùng được những Object mà trong train chỉ có đúng 1 người dùng. Khi đo thì thấy ảnh hưởng **gần như bằng 0**: 628 / 834,166 membership của test (0.08%), chỉ 2 target test có thêm Object. **Thầy đề xuất bỏ bảng này** cho đơn giản (mục 1.5). |
| 6_hgnn đã đúng công thức và đúng 2 layer chưa? | **Đúng.** Code hiện tại (`src/5_model.py`) khớp từng nhân tử của công thức HGNN (Eq. 10). Có đúng 2 lần lan truyền, mỗi lần theo sau là ReLU, giống `HGNN_embedding` của code gốc. File mới giữ nguyên các dòng này. Phần mới duy nhất là cách tính cho target val/test (Bước A/B), có 2 phép thử kiểm chứng (mục 3.5). |

---

## 1. File `4_hypergraph.py`: dựng H0 (chạy 1 lần)

### 1.1 Việc của file

- Đọc **chỉ** `train.csv`. Không đọc validation hay test, không đọc nhãn.
- Dựng hypergraph H0 trên các node train.
- Ghi kết quả ra `hypergraph.npz`.
- Không dùng GPU, không dùng torch, không có cờ dòng lệnh: `python src/4_hypergraph.py`.

### 1.2 Ba loại hyperedge được dựng (self-loop KHÔNG dựng ở đây)

Ký hiệu: v là một enrollment (node); s(v) là **ngày** bắt đầu khóa học của v.

| Loại | Định nghĩa | Ví dụ |
|---|---|---|
| **Course** | `e_C(c) = { v : course(v) = c }`, mỗi khóa học 1 hyperedge | mọi người học trong khóa "Toán cao cấp 2016_T1" |
| **Object** | `e_O(o) = { v : v có sự kiện video / bài tập / forum với object o trong ngày 0–34 }`, key `course|loại|object_id` | mọi người xem video số 17 của khóa đó |
| **User (temporal)** | mỗi enrollment a là một anchor: `e_U(a) = {a} ∪ { m : user(m) = user(a), s(m) ≤ s(a) }` | khóa hiện tại của người học A + các khóa A học trước đó |

Quy tắc chung: **bỏ hyperedge có ít hơn 2 thành viên.** Hyperedge chỉ có 1 thành viên không mang thêm thông tin, vì self-loop đã làm đúng việc đó.

Ghi chú:
- Action loại "web_page" (click_info, click_about, …) không có object riêng, nên không tạo Object.
- 0.38% sự kiện object không có id thì bị bỏ qua.

### 1.3 Self-loop: đảm bảo 100%

- Self-loop **không lưu** trong `hypergraph.npz`.
- `5_graph_data.add_self_loops` thêm cho **mọi** node khi đọc graph: `e_S(v) = {v}`, với N hyperedge, mỗi cái cỡ 1.
- Target val/test cũng luôn có self-loop của riêng nó.
- Vì vậy mọi node có ít nhất 1 hyperedge, nên bậc d(v) ≥ w_self > 0. Phép d(v)^-1/2 không bao giờ chia cho 0.

### 1.4 Mỗi loại phủ bao nhiêu node (đo thực tế)

| Loại | Số hyperedge trong H0 | Cỡ trung vị (lớn nhất) | % node train có | % target val | % target test |
|---|---|---|---|---|---|
| Course | 247 | 377 (2,361) | **100%** (khóa nhỏ nhất có 151 node train) | **100%** | **100%** |
| Object | 21,747 | 27 (1,893) | 88.4% | 88.4% | 88.4% |
| User (temporal) | 66,552 | 2 (111) | 53.4% | 53.2% | 53.6% |
| Self-loop | N (thêm khi đọc) | 1 | **100%** | **100%** | **100%** |

Khi node thiếu một loại thì sao:
- Node không có Object hoặc User chỉ đơn giản là có ít hyperedge hơn.
- Công thức vẫn chạy đúng, vì d(v) và tổng Σ_e chỉ tính trên các hyperedge mà node có.
- W học theo loại không bị lệch, vì mỗi hyperedge chỉ đóng góp khi tồn tại.

Tỷ lệ giữa train và target gần như trùng nhau ở mọi loại. Điều này cho thấy target nhìn thấy cấu trúc graph giống như node train.

### 1.5 Bảng Object 1 thành viên: để làm gì, và đề xuất bỏ

**Lý do có đề xuất này.** Luật "cỡ ≥ 2" đang được áp hơi khác nhau giữa node train và target:

| | Node train v | Target t (val/test) |
|---|---|---|
| Object o được dùng khi | o có ≥ 2 node train (v và ≥ 1 người khác) | o có trong H0, tức o có ≥ 2 node train **khác t** |
| Tức cần thêm | ≥ 1 người khác | ≥ 2 người khác |

Bảng Object 1 thành viên (32,109 object chỉ có đúng 1 node train) giúp target dùng cả những object có đúng 1 người khác. Như vậy luật sẽ y hệt node train.

**Đo ảnh hưởng thật:**

| | Validation | Test |
|---|---|---|
| Membership Object của target thuộc loại "1 thành viên train" | 274 / 388,924 (0.07%) | 628 / 834,166 (0.08%) |
| Số target có thêm Object đầu tiên nhờ bảng này | 0 | 2 |
| % target có Object: không có bảng → có bảng | 88.43% → 88.43% | 88.37% → 88.38% |

**Đề xuất: bỏ bảng.**
- Lợi ích gần như bằng 0, mà phải thêm 2 mảng vào `hypergraph.npz` và thêm một nhánh code trong `5_graph_data`.
- Thay vào đó, ghi rõ luật trong comment và trong bài: "Target dùng các hyperedge Course/Object có trong H0, cộng User temporal của nó."
- Khác biệt này chỉ chạm 0.08% membership.

Nếu em muốn khớp tuyệt đối thì giữ bảng; chỉ thêm khoảng 15 dòng code.

### 1.6 Nội dung `hypergraph.npz`

| Mảng | Kích thước | Ý nghĩa |
|---|---|---|
| `node_ids` | [M] | node train của từng membership |
| `edge_ids` | [M] | hyperedge của từng membership, đã sắp xếp nên thành viên của e nằm liền nhau |
| `edge_family` | [E] | loại của hyperedge: 0 = course, 1 = object, 2 = user |
| `edge_keys` | [E] | tên: `course_id`, `course|type|object_id`, `user|user_id|anchor` |
| `node_start` | [N] | s(v), ngày bắt đầu khóa học của node train (số ngày) |

Ở đây E = 247 + 21,747 + 66,552 = 88,546 hyperedge và N = 126,354 node train.

### 1.7 Các hàm (khoảng 110 dòng)

```
read_train_nodes(path)            → list node (node_id, user_id, course_id, s(v))
read_object_events(path)          → yield (node_id, "course|type|object_id")
course_hyperedges(nodes)          → e_C(c) = {v : course(v) = c}
object_hyperedges(path)           → e_O(o) = {v : v dùng o}
user_hyperedges(nodes)            → e_U(a) = {a} ∪ {m : user(m)=user(a), s(m) ≤ s(a)}
build_hypergraph(output_dir)      → gộp 3 loại, bỏ |e| < 2, ghi hypergraph.npz, in số hyperedge mỗi loại
```

---

## 2. File `5_graph_data.py`: đọc graph lúc train và lúc đánh giá

### 2.1 Việc của file

File này **chuẩn bị dữ liệu**, không tính toán mô hình. Nó có 2 việc:
1. **Cho train:** đọc X, nhãn và H0 của tập train, rồi thêm self-loop.
2. **Cho val/test:** với mỗi target, tìm danh sách hyperedge của nó.

Không có node val/test nào được đưa vào H0, và các target không nối với nhau.

### 2.2 `load_train_graph(output_dir)`

```
Đọc:   train/X.npy, train.csv (nhãn), hypergraph.npz
Thêm:  self-loop cho mỗi node train: e_S(v) = {v}
Trả:   X       [N, 92]   đặc trưng
       y       [N]       nhãn dropout
       graph = {num_nodes N, node_ids, edge_ids, edge_family, edge_size δ(e), node_start s(v)}
               gồm E + N hyperedge (88,546 + 126,354 self-loop)
```

### 2.3 `load_targets(output_dir, split)`: tìm hyperedge của từng target

**Ba bảng tra cứu**, dựng một lần từ `hypergraph.npz` và `train.csv`:

```
course_edge[course_id]   → id hyperedge Course trong H0
object_edge[object_key]  → id hyperedge Object trong H0 (chỉ các object có trong H0)
train_by_user[user_id]   → (các node train của người học, ngày bắt đầu của từng node)
```

**Với mỗi target t**, đọc thông tin từ `{split}.csv` và `{split}/X.npy`:

```
1. Course:  c(t)  → course_edge[c(t)]                                  (luôn có)
2. Object:  với mỗi object o mà t dùng trong ngày 0–34:
            nếu o có trong object_edge → thêm object_edge[o]            (88% target có ≥ 1)
3. User:    U(t) = { m ∈ train_by_user[user(t)] : s(m) ≤ s(t) }
            nếu U(t) khác rỗng → thêm hyperedge User mới {t} ∪ U(t)    (53% target có)
4. Self-loop của t: luôn có, xử lý trực tiếp trong 6_hgnn
5. Lớp bảo vệ thứ tự thời gian: bỏ hyperedge nào có thành viên u với s(u) > s(t)
   (Course và Object cùng ngày bắt đầu, User đã lọc s ≤ s(t), nên dòng này không bỏ gì;
    chỉ để chắc chắn)
```

**Kết quả trả về** (mảng phẳng kèm con trỏ, để 6_hgnn tính theo batch):

```
X         [T, 92]   đặc trưng target          y      [T]  nhãn (chỉ dùng để tính chỉ số)
start     [T]       s(t)
h0_ptr    [T+1]     h0_edges[h0_ptr[t]:h0_ptr[t+1]] = id các hyperedge H0 của t (Course + Object)
h0_edges  [·]
user_ptr  [T+1]     user_nodes[user_ptr[t]:user_ptr[t+1]] = U(t), các thành viên train của User(t)
user_nodes[·]
```

### 2.4 Ví dụ một target

```
Target t: người học A, khóa "Toán 2016_T2" (bắt đầu 2016-09-01), xem video 17 và 23, làm bài 5
  Course:  e_C(Toán 2016_T2)         → 377 node train            (có trong H0)
  Object:  video 17                  → 41 node train             (có trong H0)
           video 23                  → 1 node train              (KHÔNG có trong H0 → bỏ, xem 1.5)
           bài 5                     → 12 node train             (có trong H0)
  User:    A có 2 khóa train bắt đầu 2016-02-20 và 2016-09-01 → U(t) = 2 node
           (khóa A bắt đầu 2017-02-20 bị loại vì muộn hơn t)
  Self-loop: {t}
  → E(t) = {Course, video 17, bài 5, User(t), self-loop}: 5 hyperedge
```

### 2.5 Thay cho phần nào của code cũ

| Code cũ (`4_hypergraph.py`) | Code mới (`5_graph_data.py`) |
|---|---|
| `load_evaluation_split` + `build_local_graph` + `merge_local_graphs`: dựng một graph nhỏ hàng trăm node cho **mỗi** target | `load_targets`: chỉ trả danh sách hyperedge của mỗi target; việc tính toán chuyển sang `6_hgnn` (Bước B) |
| `select_families`, `node_permutation`, `candidate_*` | bỏ |

---

## 3. File `6_hgnn.py`: nhánh graph (HGNN 2 lớp)

### 3.1 Kết luận rà soát

**Công thức đúng, đúng 2 layer.** Thầy đối chiếu từng dòng của `src/5_model.py` với bài HGNN (Eq. 10) và code gốc `baseline/HGNN`. Bảng đối chiếu nằm ở mục 6.2 của PLAN_RUT_GON_v5.md. File mới **chép nguyên** các dòng lan truyền đó; chỉ đổi tên và thêm comment công thức.

### 3.2 Công thức một lớp

```
G = Dv^-1/2 · H_recv · W · De^-1 · Hᵀ · Dv^-1/2

  H        [N, E']   h(v,e) = 1 nếu v ∈ e                    (bên gửi, không chặn)
  H_recv   [N, E']   h_recv(v,e) = h(v,e)·1[s(v) ≥ max_{u∈e} s(u)]   (bên nhận, causal)
  W        [E', E']  w_e = softplus(θ_{f(e)}), 4 tham số θ: course, object, user, self_loop
  De       [E', E']  δ(e) = |e|
  Dv       [N, N]    d(v) = Σ_e h_recv(v,e)·w_e              (causal degree)
```

Viết theo từng node, đúng thứ tự trong code:

```
(1) d(v)  = Σ_e h_recv(v,e)·w_e                         bậc của node
(2) a_u   = d(u)^-1/2 · x_u                            chuẩn hóa bên gửi
(3) m_e   = (w_e / δ(e)) · Σ_{u∈e} a_u                 node → hyperedge
(4) x'_v  = d(v)^-1/2 · Σ_e h_recv(v,e) · m_e          hyperedge → node
```

### 3.3 Hai lớp: đúng 2 lần lan truyền

```
X [N,92] ──Θ1,b1──► XΘ1+b1 [N,128] ──G──► ReLU ──► Z1 [N,128]
                                                    │ Dropout(0.5) (chỉ khi train)
Z1 ──Θ2,b2──► Z1Θ2+b2 [N,128] ──G──► ReLU ──► Z_g [N,128]

Lớp 1:  Z1  = ReLU( G · (X·Θ1 + b1) )                     ← lần lan truyền thứ 1 (1 bước nhảy)
Lớp 2:  Z_g = ReLU( G · (Dropout(Z1)·Θ2 + b2) )           ← lần lan truyền thứ 2 (2 bước nhảy)
```

- **Đúng 2 lớp:** có 2 phép nhân G. Vì vậy một node nhận thông tin từ hàng xóm cách tối đa 2 bước, ví dụ: t → bạn cùng khóa u → các khóa khác của u.
- **Theo code gốc `HGNN_embedding`:** conv1 → ReLU → dropout → conv2 → ReLU. Bias cộng trước khi nhân G, giống `HGNN_conv`.
- **Dropout:** chỉ chạy khi `model.train()`. Code gốc gọi dropout thiếu `training=`; em không chép điểm này.
- Classifier không nằm trong file này. Nó ở `8_model.py`: `logit = [Z_g ‖ Z_s]·u + b`.

### 3.4 Các hàm trong file

| Hàm | Dùng khi | Công thức |
|---|---|---|
| `prepare_graph(graph, device)` | train và Bước A | dựng `H_T`, `H_recv` (causal), `δ(e)` |
| `HGNNEncoder.family_weights()` | mọi lúc | `w_f = softplus(θ_f)`, khởi tạo w = 1 |
| `HGNNEncoder.propagate(x, graph, w)` | train và Bước A | (1)–(4) ở mục 3.2 |
| `HGNNEncoder.forward(x, graph)` | **train** (full-batch H0) | 2 lớp ở mục 3.3 |
| `HGNNEncoder.cache_train_states(x, graph)` | **Bước A**, ở eval và `no_grad` | `d(u)`, `S1_e`, `Z1_u`, `S2_e` trên H0 |
| `HGNNEncoder.forward_targets(cache, targets)` | **Bước B**, ở eval và `no_grad` | 2 lớp cho target (mục 3.5) |

### 3.5 Hai lớp cho target val/test (Bước A + B)

Đây vẫn là 2 lớp với cùng công thức. Điểm khác duy nhất: target được thêm vào các hyperedge của nó, còn node train giữ trạng thái đã tính trên H0.

```
Bước A (trên H0, một lần mỗi lần đánh giá):
  S1_e = Σ_{u∈e} d(u)^-1/2 · (x_u·Θ1 + b1)         tổng gửi lớp 1 của hyperedge e
  Z1_u = ReLU( G·(XΘ1 + b1) )_u                    trạng thái lớp 1 của node train u
  S2_e = Σ_{u∈e} d(u)^-1/2 · (Z1_u·Θ2 + b2)         tổng gửi lớp 2 của hyperedge e

Bước B (cho mỗi target t, E(t) lấy từ 5_graph_data):
  d(t)  = Σ_{e∈E(t)} w_e
  Lớp 1: z1_t = ReLU( d(t)^-1/2 · Σ_{e∈E(t)} w_e/(δ(e)+1) · ( S1_e + d(t)^-1/2·(x_t·Θ1 + b1) ) )
  Lớp 2: z_t  = ReLU( d(t)^-1/2 · Σ_{e∈E(t)} w_e/(δ(e)+1) · ( S2_e + d(t)^-1/2·(z1_t·Θ2 + b2) ) )
```

Trong đó:
- δ(e) là số thành viên train của e; cộng 1 vì có thêm t.
- Với hyperedge User(t), S_e được tính trực tiếp từ U(t).
- Với self-loop của t: S_e = 0 và δ = 0.

Đối chiếu với 4 dòng (1)–(4) ở mục 3.2:

| Dòng | Lúc train (node v trong H0) | Target t |
|---|---|---|
| (1) bậc | d(v) = Σ_e h_recv(v,e)·w_e | d(t) = Σ_{e∈E(t)} w_e (t nhận từ mọi e ∈ E(t)) |
| (2)+(3) node → hyperedge | m_e = (w_e/δ(e))·Σ_{u∈e} d(u)^-1/2·x_u | m_e = (w_e/(δ(e)+1))·(S_e + d(t)^-1/2·x_t) |
| (4) hyperedge → node | x'_v = d(v)^-1/2·Σ_e h_recv(v,e)·m_e | x'_t = d(t)^-1/2·Σ_{e∈E(t)} m_e |

**Hai phép thử chứng minh code đúng** (mục 12.4 và 12.5 của bản 5):
1. Cho 1,000 node **train** đi qua Bước B, với 3 điều kiện: chế độ eval; E(v) theo H_recv; không cộng δ+1 và không cộng thêm số hạng của v. Kết quả phải **trùng** forward đầy đủ trên H0, sai lệch < 1e-5.
2. Với khoảng 50 target, tính bằng vòng lặp viết thẳng theo định nghĩa (từng hyperedge, từng thành viên) và so với Bước B, sai lệch < 1e-5.

---

## 4. Điểm em cần xác nhận cho 3 file này

- [ ] 4_hypergraph: dựng Course, Object, User temporal (so theo **ngày**), bỏ |e| < 2. Self-loop thêm lúc đọc.
- [ ] **Bỏ bảng Object 1 thành viên** (ảnh hưởng 0.08%), và ghi luật "target dùng Course/Object có trong H0 + User temporal" trong comment và trong bài. Nếu em muốn giữ bảng, đánh dấu ô này là "giữ".
- [ ] 5_graph_data: `load_train_graph` + `load_targets` trả danh sách hyperedge dạng con trỏ (mục 2.3). Bỏ local star graph.
- [ ] 6_hgnn: giữ nguyên các dòng lan truyền 2 lớp của code hiện tại, thêm Bước A/B và 2 phép thử kiểm chứng.
