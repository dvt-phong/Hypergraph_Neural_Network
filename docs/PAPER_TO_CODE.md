# Đối chiếu bài báo HGNN, HSL với code

Tài liệu này đặt từng công thức của hai bài báo cạnh đoạn code cài đặt nó, để
rà soát xem code làm đúng như bài báo hay đã điều chỉnh. Số trang là số trang
in trong proceedings (in ở chân trang PDF).

| Bài báo | Proceedings | Trang |
|---|---|---|
| **HGNN**: Feng et al., “Hypergraph Neural Networks” | AAAI-19 | 3558–3565 |
| **HSL**: Cai et al., “Hypergraph Structure Learning for Hypergraph Neural Networks” | IJCAI-22 | 1923–1929 |

Nhãn trạng thái dùng trong tài liệu:

- **Khớp**: code tính đúng công thức (có thể viết dưới dạng khác).
- **Điều chỉnh**: khác bài báo có chủ đích, cần nêu khi viết luận án/bài báo.
- **Cần xem**: điểm bài báo không nói rõ hoặc code có lựa chọn riêng nên rà lại.

Ký hiệu: bài báo viết node embedding là `x`, `x̂`; code gọi là `Z0`, `Z*`.
`H0` trong code là `H` trong bài báo, `H*` là `Ĥ`.

---

## 0. Bảng tra nhanh

| Bài báo | Mục | Trang | Công thức | Code | Trạng thái |
|---|---|---|---|---|---|
| HGNN | Hypergraph learning statement | 3560 | Eq. 1 (incidence `H`) | [4_hypergraph.py:115-126](../src/4_hypergraph.py#L115-L126) | Khớp |
| HGNN | 〃 | 3560 | `d(v)`, `δ(e)`, `Dv`, `De` | [5_model.py:64-67](../src/5_model.py#L64-L67) | Khớp |
| HGNN | 〃 | 3560 | Eq. 2–4 (regularizer, Laplacian) | — | Lý thuyết |
| HGNN | Spectral convolution | 3561 | Eq. 5–9 (dẫn xuất) | — | Lý thuyết |
| HGNN | 〃 | 3561 | Eq. 10 (hyperedge convolution) | [5_model.py:61-71](../src/5_model.py#L61-L71) | Khớp |
| HGNN | 〃 | 3561 | Eq. 11 (layer) | [5_model.py:95-98](../src/5_model.py#L95-L98) | Điều chỉnh |
| HGNN | Analysis, Fig. 4 | 3561–3562 | node → edge → node | [5_model.py:69-71](../src/5_model.py#L69-L71) | Khớp |
| HGNN | Implementation | 3562 | kNN hyperedge | [4_hypergraph.py:107-109](../src/4_hypergraph.py#L107-L109) | Điều chỉnh |
| HGNN | Visual datasets, Fig. 5 | 3563 | ghép `H = [H1 ‖ … ‖ Hm]` | [4_hypergraph.py:102-112](../src/4_hypergraph.py#L102-L112) | Khớp |
| HSL | §3.1 Problem definition | 1924 | `H ∈ {0,1}^{N×E}`, `F(X, Ĥ)` | [5_model.py:100-120](../src/5_model.py#L100-L120) | Khớp |
| HSL | §3.2 HGNNs | 1925 | Eq. 1 (two-stage message passing) | [5_model.py:61-71](../src/5_model.py#L61-L71) | Điều chỉnh |
| HSL | §3.3 Hyperedge sampling | 1925 | Eq. 2 (Gumbel, `m`) | [6_hsl.py:40-47](../src/6_hsl.py#L40-L47), [6_hsl.py:92-98](../src/6_hsl.py#L92-L98) | Điều chỉnh |
| HSL | 〃 | 1925 | Eq. 3 (`H̃ = Me ⊙ H`) | [6_hsl.py:107](../src/6_hsl.py#L107) | Khớp |
| HSL | §3.4 Incident node sampling | 1926 | Eq. 4 (similarity `S`) | [6_hsl.py:122-126](../src/6_hsl.py#L122-L126) | Điều chỉnh |
| HSL | 〃 | 1926 | Eq. 5 (`ΔH`, top `p_add`) | [6_hsl.py:113-130](../src/6_hsl.py#L113-L130) | Điều chỉnh |
| HSL | 〃 | 1926 | Eq. 6 (`Zv = σ(MLP([x‖h]))`) | [6_hsl.py:133-148](../src/6_hsl.py#L133-L148) | Khớp |
| HSL | 〃 | 1926 | Eq. 7 (Gumbel, `Mv`) | [6_hsl.py:100-105](../src/6_hsl.py#L100-L105) | Điều chỉnh |
| HSL | 〃 | 1926 | Eq. 8 (`Ĥ = Me ⊙ Mv ⊙ (H + ΔH)`) | [6_hsl.py:87-107](../src/6_hsl.py#L87-L107) | Khớp |
| HSL | 〃 | 1926 | Eq. 9 (`+ I`, self-loop) | [4_hypergraph.py:185-195](../src/4_hypergraph.py#L185-L195), [6_hsl.py:98](../src/6_hsl.py#L98), [6_hsl.py:105](../src/6_hsl.py#L105) | Khớp |
| HSL | §3.5 Contrastive | 1926 | Eq. 10 (`L_CL`) | [7_losses.py:62-85](../src/7_losses.py#L62-L85) | Điều chỉnh |
| HSL | 〃 | 1927 | Eq. 11 (`L = L_T + λ L_CL`) | [7_losses.py:88-94](../src/7_losses.py#L88-L94) | Điều chỉnh |
| HSL | §4.1 Setups | 1927 | siêu tham số, split | [8_train.py:38-59](../src/8_train.py#L38-L59) | Điều chỉnh |
| HSL | §4.2 Pruning rate | 1927 | `r^v` | [6_hsl.py:151-161](../src/6_hsl.py#L151-L161) | Cần xem |
| HSL | §4.2 Ablation, Fig. 3 | 1928 | Base, +edge, +node, −CL | [8_train.py:275-278](../src/8_train.py#L275-L278) | Cần xem |

---

## 1. HGNN (Feng et al., AAAI 2019)

### 1.1 Incidence matrix (Eq. 1, trang 3560, cột phải)

**Bài báo**

$$
h(v, e) =
\begin{cases}
1, & v \in e \\
0, & v \notin e
\end{cases}
\qquad H \in \{0,1\}^{|\mathcal V| \times |\mathcal E|}
$$

**Code:** không lưu ma trận `H`, mà lưu danh sách các ô bằng 1 (membership):
`node_ids[m]` thuộc hyperedge `edge_ids[m]`
([4_hypergraph.py:3-5](../src/4_hypergraph.py#L3-L5),
[4_hypergraph.py:115-126](../src/4_hypergraph.py#L115-L126)).

**Trạng thái:** Khớp. Đây là dạng sparse (COO) của `H`, vì `H` đầy đủ của
XuetangX không vừa bộ nhớ.

### 1.2 Bậc của node và hyperedge (trang 3560, cột phải)

**Bài báo**

$$
d(v) = \sum_{e \in \mathcal E} \omega(e)\, h(v,e), \qquad
\delta(e) = \sum_{v \in \mathcal V} h(v,e)
$$

`Dv`, `De` là ma trận đường chéo của `d(v)`, `δ(e)`.

**Code:** [5_model.py:62-67](../src/5_model.py#L62-L67)

```python
constant_weights = weights.detach()
node_degree = x.new_zeros(num_nodes).index_add_(0, node_ids, constant_weights)   # d(v)
edge_degree = x.new_zeros(num_edges).index_add_(0, edge_ids, constant_weights)   # δ(e)
node_scale = node_degree.clamp_min(1.0).rsqrt()     # Dv^-1/2
edge_scale = edge_degree.clamp_min(1.0).reciprocal() # De^-1
```

**Trạng thái:** Khớp với `ω(e) = 1` (`W = I`). Có hai chi tiết riêng của code:

- `clamp_min(1)` tránh chia cho 0 khi HSL xoá hết membership của một hyperedge.
  Bài báo không gặp trường hợp này.
- **Cần xem:** bậc được tính từ `weights.detach()`, nên gradient của mask không
  đi qua phần chuẩn hoá. Bài báo HSL không nói cách xử lý chỗ này.

### 1.3 Regularizer và hypergraph Laplacian (Eq. 2–4, trang 3560)

$$
\Omega(f) = \frac12 \sum_{e}\sum_{\{u,v\}} \frac{w(e)h(u,e)h(v,e)}{\delta(e)}
\left(\frac{f(u)}{\sqrt{d(u)}} - \frac{f(v)}{\sqrt{d(v)}}\right)^2,
\qquad
\Delta = I - D_v^{-1/2} H W D_e^{-1} H^\top D_v^{-1/2}
$$

**Code:** không có. Đây là nền tảng lý thuyết dẫn tới Eq. 10.

### 1.4 Từ spectral convolution tới Eq. 9 (Eq. 5–9, trang 3561, cột trái)

Bài báo xấp xỉ Chebyshev bậc `K = 1`, lấy `λmax ≈ 2`, rồi gộp hai tham số thành
một (Eq. 8), được:

$$
g \star x \approx \theta\, D_v^{-1/2} H W D_e^{-1} H^\top D_v^{-1/2} x \qquad (9)
$$

**Code:** không có. Đây là dẫn xuất; code chỉ cài dạng cuối (Eq. 10–11).

### 1.5 Hyperedge convolution (Eq. 10, trang 3561, cột phải)

**Bài báo**

$$
Y = D_v^{-1/2} H W D_e^{-1} H^\top D_v^{-1/2} X \Theta \qquad (10)
$$

`W` được khởi tạo bằng ma trận đơn vị.

**Code:** [5_model.py:61-71](../src/5_model.py#L61-L71), đọc từ phải sang trái:

| Phép trong Eq. 10 | Dòng code |
|---|---|
| `X Θ` | `self.layer1(x)` / `self.layer2(hidden)` ([5_model.py:96-98](../src/5_model.py#L96-L98)) |
| `Dv^-1/2 ·` | `x * node_scale` |
| `Hᵀ ·` (node → hyperedge) | `weighted_sum(..., node_ids, edge_ids, num_edges)` ([5_model.py:69](../src/5_model.py#L69)) |
| `De^-1 ·` | `edge_x * edge_scale` |
| `H ·` (hyperedge → node) | `weighted_sum(..., edge_ids, node_ids, num_nodes)` ([5_model.py:70](../src/5_model.py#L70)) |
| `Dv^-1/2 ·` | `node_x * node_scale` ([5_model.py:71](../src/5_model.py#L71)) |

**Trạng thái:** Khớp với `W = I`.

- `weights` nhân vào cả hai chiều (`Hᵀ` và `H`). Với `H0`, `weights = 1`; với
  `H*`, `weights ∈ {0,1}` nên `w² = w` và kết quả đúng bằng dùng `H*` làm
  incidence matrix.
- Bỏ một hyperedge bằng `Me` (mask 0) tương đương đặt `ω(e) = 0` trong `W` của
  HGNN.
- `nn.Linear` có bias, nên code tính `(XΘ + b)` rồi mới lan truyền. Repo gốc
  `iMoonLab/HGNN` cũng làm vậy; Eq. 10 thì không có bias.

### 1.6 Một layer HGNN (Eq. 11, trang 3561) và Fig. 4 (trang 3561–3562)

**Bài báo**

$$
X^{(l+1)} = \sigma\!\left(D_v^{-1/2} H W D_e^{-1} H^\top D_v^{-1/2} X^{(l)} \Theta^{(l)}\right) \qquad (11)
$$

Fig. 4 và đoạn đầu trang 3562 mô tả layer này là biến đổi node → hyperedge →
node: `Θ` biến đổi feature, `Hᵀ` gom thành feature hyperedge `R^{E×C2}`, rồi `H`
gom ngược về node.

**Code:** [5_model.py:95-98](../src/5_model.py#L95-L98)

```python
hidden = F.relu(hgnn_propagate(self.layer1(x), graph, weights))
hidden = F.dropout(hidden, self.dropout, self.training)
return F.relu(hgnn_propagate(self.layer2(hidden), graph, weights))
```

**Trạng thái:** Điều chỉnh. Bài báo (trang 3562, “Model for node
classification”) dùng hai layer HGNN; layer thứ hai cho ra thẳng logits rồi
qua softmax. Code thêm `ReLU` sau layer thứ hai để được embedding `Z`, rồi mới
qua `classifier = nn.Linear(hidden, 1)`
([5_model.py:106](../src/5_model.py#L106),
[5_model.py:114](../src/5_model.py#L114)). Lý do là HSL cần `Z0` và `Z*` ở
không gian ẩn cho ΔH, `Mv` và loss contrastive.

### 1.7 Dựng hypergraph (trang 3562 cột trái, trang 3563 cột phải, Fig. 5)

**Bài báo:** mỗi đỉnh cùng `K` láng giềng gần nhất (khoảng cách Euclid) tạo một
hyperedge, nên có `N` hyperedge, mỗi cái `K + 1` đỉnh. Thí nghiệm thị giác dùng
`K = 10`. Khi có nhiều modality, mỗi modality cho một `Hi` và
`H = [H1 ‖ … ‖ Hm]` (ghép theo cột).

**Code**

- kNN: [4_hypergraph.py:58-79](../src/4_hypergraph.py#L58-L79) và
  [4_hypergraph.py:107-109](../src/4_hypergraph.py#L107-L109); `k = 10` mặc
  định ([4_hypergraph.py:336](../src/4_hypergraph.py#L336)).
- Ghép nhiều nhóm hyperedge: Course, Object, Behavioral được nối vào cùng một
  danh sách, `edge_family` ghi nhóm của từng hyperedge
  ([4_hypergraph.py:102-112](../src/4_hypergraph.py#L102-L112)).

**Trạng thái**

- Ghép nhóm: Khớp với ý tưởng Fig. 5.
- kNN: Điều chỉnh. Code dùng **cosine** trên riêng các cột hành vi
  (`BEHAVIOR_FEATURE_SLICE`) thay vì Euclid trên toàn bộ feature, và chỉ nối
  train với train (chống leakage).

### 1.8 Thiết lập huấn luyện (trang 3562, cột phải)

| | HGNN (citation) | Code ([8_train.py:38-59](../src/8_train.py#L38-L59)) |
|---|---|---|
| Số layer | 2 | 2 |
| Hidden | 16 | 128 |
| Dropout | 0.5 | 0.3 |
| Activation | ReLU | ReLU |
| Optimizer | Adam, lr 0.001 | Adam, lr 1e-4, weight decay 5e-4 |
| Loss | cross-entropy | BCE có `pos_weight` (mất cân bằng lớp) |

---

## 2. HSL (Cai et al., IJCAI 2022)

### 2.1 Định nghĩa bài toán (§3.1, trang 1924, cột phải)

**Bài báo:** `H ∈ {0,1}^{N×E}`, `𝒩_i = {e ∈ ℰ | i ∈ e}`, `X ∈ R^{N×d_v}`. Mục
tiêu là học đồng thời `F(X, Ĥ)` và `Ĥ`.

**Code:** `HSLModel.forward` ([5_model.py:100-120](../src/5_model.py#L100-L120)):

```text
Z0 = encode(X, H0)                 # HGNN trên cấu trúc gốc
H* = StructureLearner(Z0, h_e, H0)
Z* = encode(X, H*)                 # cùng trọng số với lượt Z0
logits = classifier(Z*)
```

**Trạng thái:** Khớp. Khác biệt duy nhất là bài toán: bài báo làm phân loại node
transductive nhiều lớp, code làm dự đoán dropout nhị phân inductive
(validation/test dùng local graph, xem mục 2.12).

### 2.2 HGNN dạng message passing hai giai đoạn (§3.2, Eq. 1, trang 1925, cột trái)

**Bài báo**

$$
h_j = f_{\mathcal V \to \mathcal E}\big(\{x_k\}_{k \in e_j}\big), \qquad
\tilde x_i = f_{\mathcal E \to \mathcal V}\big(\{h_k\}_{k \in \mathcal N_i},\, x_i\big) \qquad (1)
$$

HSL dùng AllSetTransformer làm backbone.

**Code:** backbone là HGNN (mục 1.5). `f_{V→E}` là `Hᵀ` kèm chuẩn hoá `Dv^-1/2`,
`De^-1`; `f_{E→V}` là `H` kèm `Dv^-1/2`. Phần `x_i` trong `f_{E→V}` được đưa vào
qua self-loop (mục 2.9).

**Trạng thái:** Điều chỉnh (đã ghi trong [references.md](references.md)).

**Biểu diễn hyperedge `h_j`** mà HSL dùng cho Eq. 4 và Eq. 6: bài báo lấy từ
HGNN. Code lấy trung bình `Z0` của các thành viên trong `H0`
([5_model.py:75-80](../src/5_model.py#L75-L80)):

$$
h_e = \frac{1}{|e|} \sum_{v \in e} z^{(0)}_v
$$

**Cần xem:** `h_e` không đi qua `Dv^-1/2` như trong layer HGNN, và được tính một
lần từ `Z0` (sau layer cuối) chứ không phải từ bên trong layer.

### 2.3 Hyperedge sampling (§3.3, Eq. 2–3, trang 1925)

**Bài báo**

$$
m_i = \sigma\!\left(\frac{1}{\tau}\Big(\log\frac{z^e_i}{1-z^e_i} + (\epsilon^0 - \epsilon^1)\Big)\right),
\quad \epsilon^0, \epsilon^1 \sim \text{Gumbel}(0,1) \qquad (2)
$$

$$
\tilde H = M^e \odot H, \qquad M^e = \Gamma(m) \qquad (3)
$$

`z^e ∈ [0,1]^E` là **tham số tự do**, mỗi hyperedge một số.

**Code**

`keep_mask` ([6_hsl.py:40-47](../src/6_hsl.py#L40-L47)):

```python
uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
logistic_noise = torch.log(uniform) - torch.log1p(-uniform)  # = ε0 − ε1
soft = torch.sigmoid((logits + logistic_noise) / temperature)  # Eq. 2
hard = (soft > 0.5).float()
return hard + soft - soft.detach()                              # straight-through
```

Tính `logits` và broadcast `Γ` ([6_hsl.py:92-98](../src/6_hsl.py#L92-L98),
[6_hsl.py:107](../src/6_hsl.py#L107)):

```python
edge_logits = self.edge_scorer(torch.cat([edge_representations, family], dim=1))
edge_keep   = keep_mask(edge_logits.squeeze(-1), self.temperature, self.training)
...
weights = edge_keep[edge_ids] * membership_keep   # Γ(m): chép m_e cho mọi membership của e
```

**Đối chiếu từng phần**

| Phần | Bài báo | Code | Trạng thái |
|---|---|---|---|
| `log(z/(1−z))` | log-odds của tham số tự do `z^e_i` | `MLP([h_e ‖ one-hot family])` cho ra log-odds | Điều chỉnh (để chạy được trên local graph chưa thấy) |
| `ε0 − ε1` | hiệu hai Gumbel(0,1) | `log u − log(1−u)`, `u ~ U(0,1)`: phân phối Logistic(0,1), trùng với hiệu hai Gumbel | Khớp |
| `τ` | nhiệt độ | `temperature = 0.4` ([6_hsl.py:56](../src/6_hsl.py#L56)) | Cần xem (xem mục 3) |
| Giá trị `m_i` | liên tục trong (0,1) (Binary Concrete) | 0/1 ở forward, gradient của bản mềm (straight-through) | Điều chỉnh |
| Khởi tạo | không nêu | bias = 3, `σ(3) ≈ 0.95`: ban đầu giữ gần hết ([6_hsl.py:35-36](../src/6_hsl.py#L35-L36)) | Riêng |
| Lúc eval | không nêu | `logit > 0` (xác suất > 0.5), không có nhiễu ([6_hsl.py:41-42](../src/6_hsl.py#L41-L42)) | Riêng |
| `Γ(m)` | lặp `m` thành ma trận `N×E` | `edge_keep[edge_ids]` | Khớp |

### 2.4 Similarity node–hyperedge (§3.4, Eq. 4, trang 1926, cột trái)

**Bài báo:** cosine có trọng số, nhiều head.

$$
S_{ij} = \frac{1}{n_c}\sum_{i=1}^{n_c} \cos\big(w_i \odot \tilde x_i,\; w_i \odot h_j\big) \qquad (4)
$$

**Code:** [6_hsl.py:122-126](../src/6_hsl.py#L122-L126)

```python
similarity = F.cosine_similarity(z0[candidate_nodes], edge_representations[candidate_edges][:, None, :], dim=-1)
```

**Trạng thái:** Điều chỉnh. Code dùng `n_c = 1`, `w = 1` (cosine thường, không có
tham số học), và chỉ tính `S` cho các cặp ứng viên chứ không cho cả ma trận
`N×E`.

### 2.5 Kết nối ẩn ΔH (Eq. 5, trang 1926, cột trái)

**Bài báo**

$$
\Delta H_{ij} =
\begin{cases}
1, & H_{ij} = 0 \text{ và } S_{ij} \in \text{top}(S, p_{add}) \\
0, & H_{ij} = 1
\end{cases} \qquad (5)
$$

`top(S, p_add)` chọn các ô lớn nhất trên **toàn ma trận**, số ô bằng
`p_add × nnz(H)`.

**Code:** `implicit_connections` ([6_hsl.py:113-130](../src/6_hsl.py#L113-L130)):

- Chỉ Behavioral hyperedge được thêm node.
- Ứng viên của hyperedge có anchor `a` là láng giềng hạng `k … k_max−1` của `a`
  ([4_hypergraph.py:200-214](../src/4_hypergraph.py#L200-L214)). Các node này
  chắc chắn chưa thuộc hyperedge (`H_ij = 0`), vì thành viên hiện có là `a` và
  láng giềng hạng `0 … k−1`.
- Mỗi hyperedge lấy `add_per_edge = 2` ứng viên có `S` cao nhất (`topk` theo
  hàng).
- `@torch.no_grad()`: không có gradient, giống bài báo (`top` không khả vi).

**Trạng thái:** Điều chỉnh (đã ghi trong [references.md](references.md)).
Không có tham số tương ứng trực tiếp với `p_add`. Tỉ lệ tương đương là
`2 × (số Behavioral hyperedge) / nnz(H0)`.

### 2.6 Xác suất giữ membership (Eq. 6, trang 1926, cột trái)

**Bài báo**

$$
Z^v_{ij} = \sigma\big(\text{MLP}([\tilde x_i \,\|\, h_j])\big) \qquad (6)
$$

**Code:** [6_hsl.py:74-78](../src/6_hsl.py#L74-L78),
[6_hsl.py:133-148](../src/6_hsl.py#L133-L148)

```python
# Lớp đầu W [z_v ‖ h_e] được tách thành W_v z_v + W_e h_e
hidden = F.relu(node_part[node_ids] + edge_part[edge_ids])
return self.membership_output(hidden).squeeze(-1)   # logit; σ nằm trong keep_mask
```

**Trạng thái:** Khớp. `W[z‖h] = W_v z + W_e h` là đồng nhất đại số; tách ra để
mỗi node và mỗi hyperedge chỉ nhân ma trận một lần. Code tính trên mọi
membership của `H0 + ΔH`, nên ΔH cũng bị `Mv` lọc, đúng thứ tự của bài báo
(ΔH trước, `Mv` sau).

### 2.7 Incident node sampling (Eq. 7, trang 1926, cột trái)

**Bài báo**

$$
M^v_{ij} = \sigma\!\left(\frac{1}{\tau}\Big(\log\frac{Z^v_{ij}}{1-Z^v_{ij}} + (\epsilon^0-\epsilon^1)\Big)\right) \qquad (7)
$$

**Code:** [6_hsl.py:100-105](../src/6_hsl.py#L100-L105), cùng `keep_mask` như
Eq. 2.

**Trạng thái:** Điều chỉnh, giống mục 2.3 (mask 0/1 straight-through).

### 2.8 Cấu trúc cuối, chưa tính self-loop (Eq. 8, trang 1926, cột trái)

**Bài báo**

$$
\hat H = M^e \odot M^v \odot (H + \Delta H) \qquad (8)
$$

**Code:** [6_hsl.py:87-107](../src/6_hsl.py#L87-L107)

```python
node_ids = torch.cat([graph["node_ids"], added_nodes])   # H + ΔH
edge_ids = torch.cat([graph["edge_ids"], added_edges])
weights  = edge_keep[edge_ids] * membership_keep          # Me ⊙ Mv
```

**Trạng thái:** Khớp. `H*` là cặp (`node_ids`, `edge_ids`) cộng với `weights`;
các membership có `weight = 0` vẫn nằm trong danh sách nhưng không truyền tin.

### 2.9 Self-loop (Eq. 9, trang 1926, cột trái → cột phải)

**Bài báo:** self-loop được thêm lúc tiền xử lý, chiếm **N hyperedge cuối**, và
không tham gia cả hai bước sampling:

$$
\hat H = M^e \odot M^v \odot (H + \Delta H) + I, \qquad
I_{ij} =
\begin{cases}
1, & i + E = j + N \\
0, & \text{ngược lại}
\end{cases} \qquad (9)
$$

`i + E = j + N` nghĩa là node `i` ứng với hyperedge `j = (E − N) + i`, tức là
hyperedge thứ `i` trong khối N hyperedge cuối.

**Code**

- Thêm vào cuối: `edge_ids = first_self_loop + nodes` với
  `first_self_loop = E − N` ([4_hypergraph.py:185-195](../src/4_hypergraph.py#L185-L195)).
  Đúng quy ước `j = (E − N) + i`.
- Không bị lọc: `torch.where(is_self_loop, 1.0, …)` cho `Me`
  ([6_hsl.py:98](../src/6_hsl.py#L98)) và cho `Mv`
  ([6_hsl.py:105](../src/6_hsl.py#L105)).
- Không nhận ΔH: ΔH chỉ thêm vào Behavioral hyperedge.

**Trạng thái:** Khớp.

### 2.10 Intra-hyperedge contrastive (§3.5, Eq. 10, trang 1926, cột phải)

**Bài báo:** `T_i = {j ∈ e | e ∈ 𝒩_i}` (có cả `i`), `φ` là cosine.

$$
\mathcal L_{CL}(x_i) = -\log
\frac{\exp(\phi(x_i, \hat x_i))}
{\sum_{j \in T_i, j \ne i} \exp(\phi(x_i, x_j)) + \sum_{j \in T_i} \exp(\phi(x_i, \hat x_j))} \qquad (10)
$$

Tổng thứ hai chạy qua cả `j = i`, nên mẫu số đã chứa số hạng dương
`exp(φ(x_i, x̂_i))`.

**Code:** [7_losses.py:62-85](../src/7_losses.py#L62-L85)

```python
positive   = (anchor * other[anchors]).sum(dim=1, keepdim=True)          # φ(x_i, x̂_i)
same_view  = einsum("bd,bkd->bk", anchor, view[neighbors])               # φ(x_i, x_j)
other_view = einsum("bd,bkd->bk", anchor, other[neighbors])              # φ(x_i, x̂_j)
logits = cat([positive, same_view, other_view]) / temperature
logits = masked_fill(... neighbor == anchor ... , -inf)                  # bỏ j = i
return cross_entropy(logits, target=0)
```

Mẫu số của code là: số hạng dương + `Σ_{j≠i}` cùng view + `Σ_{j≠i}` view kia.
Mẫu số của bài báo là: `Σ_{j≠i}` cùng view + `Σ_{j∈T_i}` view kia. Hai cách
viết bằng nhau vì số hạng `j = i` của tổng thứ hai chính là số hạng dương.

**Đối chiếu từng phần**

| Phần | Bài báo | Code | Trạng thái |
|---|---|---|---|
| `x`, `x̂` | embedding từ `H`, `Ĥ` | `Z0`, `Z*` (chung encoder) | Khớp |
| `φ` | cosine | `normalize` rồi tích vô hướng, **chia τ = 0.07** | Điều chỉnh |
| Anchor | mọi node (`1/N Σ_i`) | 1024 node ngẫu nhiên mỗi epoch ([8_train.py:186-187](../src/8_train.py#L186-L187)) | Điều chỉnh |
| `T_i` | toàn bộ node chung hyperedge | 32 mẫu: chọn ngẫu nhiên 1 hyperedge của `i`, rồi 1 thành viên của nó ([7_losses.py:50-59](../src/7_losses.py#L50-L59)) | Điều chỉnh |
| Self-loop trong `T_i` | chỉ góp `i` | bị loại khỏi sampler ([7_losses.py:34](../src/7_losses.py#L34)) | Khớp |
| Chiều | một chiều (anchor ở view `H`) | trung bình hai chiều `Z0→Z*` và `Z*→Z0` | Điều chỉnh (giống code HSL công bố) |

### 2.11 Hàm mục tiêu (Eq. 11, trang 1927, cột trái)

**Bài báo**

$$
\mathcal L = \frac1N \sum_{i=1}^{N}\Big(\mathcal L_T\big(F(x_i, \hat H), y_i\big) + \lambda\, \mathcal L_{CL}(x_i)\Big),
\quad \hat H = M^e \odot M^v \odot (H + \Delta H) + I \qquad (11)
$$

**Code:** [7_losses.py:88-94](../src/7_losses.py#L88-L94)

```python
bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight)
return bce + lambda_cl * cl
```

**Trạng thái:** Điều chỉnh.

- `L_T`: code dùng BCE nhị phân có `pos_weight = #âm / #dương`
  ([7_losses.py:23-28](../src/7_losses.py#L23-L28)) thay cho cross-entropy
  nhiều lớp.
- `λ = 0.1` ([8_train.py:55](../src/8_train.py#L55)). Bài báo không công bố giá
  trị `λ`.
- BCE lấy trung bình trên N node train; `L_CL` lấy trung bình trên 1024 anchor.

### 2.12 Thiết lập thí nghiệm (§4.1, trang 1927)

| | HSL | Code |
|---|---|---|
| Thiết lập | transductive, split ngẫu nhiên 0.5/0.25/0.25, 20 lần | inductive: val/test dựng local graph chỉ nối với train ([4_hypergraph.py:263-305](../src/4_hypergraph.py#L263-L305)); split chính thức; 5 seed (`config.SEEDS`) |
| Số layer HGNN | 1 | 2 |
| Hidden | tune trong {64, 128, 256, 512} | 128 |
| Optimizer | Adam, lr tune trong {0.001, 0.0001} | Adam, lr 1e-4, weight decay 5e-4 |
| `p_add` | tune trong {0, 0.01, 0.02, 0.05} | `add_per_edge = 2` |
| Chọn mô hình | validation | validation AUC, early stopping ([8_train.py:206-227](../src/8_train.py#L206-L227)) |

### 2.13 Pruning rate `r^v` (§4.2, trang 1927, cột phải; Table 1)

**Bài báo:** `r^v` là tỉ số giữa số ô khác 0 của `Ĥ` và của `H`:

$$
r^v = \frac{\text{nnz}(\hat H)}{\text{nnz}(H)}
$$

**Code:** `summary` ([6_hsl.py:151-161](../src/6_hsl.py#L151-L161)) log
`kept_course`, `kept_object`, `kept_behavioral` (tỉ lệ membership `H0` được
giữ, theo family, bỏ self-loop) và `added` (số membership ΔH được giữ).

**Cần xem:** code không log đúng `r^v`. Có thể tính lại từ log:

$$
r^v = \frac{\sum_f \text{kept}_f \cdot |H_0^{(f)}| + \text{added}}{|H_0|}
$$

Cần quyết định có đếm self-loop trong `nnz` hay không; bài báo không nói rõ.

### 2.14 Ablation (§4.2, trang 1928, Fig. 3b)

| Biến thể trong bài báo | Ý nghĩa | Lệnh tương ứng |
|---|---|---|
| Base | HGNN không học cấu trúc | `--no-hsl` |
| +edge | chỉ hyperedge sampling | `--no-node-sampling --add-per-edge 0` |
| +node | chỉ incident node sampling (có ΔH) | `--no-edge-sampling` |
| −CL | HSL bỏ contrastive | `--lambda-cl 0` |
| HSL | đầy đủ | (mặc định) |

**Cần xem:** trong bài báo, ΔH thuộc bước incident node sampling (§3.4), nên
biến thể **+edge** không có ΔH. Nếu chỉ chạy `--no-node-sampling`, code vẫn thêm
ΔH và vì `Mv = 1` nên mọi phần thêm vào đều được giữ
([6_hsl.py:101](../src/6_hsl.py#L101)). Vì vậy phải kèm `--add-per-edge 0` để
đúng nghĩa +edge.

---

## 3. Danh sách rà soát

Các điểm dưới đây không phải lỗi chắc chắn; đó là chỗ code chọn khác hoặc bài
báo bỏ ngỏ, nên cần xác nhận hoặc nêu khi viết.

1. **Hai nhiệt độ khác nhau.** `τ` của Gumbel (Eq. 2, 7) là `temperature=0.4`,
   cố định trong `StructureLearner` ([6_hsl.py:56](../src/6_hsl.py#L56)), không
   có trong `DEFAULT_SETTINGS` và không có cờ CLI. Còn `"temperature": 0.07` trong
   [8_train.py:56](../src/8_train.py#L56) là nhiệt độ của contrastive loss. Tên
   giống nhau dễ gây nhầm khi báo cáo siêu tham số.
2. **Straight-through hay Concrete.** Bài báo dùng mask mềm (Eq. 2, 7), code
   dùng mask cứng 0/1 ở forward. Cần nêu rõ vì nó ảnh hưởng tới cách hiểu
   “số membership bị xoá” trong lúc train.
3. **Bậc không nhận gradient.** `Dv`, `De` tính từ `weights.detach()`
   ([5_model.py:63](../src/5_model.py#L63)). Gradient của mask chỉ đi qua tử số
   `H X`, không qua phần chuẩn hoá.
4. **`h_e` là trung bình thường của `Z0`** ([5_model.py:75-80](../src/5_model.py#L75-L80)),
   không có multi-head weighted cosine (Eq. 4). Có thể nêu như một bản đơn giản
   hoá `n_c = 1`.
5. **Phân phối mẫu âm không đều.** Sampler chọn hyperedge trước rồi mới chọn
   thành viên ([7_losses.py:50-59](../src/7_losses.py#L50-L59)), nên thành viên
   của hyperedge nhỏ (Behavioral, khoảng 11 node) được lấy nhiều hơn thành viên
   của Course hyperedge (hàng nghìn node). Lấy mẫu có hoàn lại nên có thể trùng
   mẫu âm. Đây vẫn là tập con của `T_i`, nhưng không phải lấy đều trên `T_i`.
6. **Ablation +edge** cần `--add-per-edge 0` (mục 2.14).
7. **`r^v`** chưa được log trực tiếp (mục 2.13).
8. **Classifier head.** Code có thêm `ReLU` + `Linear` sau layer HGNN thứ hai
   (mục 1.6); bài báo HGNN thì dùng thẳng output layer 2.
