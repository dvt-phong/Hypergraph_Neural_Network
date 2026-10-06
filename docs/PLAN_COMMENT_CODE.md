# Plan: rà soát comment code

Ngày lập: 2026-10-06. **Đã thực hiện cùng ngày** (Q1: tiếng Anh; Q2, Q3: theo đề xuất). AST của 11 file
trước và sau giống hệt; `py_compile` đạt. Chưa commit.

## 1. Mục tiêu

1. Chỗ nào có công thức thì ghi công thức ngay trên đoạn code tính nó, để đọc code là thấy toán.
2. Mỗi function có một dòng mô tả đơn giản: function này làm gì.
3. **Không đổi logic.** Chỉ sửa comment.

## 2. Phạm vi

| Có sửa | Không sửa |
|---|---|
| `src/0_config.py` … `src/10_summary.py` (11 file, 53 function/class) | `baseline/CFIN/` (code gốc của CFIN + patch), `docs/`, `README.md` |

## 3. Hiện trạng (đã đọc hết 11 file)

- Mọi function đã có comment đầu hàm bằng tiếng Anh, dạng 3 phần: mô tả, `Input:`, `Output:`.
- Công thức đã có ở phần lớn các dòng tính toán, nhưng nằm **ở cuối dòng**, ví dụ:
  `course_day = (event_date - course_start).days   # d = date(event) − date(course start)`
- File đã đủ công thức: `6_hgnn.py` (G, d(v), m_e, Z1, Z_g), `9_train.py` (loss, AUC, F1),
  `3_features.py` (log1p + z-score, age), `2_preprocess.py` (d, ⌊0.8·n⌋), `10_summary.py` (std, Δ AUC).

**Chỗ còn thiếu:**

| File | Thiếu mô tả function | Thiếu công thức |
|---|---|---|
| `3_features.py` | | `set_one_hot`: x[v, start + index(value)] = 1 |
| `4_hypergraph.py` | | `build_hypergraph`: danh sách (node_ids, edge_ids) là H dạng thưa, h(v,e) = 1 |
| `6_hgnn.py` | `HGNNEncoder` (class), `__init__`, `family_weights` (chỉ có Input/Output) | |
| `7_mlp.py` | `MLPEncoder` (class), `forward` (chỉ có Input/Output) | |
| `8_model.py` | `DropoutModel` (class), `__init__` | `forward_targets`: logit = [z_g ‖ z_s]·u + b |
| `9_train.py` | | `train_one_seed`: Adam + weight decay, θ ← Adam(∇L + λ·θ); thứ tự early stopping |
| `10_summary.py` | `summarize` (chưa có comment) | mean = (1/n)·Σ x |

## 4. Quy tắc đề xuất

**R1. Mô tả function: 1 dòng.**

```python
# Lan truyền một lớp HGNN: x' = G·x, tính theo từng node.
def propagate(self, x, graph, w):
```

**R2. Công thức đặt trên đoạn code tính nó.** Nếu một công thức được tính qua nhiều dòng, ghi
công thức đầy đủ ở trên, còn từng dòng giữ ghi chú ngắn ở cuối dòng:

```python
    # x'_v = d(v)^-1/2 · Σ_e h(v,e) · (w_e/δ(e)) · Σ_{u∈e} d(u)^-1/2 · x_u
    node_degree = torch.sparse.mm(graph["H"], w[:, None])          # d(v) = Σ_e h(v,e)·w_e
    node_scale = node_degree.clamp_min(1e-6).rsqrt()                 # d(v)^-1/2
    ...
```

Công thức chỉ nằm trên một dòng thì giữ ở cuối dòng như hiện tại, không lặp lại.

**R3. Không đụng code.** Kiểm tra bằng cách so cây cú pháp (AST) của từng file trước và sau
khi sửa: phải giống hệt, tức là chỉ comment thay đổi.

## 5. Câu hỏi em cần chốt

| # | Câu hỏi | Thầy đề xuất |
|---|---|---|
| Q1 | Viết comment bằng tiếng Việt hay giữ tiếng Anh? | **Giữ tiếng Anh**, vì code sẽ công bố kèm bài báo. Công thức là ký hiệu toán nên đọc thế nào cũng rõ. Nếu em muốn tiếng Việt thì thầy đổi cả dòng mô tả function sang tiếng Việt |
| Q2 | Bỏ hai dòng `Input:` / `Output:` để mỗi function chỉ còn 1 dòng? | **Bỏ**, trừ các hàm tính tensor (`6_hgnn`, `8_model`, `predict`): giữ một dòng ghi shape, ví dụ `[N, D] -> [N, hidden]`, vì shape giúp kiểm tra công thức |
| Q3 | Chuyển mọi công thức ở cuối dòng lên phía trên dòng code? | **Không chuyển hết.** Chỉ thêm công thức tổng ở trên các đoạn nhiều dòng (R2). Chuyển hết sẽ làm diff rất lớn (khoảng 150 dòng) mà không rõ hơn |

## 6. Các bước sau khi em duyệt

1. Sửa comment theo R1–R3 và theo câu trả lời Q1–Q3, lần lượt từ file 0 đến 10.
2. Kiểm tra AST trước/sau cho cả 11 file, rồi chạy `python -m py_compile src/*.py`.
3. Gửi em `git diff` để duyệt. **Không commit, không push** cho đến khi em bảo.
