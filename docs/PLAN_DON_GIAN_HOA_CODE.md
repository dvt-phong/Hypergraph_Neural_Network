# Plan: viết lại code theo cách đơn giản nhất

Ngày lập: 2026-10-07. **Chưa thực hiện**, chờ em duyệt (mục 6).

## 1. Mục tiêu

1. Bỏ các cách viết "nâng cao" của Python: for rút gọn (comprehension), for lồng trong một dòng,
   `if ... else` trong một dòng, `**dict`, `lambda`, `defaultdict`, hàm lồng trong hàm, decorator.
   Thay bằng `for` / `if` thường, mỗi dòng làm một việc.
2. Bỏ vài "mẹo" NumPy khó đọc (`cumsum` để đánh số lại, `repeat` + `diff` + `bincount`) bằng vòng `for`.
3. **Không đổi kết quả.** Mọi file đầu ra và mọi con số phải giống hệt trước khi sửa (mục 5).

## 2. Phạm vi

| Có sửa | Không sửa |
|---|---|
| `src/0_config.py`, `2_preprocess.py`, `3_features.py`, `4_hypergraph.py`, `5_graph_data.py`, `6_hgnn.py`, `8_model.py`, `9_train.py`, `10_summary.py` | `1_download.py`, `7_mlp.py` (đã đơn giản), `baseline/`, `docs/` |

Comment em đang sửa dở trong working copy (5–10) được giữ nguyên; chỉ đổi comment ở dòng code bị viết lại.

## 3. Quy tắc

**R1. Comprehension → `for` thường.**
```python
# Trước
ACTIONS = tuple(action for actions in ACTION_GROUPS.values() for action in actions)
# Sau
action_list = []
for group_name in ACTION_GROUPS:
    for action in ACTION_GROUPS[group_name]:
        action_list.append(action)
ACTIONS = tuple(action_list)
```

**R2. `if ... else` một dòng → `if` / `else` nhiều dòng.**
```python
# Trước
self.mlp = MLPEncoder(input_dim, hidden_dim, dropout) if use_mlp else None
# Sau
if use_mlp:
    self.mlp = MLPEncoder(input_dim, hidden_dim, dropout)
else:
    self.mlp = None
```

**R3. `defaultdict` → `dict` thường + kiểm tra khóa** (giống cách `2_preprocess.py` đang viết).
```python
if course_id not in members:
    members[course_id] = []
members[course_id].append(node_id)
```

**R4. `**dict`, `{**a, ...}` → tạo dict rồi gán từng khóa.**
```python
# Trước
settings = {**config.TRAIN, "epochs": arguments.epochs, ...}
# Sau
settings = dict(config.TRAIN)
settings["epochs"] = arguments.epochs
```

**R5. Gọi hàm lồng nhau → mỗi bước một biến.**
```python
# Trước
return self.classifier(self.join(self.hgnn(x, graph), x)).squeeze(-1)
# Sau
z_graph = self.hgnn(x, graph)
z = self.join(z_graph, x)
logits = self.classifier(z)          # [N, 1]
return logits.squeeze(-1)            # [N]
```

**R6. Giữ lại** (đã là một bước rõ ràng, hoặc viết lại sẽ chậm đi nhiều): phép toán NumPy/PyTorch trên
cả mảng (`a[mask]`, `a[index]`, `np.concatenate`, `torch.sparse.mm`, `index_add_`), `import_module`
(bắt buộc vì tên file bắt đầu bằng số), f-string, `enumerate`, `with open(...)`, `nn.Sequential`.
Chỗ nào giữ mà khó hiểu thì thêm một dòng comment có ví dụ nhỏ.

## 4. Danh sách chỗ sửa (đã đọc hết 11 file)

| # | File : dòng | Hiện tại | Viết lại |
|---|---|---|---|
| 1 | `0_config.py:51` | for lồng rút gọn tạo `ACTIONS` | R1 |
| 2 | `0_config.py:111-115` | dict comprehension 2 tầng + `if` tạo `OBJECT_ACTIONS` | 2 vòng `for` + `if family == "web_page": continue` |
| 3 | `0_config.py:131` | `*` (bắt buộc gọi bằng tên) trong `scenario(...)` | bỏ `*` (mọi chỗ gọi đã dùng tên, không đổi gì) |
| 4 | `2_preprocess.py:19-21` | `yield from` | `for row in reader: yield row`, thêm comment "đọc từng dòng, không nạp hết file" (giữ `yield`: train.csv có 22,4 triệu dòng, nạp hết vào list sẽ hết RAM) |
| 5 | `2_preprocess.py:33-39` | `load_nodes` | **xóa** (không file nào gọi) |
| 6 | `2_preprocess.py:50-52` | `binary = ...` rồi `with binary:` và `codecs.getreader("utf-8")(binary)` | `with prediction_data.extractfile(member) as binary:` + `io.TextIOWrapper(binary, encoding="utf-8")` |
| 7 | `2_preprocess.py:81` | dict comprehension `node_ids` | `for` |
| 8 | `2_preprocess.py:90-97` | `with (open(...) as a, open(...) as b, open(...) as c):` (cú pháp Python 3.10) | mở 3 file bằng một vòng `for` vào dict `files`, cuối hàm đóng bằng một vòng `for` |
| 9 | `2_preprocess.py:181` | `sum(len(n) for n in ...)` | `for` cộng dồn |
| 10 | `2_preprocess.py:199-200` | list comprehension + `all(...)` | `for` + biến `all_exist` |
| 11 | `3_features.py:93` | `{value!r}` trong f-string | `'{value}'` |
| 12 | `4_hypergraph.py:34, 51, 60, 71` | `defaultdict(set)`, `defaultdict(list)` | R3 |
| 13 | `4_hypergraph.py:54, 64, 74` | `return [(...) for ... in sorted(...)]` | `for` + `append` |
| 14 | `4_hypergraph.py:92` | lọc `[edge for edge in ... if len(edge[2]) >= 2]` | `for` + `if` |
| 15 | `4_hypergraph.py:95-105` | `np.concatenate` / `np.repeat` trên generator, 4 list comprehension trong `savez_compressed` | một vòng `for` qua các hyperedge, `append` vào 4 list (node_ids, edge_ids, edge_family, edge_keys), rồi `np.asarray`. 1,78 triệu membership: thêm khoảng 1 s |
| 16 | `4_hypergraph.py:107-108` | dict comprehension + `sum(1 for ...)` + `EDGE_FAMILIES[:SELF_LOOP]` | `for` đếm số hyperedge mỗi family |
| 17 | `5_graph_data.py:31` | `*` trong `load_targets` | bỏ `*` |
| 18 | `5_graph_data.py:34-38` | dict comprehension `edge_of`, `defaultdict` | `for`, R3 |
| 19 | `5_graph_data.py:44` | `min(...) if limit else len(nodes)` | R2 |
| 20 | `5_graph_data.py:52` | `extend(... for key in sorted(objects.get(...)) if key in edge_of)` | `for key in sorted(...)` + `if key in edge_of:` + `append` |
| 21 | `5_graph_data.py:53` | `train_by_user.get(..., [])` | `if user_id in train_by_user:` |
| 22 | `5_graph_data.py:63` | list comprehension `labels` | `for` |
| 23 | `5_graph_data.py:75` | `{**graph, ...}` | R4 |
| 24 | `5_graph_data.py:87` | list comprehension + `if` | `for` + `if` |
| 25 | `5_graph_data.py:91` | `np.cumsum(keep_edge) - 1` để đánh số lại hyperedge | vòng `for` qua 54.791 hyperedge: giữ thì gán số mới 0, 1, 2, … |
| 26 | `5_graph_data.py:103-105` | dict lồng dict, biểu thức dài | từng biến một rồi mới gộp vào dict |
| 27 | `5_graph_data.py:109-117` | `np.repeat` + `np.diff` + `np.bincount` + `np.cumsum` để lọc hyperedge của target | 2 vòng `for` lồng (target → các hyperedge của nó), giống cách `load_targets` tạo `h0_ptr`. Đã đo: 0,1 s cho validation (31.589 target, 434.046 dòng) |
| 28 | `5_graph_data.py:118-119` | `if ... else` trong dict | R2 |
| 29 | `5_graph_data.py` (mới) | | trong cùng vòng `for` của #27, lưu thêm `h0_rows` = số thứ tự target của mỗi dòng; `predict` (#41) chỉ cần cắt mảng này |
| 30 | `6_hgnn.py:25` | `num_nodes, num_edges = a, b` | 2 dòng |
| 31 | `6_hgnn.py:45` | `nn.Linear(...) if layers == 2 else None` | R2 |
| 32 | `6_hgnn.py:59, 62, 76, 77, 93` | `w[:, None]` | `w.unsqueeze(1)` + comment `[E] -> [E, 1]` |
| 33 | `6_hgnn.py:76` | chuỗi `.mm(...).clamp_min(...).rsqrt()` | 3 dòng: d(v), chặn dưới, d(v)^-1/2 |
| 34 | `6_hgnn.py:91` | `if ... else` một dòng cho `w_self` | R2 |
| 35 | `6_hgnn.py:98, 104` | `torch.zeros(...).index_add_(...)` viết liền | 2 dòng: tạo mảng 0, rồi `index_add_` |
| 36 | `6_hgnn.py:102-108` | **hàm `layer` lồng trong `forward_targets`** | viết thẳng layer 1 rồi layer 2 trong `forward_targets` (lặp khoảng 6 dòng, nhưng đọc từ trên xuống, không phải nhảy vào hàm con) |
| 37 | `6_hgnn.py` `__init__` | | lưu `self.learn_w = learn_w` (dùng ở #40) |
| 38 | `8_model.py:20-21` | `if ... else` một dòng, `hidden_dim * (2 if use_mlp else 1)` | R2 |
| 39 | `8_model.py:32, 40` | gọi hàm lồng | R5 |
| 40 | `8_model.py:43-48` | decorator `@torch.no_grad()`, `isinstance`, dict comprehension + `zip` + `if else` | `with torch.no_grad():`, `self.hgnn.learn_w`, `for` theo chỉ số family |
| 41 | `9_train.py:72, 79-87` | decorator `@torch.no_grad()`; `np.repeat(np.arange(...), np.diff(...))` | `with torch.no_grad():`; `rows = h0_rows[ptr[start]:ptr[stop]] - start` |
| 42 | `9_train.py:100-101` | dict comprehension + `if else` trong `writerow` | `for` + `if` |
| 43 | `9_train.py:121` | list comprehension `p is not family_logits` | `for` + `if` |
| 44 | `9_train.py:145-146` | `roc_auc_score(..., predict(...))` lồng | 2 dòng |
| 45 | `9_train.py:167-168` | `", ".join(f"..." for ...)` | `for` tạo list chuỗi rồi `join` |
| 46 | `9_train.py:170-182` | dict có 4 lần `**{...}` | tạo `row = {}` rồi gán từng khóa bằng `for` |
| 47 | `9_train.py:187-188` | `if else` một dòng + list comprehension | R2, `for` |
| 48 | `9_train.py:210` | `{**config.TRAIN, ...}` | R4 |
| 49 | `10_summary.py:20-22` | dict comprehension + `sort_values(key=lambda ...)` | thêm cột `order` = thứ tự kịch bản, `sort_values(["order", "seed"])`, xóa cột `order` |
| 50 | `10_summary.py:25` | `.apply(pd.to_numeric, errors="coerce")` | `for` từng cột `w_*` |
| 51 | `10_summary.py:36-40` | `row.update({...})` có `if else` và generator | gán từng khóa, R2 |
| 52 | `10_summary.py:43-44, 49` | gán 2 biến một dòng, `if else` trong f-string | tách dòng, R2 |

Tổng: 9 file, khoảng 50 chỗ. Code dài thêm khoảng 120–150 dòng.

## 5. Kiểm tra (kết quả phải giống hệt)

Trước khi sửa, lưu bản hiện tại vào scratchpad. Sau khi sửa:

1. `py_compile` cả 11 file.
2. `2_preprocess.py` + `3_features.py` chạy vào một thư mục tạm → so mã hash của 3 file CSV, 3 file `X.npy`,
   `feature_names.csv` với bản hiện tại.
3. `4_hypergraph.py` vào thư mục tạm → 6 mảng trong `hypergraph.npz` giống hệt (`np.array_equal`).
4. `load_targets` + `apply_scenario` cho cả 12 kịch bản (M … H, X1): mọi mảng giống hệt code cũ.
5. `9_train.py --scenario M W1 H --seeds 1 --epochs 10 --eval-limit 2000 --device cpu` bằng code cũ và code mới:
   các cột số trong `results.csv` giống hệt (ghi vào file tạm, không đụng `outputs/results.csv`).
6. `10_summary.py` trên cùng một `results.csv`: `summary.csv` giống hệt.

Không commit, không chạy lại thí nghiệm 5 seed.

## 6. Cần em quyết định

- **Q1.** #4: giữ `yield` ở `read_csv` / `read_prediction_data` (đề xuất: giữ, vì dữ liệu quá lớn để nạp hết)?
- **Q2.** #36: hàm `layer` lồng → viết thẳng 2 layer (đề xuất) hay tách thành method riêng của class?
- **Q3.** #5: xóa `load_nodes` không dùng (đề xuất: xóa)?
- **Q4.** Có chỗ nào em thấy khó đọc mà chưa có trong bảng không?
