# Plan A — Transductive (bản 2)

Ngày 2026-10-07. Thay cho docs/PLAN_PA1_TRANSDUCTIVE.md: cùng nội dung code, thêm đối chiếu với HGNN gốc, SIG-Net [4], MST-GCN [5] (đã đọc code gốc trong thư mục baseline/). Chỉ là kế hoạch, chưa sửa code.

## 0. Cách A trong một hình

```
HIỆN TẠI                                    CÁCH A (transductive)
H0 = 126,354 nút train                      H = 225,642 enrollment (train + val + test)
train: forward trên H0                      train: Z = model(X, H), loss = BCE(Z[train], y[train])
val/test: chèn TỪNG target vào H0           val/test: chính graph đó, đọc Z[val], Z[test]
          (forward_targets, bước A/B)
```

Chỉ đổi **giao thức**. Mô hình giữ nguyên: công thức HGNN, W theo họ hyperedge, self-loop, nhánh MLP, siêu tham số, kịch bản.

## 1. Nhãn nguồn

| Nhãn | Nghĩa |
|---|---|
| **[HGNN]** | Code gốc HGNN (Feng et al., AAAI 2019), github.com/iMoonLab/HGNN |
| **[SIG-Net]** | Code gốc SIG-Net [4], baseline/SIG-Net/src |
| **[MST-GCN]** | Code gốc MST-GCN [5], baseline/MST-GCN/MST-GCN/src_xuet |
| **[HGNN+]** | Vòng train/infer của DHG (thư viện chính thức nhóm HGNN+) và phụ lục D của LightHGNN (cùng nhóm). Chưa đọc toàn văn bài TPAMI HGNN+ |
| **[Tự suy luận]** | Thầy tự quyết, không bài nào ở trên quy định; mỗi điểm có lý do |
| **[Giữ nguyên]** | Đã có trong project, không đổi |

## 2. Ba bài em dẫn làm gì (đã kiểm code)

| | HGNN gốc | SIG-Net [4] | MST-GCN [5] |
|---|---|---|---|
| Đồ thị gồm những nút nào | H dựng trên mọi đối tượng train + test (`datasets/visual_data.py`: `construct_H_with_KNN` trên toàn bộ đặc trưng) | Đồ thị toàn cục trên `truth_all` = train + test (`graph_builder.py`, kddcup_graph) | Gộp `train_log` + `test_log`, `train_truth` + `test_truth` rồi dựng một đồ thị (`xuetangx_graph_builder.py:243-254`) |
| Cách chạy | Full-batch: forward toàn H, loss trên `idx_train` | Mỗi target lấy một đồ thị con từ đồ thị toàn cục (`get_subgraph`) | Mỗi target lấy một đồ thị con từ đồ thị toàn cục (`__getitem__`) |
| Đọc kết quả test | Lấy output ở `idx_test` của cùng forward | Chấm các target test trên cùng đồ thị toàn cục | Chấm 20% target còn lại trên cùng đồ thị toàn cục |
| Validation | **Không có**. Pha "val" dùng `idx_test`, giữ model có acc test cao nhất (`train.py`) | **Không có**. Chấm test ở epoch cuối (`trainer.py:113`) | **Không có**. Chấm sau epoch cuối (`trainer.py:240`) |
| Điểm không được chép | Chọn model trên test | Cạnh "student" được gán loại theo **nhãn dropout** của enrollment nguồn, lấy từ `truth_all` gồm cả nhãn test (`graph_builder.py:274-295`), tức rò rỉ nhãn | Chia 80/20 theo thứ tự dòng của train + test gộp, không dùng test chính thức; `StandardScaler.fit_transform` trên toàn bộ (`xuetangx_graph_builder.py:179`) |

Kết luận:
1. Cả ba cùng nguyên tắc của cách A: **một đồ thị chứa cả nút train và test, chỉ nhãn train dùng để học, kết quả test đọc trên chính đồ thị đó.**
2. Cách chạy của mình (full-batch, lọc theo chỉ số) giống **HGNN gốc**. SIG-Net và MST-GCN lấy đồ thị con quanh từng target, nhưng đồ thị gốc vẫn chứa cả train lẫn test.
3. **Không bài nào trong ba bài có validation riêng.** Phần validation và early stopping của mình lấy từ HGNN+/DHG (mục 3).

## 3. Các điểm của cách A và nguồn

### 3.1 Lấy từ tài liệu

| # | Điểm | Nguồn |
|---|---|---|
| R1 | Hypergraph dựng trên mọi enrollment train + val + test | [HGNN], [SIG-Net], [MST-GCN], [HGNN+] |
| R2 | Mỗi epoch forward toàn đồ thị, loss chỉ trên chỉ số train | [HGNN] (`idx_train`), [HGNN+] (`outs[train_idx]`) |
| R3 | Chấm val/test: forward ở chế độ eval, không gradient, rồi lấy theo chỉ số | [HGNN] (`idx_test`), [HGNN+] (`@torch.no_grad`, `net.eval()`, `outs[idx]`) |
| R4 | Tập validation riêng; giữ `deepcopy(state_dict)` khi val tốt hơn; cuối cùng nạp best state, chấm test **một lần** | [HGNN+] (DHG). Không lấy từ HGNN gốc, SIG-Net, MST-GCN vì cả ba đều không có val |
| R5 | Định nghĩa để ghi trong bài: nút val/test có mặt trong đồ thị lúc train, nhãn của chúng hoàn toàn không được dùng | [HGNN+] (LightHGNN, phụ lục D) |
| R6 | Nhiều họ hyperedge trên cùng tập nút, học trọng số theo họ | [HGNN+] (hyperedge groups), [Giữ nguyên] |
| R7 | Test chính thức của XuetangX (67,699), val = 20% ngẫu nhiên của train chính thức | [Giữ nguyên] (audit 2026-10-04). MST-GCN không dùng test chính thức |

### 3.2 Thầy tự suy luận

| # | Quyết định | Lý do | Em cần biết |
|---|---|---|---|
| S1 | Đánh số nút toàn cục theo thứ tự train → validation → test: `id = offset[split] + id trong split` (offset 0 / 126,354 / 157,943) | Các file split đã đánh số 0..n−1 riêng; ghép theo thứ tự SPLITS thì bước 2, 3 không phải sửa | — |
| S2 | Dựng lại cả 3 họ trên toàn bộ nút: Course (vẫn 247), Object gồm cả object chỉ val/test dùng, User "any" gồm mọi enrollment của học viên ở mọi split. Lọc \|e\| ≥ 2 trên toàn bộ | Là R1 áp vào dữ liệu của mình | Trường hợp `{t, u}` (single_user) biến mất. Hai enrollment test của cùng học viên nối với nhau: hợp lệ vì không dùng nhãn (khác SIG-Net, mục 2) |
| S3 | Hyperedge **chỉ dựa trên cấu trúc** (khóa học, object, học viên), không dùng nhãn | Tránh đúng lỗi rò rỉ của SIG-Net | — |
| S4 | Scaler của X vẫn chỉ fit trên train | Chặt hơn MST-GCN (fit trên toàn bộ); bước 3 không phải sửa | — |
| S5 | Early stopping theo **AUC**: val mỗi 5 epoch, patience 40, tối đa 1000 epoch | DHG dùng accuracy, val mỗi epoch, 200 epoch, cho bài nhiều lớp cân bằng. Bài của mình nhị phân và lệch lớp (dropout ≈ 0.76), nên giữ quy tắc cũ | Chỉ lệch về con số, nguyên tắc vẫn là R4 |
| S6 | Nhãn val/test **không đưa lên device**: chỉ tạo tensor nhãn cho phần train; nhãn val/test ở NumPy, chỉ dùng trong `metrics` | Chặn rò rỉ ngay trong cấu trúc code | — |
| S7 | Bỏ `--eval-limit` và `eval_batch_size` | Phải forward toàn đồ thị nên không cắt bớt target được; chạy thử nhanh dùng `--epochs 10` | README, run_tmux.sh, KICH_BAN_THUC_NGHIEM.md có ví dụ dùng `--eval-limit`, sửa theo |
| S8 | `hypergraph.npz` lưu `labels [N]`, `split [N]` (0/1/2); bỏ `edge_keys`, `train_users`, `train_labels` | `edge_keys`, `train_users` chỉ phục vụ `load_targets` | Chạy lại bước 4 (bước 2, 3 giữ nguyên) |
| S9 | Không đổi phép tích chập: vẫn HGNN đối xứng `D_v^-1/2 H W D_e^-1 Hᵀ D_v^-1/2` | Chỉ đổi giao thức, để biết kết quả thay đổi do đâu | — |
| S10 | Đổi tên `outputs/results.csv` cũ thành `results_old_inductive.csv` trước khi chạy (làm tay) | Không trộn số cũ và số mới | — |
| S11 | Ước lượng: membership 1,78 triệu → khoảng 3,2 triệu; mỗi epoch CPU ~4 s → ~7 s; mỗi lần chấm val thêm một forward ~2 s | Tuyến tính theo số nút, chưa đo | Đo thật ở K4 |

## 4. Sửa từng file

### 4_hypergraph.py
- Đọc cả 3 split bằng `read_split` (giữ nguyên hàm), ghép nút, dịch khóa `objects` theo offset (S1):
  ```
  for split_id, name in enumerate(SPLITS):
      split_nodes, split_objects = read_split(name.csv)
      offset = len(nodes)
      objects[offset + local_id] = split_objects[local_id]   # mỗi local_id
      nodes += split_nodes;  split += [split_id] * len(split_nodes)
  ```
- `course_hyperedges`, `object_hyperedges`, `user_hyperedges`: giữ nguyên, nhận danh sách nút toàn cục (R1, S2, S3).
- Lưu `node_ids, edge_ids, edge_family, labels, split` (S8). Log số nút mỗi split, số hyperedge mỗi họ.
- Đổi comment "train hypergraph H0" thành "hypergraph H over all enrollments".

### 5_graph_data.py
- `load_train_graph` + `load_targets` → một hàm `load_graph`: đọc npz, ghép X của 3 split theo thứ tự SPLITS, kiểm số dòng = N.
- `apply_scenario`: giữ phần lọc họ, đánh số lại, thêm self-loop, chọn cột; bỏ vòng val/test (h0_ptr, h0_rows, single_user); trả `train_index`, `validation_index`, `test_index` = `np.flatnonzero(split == k)` (R2, R3).
- `add_self_loops`: giữ nguyên (self-loop cho cả N nút).

### 6_hgnn.py
- Bỏ `cache_train_states`, `forward_targets` và hằng `USER`, `SELF_LOOP` chỉ chúng dùng.
- `prepare_graph`, `family_weights`, `propagate`, `forward`: giữ nguyên (S9).

### 8_model.py
- Bỏ `cache_train_states`, `forward_targets`; comment `forward`: "logits of every node (train + validation + test)".

### 9_train.py
- `predict(model, x, graph, index)` (R3):
  ```
  model.eval(); with torch.no_grad(): logits = model(x, graph)
  p = sigmoid(logits[index])            # p = σ(logit) of the chosen split
  ```
- Vòng train (R2, S6):
  ```
  logits = model(x, graph)                                   # every node
  loss = BCE(logits[train_index], train_labels)              # train labels only
  ```
- Early stopping, best state, chấm val + test một lần: giữ nguyên (R4, S5).
- `train_auc` trong log dùng `logits[train_index]`.
- `main`: gọi `load_graph`, bỏ `--eval-limit` (S7), log số nút train / val / test.

### 0_config.py
- Bỏ `eval_batch_size` (S7).

### README.md, scripts/run_tmux.sh, docs/KICH_BAN_THUC_NGHIEM.md
- Lệnh chạy thử: `--scenario all --seeds 1 --epochs 10`.
- Một dòng giao thức: "transductive: H trên mọi enrollment, loss chỉ trên train, chọn theo val AUC, test chấm một lần".

### Không sửa
1_download, 2_preprocess, 3_features, 7_mlp, 10_summary, kịch bản, siêu tham số.

## 5. Kiểm tra sau khi code (local, CPU; lệnh chạy trong scratchpad, không thêm script vào repo)

| # | Kiểm tra | Phải đạt |
|---|---|---|
| K1 | Chạy bước 4 | N = 225,642 = 126,354 + 31,589 + 67,699; Course đúng 247 hyperedge, tổng thành viên Course = N; không hyperedge nào có \|e\| < 2 |
| K2 | Lấy H mới, chỉ giữ nút train, bỏ hyperedge còn < 2 thành viên | Trùng H0 cũ (cùng thành viên cho mỗi Course / Object / User): H mới = H0 cũ + nút val/test |
| K3 | M, seed 1, 10 epoch, 2 lần: nhãn thật, và nhãn val/test đặt hết bằng 0 | Loss train từng epoch và trọng số cuối giống hệt: nhãn val/test không vào train (R5, S6) |
| K4 | `--scenario all --seeds 1 --epochs 10` | Đủ 11 dòng; ghi thời gian mỗi epoch (S11) |

## 6. Chạy lại trên server

1. `git pull`, `python src/4_hypergraph.py`.
2. Đổi tên `outputs/results.csv` thành `outputs/results_old_inductive.csv` (S10).
3. `bash scripts/run_tmux.sh --scenario all --seeds 1 11 111 1111 11111`.
4. `python src/10_summary.py`.

## 7. Câu gợi ý cho bài báo

> Following HGNN [Feng et al.], SIG-Net [4] and MST-GCN [5], we adopt the transductive setting: a single hypergraph is built over all training, validation and test enrollments, and only training labels supervise the loss. Hyperedges are defined by course, object and learner membership only, never by labels. Unlike these works, we hold out a validation set (20% of the official training set), select the model with the best validation AUC, and score the official test set once.

## 8. Không làm

- PA2 (production setting), đổi phép tích chập, đổi cách chia train / val / test.
- Kiểm định thống kê, biểu đồ, checkpoint, script mới.

## Tài liệu

- HGNN code: https://github.com/iMoonLab/HGNN (`train.py`, `datasets/visual_data.py`)
- SIG-Net code: baseline/SIG-Net/src (`graph_builder.py`, `trainer.py`)
- MST-GCN code: baseline/MST-GCN/MST-GCN/src_xuet (`xuetangx_graph_builder.py`, `trainer.py`)
- DHG, ví dụ phân loại đỉnh: https://deephypergraph.readthedocs.io/en/latest/examples/vertex_cls/hypergraph.html
- LightHGNN (ICLR 2024), phụ lục D: https://proceedings.iclr.cc/paper_files/paper/2024/file/7fc0aac0d700873ac520af9b05e3f2b6-Paper-Conference.pdf
