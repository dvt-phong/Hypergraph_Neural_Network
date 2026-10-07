# Plan PA1: chuyển validation / test sang transductive

Ngày 2026-10-07. Tiếp theo docs/PLAN_VALIDATION_TEST.md. Chỉ là kế hoạch, chưa sửa code.

## 0. Tóm tắt

```
HIỆN TẠI                                   SAU PA1
H0 = chỉ 126,354 nút train                 H = mọi 225,642 enrollment (train + val + test)
train: forward trên H0                     train: forward trên H, loss chỉ trên nút train
val/test: chèn TỪNG target vào H0          val/test: cùng một forward trên H, lấy theo chỉ số
          (forward_targets, bước A/B)
```

PA1 chỉ đổi **giao thức** (đồ thị gồm những nút nào, chấm điểm ra sao). Mô hình giữ nguyên: công thức HGNN, W theo họ hyperedge, self-loop, nhánh MLP, siêu tham số.

## 1. Nhãn nguồn dùng trong plan

| Nhãn | Nghĩa |
|---|---|
| **[HGNN+]** | Lấy từ code chính thức của nhóm HGNN+ (thư viện DHG, ví dụ phân loại đỉnh) hoặc từ định nghĩa transductive ở phụ lục D của LightHGNN (ICLR 2024, cùng nhóm Feng & Gao). Lưu ý: thầy kiểm qua code DHG và LightHGNN, **chưa đọc toàn văn bài TPAMI HGNN+**. |
| **[Tự suy luận]** | Thầy tự quyết vì HGNN+ không nói đến (dữ liệu của họ đã đánh số sẵn, bài toán khác). Mỗi điểm có lý do; em có thể bác. |
| **[Giữ nguyên]** | Đã có trong project, PA1 không đụng tới. |

## 2. Những điểm lấy từ HGNN+

| # | HGNN+ làm gì (code DHG) | Mình áp dụng |
|---|---|---|
| H1 | Hypergraph dựng trên mọi đỉnh: `G = Hypergraph(data["num_vertices"], data["edge_list"])` | H dựng trên cả train + val + test |
| H2 | Train: `outs = net(X, A)`, rồi `outs[train_idx]` mới tính loss | Forward toàn đồ thị, BCE chỉ trên chỉ số train |
| H3 | Infer: `@torch.no_grad()`, `net.eval()`, `outs = net(X, A)`, rồi `outs[idx]` | `predict` = một forward ở chế độ eval, lấy theo chỉ số val hoặc test |
| H4 | Giữ `deepcopy(net.state_dict())` khi val tốt hơn; cuối cùng nạp best state rồi chấm test **một lần** | Giống hệt (code hiện tại đã làm vậy) |
| H5 | Phụ lục D LightHGNN: nút val/test có mặt trong G lúc train, nhãn của chúng hoàn toàn không biết; val dùng để chọn "best model" | Đây là định nghĩa giao thức sẽ ghi trong bài báo |
| H6 | Nhiều nhóm hyperedge (hyperedge groups) trên cùng tập đỉnh | [Giữ nguyên] Course / Object / User / self-loop, học W theo họ |

## 3. Những điểm thầy tự suy luận

| # | Quyết định | Lý do | Rủi ro / em cần biết |
|---|---|---|---|
| S1 | Đánh số nút toàn cục theo thứ tự train → validation → test: `id = offset[split] + id trong split` (offset 0 / 126,354 / 157,943) | Các file split đã đánh số 0..n−1 riêng; ghép theo thứ tự SPLITS là đơn giản nhất. Bước 2, 3 không phải sửa | Không có |
| S2 | Dựng lại cả 3 họ hyperedge trên toàn bộ nút: Course (vẫn 247), Object gồm cả object chỉ có nút val/test dùng, User "any" gồm mọi enrollment của người học ở mọi split. Bộ lọc \|e\| ≥ 2 áp trên toàn bộ | Đây chính là "H trên mọi đỉnh" (H1) áp vào dữ liệu của mình. HGNN+ không quy định cách dựng hyperedge cho MOOC | Trường hợp đặc biệt `{t, u}` (single_user) biến mất, vì giờ nó là một hyperedge User bình thường. Hai enrollment test của cùng một người học giờ nối với nhau. Như vậy vẫn hợp lệ vì không dùng nhãn; SIG-Net và MST-GCN cũng làm vậy |
| S3 | Scaler của X vẫn chỉ fit trên train | DHG không chuẩn hóa (dữ liệu có sẵn đặc trưng). Fit trên train là cách chặt hơn, và bước 3 không phải sửa | Không có |
| S4 | Early stopping vẫn theo **AUC**: val mỗi 5 epoch, patience 40, tối đa 1000 epoch | DHG chạy 200 epoch, val mỗi epoch, đo accuracy (bài toán nhiều lớp, cân bằng). Bài của mình nhị phân và lệch lớp (dropout ≈ 0.76), nên AUC hợp hơn; giữ quy tắc cũ để dễ so | Lệch so với DHG ở con số, không lệch ở nguyên tắc (H4) |
| S5 | Nhãn val/test **không bao giờ lên device**: chỉ tạo tensor nhãn cho phần train; nhãn val/test ở NumPy, chỉ dùng trong `metrics` | Chặn rò rỉ nhãn ngay trong cấu trúc code, không chỉ dựa vào việc nhớ lọc theo chỉ số | Không có |
| S6 | Bỏ `--eval-limit` và `eval_batch_size` | Transductive phải forward toàn đồ thị nên không cắt bớt target được; chạy thử nhanh chỉ cần `--epochs 10` | README và run_tmux.sh có ví dụ dùng `--eval-limit`, phải sửa theo |
| S7 | `hypergraph.npz` lưu thêm `labels [N]` và `split [N]` (0/1/2); bỏ `edge_keys`, `train_users`, `train_labels` | `edge_keys` và `train_users` chỉ phục vụ `load_targets`, giờ không còn ai dùng | Phải chạy lại bước 4 (bước 2, 3 giữ nguyên) |
| S8 | Không chuyển sang phép tích chập HGNN+ (trung bình đỉnh → cạnh → đỉnh). Vẫn dùng công thức HGNN đối xứng `D_v^-1/2 H W D_e^-1 Hᵀ D_v^-1/2` | PA1 chỉ đổi giao thức. Đổi cả mô hình thì không biết kết quả thay đổi do đâu | Trong bài báo ghi "follow the transductive protocol of HGNN+", không ghi "use HGNN+" |
| S9 | Trước khi chạy lại, đổi tên `outputs/results.csv` cũ thành `results_old_inductive.csv` (làm tay trên server) | Không để số cũ (inductive) và số mới (transductive) lẫn trong cùng một file và một bảng tổng hợp | Không có |
| S10 | Ước lượng chi phí: số membership từ 1,78 triệu lên khoảng 3,2 triệu (×1,8 theo số nút). Mỗi epoch trên CPU từ ~4 s lên ~7 s; mỗi lần chấm val tốn một forward (~2 s) | Ước lượng tuyến tính theo số nút, chưa đo | Con số thật sẽ đo ở bước kiểm tra K4 |

## 4. Sửa từng file

### 4_hypergraph.py
- Đọc cả 3 split bằng `read_split` (hàm giữ nguyên), ghép nút và dịch khóa của `objects` theo offset (S1):
  ```
  for split_id, name in enumerate(SPLITS):
      split_nodes, split_objects = read_split(name.csv)
      offset = len(nodes)
      objects[offset + local_id] = split_objects[local_id]   # mỗi local_id
      nodes += split_nodes;  split += [split_id] * len(split_nodes)
  ```
- `course_hyperedges`, `object_hyperedges`, `user_hyperedges`: giữ nguyên, chỉ nhận danh sách nút toàn cục (H1, S2).
- Lưu: `node_ids, edge_ids, edge_family, labels, split` (S7). Log số nút mỗi split và số hyperedge mỗi họ.
- Đổi tiêu đề/comment "train hypergraph H0" thành "hypergraph H over all enrollments".

### 5_graph_data.py
- `load_train_graph` + `load_targets` → một hàm `load_graph`: đọc npz, ghép X của 3 split theo thứ tự SPLITS (S1), kiểm số dòng bằng N (H1).
- `apply_scenario`: giữ phần lọc họ hyperedge, đánh số lại, thêm self-loop, chọn cột; bỏ cả vòng val/test (h0_ptr, h0_rows, single_user). Trả thêm `train_index`, `validation_index`, `test_index` = `np.flatnonzero(split == k)` (H2, H3).
- `add_self_loops`: giữ nguyên (giờ self-loop cho cả N nút).

### 6_hgnn.py
- Bỏ `cache_train_states`, `forward_targets` và các hằng `USER`, `SELF_LOOP` chỉ chúng dùng.
- `prepare_graph`, `family_weights`, `propagate`, `forward`: giữ nguyên (S8).

### 8_model.py
- Bỏ `cache_train_states`, `forward_targets`. Sửa comment của `forward`: "logits of every node (train + validation + test)".

### 9_train.py
- `predict(model, x, graph, index)` (H3):
  ```
  model.eval(); with torch.no_grad(): logits = model(x, graph)
  p = sigmoid(logits[index])          # p = σ(logit) of the chosen split
  ```
- Vòng train (H2, S5):
  ```
  logits = model(x, graph)                                   # every node
  loss = BCE(logits[train_index], train_labels)              # train labels only
  ```
- Early stopping, best state, chấm val + test một lần: giữ nguyên (H4, S4).
- `train_auc` trong log dùng `logits[train_index]`.
- `main`: gọi `load_graph`; bỏ `--eval-limit` (S6); log số nút train / val / test.

### 0_config.py
- Bỏ `eval_batch_size` (S6).

### README.md, scripts/run_tmux.sh, docs/KICH_BAN_THUC_NGHIEM.md
- Bỏ `--eval-limit` khỏi lệnh ví dụ, thay bằng `--scenario all --seeds 1 --epochs 10`.
- Một dòng mô tả giao thức: "transductive như HGNN+ (H trên mọi enrollment, loss chỉ trên train, chọn theo val AUC)".

### Không sửa
1_download, 2_preprocess, 3_features, 7_mlp, 10_summary, danh sách kịch bản, siêu tham số.

## 5. Kiểm tra sau khi code (máy local, CPU)

Các lệnh kiểm tra chạy trong scratchpad, không thêm script vào repo.

| # | Kiểm tra | Phải đạt | Nguồn |
|---|---|---|---|
| K1 | Chạy bước 4 | N = 225,642 = 126,354 + 31,589 + 67,699; Course đúng 247 hyperedge, tổng thành viên Course = N; không hyperedge nào có \|e\| < 2 | Tự suy luận |
| K2 | Lấy H mới, chỉ giữ nút train, bỏ hyperedge còn < 2 thành viên | Trùng khớp H0 cũ (cùng tập thành viên cho mỗi Course / Object / User). Chứng minh H mới = H0 cũ + nút val/test | Tự suy luận |
| K3 | Chạy M, seed 1, 10 epoch, 2 lần: nhãn thật, và nhãn val/test đặt hết bằng 0 | Loss train từng epoch và trọng số cuối giống hệt nhau. Chứng minh nhãn val/test không vào train (H5, S5) | Tự suy luận |
| K4 | `--scenario all --seeds 1 --epochs 10` | Đủ 11 dòng trong results.csv; ghi lại thời gian mỗi epoch (kiểm S10) | Giữ nguyên |

## 6. Chạy lại trên server

1. `git pull`, chạy `python src/4_hypergraph.py` (bước 2, 3 không cần chạy lại).
2. Đổi tên `outputs/results.csv` thành `outputs/results_old_inductive.csv` (S9).
3. `bash scripts/run_tmux.sh --scenario all --seeds 1 11 111 1111 11111`.
4. `python src/10_summary.py`.

## 7. Không làm trong PA1

- PA2 (production setting như LightHGNN).
- Đổi phép tích chập sang HGNN+ (S8).
- Đổi cách chia train / val / test, chia lại val theo từng seed.
- Thêm kiểm định thống kê, biểu đồ, checkpoint hay script mới.

## Tài liệu

- DHG, ví dụ phân loại đỉnh (code train / infer của nhóm HGNN+): https://deephypergraph.readthedocs.io/en/latest/examples/vertex_cls/hypergraph.html
- HGNN+ (Gao et al., TPAMI 2022): https://www.semanticscholar.org/paper/HGNN%2B:-General-Hypergraph-Neural-Networks-Gao-Feng/4d72c36065a89dc02d54b5be8078864a9685077c
- LightHGNN (Feng et al., ICLR 2024), phụ lục D: https://proceedings.iclr.cc/paper_files/paper/2024/file/7fc0aac0d700873ac520af9b05e3f2b6-Paper-Conference.pdf
