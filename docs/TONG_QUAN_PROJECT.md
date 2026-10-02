# Tổng quan project: dự đoán bỏ học XuetangX bằng hypergraph

Cập nhật: 02/10/2026. Tài liệu gồm ba phần: tình hình hiện tại, các luồng chạy (dữ liệu,
mô hình, thực nghiệm), và ý nghĩa của từng file `.py` cùng từng hàm bên trong.

---

## 1. Tình hình hiện tại

### Bài toán

- **Đầu vào:** mỗi lượt đăng ký khoá học (enrollment) là một node. Đặc trưng của node lấy
  từ 35 ngày đầu khoá học.
- **Đầu ra:** xác suất học viên bỏ học (`truth = 1`). Lớp bỏ học chiếm 75,8%.
- **Chia dữ liệu:** giữ nguyên test chính thức (67.699 enrollment). Tập train gốc được
  chia 80/20 thành train (126.354) và validation (31.589), cố định bằng `SPLIT_SEED = 1`.

### Mô hình chính đã được thầy hướng dẫn duyệt

Hypergraph gồm **Course + Object + User + self-loop**, đưa qua HGNN 2 lớp với trọng số W
theo loại hyperedge, ghép với nhánh **MLP(X)** (skip connection). HSL, Behavioral
hyperedge và contrastive loss **không** nằm trong mô hình chính.

| Phiên bản | User hyperedge | Test AUC | Ghi chú |
|---|---|---|---|
| M-any (trước đây là `u1-hgnn`) | `any`: mọi lượt đăng ký của học viên, kể cả khoá học bắt đầu sau | 0,8749 | Có rò rỉ: `check_leakage.py` cho thấy khoảng 46% target test nhận thông tin từ khoá học bắt đầu muộn hơn |
| **M0** | `temporal` + `--causal`: chỉ các khoá học đã bắt đầu | chưa chạy | Kết quả chính, không rò rỉ (đã qua `check_leakage.py`) |

### Tiến độ

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| GĐ1 | Kiểm tra rò rỉ, danh mục kịch bản, sổ Excel tự cập nhật | Code xong, đã chạy thử đầu‑cuối trên CPU, **chưa commit**, chờ em kiểm tra |
| GĐ1 (chạy) | 12 kịch bản × 5 seed trên server | Chưa chạy |
| GĐ2 | Baseline SIG-Net, MST-GCN, CA-TFHN, CFIN, HyperGCN, chạy nguyên bản, mỗi baseline một `.venv` | Code xong (mục 10); HyperGCN, CFIN, CA-TFHN đã chạy thử đầu‑cuối trên CPU; SIG-Net, MST-GCN cần DGL nên chỉ kiểm tra được phần dựng graph, **chưa chạy trên server** |
| GĐ3 | Kế hoạch thực nghiệm trên OULAD | Chưa làm |

---

## 2. Bức tranh toàn cảnh

```text
  data/raw/xuetangx/                     (tải 1 lần)
  prediction_data.tar.gz, user_info.csv, course_info.csv
          │
          │ 1_download.py
          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ 2_preprocess.py   tách train/validation/test, giữ event ngày 0–34 │
  │   → data/processed/simple/{train,validation,test}.csv             │
  └──────────────────────────────────────────────────────────────────┘
          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ 3_features.py     92 đặc trưng/node, chuẩn hoá fit trên train      │
  │   → {train,validation,test}/X.npy, feature_names.csv               │
  └──────────────────────────────────────────────────────────────────┘
          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ 4_hypergraph.py   H0 trên train: Course, Object, Behavioral, User  │
  │   → hypergraph.npz (User "any"), hypergraph_temporal.npz           │
  └──────────────────────────────────────────────────────────────────┘
          ▼                                         (bước 2–4 chạy 1 lần;
  ┌──────────────────────────────────────┐           seed không ảnh hưởng)
  │ 8_train.py  (5_model, 6_hsl, 7_losses)│   9_baselines.py (LR, GBDT)
  │ train → chọn checkpoint trên val →    │
  │ chọn t* trên val → test 1 lần          │
  └──────────────────────────────────────┘
          ▼
  outputs/runs/*.pt                 checkpoint
  outputs/reports/*_train.json      report + lịch sử từng epoch
  outputs/reports/*_{validation,test}_probs.npz   xác suất từng target
          ▼
  scripts/collect_results.py → plot_results.py → summarize_results.py
          ▼
  result/<ngày_giờ>/   (một thư mục cho mỗi lần gọi run_all.sh)
  result/so_thi_nghiem.xlsx   (dựng lại từ mọi thư mục sau mỗi lần chạy)
```

### Đặc trưng X (92 cột, xem `feature_names.csv`)

| Khối | Cột | Nội dung |
|---|---|---|
| Hành vi | 0–34 | Số event mỗi ngày, ngày 0 đến 34 |
| Hành vi | 35–57 | Số lần mỗi loại hành động (23 loại: video, bài tập, forum, trang web) |
| Học viên | 58–72 | Giới tính, học vấn (one-hot, có cột "thiếu" và "khác"), tuổi, cờ thiếu tuổi |
| Khoá học | 73–91 | Lĩnh vực khoá học (17 loại, có cột "thiếu" và "khác") |

Khối hành vi được biến đổi `log1p` rồi chuẩn hoá. Mọi phép biến đổi chỉ fit trên train.

### Các loại hyperedge (`config.EDGE_FAMILIES`)

| Loại | Nối những ai | Số hyperedge (train) | Trong M0 |
|---|---|---|---|
| course | Mọi enrollment của cùng một khoá học | 247 | Có |
| object | Các enrollment đã dùng cùng một video, bài tập hoặc chủ đề forum (trong cùng khoá) | 21.747 | Có |
| user `any` | Mọi enrollment của cùng một học viên | 32.797 | Không (chỉ M-any) |
| user `temporal` | Một enrollment mốc + các enrollment cùng học viên ở khoá bắt đầu không muộn hơn | 66.552 | Có |
| behavioral | Một enrollment + k = 10 enrollment giống nhất về hành vi (kNN cosine) | 126.354 | Không |
| self_loop | Mỗi node một hyperedge chỉ chứa chính nó, thêm khi nạp graph | = số node | Có |

**Quy tắc chống rò rỉ:** node validation/test không bao giờ nằm trong H0. Mỗi target có
một **graph cục bộ** riêng, gồm chính nó và các node train mà nó nối tới.

---

## 3. Luồng của mô hình chính (M0)

```text
 X [N × 92]                         H: Course + Object + User(temporal) + self-loop
    │                                         │
    ├──────────────┐                          │  W_ee = softplus(θ_loại(e))   (--family-weights)
    │              │                          │  causal: v chỉ nhận tin từ hyperedge
    │              │                          │  không có thành viên nào bắt đầu muộn hơn v
    │              ▼                          ▼
    │   ┌────────────────────────────────────────────────────┐
    │   │ HGNN lớp 1: Linear(92→128) → lan truyền → ReLU      │
    │   │             → Dropout 0,5                           │
    │   │ HGNN lớp 2: Linear(128→128) → lan truyền → ReLU     │  → Z  [N × 128]
    │   │ lan truyền = Dv^-½ H W De^-1 Hᵀ Dv^-½ (·)           │
    │   └────────────────────────────────────────────────────┘
    ▼
 MLP(X): Linear(92→128) → ReLU → Dropout → Linear(128→128) → ReLU   → Z_self [N × 128]
    │                                                          (--skip-connection)
    ▼
 logit = Linear([Z ‖ Z_self]) = logit_graph + logit_self + bias
    ▼
 Loss = BCE(logit, y), pos_weight = 1        (HSL tắt nên contrastive = 0)
```

Hàm tương ứng: `HSLModel.forward` → `hyperedge_weights` → `encode` (gọi `causal_receive`,
`hgnn_propagate` hai lần) → `classify`. Tất cả nằm trong `5_model.py`.

### Mô hình gốc (HSL), để đối chiếu

Mô hình gốc khác ở chỗ: sau khi có Z0 = HGNN(X, H0), `StructureLearner` (`6_hsl.py`) học
H\* = Me ⊙ Mv ⊙ (H0 + ΔH) + I, rồi encode lại thành Z\* = HGNN(X, H\*). Loss = BCE +
λ · contrastive(Z0, Z\*). Bật bằng cách bỏ `--no-hsl`.

---

## 4. Luồng một lần train và đánh giá (`8_train.py`)

```text
train(settings, seed)
 │
 ├─ load_train_graph()          X, nhãn, H0 của train (+ self-loop)
 ├─ load_split("validation")    dữ liệu để dựng graph cục bộ cho target val
 ├─ make_model()                HSLModel theo cờ: skip, W, causal, HSL
 │
 ├─ lặp epoch 1 … 1000  (full-batch: mỗi epoch = 1 bước cập nhật trên toàn graph train)
 │    ├─ model(X, H0) → logits
 │    ├─ total_loss() → backward → clip grad 5 → Adam step
 │    └─ mỗi 5 epoch: predict() trên 5.000 target val cố định
 │          ├─ val AUPRC tốt hơn → lưu checkpoint
 │          └─ 60 lần validate liên tiếp không tốt hơn → dừng sớm
 │
 ├─ nạp checkpoint tốt nhất → predict() trên TOÀN BỘ validation
 ├─ best_threshold() → t* (F1 lớp bỏ học cao nhất)
 └─ ghi *_train.json, *_validation_probs.npz; lưu t* vào checkpoint

test(checkpoint)
 ├─ load_split("test")
 ├─ predict() trên toàn bộ test, dùng t* đã lưu (test không ảnh hưởng gì đến lựa chọn)
 ├─ classification_metrics(), branch_metrics() (AUC riêng của nhánh graph và nhánh MLP)
 └─ ghi *_test.json, *_test_probs.npz
```

### `predict()`: chấm điểm target val/test

```text
cho từng lô 8 target:
  build_local_graph(target)    target (node 0) + các node train nó nối tới:
                               course của nó, object nó đã dùng, User (theo luật),
                               Behavioral (nếu có), self-loop
  merge_local_graphs()         ghép các graph cục bộ, không có hyperedge nối hai target
  model(...)                   lấy logit của node 0 mỗi graph → sigmoid → xác suất
```

---

## 5. Luồng chạy thực nghiệm

```text
bash scripts/run_scenarios.sh           (hoặc qua run_tmux.sh để chạy nền)
 │
 ├─ scenarios.py check <mã…>      mỗi kịch bản có đúng file graph và User rule? sai → dừng, chưa train
 ├─ check_leakage.py              nếu có kịch bản dùng hypergraph_temporal.npz; lỗi → dừng
 │
 └─ với mỗi mã kịch bản (M-any, M0, A1…A5, C1, C2, B-LR, B-GBDT, B-HGNN):
      scenarios.py args <mã>  →  script + tham số
      run_all.sh --skip-prep --tag <mã> <tham số>
        ├─ ghi run_info.txt (lệnh, mã kịch bản, git commit, máy, Python, torch, GPU)
        ├─ với mỗi seed: 8_train.py (hoặc 9_baselines.py) --mode both --seeds <seed>
        ├─ collect_results.py     chép report/xác suất/checkpoint vào result/<ngày_giờ>/
        ├─ plot_results.py        history.png, probabilities.png
        └─ summarize_results.py   dựng lại result/so_thi_nghiem.xlsx từ MỌI thư mục result/
```

### Các kịch bản (`scripts/scenarios.py`)

| Mã | Nhóm | Cấu hình | Câu hỏi |
|---|---|---|---|
| M-any | Chính | Course, Object, User `any` + skip + W | Cùng điều kiện với baseline; tái hiện 0,8749 |
| **M0** | Chính | Như trên, User `temporal` + `--causal` | Kết quả chính không rò rỉ |
| A1 / A2 / A3 | Ablation | M0 bỏ User / bỏ Object / bỏ Course | Vai trò từng quan hệ |
| A4 / A5 | Ablation | M0 bỏ W / bỏ skip MLP | Vai trò của W và của MLP |
| C1 | Đối chứng | A1 trên graph xáo trộn | Lợi ích đến từ hàng xóm thật hay từ mô hình lớn hơn |
| C2 | Đối chứng | MLP (chỉ self-loop) | Mốc không dùng graph |
| B-LR / B-GBDT / B-HGNN | Baseline | LR, GBDT trên X; HGNN gốc với User `any` | Mốc tuyến tính, mốc cây, hypergraph cổ điển |
| B-HGNN-T | Baseline | HGNN gốc, User `temporal` + causal | Hypergraph cổ điển dưới cùng luật với M0 |
| B-HyperGCN, B-SIGNet, B-MSTGCN, B-CATFHN, B-CFIN | Baseline | Mô hình đã công bố, User `temporal` | So với M0 |
| B-HyperGCN-any, …, B-CFIN-any | Baseline | Cùng mô hình, User `any` | So với M-any |

Siêu tham số cố định (`DEFAULT_SETTINGS` trong `8_train.py`, cộng thêm `--epochs 1000
--patience 60`): hidden 128, dropout 0,5, lr 1e-3, weight decay 5e-4, lr của W 0,05 (không
weight decay), validate mỗi 5 epoch, chọn checkpoint theo val AUPRC, `pos_weight` 1, 5 seed
(1, 11, 111, 1111, 11111).

### Sổ `result/so_thi_nghiem.xlsx`

| Sheet | Nội dung |
|---|---|
| `Bang_chinh` | Mỗi kịch bản một dòng: chỉ số test mean ± std, Δ AUC và DeLong so với M0 và M-any |
| `Kich_ban` | Danh mục kịch bản, tham số, file graph, số seed đã chạy |
| `Lan_chay` | Mỗi thư mục `result/<ngày_giờ>` một dòng: git, máy, GPU, mọi siêu tham số, val + test |
| `Theo_seed` | Từng seed |
| `DeLong` | DeLong ghép cặp từng seed |
| `Theo_cau_hinh` | Theo cấu hình, gồm cả các lần chạy cũ chưa có mã kịch bản |
| `Giai_thich` | Ý nghĩa các cột |

Dữ liệu gốc là các thư mục `result/<ngày_giờ>/`; sổ Excel chỉ là bản nhìn tổng hợp của
chúng. Muốn dựng lại sổ: `python scripts/summarize_results.py`.

---

## 6. Ý nghĩa từng file và từng hàm: `src/`

Các file đặt tên theo số bước. Vì tên module bắt đầu bằng chữ số, các file nạp nhau bằng
`import_module("4_hypergraph")`.

### `0_config.py`: cấu hình dùng chung

Không có hàm, chỉ có hằng số.

| Nhóm hằng số | Ý nghĩa |
|---|---|
| `ROOT`, `RAW`, `PROCESSED`, `RUNS`, `REPORTS` | Đường dẫn dữ liệu và kết quả |
| `SEEDS` | 5 seed huấn luyện |
| `SPLIT_SEED`, `TRAIN_RATIO`, `OBSERVATION_DAYS` | Chia 80/20 cố định, cửa sổ 35 ngày |
| `ACTION_GROUPS`, `ACTIONS` | 23 loại hành động, chia 4 nhóm |
| `*_FEATURE_START/COUNT` | Vị trí từng khối cột trong X (hành vi, học viên, khoá học) |
| `EDGE_FAMILIES`, `GRAPH_FAMILIES` | Các loại hyperedge; `--families` chọn trong `GRAPH_FAMILIES` |
| `USER_RULES` | `any` hoặc `temporal` |
| `MEMBERSHIP_CHUNK_SIZE` | Xử lý membership theo khối 1 triệu để vừa bộ nhớ GPU |

### `1_download.py`: tải dữ liệu thô

| Hàm | Việc làm |
|---|---|
| `download(raw_dir)` | Tải 3 file thô nếu chưa có. Không kiểm checksum, không giải nén |

### `2_preprocess.py`: tách split và lọc event

| Hàm | Việc làm |
|---|---|
| `read_csv(path)` | Đọc CSV, trả về từng dòng dạng dict |
| `write_csv(path, columns, rows)` | Ghi CSV |
| `load_nodes(path)` | Đọc CSV event, trả về một dòng metadata cho mỗi node, xếp theo `node_id` (đọc `train.csv` mất khoảng 75 giây) |
| `read_prediction_data(path, names)` | Đọc tuần tự các file CSV bên trong `prediction_data.tar.gz` mà không giải nén ra đĩa |
| `split_train_enrollments(ids)` | Xáo trộn enrollment của train gốc bằng `SPLIT_SEED`, chia 80% train, 20% validation |
| `stream_events(...)` | Đọc log, chỉ giữ event có hành động hợp lệ và thuộc ngày 0–34, ghi vào 3 CSV kèm nhãn, thông tin học viên và khoá học |
| `preprocess(raw_dir, output_dir)` | Hàm chính: đọc nhãn và metadata, chia split, gọi `stream_events`. Enrollment không có event trong 35 ngày vẫn có một dòng, nên không mất node |

### `3_features.py`: tạo ma trận X

| Hàm | Việc làm |
|---|---|
| `feature_columns(feature_set)` | Chọn cột theo bộ đặc trưng: `behavior`, `behavior_user`, `behavior_course`, `full` |
| `feature_metadata()` | Tên và nguồn của 92 cột, ghi ra `feature_names.csv` |
| `set_one_hot(...)` | Bật cột one-hot của một giá trị phân loại; giá trị thiếu và lạ có cột riêng |
| `numeric_statistics(train_values)` | Fit median, mean, std của một cột số **trên train** |
| `scale_numeric(raw, stats)` | Điền giá trị thiếu bằng median rồi chuẩn hoá |
| `build_split_features(path)` | Từ CSV một split: đếm event theo ngày và theo hành động, lấy thông tin học viên và khoá học |
| `build_features(output_dir)` | Hàm chính: dựng 3 split, fit biến đổi trên train, áp dụng cho cả 3, ghi `X.npy` |

### `4_hypergraph.py`: dựng và nạp hypergraph

Hypergraph lưu dạng hai mảng song song: `node_ids[m]` là thành viên của hyperedge
`edge_ids[m]`, tức các ô khác 0 của ma trận liên thuộc H.

**Dựng H0 (chạy một lần):**

| Hàm | Việc làm |
|---|---|
| `log(message)` | In log có giờ |
| `resolve_device(name)` | `auto` → GPU nếu có, không thì CPU |
| `nearest_train_neighbors(...)` | kNN cosine chính xác trên cột hành vi, tính theo lô trên GPU (dùng cho Behavioral) |
| `read_object_events(path)` | Trả về cặp (node, khoá object) cho mỗi event video, bài tập, forum; khoá có cả mã khoá học |
| `user_hyperedges(train_nodes, rule)` | Tạo User hyperedge theo luật `any` hoặc `temporal` |
| `build_train_hyperedges(...)` | Gom node train thành hyperedge Course, Object, Behavioral, User |
| `build_hypergraph(...)` | Hàm chính: tính kNN (hoặc dùng lại bằng `--reuse-neighbors`), dựng H0, ghi bundle `.npz` kèm `user_rule` |

**Nạp graph khi train và đánh giá:**

| Hàm | Việc làm |
|---|---|
| `start_day(s)` / `start_days(nodes)` | Ngày bắt đầu khoá học → số nguyên, dùng cho `--causal` |
| `node_permutation(count, seed)` | Hoán vị ngẫu nhiên node cho graph xáo trộn (`--shuffle-graph`, kịch bản C1) |
| `add_self_loops(graph)` | Thêm mỗi node một self-loop |
| `family_ids(families)` | Tên loại → chỉ số loại |
| `select_families(graph, families)` | Chỉ giữ các loại hyperedge được chọn, đánh số lại. Không còn loại nào thì chỉ còn self-loop, tức mô hình thành MLP |
| `load_train_graph(...)` | X, nhãn và H0 của train (đã chọn loại, đã có self-loop, `node_start`) |
| `load_evaluation_split(...)` | Mọi thứ cần để dựng graph cục bộ cho target val/test: thành viên theo khoá Course/Object, enrollment train của từng học viên kèm ngày bắt đầu, kNN |
| `build_local_graph(split_data, target)` | Graph cục bộ của một target (node 0) + các node train nó nối tới. Với luật `temporal`, chỉ lấy khoá học bắt đầu không muộn hơn |
| `merge_local_graphs(graphs)` | Ghép nhiều graph cục bộ thành một lô; không có hyperedge nối hai target |

### `5_model.py`: HGNN encoder và mô hình

| Hàm / lớp | Việc làm |
|---|---|
| `graph_to_device(graph, device)` | Chuyển graph NumPy sang tensor PyTorch |
| `weighted_sum(...)`, `_weighted_sum_chunk(...)` | Phép cộng có trọng số theo membership, theo cả hai chiều (node → hyperedge và ngược lại). Chạy theo khối, có checkpoint để tiết kiệm bộ nhớ |
| `causal_receive(graph, weights)` | Mask 0/1 cho từng membership: node chỉ nhận tin từ hyperedge không có thành viên nào bắt đầu muộn hơn nó |
| `hgnn_propagate(x, graph, weights, edge_weight, receive)` | Một bước lan truyền HGNN: Dv^-½ H W De^-1 Hᵀ Dv^-½ x |
| `hyperedge_means(z, graph)` | Biểu diễn hyperedge = trung bình embedding các thành viên (dùng cho HSL) |
| `hyperedge_descriptors(x, graph)` | Đầu vào của bộ chấm α_e: [trung bình X ‖ one-hot loại ‖ log kích thước] |
| `HSLModel.__init__` | Tạo 2 lớp HGNN, cùng các phần tuỳ chọn: HSL, nhánh MLP, W theo loại, α_e theo từng hyperedge, causal |
| `HSLModel.edge_alpha` | α_e ∈ (0, 2) cho mỗi hyperedge (`--edge-weights`); self-loop giữ α = 1 |
| `HSLModel.hyperedge_weights` | Đường chéo của W: w_loại(e) × α_e |
| `HSLModel.encode` | 2 lớp HGNN: Linear → lan truyền → ReLU → Dropout → Linear → lan truyền → ReLU |
| `HSLModel.classify` | Logit từ Z, hoặc từ [Z ‖ MLP(X)] khi có skip; trả thêm `logit_graph` và `logit_self` |
| `HSLModel.weight_summary` | Giá trị w_loại và trung bình α theo loại, để ghi log |
| `HSLModel.forward` | Chạy cả luồng: W → Z0. Nếu có HSL thì thêm H\* → Z\* → logit |

### `6_hsl.py`: học cấu trúc (HSL, không dùng trong M0)

| Hàm / lớp | Việc làm |
|---|---|
| `keep_mask(logits, τ, training)` | Mẫu 0/1 giữ/bỏ bằng Gumbel straight-through khi train; ngưỡng 0,5 khi đánh giá |
| `StructureLearner.__init__` | Bộ chấm Me (giữ hyperedge) và Mv (giữ membership) |
| `StructureLearner.forward` | H0 + ΔH, rồi áp Me và Mv; self-loop không bao giờ bị bỏ |
| `StructureLearner.implicit_connections` | ΔH: thêm vào mỗi Behavioral hyperedge các ứng viên có Z0 giống hyperedge nhất |
| `StructureLearner.membership_logits`, `_membership_chunk` | Logit Mv cho mọi membership, tính theo khối |
| `StructureLearner.summary` | Tỉ lệ membership được giữ theo loại (`kept_*`) và số ΔH được giữ |

### `7_losses.py`: hàm loss

| Hàm | Việc làm |
|---|---|
| `positive_class_weight(labels, setting)` | `pos_weight` của BCE: một số cố định, hoặc `balanced` = #âm / #dương |
| `build_neighbor_sampler(graph)` | Chỉ mục membership để lấy mẫu hàng xóm nhanh (cho contrastive) |
| `sample_hyperedge_neighbors(...)` | Với mỗi node neo, lấy ngẫu nhiên các node chung hyperedge làm mẫu âm |
| `_contrastive_one_direction(...)`, `contrastive_loss(...)` | Contrastive trong hyperedge giữa Z0 và Z\* (HSL, Eq. 10), tính hai chiều rồi lấy trung bình |
| `total_loss(...)` | BCE + λ · contrastive. M0 có λ = 0 vì HSL tắt |

### `8_train.py`: train, chọn trên validation, test

| Hàm | Việc làm |
|---|---|
| `DEFAULT_SETTINGS` | Mọi siêu tham số mặc định; dòng lệnh ghi đè |
| `log`, `set_seed` | Ghi log, cố định seed (random, NumPy, torch) |
| `best_threshold(labels, probs)` | Ngưỡng t\* cho F1 lớp bỏ học cao nhất |
| `classification_metrics(labels, probs, t)` | AUC, AUPRC, ACC, Precision, Recall, F1, Macro-F1, chỉ số của lớp không bỏ học, F1 tại 0,5 |
| `format_metrics` | Định dạng chỉ số để in |
| `make_run_name(settings, seed)` | Tên run từ cấu hình, ví dụ `hgnn_course-object-user_temporal_causal_skip_fw_full_M0_seed_1` |
| `make_model(input_dim, settings)` | Tạo `HSLModel` theo cờ |
| `load_split(...)` | Gọi `load_evaluation_split` với cấu hình hiện tại |
| `predict(...)` | Chấm điểm target val/test qua graph cục bộ, theo lô 8 target |
| `branch_metrics(labels, parts)` | AUC riêng của `logit_graph` và `logit_self` (mô hình có skip) |
| `save_probabilities(...)` | Ghi `*_probs.npz` (nhãn, xác suất, t\*, phần logit) |
| `save_edge_weights(...)` | Ghi bảng α_e và W_ee của từng hyperedge (`--edge-weights`) |
| `train(settings, seed)` | Vòng train đầy đủ (xem mục 4); ghi `*_train.json` kèm `user_rule` của bundle |
| `upgrade_state_dict(state, model)` | Cho phép nạp checkpoint cũ, từ trước khi có loại User |
| `test(checkpoint)` | Đánh giá test một lần với t\* đã lưu; ghi `*_test.json` |
| `pos_weight_argument`, `families_argument` | Đọc tham số dòng lệnh `--pos-weight`, `--families` |

### `9_baselines.py`: baseline không dùng graph

| Hàm | Việc làm |
|---|---|
| `load_split(dir, split, feature_set)` | X và nhãn của một split |
| `make_model(name, seed)` | `LogisticRegression`, hoặc `HistGradientBoostingClassifier` (GBDT, dừng sớm trên 10% train) |
| `train(settings, seed)` | Fit trên train, chọn t\* trên validation, ghi report cùng định dạng với `8_train.py` (có `--tag`) |
| `test(checkpoint)` | Chấm test với t\* đã lưu |

---

## 7. Ý nghĩa từng file và từng hàm: `scripts/`

### Luồng thực nghiệm hiện tại

| File | Vai trò |
|---|---|
| `run_scenarios.sh` | Chạy các kịch bản của `scenarios.py`; kiểm tra bundle và rò rỉ trước khi train |
| `run_all.sh` | Một cấu hình: (bước 2–4 nếu không `--skip-prep`) → mọi seed → collect → plot → summarize |
| `run_tmux.sh` | Chạy `run_all.sh` hoặc `RUN_SCRIPT` trong tmux, tắt SSH vẫn tiếp tục chạy |

**`scenarios.py`: danh mục kịch bản**

| Hàm | Việc làm |
|---|---|
| `scenario(...)` | Tạo một mục kịch bản: nhóm, mô tả, câu hỏi, tham số, User rule cần |
| `hypergraph_file(spec)` | File bundle mà kịch bản dùng |
| `problems(code)` | Liệt kê lý do kịch bản không chạy được: thiếu file, sai User rule |
| `main(command, codes)` | Dòng lệnh: `list`, `codes`, `args`, `note`, `check` |

**`check_leakage.py`: kiểm tra rò rỉ**

| Hàm | Việc làm |
|---|---|
| `report(...)` | In OK / FAIL / INFO; ghi nhận lỗi |
| `check_splits` | Train, validation, test không trùng enrollment |
| `check_train_user` | Mỗi User hyperedge của H0 chỉ một học viên; với `temporal`, không thành viên nào bắt đầu sau khoá mốc |
| `check_local_graphs` | Graph cục bộ của 2.000 target val/test ngẫu nhiên: không có thành viên bắt đầu muộn hơn target (với `any`, đếm số target bị rò rỉ) |
| `check_causal` | Mask `causal_receive` của mô hình trùng với quy tắc tính lại bằng NumPy |
| `main(...)` | Chạy cả 4 kiểm tra; có lỗi thì thoát với mã 1 |

**`collect_results.py`: gom kết quả một lần chạy**

| Hàm | Việc làm |
|---|---|
| `report_paths(log)` | Tìm đường dẫn report trong log của từng seed |
| `seed_row(entry, run_dir)` | Chép report, xác suất, checkpoint vào thư mục run; lấy chỉ số của một seed |
| `summary_rows(rows, columns)` | Dòng mean và std |
| `main(run_dir)` | Ghi `results.csv` |

**`plot_results.py`: vẽ hình**

| Hàm | Việc làm |
|---|---|
| `load_reports` | Đọc các `*_train.json` |
| `plot_history` | `history.png`: loss, AUC train/val, tỉ lệ giữ của HSL, ngưỡng theo epoch |
| `plot_probabilities` | `probabilities.png`: phân bố xác suất validation của hai lớp, cùng t\* và 0,5 |
| `main(run_dir)` | Vẽ cả hai hình |

**`summarize_results.py`: sổ Excel**

| Hàm | Việc làm |
|---|---|
| `metrics_at`, `split_metrics` | Tính lại mọi chỉ số từ file xác suất (nên các lần chạy cũ cũng có đủ cột) |
| `read_run_info`, `read_durations`, `folder_time` | Đọc `run_info.txt`, `manifest.tsv`, thời điểm chạy |
| `model_name(settings)` | Tên ngắn: LR, GBDT, MLP, HGNN, HSL |
| `seed_rows(run_dir)` | Một dòng cho mỗi seed của một thư mục, gồm mã kịch bản, User rule, máy, GPU |
| `mean_std`, `text` | Tính mean ± std; định dạng giá trị |
| `summarize(group, first_columns)` | Gộp một nhóm seed thành một dòng: siêu tham số và mean ± std |
| `seed_table(rows)` | Sheet `Theo_seed` |
| `table_columns`, `write_csv` | Thứ tự cột và ghi CSV |
| `latest_by(rows, key)` | Lấy lần chạy mới nhất của mỗi (kịch bản, seed); gộp seed từ nhiều thư mục |
| `load_probabilities`, `delong_rows` | DeLong ghép cặp theo seed; báo rõ khi hai tập test lệch nhau |
| `main_table` | Sheet `Bang_chinh`, thứ tự cột cố định |
| `catalog_table` | Sheet `Kich_ban` |
| `write_xlsx`, `save_workbook` | Ghi Excel ra file tạm rồi đổi tên, nên không bao giờ hỏng sổ |
| `write_markdown` | `tong_hop_ket_qua.md` |
| `main(result_dir)` | Dựng mọi sheet và CSV |

**`delong.py`: kiểm định DeLong (dùng riêng hoặc qua `summarize_results.py`)**

| Hàm | Việc làm |
|---|---|
| `midrank` | Hạng trung bình (dùng trong DeLong nhanh của Sun & Xu, 2014) |
| `delong(labels, a, b)` | AUC hai mô hình trên cùng nhãn; z và p hai phía |
| `probability_files`, `load`, `main` | So hai thư mục `result/` theo từng seed |

**`analyze_contribution.py`: graph hay MLP mang dự đoán?**

| Hàm | Việc làm |
|---|---|
| `target_groups(processed)` | Số event của mỗi target test; học viên có lượt đăng ký trước đó hay không |
| `main()` | AUC riêng từng nhánh, và AUC + DeLong theo nhóm (không hoạt động, Q1–Q4, có/không có lịch sử) |

### Script cũ (vẫn chạy được, không nằm trong luồng chính)

| File | Ghi chú |
|---|---|
| `run_p0.sh`, `run_b.sh`, `run_o.sh`, `run_night.sh`, `run_integrity.sh` | Các đợt sàng lọc trước đây (P0, bước B, nhánh O, chạy đêm, integrity). Kết quả của chúng hiện ở sheet `Theo_cau_hinh` |
| `export_excel.py` | Ghi sổ cũ `docs/ket_qua_thi_nghiem.xlsx`; `run_all.sh` không còn gọi |

---

## 8. Bảng tra cờ dòng lệnh của `8_train.py`

| Cờ | Ý nghĩa | Dùng trong |
|---|---|---|
| `--no-hsl` | Tắt HSL, thành HGNN thường | Mọi kịch bản hiện tại |
| `--families a,b` | Loại hyperedge được giữ; `self_loop` = MLP | Mọi kịch bản |
| `--hypergraph F` | Bundle trong `data/processed/simple` | M-any (`hypergraph.npz`), M0 (`hypergraph_temporal.npz`) |
| `--causal` | Không nhận tin từ khoá học bắt đầu muộn hơn | M0, A2–A5 |
| `--skip-connection` | Thêm nhánh MLP(X) | Mọi kịch bản chính, trừ A5 |
| `--family-weights` | W theo loại hyperedge | Mọi kịch bản chính, trừ A4 |
| `--edge-weights` | α_e theo từng hyperedge | Không dùng trong kịch bản hiện tại |
| `--shuffle-graph S` | Xáo thành viên hyperedge | C1 |
| `--tag X` | Hậu tố tên run = mã kịch bản | Mọi kịch bản |
| `--epochs`, `--patience` | 1000, 60 | Mọi kịch bản |
| `--validation-limit`, `--test-limit` | Tập con khi chạy thử nhanh. Tập con test chọn theo seed train | Chỉ dùng khi thử |

---

## 9. Lưu ý

- `data/processed/simple/hypergraph.npz` trên máy Windows của em **chưa có User
  hyperedge**. Bản trên server thì có (luật `any`, do `run_o.sh` thêm vào).
  `hypergraph_temporal.npz` chỉ có trên server.
- README vẫn còn liên kết tới các tài liệu đã xoá ở commit `375ef04` (ví dụ
  `docs/references.md`, `docs/FILES_4_TO_8_GUIDE.md`).
- `6_hsl.py`, phần contrastive của `7_losses.py` và Behavioral hyperedge vẫn được giữ, để
  tái hiện mô hình gốc và các kết quả cũ, nhưng không nằm trong M0.

---

## 10. Baseline đã công bố (GĐ2)

### Nguyên tắc

Mỗi baseline giữ **mô hình** của code gốc (import thẳng từ `baseline/<repo>`, không sửa
file), chỉ viết lại phần **dữ liệu** cho XuetangX và cho giao thức chung. Giao thức chung
nằm ở `src/10_baseline_data.py`, giống hệt luật của M0:

1. Cùng split train / validation / test, cùng 35 ngày sự kiện (đọc từ 3 file CSV của bước 2).
2. Chỉ dùng **nhãn train**. Không target nào thấy nhãn của chính nó.
3. Hàng xóm và các thống kê chỉ lấy từ **enrollment train**, không bao giờ từ target
   val/test khác (giống graph cục bộ của M0).
4. Enrollment khác của cùng học viên theo `--user-rule`: `temporal` = chỉ khoá học bắt đầu
   không muộn hơn (so với M0); `any` = mọi khoá học, như bản công bố (so với M-any).
   Mỗi lần dựng danh sách đều tự kiểm tra lại luật này (dòng log `[protocol]`).
5. Scaler, k-means, thống kê theo khoá học: fit trên train.
6. Chọn epoch theo AUPRC validation, t\* trên toàn bộ validation, test một lần.
   Report cùng định dạng `8_train.py`, nên sổ Excel và DeLong đọc được ngay.

### Từng baseline

| Mã | File | Code gốc (commit) | Môi trường | Khác code gốc ở đâu, vì sao |
|---|---|---|---|---|
| B-HyperGCN | `11_hypergcn.py` | malllabiisc/HyperGCN `de04938` | `.venv` chính | `utils.Laplacian` lặp Python quá chậm (vài giờ mỗi seed) nên viết lại bằng mảng; `--check-laplacian` chứng minh kết quả trùng khớp. Dùng hypergraph của M0 (Course, Object, User) và graph cục bộ; luật temporal có mask causal. Bỏ chuẩn hoá hàng của X vì X đã chuẩn hoá |
| B-SIGNet | `12_signet.py` | Noverse0/SIG-Net `87bc320` | `.venvs/signet` (DGL) | Graph builder gốc chỉ đọc KDD/NAVER nên dựng lại cho XuetangX: 3 cửa sổ 0–11, 12–23, 24–34 ngày; đặc trưng khoá học = số object video/bài tập/forum + số enrollment train. Code gốc gán loại cạnh bằng **nhãn test** của enrollment cùng học viên (rò rỉ nhãn); ở đây chỉ dùng nhãn train đã biết tại thời điểm dự đoán, còn lại dùng quan hệ thứ 11 "chưa biết" |
| B-MSTGCN | `13_mstgcn.py` | wudongze9/MST-GCN `7350a7e` (`src_xuet`) | `.venvs/mstgcn` (DGL) | Code gốc dừng với lỗi DGL "0-in-degree" (enrollment cùng học viên không có cạnh), nên bật `allow_zero_in_degree`. Thống kê object/khoá học chỉ từ train |
| B-CATFHN | `14_catfhn.py` | codeds27/CA-TFHN `f8ef659` | `.venvs/catfhn` (PyG) | `LGB.forward` gốc lỗi (`if i == 1` làm nối với `None`), sửa thành `i == 0`. Luật temporal: node học viên "tại thời điểm khoá học bắt đầu"; graph bạn cùng lớp chỉ cho train gửi sang target; giữ tối đa 64 bạn/ node vì sampler chỉ lấy 8 rồi 4 |
| B-CFIN | `15_cfin.py` | wzfhaha/dropout_prediction `a99db2e` | `.venv` chính | Code gốc là TensorFlow 1 (không cài được với CUDA mới) nên chuyển sang PyTorch từng phép tính, giữ khởi tạo và siêu tham số. Cluster học viên tính lại bằng k-means (file gốc không rõ cách tính) |

Chung cho CFIN và CA-TFHN: mã giới tính gốc so với `"m"/"f"`, dữ liệu là `"male"/"female"`,
nên bản gốc thực chất cho mọi người giá trị 0; ở đây dùng mã đúng ý tác giả.

### Chạy

```bash
bash scripts/setup_baselines.sh                         # một lần: repo gốc + 3 môi trường
ONLY="B-CFIN B-HyperGCN" SEEDS=1 bash scripts/run_scenarios.sh      # thử 1 seed
ONLY="B-SIGNet B-MSTGCN B-CATFHN B-CFIN B-HyperGCN B-HGNN-T" RUN_SCRIPT=scripts/run_scenarios.sh bash scripts/run_tmux.sh
```

`run_scenarios.sh` tự dựng cache sự kiện nếu chưa có và tự chọn Python của `.venvs/<tên>`
cho SIG-Net, MST-GCN, CA-TFHN. Thử nhanh một baseline: thêm `--limit 2000 --epochs 1 --tag smoke`.

