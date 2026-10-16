# XuetangX dropout prediction with a hypergraph neural network

Project nghiên cứu dự đoán dropout trên XuetangX. Mỗi enrollment là một node. Các
enrollment cùng khóa học (Course), cùng dùng một video/bài tập/forum (Object) và của
cùng một người học (User) được nối bằng hyperedge; mỗi node có thêm một self-loop.

Mô hình có hai nhánh, đọc chung bởi một classifier tuyến tính:

- **Nhánh graph** (`6_hgnn.py`): HGNN 2 lớp theo `HGNN_embedding` của Feng et al. (2019),
  với một trọng số học được cho mỗi loại hyperedge (`W`, ý tưởng nhóm hyperedge của HGNN+).
- **Nhánh MLP** (`7_mlp.py`): chỉ đặc trưng riêng của node, không lan truyền.

Sơ đồ: [docs/assets/hypergraph-neural-network-v4.png](docs/assets/hypergraph-neural-network-v4.png).

## Các file

| Bước | File | Việc |
|---|---|---|
| 0 | `src/0_config.py` | Đường dẫn, seed, bố cục cột X59, loại hyperedge, siêu tham số train |
| 1 | `src/1_download.py` | Tải 2 file raw (prediction_data, course_info) |
| 2 | `src/2_preprocess.py` | Giữ sự kiện ngày 0–34 của mọi enrollment → `events.csv`; split của mỗi seed → `splits/s{seed}.csv` |
| 3 | `src/3_features.py` | X59 (chỉ log activity, như MST-GCN: 35 ngày + tổng sự kiện + số object + 22 action), log1p + z-score trên train của mỗi seed → `x59_s{seed}.npz`, file input chung của M và mọi baseline |
| 4 | `src/4_hypergraph.py` | Dựng một hypergraph H (Course, Object, User) trên mọi enrollment → `hypergraph.npz`, chạy 1 lần (không phụ thuộc seed) |
| 5 | `src/5_graph_data.py` | Đọc H và `x59_s{seed}.npz` của một seed, áp kịch bản, trả chỉ số train / val / test |
| 6 | `src/6_hgnn.py` | Nhánh graph: lan truyền HGNN, 2 lớp |
| 7 | `src/7_mlp.py` | Nhánh MLP |
| 8 | `src/8_model.py` | Ghép 2 nhánh: `logit = [z_g ‖ z_s]·u + b` |
| 9 | `src/9_train.py` | Với mỗi seed: chạy các kịch bản, early stopping theo val AUC, chấm val/test ở 0.5, ghi `outputs/results.csv` |
| 10 | `src/10_summary.py` | `results.csv` → `outputs/summary.csv` + `outputs/ket_qua.xlsx` (mean ± std, Δ so với M) |
| | `scripts/run_tmux.sh` | Chạy 9 + 10 trong tmux trên server |

Plan hiện tại: [docs/PLAN_X59.md](docs/PLAN_X59.md). Bản cũ có kịch bản, checkpoint, DeLong và
các script báo cáo nằm ở git tag `before-simple`; bản có HSL và các baseline rời nằm ở tag `full-hsl`.

## Chạy

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/1_download.py
.\.venv\Scripts\python.exe src/2_preprocess.py
.\.venv\Scripts\python.exe src/3_features.py
.\.venv\Scripts\python.exe src/4_hypergraph.py
.\.venv\Scripts\python.exe src/9_train.py --scenario M --seeds 1 11 111 1111 11111
.\.venv\Scripts\python.exe src/10_summary.py
```

Chạy một phần: `--scenario M A4 H`. Chạy thử nhanh: `--scenario all --seeds 1 --epochs 10`.
Baseline (LR, SVM, CFIN, MST-GCN, CA-TFHN) đọc cùng `x59_s{seed}.npz`: lệnh ở [docs/PLAN_X59.md](docs/PLAN_X59.md).
Trên server: `bash scripts/run_tmux.sh --scenario M --seeds 1 11 111 1111 11111` (ablation: `--scenario all`).

Siêu tham số nằm trong `TRAIN`, kịch bản nằm trong `SCENARIOS` của `src/0_config.py`. Dòng lệnh
chỉ đổi được `--scenario`, `--seeds`, `--epochs`, `--eval-every`, `--patience`, `--device`.

## Kịch bản

Mỗi kịch bản chỉ đổi một yếu tố so với mô hình chính M; mọi kịch bản dùng X59.
`--scenario all` chạy 8 kịch bản đầu × 5 seed = 40 lần; X1 chỉ chạy khi gọi tên.

| Mã | Course | Object | User | Self-loop | Layer HGNN | MLP | Học W |
|---|---|---|---|---|---|---|---|
| **M** | ✓ | ✓ | ✓ | ✓ | 2 | ✓ | ✓ |
| A1 | **✗** | ✓ | ✓ | ✓ | 2 | ✓ | ✓ |
| A2 | ✓ | **✗** | ✓ | ✓ | 2 | ✓ | ✓ |
| A3 | ✓ | ✓ | **✗** | ✓ | 2 | ✓ | ✓ |
| A4 | ✓ | ✓ | ✓ | **✗** | 2 | ✓ | ✓ |
| W1 | ✓ | ✓ | ✓ | ✓ | 2 | ✓ | **✗** (W = I cố định) |
| L1 | ✓ | ✓ | ✓ | ✓ | **1** | ✓ | ✓ |
| H | ✓ | ✓ | ✓ | ✓ | 2 | **✗** | ✓ |
| X1 (tùy chọn) | ✗ | ✗ | ✗ | ✓ | 2 | ✓ | ✓ |

## Giao thức

| Mục | Cách làm |
|---|---|
| Dữ liệu | XuetangX (CFIN, Feng et al., 2019); node = enrollment; đặc trưng = 35 ngày đầu của khóa |
| Chia | test chính thức giữ nguyên (67,699); train chính thức → 80% train (126,354) / 20% validation (31,589), xáo bằng seed của lần chạy |
| Tỉ lệ dropout | train 0.758, validation 0.760, test 0.758 (dropout là lớp đa số; AUPRC ngẫu nhiên ≈ 0.758) |
| Đặc trưng X59 | chỉ log activity: 35 số đếm theo ngày, tổng sự kiện, số object và 22 action, `x = (log(1 + c) − μ_train)/σ_train`, μ, σ chỉ tính trên train của seed. Không dùng demographic người học và thông tin khóa học |
| Graph (transductive, như HGNN gốc, SIG-Net, MST-GCN) | một hypergraph H trên mọi enrollment train + val + test |
| Train | full-batch: forward trên toàn H, BCE chỉ trên nút train; Adam, tối đa 1000 epoch, early stopping theo val AUC (mỗi 5 epoch, patience 40); trọng số tốt nhất giữ trong RAM |
| Đánh giá | nạp trọng số tốt nhất, một forward trên H ở chế độ eval, đọc xác suất ở nút val và test (như HGNN+ / DHG), ngưỡng 0.5 |
| Lặp | 5 seed (1, 11, 111, 1111, 11111); seed vừa chia train/val vừa khởi tạo mô hình (như DP-SCL); báo cáo mean ± std. Mọi baseline dùng cùng split và cùng X59 |

So với các bài tham khảo (đọc từ code gốc): SIG-Net và MST-GCN không có validation, lấy
epoch cuối, ngưỡng 0.5; CA-TFHN in kết quả test mỗi epoch.

## Hyperedge

| Loại | Định nghĩa | Số trong H | % node có |
|---|---|---|---|
| Course | `e_C(c) = {v : course(v) = c}` | 247 | 100% |
| Object | `e_O(o) = {v : v dùng video/bài tập/forum o trong ngày 0–34}` | 22,421 | 88.4% |
| User ("any", như SIG-Net, MST-GCN) | `e_U(l) = {v : user(v) = l}`: mọi enrollment của người học (train, val, test) | 64,672 | 94.5% |
| Self-loop | `{v}` | N = 225,642 | 100% |

H gồm 225,642 node (mỗi seed: train 126,354, val 31,589, test 67,699) và 3,222,033 membership (chưa kể
self-loop). Hyperedge có ít hơn 2 thành viên bị bỏ (self-loop đã bao). Hyperedge chỉ dựa trên
khóa học, object và người học.

## Công thức

Ký hiệu: H ∈ {0,1}^{N×E}, `h(v,e) = 1` khi v ∈ e; `w_e = softplus(θ_f(e))`, một θ cho mỗi
loại; `δ(e) = |e|`; `d(v) = Σ_e h(v,e)·w_e`.

```
G     = Dv^-1/2 · H · W · De^-1 · Hᵀ · Dv^-1/2                (Feng et al., 2019, Eq. 10)
Z1    = ReLU( G · (X·Θ1 + b1) )
Z_g   = ReLU( G · (Dropout(Z1)·Θ2 + b2) )                     (HGNN_embedding)
Z_s   = ReLU( Dropout(ReLU(X·A1 + c1))·A2 + c2 )              (MLP)
logit = [Z_g ‖ Z_s]·u + b,   p = σ(logit)
L     = −(1/|V_train|)·Σ_{v∈V_train} [y·log p + (1 − y)·log(1 − p)]
```

G tính trên toàn H (N = mọi enrollment train + val + test). Tổng trong L chỉ chạy trên nút
train; xác suất của nút val/test đọc từ cùng một forward ở chế độ eval.

## Kết quả

`outputs/results.csv`, mỗi (kịch bản, seed) một dòng, ghi ngay khi chạy xong:

| Nhóm | Cột |
|---|---|
| Nhận dạng | time, scenario, seed |
| Kịch bản | families, hgnn_layers, use_mlp, learn_w |
| Siêu tham số | epochs, eval_every, patience, hidden_dim, dropout, learning_rate, weight_decay |
| Dừng | best_epoch, epochs_run |
| Validation | val_auc, val_auprc, val_f1 |
| Test | test_auc, test_auprc, test_accuracy, test_precision, test_recall, test_f1 |
| W học được | w_course, w_object, w_user, w_self_loop (trống nếu kịch bản không dùng loại đó) |
| Thời gian | minutes |

`src/10_summary.py` gom theo kịch bản, theo thứ tự trong `SCENARIOS` (mean ± std; một (kịch bản,
seed) chạy nhiều lần thì lấy dòng mới nhất). Bảng có các cột ✓/✗, `delta_test_auc_vs_M` (AUC test
trung bình của kịch bản − của M) và tỉ lệ `w_f / Σ w` (chỉ tỉ lệ giữa các w có nghĩa: nhân mọi w
với cùng một hằng số thì G không đổi).
