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
| 0 | `src/0_config.py` | Đường dẫn, seed, bố cục cột X, loại hyperedge, siêu tham số train |
| 1 | `src/1_download.py` | Tải 3 file raw |
| 2 | `src/2_preprocess.py` | Chia train/validation/test, giữ sự kiện ngày 0–34, ghi 3 CSV |
| 3 | `src/3_features.py` | Ma trận đặc trưng X (90 cột) của mỗi split, fit trên train |
| 4 | `src/4_hypergraph.py` | Dựng hypergraph train H0 (Course, Object, User) → `hypergraph.npz`, chạy 1 lần |
| 5 | `src/5_graph_data.py` | Đọc H0 cho train; tìm hyperedge của từng target val/test |
| 6 | `src/6_hgnn.py` | Nhánh graph: lan truyền HGNN, 2 lớp, cách tính cho target mới |
| 7 | `src/7_mlp.py` | Nhánh MLP |
| 8 | `src/8_model.py` | Ghép 2 nhánh: `logit = [z_g ‖ z_s]·u + b` |
| 9 | `src/9_train.py` | Chạy các kịch bản: train, early stopping theo val AUC, chấm val/test ở 0.5, ghi `outputs/results.csv` |
| 10 | `src/10_summary.py` | `results.csv` → `outputs/summary.csv` + `outputs/ket_qua.xlsx` (mean ± std, Δ so với M) |
| | `scripts/run_tmux.sh` | Chạy 9 + 10 trong tmux trên server |

Bản cũ có kịch bản, checkpoint, DeLong và các script báo cáo nằm ở git tag `before-simple`;
bản có HSL và các baseline rời nằm ở tag `full-hsl`.

## Chạy

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/1_download.py
.\.venv\Scripts\python.exe src/2_preprocess.py
.\.venv\Scripts\python.exe src/3_features.py
.\.venv\Scripts\python.exe src/4_hypergraph.py
.\.venv\Scripts\python.exe src/9_train.py --scenario all --seeds 1 11 111 1111 11111
.\.venv\Scripts\python.exe src/10_summary.py
```

Chạy một phần: `--scenario M A4 B1`. Chạy thử nhanh: `--scenario all --seeds 1 --epochs 10 --eval-limit 2000`.
Trên server: `bash scripts/run_tmux.sh --scenario all --seeds 1 11 111 1111 11111`.

Siêu tham số nằm trong `TRAIN`, kịch bản nằm trong `SCENARIOS` của `src/0_config.py`. Dòng lệnh
chỉ đổi được `--scenario`, `--seeds`, `--epochs`, `--eval-every`, `--patience`, `--eval-limit`
(chỉ để thử), `--device`.

## Kịch bản

Mỗi kịch bản chỉ đổi một yếu tố so với mô hình chính M (chi tiết: [docs/KICH_BAN_THUC_NGHIEM.md](docs/KICH_BAN_THUC_NGHIEM.md)).
`--scenario all` chạy 11 kịch bản đầu × 5 seed = 55 lần; X1 chỉ chạy khi gọi tên.

| Mã | Course | Object | User | Self-loop | Feature | Layer HGNN | MLP | Học W |
|---|---|---|---|---|---|---|---|---|
| **M** | ✓ | ✓ | ✓ | ✓ | full | 2 | ✓ | ✓ |
| A1 | **✗** | ✓ | ✓ | ✓ | full | 2 | ✓ | ✓ |
| A2 | ✓ | **✗** | ✓ | ✓ | full | 2 | ✓ | ✓ |
| A3 | ✓ | ✓ | **✗** | ✓ | full | 2 | ✓ | ✓ |
| A4 | ✓ | ✓ | ✓ | **✗** | full | 2 | ✓ | ✓ |
| F1 | ✓ | ✓ | ✓ | ✓ | **feature** (58) | 2 | ✓ | ✓ |
| F2 | ✓ | ✓ | ✓ | ✓ | **feature+user** (71) | 2 | ✓ | ✓ |
| F3 | ✓ | ✓ | ✓ | ✓ | **feature+course** (77) | 2 | ✓ | ✓ |
| L1 | ✓ | ✓ | ✓ | ✓ | full | **1** | ✓ | ✓ |
| B1 | ✓ | ✓ | ✓ | ✓ | full | 2 | **✗** | ✓ |
| W1 | ✓ | ✓ | ✓ | ✓ | full | 2 | ✓ | **✗** (W = I cố định) |
| X1 (tùy chọn) | ✗ | ✗ | ✗ | ✓ | full | 2 | ✓ | ✓ |

## Giao thức

| Mục | Cách làm |
|---|---|
| Dữ liệu | XuetangX (CFIN, Feng et al., 2019); node = enrollment; đặc trưng = 35 ngày đầu của khóa |
| Chia | train chính thức → 80% train / 20% validation (xáo với seed 1); test chính thức giữ nguyên (67,699) |
| Tỉ lệ dropout | train 0.758, validation 0.760, test 0.758 (dropout là lớp đa số; AUPRC ngẫu nhiên ≈ 0.758) |
| Đặc trưng (90 cột) | hành vi: số sự kiện mỗi ngày (35) và mỗi action (23), `x = (log(1 + c) − μ_train)/σ_train`; người học: one-hot giới tính (4) và học vấn (9); khóa học: one-hot lĩnh vực (19). Mỗi one-hot có thêm cột "missing" và "other", nên không phải điền giá trị. Không dùng tuổi (71.5% thiếu năm sinh) |
| Graph | H0 chỉ gồm enrollment train; target val/test tham gia các hyperedge của nó và chỉ đọc từ node train |
| Train | full-batch, BCE, Adam, tối đa 1000 epoch, early stopping theo val AUC (mỗi 5 epoch, patience 40); trọng số tốt nhất giữ trong RAM |
| Đánh giá | val và test một lần với trọng số tốt nhất, ngưỡng 0.5 |
| Lặp | 5 seed (1, 11, 111, 1111, 11111), báo cáo mean ± std |

So với các bài tham khảo (đọc từ code gốc): SIG-Net và MST-GCN không có validation, lấy
epoch cuối, ngưỡng 0.5; CA-TFHN in kết quả test mỗi epoch.

## Hyperedge

| Loại | Định nghĩa | Số trong H0 | % node train có |
|---|---|---|---|
| Course | `e_C(c) = {v : course(v) = c}` | 247 | 100% |
| Object | `e_O(o) = {v : v dùng video/bài tập/forum o trong ngày 0–34}` | 21,747 | 88.4% |
| User ("any", như SIG-Net, MST-GCN) | `e_U(l) = {v : user(v) = l}`: mọi enrollment train của người học | 32,797 | 75.9% |
| Self-loop | `{v}` | N | 100% |

Hyperedge có ít hơn 2 thành viên bị bỏ (self-loop đã bao). Với target t (val/test): Course,
các Object có trong H0, User = {t} ∪ mọi enrollment train của người học đó, self-loop {t}.
Không có node val/test nào trong H0, không dùng nhãn của node khác, các target không nối với nhau.

## Công thức

Ký hiệu: H ∈ {0,1}^{N×E}, `h(v,e) = 1` khi v ∈ e; `w_e = softplus(θ_f(e))`, một θ cho mỗi
loại; `δ(e) = |e|`; `d(v) = Σ_e h(v,e)·w_e`.

```
G     = Dv^-1/2 · H · W · De^-1 · Hᵀ · Dv^-1/2                (Feng et al., 2019, Eq. 10)
Z1    = ReLU( G · (X·Θ1 + b1) )
Z_g   = ReLU( G · (Dropout(Z1)·Θ2 + b2) )                     (HGNN_embedding)
Z_s   = ReLU( Dropout(ReLU(X·A1 + c1))·A2 + c2 )              (MLP)
logit = [Z_g ‖ Z_s]·u + b,   p = σ(logit)
L     = −(1/N)·Σ [y·log p + (1 − y)·log(1 − p)]
```

Target t của val/test: các node train giữ trạng thái đã tính trên H0, t được thêm vào các
hyperedge E(t) của nó (nên mỗi hyperedge H0 có δ(e) + 1 thành viên). Với
`S1_e = Σ_{u∈e} d(u)^-1/2·(x_u·Θ1 + b1)` và `S2_e = Σ_{u∈e} d(u)^-1/2·(Z1_u·Θ2 + b2)`:

```
d(t)  = Σ_{e∈E(t)} w_e
z1_t  = ReLU( d(t)^-1/2 · Σ_{e∈E(t)} w_e/(δ(e)+1) · ( S1_e + d(t)^-1/2·(x_t·Θ1 + b1) ) )
z_g,t = ReLU( d(t)^-1/2 · Σ_{e∈E(t)} w_e/(δ(e)+1) · ( S2_e + d(t)^-1/2·(z1_t·Θ2 + b2) ) )
```

Cho một node train đi qua công thức này (không cộng 1, không cộng số hạng của chính nó) thì
ra đúng forward trên H0.

## Kết quả

`outputs/results.csv`, mỗi (kịch bản, seed) một dòng, ghi ngay khi chạy xong:

| Nhóm | Cột |
|---|---|
| Nhận dạng | time, scenario, seed |
| Kịch bản | families, features, hgnn_layers, use_mlp |
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
