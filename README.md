# XuetangX dropout prediction with HGSL

Project nghiên cứu dự đoán dropout trên XuetangX. Mỗi enrollment là một node;
Course, Object và Behavioral hyperedge tạo `H0`; HGNN tạo `Z0`; Hypergraph
Structure Learning (HSL, IJCAI 2022) học `H* = Me ⊙ Mv ⊙ (H0 + ΔH) + I`; cùng
HGNN tạo `Z*`; classifier sinh dropout logit. Loss là BCE cộng
intra-hyperedge contrastive giữa `Z0` và `Z*`. Các điều chỉnh so với bài báo cho
bài toán MOOC được liệt kê trong [references](docs/references.md#5-hsl).

Code chính gồm một file cấu hình, tám bước pipeline và một file baseline:

| Bước | File | Trách nhiệm |
|---|---|---|
| 0 | `0_config.py` | Quản lý đường dẫn, split, action và chỉ số feature |
| 1 | `1_download.py` | Tải ba raw file nếu chưa tồn tại |
| 2 | `2_preprocess.py` | Ghép metadata, giữ test gốc và tạo ba split CSV |
| 3 | `3_features.py` | Tạo ba ma trận feature, fit transform trên train |
| 4 | `4_hypergraph.py` | Tạo ba loại hyperedge (`H0`) và local graph cho validation/test |
| 5 | `5_model.py` | HGNN propagation và `HSLModel` |
| 6 | `6_hsl.py` | HSL: hyperedge sampling, incident node sampling, ΔH → `H*` |
| 7 | `7_losses.py` | BCE (có tuỳ chọn `pos_weight`) và intra-hyperedge contrastive loss |
| 8 | `8_train.py` | Train, validation, early stopping, chọn ngưỡng t\* và test |
| 9 | `9_baselines.py` | Baseline không graph: Logistic Regression, GBDT |
| 10 | `10_baseline_data.py` | Cache sự kiện và giao thức chung cho các baseline đã công bố (file 11–15) |
| 11–15 | `11_hypergcn.py`, `12_signet.py`, `13_mstgcn.py`, `14_catfhn.py`, `15_cfin.py` | HyperGCN, SIG-Net, MST-GCN, CA-TFHN, CFIN trên cùng split và giao thức |

Xem [dòng chảy dữ liệu và các cột](docs/data-flow-columns.md),
[nguồn tham khảo của code](docs/references.md), và
[sơ đồ mô hình](docs/assets/hypergraph-neural-network-v3.png). Để học sâu riêng
các file 4–8, đọc [hướng dẫn từ `H0` đến huấn luyện](docs/FILES_4_TO_8_GUIDE.md).

## Cài đặt và chạy

Trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/1_download.py
.\.venv\Scripts\python.exe src/2_preprocess.py
.\.venv\Scripts\python.exe src/3_features.py
.\.venv\Scripts\python.exe src/4_hypergraph.py --k 10 --k-max 20 --device auto
# Thêm User hyperedge vào graph đã có mà không tính lại kNN:
#   src/4_hypergraph.py --reuse-neighbors hypergraph_v3.npz [--user-rule temporal --hypergraph-file hypergraph_temporal.npz]
.\.venv\Scripts\python.exe src/8_train.py --seeds 1
.\.venv\Scripts\python.exe src/8_train.py --mode test --checkpoint outputs/runs/hsl_full_seed_1.pt
```

Siêu tham số mặc định nằm trong `DEFAULT_SETTINGS` ở đầu `src/8_train.py`. Chạy
nhanh để kiểm tra: `--epochs 10 --validation-limit 2000`. Chạy đủ 5 seed:
`--seeds 1 11 111 1111 11111 --mode both`.

Ngưỡng phân loại: sau khi train, best checkpoint được chạy trên **toàn bộ**
validation để chọn ngưỡng t\* làm F1 (lớp dropout) cao nhất; t\* được lưu vào
checkpoint và dùng cho test. Report có thêm `macro_f1`, `f1_negative`,
`auprc_negative` (lớp không dropout) và `f1_at_0.5` để so với các lần chạy cũ.
Chạy `--mode test` trên checkpoint cũ (chưa có t\*) sẽ tự chọn t\* trên validation
trước khi đánh giá test.

Các tuỳ chọn thường dùng:

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--pos-weight` | `1` | Trọng số BCE của lớp dropout; `balanced` = #âm / #dương (0,319, như bản cũ) |
| `--select-metric` | `auprc` | Chỉ số validation để chọn checkpoint (`auprc` hoặc `auc`) |
| `--patience` | `20` | Số lần validate liên tiếp không cải thiện thì dừng |
| `--validation-limit` | `5000` | Tập con validation cố định dùng trong lúc train (0 = toàn bộ) |
| `--lr-schedule` | `none` | `multistep`: lr × 0,9 ở epoch 100, như HGNN |
| `--families` | tất cả | Loại hyperedge được giữ, ví dụ `course,object,user`; `self_loop` = không graph |
| `--hypergraph` | `hypergraph.npz` | File graph trong `data/processed/simple`, ví dụ `hypergraph_temporal.npz` |
| `--tag` | rỗng | Hậu tố tên run, tránh ghi đè khi quét tham số, ví dụ `lr3e-3` |
| `--skip-connection` | tắt | Lớp phân loại nhận thêm `MLP(X)` (đặc trưng riêng của node, không lan truyền) |
| `--family-weights` | tắt | Học một trọng số cho mỗi loại hyperedge (ma trận `W` của HGNN), log ở `w_course`, `w_object`, … |
| `--edge-weights` | tắt | Thêm một trọng số `α_e` cho từng hyperedge (MLP nhỏ trên trung bình X, loại, kích thước); log ở `alpha_*`, bảng đầy đủ ở `outputs/reports/<run>_edge_weights.csv` |

Baseline và ablation:

```powershell
.\.venv\Scripts\python.exe src/9_baselines.py --model gbdt --seeds 1              # GBDT, không graph
.\.venv\Scripts\python.exe src/9_baselines.py --model logreg                      # Logistic Regression
.\.venv\Scripts\python.exe src/8_train.py --no-hsl --families self_loop           # MLP (cùng encoder)
.\.venv\Scripts\python.exe src/8_train.py --no-hsl                                # HGNN baseline
.\.venv\Scripts\python.exe src/8_train.py --no-node-sampling --add-per-edge 0     # chỉ hyperedge sampling
.\.venv\Scripts\python.exe src/8_train.py --no-edge-sampling                      # chỉ incident node sampling
.\.venv\Scripts\python.exe src/8_train.py --lambda-cl 0                           # bỏ contrastive
.\.venv\Scripts\python.exe src/8_train.py --families course,object                # bỏ Behavioral
```

## Kịch bản thực nghiệm

Mọi kịch bản của bài nằm trong một file, [scripts/scenarios.py](scripts/scenarios.py).
Mã kịch bản cũng là `--tag` của lần chạy. Siêu tham số không ghi trong kịch bản lấy theo
`DEFAULT_SETTINGS` của `src/8_train.py`.

| Mã | Nhóm | Cấu hình | Câu hỏi |
|---|---|---|---|
| M-any | Chính | HGNN + skip MLP + W; Course, Object, User `any`, self-loop | Cùng điều kiện với baseline (User có cả khoá học tương lai); tái hiện 0,8749 |
| **M0** | Chính | Như M-any, User `temporal` + `--causal` | Kết quả chính không rò rỉ |
| A1 / A2 / A3 | Ablation | M0 bỏ User / bỏ Object / bỏ Course | Vai trò từng quan hệ |
| A4 / A5 | Ablation | M0 bỏ W / bỏ skip MLP | Vai trò của W và của MLP |
| C1 | Đối chứng | A1 trên graph xáo trộn | Lợi ích đến từ hàng xóm thật hay từ mô hình lớn hơn |
| C2 | Đối chứng | MLP (chỉ self-loop) | Mốc không dùng graph |
| B-LR / B-GBDT / B-HGNN | Baseline | LR, GBDT trên X; HGNN gốc với User `any` | Mốc không graph và hypergraph cổ điển |
| B-HGNN-T | Baseline | HGNN gốc, User `temporal` + causal | Hypergraph cổ điển, cùng luật với M0 |
| B-HyperGCN, B-SIGNet, B-MSTGCN, B-CATFHN, B-CFIN | Baseline | Mô hình đã công bố, User `temporal` | So với M0 (không tương lai) |
| B-…-any | Baseline | Cùng mô hình, User `any` | So với M-any (như bản công bố) |

Baseline đã công bố (GĐ2) dùng mô hình trong code gốc (`baseline/<repo>`, đúng commit
ghi trong `scripts/setup_baselines.sh`) và cùng giao thức với mô hình chính: cùng
split, chỉ dùng nhãn train, hàng xóm chỉ là enrollment train, luật User `temporal`/`any`,
scaler fit trên train, chọn epoch theo AUPRC validation, t\* trên validation. Những chỗ
phải khác code gốc (và lý do) ghi ở đầu từng file `src/11`–`15`. Chuẩn bị một lần trên
server:

```bash
bash scripts/setup_baselines.sh                  # clone repo gốc + .venvs/{signet,mstgcn,catfhn}
python src/10_baseline_data.py                   # cache sự kiện (run_scenarios.sh tự chạy nếu thiếu)
python src/11_hypergcn.py --check-laplacian      # HyperGCN: bản vector hoá khớp utils.Laplacian gốc
ONLY="B-CFIN B-HyperGCN" SEEDS=1 bash scripts/run_scenarios.sh
.venvs/signet/bin/python src/12_signet.py --limit 2000 --epochs 1 --tag smoke   # thử nhanh
```

Hai bundle cần có trong `data/processed/simple`: `hypergraph.npz` với User `any` và
`hypergraph_temporal.npz` với User `temporal`. Script chạy kiểm tra điều này trước khi
train, rồi chạy `scripts/check_leakage.py` cho bundle `temporal`:

```bash
python scripts/scenarios.py list                                     # mã, mô tả, câu hỏi
python scripts/check_leakage.py                                      # kiểm tra rò rỉ của M0 (khoảng 4 phút)
python scripts/check_leakage.py --hypergraph hypergraph.npz          # đo lượng thông tin tương lai của M-any
ONLY=M0 SEEDS=1 bash scripts/run_scenarios.sh                        # thử một kịch bản, một seed
RUN_SCRIPT=scripts/run_scenarios.sh bash scripts/run_tmux.sh         # mọi kịch bản, 5 seed, trong tmux
ONLY="M-any M0" RUN_SCRIPT=scripts/run_scenarios.sh bash scripts/run_tmux.sh
```

Mỗi kịch bản là một lần gọi `scripts/run_all.sh`: một thư mục `result/<ngày_giờ>` với
report, file xác suất, checkpoint và `run_info.txt` (lệnh, mã kịch bản, git commit, máy,
Python/torch, GPU). Sau mỗi lần chạy, `scripts/summarize_results.py` dựng lại
**`result/so_thi_nghiem.xlsx`** từ mọi thư mục trong `result/`, nên sổ luôn khớp với dữ
liệu trên đĩa:

| Sheet | Nội dung |
|---|---|
| `Bang_chinh` | Mỗi kịch bản một dòng: chỉ số test mean ± std (lần chạy mới nhất của mỗi seed), Δ AUC và DeLong so với M0 và M-any |
| `Kich_ban` | Danh mục: câu hỏi, script, tham số, file graph, User rule, số seed đã chạy |
| `Lan_chay` | Mỗi thư mục một dòng: ngày, git, máy, GPU, mọi siêu tham số, val và test |
| `Theo_seed` | Từng seed |
| `DeLong` | DeLong ghép cặp từng seed, mỗi kịch bản so với M0 và M-any |
| `Theo_cau_hinh` | Mỗi cấu hình một dòng, gồm cả các lần chạy cũ chưa có mã kịch bản |
| `Giai_thich` | Ý nghĩa các cột |

`scripts/export_excel.py` (sổ cũ `docs/ket_qua_thi_nghiem.xlsx`) không còn được
`run_all.sh` gọi. Các script cũ `run_p0.sh`, `run_b.sh`, `run_o.sh`, `run_night.sh`,
`run_integrity.sh` vẫn chạy được; kết quả của chúng nằm ở sheet `Theo_cau_hinh`.

So sánh AUC test của hai lần chạy bằng kiểm định DeLong, từng seed với cùng seed:

```bash
python scripts/delong.py result/<mô hình chính> result/<baseline 1> result/<baseline 2>
```

Các bước chạy lâu đều in log có timestamp và `flush=True`. Bước 4 báo tiến độ
exact kNN và dựng/lưu `H0`; bước 8 báo từng epoch (loss, train AUC, tỉ lệ
membership HSL giữ lại theo family) và tiến độ validation/test.

`1_download.py` không kiểm checksum và không giải nén archive. Nếu một raw file đã
tồn tại, script bỏ qua file đó. Nếu lần tải trước bị lỗi, hãy xóa file tương ứng
rồi chạy lại.

`2_preprocess.py` giữ nguyên `test_log.csv` làm test cuối cùng. Chỉ các enrollment
trong `train_log.csv` được shuffle một lần và chia 80/20 thành train/validation.
Kết quả là `train.csv`, `validation.csv`, `test.csv`; mỗi dòng là một event đã
ghép enrollment, label, user và course. Enrollment không có event hợp lệ trong
35 ngày đầu vẫn có một dòng đại diện với behavior rỗng, nên node và label không
bị mất. Mỗi split có `node_id` riêng và không cần `source_partition`.

Feature và hypergraph không phụ thuộc seed huấn luyện nên chỉ cần tạo một lần.
Các seed trong `0_config.py` chỉ điều khiển quá trình train mô hình.

## Các artifact được giữ

- Dữ liệu preprocess: `train.csv`, `validation.csv`, `test.csv`.
- Metadata feature: `feature_names.csv`.
- Mỗi thư mục `train/`, `validation/`, `test/`: một ma trận `X.npy`.
- Hypergraph: một bundle `hypergraph.npz` chứa memberships của `H0`, loại/khóa
  hyperedge, train/validation/test neighbors và `k`.
- Experiment: checkpoint và train/test report trong `outputs/`.

Đây là output thực của từng bước, không phải cache thông minh. Project không dùng
hash, fingerprint, timestamp state hay run ID để quyết định artifact có hợp lệ.
Nếu đổi `k`, `k_max` hoặc feature, hãy chạy lại `4_hypergraph.py` để ghi đè bundle.

## Thiết kế thí nghiệm

- Node là enrollment; `truth=1` là dropout.
- Giữ nguyên test chính thức; raw train được chia 80/20 thành train/validation
  bằng `SPLIT_SEED` cố định.
- Các seed `1, 11, 111, 1111, 11111` chỉ dùng để lặp quá trình huấn luyện.
- Normalization và imputation chỉ fit trên train node.
- `test_truth.csv` chỉ được sử dụng khi chạy đánh giá test cuối cùng.
- `H0` có Course, Object và Behavioral hyperedge.
- Behavioral neighbor dùng exact cosine similarity theo batch bằng PyTorch.
- Hyperedge luôn được ghi theo thứ tự Course, Object, rồi Behavioral kNN.
- Mỗi node có thêm một self-loop hyperedge mà HSL không bao giờ xóa.
- HSL giữ/bỏ hyperedge (`Me`) và membership (`Mv`) bằng Gumbel straight-through
  khi train, ngưỡng 0.5 khi validation/test; ΔH chỉ thêm node vào Behavioral
  hyperedge từ neighbor `k..k_max-1`.
- Validation/test target chỉ nối tới train reference node.
- Validation AUPRC chọn checkpoint; ngưỡng t\* chọn trên toàn bộ validation;
  test chỉ chạy sau khi checkpoint và t\* đã được chọn.
- `--no-hsl` chạy HGNN baseline với cùng encoder; thêm `--families self_loop`
  thành MLP.

## Kiểm tra cú pháp

```powershell
.\.venv\Scripts\python.exe -m py_compile src/*.py
```
