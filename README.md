# XuetangX dropout prediction with a hypergraph neural network

Project nghiên cứu dự đoán dropout trên XuetangX. Mỗi enrollment là một node. Các
enrollment cùng khoá học (Course), cùng dùng một video/bài tập/forum (Object) và của
cùng một học viên (User) được nối bằng hyperedge. Mô hình chính gồm hai nhánh:

- **Nhánh graph:** HGNN 2 lớp (Feng et al., 2019). Mỗi lớp lan truyền
  node → hyperedge → node, với một trọng số học được cho mỗi loại hyperedge (`W`).
- **Nhánh MLP (skip):** chỉ dùng đặc trưng riêng của node.

Lớp phân loại nhận `[z_graph ‖ z_self]` và cho ra dropout logit. Loss là BCE.
Sơ đồ mô hình: [docs/assets/hypergraph-neural-network-v4.png](docs/assets/hypergraph-neural-network-v4.png).

| Bước | File | Trách nhiệm |
|---|---|---|
| 0 | `0_config.py` | Đường dẫn, seed, bố cục cột feature, các loại hyperedge |
| 1 | `1_download.py` | Tải ba raw file nếu chưa có |
| 2 | `2_preprocess.py` | Giữ test gốc, chia train 80/20 thành train/validation, ghi ba split CSV |
| 3 | `3_features.py` | Tạo ma trận feature `X` của từng split, fit transform trên train |
| 4 | `4_hypergraph.py` | Tạo hypergraph train `H0` và local graph cho từng target validation/test |
| 5 | `5_model.py` | `DropoutModel`: HGNN 2 lớp + nhánh MLP + trọng số theo loại hyperedge |
| 6 | `6_train.py` | Train, chọn checkpoint và ngưỡng t\* trên validation, đánh giá test |

Bản có HSL, contrastive loss, trọng số từng hyperedge và các baseline rời được giữ ở
git tag `full-hsl` (`git checkout full-hsl`).

## Cài đặt và chạy

Trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/1_download.py
.\.venv\Scripts\python.exe src/2_preprocess.py
.\.venv\Scripts\python.exe src/3_features.py
.\.venv\Scripts\python.exe src/4_hypergraph.py --k 10 --k-max 20 --device auto
# Bundle User temporal, dùng lại kNN của bundle trên:
.\.venv\Scripts\python.exe src/4_hypergraph.py --user-rule temporal --hypergraph-file hypergraph_temporal.npz --reuse-neighbors hypergraph.npz
# Mô hình chính M0, một seed, train rồi test:
.\.venv\Scripts\python.exe src/6_train.py --mode both --seeds 1 --skip-connection --family-weights --families course,object,user --hypergraph hypergraph_temporal.npz --causal
# Test lại một checkpoint:
.\.venv\Scripts\python.exe src/6_train.py --mode test --checkpoint outputs/runs/<run>.pt
```

Siêu tham số mặc định nằm trong `DEFAULT_SETTINGS` ở đầu `src/6_train.py`. Chạy nhanh
để kiểm tra: `--epochs 10 --validation-limit 2000 --test-limit 2000`.

Ngưỡng phân loại: sau khi train, best checkpoint (theo AUPRC validation) được chạy trên
**toàn bộ** validation để chọn ngưỡng t\* làm F1 (lớp dropout) cao nhất. t\* được lưu vào
checkpoint và dùng cho test. Report có thêm `macro_f1`, `f1_negative`, `auprc_negative`
(lớp không dropout) và `f1_at_0.5`.

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--families` | tất cả | Loại hyperedge được giữ, ví dụ `course,object,user`; `self_loop` = MLP |
| `--hypergraph` | `hypergraph.npz` | Bundle trong `data/processed/simple`, ví dụ `hypergraph_temporal.npz` |
| `--causal` | tắt | Node chỉ nhận từ hyperedge không có thành viên nào bắt đầu khoá học muộn hơn |
| `--skip-connection` | tắt | Thêm nhánh `MLP(X)` |
| `--family-weights` | tắt | Học một trọng số cho mỗi loại hyperedge, log ở `w_course`, `w_object`, … |
| `--shuffle-graph SEED` | tắt | Đối chứng: hyperedge giữ kích thước nhưng thành viên ngẫu nhiên |
| `--patience` | `20` | Số lần validate liên tiếp không cải thiện thì dừng |
| `--validation-limit` | `5000` | Tập con validation cố định dùng trong lúc train (0 = toàn bộ) |
| `--tag` | rỗng | Hậu tố tên run, ví dụ mã kịch bản |

## Kịch bản thực nghiệm

Mọi kịch bản nằm trong [scripts/scenarios.py](scripts/scenarios.py). Mã kịch bản cũng là
`--tag` của lần chạy.

| Mã | Nhóm | Cấu hình | Câu hỏi |
|---|---|---|---|
| M-any | Chính | HGNN + skip MLP + W; Course, Object, User `any` | Cùng điều kiện với baseline có thông tin tương lai |
| **M0** | Chính | Như M-any, User `temporal` + `--causal` | Kết quả chính không rò rỉ |
| A1 / A2 / A3 | Ablation | M0 bỏ User / bỏ Object / bỏ Course | Vai trò từng quan hệ |
| A4 / A5 | Ablation | M0 bỏ W / bỏ skip MLP | Vai trò của W và của MLP |
| C1 | Đối chứng | A1 trên graph xáo trộn | Lợi ích đến từ hàng xóm thật hay từ mô hình lớn hơn |
| C2 | Đối chứng | MLP (chỉ self-loop) | Mốc không dùng graph |
| B-HGNN / B-HGNN-T | Baseline | HGNN gốc, User `any` / `temporal` + causal | Hypergraph cổ điển |

Hai bundle cần có trong `data/processed/simple`: `hypergraph.npz` (User `any`) và
`hypergraph_temporal.npz` (User `temporal`). `run_scenarios.sh` kiểm tra điều này trước
khi train, rồi chạy `scripts/check_leakage.py` cho bundle `temporal`:

```bash
python scripts/scenarios.py list                                     # mã, mô tả, câu hỏi
python scripts/check_leakage.py                                      # kiểm tra rò rỉ của M0 (khoảng 4 phút)
ONLY=M0 SEEDS=1 bash scripts/run_scenarios.sh                        # thử một kịch bản, một seed
RUN_SCRIPT=scripts/run_scenarios.sh bash scripts/run_tmux.sh         # mọi kịch bản, 5 seed, trong tmux
bash scripts/run_integrity.sh                                        # i-mlp, i-co, i-cou, ... + phân tích đóng góp
```

Mỗi kịch bản là một lần gọi `scripts/run_all.sh`, ghi vào một thư mục
`result/<ngày_giờ>` gồm report, file xác suất, checkpoint và `run_info.txt`.
Sau mỗi lần chạy, `scripts/summarize_results.py` dựng lại **`result/so_thi_nghiem.xlsx`**
từ mọi thư mục trong `result/`:

| Sheet | Nội dung |
|---|---|
| `Bang_chinh` | Mỗi kịch bản một dòng: chỉ số test mean ± std, Δ AUC và DeLong so với M0 và M-any |
| `Kich_ban` | Danh mục: câu hỏi, tham số, file graph, User rule, số seed đã chạy |
| `Lan_chay` | Mỗi thư mục một dòng: ngày, git, máy, GPU, mọi siêu tham số, val và test |
| `Theo_seed` | Từng seed |
| `DeLong` | DeLong ghép cặp từng seed, mỗi kịch bản so với M0 và M-any |
| `Theo_cau_hinh` | Mỗi cấu hình một dòng, gồm cả các lần chạy cũ chưa có mã kịch bản |
| `Giai_thich` | Ý nghĩa các cột |

So sánh AUC test của hai lần chạy bằng kiểm định DeLong, từng seed với cùng seed:

```bash
python scripts/delong.py result/<mô hình chính> result/<lần chạy khác>
```

## Dữ liệu và thiết kế thí nghiệm

- Node là enrollment; `truth=1` là dropout. Feature lấy từ 35 ngày đầu của khoá học.
- Giữ nguyên test chính thức; raw train được chia 80/20 thành train/validation bằng
  `SPLIT_SEED` cố định. Enrollment không có event trong 35 ngày đầu vẫn được giữ, với
  behavior bằng 0.
- Normalization và imputation chỉ fit trên train. Các seed `1, 11, 111, 1111, 11111`
  chỉ điều khiển quá trình train.
- Validation/test target chỉ nối tới train node (local graph); nhãn của node khác không
  bao giờ được lan truyền.
- Mỗi node có thêm một self-loop hyperedge.
- Validation AUPRC chọn checkpoint; t\* chọn trên toàn bộ validation; test chỉ chạy sau
  khi checkpoint và t\* đã được chọn.
- Feature và hypergraph không phụ thuộc seed nên chỉ tạo một lần. Nếu đổi `k`, `k_max`
  hoặc feature, hãy chạy lại `4_hypergraph.py`.

## Kiểm tra cú pháp

```powershell
.\.venv\Scripts\python.exe -m py_compile src/*.py
```
