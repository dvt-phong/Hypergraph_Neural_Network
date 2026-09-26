# XuetangX dropout prediction with HGSL

Project nghiên cứu dự đoán dropout trên XuetangX. Mỗi enrollment là một node;
Course, Object và Behavioral hyperedge tạo `H0`; HGNN tạo `Z0`; Hypergraph
Structure Learning (HSL, IJCAI 2022) học `H* = Me ⊙ Mv ⊙ (H0 + ΔH) + I`; cùng
HGNN tạo `Z*`; classifier sinh dropout logit. Loss là weighted BCE cộng
intra-hyperedge contrastive giữa `Z0` và `Z*`. Các điều chỉnh so với bài báo cho
bài toán MOOC được liệt kê trong [references](docs/references.md#5-hsl).

Code chính gồm một file cấu hình và tám bước pipeline:

| Bước | File | Trách nhiệm |
|---|---|---|
| 0 | `0_config.py` | Quản lý đường dẫn, split, action và chỉ số feature |
| 1 | `1_download.py` | Tải ba raw file nếu chưa tồn tại |
| 2 | `2_preprocess.py` | Ghép metadata, giữ test gốc và tạo ba split CSV |
| 3 | `3_features.py` | Tạo ba ma trận feature, fit transform trên train |
| 4 | `4_hypergraph.py` | Tạo ba loại hyperedge (`H0`) và local graph cho validation/test |
| 5 | `5_model.py` | HGNN propagation và `HSLModel` |
| 6 | `6_hsl.py` | HSL: hyperedge sampling, incident node sampling, ΔH → `H*` |
| 7 | `7_losses.py` | Weighted BCE và intra-hyperedge contrastive loss |
| 8 | `8_train.py` | Train, validation, early stopping và test |

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
.\.venv\Scripts\python.exe src/8_train.py --seeds 1
.\.venv\Scripts\python.exe src/8_train.py --mode test --checkpoint outputs/runs/hsl_full_seed_1.pt
```

Siêu tham số mặc định nằm trong `DEFAULT_SETTINGS` ở đầu `src/8_train.py`. Chạy
nhanh để kiểm tra: `--epochs 10 --validation-limit 2000`. Chạy đủ 5 seed:
`--seeds 1 11 111 1111 11111 --mode both`.

Ablation theo Fig. 3 của bài báo HSL:

```powershell
.\.venv\Scripts\python.exe src/8_train.py --no-hsl                                # HGNN baseline
.\.venv\Scripts\python.exe src/8_train.py --no-node-sampling --add-per-edge 0     # chỉ hyperedge sampling
.\.venv\Scripts\python.exe src/8_train.py --no-edge-sampling                      # chỉ incident node sampling
.\.venv\Scripts\python.exe src/8_train.py --lambda-cl 0                           # bỏ contrastive
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
- Validation AUC chọn checkpoint; test chỉ chạy sau khi checkpoint đã được chọn.
- `--no-hsl` chạy HGNN baseline với cùng encoder.

## Kiểm tra cú pháp

```powershell
.\.venv\Scripts\python.exe -m py_compile src/*.py
```
