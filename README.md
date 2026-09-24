# XuetangX dropout prediction with HGSL

Project nghiên cứu dự đoán dropout trên XuetangX. Mỗi enrollment là một node;
Course, Object và Behavioral hyperedge tạo `H0`; HGNN tạo `Z0`; HSL học lại
membership để tạo `H*`; cùng HGNN tạo `Z*`; classifier sinh dropout logit.

Code chính gồm một file cấu hình và tám bước pipeline:

| Bước | File | Trách nhiệm |
|---|---|---|
| 0 | `0_config.py` | Quản lý đường dẫn, split, action và chỉ số feature |
| 1 | `1_download.py` | Tải ba raw file nếu chưa tồn tại |
| 2 | `2_preprocess.py` | Ghép metadata, giữ test gốc và tạo ba split CSV |
| 3 | `3_features.py` | Tạo ba ma trận feature, fit transform trên train |
| 4 | `4_hypergraph.py` | Tạo ba loại hyperedge và sparse `H0` |
| 5 | `5_model.py` | HGNN propagation và `HGSLModel` |
| 6 | `6_hsl.py` | Học membership và tạo `H*` |
| 7 | `7_losses.py` | Weighted BCE và contrastive loss |
| 8 | `8_train.py` | Train, validation, early stopping và test |

Xem [dòng chảy dữ liệu và các cột](docs/data-flow-columns.md),
[nguồn tham khảo của code](docs/references.md), và
[sơ đồ mô hình](docs/assets/hypergraph-neural-network-v3.png).

## Cài đặt và chạy

Trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/1_download.py
.\.venv\Scripts\python.exe src/2_preprocess.py
.\.venv\Scripts\python.exe src/3_features.py
.\.venv\Scripts\python.exe src/4_hypergraph.py --k 10 --k-max 20
.\.venv\Scripts\python.exe src/8_train.py --seed 1 --feature-set full --epochs 50
.\.venv\Scripts\python.exe src/8_train.py --mode test --checkpoint outputs/runs/simple_hgsl_full_seed_1.pt
```

`1_download.py` không kiểm checksum và không giải nén archive. Nếu một raw file đã
tồn tại, script bỏ qua file đó. Nếu lần tải trước bị lỗi, hãy xóa file tương ứng
rồi chạy lại.

`2_preprocess.py` giữ nguyên `test_log.csv` làm test cuối cùng. Chỉ các enrollment
trong `train_log.csv` được shuffle một lần và chia 80/20 thành train/validation.
Kết quả là `train.csv`, `validation.csv`, `test.csv`; mỗi dòng là một event đã
ghép enrollment, label, user và course. Mỗi split có `node_id` riêng và không
cần `source_partition`.

Feature và hypergraph không phụ thuộc seed huấn luyện nên chỉ cần tạo một lần.
Các seed trong `0_config.py` chỉ điều khiển quá trình train mô hình.

## Các artifact được giữ

- Dữ liệu preprocess: `train.csv`, `validation.csv`, `test.csv`.
- Metadata feature: `feature_names.csv`.
- Mỗi thư mục `train/`, `validation/`, `test/`: `X.npy`, `neighbors.npy`.
- Hypergraph train: `edge_memberships.csv.gz`, `edge_meta.csv`, `H0.npz` và
  `graph_config.json`.
- Experiment: checkpoint và train/test report trong `outputs/`.

Đây là output thực của từng bước, không phải cache thông minh. Project không dùng
hash, fingerprint, timestamp state hay run ID để quyết định artifact có hợp lệ.

## Thiết kế thí nghiệm

- Node là enrollment; `truth=1` là dropout.
- Giữ nguyên test chính thức; raw train được chia 80/20 thành train/validation
  bằng `SPLIT_SEED` cố định.
- Các seed `1, 11, 111, 1111, 11111` chỉ dùng để lặp quá trình huấn luyện.
- Normalization và imputation chỉ fit trên train node.
- `test_truth.csv` chỉ được sử dụng khi chạy đánh giá test cuối cùng.
- `H0` có Course, Object và Behavioral hyperedge.
- Behavioral neighbor dùng cosine similarity và FAISS HNSW.
- Validation/test target chỉ nối tới train reference node.
- Validation AUC chọn checkpoint; test chỉ chạy sau khi checkpoint đã được chọn.
- `--no-hsl` chạy HGNN ablation với cùng encoder.

## Kiểm tra cú pháp

```powershell
.\.venv\Scripts\python.exe -m py_compile src/*.py
```
