# XuetangX dropout prediction with HGSL

Project nghiên cứu dự đoán dropout trên XuetangX. Mỗi enrollment là một node;
Course, Object và Behavioral hyperedge tạo `H0`; HGNN tạo `Z0`; HSL học lại
membership để tạo `H*`; cùng HGNN tạo `Z*`; classifier sinh dropout logit.

Code chính gồm tám file và được đọc theo đúng thứ tự pipeline:

| Bước | File | Trách nhiệm |
|---|---|---|
| 1 | `download.py` | Tải ba raw file nếu chưa tồn tại |
| 2 | `preprocess.py` | Stream archive và tạo bốn bảng sạch |
| 3 | `features.py` | Tạo feature và chuẩn hóa theo train split |
| 4 | `hypergraph.py` | Tạo ba loại hyperedge và sparse `H0` |
| 5 | `model.py` | HGNN propagation và `HGSLModel` |
| 6 | `hsl.py` | Học membership và tạo `H*` |
| 7 | `losses.py` | Weighted BCE và contrastive loss |
| 8 | `train.py` | Train, validation, early stopping và test |

Xem [hướng dẫn đọc code](docs/code-guide.md),
[tài liệu pipeline đầy đủ](docs/project-guide.md) và
[sơ đồ mô hình](docs/assets/hypergraph-neural-network-v3.png).

## Cài đặt và chạy

Trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/download.py
.\.venv\Scripts\python.exe src/preprocess.py
.\.venv\Scripts\python.exe src/features.py --seed 1
.\.venv\Scripts\python.exe src/hypergraph.py --seed 1 --k 10 --k-max 20
.\.venv\Scripts\python.exe src/train.py --seed 1 --feature-set full --epochs 50
.\.venv\Scripts\python.exe src/train.py --mode test --checkpoint outputs/runs/simple_hgsl_full_seed_1.pt
```

`download.py` không kiểm checksum và không giải nén archive. Nếu một raw file đã
tồn tại, script bỏ qua file đó. Nếu lần tải trước bị lỗi, hãy xóa file tương ứng
rồi chạy lại.

`preprocess.py` đọc trực tiếp `train_log.csv`, `test_log.csv`, `train_truth.csv`
và `test_truth.csv` trong `prediction_data.tar.gz`. Nếu đủ bốn processed output
thì script bỏ qua toàn bộ raw data. Muốn preprocess lại, hãy xóa bốn file
`nodes.csv`, `users.csv`, `courses.csv`, `events_35d.csv.gz` trong
`data/processed/simple/`.

Với seed mới, cần chạy lại `features.py` và `hypergraph.py` trước khi train.
`train.py` không tự suy đoán artifact cũ còn hợp lệ hay tự rebuild pipeline.

## Các artifact được giữ

- Clean data: `nodes.csv`, `users.csv`, `courses.csv`, `events_35d.csv.gz`.
- Feature: `feature_base.npz`, `node_objects.csv.gz`, `X_seed_*.npy`,
  `feature_stats_seed_*.json`.
- Hypergraph: `neighbors_seed_*.npy`, `edge_memberships_seed_*.csv.gz`,
  `edge_meta_seed_*.csv`, `H0_seed_*.npz`, `train_ids_seed_*.npy` và
  `graph_config_seed_*.json`.
- Experiment: checkpoint và train/test report trong `outputs/`.

Đây là output thực của từng bước, không phải cache thông minh. Project không dùng
hash, fingerprint, timestamp state hay run ID để quyết định artifact có hợp lệ.

## Thiết kế thí nghiệm

- Node là enrollment; `truth=1` là dropout.
- Split 64/16/20 theo user với các seed `1, 11, 111, 1111, 11111`.
- Normalization và imputation chỉ fit trên train node.
- `H0` có Course, Object và Behavioral hyperedge.
- Behavioral neighbor dùng cosine similarity và FAISS HNSW.
- Validation/test target chỉ nối tới train reference node.
- Validation AUC chọn checkpoint; test chỉ chạy sau khi checkpoint đã được chọn.
- `--no-hsl` chạy HGNN ablation với cùng encoder.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test nhỏ kiểm tra toàn bộ pipeline trên một archive giả lập, data leakage và việc
`torch.sparse.mm()` truyền gradient tới membership scorer.
