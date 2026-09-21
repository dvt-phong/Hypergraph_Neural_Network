# Hướng dẫn đọc code

Code được tổ chức theo đúng thứ tự chạy của mô hình. Điểm bắt đầu là
`src/main.py`, đặc biệt là hai hàm `run_pipeline()` và `run_experiments()`.
Tài liệu đối chiếu chi tiết từng khối trong sơ đồ nằm ở
[hgsl-model-code-walkthrough.md](hgsl-model-code-walkthrough.md).

Mỗi hàm/class trong `src/` có comment `# Mục đích`, `# Đầu vào`, `# Đầu ra` ngay
trước định nghĩa; `# Lưu ý` được thêm khi có điều kiện cần nhớ. Đọc ba dòng đó
trước rồi xem thân hàm để đối chiếu phép biến đổi dữ liệu. Không dùng docstring
ba nháy làm comment; dấu ba nháy còn lại trong code là chuỗi SQL nhiều dòng.

## Luồng chính

```text
Raw CSV
→ preprocessing và user-disjoint split
→ behavioral/user/course features
→ course/object/behavioral hyperedges
→ sparse H0
→ HGNN tạo Z0
→ HSL tạo H*
→ HGNN tạo Z*
→ dropout logits và loss
→ validation chọn checkpoint
→ test checkpoint đã chọn
```

## Module và dữ liệu vào/ra

| Module | Nhiệm vụ | Đầu vào | Đầu ra |
|---|---|---|---|
| `main.py` | Thể hiện pipeline và cung cấp CLI | Tham số experiment | Report của từng stage |
| `config.py` | Khóa dataset contract, feature schema và seed | Không có | Constants và tên feature |
| `data/preprocess.py` | Chuẩn hóa raw XuetangX | CSV gốc | `nodes`, `events_35d`, `users`, `courses` |
| `data/split.py` | Tạo split user-disjoint | `nodes.parquet`, seed | `splits.parquet` |
| `features/` | Aggregate và transform feature, chỉ fit trên train | Canonical tables và split | `X_base`, `X_context`, transforms |
| `hypergraph/` | Tạo candidate hyperedge và sparse H0 | Events, split, features | Memberships, neighbors, H0, metadata |
| `graph_data.py` | Ghép đúng feature với train/local graph | Artifacts của feature và graph | Train graph hoặc local evaluation graph |
| `model.py` | Sparse propagation và HGNN encoder | Feature tensor và incidence operator | Node embeddings và logits |
| `hsl.py` | Sampling, membership refinement và loss | Z0, H0, metadata, labels | H*, Z*, logits và loss |
| `train.py` | Train, validation, checkpoint và test | Config và processed artifacts | Checkpoint, history và metrics report |
| `metrics.py` | Tính metric nhị phân | Labels và probability | AUC, AUPRC, F1, precision, recall |
| `artifacts.py` | I/O dùng chung | Path và payload | Atomic artifact, signature, timestamp |

## Quy tắc thí nghiệm

- Transformer và class weight chỉ được fit/tính từ train split.
- Validation AUC chọn checkpoint.
- Test split chỉ được đọc bởi `evaluate_hgsl_checkpoint()` hoặc
  `evaluate_baseline_checkpoint()` sau khi checkpoint đã được chọn.
- Mỗi seed điều khiển cả data split, model initialization và sampling.
- `check-hgsl` chỉ là một optimizer step để debug gradient; không phải kết quả
  thí nghiệm.

## Lệnh thường dùng

```powershell
# Kiểm tra nhanh HSL
.\.venv\Scripts\python.exe run.py check-hgsl --seed 1 --feature-set full

# Train một cấu hình
.\.venv\Scripts\python.exe run.py train-hgsl --seed 1 --feature-set full --epochs 50

# Test checkpoint đã chọn bằng validation
.\.venv\Scripts\python.exe run.py evaluate --seed 1 --feature-set full

# Chạy năm seed và tổng hợp mean/std
.\.venv\Scripts\python.exe run.py run-experiments --feature-sets behavior full --epochs 50
```
