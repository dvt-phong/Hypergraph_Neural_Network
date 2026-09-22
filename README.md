# XuetangX dropout prediction with HGSL

Một project, một mô hình: mỗi enrollment là một node; Course, Object và Behavioral
hyperedge tạo `H0`; HGNN tạo `Z0`; HSL tạo `H*`; cùng HGNN tạo `Z*`; classifier dự đoán
dropout. Loss là weighted BCE cộng `lambda ×` contrastive InfoNCE.

Code chính chỉ có tám file trong `src/`, theo đúng thứ tự chạy:

| Bước | File | Việc cần đọc |
|---|---|---|
| 1 | `download.py` | Tải và giải nén CSV |
| 2 | `preprocess.py` | Làm sạch event, tạo node và user-disjoint split |
| 3 | `features.py` | Tạo và chuẩn hóa node feature bằng train split |
| 4 | `hypergraph.py` | Tạo Course/Object/Behavioral hyperedge |
| 5 | `model.py` | Tạo sparse `H0`, ghép `X`, hai lượt HGNN và classifier |
| 6 | `hsl.py` | Sample và refine membership để có `H*` |
| 7 | `losses.py` | BCE, contrastive và total loss |
| 8 | `train.py` | Train, validation chọn checkpoint, test |

Xem [tài liệu đầy đủ theo sơ đồ và input/output từng file](docs/project-guide.md),
[hướng dẫn đọc code ngắn](docs/code-guide.md) và
[sơ đồ mô hình](docs/assets/hypergraph-neural-network-v3.png).

## Chạy từ dữ liệu gốc

Trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/download.py
.\.venv\Scripts\python.exe src/preprocess.py
.\.venv\Scripts\python.exe src/features.py
.\.venv\Scripts\python.exe src/hypergraph.py --seed 1 --k 10
.\.venv\Scripts\python.exe src/model.py --seed 1
.\.venv\Scripts\python.exe src/train.py --seed 1 --feature-set full --epochs 50
$checkpoint = Get-ChildItem outputs/runs/simple_hgsl_full_seed_1_*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.\.venv\Scripts\python.exe src/train.py --mode test --checkpoint $checkpoint.FullName
```

`train.py` chỉ dùng validation để chọn checkpoint. `--mode test` đánh giá test sau
khi chốt cấu hình. Dùng `--mode both` nếu cấu hình đã được chốt. Mặc định
`--feature-set behavior` là 60 chiều; `full` là 96 chiều. `--no-hsl` cho ablation
HGNN với cùng encoder. Tên checkpoint có mã 10 ký tự từ cấu hình và dữ liệu,
nên các cấu hình khác nhau có file riêng. Chạy lại cùng cấu hình trên cùng dữ liệu
sẽ ghi đè checkpoint đó. Khi test, code kiểm tra các file dữ liệu còn khớp với
lúc train; nếu đã tạo lại dữ liệu, cần train checkpoint mới.

Dữ liệu bảng lưu dạng CSV/CSV nén; `X` dùng NumPy `.npy`, `H0` dùng SciPy sparse
`.npz`. Project không dùng cơ sở dữ liệu hoặc Parquet. Hai log gốc khoảng 6 GB;
`features.py` chia các cặp session/object thành file tạm nhỏ để đếm chính xác mà
không giữ tất cả event trong RAM.

## Thiết kế hiện tại

- Node là enrollment; `truth=1` là dropout.
- Split 64/16/20 theo user, dùng năm seed `1, 11, 111, 1111, 11111`.
- `H0` chính có Course, Object và Behavioral edge. User edge trong hình chưa dùng.
- Hai lượt HGNN dùng chung trọng số. HSL refine một lượt mỗi forward.
- Train sample edge; validation/test refine toàn bộ edge trong local graph của từng
  target, chỉ lấy train node làm reference.

Các quyết định này giữ theo code trước refactor. Bản trước refactor nằm ở commit
`3d50a16` trên `main` để đối chiếu khi cần.

Khi đối chiếu `X` với bản cũ, 60 behavioral feature khớp hoàn toàn. Bản cũ gán
category trống của gender/education/course vào cột `other`; bản này gán vào cột
`missing` theo đúng tên feature. Đây là thay đổi đầu vào có chủ đích, nên metric
`full` sau refactor cần được chạy lại; không so trực tiếp với checkpoint cũ.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test nhỏ chạy toàn bộ luồng. Sau khi chạy đủ dữ liệu thật và chọn hyperparameter
bằng validation, cần chạy năm seed và các ablation trước khi báo cáo kết quả nghiên cứu.
