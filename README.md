# XuetangX Hypergraph Structure Learning

Project dự đoán dropout bằng Hypergraph Neural Network kết hợp học lại cấu trúc
hypergraph. Thí nghiệm chính chỉ dùng phần XuetangX có nhãn gồm **225.642
enrollment, 77.083 user và 247 course**.

## Quyết định chính

- Node là enrollment; identity chuẩn là `enroll_id`.
- Input chính `X_base` có 60 chiều.
- Initial hypergraph gồm Course, Object và Behavioral hyperedge.
- User hyperedge chưa dùng trong cấu hình chính vì nguy cơ temporal leakage.
- Split chính là user-disjoint train/validation/test với tỷ lệ mục tiêu 64/16/20.
- Không dùng 351 triệu full-activity event trong thí nghiệm chính vòng đầu.
- SIG-Net và MST-GCN được chạy lại trên cùng XuetangX-247; không so trực tiếp với
  kết quả XuetangX 1.213 course đã công bố.

Chi tiết dữ liệu nằm tại
[docs/xuetangx-feature-analysis.md](docs/xuetangx-feature-analysis.md). Kế hoạch
triển khai nằm tại [docs/implementation-plan.md](docs/implementation-plan.md).

## Cấu trúc code

Code đặt trực tiếp dưới `src/`, không có package trung gian `mooc_hgsl`:

```text
src/
  paths.py
  cli.py
  data/
  features/
  hypergraph/
  models/
  losses/
  training/
```

`baseline/` chứa manifest và hướng dẫn cho các repository tham khảo. Các clone cục
bộ được Git ignore và không được import vào mô hình đề xuất.

## Trạng thái

Phase 0–3 đã hoàn thành. Dữ liệu đã được audit, chia user-disjoint và chuyển thành
`X_base` 60 chiều tại `data/processed/xuetangx_247/`.

Phase 0 đã khóa:

- đường dẫn project;
- schema raw data;
- action vocabulary 23 chiều;
- dataset contract XuetangX-247;
- feature schema 60 chiều;
- năm experiment seed `1, 11, 111, 1111, 11111` và ba hyperedge family chính.

Kiểm tra contract và cấu trúc:

```powershell
.\.venv\Scripts\python.exe run.py contract
.\.venv\Scripts\python.exe run.py structure
.\.venv\Scripts\python.exe run.py audit-data
.\.venv\Scripts\python.exe run.py prepare-data
.\.venv\Scripts\python.exe run.py split-data
.\.venv\Scripts\python.exe run.py build-features
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Hai script full activity được giữ riêng để tải và chuyển JSON thành CSV.GZ:

```powershell
.\.venv\Scripts\python.exe scripts\download_xuetangx_full.py
.\.venv\Scripts\python.exe scripts\convert_xuetangx_full.py
```

Phase tiếp theo là xác định và xây dựng các nhóm hyperedge.
