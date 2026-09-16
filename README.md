# Hypergraph_Neural_Network

## XuetangX

Đã bổ sung script tải dữ liệu gốc và thống kê từng field.
Xem [hướng dẫn XuetangX](docs/xuetangx.md).

```powershell
.\.venv\Scripts\python.exe scripts/download_xuetangx.py
.\.venv\Scripts\python.exe -m pip install -r scripts/requirements-data.txt
.\.venv\Scripts\python.exe scripts/profile_xuetangx.py
```

Dữ liệu: `data/raw/xuetangx/`. Báo cáo: `outputs/reports/xuetangx/statistics.md`.

Project nghiên cứu dự đoán bỏ học MOOC và tái lập graph/hypergraph baselines.

## Trạng thái

Đã dựng cấu trúc, bổ sung script tải và thống kê XuetangX.
Chưa triển khai preprocessing cho mô hình, model hoặc training;
đã clone bảy repo tác giả trong `third_party/`; chưa cài dependency GPU.
CLI hiện chỉ cung cấp trợ giúp và báo rõ các chức năng chưa triển khai.

## Môi trường riêng

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe run.py --help
```

Ubuntu / WSL (tạo lại môi trường tại máy đích):

```bash
bash scripts/setup.sh
.venv/bin/python run.py --help
```

Không copy `.venv` giữa các máy. Không dùng `sudo pip` hoặc
`--system-site-packages`. Luôn cài bằng Python trong `.venv`.
Python hiện có trên laptop là 3.13; phiên bản môi trường benchmark cuối cùng
sẽ được chọn sau khi kiểm tra tương thích PyTorch/PyG/DGL và GPU.
Không thay đổi driver NVIDIA của server trong script setup.

## Cấu trúc

- `run.py`: CLI chung.
- `configs/`: cấu hình máy, dataset và bộ thí nghiệm.
- `src/`: xử lý dữ liệu, cấu trúc, training, evaluation dùng chung.
- `models/`: implementation/adapter do project quản lý.
- `third_party/`: repo gốc; URL và commit được lưu trong `manifest.json`.
- `scripts/`: setup, tái lập và chạy server.
- `data/raw`, `data/processed`: dữ liệu gốc và cache đã xử lý.
- `outputs/runs`: mỗi run chứa config, metadata, log, metrics và checkpoint.
- `outputs/reports`: bảng và hình tổng hợp.
- `docs/`, `notebooks/`, `tests/`: protocol, phân tích và kiểm tra.

Chỉ thêm file model khi bắt đầu triển khai. `pyproject.toml` khai báo package;
`requirements.txt` dành cho dependency được khóa sau khi kiểm tra môi trường.
Cấu hình GPU mặc định là `cuda`; training tương lai phải báo lỗi nếu người dùng
chọn CUDA nhưng CUDA không khả dụng. LR vẫn sử dụng CPU.

## Environment audit

See [docs/environment-audit.md](docs/environment-audit.md) for all six repositories,
GPU observations, candidate versions and remaining installation checks.
The current Python 3.13 venv is a scaffold environment, not the selected GPU stack.
