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

Project nghi?n c?u d? ?o?n b? h?c MOOC v? t?i l?p graph/hypergraph baselines.

## Tr?ng th?i

?? d?ng c?u tr?c. Ch?a tri?n khai preprocessing, model ho?c training;
?? clone s?u repo t?c gi? trong `third_party/`; ch?a c?i dependency GPU.
CLI hi?n ch? cung c?p tr? gi?p v? b?o r? c?c ch?c n?ng ch?a tri?n khai.

## M?i tr??ng ri?ng

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe run.py --help
```

Ubuntu / WSL (t?o l?i m?i tr??ng t?i m?y ??ch):

```bash
bash scripts/setup.sh
.venv/bin/python run.py --help
```

Kh?ng copy `.venv` gi?a c?c m?y. Kh?ng d?ng `sudo pip` ho?c
`--system-site-packages`. Lu?n c?i b?ng Python trong `.venv`.
Python hi?n c? tr?n laptop l? 3.13; phi?n b?n m?i tr??ng benchmark cu?i c?ng
s? ???c ch?n sau khi ki?m tra t??ng th?ch PyTorch/PyG/DGL v? GPU.
Kh?ng thay ??i driver NVIDIA c?a server trong script setup.

## C?u tr?c

- `run.py`: CLI chung.
- `configs/`: c?u h?nh m?y, dataset v? b? th? nghi?m.
- `src/`: x? l? d? li?u, c?u tr?c, training, evaluation d?ng chung.
- `models/`: implementation/adapter do project qu?n l?.
- `third_party/`: repo g?c; URL v? commit ???c l?u trong `manifest.json`.
- `scripts/`: setup, t?i l?p v? ch?y server.
- `data/raw`, `data/processed`: d? li?u g?c v? cache ?? x? l?.
- `outputs/runs`: m?i run ch?a config, metadata, log, metrics v? checkpoint.
- `outputs/reports`: b?ng v? h?nh t?ng h?p.
- `docs/`, `notebooks/`, `tests/`: protocol, ph?n t?ch v? ki?m tra.

Ch? th?m file model khi b?t ??u tri?n khai. `pyproject.toml` khai b?o package;
`requirements.txt` d?nh cho dependency ???c kh?a sau khi ki?m tra m?i tr??ng.
C?u h?nh GPU m?c ??nh l? `cuda`; training t??ng lai ph?i b?o l?i n?u ng??i d?ng
ch?n CUDA nh?ng CUDA kh?ng kh? d?ng. LR v?n s? d?ng CPU.

## Environment audit

See [docs/environment-audit.md](docs/environment-audit.md) for all six repositories,
GPU observations, candidate versions and remaining installation checks.
The current Python 3.13 venv is a scaffold environment, not the selected GPU stack.
