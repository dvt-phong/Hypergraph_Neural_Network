# Environment audit ? 2026-09-13

## Status
Static audit completed for all six pinned repositories: README/Dockerfile,
requirements, recursive Python imports and selected compatibility-sensitive APIs.
No GPU packages installed; no environment or model runtime validated.
See dependency-audit.json for import locations and wheel-audit.json for wheel evidence.

## Laptop observations
- Windows; NVIDIA Quadro T1000, 4096 MiB VRAM, driver 581.32.
- nvidia-smi displays CUDA 13.0: this is a driver capability indication, not proof
  of an installed CUDA toolkit or the runtime used by PyTorch.
- Only registered Python: 3.13.15. Existing project .venv uses this interpreter.
- WSL optional component is not enabled (WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED).
- Ubuntu server hardware, architecture, driver and scheduler have not been inspected.

## Per-repository evidence
| Repository | Declared environment | Actual additional imports / findings |
| --- | --- | --- |
| SIG-Net | requirements pins torch 1.12.1; Docker base torch 1.9.1/CUDA 11.1; DGL wheel index cu117, no DGL pin | pandas, numpy, tqdm, sklearn, openpyxl; DGL RelGraphConv/GATConv/SAGEConv/GINConv. Docker recipe is not a consistent frozen environment. dglgo is installed by Docker but not directly imported by Python code. |
| MST-GCN | README only gives run commands; no dependency versions | torch, dgl, pandas, numpy, sklearn, tqdm, matplotlib, seaborn, scipy. KDD and XuetangX have separate source directories; inspect both. |
| HGNN | Python 3.6, torch 0.4.0, CUDA 9.0, Ubuntu 16.04 | numpy, scipy, PyYAML. config/config.py:21 calls yaml.load without Loader; incompatible with PyYAML 6. Custom !join/!concat constructors require deliberate loader handling. |
| HyperGCN | Compatible with torch 1.0/Python 3.x; incomplete pins | numpy, scipy, tqdm, configargparse, PyYAML. No PyG/DGL requirement in core code. |
| UniGNN | Python >=3.6; historical recommendation to install newest torch/PyG | scipy, path, tqdm; torch_scatter and torch_sparse are directly required, not merely optional. Hard-coded CUDA calls. Data is obtained separately from HyperGCN. |
| AllSet | Python 3.7, torch 1.4.0, CUDA 10.0, PyG 1.6.3, scatter 2.0.4, sparse 0.6.0, cluster 1.5.2 | ipdb, numpy, scipy, pandas, matplotlib, sklearn, tqdm. np.int in load_other_datasets.py:166 is incompatible with NumPy >=1.24. Custom MessagePassing code requires runtime checking after a PyG upgrade. DGL_HAN is an auxiliary baseline, not the AllSetTransformer core. |

## Recommended installation route
Use Ubuntu x86_64 (WSL2 on laptop, native Ubuntu on server) for the combined
GPU environment. Keep all Python dependencies in the project venv; do not
change system Python, global packages or NVIDIA drivers.

A **candidate compatibility environment**, NOT an original-paper environment:
- Python 3.10.x
- torch 2.1.2+cu118
- DGL 1.1.3+cu118
- torch-geometric 2.3.1 (candidate for older MessagePassing code)
- torch-scatter 2.1.2+pt21cu118
- torch-sparse 0.6.18+pt21cu118
- numpy 1.23.5, scipy 1.10.1, pandas 1.5.3
- scikit-learn 1.3.2, xgboost 2.0.3
- PyYAML, tqdm, openpyxl, matplotlib, seaborn, configargparse, path, ipdb
- tensorboard and pytest for shared project development

Why this candidate: vendor indexes contain CPython 3.10 CUDA 11.8 wheels
for torch and PyG extensions on Windows/Linux, and DGL 1.1.3 CUDA 11.8 on Linux.
The inspected DGL index has no matching Windows wheel for this DGL pin.
This is wheel-availability evidence, not a completed dependency resolution,
ABI check, security endorsement, or model reproduction result.
Old versions are selected for historical code compatibility, not as a general
recommendation for new production software.

Do not install CPU DGL silently when GPU DGL is unavailable.
Do not install all historical requirements files into the same environment.
Do not force torch 0.4/1.4 into the existing Python 3.13 venv.
Do not copy the Windows venv to WSL or Ubuntu; create a fresh Linux venv.

## Remaining installation gates
1. Enable/install WSL2 Ubuntu if using the laptop for DGL GPU; OS feature
   installation/restart is separate from the Python venv. Alternatively use
   the school's Ubuntu machine for DGL and native Windows for other development.
2. Inspect target Linux GPU/driver, architecture, Python 3.10 and free disk/RAM.
   A newer server GPU may need another CUDA/PyTorch stack; do not blindly reuse cu118.
3. Resolve/install candidate dependencies in a fresh project-local environment.
4. Run pip check; CUDA forward/backward for torch, PyG scatter/sparse and DGL
   RelGraphConv; XGBoost GPU training on a tiny synthetic dataset.
5. Import and run synthetic model forward/backward for each original repo in
   separate processes, respecting its working directory and local module imports.
6. Record necessary compatibility patches separately; keep original repos clean.
7. Only after success generate per-platform pip freeze/lock and a repeatable installer.
8. Original-paper numerical reproduction still needs data, split and metric validation.

A single scripted installation can install several isolated venvs if necessary.
A single venv cannot preserve all conflicting historical versions simultaneously.

## Deliberately deferred dependencies
- torch-cluster: listed by historical AllSet installation but not directly imported
  by the inspected Python code; add only if the selected execution path requires it.
- torchvision/torchaudio: not directly imported by the six repositories.
- dglgo/torchdata: not required by the proposed DGL 1.1.3 core execution path.
- DeepHypergraph and HSL: not among the six cloned codebases; no claim of coverage.
- Notebook tooling: optional development extra, not required for model training.
- CUDA toolkit/compiler: avoid source builds initially; use matching binary wheels.

## Primary references
- https://pytorch.org/get-started/previous-versions/
- https://data.dgl.ai/wheels/cu118/repo.html
- https://data.pyg.org/whl/torch-2.1.0+cu118.html
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- Local repository files at commits recorded in third_party/manifest.json.
