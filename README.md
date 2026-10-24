# Hypergraph Neural Network for MOOC Dropout Prediction

This project predicts whether a learner drops out of a MOOC course from the learner's activity
log in the first 35 days. Each enrollment (one learner in one course) is a node of a hypergraph.
Enrollments that share a course, a learning object or a learner are linked by hyperedges, so
every prediction can use both the enrollment's own activity and the activity of related
enrollments.

The main dataset is XuetangX (Feng et al., 2019). The same pipeline also runs on OULAD.

## Model

![Model architecture](docs/assets/Hypergraph%20Final.png)

**Nodes.** One node per enrollment. Its features X59 come only from the activity log:
35 daily event counts, the total number of events, the number of distinct objects used and
22 action counts. Every count c becomes `x = (log(1 + c) − μ_train) / σ_train`, with μ and σ
computed on the training nodes only.

**Hyperedges.** Four families, built once over every enrollment (train, validation and test):

| Family | Hyperedge |
|---|---|
| Course | all enrollments of the same course |
| Object | all enrollments that used the same video, problem or forum thread in days 0–34 |
| User | all enrollments of the same learner |
| Self-loop | one hyperedge `{v}` per node |

**Two branches.**

- *Hypergraph branch*: a 2-layer HGNN (Feng et al., 2019) on the incidence matrix H. Each
  hyperedge family has one learnable weight, so the model learns how much each kind of
  relation matters.
- *Self branch*: an MLP on the node's own features, with no propagation.

The two embeddings are concatenated and a linear classifier outputs the dropout probability.
The loss is binary cross-entropy on the training nodes.

```
G      = Dv^-1/2 · H · W · De^-1 · Hᵀ · Dv^-1/2,   W = diag(softplus(θ_family))
Z1     = ReLU(G · (X·Θ1 + b1))
Z      = ReLU(G · (Dropout(Z1)·Θ2 + b2))                    hypergraph embedding
Z_self = ReLU(Dropout(ReLU(X·A1 + c1))·A2 + c2)             self embedding
p      = σ([Z ‖ Z_self]·u + b)
L_BCE  = −(1/|V_train|) Σ_{v ∈ V_train} [y·log p + (1 − y)·log(1 − p)]
```

Training is full-batch and transductive: the forward pass runs over the whole hypergraph and the
loss uses only training nodes. Validation AUC selects the best epoch; test nodes are scored with
those weights at threshold 0.5.

## Project layout

| File | Purpose |
|---|---|
| `src/0_config.py` | Paths, seeds, feature layout, hyperedge families, scenarios, training hyperparameters |
| `src/1_download.py` | Download the raw XuetangX files |
| `src/2_preprocess.py` | Keep events of days 0–34; write the train / validation / test split of every seed |
| `src/3_features.py` | Build X59 for every seed → `x59_s{seed}.npz` (shared by the model and all baselines) |
| `src/4_hypergraph.py` | Build the hypergraph H once → `hypergraph.npz` |
| `src/5_graph_data.py` | Load H and the features of one seed, apply a scenario |
| `src/6_hgnn.py`, `src/7_mlp.py`, `src/8_model.py` | Hypergraph branch, self branch, full model |
| `src/9_train.py` | Train and evaluate → `outputs/results.csv` |
| `src/10_summary.py` | Mean ± std over seeds → `outputs/summary.csv`, `outputs/ket_qua.xlsx` |
| `scripts/run_tmux.sh` | Run training and summary in a detached tmux session on a server |
| `baseline/` | LR, SVM, XGBoost, CFIN, MST-GCN, CA-TFHN, and the OULAD pipeline |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A pinned GPU environment is also provided: `conda env create -f environment-gpu.yml`.

## Run on XuetangX

```bash
python src/1_download.py
python src/2_preprocess.py
python src/3_features.py
python src/4_hypergraph.py
python src/9_train.py --scenario M --seeds 1 11 111 1111 11111
python src/10_summary.py
```

Each seed (1, 11, 111, 1111, 11111) splits all labeled enrollments into 60% train, 10% validation
and 30% test, stratified by label, and initialises the model. Results are reported as mean ± std
over the five seeds.

Options of `src/9_train.py`: `--scenario`, `--seeds`, `--epochs`, `--eval-every`, `--patience`,
`--device {auto,cpu,cuda}`. A quick test: `python src/9_train.py --scenario M --seeds 1 --epochs 10`.

On a server, the same run inside tmux:

```bash
bash scripts/run_tmux.sh --scenario M --seeds 1 11 111 1111 11111
```

### Ablations

`--scenario all` runs the main model M and its ablations; each changes one component of M.

| Scenario | Change from M |
|---|---|
| M | Course + Object + User + self-loop, 2 HGNN layers, MLP branch, learned W |
| A1 / A2 / A3 / A4 | drop Course / Object / User / self-loop hyperedges |
| W1 | fixed W = I (as the original HGNN) |
| L1 | 1 HGNN layer |
| H | no MLP branch (hypergraph only) |
| X1 (optional, run by name) | self-loops only, no neighbours |

## Baselines

Every baseline reads the same `x59_s{seed}.npz` (same features, same split per seed). Run them
after `src/3_features.py`.

```bash
python baseline/ML/LR.py
python baseline/ML/SVM.py
python baseline/ML/XGBoost.py

bash baseline/CFIN/run_cfin.sh setup && bash baseline/CFIN/run_cfin.sh full
bash baseline/MST-GCN/run_mstgcn.sh setup && bash baseline/MST-GCN/run_mstgcn.sh features && bash baseline/MST-GCN/run_mstgcn.sh full
bash baseline/CA-TFHN/run_catfhn.sh setup && bash baseline/CA-TFHN/run_catfhn.sh features && bash baseline/CA-TFHN/run_catfhn.sh full

python baseline/summary.py
```

CFIN, MST-GCN and CA-TFHN use the authors' original code at a pinned commit plus a small patch
(`*_changes.patch`). `setup` clones the code and applies the patch. Each one needs its own
environment; see the header of its `run_*.sh` and its `requirements.txt`.

## Run on OULAD

The OULAD pipeline (X55 features) lives in `baseline/new_dataset/OULAD/` and mirrors `src/`:

```bash
python baseline/new_dataset/OULAD/1_download.py
python baseline/new_dataset/OULAD/2_preprocess.py
python baseline/new_dataset/OULAD/3_features.py
python baseline/new_dataset/OULAD/4_hypergraph.py
python baseline/new_dataset/OULAD/9_train.py --scenario M --seeds 1 11 111 1111 11111
python baseline/new_dataset/OULAD/10_summary.py
```

Baselines on OULAD:

```bash
python baseline/ML/LR.py --x-dir data/processed/oulad --x-name x55 --output-dir outputs/oulad/baselines/LR
DATASET=oulad bash baseline/CFIN/run_cfin.sh full
python baseline/summary.py --results-dir outputs/oulad/baselines --x-dir data/processed/oulad --x-name x55
```

SVM and XGBoost take the same arguments as LR; MST-GCN and CA-TFHN take `DATASET=oulad` as CFIN.

## References

- Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). Hypergraph Neural Networks. AAAI.
- Feng, W., Tang, J., & Liu, T. X. (2019). Understanding Dropouts in MOOCs. AAAI. (XuetangX, CFIN)
- Kuzilek, J., Hlosta, M., & Wolff, A. (2017). Open University Learning Analytics Dataset. Scientific Data.
