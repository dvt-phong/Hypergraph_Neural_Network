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
| `scripts/run_ablation.sh` | Run the ablation study on OULAD and XuetangX in a detached tmux session |
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

## Ablation

| Code | Scenario |
|---|---|
| M | Full model |
| H | w/o MLP branch |
| A1 | w/o Course hyperedge |
| A2 | w/o Object hyperedge |
| A3 | w/o User hyperedge |
| A4 | w/o self-loop |
| W1 | w/o adaptive weight (W = I) |

```bash
bash scripts/run_ablation.sh
```

The script trains every scenario with the five seeds on OULAD and then XuetangX. Results,
summary and log of each dataset go to `outputs/ABLATION/<dataset>/`
(`results.csv`, `summary.csv`, `ket_qua.xlsx`, `logs/`).

Extra scenarios (`docs/PLAN_EXTRA_EXPERIMENTS.md`) are chosen with `SCENARIOS` and appended to
the same `results.csv`:

| Code | Scenario |
|---|---|
| P0 | MLP-only: no HGNN branch |
| S1 | M on a shuffled H: node ids permuted within each family, hyperedge sizes and d(v) kept |
| C1 | Course + self-loop only |
| O1 | Object + self-loop only |
| U1 | User + self-loop only |
| L1 | HGNN with 1 layer |
| L3 | HGNN with 3 layers |
| UC | M with a causal User hyperedge: node v only receives from enrollments of the same learner with start ≤ start(v) (XuetangX only) |

```bash
SCENARIOS="P0 S1" bash scripts/run_ablation.sh
SCENARIOS="C1 O1 U1 L1 L3" bash scripts/run_ablation.sh
```
