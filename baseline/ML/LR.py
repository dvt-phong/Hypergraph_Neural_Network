# Baseline - Logistic Regression
#
#   python baseline/ML/LR.py
#
# Protocol of docs/PLAN_BASELINE_v3.md: learner split of baseline/prepare_raw.py (one split per
# seed, split seed = SGD seed), alpha and epoch chosen on validation AUC, metrics AUC, precision,
# recall and F1 of the dropout class at threshold 0.5 (as src/9_train.py of M).
# LR has no preprocessing of its own, so it uses the 89 columns X of the main model M
# (data/processed/simple/{split}/X.npy, read only). Its rows are re-ordered by enroll_id and the
# z-scored columns (59 behavior columns, age) are z-scored again with the learner-train rows; the
# z-score of an affine map of a column is the same, so this equals log1p + z-score fitted on the
# learner-train rows. One-hot columns stay 0/1.

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
X_DIR = ROOT / "data" / "processed" / "simple"
RAW_DIR = ROOT / "data" / "processed" / "baseline_raw"
OUTPUT_DIR = ROOT / "outputs" / "baselines" / "LR"
M_SPLITS = ("train", "validation", "test")   # folders of X.npy (official split of M)
SPLITS = ("train", "validation", "test")
SEEDS = (1, 11, 111, 1111, 11111)
FEATURE_COUNT = 89
ZSCORE_COLUMNS = list(range(59)) + [70]      # X[:, 0:59] log1p + z-score, X[:, 70] age z-score
ALPHA_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
MAX_EPOCHS = 200
PATIENCE = 10
THRESHOLD = 0.5                              # ŷ = 1[p ≥ 0.5], as config.THRESHOLD of M
METRICS = ("auc", "precision", "recall", "f1")
RESULT_COLUMNS = (
    "time", "model", "seed", "alpha", "best_epoch", "epochs_run",
    "val_auc", "val_precision", "val_recall", "val_f1",
    "test_auc", "test_precision", "test_recall", "test_f1", "minutes",
)


# Print a message with the time and a scope, e.g. [lr].
def log(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


# enroll_id of every row of X.npy (row = node_id of the split CSV); cached in RAW_DIR.
def x_enroll_ids(x_dir, cache_path):
    if cache_path.is_file():
        rows = pd.read_csv(cache_path)
    else:
        parts = []
        for name in M_SPLITS:
            pairs = []
            for chunk in pd.read_csv(x_dir / f"{name}.csv", usecols=["node_id", "enroll_id"], chunksize=5_000_000):
                pairs.append(chunk.drop_duplicates("node_id"))
            pairs = pd.concat(pairs).drop_duplicates("node_id").sort_values("node_id")
            pairs.insert(0, "m_split", name)
            parts.append(pairs)
            log("setup", f"{name}.csv: {len(pairs):,} nodes")
        rows = pd.concat(parts, ignore_index=True)
        rows.to_csv(cache_path, index=False)
    enroll_ids = {}
    for name in M_SPLITS:
        part = rows[rows["m_split"] == name]
        if not np.array_equal(part["node_id"].to_numpy(), np.arange(len(part))):
            raise ValueError(f"{name}: node_id is not 0..n-1")
        enroll_ids[name] = part["enroll_id"].to_numpy()
    return enroll_ids


# X (89 columns) and label of every enrollment, indexed by enroll_id.
def load_x(x_dir, raw_dir):
    enroll_ids = x_enroll_ids(x_dir, raw_dir / "x89_enroll_ids.csv")
    blocks, ids = [], []
    for name in M_SPLITS:
        x = np.load(x_dir / name / "X.npy").astype(np.float64)
        if x.shape != (len(enroll_ids[name]), FEATURE_COUNT):
            raise ValueError(f"{name}/X.npy has shape {x.shape}; rerun src/3_features.py")
        blocks.append(x)
        ids.append(enroll_ids[name])
    x = pd.DataFrame(np.vstack(blocks), index=np.concatenate(ids))
    truth = pd.read_csv(raw_dir / "prediction_log" / "truth.csv", index_col="enroll_id")["truth"]
    if len(x) != len(truth) or not x.index.isin(truth.index).all():
        raise ValueError("X rows and baseline_raw truth.csv cover different enrollments")
    return x, truth


# Rows of one learner split; z-scored columns refitted on its train rows.
def make_splits(x, truth, split_path):
    split = pd.read_csv(split_path, index_col="enroll_id")["split"]
    train_x = x.loc[split.index[split == "train"]].to_numpy()
    mean = train_x[:, ZSCORE_COLUMNS].mean(axis=0)                       # μ_train per column
    std = train_x[:, ZSCORE_COLUMNS].std(axis=0)                         # σ_train per column
    std[std == 0] = 1.0                                                  # constant column -> z = 0
    splits = {}
    for name in SPLITS:
        ids = split.index[split == name].to_numpy()
        values = x.loc[ids].to_numpy().copy()
        values[:, ZSCORE_COLUMNS] = (values[:, ZSCORE_COLUMNS] - mean) / std   # z = (x − μ_train) / σ_train
        splits[name] = {"x": values, "y": truth.loc[ids].to_numpy().astype(np.int64), "enroll_id": ids}
    return splits


# AUC, precision, recall and F1 of the dropout class (label 1) at THRESHOLD.
def metrics(labels, probabilities):
    predicted = probabilities >= THRESHOLD                              # ŷ = 1[p ≥ 0.5]
    # P = TP/(TP + FP), R = TP/(TP + FN), F1 = 2·P·R/(P + R)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=[1], zero_division=0)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),             # AUC = P(p_dropout > p_non-dropout)
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


# Train one alpha with SGD and early stopping on validation AUC; return the best epoch's model.
def train_sgd(alpha, seed, train, validation):
    # min_w,b  (1/n)·Σ_i log(1 + exp(−y_i·(w·x_i + b))) + (α/2)·‖w‖²
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, random_state=seed)
    rng = np.random.default_rng(seed)                                    # seed -> order of the train rows
    best = {"auc": -1.0, "epoch": 0, "model": None}
    stale = 0          # epochs in a row without a better validation AUC
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(len(train["y"]))                         # new shuffle every epoch
        model.partial_fit(train["x"][order], train["y"][order], classes=np.array([0, 1]))
        validation_auc = roc_auc_score(validation["y"], model.predict_proba(validation["x"])[:, 1])
        if validation_auc > best["auc"]:                                 # best = argmax_epoch AUC_val
            best = {"auc": validation_auc, "epoch": epoch, "model": copy.deepcopy(model)}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    best["epochs_run"] = epoch
    return best


# Write split, enroll_id, label, prob of one split to a CSV file.
def write_predictions(path, split_name, data, probability):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("split", "enroll_id", "label", "prob"))
        for enroll_id, label, prob in zip(data["enroll_id"], data["y"], probability):
            writer.writerow((split_name, int(enroll_id), int(label), f"{prob:.8f}"))


# Append one row to results.csv (header written when the file is new).
def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        text_row = {}
        for name in row:
            value = row[name]
            if isinstance(value, float):
                text_row[name] = f"{value:.6f}"
            else:
                text_row[name] = value
        writer.writerow(text_row)


# Choose alpha on validation AUC, score validation and test once, save predictions and one result row.
def run_one_seed(seed, splits, output_dir):
    started_at = time.perf_counter()
    train, validation, test = splits["train"], splits["validation"], splits["test"]

    best = {"auc": -1.0}
    for alpha in ALPHA_GRID:
        result = train_sgd(alpha, seed, train, validation)
        log("lr", f"seed {seed} alpha={alpha:g}: val_auc={result['auc']:.4f} "
                  f"(best epoch {result['epoch']} of {result['epochs_run']})")
        if result["auc"] > best["auc"]:                                  # α* = argmax_α AUC_val
            best = dict(result, alpha=alpha)

    scores = {}
    for split_name, data in (("validation", validation), ("test", test)):
        probability = best["model"].predict_proba(data["x"])[:, 1]       # p = σ(w·x + b)
        scores[split_name] = metrics(data["y"], probability)
        write_predictions(output_dir / f"seed{seed}_{split_name}.csv", split_name, data, probability)
        log(split_name, f"lr seed {seed} alpha={best['alpha']:g}: "
                        + ", ".join(f"{name}={scores[split_name][name]:.4f}" for name in METRICS))

    row = {}
    row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["model"] = "LR"
    row["seed"] = seed
    row["alpha"] = best["alpha"]
    row["best_epoch"] = best["epoch"]
    row["epochs_run"] = best["epochs_run"]
    for split_name, prefix in (("validation", "val"), ("test", "test")):
        for metric_name in METRICS:
            row[f"{prefix}_{metric_name}"] = scores[split_name][metric_name]
    row["minutes"] = (time.perf_counter() - started_at) / 60
    append_result(output_dir / "results.csv", row)
    return scores["test"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR trained by SGD on learner splits, alpha and epoch on validation AUC.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--x-dir", default=X_DIR, help="folder of M with {split}/X.npy and {split}.csv")
    parser.add_argument("--raw-dir", default=RAW_DIR, help="output of baseline/prepare_raw.py")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="folder for results.csv and predictions")
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir)
    raw_dir = Path(arguments.raw_dir)
    x, truth = load_x(Path(arguments.x_dir), raw_dir)
    log("setup", f"X {FEATURE_COUNT} columns, {len(x):,} enrollments, seeds={arguments.seeds}")

    test_scores = []
    for seed in arguments.seeds:
        splits = make_splits(x, truth, raw_dir / "splits" / f"learner_s{seed}.csv")
        log("setup", f"learner_s{seed}: " + ", ".join(f"{name}={len(splits[name]['y']):,}" for name in SPLITS))
        test_scores.append(run_one_seed(seed, splits, output_dir))
    # mean ± std (ddof = 1) over seeds of the test metrics
    parts = []
    for metric_name in METRICS:
        values = [scores[metric_name] for scores in test_scores]
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        parts.append(f"{metric_name}={np.mean(values):.4f}+-{std:.4f}")
    log("summary", f"LR test over {len(test_scores)} learner splits: " + ", ".join(parts))
    log("done", f"results and predictions in {output_dir}")
