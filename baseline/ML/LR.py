# Baseline - Logistic Regression

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "simple"
OUTPUT_DIR = ROOT / "outputs" / "baselines" / "LR"
SPLITS = ("train", "validation", "test")
SEEDS = (1, 11, 111, 1111, 11111)
FEATURE_COUNT = 89
THRESHOLD = 0.5
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
MAX_ITER = 5000
RESULT_COLUMNS = (
    "time", "model", "seed", "C", "converged",
    "val_auc", "val_auprc", "val_f1",
    "test_auc", "test_auprc", "test_accuracy", "test_precision", "test_recall", "test_f1",
    "minutes",
)


# Print a message with the time and a scope, e.g. [lr].
def log(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


# X and labels of every split; rows of hypergraph.npz are ordered train -> validation -> test.
def load_splits(data_dir):
    with np.load(data_dir / "hypergraph.npz") as bundle:
        labels = bundle["labels"].astype(np.int64)
        split = bundle["split"]
    splits = {}
    for split_id in range(len(SPLITS)):
        name = SPLITS[split_id]
        x = np.load(data_dir / name / "X.npy")
        y = labels[split == split_id]
        if len(x) != len(y):
            raise ValueError(f"{name}/X.npy and hypergraph.npz do not align; rerun src/4_hypergraph.py")
        if x.shape[1] != FEATURE_COUNT:
            raise ValueError(f"{name}/X.npy has {x.shape[1]} columns, expected {FEATURE_COUNT}; "
                             f"rerun src/3_features.py")
        splits[name] = {"x": x, "y": y, "node_id": np.arange(len(y))}   # node_id = row of X.npy
    return splits


# Metrics of the dropout class (label 1) at threshold 0.5, as src/9_train.metrics.
def metrics(labels, probabilities):
    predicted = probabilities >= THRESHOLD                               # ŷ = 1[p ≥ 0.5]
    # P = TP/(TP + FP), R = TP/(TP + FN), F1 = 2·P·R/(P + R)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[1], zero_division=0)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),             # AUC = P(p_dropout > p_non-dropout)
        "auprc": float(average_precision_score(labels, probabilities)), # AUPRC = Σ_n (R_n − R_{n−1})·P_n
        "accuracy": float(np.mean(predicted == labels)),                # Acc = (TP + TN)/n
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


# Fit on train; also report whether lbfgs converged.
def fit(model, x, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    converged = True
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            converged = False
    return converged


# Write split, node_id, label, prob of one split to a CSV file.
def write_predictions(path, split_name, data, probability):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("split", "node_id", "label", "prob"))
        for node_id, label, prob in zip(data["node_id"], data["y"], probability):
            writer.writerow((split_name, int(node_id), int(label), f"{prob:.8f}"))


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


# Choose C on validation AUC, score validation and test once, save predictions and one result row.
def run_one_seed(seed, splits, output_dir):
    started_at = time.perf_counter()
    train, validation, test = splits["train"], splits["validation"], splits["test"]

    best = {"auc": -1.0, "C": None, "model": None, "converged": None}
    for c_value in C_GRID:
        # min_w,b  ½‖w‖² + C·Σ_i log(1 + exp(−y_i·(w·x_i + b)))
        model = LogisticRegression(C=c_value, max_iter=MAX_ITER, random_state=seed)
        converged = fit(model, train["x"], train["y"])
        validation_auc = roc_auc_score(validation["y"], model.predict_proba(validation["x"])[:, 1])
        log("lr", f"seed {seed} C={c_value:g}: val_auc={validation_auc:.4f}"
                  + ("" if converged else " (not converged)"))
        if validation_auc > best["auc"]:                                 # C* = argmax_C AUC_val
            best = {"auc": validation_auc, "C": c_value, "model": model, "converged": converged}

    scores = {}
    for split_name, data in (("validation", validation), ("test", test)):
        probability = best["model"].predict_proba(data["x"])[:, 1]       # p = σ(w·x + b)
        scores[split_name] = metrics(data["y"], probability)
        write_predictions(output_dir / f"seed{seed}_{split_name}.csv", split_name, data, probability)
        parts = []
        for metric_name in scores[split_name]:
            parts.append(f"{metric_name}={scores[split_name][metric_name]:.4f}")
        log(split_name, f"lr seed {seed} C={best['C']:g}: " + ", ".join(parts))

    row = {}
    row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["model"] = "LR"
    row["seed"] = seed
    row["C"] = best["C"]
    row["converged"] = best["converged"]
    for metric_name in ("auc", "auprc", "f1"):
        row[f"val_{metric_name}"] = scores["validation"][metric_name]
    for metric_name in scores["test"]:
        row[f"test_{metric_name}"] = scores["test"][metric_name]
    row["minutes"] = (time.perf_counter() - started_at) / 60
    append_result(output_dir / "results.csv", row)
    return scores["test"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR on X with C chosen on validation AUC.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--data-dir", default=DATA_DIR, help="folder with {split}/X.npy and hypergraph.npz")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="folder for results.csv and predictions")
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir)
    splits = load_splits(Path(arguments.data_dir))
    counts = []
    for split_name in SPLITS:
        counts.append(f"{split_name}={len(splits[split_name]['y']):,}")
    log("setup", f"X {FEATURE_COUNT} columns, " + ", ".join(counts) + f", seeds={arguments.seeds}")

    test_scores = []
    for seed in arguments.seeds:
        test_scores.append(run_one_seed(seed, splits, output_dir))
    # mean +- std over seeds of every test metric
    parts = []
    for metric_name in test_scores[0]:
        values = []
        for scores in test_scores:
            values.append(scores[metric_name])
        parts.append(f"{metric_name}={np.mean(values):.4f}+-{np.std(values):.4f}")
    log("summary", f"LR test over {len(test_scores)} seeds: " + ", ".join(parts))
    log("done", f"results and predictions in {output_dir}")
