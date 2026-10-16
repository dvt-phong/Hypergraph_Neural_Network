# Baseline - SVM (linear, hinge loss)
#
#   python baseline/ML/SVM.py
#
# Input: X59 of the main model M (data/processed/simple/x59_s{seed}.npz, written by
# src/3_features.py): same features and same train / validation / test as M for every seed.
# Seed s: the split of seed s and the SGD seed s. alpha and the epoch are chosen on validation AUC.
# Metrics: AUC, precision, recall and F1 of the dropout class at threshold 0.5 (as M).

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
X_DIR = ROOT / "data" / "processed" / "simple"
OUTPUT_DIR = ROOT / "outputs" / "baselines" / "SVM"
SPLITS = ("train", "validation", "test")
SEEDS = (1, 11, 111, 1111, 11111)
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


# Print a message with the time and a scope, e.g. [svm].
def log(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


# X, label and enroll_id of train / validation / test of one seed
def load_splits(x_dir, seed):
    data = np.load(Path(x_dir) / f"x59_s{seed}.npz")
    splits = {}
    for name in SPLITS:
        rows = data["split"] == name
        splits[name] = {
            "x": data["X"][rows].astype(np.float64),
            "y": data["label"][rows],
            "enroll_id": data["enroll_id"][rows],
        }
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


# SVM has no probability: p = σ(score), score = w·x + b; σ is monotone, so the AUC is the one of
# the score. p is not calibrated, but p ≥ 0.5 ⇔ score ≥ 0, the SVM decision rule.
def dropout_probabilities(model, x):
    score = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-score))                                  # p = σ(score)


# Train one alpha with SGD and early stopping on validation AUC; return the best epoch's model.
def train_sgd(alpha, seed, train, validation):
    # min_w,b  (1/n)·Σ_i max(0, 1 − y_i·(w·x_i + b)) + (α/2)·‖w‖²   (hinge = linear SVM)
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=alpha, random_state=seed)
    rng = np.random.default_rng(seed)                                    # seed -> order of the train rows
    best = {"auc": -1.0, "epoch": 0, "model": None}
    stale = 0          # epochs in a row without a better validation AUC
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(len(train["y"]))                         # new shuffle every epoch
        model.partial_fit(train["x"][order], train["y"][order], classes=np.array([0, 1]))
        validation_auc = roc_auc_score(validation["y"], dropout_probabilities(model, validation["x"]))
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
        for index in range(len(data["y"])):
            writer.writerow((split_name, int(data["enroll_id"][index]), int(data["y"][index]),
                             f"{probability[index]:.8f}"))


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
    train = splits["train"]
    validation = splits["validation"]

    best = {"auc": -1.0}
    for alpha in ALPHA_GRID:
        result = train_sgd(alpha, seed, train, validation)
        log("svm", f"seed {seed} alpha={alpha:g}: val_auc={result['auc']:.4f} "
                  f"(best epoch {result['epoch']} of {result['epochs_run']})")
        if result["auc"] > best["auc"]:                                  # α* = argmax_α AUC_val
            best = result
            best["alpha"] = alpha

    row = {}
    row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["model"] = "SVM"
    row["seed"] = seed
    row["alpha"] = best["alpha"]
    row["best_epoch"] = best["epoch"]
    row["epochs_run"] = best["epochs_run"]
    for split_name, prefix in (("validation", "val"), ("test", "test")):
        data = splits[split_name]
        probability = dropout_probabilities(best["model"], data["x"])
        scores = metrics(data["y"], probability)
        write_predictions(output_dir / f"seed{seed}_{split_name}.csv", split_name, data, probability)
        texts = []
        for name in METRICS:
            row[f"{prefix}_{name}"] = scores[name]
            texts.append(f"{name}={scores[name]:.4f}")
        log(split_name, f"svm seed {seed} alpha={best['alpha']:g}: " + ", ".join(texts))
    row["minutes"] = (time.perf_counter() - started_at) / 60
    append_result(output_dir / "results.csv", row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear SVM trained by SGD on X59, alpha and epoch on validation AUC.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--x-dir", default=X_DIR, help="folder with x59_s{seed}.npz")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="folder for results.csv and predictions")
    arguments = parser.parse_args()

    for seed in arguments.seeds:
        splits = load_splits(arguments.x_dir, seed)
        counts = []
        for name in SPLITS:
            counts.append(f"{name}={len(splits[name]['y']):,}")
        log("setup", f"seed {seed}: X59, " + ", ".join(counts))
        run_one_seed(seed, splits, Path(arguments.output_dir))
    log("done", f"results and predictions in {arguments.output_dir}")
