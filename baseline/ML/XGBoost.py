# Baseline - XGBoost (gradient boosted trees)
#
#   python baseline/ML/XGBoost.py
#   python baseline/ML/XGBoost.py --x-dir data/processed/oulad --x-name x55 --output-dir outputs/oulad/baselines/XGBoost
#
# Input: X59 of the main model M (data/processed/simple/x59_s{seed}.npz, written by
# src/3_features.py): same features and same train / validation / test as M for every seed.
# Seed s: the split of seed s and the row / column subsampling seed s. max_depth and the number of
# trees are chosen on validation AUC.
# Metrics: AUC, precision, recall and F1 of the dropout class at threshold 0.5 (as M).

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
X_DIR = ROOT / "data" / "processed" / "simple"
OUTPUT_DIR = ROOT / "outputs" / "baselines" / "XGBoost"
SPLITS = ("train", "validation", "test")
SEEDS = (1, 11, 111, 1111, 11111)
MAX_DEPTH_GRID = (3, 4, 6, 8, 10)
LEARNING_RATE = 0.05
MAX_ROUNDS = 3000                            # upper bound on the number of trees
PATIENCE = 50                                # rounds without a better validation AUC
SUBSAMPLE = 0.8                              # share of train rows per tree
COLSAMPLE = 0.8                              # share of the 59 features per tree
THRESHOLD = 0.5                              # ŷ = 1[p ≥ 0.5], as config.THRESHOLD of M
METRICS = ("auc", "precision", "recall", "f1")
RESULT_COLUMNS = (
    "time", "model", "seed", "max_depth", "best_round", "rounds_run",
    "val_auc", "val_precision", "val_recall", "val_f1",
    "test_auc", "test_precision", "test_recall", "test_f1", "minutes",
)


# Print a message with the time and a scope, e.g. [xgb].
def log(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


# X, label and enroll_id of train / validation / test of one seed
def load_splits(x_dir, x_name, seed):
    data = np.load(Path(x_dir) / f"{x_name}_s{seed}.npz")
    splits = {}
    for name in SPLITS:
        rows = data["split"] == name
        splits[name] = {
            "x": data["X"][rows].astype(np.float32),
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


# Dropout probability of every row with the trees up to the best round: p = σ(Σ_k f_k(x))
def dropout_probabilities(model, x):
    return model.predict_proba(x)[:, 1]


# Train one max_depth with early stopping on validation AUC; the model keeps the best round.
def train_xgboost(max_depth, seed, train, validation):
    # min_f  Σ_i log(1 + exp(−y_i·F(x_i))) + Σ_k Ω(f_k),  F = Σ_k η·f_k,  f_k = tree of depth ≤ max_depth
    model = XGBClassifier(
        n_estimators=MAX_ROUNDS,
        learning_rate=LEARNING_RATE,
        max_depth=max_depth,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=PATIENCE,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train["x"], train["y"], eval_set=[(validation["x"], validation["y"])], verbose=False)
    rounds_run = len(model.evals_result()["validation_0"]["auc"])
    return {
        "auc": float(model.best_score),                                  # best = argmax_round AUC_val
        "round": int(model.best_iteration) + 1,
        "rounds_run": rounds_run,
        "model": model,
    }


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


# Choose max_depth on validation AUC, score validation and test once, save predictions and one result row.
def run_one_seed(seed, splits, output_dir):
    started_at = time.perf_counter()
    train = splits["train"]
    validation = splits["validation"]

    best = {"auc": -1.0}
    for max_depth in MAX_DEPTH_GRID:
        result = train_xgboost(max_depth, seed, train, validation)
        log("xgb", f"seed {seed} max_depth={max_depth}: val_auc={result['auc']:.4f} "
                   f"(best round {result['round']} of {result['rounds_run']})")
        if result["auc"] > best["auc"]:                                  # depth* = argmax_depth AUC_val
            best = result
            best["max_depth"] = max_depth

    row = {}
    row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["model"] = "XGBoost"
    row["seed"] = seed
    row["max_depth"] = best["max_depth"]
    row["best_round"] = best["round"]
    row["rounds_run"] = best["rounds_run"]
    for split_name, prefix in (("validation", "val"), ("test", "test")):
        data = splits[split_name]
        probability = dropout_probabilities(best["model"], data["x"])
        scores = metrics(data["y"], probability)
        write_predictions(output_dir / f"seed{seed}_{split_name}.csv", split_name, data, probability)
        texts = []
        for name in METRICS:
            row[f"{prefix}_{name}"] = scores[name]
            texts.append(f"{name}={scores[name]:.4f}")
        log(split_name, f"xgb seed {seed} max_depth={best['max_depth']}: " + ", ".join(texts))
    row["minutes"] = (time.perf_counter() - started_at) / 60
    append_result(output_dir / "results.csv", row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost on X59, max_depth and number of trees on validation AUC.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--x-dir", default=X_DIR, help="folder with x59_s{seed}.npz")
    parser.add_argument("--x-name", default="x59", help="x59 (XuetangX) or x55 (OULAD)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="folder for results.csv and predictions")
    arguments = parser.parse_args()

    for seed in arguments.seeds:
        splits = load_splits(arguments.x_dir, arguments.x_name, seed)
        counts = []
        for name in SPLITS:
            counts.append(f"{name}={len(splits[name]['y']):,}")
        log("setup", f"seed {seed}: {arguments.x_name}, " + ", ".join(counts))
        run_one_seed(seed, splits, Path(arguments.output_dir))
    log("done", f"results and predictions in {arguments.output_dir}")
