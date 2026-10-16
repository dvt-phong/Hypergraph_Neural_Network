# Baseline table (docs/PLAN_X59.md)
#
#   python baseline/summary.py                    # fixed threshold 0.5, as src/9_train.py of M (default)
#   python baseline/summary.py --threshold val    # threshold chosen on validation (max F1)
#
# Every metric is computed from the saved predictions outputs/baselines/<model>/seed<s>_{validation,
# test}.csv (split, enroll_id, label, prob), so no model is rerun:
#   AUC                           threshold-free;
#   precision, recall, F1         of the dropout class (label 1) at threshold t: either a fixed t
#                                 (default 0.5) or t chosen on the validation predictions of the same
#                                 seed (the t with the best validation F1, then applied unchanged to test).
# Prints test mean ± std (ddof = 1) over the seeds and writes outputs/baselines/summary.csv (one row
# per model) and summary_per_seed.csv. Checks that every seed<s>_test.csv holds exactly the test
# enrollments of x59_s<s>.npz.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("LR", "SVM", "CFIN", "CA-TFHN", "MST-GCN")
SEEDS = (1, 11, 111, 1111, 11111)
METRICS = ("auc", "precision", "recall", "f1")


# Problems of the test prediction file of one seed (other enrollments, NaN prob).
def check_test_file(predictions, x59_path):
    data = np.load(x59_path)
    expected = set(data["enroll_id"][data["split"] == "test"].tolist())
    problems = []
    if set(predictions["enroll_id"].tolist()) != expected or len(predictions) != len(expected):
        problems.append(f"{len(predictions):,} rows, expected the {len(expected):,} test enrollments")
    if predictions["prob"].isna().any():
        problems.append("NaN prob")
    return problems


# Threshold with the best F1 of the dropout class on the validation predictions.
def best_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    # F1 = 2·P·R/(P + R) at every candidate threshold (the last P/R point has no threshold)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


# AUC and precision / recall / F1 of the dropout class at threshold t.
def metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold                               # ŷ = 1[p ≥ t]
    # P = TP/(TP + FP), R = TP/(TP + FN), F1 = 2·P·R/(P + R)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=[1], zero_division=0)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),             # AUC = P(p_dropout > p_non-dropout)
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean ± std of the baselines over the seeds.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs" / "baselines")
    parser.add_argument("--x-dir", type=Path, default=ROOT / "data" / "processed" / "simple")
    parser.add_argument("--threshold", default="0.5",
                        help="threshold for precision/recall/F1: a number (default 0.5, as M) or 'val' "
                             "(best validation F1 per seed)")
    arguments = parser.parse_args()
    if arguments.threshold == "val":
        rule = "best validation F1"
    else:
        rule = f"fixed {float(arguments.threshold):g}"
    print(f"Threshold for precision / recall / F1: {rule}")

    rows = []
    seed_rows = []
    for model in MODELS:
        model_dir = arguments.results_dir / model
        scores = []
        problems = []
        for seed in SEEDS:
            test_path = model_dir / f"seed{seed}_test.csv"
            validation_path = model_dir / f"seed{seed}_validation.csv"
            if not test_path.is_file() or not validation_path.is_file():
                problems.append(f"seed {seed}: no predictions")
                continue
            test = pd.read_csv(test_path)
            validation = pd.read_csv(validation_path)
            for problem in check_test_file(test, arguments.x_dir / f"x59_s{seed}.npz"):
                problems.append(f"seed {seed}: {problem}")
            if arguments.threshold == "val":
                threshold = best_f1_threshold(validation["label"].to_numpy(), validation["prob"].to_numpy())
            else:
                threshold = float(arguments.threshold)
            score = metrics(test["label"].to_numpy(), test["prob"].to_numpy(), threshold)
            scores.append(score)
            seed_row = {"model": model, "seed": seed, "threshold": threshold}
            seed_row.update(score)
            seed_rows.append(seed_row)
        if not scores:
            print(f"{model:8s} no predictions yet")
            continue

        row = {"model": model, "seeds": len(scores), "threshold_rule": rule}
        texts = []
        for name in METRICS:
            values = []
            for score in scores:
                values.append(score[name])
            # mean = (1/n)·Σ x,  std = √(Σ(x − mean)²/(n − 1))  over the n seeds
            row[f"test_{name}_mean"] = float(np.mean(values))
            if len(values) > 1:
                row[f"test_{name}_std"] = float(np.std(values, ddof=1))
            else:
                row[f"test_{name}_std"] = 0.0
            texts.append(f"{name.upper()} {row[f'test_{name}_mean']:.4f} +- {row[f'test_{name}_std']:.4f}")
        if problems:
            row["check"] = "; ".join(problems)
        else:
            row["check"] = "ok"
        rows.append(row)
        print(f"{model:8s} " + "   ".join(texts) + f"   ({row['seeds']} seeds; {row['check']})")

    if rows:
        pd.DataFrame(rows).to_csv(arguments.results_dir / "summary.csv", index=False, float_format="%.6f")
        pd.DataFrame(seed_rows).to_csv(arguments.results_dir / "summary_per_seed.csv", index=False, float_format="%.6f")
        print(f"Saved {arguments.results_dir / 'summary.csv'} and summary_per_seed.csv")
