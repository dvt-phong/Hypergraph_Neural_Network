# Baseline table (docs/PLAN_BASELINE_v3.md)
#
#   python baseline/summary.py                    # threshold chosen on validation (max F1), default
#   python baseline/summary.py --threshold 0.5    # fixed threshold 0.5, as src/9_train.py of M
#
# Every metric is computed from the saved predictions outputs/baselines/<model>/seed<s>_{validation,
# test}.csv (split, enroll_id, label, prob), one learner split per seed, so no model is rerun:
#   AUC, AUPRC                    threshold-free;
#   precision, recall, F1         of the dropout class (label 1) at threshold t: either t chosen on
#                                 the validation predictions of the same split (the t with the best
#                                 validation F1, then applied unchanged to test) or a fixed t.
# Prints test mean ± std (ddof = 1) over the splits and writes outputs/baselines/summary.csv (one row
# per model) and summary_per_split.csv. Checks that every seed<s>_test.csv holds exactly the test
# enrollments of splits/learner_s<s>.csv.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("LR", "SVM", "CFIN", "CA-TFHN", "MST-GCN", "SIG-Net")
SEEDS = (1, 11, 111, 1111, 11111)
METRICS = ("auc", "auprc", "precision", "recall", "f1")


# Problems of the test prediction file of one split (other enrollments, NaN prob).
def check_test_file(predictions, split_path):
    split = pd.read_csv(split_path)
    expected = set(split.loc[split["split"] == "test", "enroll_id"])
    problems = []
    if set(predictions["enroll_id"]) != expected or len(predictions) != len(expected):
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


# AUC, AUPRC and precision / recall / F1 of the dropout class at threshold t.
def metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold                               # ŷ = 1[p ≥ t]
    # P = TP/(TP + FP), R = TP/(TP + FN), F1 = 2·P·R/(P + R)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=[1], zero_division=0)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),             # AUC = P(p_dropout > p_non-dropout)
        "auprc": float(average_precision_score(labels, probabilities)), # AUPRC = Σ_n (R_n − R_{n−1})·P_n
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean ± std of the baselines over the learner splits.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs" / "baselines")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "baseline_raw" / "splits")
    parser.add_argument("--threshold", type=float, default=None,
                        help="fixed threshold for precision/recall/F1; default: best validation F1 per split")
    arguments = parser.parse_args()
    rule = "best validation F1" if arguments.threshold is None else f"fixed {arguments.threshold:g}"
    print(f"Threshold for precision / recall / F1: {rule}")

    rows, split_rows = [], []
    for model in MODELS:
        model_dir = arguments.results_dir / model
        scores, problems = [], []
        for seed in SEEDS:
            test_path = model_dir / f"seed{seed}_test.csv"
            validation_path = model_dir / f"seed{seed}_validation.csv"
            if not test_path.is_file() or not validation_path.is_file():
                problems.append(f"seed {seed}: no predictions")
                continue
            test = pd.read_csv(test_path)
            validation = pd.read_csv(validation_path)
            problems += [f"seed {seed}: {p}" for p in check_test_file(test, arguments.split_dir / f"learner_s{seed}.csv")]
            if arguments.threshold is None:
                threshold = best_f1_threshold(validation["label"].to_numpy(), validation["prob"].to_numpy())
            else:
                threshold = arguments.threshold
            score = metrics(test["label"].to_numpy(), test["prob"].to_numpy(), threshold)
            scores.append(score)
            split_rows.append(dict(model=model, seed=seed, threshold=threshold, **score))
        if not scores:
            print(f"{model:8s} no predictions yet")
            continue
        row = {"model": model, "splits": len(scores), "threshold_rule": rule}
        for name in METRICS:
            values = np.array([score[name] for score in scores])
            row[f"test_{name}_mean"] = float(np.mean(values))
            row[f"test_{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["check"] = "ok" if not problems else "; ".join(problems)
        rows.append(row)
        print(f"{model:8s} " + "   ".join(
            f"{name.upper()} {row[f'test_{name}_mean']:.4f} +- {row[f'test_{name}_std']:.4f}" for name in METRICS)
            + f"   ({row['splits']} splits; {row['check']})")
    if rows:
        pd.DataFrame(rows).to_csv(arguments.results_dir / "summary.csv", index=False, float_format="%.6f")
        pd.DataFrame(split_rows).to_csv(arguments.results_dir / "summary_per_split.csv", index=False, float_format="%.6f")
        print(f"Saved {arguments.results_dir / 'summary.csv'} and summary_per_split.csv")
