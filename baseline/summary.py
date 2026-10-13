# Baseline table (docs/PLAN_BASELINE_v3.md)
#
#   python baseline/summary.py
#
# Reads outputs/baselines/<model>/results.csv (one row per learner split, seed = split seed), prints
# test AUC and AUPRC as mean ± std (ddof = 1) and writes outputs/baselines/summary.csv. Checks that
# every seed<s>_test.csv holds exactly the test enrollments of splits/learner_s<s>.csv.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("LR", "SVM", "CFIN", "CA-TFHN", "MST-GCN", "SIG-Net")
SEEDS = (1, 11, 111, 1111, 11111)


# Problems of the test prediction files of one model (missing file, other enrollments, NaN prob).
def check_predictions(model_dir, split_dir, seeds):
    problems = []
    for seed in seeds:
        path = model_dir / f"seed{seed}_test.csv"
        if not path.is_file():
            problems.append(f"seed {seed}: no {path.name}")
            continue
        predictions = pd.read_csv(path)
        split = pd.read_csv(split_dir / f"learner_s{seed}.csv")
        expected = set(split.loc[split["split"] == "test", "enroll_id"])
        if set(predictions["enroll_id"]) != expected or len(predictions) != len(expected):
            problems.append(f"seed {seed}: {len(predictions):,} rows, expected the {len(expected):,} test enrollments")
        if predictions["prob"].isna().any():
            problems.append(f"seed {seed}: NaN prob")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean ± std of the baselines over the learner splits.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs" / "baselines")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "baseline_raw" / "splits")
    arguments = parser.parse_args()

    rows = []
    for model in MODELS:
        path = arguments.results_dir / model / "results.csv"
        if not path.is_file():
            print(f"{model:8s} no results yet")
            continue
        results = pd.read_csv(path).drop_duplicates("seed", keep="last")     # a rerun of a seed replaces it
        row = {"model": model, "splits": len(results)}
        for name in ("test_auc", "test_auprc"):
            values = results[name].to_numpy()
            row[name + "_mean"] = float(np.mean(values))
            row[name + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        problems = check_predictions(arguments.results_dir / model, arguments.split_dir, results["seed"].tolist())
        missing = sorted(set(SEEDS) - set(results["seed"]))
        if missing:
            problems.append(f"missing seeds {missing}")
        row["check"] = "ok" if not problems else "; ".join(problems)
        rows.append(row)
        print(f"{model:8s} AUC {row['test_auc_mean']:.4f} +- {row['test_auc_std']:.4f}   "
              f"AUPRC {row['test_auprc_mean']:.4f} +- {row['test_auprc_std']:.4f}   "
              f"({row['splits']} splits; {row['check']})")
    if rows:
        pd.DataFrame(rows).to_csv(arguments.results_dir / "summary.csv", index=False, float_format="%.6f")
        print(f"Saved {arguments.results_dir / 'summary.csv'}")
