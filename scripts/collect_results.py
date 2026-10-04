# Collect the train/test reports of one scripts/run_all.sh run into results.csv.
#
#   python scripts/collect_results.py result/<dd-mm-yyyy_HH-MM>
#
# Reads <run>/manifest.tsv (one row per seed), finds the report paths that
# 6_train.py printed in each seed's log ("report=..._train.json" and
# "report=..._test.json"), copies those reports and the matching
# *_probs.npz files into <run>/reports/ and the selected checkpoint into
# <run>/checkpoints/, and writes <run>/results.csv: one row
# per seed, then mean and std (sample std, n - 1) over the seeds that finished.
# Reports written before a metric existed leave its cell empty.

import csv
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

METRICS = ("auc", "auprc", "f1", "precision", "recall",
           "macro_f1", "f1_negative", "auprc_negative", "f1_at_0.5", "accuracy")
# Learned family weights (--family-weights) at the best epoch.
FAMILY_WEIGHTS = ("w_course", "w_object", "w_behavioral", "w_self_loop", "w_user")
# Mean per-hyperedge weight α_e of each family (--edge-weights) at the best epoch.
EDGE_WEIGHTS = ("alpha_course", "alpha_object", "alpha_behavioral", "alpha_user")
REPORT_PATTERN = re.compile(r"report=(.+_(train|test)\.json)\s*$")


def report_paths(log_path):
    paths = {}
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = REPORT_PATTERN.search(line)
        if match:
            paths[match.group(2)] = Path(match.group(1))
    return paths


def seed_row(entry, run_dir):
    row = {
        "seed": entry["seed"],
        "status": entry["status"],
        "duration_sec": entry["duration_sec"],
    }
    paths = report_paths(entry["log"]) if Path(entry["log"]).exists() else {}
    for kind, path in paths.items():
        if path.exists():
            shutil.copy2(path, run_dir / "reports" / path.name)
            run_name = path.name.removesuffix(f"_{kind}.json")
            for extra in [*path.parent.glob(f"{run_name}_*_probs.npz"),
                          *path.parent.glob(f"{run_name}_edge_weights.csv")]:
                shutil.copy2(extra, run_dir / "reports" / extra.name)

    train_path = paths.get("train")
    if train_path is not None and train_path.exists():
        train = json.loads(train_path.read_text(encoding="utf-8"))
        checkpoint = Path(train["checkpoint"])
        if checkpoint.exists():
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            shutil.copy2(checkpoint, run_dir / "checkpoints" / checkpoint.name)
        row["run_name"] = train_path.name.removesuffix("_train.json")
        row["best_epoch"] = train["best_epoch"]
        row["epochs_run"] = len(train["history"])
        row["threshold"] = train.get("threshold", "")
        for name in METRICS:
            row[f"val_{name}"] = (train["best_validation"] or {}).get(name, "")
        best = next((record for record in train["history"] if record["epoch"] == train["best_epoch"]), {})
        for name in FAMILY_WEIGHTS + EDGE_WEIGHTS:
            row[name] = best.get(name, "")

    test_path = paths.get("test")
    if test_path is not None and test_path.exists():
        test = json.loads(test_path.read_text(encoding="utf-8"))
        for name in METRICS:
            row[f"test_{name}"] = test["test"].get(name, "")
    elif row["status"] == "ok":
        row["status"] = "no test report"
    return row


def summary_rows(rows, columns):
    finished = [row for row in rows if row.get("test_auc", "") != ""]
    mean_row = {"seed": "mean", "status": f"n={len(finished)}"}
    std_row = {"seed": "std", "status": f"n={len(finished)}"}
    for column in columns:
        if column.startswith(("val_", "test_", "w_", "alpha_")) or column in ("threshold", "duration_sec"):
            values = [float(row[column]) for row in finished if row.get(column, "") != ""]
            if values:
                mean_row[column] = statistics.mean(values)
                std_row[column] = statistics.stdev(values) if len(values) > 1 else 0.0
    return [mean_row, std_row]


def main(run_dir):
    run_dir = Path(run_dir)
    (run_dir / "reports").mkdir(exist_ok=True)
    with open(run_dir / "manifest.tsv", encoding="utf-8") as manifest:
        entries = list(csv.DictReader(manifest, delimiter="\t"))

    rows = [seed_row(entry, run_dir) for entry in entries]
    columns = (
        ["seed", "run_name", "status", "best_epoch", "epochs_run", "threshold"]
        + [f"val_{name}" for name in METRICS]
        + [f"test_{name}" for name in METRICS]
        + list(FAMILY_WEIGHTS)
        + list(EDGE_WEIGHTS)
        + ["duration_sec"]
    )
    rows += summary_rows(rows, columns)

    output_path = run_dir / "results.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: f"{value:.4f}" if isinstance(value, float) else value
                for name, value in row.items()
            })

    print(f"Saved {output_path}")
    for row in rows:
        print(f"  seed={row['seed']:<6} status={row['status']:<10} "
              f"val_auc={row.get('val_auc', '')!s:<8.8} test_auc={row.get('test_auc', '')!s:<8.8}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/collect_results.py result/<run folder>")
    main(sys.argv[1])
