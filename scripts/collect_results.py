# Collect the train/test reports of one scripts/run_all.sh run into results.csv.
#
#   python scripts/collect_results.py result/<dd-mm-yyyy_HH-MM>
#
# Reads <run>/manifest.tsv (one row per seed), finds the report paths that
# 8_train.py printed in each seed's log ("report=..._train.json" and
# "report=..._test.json"), copies those reports into <run>/reports/, and writes
# <run>/results.csv: one row per seed, then mean and std (sample std, n - 1)
# over the seeds that finished.

import csv
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

METRICS = ("auc", "auprc", "f1", "precision", "recall")
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

    train_path = paths.get("train")
    if train_path is not None and train_path.exists():
        train = json.loads(train_path.read_text(encoding="utf-8"))
        row["run_name"] = train_path.name.removesuffix("_train.json")
        row["best_epoch"] = train["best_epoch"]
        row["epochs_run"] = len(train["history"])
        for name in METRICS:
            row[f"val_{name}"] = (train["best_validation"] or {}).get(name, "")

    test_path = paths.get("test")
    if test_path is not None and test_path.exists():
        test = json.loads(test_path.read_text(encoding="utf-8"))
        for name in METRICS:
            row[f"test_{name}"] = test["test"][name]
    elif row["status"] == "ok":
        row["status"] = "no test report"
    return row


def summary_rows(rows, columns):
    finished = [row for row in rows if row.get("test_auc", "") != ""]
    mean_row = {"seed": "mean", "status": f"n={len(finished)}"}
    std_row = {"seed": "std", "status": f"n={len(finished)}"}
    for column in columns:
        if column.startswith(("val_", "test_")) or column == "duration_sec":
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
        ["seed", "run_name", "status", "best_epoch", "epochs_run"]
        + [f"val_{name}" for name in METRICS]
        + [f"test_{name}" for name in METRICS]
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
