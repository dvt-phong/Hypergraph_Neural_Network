# Does the graph or the node's own features (MLP) carry the prediction?
#
#   python scripts/analyze_contribution.py result/<MLP run> result/<graph run> [--processed DIR]
#
# Both folders hold reports/*_seed_<n>_test_probs.npz from scripts/run_all.sh.
# For every seed found in both, on the test split:
#   1. Branch split (graph run with --skip-connection): the logit is
#      logit_graph + logit_self + bias. Reports the AUC of each part alone
#      (= the model with the other branch held constant) and their spread.
#   2. Subgroups: AUC of the MLP and of the graph run, their difference and the
#      paired DeLong p-value, for
#        activity   events in the 35-day window: none, then quartiles Q1..Q4
#        history    the learner has a train enrollment in a course that started
#                   no later (what a temporal User hyperedge can link), or not
# Writes <graph run>/contribution.csv with one row per (seed, group).

import argparse
import csv
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
config = import_module("0_config")
load_nodes = import_module("2_preprocess").load_nodes
delong_module = import_module("delong")


# Events per test target in the window, and whether the learner has an earlier
# (or same-day) train enrollment.
def target_groups(processed):
    events = Counter()
    with open(processed / "test.csv", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["action"]:
                events[int(row["node_id"])] += 1
    test_nodes = load_nodes(processed / "test.csv")
    earliest_train_start = {}
    for node in load_nodes(processed / "train.csv"):
        start = node["course_start"]
        if node["user_id"] not in earliest_train_start or start < earliest_train_start[node["user_id"]]:
            earliest_train_start[node["user_id"]] = start

    counts = np.asarray([events[node_id] for node_id in range(len(test_nodes))])
    active = counts > 0
    edges = np.quantile(counts[active], [0.25, 0.5, 0.75])
    quartile = np.char.add("Q", (1 + np.searchsorted(edges, counts, side="right")).astype(str))
    activity = np.where(active, quartile, "none")
    history = np.asarray([
        "has earlier course" if node["user_id"] in earliest_train_start
        and earliest_train_start[node["user_id"]] <= node["course_start"] else "no earlier course"
        for node in test_nodes
    ])
    groups = {"all": np.ones(len(test_nodes), dtype=bool)}
    for name in ["none", "Q1", "Q2", "Q3", "Q4"]:
        groups[f"activity {name}"] = activity == name
    for name in ["no earlier course", "has earlier course"]:
        groups[f"history: {name}"] = history == name
    print("activity quartile edges (events): " + ", ".join(f"{edge:.0f}" for edge in edges))
    return groups


def main():
    parser = argparse.ArgumentParser(description="MLP vs graph contribution on the test split")
    parser.add_argument("mlp_run", type=Path)
    parser.add_argument("graph_run", type=Path)
    parser.add_argument("--processed", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()

    mlp_files = delong_module.probability_files(arguments.mlp_run)
    graph_files = delong_module.probability_files(arguments.graph_run)
    seeds = sorted(set(mlp_files) & set(graph_files))
    if not seeds:
        sys.exit("No seed is present in both folders")
    groups = target_groups(arguments.processed)

    rows = []
    for seed in seeds:
        with np.load(mlp_files[seed]) as data:
            labels, mlp = data["labels"].astype(int), data["probabilities"]
        with np.load(graph_files[seed]) as data:
            graph_labels, model = data["labels"].astype(int), data["probabilities"]
            parts = {name: data[name] for name in ("logit_graph", "logit_self") if name in data.files}
        if not np.array_equal(labels, graph_labels) or len(labels) != len(groups["all"]):
            sys.exit(f"seed {seed}: the two runs or test.csv hold different test targets")

        print(f"\n== seed {seed}")
        if parts:
            own, graph = parts["logit_self"], parts["logit_graph"]
            print(f"  branches: AUC full {roc_auc_score(labels, model):.4f} | "
                  f"self branch alone {roc_auc_score(labels, own):.4f} | "
                  f"graph branch alone {roc_auc_score(labels, graph):.4f} | "
                  f"std logit self {np.std(own):.3f}, graph {np.std(graph):.3f}")
        else:
            print("  (no logit split saved: graph run without --skip-connection or older run)")

        print(f"  {'group':<30}{'n':>8}{'dropout':>9}{'AUC MLP':>10}{'AUC graph':>11}{'delta':>9}{'p':>11}")
        for name, mask in groups.items():
            group_labels = labels[mask]
            if mask.sum() < 50 or len(set(group_labels)) < 2:
                continue
            auc_mlp, auc_graph, _, p = delong_module.delong(group_labels, mlp[mask], model[mask])
            row = {"seed": seed, "group": name, "n": int(mask.sum()),
                   "dropout_rate": float(group_labels.mean()), "auc_mlp": auc_mlp,
                   "auc_graph": auc_graph, "delta": auc_graph - auc_mlp, "p_delong": p}
            if parts:
                row["auc_self_branch"] = roc_auc_score(group_labels, parts["logit_self"][mask])
                row["auc_graph_branch"] = roc_auc_score(group_labels, parts["logit_graph"][mask])
            rows.append(row)
            print(f"  {name:<30}{row['n']:>8,}{row['dropout_rate']:>9.3f}{auc_mlp:>10.4f}"
                  f"{auc_graph:>11.4f}{row['delta']:>+9.4f}{p:>11.1e}")

    output = arguments.graph_run / "contribution.csv"
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with open(output, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
