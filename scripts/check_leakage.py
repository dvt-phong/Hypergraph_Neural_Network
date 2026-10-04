# Check that the graphs a run sees carry no information from after a node's
# 35-day window, on the graphs the model actually uses.
#
#   python scripts/check_leakage.py                                  hypergraph_temporal.npz (M0)
#   python scripts/check_leakage.py --hypergraph hypergraph.npz      rule "any" (M-any): measure the leakage
#
# Checks (a FAIL ends with exit code 1):
#   1. splits     train, validation and test enrollments are disjoint
#   2. train User every User hyperedge of H0 holds one learner; with rule
#                 "temporal" no member started its course after the anchor
#   3. local      local graphs of --targets random validation and test targets
#                 (Course, Object, User): no member of a hyperedge the target
#                 receives from started its course later than the target
#   4. causal     the --causal mask of 5_model.py on the train graph equals the
#                 rule recomputed here with NumPy: node v receives from e only
#                 when no member of e started later than v
# With rule "any", checks 2 and 3 report how much future information reaches
# the nodes instead of failing (that is the known leakage of M-any).
#
# Done by construction elsewhere, so not repeated here: events outside days
# 0-34 are dropped (2_preprocess.py), transforms are fitted on train only
# (3_features.py), validation/test targets are never members of H0 and only
# link to train enrollments (4_hypergraph.py).

import argparse
import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
config = import_module("0_config")
hypergraph_module = import_module("4_hypergraph")
model_module = import_module("5_model")

USER = config.EDGE_FAMILIES.index("user")
FAMILIES = ["course", "object", "user"]
failures = []

# Reading train.csv takes about a minute; every loader below gets the same list.
load_nodes = lru_cache(maxsize=None)(hypergraph_module.load_nodes)
hypergraph_module.load_nodes = load_nodes


def report(name, ok, message, *, fail=True):
    status = "OK  " if ok else ("FAIL" if fail else "INFO")
    print(f"[{status}] {name}: {message}", flush=True)
    if not ok and fail:
        failures.append(name)


def check_splits(output_dir):
    enrollments = {split: {node["enroll_id"] for node in load_nodes(output_dir / f"{split}.csv")}
                   for split in config.SPLITS}
    overlaps = {f"{a}/{b}": len(enrollments[a] & enrollments[b])
                for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    report("1. splits", not any(overlaps.values()),
           f"enrollments train={len(enrollments['train']):,}, validation={len(enrollments['validation']):,}, "
           f"test={len(enrollments['test']):,}; shared {overlaps}")


def check_train_user(output_dir, hypergraph_file, rule):
    train_nodes = load_nodes(output_dir / "train.csv")
    user_of = np.asarray([node["user_id"] for node in train_nodes])
    start_of = np.asarray([node["course_start"] for node in train_nodes])
    with np.load(output_dir / hypergraph_file) as bundle:
        node_ids, edge_ids = bundle["node_ids"], bundle["edge_ids"]
        user_edges = np.flatnonzero(bundle["edge_family"] == USER)
        keys = bundle["edge_keys"][user_edges]
        start = np.searchsorted(edge_ids, np.arange(len(bundle["edge_family"]) + 1))
    mixed_users = later_members = 0
    for edge, key in zip(user_edges, keys):
        members = node_ids[start[edge]:start[edge + 1]]
        if np.any(user_of[members] != str(key).split("|")[1]):
            mixed_users += 1
        # Rule "temporal" keys the anchor; rule "any" has none, so compare with the earliest member.
        parts = str(key).split("|")
        reference = start_of[int(parts[2])] if len(parts) == 3 else min(start_of[members])
        later_members += int(np.any(start_of[members] > reference))
    report("2. train User learner", mixed_users == 0,
           f"{len(user_edges):,} User hyperedges, {mixed_users} mix learners")
    if rule == "temporal":
        report("2. train User temporal", later_members == 0,
               f"{later_members} hyperedges have a member that started after the anchor")
    else:
        report("2. train User any", False,
               f"{later_members:,} of {len(user_edges):,} hyperedges join courses with different starts "
               "(future information for the earlier ones)", fail=False)


def check_local_graphs(output_dir, hypergraph_file, rule, target_count):
    rng = np.random.default_rng(0)
    for split in ("validation", "test"):
        split_data = hypergraph_module.load_evaluation_split(
            output_dir, split_name=split, families=FAMILIES, hypergraph_file=hypergraph_file)
        targets = rng.choice(len(split_data["nodes"]), min(target_count, len(split_data["nodes"])), replace=False)
        with_user = leaking = 0
        later_by_family = {}
        for target in targets:
            graph = hypergraph_module.build_local_graph(split_data, int(target))["graph"]
            node_start = graph["node_start"]
            family = graph["edge_family"][graph["edge_ids"]]
            later = node_start[graph["node_ids"]] > node_start[0]
            # Edges the target (local node 0) belongs to, and which of them hold a later member.
            target_edges = np.unique(graph["edge_ids"][graph["node_ids"] == 0])
            later_edges = np.unique(graph["edge_ids"][later])
            for edge in np.intersect1d(target_edges, later_edges):
                name = config.EDGE_FAMILIES[graph["edge_family"][edge]]
                later_by_family[name] = later_by_family.get(name, 0) + 1
            if np.any(family == USER):
                with_user += 1
                leaking += int(np.any(later & (family == USER)))
        message = (f"{len(targets):,} targets, {with_user:,} with a User hyperedge, {leaking:,} of them "
                   f"reach a later course; hyperedges with a later member by family: {later_by_family or 'none'}")
        if rule == "temporal":
            report(f"3. local {split}", not later_by_family, message)
        else:
            report(f"3. local {split}", False, message, fail=False)


def check_causal(output_dir, hypergraph_file):
    import torch

    data = hypergraph_module.load_train_graph(output_dir, families=FAMILIES, hypergraph_file=hypergraph_file)
    graph = data["graph"]
    # Receive matrix the model uses (one value per membership).
    receive_matrix = model_module.prepare_graph(graph, torch.device("cpu"), causal=True)["H_recv"]
    node_ids, edge_ids = receive_matrix.indices().numpy()
    receive = receive_matrix.values().numpy() > 0

    member_start = graph["node_start"][node_ids]
    latest = np.full(len(graph["edge_family"]), np.iinfo(np.int64).min)
    np.maximum.at(latest, edge_ids, member_start)
    expected = member_start >= latest[edge_ids]
    family = graph["edge_family"][edge_ids]
    cut = {config.EDGE_FAMILIES[f]: int(np.sum(~expected & (family == f))) for f in np.unique(family)}
    report("4. causal", np.array_equal(receive, expected),
           f"{len(receive):,} memberships, {int(np.sum(~receive)):,} may not receive; by family {cut}")


def main(output_dir, hypergraph_file, target_count):
    with np.load(output_dir / hypergraph_file) as bundle:
        rule = str(bundle["user_rule"]) if "user_rule" in bundle.files else None
    print(f"{output_dir / hypergraph_file}: User rule {rule!r}", flush=True)
    if rule is None:
        sys.exit("This bundle has no User hyperedges; build one with src/4_hypergraph.py --user-rule ...")
    check_splits(output_dir)
    check_train_user(output_dir, hypergraph_file, rule)
    check_local_graphs(output_dir, hypergraph_file, rule, target_count)
    check_causal(output_dir, hypergraph_file)
    if failures:
        sys.exit(f"FAILED: {', '.join(failures)}")
    print("All checks passed." if rule == "temporal" else
          "No check failed; the INFO lines measure the future information of rule 'any'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a hypergraph bundle for future information.")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--hypergraph", default="hypergraph_temporal.npz")
    parser.add_argument("--targets", type=int, default=2000, help="random targets per split for check 3")
    arguments = parser.parse_args()
    main(arguments.output_dir, arguments.hypergraph, arguments.targets)
