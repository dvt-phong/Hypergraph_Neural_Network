# 5. Load the graph data a training run needs, then cut it down to one scenario.
#
#    Loading (once per run, all hyperedge families, all 89 columns of X):
#    - train: features X, labels and H0 from 4_hypergraph.py;
#    - validation/test: for every target t, the H0 hyperedges t joins. Targets are
#      never added to H0 and never linked to each other; they only read from train nodes.
#    Scenario (apply_scenario, once per scenario, cheap):
#    - keep only the hyperedge families of the scenario, renumber them 0..E'-1,
#      add the self-loops only when "self_loop" is kept, and keep only the X columns
#      of the scenario's feature set.
#    No model computation here: 6_hgnn.py turns these lists into representations.

from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
hypergraph_module = import_module("4_hypergraph")
feature_columns = import_module("3_features").feature_columns
read_split = hypergraph_module.read_split
HYPERGRAPH_FILE = hypergraph_module.HYPERGRAPH_FILE
SELF_LOOP = hypergraph_module.SELF_LOOP


# ---------------------------------------------------------------------------
# Loading (all families, all columns)
# ---------------------------------------------------------------------------

# Load the train split: features [N, 89], labels [N] and graph = H0 without self-loops
# (num_nodes, node_ids, edge_ids, edge_family).
def load_train_graph(output_dir=config.PROCESSED):
    output_dir = Path(output_dir)
    with np.load(output_dir / HYPERGRAPH_FILE) as bundle:
        labels = bundle["train_labels"]
        graph = {"num_nodes": len(labels), "node_ids": bundle["node_ids"],
                 "edge_ids": bundle["edge_ids"], "edge_family": bundle["edge_family"]}
    features = np.load(output_dir / "train" / "X.npy")
    if len(features) != len(labels):
        raise ValueError("train/X.npy and hypergraph.npz do not align; rerun 4_hypergraph.py")
    return {"features": features, "labels": labels, "graph": graph}


# Load the validation or test targets and the hyperedges each target t joins
# (limit > 0: only the first `limit` targets, for quick checks):
#   Course  the H0 hyperedge of its course (every course has train enrollments)
#   Object  the H0 hyperedge of every object t used in days 0–34, when that object is in
#           H0 (≥ 2 train members). Objects with a single train member are left out, as
#           in H0 (they touch 0.08% of the target memberships).
#   User    {t} ∪ all train enrollments of the same learner (rule "any"):
#           ≥ 2 train enrollments  -> t joins the learner's H0 hyperedge
#           exactly 1              -> a new hyperedge {t, u}; u is stored in single_user
#           none                   -> no User hyperedge
#   Self-loop {t}: handled in 6_hgnn.py when the scenario keeps self-loops.
# Returned arrays: features [T, 89], labels [T],
#   h0_ptr [T + 1], h0_edges: the H0 hyperedges of target t are h0_edges[h0_ptr[t]:h0_ptr[t + 1]]
#   single_user [T]: train node u of the hyperedge {t, u}, or -1
def load_targets(output_dir=config.PROCESSED, *, split_name, limit=0):
    output_dir = Path(output_dir)
    with np.load(output_dir / HYPERGRAPH_FILE) as bundle:
        edge_of = {str(key): edge for edge, key in enumerate(bundle["edge_keys"])}   # key -> H0 hyperedge id
        train_users = bundle["train_users"]
    train_by_user = defaultdict(list)                                                # learner -> train nodes
    for node_id, user_id in enumerate(train_users.tolist()):
        train_by_user[user_id].append(node_id)

    nodes, objects = read_split(output_dir / f"{split_name}.csv")
    features = np.load(output_dir / split_name / "X.npy")
    if len(features) != len(nodes):
        raise ValueError(f"{split_name}.csv and {split_name}/X.npy do not align")
    count = min(limit, len(nodes)) if limit else len(nodes)

    h0_ptr = [0]
    h0_edges = []
    single_user = np.full(count, -1, dtype=np.int64)
    for target in range(count):
        node = nodes[target]
        h0_edges.append(edge_of[node["course_id"]])                                  # Course
        h0_edges.extend(edge_of[key] for key in sorted(objects.get(target, ())) if key in edge_of)   # Object
        same_user = train_by_user.get(int(node["user_id"]), [])                       # User ("any")
        if len(same_user) >= 2:
            h0_edges.append(edge_of[f"user|{int(node['user_id'])}"])
        elif len(same_user) == 1:
            single_user[target] = same_user[0]
        h0_ptr.append(len(h0_edges))

    return {
        "split_name": split_name,
        "features": features[:count],
        "labels": np.asarray([int(node["label"]) for node in nodes[:count]], dtype=np.float32),
        "h0_ptr": np.asarray(h0_ptr, dtype=np.int64),
        "h0_edges": np.asarray(h0_edges, dtype=np.int64),
        "single_user": single_user,
    }


# ---------------------------------------------------------------------------
# One scenario: keep some hyperedge families and some X columns
# ---------------------------------------------------------------------------

# Add one self-loop hyperedge per node after the existing E hyperedges (E + N in total):
#   e_S(v) = {v},  id(e_S(v)) = E + v
def add_self_loops(graph):
    nodes = np.arange(graph["num_nodes"], dtype=np.int64)
    first_self_loop = len(graph["edge_family"])
    return {
        **graph,
        "node_ids": np.concatenate([graph["node_ids"], nodes]),
        "edge_ids": np.concatenate([graph["edge_ids"], first_self_loop + nodes]),
        "edge_family": np.concatenate([graph["edge_family"],
                                       np.full(len(nodes), SELF_LOOP, dtype=np.int64)]),
    }


# Cut the loaded data down to one scenario (config.SCENARIOS); the result has the same shape:
#   train.graph  only the kept H0 hyperedges, renumbered 0..E'-1, plus one self-loop per
#                node when "self_loop" is kept; graph["self_loop"] says which
#   features     only the scenario's X columns (train and targets)
#   targets      h0_edges filtered and renumbered the same way; single_user = -1 when
#                "user" is not kept
def apply_scenario(data, scenario):
    families = scenario["families"]
    columns = feature_columns(scenario["features"])
    kept_families = [config.EDGE_FAMILIES.index(name) for name in families if name != "self_loop"]

    graph = data["train"]["graph"]
    keep_edge = np.isin(graph["edge_family"], kept_families)                  # hyperedges of kept families
    new_edge_id = np.cumsum(keep_edge) - 1                                    # old id -> new id 0..E'-1
    keep_membership = keep_edge[graph["edge_ids"]]
    train_graph = {
        "num_nodes": graph["num_nodes"],
        "node_ids": graph["node_ids"][keep_membership],
        "edge_ids": new_edge_id[graph["edge_ids"][keep_membership]],
        "edge_family": graph["edge_family"][keep_edge],
    }
    if "self_loop" in families:
        train_graph = add_self_loops(train_graph)
    train_graph["self_loop"] = "self_loop" in families                        # read by 6_hgnn.py

    result = {"train": {"features": np.ascontiguousarray(data["train"]["features"][:, columns],
                                                         dtype=np.float32),
                        "labels": data["train"]["labels"], "graph": train_graph}}
    for split in ("validation", "test"):
        targets = data[split]
        count = len(targets["labels"])
        rows = np.repeat(np.arange(count), np.diff(targets["h0_ptr"]))       # target of every entry
        keep = keep_edge[targets["h0_edges"]]                                 # entry of a kept family?
        kept_per_target = np.bincount(rows[keep], minlength=count)
        result[split] = {
            "split_name": split,
            "features": np.ascontiguousarray(targets["features"][:, columns], dtype=np.float32),
            "labels": targets["labels"],
            "h0_ptr": np.concatenate([[0], np.cumsum(kept_per_target)]).astype(np.int64),
            "h0_edges": new_edge_id[targets["h0_edges"][keep]],
            "single_user": targets["single_user"] if "user" in families
                           else np.full(count, -1, dtype=np.int64),
        }
    return result
