# 5. Load the graph data (one hypergraph over train, validation and test)

from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
hypergraph_module = import_module("4_hypergraph")
feature_columns = import_module("3_features").feature_columns
HYPERGRAPH_FILE = hypergraph_module.HYPERGRAPH_FILE
SELF_LOOP = hypergraph_module.SELF_LOOP


# Load the hypergraph H, the labels, the split of every node and X of the three splits
# stacked in the order of config.SPLITS (the node order of 4_hypergraph.py).
def load_graph(output_dir=config.PROCESSED):
    output_dir = Path(output_dir)
    with np.load(output_dir / HYPERGRAPH_FILE) as bundle:
        labels = bundle["labels"]
        split = bundle["split"]
        graph = {
            "num_nodes": len(labels),
            "node_ids": bundle["node_ids"],
            "edge_ids": bundle["edge_ids"],
            "edge_family": bundle["edge_family"],
        }

    split_features = []
    for split_id in range(len(config.SPLITS)):
        split_name = config.SPLITS[split_id]
        features = np.load(output_dir / split_name / "X.npy")
        if len(features) != np.sum(split == split_id):
            raise ValueError(f"{split_name}/X.npy and hypergraph.npz do not align; rerun 4_hypergraph.py")
        split_features.append(features)
    features = np.concatenate(split_features)                       # X = [X_train; X_validation; X_test]
    return {"features": features, "labels": labels, "split": split, "graph": graph}


# Add one self-loop hyperedge
def add_self_loops(graph):
    nodes = np.arange(graph["num_nodes"], dtype=np.int64)          # 0, 1, ..., N-1
    first_self_loop = len(graph["edge_family"])                    # self-loop of node v gets id E + v
    self_loop_family = np.full(len(nodes), SELF_LOOP, dtype=np.int64)

    new_graph = dict(graph)
    new_graph["node_ids"] = np.concatenate([graph["node_ids"], nodes])
    new_graph["edge_ids"] = np.concatenate([graph["edge_ids"], first_self_loop + nodes])
    new_graph["edge_family"] = np.concatenate([graph["edge_family"], self_loop_family])
    return new_graph


# This function for scenario when experiment
def apply_scenario(data, scenario):
    families = scenario["families"]
    columns = feature_columns(scenario["features"])

    # Family ids of the kept hyperedge families (self-loops are added separately).
    kept_families = []
    for name in families:
        if name != "self_loop":
            kept_families.append(config.EDGE_FAMILIES.index(name))

    graph = data["graph"]
    keep_edge = np.isin(graph["edge_family"], kept_families)                  # hyperedges of kept families

    # Renumber the kept hyperedges 0, 1, 2, ... (E' kept); a dropped hyperedge gets -1.
    new_edge_list = []
    next_id = 0
    for keep in keep_edge.tolist():
        if keep:
            new_edge_list.append(next_id)
            next_id += 1
        else:
            new_edge_list.append(-1)
    new_edge_id = np.asarray(new_edge_list, dtype=np.int64)

    keep_membership = keep_edge[graph["edge_ids"]]                           # membership in a kept hyperedge?
    kept_node_ids = graph["node_ids"][keep_membership]
    kept_edge_ids = graph["edge_ids"][keep_membership]
    scenario_graph = {
        "num_nodes": graph["num_nodes"],
        "node_ids": kept_node_ids,
        "edge_ids": new_edge_id[kept_edge_ids],
        "edge_family": graph["edge_family"][keep_edge],
    }
    if "self_loop" in families:
        scenario_graph = add_self_loops(scenario_graph)

    features = data["features"][:, columns]
    return {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "labels": data["labels"],
        "graph": scenario_graph,
        "train_index": np.flatnonzero(data["split"] == 0),                    # nodes of the loss
        "validation_index": np.flatnonzero(data["split"] == 1),               # nodes of early stopping
        "test_index": np.flatnonzero(data["split"] == 2),                     # nodes scored once at the end
    }
