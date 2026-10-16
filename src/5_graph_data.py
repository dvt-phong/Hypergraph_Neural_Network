# 5. Load the graph data of one seed

from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
hypergraph_module = import_module("4_hypergraph")
x59_path = import_module("3_features").x59_path
HYPERGRAPH_FILE = hypergraph_module.HYPERGRAPH_FILE
SELF_LOOP = hypergraph_module.SELF_LOOP


# Load the hypergraph H and the X59, labels and split of one seed
def load_graph(seed, output_dir=config.PROCESSED):
    output_dir = Path(output_dir)
    with np.load(output_dir / HYPERGRAPH_FILE) as bundle:
        graph = {
            "num_nodes": int(bundle["num_nodes"]),
            "node_ids": bundle["node_ids"],
            "edge_ids": bundle["edge_ids"],
            "edge_family": bundle["edge_family"],
        }
    with np.load(x59_path(output_dir, seed)) as bundle:
        features = bundle["X"]
        labels = bundle["label"].astype(np.float32)
        split = bundle["split"]
    if len(features) != graph["num_nodes"]:
        raise ValueError("x59 and hypergraph.npz do not align; rerun 3_features.py and 4_hypergraph.py")
    return {"features": features, "labels": labels, "split": split, "graph": graph}


# Add one self-loop hyperedge
def add_self_loops(graph):
    nodes = np.arange(graph["num_nodes"], dtype=np.int64)
    first_self_loop = len(graph["edge_family"])
    self_loop_family = np.full(len(nodes), SELF_LOOP, dtype=np.int64)

    new_graph = dict(graph)
    new_graph["node_ids"] = np.concatenate([graph["node_ids"], nodes])
    new_graph["edge_ids"] = np.concatenate([graph["edge_ids"], first_self_loop + nodes])
    new_graph["edge_family"] = np.concatenate([graph["edge_family"], self_loop_family])
    return new_graph


# Keep the hyperedge families of one scenario
def apply_scenario(data, scenario):
    families = scenario["families"]

    kept_families = []
    for name in families:
        if name != "self_loop":
            kept_families.append(config.EDGE_FAMILIES.index(name))

    graph = data["graph"]
    keep_edge = np.isin(graph["edge_family"], kept_families)

    # New hyperedge id of every kept hyperedge (0, 1, 2, ...), -1 for a removed one.
    new_edge_list = []
    next_id = 0
    for keep in keep_edge.tolist():
        if keep:
            new_edge_list.append(next_id)
            next_id += 1
        else:
            new_edge_list.append(-1)
    new_edge_id = np.asarray(new_edge_list, dtype=np.int64)

    keep_membership = keep_edge[graph["edge_ids"]]
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

    return {
        "features": np.ascontiguousarray(data["features"], dtype=np.float32),
        "labels": data["labels"],
        "graph": scenario_graph,
        "train_index": np.flatnonzero(data["split"] == "train"),             # nodes of the loss
        "validation_index": np.flatnonzero(data["split"] == "validation"),   # nodes of early stopping
        "test_index": np.flatnonzero(data["split"] == "test"),               # nodes scored once at the end
    }
