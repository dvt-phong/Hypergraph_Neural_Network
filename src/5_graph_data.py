# 5. Load the graph data

from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
hypergraph_module = import_module("4_hypergraph")
feature_columns = import_module("3_features").feature_columns
read_split = hypergraph_module.read_split
HYPERGRAPH_FILE = hypergraph_module.HYPERGRAPH_FILE
SELF_LOOP = hypergraph_module.SELF_LOOP


# Load graph data for train
def load_train_graph(output_dir=config.PROCESSED):
    output_dir = Path(output_dir)
    with np.load(output_dir / HYPERGRAPH_FILE) as bundle:
        labels = bundle["train_labels"]
        graph = {
            "num_nodes": len(labels),
            "node_ids": bundle["node_ids"],
            "edge_ids": bundle["edge_ids"],
            "edge_family": bundle["edge_family"],
        }
    features = np.load(output_dir / "train" / "X.npy")
    if len(features) != len(labels):
        raise ValueError("train/X.npy and hypergraph.npz do not align; rerun 4_hypergraph.py")
    return {"features": features, "labels": labels, "graph": graph}


# Load graph data for validation and test
def load_targets(split_name, output_dir=config.PROCESSED, limit=0):
    output_dir = Path(output_dir)
    with np.load(output_dir / HYPERGRAPH_FILE) as bundle:
        edge_keys = bundle["edge_keys"]
        train_users = bundle["train_users"]

    # edge_of[key] = H0 hyperedge id of that key
    edge_of = {}
    for edge_id in range(len(edge_keys)):
        edge_of[str(edge_keys[edge_id])] = edge_id

    # train_by_user[learner] = train nodes of that learner
    train_by_user = {}
    train_user_list = train_users.tolist()
    for node_id in range(len(train_user_list)):
        user_id = train_user_list[node_id]
        if user_id not in train_by_user:
            train_by_user[user_id] = []
        train_by_user[user_id].append(node_id)

    nodes, objects = read_split(output_dir / f"{split_name}.csv")
    features = np.load(output_dir / split_name / "X.npy")
    if len(features) != len(nodes):
        raise ValueError(f"{split_name}.csv and {split_name}/X.npy do not align")
    if limit != 0:
        count = min(limit, len(nodes))
    else:
        count = len(nodes)

    # The H0 hyperedges of target t are h0_edges[h0_ptr[t]:h0_ptr[t + 1]].
    h0_ptr = [0]
    h0_edges = []
    single_user = np.full(count, -1, dtype=np.int64)
    for target in range(count):
        node = nodes[target]
        user_id = int(node["user_id"])

        # Course
        h0_edges.append(edge_of[node["course_id"]])

        # Object: only objects that are in H0
        if target in objects:
            for key in sorted(objects[target]):
                if key in edge_of:
                    h0_edges.append(edge_of[key])

        # User ("any"): join the learner's H0 hyperedge, or make {t, u} with the one train node u
        if user_id in train_by_user:
            same_user = train_by_user[user_id]
        else:
            same_user = []
        if len(same_user) >= 2:
            h0_edges.append(edge_of[f"user|{user_id}"])
        elif len(same_user) == 1:
            single_user[target] = same_user[0]

        h0_ptr.append(len(h0_edges))

    labels = []
    for target in range(count):
        labels.append(int(nodes[target]["label"]))

    return {
        "split_name": split_name,
        "features": features[:count],
        "labels": np.asarray(labels, dtype=np.float32),
        "h0_ptr": np.asarray(h0_ptr, dtype=np.int64),
        "h0_edges": np.asarray(h0_edges, dtype=np.int64),
        "single_user": single_user,
    }


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

    graph = data["train"]["graph"]
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
    train_graph = {
        "num_nodes": graph["num_nodes"],
        "node_ids": kept_node_ids,
        "edge_ids": new_edge_id[kept_edge_ids],
        "edge_family": graph["edge_family"][keep_edge],
    }
    if "self_loop" in families:
        train_graph = add_self_loops(train_graph)
    train_graph["self_loop"] = "self_loop" in families                        # read by 6_hgnn.py

    train_features = data["train"]["features"][:, columns]
    result = {}
    result["train"] = {
        "features": np.ascontiguousarray(train_features, dtype=np.float32),
        "labels": data["train"]["labels"],
        "graph": train_graph,
    }

    keep_list = keep_edge.tolist()
    for split in ("validation", "test"):
        targets = data[split]
        count = len(targets["labels"])
        old_ptr = targets["h0_ptr"].tolist()
        old_edges = targets["h0_edges"].tolist()

        # Keep only the hyperedges of kept families, with their new ids.
        # h0_rows[k] = target of entry k (used by 9_train.predict).
        h0_ptr = [0]
        h0_edges = []
        h0_rows = []
        for target in range(count):
            for k in range(old_ptr[target], old_ptr[target + 1]):
                edge = old_edges[k]
                if keep_list[edge]:
                    h0_edges.append(new_edge_list[edge])
                    h0_rows.append(target)
            h0_ptr.append(len(h0_edges))

        if "user" in families:
            single_user = targets["single_user"]
        else:
            single_user = np.full(count, -1, dtype=np.int64)              # no User family: no {t, u}

        target_features = targets["features"][:, columns]
        result[split] = {
            "split_name": split,
            "features": np.ascontiguousarray(target_features, dtype=np.float32),
            "labels": targets["labels"],
            "h0_ptr": np.asarray(h0_ptr, dtype=np.int64),
            "h0_edges": np.asarray(h0_edges, dtype=np.int64),
            "h0_rows": np.asarray(h0_rows, dtype=np.int64),
            "single_user": single_user,
        }
    return result
