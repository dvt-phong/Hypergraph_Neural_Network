# 5. Load the graph data of one seed

from datetime import date
from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
hypergraph_module = import_module("4_hypergraph")
x59_path = import_module("3_features").x59_path
read_csv = import_module("2_preprocess").read_csv
HYPERGRAPH_FILE = hypergraph_module.HYPERGRAPH_FILE
SELF_LOOP = hypergraph_module.SELF_LOOP
USER = hypergraph_module.USER


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
        user_id = bundle["user_id"]
        course_id = bundle["course_id"]
    if len(features) != graph["num_nodes"]:
        raise ValueError("x59 and hypergraph.npz do not align; rerun 3_features.py and 4_hypergraph.py")
    return {"features": features, "labels": labels, "split": split, "graph": graph,
            "user_id": user_id, "course_id": course_id}


# Add one self-loop hyperedge
def add_self_loops(graph):
    nodes = np.arange(graph["num_nodes"], dtype=np.int64)
    first_self_loop = len(graph["edge_family"])
    self_loop_family = np.full(len(nodes), SELF_LOOP, dtype=np.int64)

    new_graph = dict(graph)
    new_graph["node_ids"] = np.concatenate([graph["node_ids"], nodes])
    new_graph["edge_ids"] = np.concatenate([graph["edge_ids"], first_self_loop + nodes])
    new_graph["edge_family"] = np.concatenate([graph["edge_family"], self_loop_family])
    if "receives" in graph:
        new_graph["receives"] = np.concatenate([graph["receives"], np.ones(len(nodes), dtype=bool)])
    return new_graph


# Course start of every node, in days since the earliest start (course_info.csv, as 2_preprocess.py)
def course_start_days(course_ids, raw_dir=config.RAW):
    starts = {}
    for row in read_csv(Path(raw_dir) / "course_info.csv"):
        starts[row["course_id"]] = date.fromisoformat(row["start"][:10])
    first = min(starts.values())
    days = []
    for course_id in course_ids.tolist():
        days.append((starts[course_id] - first).days)
    return np.asarray(days, dtype=np.int64)


# Causal User hyperedges (δ = 0, directed): one hyperedge e_v per node v whose learner has another
# enrollment u with start(u) ≤ start(v). Every member of e_v sends (v and those u), only v receives,
# so v never reads the features of a course that starts after its own course: the window of u ends
# at start(u) + 35 ≤ t_obs(v).
def add_causal_user_edges(graph, user_id, start_day):
    order = np.lexsort((start_day, user_id))                             # by learner, then by start
    first_rows = np.flatnonzero(np.r_[True, user_id[order][1:] != user_id[order][:-1]])
    last_rows = np.r_[first_rows[1:], len(order)]

    node_ids = []
    edge_ids = []
    receives = []
    next_edge = len(graph["edge_family"])
    for begin, end in zip(first_rows.tolist(), last_rows.tolist()):
        nodes = order[begin:end]                                         # enrollments of one learner
        if len(nodes) < 2:
            continue
        for v in nodes.tolist():
            senders = nodes[start_day[nodes] <= start_day[v]]            # v itself and the earlier ones
            if len(senders) < 2:                                         # keep |e| ≥ 2
                continue
            node_ids.append(senders)
            edge_ids.append(np.full(len(senders), next_edge, dtype=np.int64))
            receives.append(senders == v)
            next_edge += 1

    new_graph = dict(graph)
    new_graph["node_ids"] = np.concatenate([graph["node_ids"]] + node_ids)
    new_graph["edge_ids"] = np.concatenate([graph["edge_ids"]] + edge_ids)
    new_graph["edge_family"] = np.concatenate(
        [graph["edge_family"], np.full(next_edge - len(graph["edge_family"]), USER, dtype=np.int64)])
    new_graph["receives"] = np.concatenate([np.ones(len(graph["node_ids"]), dtype=bool)] + receives)
    return new_graph


# Shuffled incidence: permute the node ids of the memberships of each family, so every hyperedge
# keeps its size δ(e) and every node keeps its number of hyperedges per family (same d(v)).
def shuffle_members(graph, seed):
    rng = np.random.default_rng(seed)
    membership_family = graph["edge_family"][graph["edge_ids"]]          # family of every (v, e) pair
    node_ids = graph["node_ids"].copy()
    for family in range(len(config.EDGE_FAMILIES)):
        if config.EDGE_FAMILIES[family] == "self_loop":
            continue
        positions = np.flatnonzero(membership_family == family)
        node_ids[positions] = rng.permutation(node_ids[positions])

    new_graph = dict(graph)
    new_graph["node_ids"] = node_ids
    return new_graph


# Keep the hyperedge families of one scenario (shuffled with the seed when the scenario asks)
def apply_scenario(data, scenario, seed):
    families = scenario["families"]

    kept_families = []
    for name in families:
        if name == "self_loop":
            continue
        if name == "user" and scenario["causal_user"]:
            continue                                                     # replaced by the causal ones below
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
    if scenario["shuffle"]:
        scenario_graph = shuffle_members(scenario_graph, seed)
    if "user" in families and scenario["causal_user"]:
        start_day = course_start_days(data["course_id"])
        scenario_graph = add_causal_user_edges(scenario_graph, data["user_id"], start_day)
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
