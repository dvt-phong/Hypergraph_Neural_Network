# 4. Build the train hypergraph H0 (Course, Object, Behavioral, User hyperedges)
#    and the local graph of every validation/test target.
# Tham khảo từ project/bài báo:
# - HGNN, AAAI 2019 (Feng et al.): https://doi.org/10.1609/aaai.v33i01.33013558
#   Code: https://github.com/iMoonLab/HGNN
# - SIG-Net, ACM SAC 2024: https://doi.org/10.1145/3605098.3636002
#   Code: https://github.com/Noverse0/SIG-Net
# - CA-TFHN, ICONIP 2023: https://doi.org/10.1007/978-981-99-8184-7_31
#   Code: https://github.com/codeds27/CA-TFHN

import argparse
import time
from collections import defaultdict
from datetime import date
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

config = import_module("0_config")
preprocess_module = import_module("2_preprocess")
feature_module = import_module("3_features")
read_csv = preprocess_module.read_csv
load_nodes = preprocess_module.load_nodes
feature_columns = feature_module.feature_columns

HYPERGRAPH_FILE_NAME = "hypergraph.npz"
COURSE = config.EDGE_FAMILIES.index("course")
OBJECT = config.EDGE_FAMILIES.index("object")
BEHAVIORAL = config.EDGE_FAMILIES.index("behavioral")
USER = config.EDGE_FAMILIES.index("user")
SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")


# Print a message with the time.
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}][hypergraph] {message}", flush=True)


# Course start as a day number, so start days compare as integers.
# Input:  date text, e.g. "2016-11-16 08:00:00".
# Output: int day number.
def start_day(course_start):
    return date.fromisoformat(course_start[:10]).toordinal()


# Course start day of every node.
# Input:  node rows from load_nodes.
# Output: int64 array [N].
def start_days(nodes):
    return np.asarray([start_day(node["course_start"]) for node in nodes], dtype=np.int64)


# Random relabeling of the train nodes for the shuffled-graph control.
# Input:  number of train nodes, seed (None = no shuffle).
# Output: permutation array [count], or None.
def node_permutation(count, shuffle_seed):
    if shuffle_seed is None:
        return None
    return np.random.default_rng(shuffle_seed).permutation(count)


# Input:  "auto", "cpu" or "cuda".
# Output: torch.device ("auto" = cuda when available).
def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(device_name)


# ---------------------------------------------------------------------------
# Building H0 (run once with `python src/4_hypergraph.py`)
# ---------------------------------------------------------------------------

# Nearest train nodes by cosine similarity of the behavior columns (in batches).
# Input:  train X, query X, number of neighbors, exclude_self (True when the
#         queries are the train nodes), device, batch size.
# Output: int64 array [queries, count] of train node ids, most similar first.
def nearest_train_neighbors(
    train_features, query_features, count, *, exclude_self, device, batch_size
):
    behavior = config.BEHAVIOR_FEATURE_SLICE
    train = torch.as_tensor(np.ascontiguousarray(train_features[:, behavior]))
    query = torch.as_tensor(np.ascontiguousarray(query_features[:, behavior]))
    train = F.normalize(train.to(device), dim=1)
    query = F.normalize(query.to(device), dim=1)

    neighbors = np.empty((len(query), count), dtype=np.int64)
    batch_count = (len(query) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(query), batch_size)):
        stop = min(start + batch_size, len(query))
        similarity = query[start:stop] @ train.T
        if exclude_self:
            # A train node must not be its own neighbor.
            rows = torch.arange(stop - start, device=device)
            similarity[rows, rows + start] = -torch.inf
        neighbors[start:stop] = similarity.topk(count, dim=1).indices.cpu().numpy()
        if batch_number % 25 == 0 or stop == len(query):
            log(f"  kNN {stop:,}/{len(query):,} queries (batch {batch_number + 1}/{batch_count})")
    return neighbors


# Objects (video, problem, forum) used by each node.
# Input:  path of a split CSV.
# Output: yields (node_id, "course|type|object_id").
def read_object_events(events_path):
    for row in read_csv(events_path):
        object_type = config.OBJECT_ACTIONS.get(row["action"])
        object_id = row["object_id"].strip()
        if object_type and object_id.lower() not in config.MISSING_VALUES:
            yield int(row["node_id"]), f"{row['course_id']}|{object_type}|{object_id}"


# User hyperedges of the train split.
#   "any":      one hyperedge per learner with all their enrollments.
#   "temporal": one hyperedge per enrollment (anchor) + the learner's enrollments
#               in courses that started no later.
# Input:  train node rows, user rule.
# Output: list of (USER, key, member node ids).
def user_hyperedges(train_nodes, user_rule):
    by_user = defaultdict(list)
    for node in train_nodes:
        by_user[node["user_id"]].append(int(node["node_id"]))
    hyperedges = []
    for user_id in sorted(by_user, key=int):
        members = by_user[user_id]
        if user_rule == "any":
            hyperedges.append((USER, f"user|{user_id}", members))
            continue
        for anchor in members:
            start = train_nodes[anchor]["course_start"]
            earlier = [m for m in members if m != anchor and train_nodes[m]["course_start"] <= start]
            hyperedges.append((USER, f"user|{user_id}|{anchor}", [anchor] + earlier))
    return hyperedges


# Build H0: Course, Object, Behavioral (anchor + k nearest) and User hyperedges.
# Input:  train node rows, train.csv path, train kNN lists, k, user rule.
# Output: dict node_ids, edge_ids (one entry per membership, sorted by edge),
#         edge_family, edge_keys (one entry per hyperedge).
def build_train_hyperedges(train_nodes, train_events_path, neighbors, k, user_rule):
    course_hyperedge_members = defaultdict(list)
    for node in train_nodes:
        course_hyperedge_members[node["course_id"]].append(int(node["node_id"]))

    object_hyperedge_members = defaultdict(set)
    for node_id, object_key in read_object_events(train_events_path):
        object_hyperedge_members[object_key].add(node_id)

    hyperedges = []  # (family, key, members)
    for course_id in sorted(course_hyperedge_members):
        hyperedges.append((COURSE, course_id, course_hyperedge_members[course_id]))
    for object_key in sorted(object_hyperedge_members):
        hyperedges.append((OBJECT, object_key, sorted(object_hyperedge_members[object_key])))
    for anchor in range(len(train_nodes)):
        members = [anchor] + neighbors[anchor, :k].tolist()
        hyperedges.append((BEHAVIORAL, str(anchor), members))
    hyperedges.extend(user_hyperedges(train_nodes, user_rule))

    # Drop single-member hyperedges; the self-loop already covers them.
    hyperedges = [edge for edge in hyperedges if len(edge[2]) >= 2]

    node_ids = []
    edge_ids = []
    for edge_id, (_, _, members) in enumerate(hyperedges):
        node_ids.extend(members)
        edge_ids.extend([edge_id] * len(members))

    return {
        "node_ids": np.asarray(node_ids, dtype=np.int64),
        "edge_ids": np.asarray(edge_ids, dtype=np.int64),
        "edge_family": np.asarray([edge[0] for edge in hyperedges], dtype=np.int64),
        "edge_keys": np.asarray([edge[1] for edge in hyperedges]),
    }


# Run step 4: compute the kNN lists, build H0 and save them in one bundle.
# Input:  output_dir, k (Behavioral size), k_max (k..k_max-1 = HSL candidates),
#         device, batch size, user rule, bundle name, reuse_neighbors (bundle to
#         copy the kNN lists from instead of recomputing them).
# Output: none (writes output_dir/hypergraph_file).
def build_hypergraph(
    output_dir=config.PROCESSED, *, k=10, k_max=20, device_name="auto", batch_size=1024,
    user_rule="any", hypergraph_file=HYPERGRAPH_FILE_NAME, reuse_neighbors=None,
):
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    if not 0 < k < k_max:
        raise ValueError("Require 0 < k < k_max (neighbors k..k_max are HSL candidates)")
    if user_rule not in config.USER_RULES:
        raise ValueError(f"user_rule must be one of {config.USER_RULES}")

    train_nodes = load_nodes(output_dir / "train.csv")
    train_features = np.load(output_dir / "train" / "X.npy")
    if len(train_nodes) != len(train_features):
        raise ValueError("train.csv and train/X.npy do not align")
    device = resolve_device(device_name)
    log(f"train nodes={len(train_nodes):,}, k={k}, k_max={k_max}, user_rule={user_rule}, device={device}")

    neighbors = {}
    if reuse_neighbors is not None:
        with np.load(output_dir / reuse_neighbors) as old:
            for split_name in ("train", "validation", "test"):
                neighbors[split_name] = old[f"{split_name}_neighbors"]
        if neighbors["train"].shape != (len(train_nodes), k_max):
            raise ValueError(f"{reuse_neighbors} has neighbors of shape {neighbors['train'].shape}, "
                             f"expected ({len(train_nodes)}, {k_max})")
        log(f"reusing kNN neighbors from {reuse_neighbors}")
    else:
        log("train -> train neighbors")
        neighbors["train"] = nearest_train_neighbors(
            train_features, train_features, k_max,
            exclude_self=True, device=device, batch_size=batch_size,
        )
        for split_name in ("validation", "test"):
            log(f"{split_name} -> train neighbors")
            neighbors[split_name] = nearest_train_neighbors(
                train_features, np.load(output_dir / split_name / "X.npy"), k_max,
                exclude_self=False, device=device, batch_size=batch_size,
            )

    log("grouping train events into hyperedges")
    h0 = build_train_hyperedges(train_nodes, output_dir / "train.csv", neighbors["train"], k, user_rule)

    bundle_path = output_dir / hypergraph_file
    np.savez_compressed(
        bundle_path,
        **h0,
        train_neighbors=neighbors["train"],
        validation_neighbors=neighbors["validation"],
        test_neighbors=neighbors["test"],
        k=np.asarray(k),
        user_rule=np.asarray(user_rule),
    )
    family_counts = {
        name: int(np.count_nonzero(h0["edge_family"] == family))
        for family, name in enumerate(config.EDGE_FAMILIES[:SELF_LOOP])
    }
    log(
        f"saved {bundle_path}: hyperedges={family_counts}, "
        f"memberships={len(h0['node_ids']):,}, "
        f"elapsed={time.perf_counter() - started_at:.1f}s"
    )


# ---------------------------------------------------------------------------
# Loading graphs for training and evaluation
# ---------------------------------------------------------------------------

# Add one self-loop hyperedge per node.
# Input:  graph dict (num_nodes, node_ids, edge_ids, edge_family, ...).
# Output: new graph dict with the self-loops appended last.
def add_self_loops(graph):
    nodes = np.arange(graph["num_nodes"], dtype=np.int64)
    first_self_loop = len(graph["edge_family"])
    return {
        **graph,
        "node_ids": np.concatenate([graph["node_ids"], nodes]),
        "edge_ids": np.concatenate([graph["edge_ids"], first_self_loop + nodes]),
        "edge_family": np.concatenate([
            graph["edge_family"], np.full(len(nodes), SELF_LOOP, dtype=np.int64)
        ]),
    }


# Input:  family names, e.g. ["course", "object"].
# Output: their indices in EDGE_FAMILIES (self_loop left out).
def family_ids(families):
    return [config.EDGE_FAMILIES.index(name) for name in families if name != "self_loop"]


# Keep only the hyperedges of the given families and renumber them.
# Input:  graph dict, family names (none left = MLP).
# Output: new graph dict.
def select_families(graph, families):
    keep_edge = np.isin(graph["edge_family"], family_ids(families))
    new_edge_id = np.cumsum(keep_edge) - 1
    keep_membership = keep_edge[graph["edge_ids"]]
    keep_candidate = keep_edge[graph["candidate_edge_ids"]]
    if "edge_keys" in graph:
        graph = {**graph, "edge_keys": graph["edge_keys"][keep_edge]}
    return {
        **graph,
        "node_ids": graph["node_ids"][keep_membership],
        "edge_ids": new_edge_id[graph["edge_ids"][keep_membership]],
        "edge_family": graph["edge_family"][keep_edge],
        "candidate_edge_ids": new_edge_id[graph["candidate_edge_ids"][keep_candidate]],
        "candidate_node_ids": graph["candidate_node_ids"][keep_candidate],
    }


# Load the train split for training.
# Input:  output_dir, feature set, families, bundle name, shuffle seed.
# Output: dict features [N, D], labels [N], graph (H0 + self-loops, plus the HSL
#         candidates k..k_max-1 of every Behavioral hyperedge), edge_keys.
def load_train_graph(
    output_dir=config.PROCESSED, *, feature_set="full", families=config.GRAPH_FAMILIES,
    hypergraph_file=HYPERGRAPH_FILE_NAME, shuffle_seed=None,
):
    output_dir = Path(output_dir)
    with np.load(output_dir / hypergraph_file) as bundle:
        if USER in family_ids(families) and "user_rule" not in bundle.files:
            raise ValueError(
                f"{hypergraph_file} has no User hyperedges. Rebuild it with 4_hypergraph.py "
                "--user-rule, or leave user out of --families."
            )
        k = int(bundle["k"])
        edge_family = bundle["edge_family"]
        behavioral_edges = np.flatnonzero(edge_family == BEHAVIORAL)
        anchors = bundle["edge_keys"][behavioral_edges].astype(np.int64)
        graph = {
            "node_ids": bundle["node_ids"],
            "edge_ids": bundle["edge_ids"],
            "edge_family": edge_family,
            "edge_keys": bundle["edge_keys"],
            "candidate_edge_ids": behavioral_edges,
            "candidate_node_ids": bundle["train_neighbors"][anchors, k:],
        }

    graph = select_families(graph, families)
    # Hyperedge names, only used by the per-hyperedge weight table.
    edge_keys = graph.pop("edge_keys")
    nodes = load_nodes(output_dir / "train.csv")
    features = np.load(output_dir / "train" / "X.npy")[:, feature_columns(feature_set)]
    graph["num_nodes"] = len(nodes)
    graph["node_start"] = start_days(nodes)
    permutation = node_permutation(len(nodes), shuffle_seed)
    if permutation is not None:
        graph["node_ids"] = permutation[graph["node_ids"]]
        graph["candidate_node_ids"] = permutation[graph["candidate_node_ids"]]
    return {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "labels": np.asarray([int(node["label"]) for node in nodes], dtype=np.float32),
        "graph": add_self_loops(graph),
        "edge_keys": edge_keys,
    }


# Load a validation/test split: everything build_local_graph needs.
# Input:  output_dir, split name, feature set, families, bundle name, shuffle seed.
# Output: dict with target nodes and features, train features, train members of
#         every Course/Object hyperedge, kNN lists, train enrollments per user.
def load_evaluation_split(
    output_dir=config.PROCESSED, *, split_name, feature_set="full", families=config.GRAPH_FAMILIES,
    hypergraph_file=HYPERGRAPH_FILE_NAME, shuffle_seed=None,
):
    output_dir = Path(output_dir)
    columns = feature_columns(feature_set)
    with np.load(output_dir / hypergraph_file) as bundle:
        node_ids = bundle["node_ids"]
        edge_family = bundle["edge_family"]
        edge_keys = bundle["edge_keys"]
        # edge_ids is sorted, so the members of edge e are node_ids[start[e]:start[e + 1]].
        start = np.searchsorted(bundle["edge_ids"], np.arange(len(edge_family) + 1))
        neighbors = bundle[f"{split_name}_neighbors"]
        k = int(bundle["k"])
        # Old bundles have no User hyperedges.
        user_rule = str(bundle["user_rule"]) if "user_rule" in bundle.files else None

    # Train members of every Course and Object hyperedge, looked up by key.
    members_by_key = {}
    for edge_id in np.flatnonzero(np.isin(edge_family, [COURSE, OBJECT])):
        members_by_key[str(edge_keys[edge_id])] = node_ids[start[edge_id]:start[edge_id + 1]]

    train_nodes = load_nodes(output_dir / "train.csv")
    # Train enrollments of every learner, with their course start, for User hyperedges.
    train_by_user = {}
    if user_rule is not None:
        grouped = defaultdict(list)
        for node in train_nodes:
            grouped[node["user_id"]].append((int(node["node_id"]), node["course_start"]))
        for user_id, rows in grouped.items():
            train_by_user[user_id] = (np.asarray([r[0] for r in rows], dtype=np.int64),
                                      np.asarray([r[1] for r in rows]))

    # Same π as load_train_graph: relabel train members after choosing them.
    permutation = node_permutation(len(train_nodes), shuffle_seed)
    if permutation is not None:
        members_by_key = {key: permutation[members] for key, members in members_by_key.items()}
        neighbors = permutation[neighbors]
        train_by_user = {user_id: (permutation[ids], starts)
                         for user_id, (ids, starts) in train_by_user.items()}

    object_keys = defaultdict(set)
    for node_id, object_key in read_object_events(output_dir / f"{split_name}.csv"):
        object_keys[node_id].add(object_key)

    train_features = np.load(output_dir / "train" / "X.npy")[:, columns]
    target_features = np.load(output_dir / split_name / "X.npy")[:, columns]
    return {
        "split_name": split_name,
        "nodes": load_nodes(output_dir / f"{split_name}.csv"),
        "features": np.ascontiguousarray(target_features, dtype=np.float32),
        "train_features": np.ascontiguousarray(train_features, dtype=np.float32),
        "train_start": start_days(train_nodes),
        "members_by_key": members_by_key,
        "object_keys": object_keys,
        "neighbors": neighbors,
        "k": k,
        "families": set(family_ids(families)),
        "user_rule": user_rule,
        "train_by_user": train_by_user,
    }


# Local graph of one validation/test target: the target (local node 0) joins its
# Course, Object, User and Behavioral hyperedges, whose other members are train nodes.
# Input:  output of load_evaluation_split, target id.
# Output: dict features [n, D], graph, label.
def build_local_graph(split_data, target_id):
    target = split_data["nodes"][target_id]
    neighbors = split_data["neighbors"][target_id]
    k = split_data["k"]
    families = split_data["families"]

    hyperedges = []  # (family, train members)
    course_members = split_data["members_by_key"].get(target["course_id"])
    if COURSE in families and course_members is not None:
        hyperedges.append((COURSE, course_members))
    if OBJECT in families:
        for object_key in sorted(split_data["object_keys"].get(target_id, ())):
            object_members = split_data["members_by_key"].get(object_key)
            if object_members is not None:
                hyperedges.append((OBJECT, object_members))
    same_user = split_data["train_by_user"].get(target["user_id"])
    if USER in families and same_user is not None:
        user_members, course_starts = same_user
        if split_data["user_rule"] == "temporal":
            user_members = user_members[course_starts <= target["course_start"]]
        if len(user_members):
            hyperedges.append((USER, user_members))
    # Behavioral stays last: its index is the ΔH candidate hyperedge below.
    if BEHAVIORAL in families:
        hyperedges.append((BEHAVIORAL, neighbors[:k]))
        candidates = neighbors[k:]
    else:
        candidates = neighbors[:0]

    train_ids = np.unique(np.concatenate([members for _, members in hyperedges] + [candidates]))

    def local_ids(ids):
        return np.searchsorted(train_ids, ids) + 1

    node_ids = [np.zeros(0, dtype=np.int64)]
    edge_ids = [np.zeros(0, dtype=np.int64)]
    for edge_id, (_, members) in enumerate(hyperedges):
        node_ids.append(np.concatenate([[0], local_ids(members)]))
        edge_ids.append(np.full(len(members) + 1, edge_id, dtype=np.int64))

    # ΔH candidates belong to the Behavioral hyperedge, which is added last.
    if BEHAVIORAL in families:
        candidate_edge_ids = np.asarray([len(hyperedges) - 1])
        candidate_node_ids = local_ids(candidates)[None, :]
    else:
        candidate_edge_ids = np.zeros(0, dtype=np.int64)
        candidate_node_ids = np.zeros((0, len(neighbors) - k), dtype=np.int64)
    graph = add_self_loops({
        "num_nodes": len(train_ids) + 1,
        "node_start": np.concatenate([[start_day(target["course_start"])],
                                      split_data["train_start"][train_ids]]).astype(np.int64),
        "node_ids": np.concatenate(node_ids).astype(np.int64),
        "edge_ids": np.concatenate(edge_ids),
        "edge_family": np.asarray([family for family, _ in hyperedges], dtype=np.int64),
        "candidate_edge_ids": candidate_edge_ids,
        "candidate_node_ids": candidate_node_ids,
    })
    features = np.concatenate([
        split_data["features"][target_id:target_id + 1],
        split_data["train_features"][train_ids],
    ])
    return {"features": features, "graph": graph, "label": int(target["label"])}


# Put several local graphs side by side in one batch; they stay disconnected.
# Input:  list of outputs of build_local_graph.
# Output: (features, merged graph, row of each target, labels).
def merge_local_graphs(local_graphs):
    parts = defaultdict(list)
    node_offset = 0
    edge_offset = 0
    target_rows = []
    for local_graph in local_graphs:
        graph = local_graph["graph"]
        parts["node_start"].append(graph["node_start"])
        parts["node_ids"].append(graph["node_ids"] + node_offset)
        parts["edge_ids"].append(graph["edge_ids"] + edge_offset)
        parts["edge_family"].append(graph["edge_family"])
        parts["candidate_edge_ids"].append(graph["candidate_edge_ids"] + edge_offset)
        parts["candidate_node_ids"].append(graph["candidate_node_ids"] + node_offset)
        target_rows.append(node_offset)
        node_offset += graph["num_nodes"]
        edge_offset += len(graph["edge_family"])

    merged = {name: np.concatenate(arrays) for name, arrays in parts.items()}
    merged["num_nodes"] = node_offset
    features = np.concatenate([local_graph["features"] for local_graph in local_graphs])
    labels = np.asarray([local_graph["label"] for local_graph in local_graphs])
    return features, merged, np.asarray(target_rows), labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config.HYPERGRAPH_CLI_DESCRIPTION)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--neighbor-batch-size", type=int, default=1024)
    parser.add_argument("--user-rule", choices=config.USER_RULES, default="any")
    parser.add_argument("--hypergraph-file", default=HYPERGRAPH_FILE_NAME,
                        help="bundle name inside --output-dir, e.g. hypergraph_temporal.npz")
    parser.add_argument("--reuse-neighbors", metavar="BUNDLE",
                        help="take the kNN lists from this existing bundle instead of recomputing")
    arguments = parser.parse_args()
    build_hypergraph(
        arguments.output_dir,
        k=arguments.k,
        k_max=arguments.k_max,
        device_name=arguments.device,
        batch_size=arguments.neighbor_batch_size,
        user_rule=arguments.user_rule,
        hypergraph_file=arguments.hypergraph_file,
        reuse_neighbors=arguments.reuse_neighbors,
    )
