# 4. Build the train hypergraph H0 and the local graphs used for validation/test.
#
# A hypergraph is stored as two parallel arrays with one entry per membership:
#     node_ids[m] is a member of hyperedge edge_ids[m]
# These are the non-zero cells of the incidence matrix H (rows = nodes,
# columns = hyperedges). edge_family[e] says which relation built hyperedge e:
#   course      enrollments of the same course
#   object      enrollments that used the same video / problem / forum object
#   behavioral  one enrollment (the anchor) + its k most similar train enrollments
#   user        enrollments of the same learner in different courses, by --user-rule:
#                 any       one hyperedge per learner with all their train enrollments
#                 temporal  one hyperedge per enrollment (the anchor) + the learner's
#                           enrollments in courses that started no later, so only
#                           behavior already observable at prediction time is used
#               Only features are propagated, never the labels of other enrollments.
#   self_loop   one per node, added when a graph is loaded (HSL, Eq. 9)
#
# Leakage rule: validation/test enrollments are never added to H0. Each target
# gets its own local graph = the target + the train enrollments it is linked to.
# Tham khảo: HGNN (Feng et al., 2019), SIG-Net, CA-TFHN; xem docs/references.md.

import argparse
import time
from collections import defaultdict
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


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}][hypergraph] {message}", flush=True)


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(device_name)


# ---------------------------------------------------------------------------
# Building H0 (run once with `python src/4_hypergraph.py`)
# ---------------------------------------------------------------------------

# Exact cosine kNN on the behavior columns. Queries are processed in batches so
# the full query x train similarity matrix is never stored.
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


# Yield (node_id, object_key) for every video, problem, or forum event.
# The course is part of the key so equal object IDs in two courses stay apart.
def read_object_events(events_path):
    for row in read_csv(events_path):
        object_type = config.OBJECT_ACTIONS.get(row["action"])
        object_id = row["object_id"].strip()
        if object_type and object_id.lower() not in config.MISSING_VALUES:
            yield int(row["node_id"]), f"{row['course_id']}|{object_type}|{object_id}"


# User hyperedges of the train split: (USER, key, members), see the file header.
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


# Group train enrollments into Course, Object, Behavioral, and User hyperedges.
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
    # Added last, so Course, Object, and Behavioral keep the ids they had before.
    hyperedges.extend(user_hyperedges(train_nodes, user_rule))

    # A hyperedge with a single member links nobody; the self-loop covers it.
    hyperedges = [edge for edge in hyperedges if len(edge[2]) >= 2]

    # Memberships are written hyperedge by hyperedge, so edge_ids is sorted.
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


# reuse_neighbors: take the kNN lists from an existing bundle with the same
# k_max instead of recomputing them (the slow part), e.g. to add User hyperedges.
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

# Append one self-loop hyperedge per node.
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


def family_ids(families):
    return [config.EDGE_FAMILIES.index(name) for name in families if name != "self_loop"]


# Keep only the hyperedges of the given families and renumber them, so no
# hyperedge is left empty. With no family left, only self-loops remain and the
# HGNN encoder reduces to an MLP.
def select_families(graph, families):
    keep_edge = np.isin(graph["edge_family"], family_ids(families))
    new_edge_id = np.cumsum(keep_edge) - 1
    keep_membership = keep_edge[graph["edge_ids"]]
    keep_candidate = keep_edge[graph["candidate_edge_ids"]]
    return {
        **graph,
        "node_ids": graph["node_ids"][keep_membership],
        "edge_ids": new_edge_id[graph["edge_ids"][keep_membership]],
        "edge_family": graph["edge_family"][keep_edge],
        "candidate_edge_ids": new_edge_id[graph["candidate_edge_ids"][keep_candidate]],
        "candidate_node_ids": graph["candidate_node_ids"][keep_candidate],
    }


# X, labels, and H0 of the train split.
#
# HSL candidates (ΔH): the Behavioral hyperedge of anchor a may recruit the
# anchor's next neighbors a[k], ..., a[k_max-1] that are not yet members.
def load_train_graph(
    output_dir=config.PROCESSED, *, feature_set="full", families=config.GRAPH_FAMILIES,
    hypergraph_file=HYPERGRAPH_FILE_NAME,
):
    output_dir = Path(output_dir)
    with np.load(output_dir / hypergraph_file) as bundle:
        k = int(bundle["k"])
        edge_family = bundle["edge_family"]
        behavioral_edges = np.flatnonzero(edge_family == BEHAVIORAL)
        anchors = bundle["edge_keys"][behavioral_edges].astype(np.int64)
        graph = {
            "node_ids": bundle["node_ids"],
            "edge_ids": bundle["edge_ids"],
            "edge_family": edge_family,
            "candidate_edge_ids": behavioral_edges,
            "candidate_node_ids": bundle["train_neighbors"][anchors, k:],
        }

    graph = select_families(graph, families)
    nodes = load_nodes(output_dir / "train.csv")
    features = np.load(output_dir / "train" / "X.npy")[:, feature_columns(feature_set)]
    graph["num_nodes"] = len(nodes)
    return {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "labels": np.asarray([int(node["label"]) for node in nodes], dtype=np.float32),
        "graph": add_self_loops(graph),
    }


# Everything needed to build the local graph of any validation/test target.
def load_evaluation_split(
    output_dir=config.PROCESSED, *, split_name, feature_set="full", families=config.GRAPH_FAMILIES,
    hypergraph_file=HYPERGRAPH_FILE_NAME,
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
        # Bundles built before User hyperedges existed have none, in training either.
        user_rule = str(bundle["user_rule"]) if "user_rule" in bundle.files else None

    # Train members of every Course and Object hyperedge, looked up by key.
    members_by_key = {}
    for edge_id in np.flatnonzero(np.isin(edge_family, [COURSE, OBJECT])):
        members_by_key[str(edge_keys[edge_id])] = node_ids[start[edge_id]:start[edge_id + 1]]

    # Train enrollments of every learner, with their course start, for User hyperedges.
    train_by_user = {}
    if user_rule is not None:
        grouped = defaultdict(list)
        for node in load_nodes(output_dir / "train.csv"):
            grouped[node["user_id"]].append((int(node["node_id"]), node["course_start"]))
        for user_id, rows in grouped.items():
            train_by_user[user_id] = (np.asarray([r[0] for r in rows], dtype=np.int64),
                                      np.asarray([r[1] for r in rows]))

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
        "members_by_key": members_by_key,
        "object_keys": object_keys,
        "neighbors": neighbors,
        "k": k,
        "families": set(family_ids(families)),
        "user_rule": user_rule,
        "train_by_user": train_by_user,
    }


# Local graph of one validation/test target. Local node 0 is the target; the
# other nodes are train enrollments. The target joins its course hyperedge, the
# object hyperedges of objects it used, a User hyperedge with the learner's train
# enrollments (only those in courses that started no later, for rule
# "temporal"), and its own Behavioral hyperedge.
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


# Put several local graphs side by side in one graph. No hyperedge links two
# local graphs, so the targets never see each other.
def merge_local_graphs(local_graphs):
    parts = defaultdict(list)
    node_offset = 0
    edge_offset = 0
    target_rows = []
    for local_graph in local_graphs:
        graph = local_graph["graph"]
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
