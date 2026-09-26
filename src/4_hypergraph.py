# 4. Build the train hypergraph H0 and the local graphs used for validation/test.
#
# A hypergraph is stored as two parallel arrays with one entry per membership:
#     node_ids[m] is a member of hyperedge edge_ids[m]
# These are the non-zero cells of the incidence matrix H (rows = nodes,
# columns = hyperedges). edge_family[e] says which relation built hyperedge e:
#   course      enrollments of the same course
#   object      enrollments that used the same video / problem / forum object
#   behavioral  one enrollment (the anchor) + its k most similar train enrollments
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


# Group train enrollments into Course, Object, and Behavioral hyperedges.
def build_train_hyperedges(train_nodes, train_events_path, neighbors, k):
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


def build_hypergraph(
    output_dir=config.PROCESSED, *, k=10, k_max=20, device_name="auto", batch_size=1024
):
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    if not 0 < k < k_max:
        raise ValueError("Require 0 < k < k_max (neighbors k..k_max are HSL candidates)")

    train_nodes = load_nodes(output_dir / "train.csv")
    train_features = np.load(output_dir / "train" / "X.npy")
    if len(train_nodes) != len(train_features):
        raise ValueError("train.csv and train/X.npy do not align")
    device = resolve_device(device_name)
    log(f"train nodes={len(train_nodes):,}, k={k}, k_max={k_max}, device={device}")

    neighbors = {}
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
    h0 = build_train_hyperedges(train_nodes, output_dir / "train.csv", neighbors["train"], k)

    bundle_path = output_dir / HYPERGRAPH_FILE_NAME
    np.savez_compressed(
        bundle_path,
        **h0,
        train_neighbors=neighbors["train"],
        validation_neighbors=neighbors["validation"],
        test_neighbors=neighbors["test"],
        k=np.asarray(k),
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


# X, labels, and H0 of the train split.
#
# HSL candidates (ΔH): the Behavioral hyperedge of anchor a may recruit the
# anchor's next neighbors a[k], ..., a[k_max-1] that are not yet members.
def load_train_graph(output_dir=config.PROCESSED, *, feature_set="full"):
    output_dir = Path(output_dir)
    with np.load(output_dir / HYPERGRAPH_FILE_NAME) as bundle:
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

    nodes = load_nodes(output_dir / "train.csv")
    features = np.load(output_dir / "train" / "X.npy")[:, feature_columns(feature_set)]
    graph["num_nodes"] = len(nodes)
    return {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "labels": np.asarray([int(node["label"]) for node in nodes], dtype=np.float32),
        "graph": add_self_loops(graph),
    }


# Everything needed to build the local graph of any validation/test target.
def load_evaluation_split(output_dir=config.PROCESSED, *, split_name, feature_set="full"):
    output_dir = Path(output_dir)
    columns = feature_columns(feature_set)
    with np.load(output_dir / HYPERGRAPH_FILE_NAME) as bundle:
        node_ids = bundle["node_ids"]
        edge_family = bundle["edge_family"]
        edge_keys = bundle["edge_keys"]
        # edge_ids is sorted, so the members of edge e are node_ids[start[e]:start[e + 1]].
        start = np.searchsorted(bundle["edge_ids"], np.arange(len(edge_family) + 1))
        neighbors = bundle[f"{split_name}_neighbors"]
        k = int(bundle["k"])

    # Train members of every Course and Object hyperedge, looked up by key.
    members_by_key = {}
    for edge_id in np.flatnonzero(edge_family != BEHAVIORAL):
        members_by_key[str(edge_keys[edge_id])] = node_ids[start[edge_id]:start[edge_id + 1]]

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
    }


# Local graph of one validation/test target. Local node 0 is the target; the
# other nodes are train enrollments. The target joins its course hyperedge, the
# object hyperedges of objects it used, and its own Behavioral hyperedge.
def build_local_graph(split_data, target_id):
    target = split_data["nodes"][target_id]
    neighbors = split_data["neighbors"][target_id]
    k = split_data["k"]

    hyperedges = []  # (family, train members)
    course_members = split_data["members_by_key"].get(target["course_id"])
    if course_members is not None:
        hyperedges.append((COURSE, course_members))
    for object_key in sorted(split_data["object_keys"].get(target_id, ())):
        object_members = split_data["members_by_key"].get(object_key)
        if object_members is not None:
            hyperedges.append((OBJECT, object_members))
    hyperedges.append((BEHAVIORAL, neighbors[:k]))
    candidates = neighbors[k:]

    train_ids = np.unique(np.concatenate([members for _, members in hyperedges] + [candidates]))

    def local_ids(ids):
        return np.searchsorted(train_ids, ids) + 1

    node_ids = []
    edge_ids = []
    for edge_id, (_, members) in enumerate(hyperedges):
        node_ids.append(np.concatenate([[0], local_ids(members)]))
        edge_ids.append(np.full(len(members) + 1, edge_id, dtype=np.int64))

    graph = add_self_loops({
        "num_nodes": len(train_ids) + 1,
        "node_ids": np.concatenate(node_ids).astype(np.int64),
        "edge_ids": np.concatenate(edge_ids),
        "edge_family": np.asarray([family for family, _ in hyperedges], dtype=np.int64),
        "candidate_edge_ids": np.asarray([len(hyperedges) - 1]),  # the Behavioral edge
        "candidate_node_ids": local_ids(candidates)[None, :],
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
    arguments = parser.parse_args()
    build_hypergraph(
        arguments.output_dir,
        k=arguments.k,
        k_max=arguments.k_max,
        device_name=arguments.device,
        batch_size=arguments.neighbor_batch_size,
    )
