# 4. Build Course, Object, and Behavioral hyperedges, then create sparse H0.
#
# H0 columns always follow this order:
#   Course hyperedges -> Object hyperedges -> Behavioral hyperedges.

import argparse
import time
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

config = import_module("0_config")
preprocess_module = import_module("2_preprocess")
feature_module = import_module("3_features")
read_csv = preprocess_module.read_csv
load_nodes = preprocess_module.load_nodes
feature_columns = feature_module.feature_columns

HYPERGRAPH_FILE_NAME = "hypergraph.npz"


def log_event(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(device_name)


# Find exact cosine neighbors in batches so the full N x N matrix is never stored.
def behavioral_neighbors(
    train_features,
    query_features,
    maximum_neighbor_count,
    *,
    exclude_self=False,
    device_name="auto",
    batch_size=256,
    progress_name="behavioral kNN",
):
    train_count = len(train_features)
    query_count = len(query_features)
    available_references = train_count - int(exclude_self)
    if maximum_neighbor_count <= 0:
        raise ValueError("maximum_neighbor_count must be positive")
    if maximum_neighbor_count > available_references:
        raise ValueError("Not enough train nodes for the requested neighbor count")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if exclude_self and query_count != train_count:
        raise ValueError("Self-exclusion requires aligned train queries")

    train_behavior = np.asarray(
        train_features[:, config.BEHAVIOR_FEATURE_SLICE],
        dtype=np.float32,
    ).copy()
    query_behavior = np.asarray(
        query_features[:, config.BEHAVIOR_FEATURE_SLICE],
        dtype=np.float32,
    ).copy()

    train_norms = np.linalg.norm(train_behavior, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_behavior, axis=1, keepdims=True)
    np.divide(
        train_behavior,
        train_norms,
        out=train_behavior,
        where=train_norms != 0,
    )
    np.divide(
        query_behavior,
        query_norms,
        out=query_behavior,
        where=query_norms != 0,
    )
    zero_query_rows = query_norms.ravel() == 0

    device = resolve_device(device_name)
    started_at = time.perf_counter()
    log_event(
        "hypergraph",
        f"{progress_name}: queries={query_count:,}, references={train_count:,}, "
        f"k_max={maximum_neighbor_count}, device={device}, batch={batch_size}",
    )

    train_tensor = torch.as_tensor(
        np.ascontiguousarray(train_behavior),
        device=device,
    )
    train_tensor_transposed = train_tensor.transpose(0, 1)
    neighbor_node_ids = np.empty(
        (query_count, maximum_neighbor_count),
        dtype=np.int32,
    )
    all_train_node_ids = np.arange(train_count, dtype=np.int32)

    next_progress_percent = 10
    with torch.inference_mode():
        for batch_start in range(0, query_count, batch_size):
            batch_stop = min(batch_start + batch_size, query_count)
            batch_zero_rows = zero_query_rows[batch_start:batch_stop]
            nonzero_positions = np.flatnonzero(~batch_zero_rows)

            if len(nonzero_positions) > 0:
                query_tensor = torch.as_tensor(
                    query_behavior[batch_start:batch_stop][nonzero_positions],
                    device=device,
                )
                cosine_similarities = query_tensor @ train_tensor_transposed

                if exclude_self:
                    local_query_ids = torch.arange(
                        len(nonzero_positions),
                        device=device,
                    )
                    global_query_ids = torch.as_tensor(
                        batch_start + nonzero_positions,
                        dtype=torch.int64,
                        device=device,
                    )
                    cosine_similarities[local_query_ids, global_query_ids] = (
                        -torch.inf
                    )

                top_scores, top_node_ids = torch.topk(
                    cosine_similarities,
                    maximum_neighbor_count,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                top_scores = top_scores.cpu().numpy()
                top_node_ids = top_node_ids.cpu().numpy().astype(
                    np.int32,
                    copy=False,
                )

                # Node ID is the stable tie-breaker for equal cosine scores.
                for result_row, local_position in enumerate(nonzero_positions):
                    order = np.lexsort(
                        (top_node_ids[result_row], -top_scores[result_row])
                    )
                    query_node_id = batch_start + int(local_position)
                    neighbor_node_ids[query_node_id] = top_node_ids[
                        result_row,
                        order,
                    ]

            # A zero behavior vector has equal cosine similarity to every node.
            for local_position in np.flatnonzero(batch_zero_rows):
                query_node_id = batch_start + int(local_position)
                valid_node_ids = all_train_node_ids
                if exclude_self:
                    valid_node_ids = all_train_node_ids[
                        all_train_node_ids != query_node_id
                    ]
                neighbor_node_ids[query_node_id] = valid_node_ids[
                    :maximum_neighbor_count
                ]

            completed_percent = int(100 * batch_stop / max(query_count, 1))
            if completed_percent >= next_progress_percent or batch_stop == query_count:
                log_event(
                    "hypergraph",
                    f"{progress_name}: {batch_stop:,}/{query_count:,} "
                    f"({completed_percent}%) in "
                    f"{time.perf_counter() - started_at:.1f}s",
                )
                while next_progress_percent <= completed_percent:
                    next_progress_percent += 10

    return neighbor_node_ids


# Enrollment nodes from the same course form one Course hyperedge.
def build_course_hyperedges(nodes):
    members_by_course = defaultdict(list)
    for node in nodes:
        members_by_course[node["course_id"]].append(int(node["node_id"]))

    course_hyperedges = {}
    for course_id in sorted(members_by_course):
        course_hyperedges[course_id] = sorted(set(members_by_course[course_id]))
    return course_hyperedges


# Enrollment nodes using the same object in the same course form one Object edge.
def build_object_hyperedges(data_path):
    started_at = time.perf_counter()
    members_by_object = defaultdict(set)
    event_count = 0

    for event_count, event in enumerate(read_csv(data_path), start=1):
        object_type = config.OBJECT_ACTIONS.get(event["action"], "")
        object_id = event["object_id"].strip()
        if object_type and object_id.lower() not in config.MISSING_VALUES:
            # Course and object type prevent unrelated objects from being merged.
            object_key = (event["course_id"], object_type, object_id)
            members_by_object[object_key].add(int(event["node_id"]))

        if event_count % 1_000_000 == 0:
            log_event(
                "hypergraph",
                f"object grouping: scanned {event_count:,} events",
            )

    object_hyperedges = {}
    for object_key in sorted(members_by_object):
        object_hyperedges[object_key] = sorted(members_by_object[object_key])

    log_event(
        "hypergraph",
        f"object grouping completed: events={event_count:,}, "
        f"objects={len(object_hyperedges):,}, "
        f"elapsed={time.perf_counter() - started_at:.1f}s",
    )
    return object_hyperedges


# One Behavioral edge contains its anchor and the first k saved neighbors.
def build_behavioral_hyperedges(neighbor_node_ids, neighbor_count):
    behavioral_hyperedges = []
    for anchor_node_id in range(len(neighbor_node_ids)):
        members = [anchor_node_id]
        for neighbor_node_id in neighbor_node_ids[anchor_node_id, :neighbor_count]:
            members.append(int(neighbor_node_id))
        behavioral_hyperedges.append(sorted(set(members)))
    return behavioral_hyperedges


# Convert the three ordered hyperedge families directly to sparse CSR H0.
def build_h0(
    node_count,
    course_hyperedges,
    object_hyperedges,
    behavioral_hyperedges,
):
    hyperedges = []
    edge_families = []

    for members in course_hyperedges.values():
        if len(members) >= 2:
            hyperedges.append(members)
            edge_families.append("course")

    for members in object_hyperedges.values():
        if len(members) >= 2:
            hyperedges.append(members)
            edge_families.append("object")

    for members in behavioral_hyperedges:
        if len(members) >= 2:
            hyperedges.append(members)
            edge_families.append("behavioral")

    if not hyperedges:
        raise ValueError("No non-singleton hyperedges were constructed")

    rows = []
    columns = []
    edge_sizes = []
    for edge_id, members in enumerate(hyperedges):
        edge_sizes.append(len(members))
        for node_id in members:
            rows.append(node_id)
            columns.append(edge_id)

    values = np.ones(len(rows), dtype=np.uint8)
    initial_incidence_matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(node_count, len(hyperedges)),
    ).tocsr()

    edge_families = np.asarray(edge_families, dtype="<U10")
    edge_sizes = np.asarray(edge_sizes, dtype=np.int32)
    family_counts = {
        "course": int(np.count_nonzero(edge_families == "course")),
        "object": int(np.count_nonzero(edge_families == "object")),
        "behavioral": int(np.count_nonzero(edge_families == "behavioral")),
    }
    return initial_incidence_matrix, edge_families, edge_sizes, family_counts


def build_hypergraph(
    output_dir=config.PROCESSED,
    *,
    k=10,
    k_max=20,
    device_name="auto",
    neighbor_batch_size=256,
):
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    if k <= 0 or k_max <= 0 or k > k_max:
        raise ValueError("Require 0 < k <= k_max")

    train_nodes = load_nodes(output_dir / "train.csv")
    train_features = np.load(output_dir / "train" / "X.npy")
    if len(train_nodes) != len(train_features):
        raise ValueError("Train nodes and feature rows do not align")
    if k_max >= len(train_nodes):
        raise ValueError("k_max must be smaller than the train node count")

    log_event(
        "hypergraph",
        f"build started: train_nodes={len(train_nodes):,}, k={k}, k_max={k_max}",
    )
    train_neighbors = behavioral_neighbors(
        train_features,
        train_features,
        k_max,
        exclude_self=True,
        device_name=device_name,
        batch_size=neighbor_batch_size,
        progress_name="train-to-train neighbors",
    )

    evaluation_neighbors = {}
    for split_name in ("validation", "test"):
        target_features = np.load(output_dir / split_name / "X.npy")
        target_nodes = load_nodes(output_dir / f"{split_name}.csv")
        if len(target_nodes) != len(target_features):
            raise ValueError(f"{split_name} nodes and feature rows do not align")
        evaluation_neighbors[split_name] = behavioral_neighbors(
            train_features,
            target_features,
            k_max,
            device_name=device_name,
            batch_size=neighbor_batch_size,
            progress_name=f"{split_name}-to-train neighbors",
        )

    course_hyperedges = build_course_hyperedges(train_nodes)
    object_hyperedges = build_object_hyperedges(output_dir / "train.csv")
    behavioral_hyperedges = build_behavioral_hyperedges(train_neighbors, k)
    (
        initial_incidence_matrix,
        edge_families,
        edge_sizes,
        family_counts,
    ) = build_h0(
        len(train_nodes),
        course_hyperedges,
        object_hyperedges,
        behavioral_hyperedges,
    )

    # H0: [num_train_nodes, num_hyperedges]
    log_event(
        "hypergraph",
        f"H0 ready: shape={initial_incidence_matrix.shape}, "
        f"nnz={initial_incidence_matrix.nnz:,}, families={family_counts}",
    )

    bundle_path = output_dir / HYPERGRAPH_FILE_NAME
    matrix = initial_incidence_matrix.tocsr()
    np.savez_compressed(
        bundle_path,
        h0_data=matrix.data,
        h0_indices=matrix.indices,
        h0_indptr=matrix.indptr,
        h0_shape=np.asarray(matrix.shape, dtype=np.int64),
        edge_families=edge_families,
        edge_sizes=edge_sizes,
        train_neighbors=train_neighbors,
        validation_neighbors=evaluation_neighbors["validation"],
        test_neighbors=evaluation_neighbors["test"],
        k=np.asarray(k, dtype=np.int32),
        k_max=np.asarray(k_max, dtype=np.int32),
        neighbor_backend=np.asarray("torch_exact_cosine"),
    )

    report = {
        "artifact": str(bundle_path),
        "neighbor_backend": "torch_exact_cosine",
        "k": k,
        "k_max": k_max,
        "train_nodes": len(train_nodes),
        "edges": initial_incidence_matrix.shape[1],
        "incidences": initial_incidence_matrix.nnz,
        "families": family_counts,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    log_event(
        "hypergraph",
        f"build completed in {report['elapsed_seconds']:.1f}s: {report}",
    )
    return report


def load_train_graph(output_dir=config.PROCESSED, *, feature_set="behavior"):
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    with np.load(output_dir / HYPERGRAPH_FILE_NAME, allow_pickle=False) as bundle:
        matrix_shape = tuple(int(value) for value in bundle["h0_shape"])
        initial_incidence_matrix = sparse.csr_matrix(
            (
                bundle["h0_data"],
                bundle["h0_indices"],
                bundle["h0_indptr"],
            ),
            shape=matrix_shape,
        )
        edge_families = bundle["edge_families"].copy()
        edge_sizes = bundle["edge_sizes"].astype(np.int64, copy=True)

    all_features = np.load(output_dir / "train" / "X.npy")
    selected_columns = feature_columns(feature_set)
    node_features = np.asarray(
        all_features[:, selected_columns],
        dtype=np.float32,
    )
    labels = np.asarray(
        [int(node["label"]) for node in load_nodes(output_dir / "train.csv")],
        dtype=np.float32,
    )

    if initial_incidence_matrix.shape[0] != len(node_features):
        raise ValueError("H0 and train features do not align")
    if initial_incidence_matrix.shape[1] != len(edge_families):
        raise ValueError("H0 and edge metadata do not align")

    log_event(
        "hypergraph",
        f"train graph loaded: X={node_features.shape}, "
        f"H0={initial_incidence_matrix.shape}, nnz={initial_incidence_matrix.nnz:,}, "
        f"elapsed={time.perf_counter() - started_at:.1f}s",
    )
    return {
        "features": node_features,
        "incidence_matrix": initial_incidence_matrix,
        "labels": labels,
        "families": edge_families,
        "sizes": edge_sizes,
    }


# Load one target split while keeping every graph reference in the train split.
def load_evaluation_data(
    output_dir=config.PROCESSED,
    *,
    split_name="validation",
    feature_set="behavior",
):
    if split_name not in ("validation", "test"):
        raise ValueError("split_name must be validation or test")

    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    train_nodes = load_nodes(output_dir / "train.csv")
    target_nodes = load_nodes(output_dir / f"{split_name}.csv")
    with np.load(output_dir / HYPERGRAPH_FILE_NAME, allow_pickle=False) as bundle:
        neighbors = bundle[f"{split_name}_neighbors"].copy()
        neighbor_count = int(bundle["k"])

    target_objects = defaultdict(set)
    for event in read_csv(output_dir / f"{split_name}.csv"):
        object_type = config.OBJECT_ACTIONS.get(event["action"], "")
        object_id = event["object_id"].strip()
        if object_type and object_id.lower() not in config.MISSING_VALUES:
            object_key = (event["course_id"], object_type, object_id)
            target_objects[int(event["node_id"])].add(object_key)

    if len(neighbors) != len(target_nodes):
        raise ValueError("Evaluation nodes and neighbor rows do not align")

    evaluation_data = {
        "split_name": split_name,
        "nodes": target_nodes,
        "course_references": build_course_hyperedges(train_nodes),
        "object_references": build_object_hyperedges(output_dir / "train.csv"),
        "target_objects": target_objects,
        "neighbors": neighbors,
        "train_features": np.load(output_dir / "train" / "X.npy"),
        "target_features": np.load(output_dir / split_name / "X.npy"),
        "selected_columns": feature_columns(feature_set),
        "neighbor_count": neighbor_count,
    }
    log_event(
        "hypergraph",
        f"{split_name} data loaded: targets={len(target_nodes):,}, "
        f"neighbors={neighbors.shape}, elapsed={time.perf_counter() - started_at:.1f}s",
    )
    return evaluation_data


# Build: one target validation/test node + train reference nodes -> local graph.
def build_local_graph(evaluation_data, target_node_id):
    target_node = evaluation_data["nodes"][target_node_id]
    local_edges = []

    course_nodes = evaluation_data["course_references"].get(
        target_node["course_id"],
        [],
    )
    if course_nodes:
        local_edges.append(("course", course_nodes))

    object_keys = sorted(evaluation_data["target_objects"].get(target_node_id, []))
    for object_key in object_keys:
        object_nodes = evaluation_data["object_references"].get(object_key, [])
        if object_nodes:
            local_edges.append(("object", object_nodes))

    behavioral_nodes = evaluation_data["neighbors"][
        target_node_id,
        :evaluation_data["neighbor_count"],
    ]
    local_edges.append(("behavioral", behavioral_nodes))

    train_reference_ids = set()
    for _, node_ids in local_edges:
        for node_id in node_ids:
            train_reference_ids.add(int(node_id))
    train_reference_ids = sorted(train_reference_ids)

    local_position = {}
    for position, train_node_id in enumerate(train_reference_ids, start=1):
        local_position[train_node_id] = position

    rows = []
    columns = []
    for edge_id, (_, train_node_ids) in enumerate(local_edges):
        rows.append(0)  # The target node is always local row 0.
        columns.append(edge_id)
        for train_node_id in train_node_ids:
            rows.append(local_position[int(train_node_id)])
            columns.append(edge_id)

    values = np.ones(len(rows), dtype=np.uint8)
    initial_incidence_matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(train_reference_ids) + 1, len(local_edges)),
    ).tocsr()

    selected_columns = evaluation_data["selected_columns"]
    target_features = evaluation_data["target_features"][
        target_node_id:target_node_id + 1,
        selected_columns,
    ]
    train_features = evaluation_data["train_features"][train_reference_ids][
        :,
        selected_columns,
    ]
    node_features = np.concatenate(
        (target_features, train_features),
        axis=0,
    ).astype(np.float32, copy=False)

    # X_local: [1 target + train references, input_dim]
    # H0_local: [1 target + train references, local_hyperedges]
    edge_families = np.asarray(
        [family for family, _ in local_edges],
        dtype="<U10",
    )
    edge_sizes = np.asarray(
        initial_incidence_matrix.sum(axis=0)
    ).ravel().astype(np.int64)
    return {
        "features": node_features,
        "incidence_matrix": initial_incidence_matrix,
        "families": edge_families,
        "sizes": edge_sizes,
        "label": int(target_node["label"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config.HYPERGRAPH_CLI_DESCRIPTION)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--neighbor-batch-size", type=int, default=256)
    arguments = parser.parse_args()
    build_hypergraph(
        arguments.output_dir,
        k=arguments.k,
        k_max=arguments.k_max,
        device_name=arguments.device,
        neighbor_batch_size=arguments.neighbor_batch_size,
    )
