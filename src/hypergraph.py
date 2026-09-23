# Build and load Course, Object, and Behavioral hypergraphs.

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from features import BEHAVIOR_FEATURE_SLICE, feature_columns
from preprocess import PROCESSED, SEEDS, read_csv, split_nodes

CLI_DESCRIPTION = "Build Course, Object, and Behavioral hypergraphs."


# Find each node's nearest train nodes by behavioral cosine similarity.
def _behavioral_neighbors(node_features, train_node_ids, maximum_neighbor_count):
    import faiss

    if len(train_node_ids) <= maximum_neighbor_count:
        raise ValueError("Need more train nodes than behavioral neighbors")

    behavior_features = node_features[:, BEHAVIOR_FEATURE_SLICE].astype(np.float32)
    normalized_features = np.ascontiguousarray(behavior_features.copy())
    feature_norms = np.linalg.norm(normalized_features, axis=1, keepdims=True)
    np.divide(
        normalized_features,
        feature_norms,
        out=normalized_features,
        where=feature_norms != 0,
    )

    index = faiss.IndexHNSWFlat(
        normalized_features.shape[1],
        32,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 128
    index.add(np.ascontiguousarray(normalized_features[train_node_ids]))

    neighbor_node_ids = np.empty(
        (len(node_features), maximum_neighbor_count),
        dtype=np.int32,
    )
    search_count = min(maximum_neighbor_count + 8, len(train_node_ids))
    batch_size = 25_000
    for batch_start in range(0, len(node_features), batch_size):
        batch_stop = min(batch_start + batch_size, len(node_features))
        query_features = np.ascontiguousarray(
            normalized_features[batch_start:batch_stop]
        )
        search_results = index.search(
            query_features,
            search_count,
        )
        nearest_positions = search_results[1]

        for offset, positions in enumerate(nearest_positions):
            if np.any(positions < 0):
                raise ValueError("FAISS did not find enough neighbors")

            anchor_node_id = batch_start + offset
            found_node_ids = train_node_ids[positions]
            found_node_ids = found_node_ids[found_node_ids != anchor_node_id]
            selected_node_ids = found_node_ids[:maximum_neighbor_count]
            if len(selected_node_ids) < maximum_neighbor_count:
                raise ValueError("Not enough distinct train neighbors")
            if len(np.unique(selected_node_ids)) != maximum_neighbor_count:
                raise ValueError("FAISS returned repeated train neighbors")
            neighbor_node_ids[anchor_node_id] = selected_node_ids
    return neighbor_node_ids


# Group train node identifiers by course.
def _course_groups(nodes, split):
    groups = defaultdict(set)
    for node in nodes:
        node_id = int(node["node_id"])
        if split[node_id] == "train":
            groups[node["course_id"]].add(node_id)
    return groups


# Group train node identifiers by observed course object.
def _object_groups(output_dir, split):
    groups = defaultdict(set)
    object_path = output_dir / "node_objects.csv.gz"
    with gzip.open(object_path, "rt", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            node_id = int(row["node_id"])
            if split[node_id] != "train":
                continue
            key = (row["course_id"], row["object_id"], row["object_type"])
            groups[key].add(node_id)
    return groups


# Sort object keys by course, object type, and object identifier.
def _object_key_sort_key(object_key):
    return object_key[0], object_key[2], object_key[1]


# Write one non-singleton hyperedge and return the next edge identifier.
def _write_edge(
    edge_writer,
    metadata_writer,
    family_counts,
    edge_id,
    family,
    members,
    course_id="",
    object_id="",
    object_type="",
    anchor_id="",
):
    sorted_members = sorted(members)
    if len(sorted_members) < 2:
        return edge_id

    metadata_writer.writerow((
        edge_id,
        family,
        course_id,
        object_id,
        object_type,
        anchor_id,
        len(sorted_members),
    ))
    for node_id in sorted_members:
        edge_writer.writerow((edge_id, node_id))
    family_counts[family] += 1
    return edge_id + 1


# Write course, object, and behavioral hyperedges with stable identifiers.
def _write_hyperedges(
    output_dir,
    seed,
    train_ids,
    course_groups,
    object_groups,
    neighbor_node_ids,
    neighbor_count,
):
    edge_path = output_dir / f"edge_memberships_seed_{seed}.csv.gz"
    metadata_path = output_dir / f"edge_meta_seed_{seed}.csv"
    family_counts = {"course": 0, "object": 0, "behavioral": 0}

    with gzip.open(
        edge_path,
        "wt",
        newline="",
        encoding="utf-8",
        compresslevel=1,
    ) as edge_file:
        with open(metadata_path, "w", newline="", encoding="utf-8") as metadata_file:
            edge_writer = csv.writer(edge_file)
            metadata_writer = csv.writer(metadata_file)
            edge_writer.writerow(("edge_id", "node_id"))
            metadata_writer.writerow((
                "edge_id", "family", "course_id", "object_id",
                "object_type", "anchor_id", "size",
            ))
            edge_id = 0

            for course_id in sorted(course_groups):
                edge_id = _write_edge(
                    edge_writer,
                    metadata_writer,
                    family_counts,
                    edge_id,
                    "course",
                    course_groups[course_id],
                    course_id=course_id,
                )

            object_keys = sorted(object_groups, key=_object_key_sort_key)
            for course_id, object_id, object_type in object_keys:
                edge_id = _write_edge(
                    edge_writer,
                    metadata_writer,
                    family_counts,
                    edge_id,
                    "object",
                    object_groups[(course_id, object_id, object_type)],
                    course_id=course_id,
                    object_id=object_id,
                    object_type=object_type,
                )

            unique_behavioral_edges = {}
            for anchor_node_id in train_ids:
                members = {int(anchor_node_id)}
                for neighbor_node_id in neighbor_node_ids[
                    anchor_node_id,
                    :neighbor_count,
                ]:
                    members.add(int(neighbor_node_id))
                member_tuple = tuple(sorted(members))
                unique_behavioral_edges.setdefault(
                    member_tuple,
                    int(anchor_node_id),
                )

            for members in sorted(unique_behavioral_edges):
                edge_id = _write_edge(
                    edge_writer,
                    metadata_writer,
                    family_counts,
                    edge_id,
                    "behavioral",
                    members,
                    anchor_id=unique_behavioral_edges[members],
                )

    return edge_id, family_counts


# Turn train edge membership rows into the sparse initial incidence matrix H0.
def _build_h0(output_dir, seed, train_ids):
    global_to_train = {}
    for train_row, node_id in enumerate(train_ids):
        global_to_train[int(node_id)] = train_row

    metadata = list(read_csv(output_dir / f"edge_meta_seed_{seed}.csv"))
    rows = []
    columns = []
    membership_path = output_dir / f"edge_memberships_seed_{seed}.csv.gz"
    with gzip.open(membership_path, "rt", newline="", encoding="utf-8") as source:
        for membership in csv.DictReader(source):
            node_id = int(membership["node_id"])
            if node_id not in global_to_train:
                raise ValueError(f"Non-train node in H0 membership: {node_id}")
            rows.append(global_to_train[node_id])
            columns.append(int(membership["edge_id"]))

    values = np.ones(len(rows), dtype=np.uint8)
    initial_incidence_matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(train_ids), len(metadata)),
    ).tocsr()
    edge_sizes = np.asarray(initial_incidence_matrix.sum(axis=0)).ravel()
    node_degrees = np.asarray(initial_incidence_matrix.sum(axis=1)).ravel()
    if np.any(edge_sizes < 2):
        raise ValueError("H0 contains an empty or singleton hyperedge")
    if np.any(node_degrees == 0):
        raise ValueError("H0 contains an isolated train node")

    sparse.save_npz(
        output_dir / f"H0_seed_{seed}.npz",
        initial_incidence_matrix,
    )
    np.save(output_dir / f"train_ids_seed_{seed}.npy", train_ids)
    return initial_incidence_matrix


# Build all three hyperedge families and save their train incidence H0.
def build_hypergraph(output_dir=PROCESSED, *, seed=1, k=10, k_max=20, split=None):
    output_dir = Path(output_dir)
    if not 0 < k <= k_max:
        raise ValueError("k must be between 1 and k_max")

    nodes = list(read_csv(output_dir / "nodes.csv"))
    if split is None:
        split = split_nodes(nodes, seed)
    if len(nodes) != len(split):
        raise ValueError("Split and node counts differ")

    split_array = np.asarray(split)
    train_ids = np.flatnonzero(split_array == "train")
    node_features = np.load(output_dir / f"X_seed_{seed}.npy", mmap_mode="r")
    if len(node_features) != len(nodes):
        raise ValueError("X and nodes.csv have different row counts")

    neighbor_node_ids = _behavioral_neighbors(node_features, train_ids, k_max)
    np.save(output_dir / f"neighbors_seed_{seed}.npy", neighbor_node_ids)

    course_groups = _course_groups(nodes, split)
    object_groups = _object_groups(output_dir, split)
    edge_count, family_counts = _write_hyperedges(
        output_dir,
        seed,
        train_ids,
        course_groups,
        object_groups,
        neighbor_node_ids,
        k,
    )
    initial_incidence_matrix = _build_h0(output_dir, seed, train_ids)

    report = {
        "seed": seed,
        "k": k,
        "k_max": k_max,
        "train_nodes": len(train_ids),
        "edges": edge_count,
        "incidences": initial_incidence_matrix.nnz,
        "families": family_counts,
    }
    config_path = output_dir / f"graph_config_seed_{seed}.json"
    config_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report, flush=True)
    return report


# Load node features, H0, labels, and edge metadata for one train split.
def load_train_graph(output_dir=PROCESSED, *, seed=1, feature_set="behavior"):
    output_dir = Path(output_dir)
    initial_incidence_matrix = sparse.load_npz(output_dir / f"H0_seed_{seed}.npz")
    train_ids = np.load(output_dir / f"train_ids_seed_{seed}.npy")
    node_features = np.load(output_dir / f"X_seed_{seed}.npy", mmap_mode="r")
    columns = feature_columns(feature_set)
    train_node_features = np.asarray(
        node_features[train_ids][:, columns],
        dtype=np.float32,
    )

    nodes = list(read_csv(output_dir / "nodes.csv"))
    labels = []
    for node_id in train_ids:
        labels.append(int(nodes[node_id]["label"]))
    labels = np.asarray(labels, dtype=np.float32)

    metadata = list(read_csv(output_dir / f"edge_meta_seed_{seed}.csv"))
    families = []
    sizes = []
    for metadata_row in metadata:
        families.append(metadata_row["family"])
        sizes.append(int(metadata_row["size"]))
    families = np.asarray(families)
    sizes = np.asarray(sizes, dtype=np.int64)
    if initial_incidence_matrix.shape != (len(train_node_features), len(metadata)):
        raise ValueError("X, H0, and edge metadata do not align")
    return train_node_features, initial_incidence_matrix, labels, families, sizes


# Load the data needed to build leakage-free validation and test graphs.
def load_evaluation_data(output_dir=PROCESSED, *, seed=1, feature_set="behavior"):
    output_dir = Path(output_dir)
    nodes = list(read_csv(output_dir / "nodes.csv"))
    split = split_nodes(nodes, seed)
    course_references = defaultdict(list)
    object_references = defaultdict(list)
    objects_by_node = defaultdict(set)

    for node_id, node in enumerate(nodes):
        if split[node_id] == "train":
            course_references[node["course_id"]].append(node_id)

    object_path = output_dir / "node_objects.csv.gz"
    with gzip.open(object_path, "rt", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            node_id = int(row["node_id"])
            key = (row["course_id"], row["object_id"], row["object_type"])
            objects_by_node[node_id].add(key)
            if split[node_id] == "train":
                object_references[key].append(node_id)

    config_path = output_dir / f"graph_config_seed_{seed}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "nodes": nodes,
        "split": split,
        "course_references": course_references,
        "object_references": object_references,
        "objects_by_node": objects_by_node,
        "neighbors": np.load(output_dir / f"neighbors_seed_{seed}.npy"),
        "x": np.load(output_dir / f"X_seed_{seed}.npy"),
        "columns": feature_columns(feature_set),
        "k": int(config["k"]),
    }


# Build one evaluation target graph using train nodes as references.
def build_local_graph(evaluation_data, target_node_id):
    if evaluation_data["split"][target_node_id] == "train":
        raise ValueError("A local evaluation target must be validation or test")

    target_node = evaluation_data["nodes"][target_node_id]
    edges = []
    course_nodes = evaluation_data["course_references"][target_node["course_id"]]
    if course_nodes:
        edges.append(("course", course_nodes))

    for object_key in sorted(evaluation_data["objects_by_node"][target_node_id]):
        object_nodes = evaluation_data["object_references"][object_key]
        if object_nodes:
            edges.append(("object", object_nodes))

    behavioral_nodes = evaluation_data["neighbors"][
        target_node_id,
        :evaluation_data["k"],
    ]
    edges.append(("behavioral", behavioral_nodes))

    reference_set = set()
    for edge_data in edges:
        reference_nodes = edge_data[1]
        for node_id in reference_nodes:
            reference_set.add(int(node_id))
    references = sorted(reference_set)
    node_ids = np.asarray([target_node_id, *references], dtype=np.int64)

    positions = {}
    for local_position, node_id in enumerate(references, start=1):
        positions[node_id] = local_position

    rows = []
    columns = []
    for edge_id, edge_data in enumerate(edges):
        reference_nodes = edge_data[1]
        rows.append(0)
        columns.append(edge_id)
        for node_id in reference_nodes:
            rows.append(positions[int(node_id)])
            columns.append(edge_id)

    values = np.ones(len(rows), dtype=np.uint8)
    initial_incidence_matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(node_ids), len(edges)),
    ).tocsr()
    node_features = np.asarray(
        evaluation_data["x"][node_ids][:, evaluation_data["columns"]],
        dtype=np.float32,
    )
    families = []
    for edge_data in edges:
        families.append(edge_data[0])
    families = np.asarray(families)
    sizes = np.asarray(initial_incidence_matrix.sum(axis=0)).ravel().astype(np.int64)
    label = int(target_node["label"])
    return node_features, initial_incidence_matrix, families, sizes, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=20)
    arguments = parser.parse_args()
    build_hypergraph(
        arguments.output_dir,
        seed=arguments.seed,
        k=arguments.k,
        k_max=arguments.k_max,
    )
