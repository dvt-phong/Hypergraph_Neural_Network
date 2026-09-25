# 4. Build and load Course, Object, and Behavioral hypergraphs.
# Tham khảo từ project/bài báo:
# - HGNN, AAAI 2019: https://doi.org/10.1609/aaai.v33i01.33013558
#   Code: https://github.com/iMoonLab/HGNN
# - SIG-Net, ACM SAC 2024: https://doi.org/10.1145/3605098.3636002
#   Code: https://github.com/Noverse0/SIG-Net
# - CA-TFHN, ICONIP 2023: https://doi.org/10.1007/978-981-99-8184-7_31
#   Code: https://github.com/codeds27/CA-TFHN
# - MST-GCN, Scientific Reports 2026:
#   https://doi.org/10.1038/s41598-026-40502-w
#   Code: https://github.com/wudongze9/MST-GCN
# Course, Object và Behavioral hyperedges là cách điều chỉnh riêng của project
# từ ý tưởng quan hệ bậc cao, student interaction và classmates similarity.

import argparse
import csv
import gzip
import json
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy import sparse

config = import_module("0_config")
preprocess_module = import_module("2_preprocess")
feature_module = import_module("3_features")
read_csv = preprocess_module.read_csv
load_nodes = preprocess_module.load_nodes
feature_columns = feature_module.feature_columns


# Find each query node's nearest train nodes by behavioral cosine similarity.
def behavioral_neighbors(
    train_features,
    query_features,
    maximum_neighbor_count,
    exclude_self=False,
):
    import faiss

    # Use only the 35 daily counts and 23 action counts for similarity.
    train_behavior = np.asarray(
        train_features[:, config.BEHAVIOR_FEATURE_SLICE],
        dtype=np.float32,
    ).copy()
    query_behavior = np.asarray(
        query_features[:, config.BEHAVIOR_FEATURE_SLICE],
        dtype=np.float32,
    ).copy()

    # Compute one L2 norm per node: ||x|| = sqrt(sum(x_i^2)).
    train_norms = np.linalg.norm(train_behavior, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_behavior, axis=1, keepdims=True)

    # After L2 normalization, inner product equals cosine similarity.
    np.divide(train_behavior, train_norms, out=train_behavior, where=train_norms != 0)
    np.divide(query_behavior, query_norms, out=query_behavior, where=query_norms != 0)

    # Build an approximate HNSW index with 32 graph links per node.
    index = faiss.IndexHNSWFlat(
        train_behavior.shape[1],
        32,
        faiss.METRIC_INNER_PRODUCT,
    )
    # Larger ef values improve neighbor recall but require more work.
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 128
    # FAISS expects a contiguous float32 matrix of reference vectors.
    index.add(np.ascontiguousarray(train_behavior))

    search_count = maximum_neighbor_count
    # Search for one extra result because a train node finds itself first.
    if exclude_self:
        search_count += 1
    # Row i stores the train node IDs nearest to query node i.
    neighbor_node_ids = np.empty(
        (len(query_features), maximum_neighbor_count),
        dtype=np.int32,
    )
    # Search in batches to limit peak memory on large datasets.
    batch_size = 25_000
    for batch_start in range(0, len(query_features), batch_size):
        batch_stop = min(batch_start + batch_size, len(query_features))
        # FAISS returns (similarity scores, reference node IDs).
        search_results = index.search(
            np.ascontiguousarray(query_behavior[batch_start:batch_stop]),
            search_count,
        )
        # Only neighbor IDs are needed; similarity scores are not saved.
        nearest_positions = search_results[1]

        for offset, positions in enumerate(nearest_positions):
            query_node_id = batch_start + offset
            # Remove the query itself during train-to-train search.
            if exclude_self:
                positions = positions[positions != query_node_id]
            neighbor_node_ids[query_node_id] = positions[:maximum_neighbor_count]
    return neighbor_node_ids


# Group train node identifiers by course.
def course_groups(nodes):
    groups = defaultdict(set)
    # All train nodes in the same course form one course hyperedge.
    for node in nodes:
        groups[node["course_id"]].add(int(node["node_id"]))
    return groups


# Group node identifiers by observed course object from one split CSV.
def object_groups(data_path):
    groups = defaultdict(set)
    for event in read_csv(data_path):
        # Web-page actions do not create object hyperedges.
        object_type = config.OBJECT_ACTIONS.get(event["action"], "")
        object_id = event["object_id"].strip()
        if object_type and object_id.lower() not in config.MISSING_VALUES:
            # Include course and type so equal object IDs do not collide.
            key = (event["course_id"], object_id, object_type)
            groups[key].add(int(event["node_id"]))
    return groups


# Group each target node's observed course objects.
def objects_by_node(data_path):
    objects_by_node = defaultdict(set)
    for event in read_csv(data_path):
        object_type = config.OBJECT_ACTIONS.get(event["action"], "")
        object_id = event["object_id"].strip()
        if object_type and object_id.lower() not in config.MISSING_VALUES:
            # Save each valid object observed by this target node.
            key = (event["course_id"], object_id, object_type)
            objects_by_node[int(event["node_id"])].add(key)
    return objects_by_node


# Sort object keys by course, object type, and object identifier.
def object_key_sort_key(object_key):
    return object_key[0], object_key[2], object_key[1]


# Write one non-singleton hyperedge and return the next edge identifier.
def write_edge(
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
    # A singleton cannot model a relation between different nodes.
    if len(sorted_members) < 2:
        return edge_id

    # Store one metadata row for the hyperedge itself.
    metadata_writer.writerow((
        edge_id,
        family,
        course_id,
        object_id,
        object_type,
        anchor_id,
        len(sorted_members),
    ))
    # Store one incidence row for every node that belongs to the edge.
    for node_id in sorted_members:
        edge_writer.writerow((edge_id, node_id))
    family_counts[family] += 1
    return edge_id + 1


# Write train Course, Object, and Behavioral hyperedges.
def write_hyperedges(
    output_dir,
    train_count,
    course_groups,
    object_groups,
    neighbor_node_ids,
    neighbor_count,
):
    edge_path = output_dir / "edge_memberships.csv.gz"
    metadata_path = output_dir / "edge_meta.csv"
    # Count saved edges separately for each relation family.
    family_counts = {"course": 0, "object": 0, "behavioral": 0}

    with gzip.open(
        edge_path,
        "wt",
        newline="",
        encoding="utf-8",
        compresslevel=1,
    ) as edge_file, open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as metadata_file:
        edge_writer = csv.writer(edge_file)
        metadata_writer = csv.writer(metadata_file)
        edge_writer.writerow(("edge_id", "node_id"))
        metadata_writer.writerow((
            "edge_id", "family", "course_id", "object_id",
            "object_type", "anchor_id", "size",
        ))
        edge_id = 0

        # Write one shared-course hyperedge for each course.
        for course_id in sorted(course_groups):
            edge_id = write_edge(
                edge_writer,
                metadata_writer,
                family_counts,
                edge_id,
                "course",
                course_groups[course_id],
                course_id=course_id,
            )

        # Write one shared-object hyperedge for each course object.
        object_keys = sorted(object_groups, key=object_key_sort_key)
        for course_id, object_id, object_type in object_keys:
            edge_id = write_edge(
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

        # A behavioral edge contains one anchor and its top-k neighbors.
        unique_behavioral_edges = {}
        for anchor_node_id in range(train_count):
            members = {anchor_node_id}
            for neighbor_node_id in neighbor_node_ids[
                anchor_node_id,
                :neighbor_count,
            ]:
                members.add(int(neighbor_node_id))
            # Sets remove repeated nodes and sorting gives a stable edge key.
            member_tuple = tuple(sorted(members))
            # Keep only one copy when two anchors create the same member set.
            unique_behavioral_edges.setdefault(member_tuple, anchor_node_id)

        # Write each unique behavioral member set as one hyperedge.
        for members in sorted(unique_behavioral_edges):
            edge_id = write_edge(
                edge_writer,
                metadata_writer,
                family_counts,
                edge_id,
                "behavioral",
                members,
                anchor_id=unique_behavioral_edges[members],
            )

    return edge_id, family_counts


# Turn train edge memberships into the sparse initial incidence matrix H0.
def build_h0(output_dir, train_count):
    metadata = list(read_csv(output_dir / "edge_meta.csv"))
    rows = []
    columns = []
    membership_path = output_dir / "edge_memberships.csv.gz"
    # Each CSV pair (node, edge) becomes a nonzero position in H0.
    with gzip.open(membership_path, "rt", newline="", encoding="utf-8") as source:
        for membership in csv.DictReader(source):
            rows.append(int(membership["node_id"]))
            columns.append(int(membership["edge_id"]))

    # H0[v, e] = 1 means node v belongs to hyperedge e.
    values = np.ones(len(rows), dtype=np.uint8)
    initial_incidence_matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(train_count, len(metadata)),
    ).tocsr()
    # CSR format supports efficient row-based hypergraph operations.
    sparse.save_npz(output_dir / "H0.npz", initial_incidence_matrix)
    return initial_incidence_matrix


# Build the train hypergraph and train-reference neighbors for evaluation.
def build_hypergraph(output_dir=config.PROCESSED, *, k=10, k_max=20):
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    train_nodes = load_nodes(output_dir / "train.csv")
    train_features = np.load(train_dir / "X.npy")

    # Find train neighbors from train references and remove self-matches.
    train_neighbors = behavioral_neighbors(
        train_features,
        train_features,
        k_max,
        exclude_self=True,
    )
    np.save(train_dir / "neighbors.npy", train_neighbors)

    # Validation and test nodes may only use train nodes as references.
    for split_name in ("validation", "test"):
        split_dir = output_dir / split_name
        target_features = np.load(split_dir / "X.npy")
        target_neighbors = behavioral_neighbors(
            train_features,
            target_features,
            k_max,
        )
        np.save(split_dir / "neighbors.npy", target_neighbors)

    # Build the three train-only relation families.
    train_course_groups = course_groups(train_nodes)
    train_object_groups = object_groups(output_dir / "train.csv")
    edge_count, family_counts = write_hyperedges(
        output_dir,
        len(train_nodes),
        train_course_groups,
        train_object_groups,
        train_neighbors,
        k,
    )
    # Convert saved memberships into the initial sparse incidence matrix.
    initial_incidence_matrix = build_h0(output_dir, len(train_nodes))

    report = {
        "k": k,
        "k_max": k_max,
        "train_nodes": len(train_nodes),
        "edges": edge_count,
        "incidences": initial_incidence_matrix.nnz,
        "families": family_counts,
    }
    (output_dir / "graph_config.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(report, flush=True)
    return report


# Load train features, H0, labels, and edge metadata.
def load_train_graph(output_dir=config.PROCESSED, *, feature_set="behavior"):
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    initial_incidence_matrix = sparse.load_npz(output_dir / "H0.npz")
    node_features = np.load(train_dir / "X.npy")
    # Select behavior, user, course, or full columns for this experiment.
    columns = feature_columns(feature_set)
    train_node_features = np.asarray(node_features[:, columns], dtype=np.float32)

    labels = []
    # Node IDs are local and contiguous, so sorted node rows align with X.
    for node in load_nodes(output_dir / "train.csv"):
        labels.append(int(node["label"]))
    labels = np.asarray(labels, dtype=np.float32)

    families = []
    sizes = []
    for metadata_row in read_csv(output_dir / "edge_meta.csv"):
        families.append(metadata_row["family"])
        sizes.append(int(metadata_row["size"]))
    families = np.asarray(families)
    sizes = np.asarray(sizes, dtype=np.int64)
    return train_node_features, initial_incidence_matrix, labels, families, sizes


# Load one evaluation split and its train-only graph references.
def load_evaluation_data(
    output_dir=config.PROCESSED,
    *,
    split_name="validation",
    feature_set="behavior",
):
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    target_dir = output_dir / split_name
    train_nodes = load_nodes(output_dir / "train.csv")
    target_nodes = load_nodes(output_dir / f"{split_name}.csv")

    # Build train-only course references for each evaluation target.
    course_references = defaultdict(list)
    for node in train_nodes:
        course_references[node["course_id"]].append(int(node["node_id"]))

    graph_settings = json.loads(
        (output_dir / "graph_config.json").read_text(encoding="utf-8")
    )
    return {
        "nodes": target_nodes,
        "course_references": course_references,
        "object_references": object_groups(
            output_dir / "train.csv"
        ),
        "objects_by_node": objects_by_node(
            output_dir / f"{split_name}.csv"
        ),
        "neighbors": np.load(target_dir / "neighbors.npy"),
        "train_x": np.load(train_dir / "X.npy"),
        "target_x": np.load(target_dir / "X.npy"),
        "columns": feature_columns(feature_set),
        "k": int(graph_settings["k"]),
    }


# Build one evaluation target graph using train nodes as references.
def build_local_graph(evaluation_data, target_node_id):
    target_node = evaluation_data["nodes"][target_node_id]
    edges = []
    # Add train nodes that share the target course.
    course_nodes = evaluation_data["course_references"][target_node["course_id"]]
    if course_nodes:
        edges.append(("course", course_nodes))

    # Add one train-reference edge for each object observed by the target.
    for object_key in sorted(evaluation_data["objects_by_node"][target_node_id]):
        object_nodes = evaluation_data["object_references"][object_key]
        if object_nodes:
            edges.append(("object", object_nodes))

    # Add the top-k behaviorally similar train nodes.
    behavioral_nodes = evaluation_data["neighbors"][target_node_id, :evaluation_data["k"]]
    edges.append(("behavioral", behavioral_nodes))

    # Merge references so each train node appears once in the local graph.
    reference_set = set()
    for edge_data in edges:
        for node_id in edge_data[1]:
            reference_set.add(int(node_id))
    references = sorted(reference_set)

    positions = {}
    # Local row 0 is the target; train references start from row 1.
    for local_position, node_id in enumerate(references, start=1):
        positions[node_id] = local_position

    rows = []
    columns = []
    # Put the target and all matching train references into each local edge.
    for edge_id, edge_data in enumerate(edges):
        rows.append(0)
        columns.append(edge_id)
        for node_id in edge_data[1]:
            rows.append(positions[int(node_id)])
            columns.append(edge_id)

    # Local H0 has one row for the target plus one row per train reference.
    values = np.ones(len(rows), dtype=np.uint8)
    initial_incidence_matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(references) + 1, len(edges)),
    ).tocsr()

    # Keep target first so its prediction is always read from local row 0.
    selected_columns = evaluation_data["columns"]
    target_features = evaluation_data["target_x"][
        target_node_id:target_node_id + 1,
        selected_columns,
    ]
    reference_features = evaluation_data["train_x"][references][:, selected_columns]
    node_features = np.concatenate(
        (target_features, reference_features),
        axis=0,
    ).astype(np.float32, copy=False)

    families = []
    for edge_data in edges:
        families.append(edge_data[0])
    families = np.asarray(families)
    sizes = np.asarray(initial_incidence_matrix.sum(axis=0)).ravel().astype(np.int64)
    label = int(target_node["label"])
    return node_features, initial_incidence_matrix, families, sizes, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config.HYPERGRAPH_CLI_DESCRIPTION)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=20)
    arguments = parser.parse_args()
    build_hypergraph(
        arguments.output_dir,
        k=arguments.k,
        k_max=arguments.k_max,
    )
