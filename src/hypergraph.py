"""Build Course, Object and Behavioral hyperedges from node features."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from preprocess import PROCESSED, SEEDS, read_csv


def _neighbors(x, train_ids, k_max):
    import faiss

    if len(train_ids) <= k_max:
        raise ValueError("Need more train nodes than behavioral neighbors")
    normalized = np.ascontiguousarray(x[:, :60].astype(np.float32).copy())
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    np.divide(normalized, norms, out=normalized, where=norms != 0)
    index = faiss.IndexHNSWFlat(normalized.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 128
    index.add(np.ascontiguousarray(normalized[train_ids]))
    result = np.empty((len(x), k_max), dtype=np.int32)
    search_k = min(k_max + 8, len(train_ids))
    for start in range(0, len(x), 25000):
        stop = min(start + 25000, len(x))
        _, nearest = index.search(np.ascontiguousarray(normalized[start:stop]), search_k)
        for offset, local in enumerate(nearest):
            if np.any(local < 0):
                raise ValueError("FAISS did not find enough neighbors")
            anchor = start + offset
            found = train_ids[local]
            found = found[found != anchor]
            if len(found) < k_max or len(np.unique(found[:k_max])) != k_max:
                raise ValueError("Not enough distinct train neighbors")
            result[anchor] = found[:k_max]
    return result


def build_hyperedges(output_dir=PROCESSED, *, seed=1, k=10, k_max=20):
    output_dir = Path(output_dir)
    if not 0 < k <= k_max:
        raise ValueError("k must be between 1 and k_max")
    nodes = list(read_csv(output_dir / "nodes.csv"))
    split = [row["split"] for row in read_csv(output_dir / f"split_seed_{seed}.csv")]
    if len(nodes) != len(split):
        raise ValueError("Split and node counts differ")
    train_ids = np.flatnonzero(np.array(split) == "train")
    x = np.load(output_dir / f"X_seed_{seed}.npy", mmap_mode="r")
    neighbors = _neighbors(x, train_ids, k_max)
    np.save(output_dir / f"neighbors_seed_{seed}.npy", neighbors)

    course_groups = defaultdict(set)
    for row in nodes:
        node_id = int(row["node_id"])
        if split[node_id] == "train":
            course_groups[row["course_id"]].add(node_id)
    object_groups = defaultdict(set)
    with gzip.open(output_dir / "node_objects.csv.gz", "rt", newline="",
                   encoding="utf-8") as source:
        for row in csv.DictReader(source):
            node_id = int(row["node_id"])
            if split[node_id] == "train":
                key = (row["course_id"], row["object_id"], row["object_type"])
                object_groups[key].add(node_id)

    edge_path = output_dir / f"edge_memberships_seed_{seed}.csv.gz"
    meta_path = output_dir / f"edge_meta_seed_{seed}.csv"
    counts = {"course": 0, "object": 0, "behavioral": 0}
    with gzip.open(edge_path, "wt", newline="", encoding="utf-8",
                   compresslevel=1) as edge_file, open(meta_path, "w", newline="",
                                                 encoding="utf-8") as meta_file:
        edge_writer, meta_writer = csv.writer(edge_file), csv.writer(meta_file)
        edge_writer.writerow(("edge_id", "node_id"))
        meta_writer.writerow(("edge_id", "family", "course_id", "object_id",
                              "object_type", "anchor_id", "size"))
        edge_id = 0

        def add_edge(family, members, course="", obj="", obj_type="", anchor=""):
            nonlocal edge_id
            if len(members) < 2:
                return
            members = sorted(members)
            meta_writer.writerow((edge_id, family, course, obj, obj_type,
                                  anchor, len(members)))
            edge_writer.writerows((edge_id, node_id) for node_id in members)
            counts[family] += 1
            edge_id += 1

        for course_id in sorted(course_groups):
            add_edge("course", course_groups[course_id], course=course_id)
        for course_id, object_id, object_type in sorted(
            object_groups, key=lambda key: (key[0], key[2], key[1])):
            add_edge("object", object_groups[(course_id, object_id, object_type)],
                     course=course_id, obj=object_id, obj_type=object_type)
        behavioral_edges = {}
        for anchor in train_ids:
            members = tuple(sorted((int(anchor), *(int(v) for v in neighbors[anchor, :k]))))
            behavioral_edges.setdefault(members, int(anchor))
        for members in sorted(behavioral_edges):
            add_edge("behavioral", members, anchor=behavioral_edges[members])

    report = {"seed": seed, "train_nodes": len(train_ids), "edges": edge_id,
              "families": counts, "k": k, "k_max": k_max}
    (output_dir / f"graph_config_seed_{seed}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(report, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=20)
    args = parser.parse_args()
    build_hyperedges(args.output_dir, seed=args.seed, k=args.k, k_max=args.k_max)
