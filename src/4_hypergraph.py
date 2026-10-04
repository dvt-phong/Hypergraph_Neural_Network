# 4. Build the train hypergraph H0 once (python src/4_hypergraph.py). Nodes are the
#    train enrollments; hyperedges are Course, Object and User. Self-loops are not
#    stored: 5_graph_data.py adds one per node when it loads the graph.
#    Only train.csv is read: no validation/test enrollment and no label is used here.
# Tham khảo từ project/bài báo:
# - HGNN, AAAI 2019 (Feng et al.): https://doi.org/10.1609/aaai.v33i01.33013558
#   Code: https://github.com/iMoonLab/HGNN
# - SIG-Net, ACM SAC 2024 / MST-GCN, Scientific Reports 2026: enrollments of the
#   same learner are linked whatever the course start (User "any")
#   Code: https://github.com/Noverse0/SIG-Net, https://github.com/wudongze9/MST-GCN

import argparse
import time
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
read_csv = import_module("2_preprocess").read_csv

HYPERGRAPH_FILE = "hypergraph.npz"
COURSE = config.EDGE_FAMILIES.index("course")
OBJECT = config.EDGE_FAMILIES.index("object")
USER = config.EDGE_FAMILIES.index("user")
SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")


# Print a message with the time.
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}][hypergraph] {message}", flush=True)


# Read a split CSV (one row per event) in one pass.
# Input:  path of train.csv, validation.csv or test.csv.
# Output: nodes   list, nodes[node_id] = first row of that node (user, course, label, ...)
#         objects dict node_id -> set of object keys "course|family|object_id" the node used
def read_split(path):
    nodes = {}
    objects = defaultdict(set)
    for row in read_csv(path):
        node_id = int(row["node_id"])
        if node_id not in nodes:
            nodes[node_id] = row
        family = config.OBJECT_ACTIONS.get(row["action"])    # video / assignment / forum; web pages have none
        object_id = row["object_id"].strip()
        if family and object_id.lower() not in config.MISSING_VALUES:
            objects[node_id].add(f"{row['course_id']}|{family}|{object_id}")
    if sorted(nodes) != list(range(len(nodes))):
        raise ValueError(f"{path}: node ids are not 0..n-1")
    return [nodes[node_id] for node_id in range(len(nodes))], objects


# Course hyperedges: e_C(c) = {v : course(v) = c}, one per course.
# Input:  train node rows.
# Output: list of (COURSE, course_id, member node ids).
def course_hyperedges(nodes):
    members = defaultdict(list)
    for node_id, node in enumerate(nodes):
        members[node["course_id"]].append(node_id)
    return [(COURSE, course_id, members[course_id]) for course_id in sorted(members)]


# Object hyperedges: e_O(o) = {v : v used object o in days 0–34}, one per object.
# Input:  dict node_id -> object keys (from read_split).
# Output: list of (OBJECT, "course|family|object_id", member node ids).
def object_hyperedges(objects):
    members = defaultdict(list)
    for node_id in sorted(objects):
        for key in objects[node_id]:
            members[key].append(node_id)
    return [(OBJECT, key, members[key]) for key in sorted(members)]


# User hyperedges (rule "any"): e_U(l) = {v : user(v) = l}, one per learner, with all
# of the learner's train enrollments whatever their course start.
# Input:  train node rows.
# Output: list of (USER, "user|<user_id>", member node ids).
def user_hyperedges(nodes):
    members = defaultdict(list)
    for node_id, node in enumerate(nodes):
        members[int(node["user_id"])].append(node_id)
    return [(USER, f"user|{user_id}", members[user_id]) for user_id in sorted(members)]


# Run step 4: build H0 and save it.
# Input:  output_dir with train.csv.
# Output: none (writes output_dir/hypergraph.npz):
#   node_ids, edge_ids  [M]  one entry per membership (v, e), sorted by e, so the members
#                            of e are node_ids[start[e]:start[e + 1]]
#   edge_family [E]          COURSE / OBJECT / USER
#   edge_keys   [E]          course_id, "course|family|object_id" or "user|<user_id>"
#   train_labels [N]         dropout label of every train node (used for the loss only)
#   train_users  [N]         user id of every train node (User hyperedges of targets)
def build_hypergraph(output_dir=config.PROCESSED):
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    log("reading train.csv")
    nodes, objects = read_split(output_dir / "train.csv")

    hyperedges = course_hyperedges(nodes) + object_hyperedges(objects) + user_hyperedges(nodes)
    # Keep |e| ≥ 2: a one-member hyperedge carries nothing beyond the node's self-loop.
    hyperedges = [edge for edge in hyperedges if len(edge[2]) >= 2]

    node_ids = np.concatenate([np.asarray(members, dtype=np.int64) for _, _, members in hyperedges])
    edge_ids = np.repeat(np.arange(len(hyperedges), dtype=np.int64),
                         [len(members) for _, _, members in hyperedges])
    np.savez_compressed(
        output_dir / HYPERGRAPH_FILE,
        node_ids=node_ids,
        edge_ids=edge_ids,
        edge_family=np.asarray([family for family, _, _ in hyperedges], dtype=np.int64),
        edge_keys=np.asarray([key for _, key, _ in hyperedges]),
        train_labels=np.asarray([int(node["label"]) for node in nodes], dtype=np.float32),
        train_users=np.asarray([int(node["user_id"]) for node in nodes], dtype=np.int64),
    )
    counts = {name: sum(1 for family, _, _ in hyperedges if family == index)
              for index, name in enumerate(config.EDGE_FAMILIES[:SELF_LOOP])}
    log(f"saved {output_dir / HYPERGRAPH_FILE}: train nodes={len(nodes):,}, hyperedges={counts}, "
        f"memberships={len(node_ids):,}, elapsed={time.perf_counter() - started_at:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the train hypergraph H0 (Course, Object, User).")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()
    build_hypergraph(arguments.output_dir)
