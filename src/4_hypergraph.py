# 4. Build the train hypergraph H0
# - HGNN https://github.com/iMoonLab/HGNN
# - MST-GCN https://github.com/wudongze9/MST-GCN
# - SIG-Net https://github.com/Noverse0/SIG-Net

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


# Read a train / validation / test CSV
# nodes[node_id] = first row of that node,
# objects[node_id] = set of object keys "course|family|object_id" the node used.
def read_split(path):
    nodes = {}
    objects = defaultdict(set)
    for row in read_csv(path):
        node_id = int(row["node_id"])
        if node_id not in nodes:
            nodes[node_id] = row
        family = config.OBJECT_ACTIONS.get(row["action"])    # video / assignment / forum
        object_id = row["object_id"].strip()
        if family and object_id.lower() not in config.MISSING_VALUES:
            objects[node_id].add(f"{row['course_id']}|{family}|{object_id}")
    if sorted(nodes) != list(range(len(nodes))):
        raise ValueError(f"{path}: node ids are not 0..n-1")
    return [nodes[node_id] for node_id in range(len(nodes))], objects


# Course hyperedges, one per course, as (COURSE, course_id, members):
# e_C(c) = {v : course(v) = c}
def course_hyperedges(nodes):
    members = defaultdict(list)
    for node_id, node in enumerate(nodes):
        members[node["course_id"]].append(node_id)
    return [(COURSE, course_id, members[course_id]) for course_id in sorted(members)]


# Object hyperedges, one per object, as (OBJECT, "course|family|object_id", members):
# e_O(o) = {v : v used object o in days 0–34}
def object_hyperedges(objects):
    members = defaultdict(list)
    for node_id in sorted(objects):
        for key in objects[node_id]:
            members[key].append(node_id)
    return [(OBJECT, key, members[key]) for key in sorted(members)]


# User hyperedges, one per learner, as (USER, "user|<user_id>", members), with all of the
# learner's train enrollments whatever their course start:
# e_U(l) = {v : user(v) = l}
def user_hyperedges(nodes):
    members = defaultdict(list)
    for node_id, node in enumerate(nodes):
        members[int(node["user_id"])].append(node_id)
    return [(USER, f"user|{user_id}", members[user_id]) for user_id in sorted(members)]


# Run step 4: build the train hypergraph H0 -> hypergraph.npz:
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

    # Sparse incidence matrix H0 as a membership list: h(v,e) = 1 for every (node_ids[k], edge_ids[k]).
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
