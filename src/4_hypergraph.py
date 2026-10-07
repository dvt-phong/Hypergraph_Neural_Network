# 4. Build the train hypergraph H0
# - HGNN https://github.com/iMoonLab/HGNN
# - MST-GCN https://github.com/wudongze9/MST-GCN
# - SIG-Net https://github.com/Noverse0/SIG-Net

import argparse
import time
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
    objects = {}
    for row in read_csv(path):
        node_id = int(row["node_id"])
        if node_id not in nodes:
            nodes[node_id] = row
        action = row["action"]
        object_id = row["object_id"].strip()
        if action in config.OBJECT_ACTIONS and object_id.lower() not in config.MISSING_VALUES:
            family = config.OBJECT_ACTIONS[action]                   # video / assignment / forum
            if node_id not in objects:
                objects[node_id] = set()
            objects[node_id].add(f"{row['course_id']}|{family}|{object_id}")
    if sorted(nodes) != list(range(len(nodes))):
        raise ValueError(f"{path}: node ids are not 0..n-1")
    node_list = []
    for node_id in range(len(nodes)):
        node_list.append(nodes[node_id])
    return node_list, objects


# Course hyperedges, one per course, as (COURSE, course_id, members):
# e_C(c) = {v : course(v) = c}
def course_hyperedges(nodes):
    members = {}
    for node_id in range(len(nodes)):
        course_id = nodes[node_id]["course_id"]
        if course_id not in members:
            members[course_id] = []
        members[course_id].append(node_id)
    hyperedges = []
    for course_id in sorted(members):
        hyperedges.append((COURSE, course_id, members[course_id]))
    return hyperedges


# Object hyperedges, one per object, as (OBJECT, "course|family|object_id", members):
# e_O(o) = {v : v used object o in days 0–34}
def object_hyperedges(objects):
    members = {}
    for node_id in sorted(objects):
        for key in objects[node_id]:
            if key not in members:
                members[key] = []
            members[key].append(node_id)
    hyperedges = []
    for key in sorted(members):
        hyperedges.append((OBJECT, key, members[key]))
    return hyperedges


# User hyperedges, one per learner, as (USER, "user|<user_id>", members), with all of the
# learner's train enrollments whatever their course start:
# e_U(l) = {v : user(v) = l}
def user_hyperedges(nodes):
    members = {}
    for node_id in range(len(nodes)):
        user_id = int(nodes[node_id]["user_id"])
        if user_id not in members:
            members[user_id] = []
        members[user_id].append(node_id)
    hyperedges = []
    for user_id in sorted(members):
        hyperedges.append((USER, f"user|{user_id}", members[user_id]))
    return hyperedges


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

    all_hyperedges = course_hyperedges(nodes) + object_hyperedges(objects) + user_hyperedges(nodes)
    # Keep |e| ≥ 2: a one-member hyperedge carries nothing beyond the node's self-loop.
    hyperedges = []
    for family, key, members in all_hyperedges:
        if len(members) >= 2:
            hyperedges.append((family, key, members))

    # Sparse incidence matrix H0 as a membership list: h(v,e) = 1 for every (node_ids[k], edge_ids[k]).
    node_ids = []
    edge_ids = []
    edge_family = []
    edge_keys = []
    for edge_id in range(len(hyperedges)):
        family, key, members = hyperedges[edge_id]
        edge_family.append(family)
        edge_keys.append(key)
        for node_id in members:
            node_ids.append(node_id)
            edge_ids.append(edge_id)

    train_labels = []
    train_users = []
    for node in nodes:
        train_labels.append(int(node["label"]))
        train_users.append(int(node["user_id"]))

    np.savez_compressed(
        output_dir / HYPERGRAPH_FILE,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        edge_ids=np.asarray(edge_ids, dtype=np.int64),
        edge_family=np.asarray(edge_family, dtype=np.int64),
        edge_keys=np.asarray(edge_keys),
        train_labels=np.asarray(train_labels, dtype=np.float32),
        train_users=np.asarray(train_users, dtype=np.int64),
    )

    # Number of hyperedges of each family, for the log.
    counts = {"course": 0, "object": 0, "user": 0}
    for family, key, members in hyperedges:
        counts[config.EDGE_FAMILIES[family]] += 1
    log(f"saved {output_dir / HYPERGRAPH_FILE}: train nodes={len(nodes):,}, hyperedges={counts}, "
        f"memberships={len(node_ids):,}, elapsed={time.perf_counter() - started_at:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the train hypergraph H0 (Course, Object, User).")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()
    build_hypergraph(arguments.output_dir)
