# 4. Build one hypergraph H over every enrollment (train + validation + test of every seed)
# - HGNN https://github.com/iMoonLab/HGNN
# - MST-GCN https://github.com/wudongze9/MST-GCN
# H does not depend on the seed: the seed only says which nodes are train / validation / test.

import argparse
import time
from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
preprocess_module = import_module("2_preprocess")
read_csv = preprocess_module.read_csv

HYPERGRAPH_FILE = "hypergraph.npz"
COURSE = config.EDGE_FAMILIES.index("course")
OBJECT = config.EDGE_FAMILIES.index("object")
USER = config.EDGE_FAMILIES.index("user")
SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")


# Print a message with the time
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}][hypergraph] {message}", flush=True)


# Read the course, user and objects of every node from events.csv
def read_events(path):
    nodes = {}         # node_id -> (course_id, user_id)
    objects = {}       # node_id -> set of "course|family|object"
    for row in read_csv(path):
        node_id = int(row["node_id"])
        if node_id not in nodes:
            nodes[node_id] = (row["course_id"], int(row["user_id"]))
        action = row["action"]
        object_id = row["object_id"].strip()
        if action in config.OBJECT_ACTIONS and object_id.lower() not in config.MISSING_VALUES:
            family = config.OBJECT_ACTIONS[action]                   # video / assignment / forum
            if node_id not in objects:
                objects[node_id] = set()
            objects[node_id].add(f"{row['course_id']}|{family}|{object_id}")
    if sorted(nodes) != list(range(len(nodes))):
        raise ValueError(f"{path}: node ids are not 0..n-1")
    return nodes, objects


# Hyperedges of one family from a dict key -> member nodes
def make_hyperedges(family, members):
    hyperedges = []
    for key in sorted(members):
        hyperedges.append((family, key, members[key]))
    return hyperedges


# Course hyperedges: the nodes of one course
def course_hyperedges(nodes):
    members = {}
    for node_id in range(len(nodes)):
        course_id = nodes[node_id][0]
        if course_id not in members:
            members[course_id] = []
        members[course_id].append(node_id)
    return make_hyperedges(COURSE, members)


# Object hyperedges: the nodes that used one object
def object_hyperedges(objects):
    members = {}
    for node_id in sorted(objects):
        for key in objects[node_id]:
            if key not in members:
                members[key] = []
            members[key].append(node_id)
    return make_hyperedges(OBJECT, members)


# User hyperedges: the nodes of one learner
def user_hyperedges(nodes):
    members = {}
    for node_id in range(len(nodes)):
        user_id = nodes[node_id][1]
        if user_id not in members:
            members[user_id] = []
        members[user_id].append(node_id)
    return make_hyperedges(USER, members)


# Run step 4: build the hypergraph
def build_hypergraph(output_dir=config.PROCESSED):
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    log(f"reading {preprocess_module.EVENTS_FILE}")
    nodes, objects = read_events(output_dir / preprocess_module.EVENTS_FILE)

    all_hyperedges = course_hyperedges(nodes) + object_hyperedges(objects) + user_hyperedges(nodes)
    # Keep |e| ≥ 2
    hyperedges = []
    for family, key, members in all_hyperedges:
        if len(members) >= 2:
            hyperedges.append((family, key, members))

    # Sparse incidence matrix H: one (node, hyperedge) pair per membership
    node_ids = []
    edge_ids = []
    edge_family = []
    for edge_id in range(len(hyperedges)):
        family, key, members = hyperedges[edge_id]
        edge_family.append(family)
        for node_id in members:
            node_ids.append(node_id)
            edge_ids.append(edge_id)

    np.savez_compressed(
        output_dir / HYPERGRAPH_FILE,
        num_nodes=len(nodes),
        node_ids=np.asarray(node_ids, dtype=np.int64),
        edge_ids=np.asarray(edge_ids, dtype=np.int64),
        edge_family=np.asarray(edge_family, dtype=np.int64),
    )

    # Number of hyperedges of each family, for the log.
    counts = {"course": 0, "object": 0, "user": 0}
    for family, key, members in hyperedges:
        counts[config.EDGE_FAMILIES[family]] += 1
    log(f"saved {output_dir / HYPERGRAPH_FILE}: nodes={len(nodes):,}, hyperedges={counts}, "
        f"memberships={len(node_ids):,}, elapsed={time.perf_counter() - started_at:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build one hypergraph H (Course, Object, User) over "
                                                 "every enrollment.")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()
    build_hypergraph(arguments.output_dir)
