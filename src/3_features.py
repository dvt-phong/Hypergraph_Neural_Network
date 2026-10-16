# 3. Build the node feature matrix X59 (log activity only) of every seed
# - MST-GCN https://github.com/wudongze9/MST-GCN (the 59 enrollment features)
# - CFIN https://github.com/wzfhaha/dropout_prediction (log1p before the z-score)
#
# X59 = [35 daily event counts | total events | distinct objects | 22 action counts]
# x = (log(1 + c) − μ_train) / σ_train, μ and σ from the train rows of the seed.
# Output data/processed/simple/x59_s{seed}.npz, one row per node (row = node_id):
#   enroll_id, user_id, course_id, label, split ("train" / "validation" / "test"), X [N, 59]
# Every model (M and every baseline) reads this file.

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
preprocess_module = import_module("2_preprocess")
read_csv = preprocess_module.read_csv
write_csv = preprocess_module.write_csv


# Path of the X59 file of one seed
def x59_path(output_dir, seed):
    return Path(output_dir) / f"x59_s{seed}.npz"


# Name and source of the 59 columns
def feature_names():
    names = []
    for day_number in range(config.DAY_FEATURE_COUNT):
        names.append((f"activity_day_{day_number}", f"events on course day {day_number}"))
    names.append(("total_events", "events in course days 0-34 (MST-GCN total_actions)"))
    names.append(("distinct_objects", "distinct objects used in course days 0-34 (MST-GCN unique_objects)"))
    for family in config.ACTION_GROUPS:
        action_number = 1
        for action_name in config.ACTION_GROUPS[family]:
            names.append((f"action_{family}_{action_number}", action_name))
            action_number += 1
    return names


# Count the 59 values of every node, and read its enroll_id, user_id, course_id and label
def count_events(events_path):
    action_columns = {}
    for offset in range(len(config.ACTIONS)):
        action_columns[config.ACTIONS[offset]] = config.ACTION_FEATURE_START + offset

    counts = {}        # node_id -> 59 counts
    objects = {}       # node_id -> set of object ids
    nodes = {}         # node_id -> (enroll_id, user_id, course_id, label)
    for row in read_csv(events_path):
        node_id = int(row["node_id"])
        if node_id not in nodes:
            nodes[node_id] = (int(row["enroll_id"]), int(row["user_id"]), row["course_id"], int(row["label"]))
            counts[node_id] = np.zeros(config.FEATURE_COUNT, dtype=np.int64)
            objects[node_id] = set()

        action = row["action"]
        if not action:
            continue                                                     # enrollment with no event
        course_day = int(row["course_day"])
        counts[node_id][course_day] += 1                                 # events on that day
        counts[node_id][config.TOTAL_EVENTS_INDEX] += 1                  # all events
        counts[node_id][action_columns[action]] += 1                     # events of that action
        object_id = row["object_id"].strip()
        if object_id.lower() not in config.MISSING_VALUES:
            objects[node_id].add(object_id)

    node_count = len(nodes)
    if sorted(nodes) != list(range(node_count)):
        raise ValueError(f"{events_path}: node ids are not 0..n-1")

    count_matrix = np.zeros((node_count, config.FEATURE_COUNT), dtype=np.int64)
    enroll_ids = np.zeros(node_count, dtype=np.int64)
    user_ids = np.zeros(node_count, dtype=np.int64)
    labels = np.zeros(node_count, dtype=np.int64)
    course_ids = []
    for node_id in range(node_count):
        count_matrix[node_id] = counts[node_id]
        count_matrix[node_id, config.OBJECT_COUNT_INDEX] = len(objects[node_id])
        enroll_id, user_id, course_id, label = nodes[node_id]
        enroll_ids[node_id] = enroll_id
        user_ids[node_id] = user_id
        labels[node_id] = label
        course_ids.append(course_id)
    return {"counts": count_matrix, "enroll_id": enroll_ids, "user_id": user_ids,
            "course_id": np.array(course_ids), "label": labels}


# Split name of every node for one seed
def read_split(output_dir, seed, node_count):
    split = np.empty(node_count, dtype="<U10")
    for row in read_csv(preprocess_module.split_path(output_dir, seed)):
        split[int(row["node_id"])] = row["split"]
    return split


# log1p, then z-score with the train rows of the seed
def scale_counts(counts, is_train):
    # x = (log(1 + c) − μ_train) / σ_train
    log_counts = np.log1p(counts.astype(np.float64))                     # log(1 + c)
    mean = np.mean(log_counts[is_train], axis=0)                         # μ_train per column
    std = np.std(log_counts[is_train], axis=0)                           # σ_train per column
    std[std == 0] = 1.0                                                  # constant column -> z = 0
    scaled = (log_counts - mean) / std
    return scaled.astype(np.float32)


# Run step 3: events.csv -> x59_s{seed}.npz for every seed
def build_features(output_dir=config.PROCESSED):
    output_dir = Path(output_dir)
    data = count_events(output_dir / preprocess_module.EVENTS_FILE)
    node_count = len(data["label"])
    print(f"Counted {config.FEATURE_COUNT} columns of {node_count:,} nodes", flush=True)

    for seed in config.SEEDS:
        split = read_split(output_dir, seed, node_count)
        x = scale_counts(data["counts"], split == "train")
        np.savez(
            x59_path(output_dir, seed),
            enroll_id=data["enroll_id"],
            user_id=data["user_id"],
            course_id=data["course_id"],
            label=data["label"],
            split=split,
            X=x,
        )
        print(f"Saved {x59_path(output_dir, seed)}: X {x.shape}", flush=True)

    rows = []
    names = feature_names()
    for index in range(len(names)):
        rows.append((index, names[index][0], names[index][1]))
    write_csv(output_dir / "feature_names.csv", ("feature_index", "feature_name", "source"), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build X59 of every seed.")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()
    build_features(arguments.output_dir)
