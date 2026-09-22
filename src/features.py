"""Make node features from clean CSV events; fit transforms on train nodes only."""

import argparse
import csv
import gzip
import json
from contextlib import ExitStack
from datetime import date
from pathlib import Path

import numpy as np

from preprocess import (ACTIONS, ACTION_GROUPS, PROCESSED, SEEDS, read_csv)


GENDERS = ("female", "male")
EDUCATIONS = ("Associate", "Bachelor's", "Doctorate", "High", "Master's",
              "Middle", "Primary")
CATEGORIES = ("art", "biology", "business", "chemistry", "computer", "economics",
              "education", "electrical", "engineering", "foreign language", "history",
              "literature", "math", "medicine", "philosophy", "physics", "social science")
OBJECT_ACTIONS = {action: family for family, actions in ACTION_GROUPS.items()
                  if family != "web_page" for action in actions}


def feature_names():
    behavior = [f"day_{day:02d}" for day in range(35)]
    behavior += [f"action_{action}" for action in ACTIONS]
    behavior += ["session_count", "distinct_observed_objects"]
    user = ["gender_female", "gender_male", "gender_missing", "gender_other"]
    user += ["education_" + value.lower().replace("'", "").replace(" ", "_")
             for value in EDUCATIONS]
    user += ["education_missing", "education_other", "age_at_course_start", "age_missing"]
    course = [f"category_{value.replace(' ', '_')}" for value in CATEGORIES]
    course += ["category_missing", "category_other", "course_duration_days",
               "course_duration_missing"]
    return behavior + user + course


def _one_hot(matrix, row, value, vocabulary, start):
    if not value:
        index = len(vocabulary)
    else:
        try:
            index = vocabulary.index(value)
        except ValueError:
            index = len(vocabulary) + 1
    matrix[row, start + index] = 1.0


def _scale_numeric(values, train_ids):
    missing = ~np.isfinite(values)
    observed = values[train_ids][~missing[train_ids]]
    if observed.size == 0:
        raise ValueError("No observed train values for context feature")
    median = float(np.median(observed))
    filled = np.where(missing, median, values)
    mean = float(np.mean(filled[train_ids], dtype=np.float64))
    std = float(np.std(filled[train_ids], dtype=np.float64)) or 1.0
    return ((filled - mean) / std).astype(np.float32), missing.astype(np.float32), {
        "median": median, "mean": mean, "std": std,
    }


def build_features(output_dir=PROCESSED, *, buckets=64, seeds=SEEDS):
    output_dir = Path(output_dir)
    nodes = list(read_csv(output_dir / "nodes.csv"))
    count = len(nodes)
    node_by_enrollment = {(int(row["enroll_id"]), row["source_partition"]):
                          int(row["node_id"]) for row in nodes}
    if len(node_by_enrollment) != count:
        raise ValueError("Enrollment keys are not unique")
    raw = np.zeros((count, 60), dtype=np.int32)
    action_index = {action: 35 + i for i, action in enumerate(ACTIONS)}

    # All events of one node go to one small file. This keeps distinct session and
    # object counts exact without holding sets for 40 million events in memory.
    temp_dir = output_dir / "feature_buckets"
    temp_dir.mkdir(exist_ok=True)
    bucket_paths = [temp_dir / f"bucket_{i:02d}.csv" for i in range(buckets)]
    total_events = 0
    with ExitStack() as stack:
        writers = []
        for path in bucket_paths:
            handle = stack.enter_context(open(path, "w", newline="", encoding="utf-8"))
            writers.append(csv.writer(handle))
        with gzip.open(output_dir / "events_35d.csv.gz", "rt", newline="",
                       encoding="utf-8") as source:
            for event in csv.DictReader(source):
                key = (int(event["enroll_id"]), event["source_partition"])
                node_id = node_by_enrollment[key]
                day = int(event["course_day"])
                if not 0 <= day < 35:
                    raise ValueError(f"Invalid course day: {day}")
                action = event["action"]
                raw[node_id, day] += 1
                raw[node_id, action_index[action]] += 1
                writers[node_id % buckets].writerow((node_id, event["session_id"],
                    event["course_id"], event["object_id"], OBJECT_ACTIONS.get(action, "")))
                total_events += 1
                if total_events % 5_000_000 == 0:
                    print(f"Aggregated {total_events:,} events", flush=True)

    object_path = output_dir / "node_objects.csv.gz"
    with gzip.open(object_path, "wt", newline="", encoding="utf-8",
                   compresslevel=1) as target:
        writer = csv.writer(target)
        writer.writerow(("node_id", "course_id", "object_id", "object_type"))
        for path in bucket_paths:
            sessions = {}
            observed = {}
            memberships = set()
            with open(path, newline="", encoding="utf-8") as source:
                for node_text, session, course, object_id, object_type in csv.reader(source):
                    node_id = int(node_text)
                    sessions.setdefault(node_id, set()).add(session)
                    if object_id:
                        observed.setdefault(node_id, set()).add((course, object_id))
                        if object_type:
                            memberships.add((node_id, course, object_id, object_type))
            for node_id, values in sessions.items():
                raw[node_id, 58] = len(values)
            for node_id, values in observed.items():
                raw[node_id, 59] = len(values)
            writer.writerows(sorted(memberships))
            path.unlink()
    temp_dir.rmdir()
    if int(raw[:, :35].sum()) != total_events or int(raw[:, 35:58].sum()) != total_events:
        raise ValueError("Daily/action feature sums do not match the event count")

    users = {int(row["user_id"]): row for row in read_csv(output_dir / "users.csv")}
    courses = {row["course_id"]: row for row in read_csv(output_dir / "courses.csv")}
    context = np.zeros((count, 36), dtype=np.float32)
    ages = np.full(count, np.nan, dtype=np.float64)
    durations = np.full(count, np.nan, dtype=np.float64)
    for row in nodes:
        node_id = int(row["node_id"])
        user, course = users[int(row["user_id"])], courses[row["course_id"]]
        _one_hot(context, node_id, user["gender"].strip(), GENDERS, 0)
        _one_hot(context, node_id, user["education"].strip(), EDUCATIONS, 4)
        _one_hot(context, node_id, course["category"].strip(), CATEGORIES, 15)
        start = date.fromisoformat(course["start"][:10])
        duration = (date.fromisoformat(course["end"][:10]) - start).days
        if duration >= 0:
            durations[node_id] = duration
        if user["birth"]:
            age = start.year - int(float(user["birth"]))
            if 10 <= age <= 100:
                ages[node_id] = age

    logged = np.log1p(raw.astype(np.float32))
    for seed in seeds:
        split = list(read_csv(output_dir / f"split_seed_{seed}.csv"))
        train_ids = np.array([int(row["node_id"]) for row in split
                              if row["split"] == "train"], dtype=np.int64)
        mean = np.mean(logged[train_ids], axis=0, dtype=np.float64).astype(np.float32)
        std = np.std(logged[train_ids], axis=0, dtype=np.float64).astype(np.float32)
        std[std == 0] = 1
        x = np.empty((count, 96), dtype=np.float32)
        x[:, :60] = (logged - mean) / std
        x[:, 60:] = context
        age, age_missing, age_stats = _scale_numeric(ages, train_ids)
        duration, duration_missing, duration_stats = _scale_numeric(durations, train_ids)
        x[:, 73], x[:, 74] = age, age_missing
        x[:, 94], x[:, 95] = duration, duration_missing
        if not np.isfinite(x).all():
            raise ValueError("Node features contain NaN or infinity")
        np.save(output_dir / f"X_seed_{seed}.npy", x)
        (output_dir / f"feature_stats_seed_{seed}.json").write_text(json.dumps({
            "names": feature_names(), "behavior_mean": mean.tolist(),
            "behavior_std": std.tolist(), "age": age_stats, "duration": duration_stats,
            "train_nodes": len(train_ids),
        }, indent=2), encoding="utf-8")
    report = {"nodes": count, "events": total_events, "features": 96,
              "object_memberships": str(object_path)}
    print(report, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--buckets", type=int, default=64)
    args = parser.parse_args()
    build_features(args.output_dir, buckets=args.buckets)
