"""Clean raw XuetangX CSV files and split enrollment nodes by user."""

import argparse
import csv
import gzip
import math
import random
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xuetangx"
PROCESSED = ROOT / "data" / "processed" / "simple"
SEEDS = (1, 11, 111, 1111, 11111)
SPLITS = ("train", "validation", "test")
RATIOS = (0.64, 0.16, 0.20)
OBSERVATION_DAYS = 35
ACTION_GROUPS = {
    "video": ("seek_video", "play_video", "pause_video", "stop_video", "load_video"),
    "assignment": ("problem_get", "problem_check", "problem_save", "reset_problem",
                   "problem_check_correct", "problem_check_incorrect"),
    "forum": ("create_thread", "create_comment", "delete_thread", "delete_comment",
              "close_forum"),
    "web_page": ("click_info", "click_courseware", "click_about", "click_forum",
                 "click_progress", "close_courseware", "close_info"),
}
ACTIONS = tuple(action for group in ACTION_GROUPS.values() for action in group)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as source:
        yield from csv.DictReader(source)


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


def _split_users(nodes, seed):
    """Keep a user's enrollments together and balance user counts and labels."""
    groups = defaultdict(lambda: [0, 0])
    for node in nodes:
        counts = groups[node["user_id"]]
        counts[0] += 1
        counts[1] += node["label"]
    ordered = [(user, *counts) for user, counts in sorted(groups.items())]
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda group: (group[1], group[2]), reverse=True)

    exact = [len(ordered) * ratio for ratio in RATIOS]
    targets = [math.floor(number) for number in exact]
    for index in sorted(range(3), key=lambda i: (-(exact[i] - targets[i]), i))[
        :len(ordered) - sum(targets)
    ]:
        targets[index] += 1
    slots = []
    used = [0, 0, 0]
    for _ in ordered:
        choice = min(
            (i for i in range(3) if used[i] < targets[i]),
            key=lambda i: ((used[i] + 1) / targets[i], i),
        )
        slots.append(SPLITS[choice])
        used[choice] += 1
    return {user: split for (user, _, _), split in zip(ordered, slots)}


def preprocess(raw_dir=RAW, output_dir=PROCESSED, *, check_full_counts=True):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The course start date defines course_day, so read it before streaming logs.
    courses = {}
    for row in read_csv(raw_dir / "course_info.csv"):
        course_id = row["course_id"]
        if course_id in courses:
            raise ValueError(f"Duplicate course {course_id}")
        courses[course_id] = row

    labels = {}
    for source in ("train", "test"):
        for row in read_csv(raw_dir / f"{source}_truth.csv"):
            key = (int(row["enroll_id"]), source)
            if key in labels or row["truth"] not in ("0", "1"):
                raise ValueError(f"Invalid or repeated label: {key}")
            labels[key] = int(row["truth"])

    metadata = {}
    raw_events = retained_events = 0
    event_path = output_dir / "events_35d.csv.gz"
    with gzip.open(event_path, "wt", newline="", encoding="utf-8", compresslevel=1) as target:
        writer = csv.writer(target)
        writer.writerow(("enroll_id", "source_partition", "session_id", "action",
                         "object_id", "course_day", "course_id"))
        for source in ("train", "test"):
            for row in read_csv(raw_dir / f"{source}_log.csv"):
                raw_events += 1
                if raw_events % 5_000_000 == 0:
                    print(f"Read {raw_events:,} raw events", flush=True)
                key = (int(row["enroll_id"]), source)
                if key not in labels:
                    raise ValueError(f"Log without label: {key}")
                user_id = int(row["username"])
                course_id = row["course_id"]
                if course_id not in courses:
                    raise ValueError(f"Unknown course: {course_id}")
                if not row["session_id"].strip() or row["action"] not in ACTIONS:
                    raise ValueError(f"Invalid event: {key}")
                existing = metadata.setdefault(key, (user_id, course_id))
                if existing != (user_id, course_id):
                    raise ValueError(f"Conflicting user/course for {key}")
                course_day = (
                    date.fromisoformat(row["time"][:10])
                    - date.fromisoformat(courses[course_id]["start"][:10])
                ).days
                if 0 <= course_day < OBSERVATION_DAYS:
                    writer.writerow((key[0], source, row["session_id"], row["action"],
                                     row["object"].strip(), course_day, course_id))
                    retained_events += 1

    if metadata.keys() != labels.keys():
        raise ValueError("Some labeled enrollments have no log events")
    nodes = []
    for node_id, key in enumerate(sorted(metadata)):
        user_id, course_id = metadata[key]
        nodes.append({"node_id": node_id, "enroll_id": key[0], "user_id": user_id,
                      "course_id": course_id, "source_partition": key[1],
                      "label": labels[key]})
    write_csv(output_dir / "nodes.csv",
              ("node_id", "enroll_id", "user_id", "course_id", "source_partition", "label"),
              ((node[column] for column in ("node_id", "enroll_id", "user_id",
                                            "course_id", "source_partition", "label"))
               for node in nodes))

    used_users = {node["user_id"] for node in nodes}
    users = {}
    for row in read_csv(raw_dir / "user_info.csv"):
        user_id = int(row["user_id"])
        if user_id in used_users:
            if user_id in users:
                raise ValueError(f"Duplicate user {user_id}")
            users[user_id] = row
    if users.keys() != used_users:
        raise ValueError("Missing user metadata")
    write_csv(output_dir / "users.csv", ("user_id", "gender", "education", "birth"),
              ((user_id, users[user_id]["gender"], users[user_id]["education"],
                users[user_id]["birth"]) for user_id in sorted(users)))

    used_courses = {node["course_id"] for node in nodes}
    write_csv(output_dir / "courses.csv",
              ("course_id", "start", "end", "category"),
              ((course_id, courses[course_id]["start"], courses[course_id]["end"],
                courses[course_id]["category"]) for course_id in sorted(used_courses)))

    split_counts = {}
    for seed in SEEDS:
        assignment = _split_users(nodes, seed)
        split_counts[seed] = {name: 0 for name in SPLITS}
        for node in nodes:
            split_counts[seed][assignment[node["user_id"]]] += 1
        write_csv(output_dir / f"split_seed_{seed}.csv", ("node_id", "split"),
                  ((node["node_id"], assignment[node["user_id"]]) for node in nodes))

    if check_full_counts and (len(nodes), len(used_users), len(used_courses),
                              raw_events, retained_events) != (
                                  225642, 77083, 247, 42110402, 40558640):
        raise ValueError("XuetangX-247 counts differ from the recorded dataset")
    report = {"nodes": len(nodes), "users": len(used_users), "courses": len(used_courses),
              "raw_events": raw_events, "events_35d": retained_events,
              "split_counts": split_counts}
    print(report, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()
    preprocess(args.raw_dir, args.output_dir, check_full_counts=not args.allow_small)
