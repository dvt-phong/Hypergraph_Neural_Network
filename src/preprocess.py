# Preprocess dataset XuetangX

import argparse
import csv
import gzip
import math
import random
import tarfile
from collections import defaultdict
from datetime import date
from pathlib import Path

# Define some constant
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xuetangx"
PROCESSED = ROOT / "data" / "processed" / "simple"

# Seed for train and result is mean +- std
SEEDS = (1, 11, 111, 1111, 11111)

# Split dataset into 3 set train/validation/test
# With 64% 16% 20%
SPLITS = ("train", "validation", "test")
RATIOS = (0.64, 0.16, 0.20)

# Event in 0...34 days
OBSERVATION_DAYS = 35

# Define action in log activity
ACTION_GROUPS = {
    "video": (
        "seek_video", "play_video", "pause_video", "stop_video", "load_video"
    ),
    "assignment": (
        "problem_get", "problem_check", "problem_save", "reset_problem",
        "problem_check_correct", "problem_check_incorrect",
    ),
    "forum": (
        "create_thread", "create_comment", "delete_thread", "delete_comment",
        "close_forum",
    ),
    "web_page": (
        "click_info", "click_courseware", "click_about", "click_forum",
        "click_progress", "close_courseware", "close_info",
    ),
}
ACTIONS = tuple(action for group in ACTION_GROUPS.values() for action in group)

# Save data after preprocessed to save time for after run
PROCESSED_FILES = ("nodes.csv", "users.csv", "courses.csv", "events_35d.csv.gz")
TRUTH_FILES = ("train_truth.csv", "test_truth.csv")
LOG_FILES = ("train_log.csv", "test_log.csv")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as source:
        yield from csv.DictReader(source)


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


def _read_archive_tables(archive_path, selected_names):
    """Read selected CSV members in one sequential pass through the archive."""
    found = set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in selected_names:
                continue

            binary = archive.extractfile(member)
            if binary is None:
                raise OSError(f"Cannot read {member.name} from prediction_data.tar.gz")
            found.add(name)
            with binary:
                decoded_lines = (line.decode("utf-8") for line in binary)
                for row in csv.DictReader(decoded_lines):
                    yield name, row

    missing = [name for name in selected_names if name not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing files in prediction_data.tar.gz: {', '.join(missing)}"
        )


def _split_users(nodes, seed):
    """Assign every enrollment of one user to the same experiment split."""
    groups = defaultdict(lambda: [0, 0])
    for node in nodes:
        counts = groups[int(node["user_id"])]
        counts[0] += 1
        counts[1] += int(node["label"])

    ordered = [(user_id, *counts) for user_id, counts in sorted(groups.items())]
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda group: (group[1], group[2]), reverse=True)

    exact_targets = [len(ordered) * ratio for ratio in RATIOS]
    targets = [math.floor(number) for number in exact_targets]
    remaining = len(ordered) - sum(targets)
    target_order = sorted(
        range(len(SPLITS)),
        key=lambda index: (-(exact_targets[index] - targets[index]), index),
    )
    for index in target_order[:remaining]:
        targets[index] += 1

    slots = []
    used = [0, 0, 0]
    for _ in ordered:
        available = [index for index in range(3) if used[index] < targets[index]]
        choice = min(
            available,
            key=lambda index: ((used[index] + 1) / targets[index], index),
        )
        slots.append(SPLITS[choice])
        used[choice] += 1

    assignment = {}
    for group, split_name in zip(ordered, slots):
        assignment[group[0]] = split_name
    return assignment


def split_nodes(nodes, seed):
    """Return deterministic split names aligned with consecutive node IDs."""
    for index, node in enumerate(nodes):
        if int(node["node_id"]) != index:
            raise ValueError("nodes.csv must be ordered by consecutive node_id")

    assignment = _split_users(nodes, seed)
    return [assignment[int(node["user_id"])] for node in nodes]


def _required_raw_files(raw_dir):
    paths = {
        "prediction_data.tar.gz": raw_dir / "prediction_data.tar.gz",
        "user_info.csv": raw_dir / "user_info.csv",
        "course_info.csv": raw_dir / "course_info.csv",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {', '.join(missing)}")
    return paths


def preprocess(raw_dir=RAW, output_dir=PROCESSED):
    """Stream raw logs and write nodes, users, courses, and 35-day events."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_paths = [output_dir / name for name in PROCESSED_FILES]
    if all(path.is_file() for path in output_paths):
        report = {"skipped": True, "output_dir": str(output_dir)}
        print(report, flush=True)
        return report

    raw_paths = _required_raw_files(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_paths:
        path.unlink(missing_ok=True)

    courses = {}
    for row in read_csv(raw_paths["course_info.csv"]):
        course_id = row["course_id"]
        if course_id in courses:
            raise ValueError(f"Duplicate course: {course_id}")
        courses[course_id] = row

    labels = {}
    metadata = {}
    raw_events = 0
    retained_events = 0
    archive_path = raw_paths["prediction_data.tar.gz"]
    for truth_name, row in _read_archive_tables(archive_path, TRUTH_FILES):
        source_partition = truth_name.removesuffix("_truth.csv")
        key = (int(row["enroll_id"]), source_partition)
        if key in labels:
            raise ValueError(f"Repeated label: {key}")
        if row["truth"] not in ("0", "1"):
            raise ValueError(f"Invalid label for enrollment: {key}")
        labels[key] = int(row["truth"])

    event_path = output_dir / "events_35d.csv.gz"
    with gzip.open(
        event_path,
        "wt",
        newline="",
        encoding="utf-8",
        compresslevel=1,
    ) as target:
        writer = csv.writer(target)
        writer.writerow((
            "enroll_id", "source_partition", "session_id", "action",
            "object_id", "course_day", "course_id",
        ))

        for log_name, row in _read_archive_tables(archive_path, LOG_FILES):
            source_partition = log_name.removesuffix("_log.csv")
            raw_events += 1
            if raw_events % 5_000_000 == 0:
                print(f"Read {raw_events:,} raw events", flush=True)

            key = (int(row["enroll_id"]), source_partition)
            if key not in labels:
                raise ValueError(f"Log without label: {key}")

            user_id = int(row["username"])
            course_id = row["course_id"]
            if course_id not in courses:
                raise ValueError(f"Unknown course: {course_id}")
            if not row["session_id"].strip():
                raise ValueError(f"Missing session for enrollment: {key}")
            if row["action"] not in ACTIONS:
                raise ValueError(f"Unknown action for enrollment: {key}")

            node_metadata = (user_id, course_id)
            previous_metadata = metadata.setdefault(key, node_metadata)
            if previous_metadata != node_metadata:
                raise ValueError(f"Conflicting user or course for: {key}")

            event_date = date.fromisoformat(row["time"][:10])
            course_start = date.fromisoformat(courses[course_id]["start"][:10])
            course_day = (event_date - course_start).days
            if 0 <= course_day < OBSERVATION_DAYS:
                writer.writerow((
                    key[0], source_partition, row["session_id"], row["action"],
                    row["object"].strip(), course_day, course_id,
                ))
                retained_events += 1

    if metadata.keys() != labels.keys():
        raise ValueError("Some labeled enrollments have no log events")

    nodes = []
    for node_id, key in enumerate(sorted(metadata)):
        user_id, course_id = metadata[key]
        nodes.append({
            "node_id": node_id,
            "enroll_id": key[0],
            "user_id": user_id,
            "course_id": course_id,
            "source_partition": key[1],
            "label": labels[key],
        })

    node_columns = (
        "node_id", "enroll_id", "user_id", "course_id", "source_partition", "label"
    )
    node_rows = []
    for node in nodes:
        node_rows.append(tuple(node[column] for column in node_columns))
    write_csv(output_dir / "nodes.csv", node_columns, node_rows)

    used_users = {node["user_id"] for node in nodes}
    users = {}
    for row in read_csv(raw_paths["user_info.csv"]):
        user_id = int(row["user_id"])
        if user_id not in used_users:
            continue
        if user_id in users:
            raise ValueError(f"Duplicate user: {user_id}")
        users[user_id] = row
    if users.keys() != used_users:
        raise ValueError("Missing user metadata")

    user_rows = []
    for user_id in sorted(users):
        user = users[user_id]
        user_rows.append((user_id, user["gender"], user["education"], user["birth"]))
    write_csv(
        output_dir / "users.csv",
        ("user_id", "gender", "education", "birth"),
        user_rows,
    )

    used_courses = {node["course_id"] for node in nodes}
    course_rows = []
    for course_id in sorted(used_courses):
        course = courses[course_id]
        course_rows.append((
            course_id, course["start"], course["end"], course["category"]
        ))
    write_csv(
        output_dir / "courses.csv",
        ("course_id", "start", "end", "category"),
        course_rows,
    )

    report = {
        "nodes": len(nodes),
        "users": len(used_users),
        "courses": len(used_courses),
        "raw_events": raw_events,
        "events_35d": retained_events,
    }
    print(report, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    arguments = parser.parse_args()
    preprocess(arguments.raw_dir, arguments.output_dir)
