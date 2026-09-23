# Preprocess the XuetangX dataset.

import argparse
import csv
import gzip
import math
import random
import tarfile
from collections import defaultdict
from datetime import date
from functools import partial
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
ACTIONS = []
for action_group in ACTION_GROUPS.values():
    for action_name in action_group:
        ACTIONS.append(action_name)
ACTIONS = tuple(ACTIONS)

# Save data after preprocessed to save time for after run
PROCESSED_FILES = ("nodes.csv", "users.csv", "courses.csv", "events_35d.csv.gz")
TRUTH_FILES = ("train_truth.csv", "test_truth.csv")
LOG_FILES = ("train_log.csv", "test_log.csv")
CLI_DESCRIPTION = "Preprocess the XuetangX dataset."


# Yield rows from a UTF-8 CSV file as dictionaries.
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as source:
        yield from csv.DictReader(source)


# Write column names and data rows to a UTF-8 CSV file.
def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


# Decode binary archive lines without loading a complete CSV into memory.
def _decode_utf8_lines(binary_source):
    for binary_line in binary_source:
        yield binary_line.decode("utf-8")


# Read selected CSV members in one sequential pass through the archive.
def _read_archive_tables(archive_path, selected_names):
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
                decoded_lines = _decode_utf8_lines(binary)
                for row in csv.DictReader(decoded_lines):
                    yield name, row

    missing = []
    for selected_name in selected_names:
        if selected_name not in found:
            missing.append(selected_name)
    if missing:
        raise FileNotFoundError(
            f"Missing files in prediction_data.tar.gz: {', '.join(missing)}"
        )


# Create an empty enrollment and positive-label counter for one user.
def _empty_user_counts():
    return [0, 0]


# Sort user groups by enrollment count and positive-label count.
def _user_group_sort_key(user_group):
    return user_group[1], user_group[2]


# Sort split indices by largest fractional target remainder first.
def _split_remainder_sort_key(split_index, exact_targets, target_counts):
    fractional_remainder = exact_targets[split_index] - target_counts[split_index]
    return -fractional_remainder, split_index


# Rank available splits by their projected fill ratio.
def _split_fill_sort_key(split_index, used_counts, target_counts):
    projected_fill_ratio = (used_counts[split_index] + 1) / target_counts[split_index]
    return projected_fill_ratio, split_index


# Assign every enrollment of one user to the same experiment split.
def _split_users(nodes, seed):
    groups = defaultdict(_empty_user_counts)
    for node in nodes:
        counts = groups[int(node["user_id"])]
        counts[0] += 1
        counts[1] += int(node["label"])

    ordered_groups = []
    for user_id, user_counts in sorted(groups.items()):
        ordered_groups.append((user_id, *user_counts))
    random.Random(seed).shuffle(ordered_groups)
    ordered_groups.sort(key=_user_group_sort_key, reverse=True)

    exact_targets = []
    for split_ratio in RATIOS:
        exact_targets.append(len(ordered_groups) * split_ratio)
    target_counts = []
    for exact_target in exact_targets:
        target_counts.append(math.floor(exact_target))
    remaining = len(ordered_groups) - sum(target_counts)
    target_order_key = partial(
        _split_remainder_sort_key,
        exact_targets=exact_targets,
        target_counts=target_counts,
    )
    target_order = sorted(
        range(len(SPLITS)),
        key=target_order_key,
    )
    for split_index in target_order[:remaining]:
        target_counts[split_index] += 1

    assigned_split_names = []
    used_counts = [0, 0, 0]
    for user_group in ordered_groups:
        available_splits = []
        for split_index in range(len(SPLITS)):
            if used_counts[split_index] < target_counts[split_index]:
                available_splits.append(split_index)
        split_fill_key = partial(
            _split_fill_sort_key,
            used_counts=used_counts,
            target_counts=target_counts,
        )
        choice = min(
            available_splits,
            key=split_fill_key,
        )
        assigned_split_names.append(SPLITS[choice])
        used_counts[choice] += 1

    split_by_user = {}
    for user_group, split_name in zip(ordered_groups, assigned_split_names):
        split_by_user[user_group[0]] = split_name
    return split_by_user


# Return deterministic split names aligned with consecutive node IDs.
def split_nodes(nodes, seed):
    for index, node in enumerate(nodes):
        if int(node["node_id"]) != index:
            raise ValueError("nodes.csv must be ordered by consecutive node_id")

    split_by_user = _split_users(nodes, seed)
    node_splits = []
    for node in nodes:
        user_id = int(node["user_id"])
        node_splits.append(split_by_user[user_id])
    return node_splits


# Resolve required raw paths and fail if any input file is missing.
def _required_raw_files(raw_dir):
    paths = {
        "prediction_data.tar.gz": raw_dir / "prediction_data.tar.gz",
        "user_info.csv": raw_dir / "user_info.csv",
        "course_info.csv": raw_dir / "course_info.csv",
    }
    missing = []
    for file_name, file_path in paths.items():
        if not file_path.is_file():
            missing.append(file_name)
    if missing:
        raise FileNotFoundError(f"Missing raw files: {', '.join(missing)}")
    return paths


# Load course rows and reject duplicate course identifiers.
def _load_courses(course_path):
    courses = {}
    for row in read_csv(course_path):
        course_id = row["course_id"]
        if course_id in courses:
            raise ValueError(f"Duplicate course: {course_id}")
        courses[course_id] = row
    return courses


# Load enrollment labels from both source partitions in the archive.
def _load_enrollment_labels(archive_path):
    labels = {}
    for truth_name, row in _read_archive_tables(archive_path, TRUTH_FILES):
        source_partition = truth_name.removesuffix("_truth.csv")
        enrollment_key = (int(row["enroll_id"]), source_partition)
        if enrollment_key in labels:
            raise ValueError(f"Repeated label: {enrollment_key}")
        if row["truth"] not in ("0", "1"):
            raise ValueError(f"Invalid label for enrollment: {enrollment_key}")
        labels[enrollment_key] = int(row["truth"])
    return labels


# Validate one raw log row and return its stable enrollment metadata.
def _validate_log_row(row, source_partition, labels, courses):
    enrollment_key = (int(row["enroll_id"]), source_partition)
    if enrollment_key not in labels:
        raise ValueError(f"Log without label: {enrollment_key}")

    user_id = int(row["username"])
    course_id = row["course_id"]
    if course_id not in courses:
        raise ValueError(f"Unknown course: {course_id}")
    if not row["session_id"].strip():
        raise ValueError(f"Missing session for enrollment: {enrollment_key}")
    if row["action"] not in ACTIONS:
        raise ValueError(f"Unknown action for enrollment: {enrollment_key}")
    return enrollment_key, user_id, course_id


# Stream raw logs, retain first-window events, and collect enrollment metadata.
def _stream_events(archive_path, event_path, labels, courses):
    enrollment_metadata = {}
    raw_event_count = 0
    retained_event_count = 0
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
            raw_event_count += 1
            if raw_event_count % 5_000_000 == 0:
                print(f"Read {raw_event_count:,} raw events", flush=True)

            enrollment_key, user_id, course_id = _validate_log_row(
                row,
                source_partition,
                labels,
                courses,
            )

            node_metadata = (user_id, course_id)
            previous_metadata = enrollment_metadata.setdefault(
                enrollment_key,
                node_metadata,
            )
            if previous_metadata != node_metadata:
                raise ValueError(
                    f"Conflicting user or course for: {enrollment_key}"
                )

            event_date = date.fromisoformat(row["time"][:10])
            course_start = date.fromisoformat(courses[course_id]["start"][:10])
            course_day = (event_date - course_start).days
            if 0 <= course_day < OBSERVATION_DAYS:
                writer.writerow((
                    enrollment_key[0], source_partition,
                    row["session_id"], row["action"],
                    row["object"].strip(), course_day, course_id,
                ))
                retained_event_count += 1

    return enrollment_metadata, raw_event_count, retained_event_count


# Convert enrollment metadata into rows with consecutive node identifiers.
def _build_nodes(enrollment_metadata, labels):
    nodes = []
    for node_id, enrollment_key in enumerate(sorted(enrollment_metadata)):
        user_id, course_id = enrollment_metadata[enrollment_key]
        nodes.append({
            "node_id": node_id,
            "enroll_id": enrollment_key[0],
            "user_id": user_id,
            "course_id": course_id,
            "source_partition": enrollment_key[1],
            "label": labels[enrollment_key],
        })
    return nodes


# Write node dictionaries in the stable nodes.csv column order.
def _write_nodes(output_dir, nodes):
    node_columns = (
        "node_id", "enroll_id", "user_id", "course_id", "source_partition", "label"
    )
    node_rows = []
    for node in nodes:
        node_row = []
        for column_name in node_columns:
            node_row.append(node[column_name])
        node_rows.append(tuple(node_row))
    write_csv(output_dir / "nodes.csv", node_columns, node_rows)


# Write metadata only for users referenced by the processed nodes.
def _write_selected_users(user_path, output_dir, nodes):
    used_user_ids = set()
    for node in nodes:
        used_user_ids.add(node["user_id"])

    users_by_id = {}
    for row in read_csv(user_path):
        user_id = int(row["user_id"])
        if user_id not in used_user_ids:
            continue
        if user_id in users_by_id:
            raise ValueError(f"Duplicate user: {user_id}")
        users_by_id[user_id] = row
    if users_by_id.keys() != used_user_ids:
        raise ValueError("Missing user metadata")

    user_rows = []
    for user_id in sorted(users_by_id):
        user = users_by_id[user_id]
        user_rows.append((user_id, user["gender"], user["education"], user["birth"]))
    write_csv(
        output_dir / "users.csv",
        ("user_id", "gender", "education", "birth"),
        user_rows,
    )
    return len(used_user_ids)


# Write metadata only for courses referenced by the processed nodes.
def _write_selected_courses(output_dir, nodes, courses):
    used_course_ids = set()
    for node in nodes:
        used_course_ids.add(node["course_id"])

    course_rows = []
    for course_id in sorted(used_course_ids):
        course = courses[course_id]
        course_rows.append((
            course_id, course["start"], course["end"], course["category"]
        ))
    write_csv(
        output_dir / "courses.csv",
        ("course_id", "start", "end", "category"),
        course_rows,
    )
    return len(used_course_ids)


# Resolve every processed output path in its declared order.
def _processed_output_paths(output_dir):
    output_paths = []
    for file_name in PROCESSED_FILES:
        output_paths.append(output_dir / file_name)
    return output_paths


# Return whether every expected processed output already exists.
def _all_outputs_exist(output_paths):
    for output_path in output_paths:
        if not output_path.is_file():
            return False
    return True


# Stream raw logs and write nodes, users, courses, and 35-day events.
def preprocess(raw_dir=RAW, output_dir=PROCESSED):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_paths = _processed_output_paths(output_dir)
    if _all_outputs_exist(output_paths):
        report = {"skipped": True, "output_dir": str(output_dir)}
        print(report, flush=True)
        return report

    raw_paths = _required_raw_files(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for output_path in output_paths:
        output_path.unlink(missing_ok=True)

    courses = _load_courses(raw_paths["course_info.csv"])
    archive_path = raw_paths["prediction_data.tar.gz"]
    labels = _load_enrollment_labels(archive_path)
    enrollment_metadata, raw_event_count, retained_event_count = _stream_events(
        archive_path,
        output_dir / "events_35d.csv.gz",
        labels,
        courses,
    )

    if enrollment_metadata.keys() != labels.keys():
        raise ValueError("Some labeled enrollments have no log events")

    nodes = _build_nodes(enrollment_metadata, labels)
    _write_nodes(output_dir, nodes)
    user_count = _write_selected_users(raw_paths["user_info.csv"], output_dir, nodes)
    course_count = _write_selected_courses(output_dir, nodes, courses)

    report = {
        "nodes": len(nodes),
        "users": user_count,
        "courses": course_count,
        "raw_events": raw_event_count,
        "events_35d": retained_event_count,
    }
    print(report, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    arguments = parser.parse_args()
    preprocess(arguments.raw_dir, arguments.output_dir)
