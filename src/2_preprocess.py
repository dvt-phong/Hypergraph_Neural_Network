# 2. Preprocess XuetangX and preserve its official test partition.

import argparse
import codecs
import csv
import gzip
import random
import tarfile
from datetime import date
from importlib import import_module
from pathlib import Path

config = import_module("0_config")


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


# Read selected CSV members in one sequential pass through the archive.
def _read_archive_tables(archive_path, selected_names):
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in selected_names:
                continue

            binary = archive.extractfile(member)
            with binary:
                text = codecs.getreader("utf-8")(binary)
                for row in csv.DictReader(text):
                    yield name, row


# Split only official-train enrollments into train and validation.
def split_train_enrollments(enrollment_ids):
    shuffled_ids = list(enrollment_ids)
    random.Random(config.SPLIT_SEED).shuffle(shuffled_ids)
    train_count = int(len(shuffled_ids) * config.TRAIN_RATIO)

    split_by_enrollment = {}
    for position, enrollment_id in enumerate(shuffled_ids):
        if position < train_count:
            split_by_enrollment[enrollment_id] = "train"
        else:
            split_by_enrollment[enrollment_id] = "validation"
    return split_by_enrollment


# Stream both raw logs into their final train, validation, and test files.
def _stream_events(
    archive_path,
    output_dir,
    courses,
    split_by_enrollment,
    node_by_enrollment,
):
    event_counts = {"train": 0, "validation": 0, "test": 0}
    enrollment_metadata = {}
    raw_event_count = 0
    event_columns = (
        "node_id", "action", "object_id", "course_day", "course_id",
    )

    train_path = output_dir / "train" / "events_35d.csv.gz"
    validation_path = output_dir / "validation" / "events_35d.csv.gz"
    test_path = output_dir / "test" / "events_35d.csv.gz"
    with gzip.open(
        train_path, "wt", newline="", encoding="utf-8", compresslevel=1
    ) as train_target, gzip.open(
        validation_path, "wt", newline="", encoding="utf-8", compresslevel=1
    ) as validation_target, gzip.open(
        test_path, "wt", newline="", encoding="utf-8", compresslevel=1
    ) as test_target:
        writers = {
            "train": csv.writer(train_target),
            "validation": csv.writer(validation_target),
            "test": csv.writer(test_target),
        }
        for writer in writers.values():
            writer.writerow(event_columns)

        for log_name, row in _read_archive_tables(archive_path, config.LOG_FILES):
            raw_event_count += 1
            enrollment_id = int(row["enroll_id"])
            if log_name == "test_log.csv":
                split_name = "test"
            else:
                split_name = split_by_enrollment[enrollment_id]

            user_id = int(row["username"])
            course_id = row["course_id"]
            enrollment_metadata[enrollment_id] = (user_id, course_id)

            action = row["action"]
            if action not in config.ACTIONS:
                continue

            event_date = date.fromisoformat(row["time"][:10])
            course_start = date.fromisoformat(courses[course_id]["start"][:10])
            course_day = (event_date - course_start).days
            if 0 <= course_day < config.OBSERVATION_DAYS:
                writers[split_name].writerow((
                    node_by_enrollment[enrollment_id],
                    action,
                    row["object"].strip(),
                    course_day,
                    course_id,
                ))
                event_counts[split_name] += 1

    return enrollment_metadata, raw_event_count, event_counts


# Run preprocessing and write explicit train, validation, and test datasets.
def preprocess(raw_dir=config.RAW, output_dir=config.PROCESSED):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    archive_path = raw_dir / "prediction_data.tar.gz"
    user_path = raw_dir / "user_info.csv"
    course_path = raw_dir / "course_info.csv"
    user_output_path = output_dir / "users.csv"
    course_output_path = output_dir / "courses.csv"

    output_paths = [user_output_path, course_output_path]
    for split_name in config.SPLITS:
        split_dir = output_dir / split_name
        output_paths.append(split_dir / "nodes.csv")
        output_paths.append(split_dir / "events_35d.csv.gz")
    outputs_exist = True
    for path in output_paths:
        if not path.is_file():
            outputs_exist = False
    if outputs_exist:
        report = {"skipped": True, "output_dir": str(output_dir)}
        print(report, flush=True)
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in config.SPLITS:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    courses = {}
    for row in read_csv(course_path):
        courses[row["course_id"]] = row

    train_labels = {}
    test_labels = {}
    for truth_name, row in _read_archive_tables(archive_path, config.TRUTH_FILES):
        enrollment_id = int(row["enroll_id"])
        label = int(row["truth"])
        if truth_name == "train_truth.csv":
            train_labels[enrollment_id] = label
        else:
            test_labels[enrollment_id] = label

    split_by_enrollment = split_train_enrollments(sorted(train_labels))
    for enrollment_id in test_labels:
        split_by_enrollment[enrollment_id] = "test"

    enrollments_by_split = {"train": [], "validation": [], "test": []}
    for enrollment_id in sorted(split_by_enrollment):
        split_name = split_by_enrollment[enrollment_id]
        enrollments_by_split[split_name].append(enrollment_id)

    node_by_enrollment = {}
    for split_name in config.SPLITS:
        for node_id, enrollment_id in enumerate(enrollments_by_split[split_name]):
            node_by_enrollment[enrollment_id] = node_id

    enrollment_metadata, raw_event_count, event_counts = _stream_events(
        archive_path,
        output_dir,
        courses,
        split_by_enrollment,
        node_by_enrollment,
    )

    labels = {}
    labels.update(train_labels)
    labels.update(test_labels)
    used_user_ids = set()
    used_course_ids = set()
    node_counts = {}
    for split_name in config.SPLITS:
        node_rows = []
        for enrollment_id in enrollments_by_split[split_name]:
            user_id, course_id = enrollment_metadata[enrollment_id]
            node_rows.append((
                node_by_enrollment[enrollment_id],
                enrollment_id,
                user_id,
                course_id,
                labels[enrollment_id],
            ))
            used_user_ids.add(user_id)
            used_course_ids.add(course_id)
        write_csv(
            output_dir / split_name / "nodes.csv",
            ("node_id", "enroll_id", "user_id", "course_id", "label"),
            node_rows,
        )
        node_counts[split_name] = len(node_rows)

    user_rows = []
    for row in read_csv(user_path):
        user_id = int(row["user_id"])
        if user_id in used_user_ids:
            user_rows.append((
                user_id,
                row["gender"],
                row["education"],
                row["birth"],
            ))
    write_csv(
        user_output_path,
        ("user_id", "gender", "education", "birth"),
        user_rows,
    )

    course_rows = []
    for course_id in sorted(used_course_ids):
        course = courses[course_id]
        course_rows.append((
            course_id,
            course["start"],
            course["end"],
            course["category"],
        ))
    write_csv(
        course_output_path,
        ("course_id", "start", "end", "category"),
        course_rows,
    )

    report = {
        "nodes": node_counts,
        "users": len(user_rows),
        "courses": len(course_rows),
        "raw_events": raw_event_count,
        "events_35d": event_counts,
    }
    print(report, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config.PREPROCESS_CLI_DESCRIPTION)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()
    preprocess(arguments.raw_dir, arguments.output_dir)
