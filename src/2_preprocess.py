# 2. Write the events of every labeled XuetangX enrollment and the split of every seed
# - MST-GCN https://github.com/wudongze9/MST-GCN
# - CA-TFHN https://github.com/codeds27/CA-TFHN
# - CFIN https://github.com/wzfhaha/dropout_prediction

import argparse
import codecs
import csv
import tarfile
from datetime import date
from importlib import import_module
from pathlib import Path

from sklearn.model_selection import train_test_split

config = import_module("0_config")

EVENTS_FILE = "events.csv"


# Read a UTF-8 CSV file row by row, one dict per row
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        for row in reader:
            yield row


# Write a UTF-8 CSV file from column names and rows
def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


# Read selected CSV files inside prediction_data.tar.gz
def read_prediction_data(prediction_data_path, selected_names):
    with tarfile.open(prediction_data_path, "r|gz") as prediction_data:
        for member in prediction_data:
            name = Path(member.name).name
            if not member.isfile() or name not in selected_names:
                continue

            with prediction_data.extractfile(member) as binary:
                text = codecs.getreader("utf-8")(binary)                 # bytes -> text
                for row in csv.DictReader(text):
                    yield name, row


# Path of the split file of one seed
def split_path(output_dir, seed):
    return Path(output_dir) / "splits" / f"s{seed}.csv"


# Split of one seed: every labeled enrollment -> "train" (60%), "validation" (10%) or "test" (30%),
# stratified by label (the same dropout rate in the three parts), as DP-SCL
def split_enrollments(labels, seed):
    enrollment_ids = sorted(labels)
    y = []
    for enrollment_id in enrollment_ids:
        y.append(labels[enrollment_id])
    # n_test = ⌈0.3·n⌉, drawn separately from the dropouts and the non-dropouts
    rest_ids, test_ids = train_test_split(enrollment_ids, test_size=config.TEST_RATIO, stratify=y,
                                          random_state=seed)

    rest_y = []
    for enrollment_id in rest_ids:
        rest_y.append(labels[enrollment_id])
    # validation share of the rest = 0.1 / (0.6 + 0.1) = 1/7, so n_validation ≈ 0.1·n
    validation_share = config.VALIDATION_RATIO / (config.TRAIN_RATIO + config.VALIDATION_RATIO)
    train_ids, validation_ids = train_test_split(rest_ids, test_size=validation_share, stratify=rest_y,
                                                 random_state=seed)

    split_by_enrollment = {}
    for enrollment_id in train_ids:
        split_by_enrollment[enrollment_id] = "train"
    for enrollment_id in validation_ids:
        split_by_enrollment[enrollment_id] = "validation"
    for enrollment_id in test_ids:
        split_by_enrollment[enrollment_id] = "test"
    return split_by_enrollment


# Write the kept events of every labeled enrollment to events.csv
def write_events(prediction_data_path, output_dir, courses, labels, node_ids):
    columns = ("node_id", "enroll_id", "user_id", "course_id", "label", "action", "object_id", "course_day")
    first_rows = {}            # enroll_id -> (user_id, course_id) of its first raw row
    has_event = set()          # enroll_ids with at least one kept event
    event_count = 0

    with open(output_dir / EVENTS_FILE, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)

        for _, row in read_prediction_data(prediction_data_path, config.LOG_FILES):
            enrollment_id = int(row["enroll_id"])
            if enrollment_id not in labels:
                continue
            user_id = int(row["username"])
            course_id = row["course_id"]
            if enrollment_id not in first_rows:
                first_rows[enrollment_id] = (user_id, course_id)

            action = row["action"]
            if action not in config.ACTIONS:
                continue
            event_date = date.fromisoformat(row["time"][:10])
            course_start = date.fromisoformat(courses[course_id]["start"][:10])
            course_day = (event_date - course_start).days                # d = date(event) − date(course start)
            if course_day < 0 or course_day >= config.OBSERVATION_DAYS:  # keep 0 ≤ d < 35
                continue

            writer.writerow((
                node_ids[enrollment_id],
                enrollment_id,
                user_id,
                course_id,
                labels[enrollment_id],
                action,
                row["object"].strip(),
                course_day,
            ))
            has_event.add(enrollment_id)
            event_count += 1

        # Enrollments with no kept event: one row with an empty action (0 in every count).
        empty_count = 0
        for enrollment_id in sorted(labels):
            if enrollment_id in has_event:
                continue
            if enrollment_id not in first_rows:
                raise ValueError(f"Enrollment {enrollment_id} has a label but no raw log row")
            user_id, course_id = first_rows[enrollment_id]
            writer.writerow((
                node_ids[enrollment_id],
                enrollment_id,
                user_id,
                course_id,
                labels[enrollment_id],
                "",
                "",
                "",
            ))
            empty_count += 1

    print(f"Wrote {event_count:,} events of {len(labels):,} enrollments "
          f"({empty_count:,} with no event) to {output_dir / EVENTS_FILE}", flush=True)


# Write splits/s{seed}.csv for every seed
def write_splits(output_dir, node_ids, labels):
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    for seed in config.SEEDS:
        split_by_enrollment = split_enrollments(labels, seed)
        rows = []
        for enrollment_id in sorted(split_by_enrollment):
            rows.append((node_ids[enrollment_id], enrollment_id, split_by_enrollment[enrollment_id]))
        write_csv(split_path(output_dir, seed), ("node_id", "enroll_id", "split"), rows)

        counts = {}
        for split_name in config.SPLITS:
            counts[split_name] = 0
        for enrollment_id in split_by_enrollment:
            counts[split_by_enrollment[enrollment_id]] += 1
        print(f"Saved {split_path(output_dir, seed)}: {counts}", flush=True)


# Run step 2: raw files -> events.csv and the split files (skipped when they already exist; an
# existing events.csv is kept, it does not depend on the split)
def preprocess(raw_dir=config.RAW, output_dir=config.PROCESSED):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    prediction_data_path = raw_dir / "prediction_data.tar.gz"

    all_exist = (output_dir / EVENTS_FILE).is_file()
    for seed in config.SEEDS:
        if not split_path(output_dir, seed).is_file():
            all_exist = False
    if all_exist:
        print(f"Skipped existing outputs in {output_dir}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    courses = {}
    for row in read_csv(raw_dir / "course_info.csv"):
        courses[row["course_id"]] = row

    # Labels of the official train and the official test together; the split ignores which file.
    labels = {}
    for _, row in read_prediction_data(prediction_data_path, config.TRUTH_FILES):
        labels[int(row["enroll_id"])] = int(row["truth"])

    node_ids = {}
    for enrollment_id in sorted(labels):
        node_ids[enrollment_id] = len(node_ids)                          # node_id = rank of enroll_id

    write_splits(output_dir, node_ids, labels)
    if (output_dir / EVENTS_FILE).is_file():
        print(f"Kept existing {output_dir / EVENTS_FILE}", flush=True)
    else:
        write_events(prediction_data_path, output_dir, courses, labels, node_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write events.csv and the train / validation / test "
                                                 "split of every seed.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()
    preprocess(arguments.raw_dir, arguments.output_dir)
