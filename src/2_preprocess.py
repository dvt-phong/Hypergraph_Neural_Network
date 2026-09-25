# 2. Preprocess XuetangX and preserve its official test partition.
# Tham khảo từ project/bài báo:
# - SIG-Net, ACM SAC 2024: https://doi.org/10.1145/3605098.3636002
#   Code: https://github.com/Noverse0/SIG-Net
# - MST-GCN, Scientific Reports 2026:
#   https://doi.org/10.1038/s41598-026-40502-w
#   Code: https://github.com/wudongze9/MST-GCN
# - CA-TFHN, ICONIP 2023: https://doi.org/10.1007/978-981-99-8184-7_31
#   Code: https://github.com/codeds27/CA-TFHN
# Các nguồn trên được dùng để đối chiếu cách tổ chức dữ liệu tương tác MOOC.
# Cách chia split và CSV hợp nhất trong file này là thiết kế của project.

import argparse
import codecs
import csv
import random
import tarfile
from datetime import date
from importlib import import_module
from pathlib import Path

config_constant = import_module("0_config")


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


# Return one metadata row per node, ordered by local node identifier.
def load_nodes(path):
    nodes = {}
    for row in read_csv(path):
        node_id = int(row["node_id"])
        if node_id not in nodes:
            nodes[node_id] = row
    return [nodes[node_id] for node_id in sorted(nodes)]


# Read selected CSV members in one sequential pass through the archive.
def read_prediction_data(prediction_data_path, selected_names):
    with tarfile.open(prediction_data_path, "r|gz") as prediction_data:
        for member in prediction_data:
            name = Path(member.name).name
            if not member.isfile() or name not in selected_names:
                continue

            binary = prediction_data.extractfile(member)
            with binary:
                text = codecs.getreader("utf-8")(binary)
                for row in csv.DictReader(text):
                    yield name, row


# Split enrollments in train data into train set and validation set.
def split_train_enrollments(enrollment_ids):
    shuffled_ids = list(enrollment_ids)
    random.Random(config_constant.SPLIT_SEED).shuffle(shuffled_ids)
    train_count = int(len(shuffled_ids) * config_constant.TRAIN_RATIO)

    split_by_enrollment = {}
    for position, enrollment_id in enumerate(shuffled_ids):
        if position < train_count:
            split_by_enrollment[enrollment_id] = "train"
        else:
            split_by_enrollment[enrollment_id] = "validation"
    return split_by_enrollment


# Stream valid events from both raw logs into three split CSV files.
def stream_events(
    prediction_data_path,
    output_dir,
    users,
    courses,
    labels,
    split_by_enrollment,
):
    node_ids = {split_name: {} for split_name in config_constant.SPLITS}
    columns = (
        "node_id", "enroll_id", "user_id", "course_id", "label",
        "gender", "education", "birth", "course_start", "course_end",
        "category", "action", "object_id", "course_day",
    )

    with (
        open(output_dir / "train.csv", "w", newline="", encoding="utf-8")
        as train_target,
        open(output_dir / "validation.csv", "w", newline="", encoding="utf-8")
        as validation_target,
        open(output_dir / "test.csv", "w", newline="", encoding="utf-8")
        as test_target,
    ):
        writers = {
            "train": csv.writer(train_target),
            "validation": csv.writer(validation_target),
            "test": csv.writer(test_target),
        }
        for writer in writers.values():
            writer.writerow(columns)

        for _, row in read_prediction_data(prediction_data_path, config_constant.LOG_FILES):
            enrollment_id = int(row["enroll_id"])
            user_id = int(row["username"])
            course_id = row["course_id"]

            action = row["action"]
            if action not in config_constant.ACTIONS:
                continue

            event_date = date.fromisoformat(row["time"][:10])
            course_start = date.fromisoformat(courses[course_id]["start"][:10])
            course_day = (event_date - course_start).days
            if 0 <= course_day < config_constant.OBSERVATION_DAYS:
                split_name = split_by_enrollment[enrollment_id]
                split_node_ids = node_ids[split_name]
                if enrollment_id not in split_node_ids:
                    split_node_ids[enrollment_id] = len(split_node_ids)

                user = users[user_id]
                course = courses[course_id]
                writers[split_name].writerow((
                    split_node_ids[enrollment_id],
                    enrollment_id,
                    user_id,
                    course_id,
                    labels[enrollment_id],
                    user["gender"],
                    user["education"],
                    user["birth"],
                    course["start"],
                    course["end"],
                    course["category"],
                    action,
                    row["object"].strip(),
                    course_day,
                ))


# Run preprocessing and write explicit train, validation, and test datasets.
def preprocess(raw_dir=config_constant.RAW, output_dir=config_constant.PROCESSED):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    prediction_data_path = raw_dir / "prediction_data.tar.gz"
    user_path = raw_dir / "user_info.csv"
    course_path = raw_dir / "course_info.csv"

    output_paths = [output_dir / f"{split_name}.csv" for split_name in config_constant.SPLITS]
    if all(path.is_file() for path in output_paths):
        print(f"Skipped existing outputs in {output_dir}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    users = {}
    for row in read_csv(user_path):
        users[int(row["user_id"])] = row

    courses = {}
    for row in read_csv(course_path):
        courses[row["course_id"]] = row

    train_labels = {}
    test_labels = {}
    for truth_name, row in read_prediction_data(prediction_data_path, config_constant.TRUTH_FILES):
        enrollment_id = int(row["enroll_id"])
        label = int(row["truth"])
        if truth_name == "train_truth.csv":
            train_labels[enrollment_id] = label
        else:
            test_labels[enrollment_id] = label

    split_by_enrollment = split_train_enrollments(sorted(train_labels))
    for enrollment_id in test_labels:
        split_by_enrollment[enrollment_id] = "test"

    labels = {}
    labels.update(train_labels)
    labels.update(test_labels)
    stream_events(
        prediction_data_path,
        output_dir,
        users,
        courses,
        labels,
        split_by_enrollment,
    )
    print(f"Saved preprocessed data to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config_constant.PREPROCESS_CLI_DESCRIPTION)
    parser.add_argument("--raw-dir", type=Path, default=config_constant.RAW)
    parser.add_argument("--output-dir", type=Path, default=config_constant.PROCESSED)
    arguments = parser.parse_args()
    preprocess(arguments.raw_dir, arguments.output_dir)
