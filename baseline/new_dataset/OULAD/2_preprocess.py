# 2. Split OULAD into train / validation / test files
# - OULAD https://doi.org/10.1038/sdata.2017.171
# - Wu et al. (2026) code: run_oulad_revision_pipeline.py (label, assessment set)

import argparse
import csv
import io
import random
import zipfile
from importlib import import_module
from pathlib import Path

config_constant = import_module("0_config")

ENROLLMENT_KEY = ("code_module", "code_presentation", "id_student")
CONTEXT_COLUMNS = (
    "final_result", "code_module", "code_presentation", "gender", "region", "highest_education",
    "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability",
    "date_registration", "date_unregistration",
)
ASSESSMENT_COLUMNS = (
    "node_id", "enroll_id", "id_assessment", "assessment_type", "due_date", "weight",
    "date_submitted", "is_banked", "score",
)


# Read a UTF-8 CSV file row by row, one dict per row.
# yield gives one row at a time, so a large file is never loaded into memory at once.
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        for row in reader:
            yield row


# Write a UTF-8 CSV file from column names and rows.
def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


# Read one CSV file inside oulad.zip row by row; a missing value ("?" or "") becomes "".
def read_zip_csv(zip_path, file_name):
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(file_name) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8", newline="")     # bytes -> text
            for row in csv.DictReader(text):
                clean_row = {}
                for name, value in row.items():
                    value = value.strip()
                    if value in config_constant.MISSING_VALUES:
                        value = ""
                    clean_row[name] = value
                yield clean_row


# Split enrollments into enrollment id -> "train", "validation" or "test".
# Shuffle once with seed 1: the first 80% are split 80/20 into train / validation, the last 20% are test.
def split_enrollments(enrollment_ids):
    shuffled_ids = list(enrollment_ids)
    random.Random(config_constant.SPLIT_SEED).shuffle(shuffled_ids)                     # random order, seed 1
    pool_count = int(len(shuffled_ids) * (1 - config_constant.TEST_RATIO))             # n_pool = ⌊0.8·n⌋
    train_count = int(pool_count * config_constant.TRAIN_RATIO)                         # n_train = ⌊0.8·n_pool⌋

    split_by_enrollment = {}
    for position, enrollment_id in enumerate(shuffled_ids):
        if position < train_count:
            split_by_enrollment[enrollment_id] = "train"
        elif position < pool_count:
            split_by_enrollment[enrollment_id] = "validation"
        else:
            split_by_enrollment[enrollment_id] = "test"
    return split_by_enrollment


# Write the clicks of every enrollment in days 0–34 to train.csv, validation.csv or test.csv.
def stream_clicks(zip_path, output_dir, enrollments, enrollment_ids, split_by_enrollment, node_ids, activity_by_site):
    columns = ("node_id", "enroll_id", "user_id", "course_id", "label") + CONTEXT_COLUMNS + (
        "action", "object_id", "course_day", "clicks")

    # One output file and one CSV writer per split; the files are closed at the end.
    files = {}
    writers = {}
    for split_name in config_constant.SPLITS:
        files[split_name] = open(output_dir / f"{split_name}.csv", "w", newline="", encoding="utf-8")
        writers[split_name] = csv.writer(files[split_name])
        writers[split_name].writerow(columns)

    # Every enrollment's first columns, the same on each of its rows.
    def node_columns(enrollment_id):
        enrollment = enrollments[enrollment_id]
        split_name = split_by_enrollment[enrollment_id]
        context = []
        for name in CONTEXT_COLUMNS:
            context.append(enrollment[name])
        return (node_ids[split_name][enrollment_id], enrollment_id, enrollment["id_student"],
                enrollment["course_id"], enrollment["label"]) + tuple(context)

    clicked = set()
    click_rows = 0
    for row in read_zip_csv(zip_path, "studentVle.csv"):
        course_day = int(row["date"])                                    # d = day since the module start
        if not 0 <= course_day < config_constant.OBSERVATION_DAYS:        # keep 0 ≤ d < 35
            continue
        key = (row["code_module"], row["code_presentation"], row["id_student"])
        if key not in enrollment_ids:
            raise ValueError(f"studentVle row of {key} has no studentInfo row")
        enrollment_id = enrollment_ids[key]
        clicked.add(enrollment_id)
        split_name = split_by_enrollment[enrollment_id]
        writers[split_name].writerow(node_columns(enrollment_id) + (
            activity_by_site[row["id_site"]], row["id_site"], course_day, int(row["sum_click"])))
        click_rows += 1

    # Keep enrollments with no click -> 0 in every click count
    empty_behavior_count = 0
    for enrollment_id in sorted(enrollments):
        if enrollment_id in clicked:
            continue
        split_name = split_by_enrollment[enrollment_id]
        writers[split_name].writerow(node_columns(enrollment_id) + ("", "", "", ""))
        empty_behavior_count += 1

    for split_name in config_constant.SPLITS:
        files[split_name].close()
    print(f"Wrote {click_rows:,} click rows; preserved {empty_behavior_count:,} enrollments "
          f"with no click in days 0-{config_constant.OBSERVATION_DAYS - 1}", flush=True)


# Write the assessment records used by 3_features.py to {split}_assessment.csv.
# Assessment set as Wu et al.: due date known and 0 ≤ due < 35 (an exam without a due date is left out;
# filling it with 0 gives the same X, because no exam is submitted before day 229).
# Unlike Wu et al., only submissions before day 35 count, so nothing after the window enters X.
# Banked results (date_submitted = −1) are kept as they are, as Wu et al.
def write_assessments(zip_path, output_dir, enrollment_ids, split_by_enrollment, node_ids):
    assessments = {}
    for row in read_zip_csv(zip_path, "assessments.csv"):
        assessments[row["id_assessment"]] = row

    rows = {}
    for split_name in config_constant.SPLITS:
        rows[split_name] = []
    due_count = 0
    for row in read_zip_csv(zip_path, "studentAssessment.csv"):
        assessment = assessments[row["id_assessment"]]
        if not assessment["date"]:
            continue                                                     # no due date
        due_date = int(float(assessment["date"]))
        if not 0 <= due_date < config_constant.OBSERVATION_DAYS:         # keep 0 ≤ due < 35
            continue
        due_count += 1
        date_submitted = int(row["date_submitted"])
        if date_submitted >= config_constant.OBSERVATION_DAYS:           # keep submitted < 35
            continue
        key = (assessment["code_module"], assessment["code_presentation"], row["id_student"])
        if key not in enrollment_ids:
            raise ValueError(f"studentAssessment row of {key} has no studentInfo row")
        enrollment_id = enrollment_ids[key]
        split_name = split_by_enrollment[enrollment_id]
        rows[split_name].append((
            node_ids[split_name][enrollment_id], enrollment_id, row["id_assessment"],
            assessment["assessment_type"], due_date, assessment["weight"], date_submitted,
            row["is_banked"], row["score"],
        ))

    kept_count = 0
    for split_name in config_constant.SPLITS:
        write_csv(output_dir / f"{split_name}_assessment.csv", ASSESSMENT_COLUMNS, rows[split_name])
        kept_count += len(rows[split_name])
    print(f"Assessment records due in days 0-{config_constant.OBSERVATION_DAYS - 1}: {due_count:,}; "
          f"kept {kept_count:,} submitted before day {config_constant.OBSERVATION_DAYS}", flush=True)


# Run step 2: oulad.zip -> three split CSVs + three assessment CSVs (skipped when they already exist).
def preprocess(raw_dir=config_constant.RAW, output_dir=config_constant.PROCESSED):
    zip_path = Path(raw_dir) / config_constant.ZIP_FILE
    output_dir = Path(output_dir)

    all_exist = True
    for split_name in config_constant.SPLITS:
        for file_name in (f"{split_name}.csv", f"{split_name}_assessment.csv"):
            if not (output_dir / file_name).is_file():
                all_exist = False
    if all_exist:
        print(f"Skipped existing outputs in {output_dir}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # enroll_id = row number in studentInfo.csv (0, 1, 2, ...); one enrollment per row.
    enrollments = {}
    enrollment_ids = {}
    for row in read_zip_csv(zip_path, "studentInfo.csv"):
        enrollment_id = len(enrollments)
        key = (row["code_module"], row["code_presentation"], row["id_student"])
        if key in enrollment_ids:
            raise ValueError(f"{key} has two studentInfo rows")
        row["course_id"] = f"{row['code_module']}_{row['code_presentation']}"
        row["label"] = int(row["final_result"] == config_constant.DROPOUT_RESULT)   # y = 1[Withdrawn]
        enrollments[enrollment_id] = row
        enrollment_ids[key] = enrollment_id

    for row in read_zip_csv(zip_path, "studentRegistration.csv"):
        key = (row["code_module"], row["code_presentation"], row["id_student"])
        enrollment = enrollments[enrollment_ids[key]]
        enrollment["date_registration"] = row["date_registration"]
        enrollment["date_unregistration"] = row["date_unregistration"]

    activity_by_site = {}
    for row in read_zip_csv(zip_path, "vle.csv"):
        activity_by_site[row["id_site"]] = row["activity_type"]

    split_by_enrollment = split_enrollments(sorted(enrollments))

    # node_ids[split][enrollment id] = node id 0, 1, 2, ... inside that split, in enroll_id order
    node_ids = {}
    for split_name in config_constant.SPLITS:
        node_ids[split_name] = {}
    for enrollment_id in sorted(enrollments):
        split_nodes = node_ids[split_by_enrollment[enrollment_id]]
        split_nodes[enrollment_id] = len(split_nodes)

    for split_name in config_constant.SPLITS:
        labels = []
        for enrollment_id in node_ids[split_name]:
            labels.append(enrollments[enrollment_id]["label"])
        print(f"{split_name}: {len(labels):,} enrollments, dropout rate {sum(labels) / len(labels):.4f}",
              flush=True)

    stream_clicks(zip_path, output_dir, enrollments, enrollment_ids, split_by_enrollment, node_ids,
                  activity_by_site)
    write_assessments(zip_path, output_dir, enrollment_ids, split_by_enrollment, node_ids)
    print(f"Saved preprocessed data to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split OULAD into train, validation and test CSVs.")
    parser.add_argument("--raw-dir", type=Path, default=config_constant.RAW)
    parser.add_argument("--output-dir", type=Path, default=config_constant.PROCESSED)
    arguments = parser.parse_args()
    preprocess(arguments.raw_dir, arguments.output_dir)
