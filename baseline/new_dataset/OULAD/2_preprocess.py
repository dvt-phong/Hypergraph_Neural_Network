# 2. Write the clicks of every OULAD enrollment and the split of every seed
# - OULAD https://doi.org/10.1038/sdata.2017.171
# - Wu et al. (2026) code: run_oulad_revision_pipeline.py (label)
# The same files as src/2_preprocess.py of XuetangX, plus the column clicks (OULAD sums the clicks
# of one site on one day in one row).

import argparse
import csv
import io
import zipfile
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


# Read one CSV file inside oulad.zip row by row; a missing value ("?" or "") becomes ""
def read_zip_csv(zip_path, file_name):
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(file_name) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8", newline="")     # bytes -> text
            for row in csv.DictReader(text):
                clean_row = {}
                for name, value in row.items():
                    value = value.strip()
                    if value in config.MISSING_VALUES:
                        value = ""
                    clean_row[name] = value
                yield clean_row


# Path of the split file of one seed
def split_path(output_dir, seed):
    return Path(output_dir) / "splits" / f"s{seed}.csv"


# Split of one seed: every enrollment -> "train" (60%), "validation" (10%) or "test" (30%),
# stratified by label (the same dropout rate in the three parts), as src/2_preprocess.py
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


# Read studentInfo.csv: enroll_id = row number (0, 1, 2, ...), one enrollment per row
def read_enrollments(zip_path):
    enrollments = {}       # enroll_id -> (user_id, course_id, label)
    enrollment_ids = {}    # (code_module, code_presentation, id_student) -> enroll_id
    for row in read_zip_csv(zip_path, "studentInfo.csv"):
        enrollment_id = len(enrollments)
        key = (row["code_module"], row["code_presentation"], row["id_student"])
        if key in enrollment_ids:
            raise ValueError(f"{key} has two studentInfo rows")
        course_id = f"{row['code_module']}_{row['code_presentation']}"
        label = int(row["final_result"] == config.DROPOUT_RESULT)          # y = 1[Withdrawn]
        enrollments[enrollment_id] = (int(row["id_student"]), course_id, label)
        enrollment_ids[key] = enrollment_id
    return enrollments, enrollment_ids


# Write the clicks of days 0–34 of every enrollment to events.csv
def write_events(zip_path, output_dir, enrollments, enrollment_ids):
    columns = ("node_id", "enroll_id", "user_id", "course_id", "label", "action", "object_id",
               "course_day", "clicks")
    activity_by_site = {}
    for row in read_zip_csv(zip_path, "vle.csv"):
        activity_by_site[row["id_site"]] = row["activity_type"]

    has_click = set()      # enroll_ids with at least one kept click
    row_count = 0
    click_count = 0
    with open(output_dir / EVENTS_FILE, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)

        for row in read_zip_csv(zip_path, "studentVle.csv"):
            course_day = int(row["date"])                                    # d = day since the module start
            if course_day < 0 or course_day >= config.OBSERVATION_DAYS:      # keep 0 ≤ d < 35
                continue
            key = (row["code_module"], row["code_presentation"], row["id_student"])
            if key not in enrollment_ids:
                raise ValueError(f"studentVle row of {key} has no studentInfo row")
            enrollment_id = enrollment_ids[key]
            user_id, course_id, label = enrollments[enrollment_id]
            clicks = int(row["sum_click"])
            writer.writerow((
                enrollment_id,                                               # node_id = enroll_id
                enrollment_id,
                user_id,
                course_id,
                label,
                activity_by_site[row["id_site"]],                            # action = activity type of the site
                row["id_site"],
                course_day,
                clicks,
            ))
            has_click.add(enrollment_id)
            row_count += 1
            click_count += clicks

        # Enrollments with no click: one row with an empty action (0 in every count).
        empty_count = 0
        for enrollment_id in sorted(enrollments):
            if enrollment_id in has_click:
                continue
            user_id, course_id, label = enrollments[enrollment_id]
            writer.writerow((enrollment_id, enrollment_id, user_id, course_id, label, "", "", "", ""))
            empty_count += 1

    print(f"Wrote {row_count:,} rows ({click_count:,} clicks) of {len(enrollments):,} enrollments "
          f"({empty_count:,} with no click) to {output_dir / EVENTS_FILE}", flush=True)


# Write splits/s{seed}.csv for every seed
def write_splits(output_dir, labels):
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    for seed in config.SEEDS:
        split_by_enrollment = split_enrollments(labels, seed)
        rows = []
        for enrollment_id in sorted(split_by_enrollment):
            rows.append((enrollment_id, enrollment_id, split_by_enrollment[enrollment_id]))
        write_csv(split_path(output_dir, seed), ("node_id", "enroll_id", "split"), rows)

        counts = {}
        for split_name in config.SPLITS:
            counts[split_name] = 0
        for enrollment_id in split_by_enrollment:
            counts[split_by_enrollment[enrollment_id]] += 1
        print(f"Saved {split_path(output_dir, seed)}: {counts}", flush=True)


# Run step 2: oulad.zip -> events.csv and the split files (skipped when they already exist; an
# existing events.csv is kept, it does not depend on the split)
def preprocess(raw_dir=config.RAW, output_dir=config.PROCESSED):
    zip_path = Path(raw_dir) / config.ZIP_FILE
    output_dir = Path(output_dir)

    all_exist = (output_dir / EVENTS_FILE).is_file()
    for seed in config.SEEDS:
        if not split_path(output_dir, seed).is_file():
            all_exist = False
    if all_exist:
        print(f"Skipped existing outputs in {output_dir}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    enrollments, enrollment_ids = read_enrollments(zip_path)
    labels = {}
    for enrollment_id in enrollments:
        labels[enrollment_id] = enrollments[enrollment_id][2]

    write_splits(output_dir, labels)
    if (output_dir / EVENTS_FILE).is_file():
        print(f"Kept existing {output_dir / EVENTS_FILE}", flush=True)
    else:
        write_events(zip_path, output_dir, enrollments, enrollment_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write events.csv and the train / validation / test "
                                                 "split of every seed.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()
    preprocess(arguments.raw_dir, arguments.output_dir)
