# Raw XuetangX data and learner splits for the baselines (LR, SVM, CFIN, CA-TFHN, MST-GCN, SIG-Net)
#
#   python baseline/prepare_raw.py
#
# Writes data/processed/baseline_raw/ with the original raw columns, so every baseline reads it
# with its own code (docs/PLAN_BASELINE_v3.md):
#   - the same events as the main model M (t_obs): action in the 22 config.ACTIONS and
#     0 ≤ date(time) − date(course start) < 35 (src/2_preprocess.stream_events);
#   - every labeled enrollment of the official train + test files (225,642) in one pool;
#   - learner splits: the learners (username) are shuffled with a seed and cut 60 / 20 / 20,
#     so every learner is in exactly one of train / validation / test; one split per seed.
# Files:
#   prediction_log/log.csv            enroll_id, username, course_id, session_id, action, object,
#                                     time (original values) of every kept event
#   prediction_log/truth.csv          enroll_id, truth of every labeled enrollment
#   prediction_log/enrollments.csv    enroll_id, username, course_id of every labeled enrollment,
#                                     also those with no kept event
#   splits/learner_s{seed}.csv        enroll_id, split (train / validation / test)
#   user_info.csv, course_info.csv    copied unchanged

import argparse
import csv
import shutil
import sys
from datetime import date
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
config = import_module("0_config")                 # read only: paths, ACTIONS, OBSERVATION_DAYS
preprocess_module = import_module("2_preprocess")  # read only: tar and CSV readers

OUTPUT_DIR = ROOT / "data" / "processed" / "baseline_raw"
LOG_COLUMNS = ("enroll_id", "username", "course_id", "session_id", "action", "object", "time")
SPLITS = ("train", "validation", "test")
SEEDS = (1, 11, 111, 1111, 11111)
LEARNER_RATIOS = (0.6, 0.2, 0.2)


# Label of every enrollment in the official train and test truth files.
def read_labels(prediction_data_path):
    labels = {}
    for _, row in preprocess_module.read_prediction_data(prediction_data_path, config.TRUTH_FILES):
        labels[int(row["enroll_id"])] = int(row["truth"])
    return labels


# Copy the kept log rows to log.csv; return enrollment metadata and row counts by reason.
def write_log(prediction_data_path, log_dir, courses, labels):
    metadata = {}      # enroll_id -> (username, course_id), from its first raw row
    counts = {"kept": 0, "not_labeled": 0, "other_action": 0, "before_day_0": 0, "after_day_34": 0}
    with open(log_dir / "log.csv", "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(LOG_COLUMNS)
        for _, row in preprocess_module.read_prediction_data(prediction_data_path, config.LOG_FILES):
            enrollment_id = int(row["enroll_id"])
            if enrollment_id not in labels:
                counts["not_labeled"] += 1
                continue
            if enrollment_id not in metadata:
                metadata[enrollment_id] = (row["username"], row["course_id"])
            if row["action"] not in config.ACTIONS:
                counts["other_action"] += 1
                continue
            event_date = date.fromisoformat(row["time"][:10])
            course_start = date.fromisoformat(courses[row["course_id"]]["start"][:10])
            course_day = (event_date - course_start).days               # d = date(event) − date(course start)
            if course_day < 0:
                counts["before_day_0"] += 1
                continue
            if course_day >= config.OBSERVATION_DAYS:                    # keep 0 ≤ d < 35
                counts["after_day_34"] += 1
                continue
            writer.writerow([row[name] for name in LOG_COLUMNS])
            counts["kept"] += 1
    return metadata, counts


# truth.csv and enrollments.csv (every labeled enrollment, also those with no kept event).
def write_enrollments(log_dir, labels, metadata):
    missing = [enrollment_id for enrollment_id in labels if enrollment_id not in metadata]
    if missing:
        raise ValueError(f"{len(missing)} labeled enrollments have no raw log row, e.g. {missing[:5]}")
    enrollment_ids = sorted(labels)
    preprocess_module.write_csv(log_dir / "truth.csv", ("enroll_id", "truth"),
                                [(enrollment_id, labels[enrollment_id]) for enrollment_id in enrollment_ids])
    preprocess_module.write_csv(log_dir / "enrollments.csv", ("enroll_id", "username", "course_id"),
                                [(enrollment_id,) + metadata[enrollment_id] for enrollment_id in enrollment_ids])


# Learner split: shuffle the sorted usernames with the seed, cut 60 / 20 / 20; enroll_id -> split.
def split_by_learner(usernames_by_enrollment, seed):
    learners = np.array(sorted(set(usernames_by_enrollment.values())))
    order = np.random.default_rng(seed).permutation(len(learners))
    train_end = int(len(learners) * LEARNER_RATIOS[0])                   # n_train = ⌊0.6·n_learners⌋
    validation_end = int(len(learners) * (LEARNER_RATIOS[0] + LEARNER_RATIOS[1]))
    split_by_learner_name = {}
    for position in range(len(learners)):
        learner = learners[order[position]]
        if position < train_end:
            split_by_learner_name[learner] = "train"
        elif position < validation_end:
            split_by_learner_name[learner] = "validation"
        else:
            split_by_learner_name[learner] = "test"
    split_by_enrollment = {}
    for enrollment_id in sorted(usernames_by_enrollment):
        split_by_enrollment[enrollment_id] = split_by_learner_name[usernames_by_enrollment[enrollment_id]]
    return split_by_enrollment


# splits/learner_s{seed}.csv for every seed; print sizes, dropout rates and the learner check.
def write_splits(output_dir, seeds):
    enrollments = list(preprocess_module.read_csv(output_dir / "prediction_log" / "enrollments.csv"))
    labels = {int(row["enroll_id"]): int(row["truth"])
              for row in preprocess_module.read_csv(output_dir / "prediction_log" / "truth.csv")}
    usernames_by_enrollment = {int(row["enroll_id"]): row["username"] for row in enrollments}
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        split_by_enrollment = split_by_learner(usernames_by_enrollment, seed)
        preprocess_module.write_csv(split_dir / f"learner_s{seed}.csv", ("enroll_id", "split"),
                                    sorted(split_by_enrollment.items()))
        learners_by_split = {name: set() for name in SPLITS}
        for enrollment_id in split_by_enrollment:
            learners_by_split[split_by_enrollment[enrollment_id]].add(usernames_by_enrollment[enrollment_id])
        shared = (len(learners_by_split["train"] & learners_by_split["validation"])
                  + len(learners_by_split["train"] & learners_by_split["test"])
                  + len(learners_by_split["validation"] & learners_by_split["test"]))
        if shared:
            raise ValueError(f"seed {seed}: {shared} learners are in two splits")
        parts = []
        for name in SPLITS:
            members = [enrollment_id for enrollment_id in split_by_enrollment
                       if split_by_enrollment[enrollment_id] == name]
            rate = sum(labels[enrollment_id] for enrollment_id in members) / len(members)
            parts.append(f"{name}={len(members):,} (dropout {rate:.4f})")
        print(f"learner_s{seed}: " + ", ".join(parts) + "; no learner in two splits", flush=True)


# Run: raw files -> data/processed/baseline_raw (the log is skipped when it already exists).
def prepare_raw(raw_dir=config.RAW, output_dir=OUTPUT_DIR, seeds=SEEDS):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    log_dir = output_dir / "prediction_log"
    if (log_dir / "enrollments.csv").is_file() and (log_dir / "log.csv").is_file():
        print(f"Skipped existing {log_dir}", flush=True)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        prediction_data_path = raw_dir / "prediction_data.tar.gz"
        courses = {}
        for row in preprocess_module.read_csv(raw_dir / "course_info.csv"):
            courses[row["course_id"]] = row
        labels = read_labels(prediction_data_path)
        metadata, counts = write_log(prediction_data_path, log_dir, courses, labels)
        write_enrollments(log_dir, labels, metadata)
        shutil.copyfile(raw_dir / "user_info.csv", output_dir / "user_info.csv")
        shutil.copyfile(raw_dir / "course_info.csv", output_dir / "course_info.csv")
        print(f"Enrollments: {len(labels):,}", flush=True)
        print("Log rows: " + ", ".join(f"{name}={value:,}" for name, value in counts.items()), flush=True)
    write_splits(output_dir, seeds)
    print(f"Saved {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw XuetangX files and learner splits for the baselines.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    arguments = parser.parse_args()
    prepare_raw(arguments.raw_dir, arguments.output_dir, arguments.seeds)
