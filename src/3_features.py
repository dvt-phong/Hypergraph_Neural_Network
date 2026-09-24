# 3. Build separate train, validation, and test node features.

import argparse
import csv
import gzip
from datetime import date
from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
preprocess_module = import_module("2_preprocess")
read_csv = preprocess_module.read_csv
write_csv = preprocess_module.write_csv


# Return the columns belonging to one experimental feature set.
def feature_columns(feature_set):
    feature_sets = {
        "behavior": config.BEHAVIOR_FEATURE_SLICE,
        "behavior_user": slice(0, config.COURSE_FEATURE_START),
        "behavior_course": np.r_[
            0:config.BEHAVIOR_FEATURE_COUNT,
            config.COURSE_FEATURE_START:config.TOTAL_FEATURE_COUNT,
        ],
        "full": slice(0, config.TOTAL_FEATURE_COUNT),
    }
    return feature_sets[feature_set]


# Build readable names and source descriptions in matrix-column order.
def _feature_metadata():
    metadata = []

    for day_number in range(config.DAY_FEATURE_COUNT):
        metadata.append((
            f"activity_day_{day_number}",
            f"course_day={day_number}",
        ))

    for family, actions in config.ACTION_GROUPS.items():
        action_number = 1
        for action_name in actions:
            metadata.append((f"action_{family}_{action_number}", action_name))
            action_number += 1

    for gender in config.GENDERS:
        metadata.append((f"gender_{gender}", f"gender={gender}"))
    metadata.extend((
        ("gender_missing", "gender is missing"),
        ("gender_other", "gender outside vocabulary"),
    ))

    for education in config.EDUCATIONS:
        clean_education = education.lower().replace("'", "").replace(" ", "_")
        metadata.append((
            f"education_{clean_education}",
            f"education={education}",
        ))
    metadata.extend((
        ("education_missing", "education is missing"),
        ("education_other", "education outside vocabulary"),
        ("age_at_course_start", "course start year - birth year"),
        ("age_missing", "birth is missing or invalid"),
    ))

    for category in config.CATEGORIES:
        clean_category = category.replace(" ", "_")
        metadata.append((f"category_{clean_category}", f"category={category}"))
    metadata.extend((
        ("category_missing", "category is missing"),
        ("category_other", "category outside vocabulary"),
        ("course_duration_days", "course end - course start"),
        ("course_duration_missing", "course end is missing or invalid"),
    ))
    return metadata


# Set the known, missing, or other one-hot column for one categorical value.
def _set_one_hot(feature_matrix, row_index, value, vocabulary, feature_start):
    if not value:
        value_index = len(vocabulary)
    else:
        try:
            value_index = vocabulary.index(value)
        except ValueError:
            value_index = len(vocabulary) + 1
    feature_matrix[row_index, feature_start + value_index] = 1.0


# Fit median, mean, and standard deviation on one train numeric feature.
def _numeric_statistics(train_values):
    observed_values = train_values[np.isfinite(train_values)]
    median = float(np.median(observed_values))
    filled_values = np.where(np.isfinite(train_values), train_values, median)
    mean = float(np.mean(filled_values, dtype=np.float64))
    standard_deviation = float(np.std(filled_values, dtype=np.float64))
    if standard_deviation == 0:
        standard_deviation = 1.0
    return median, mean, standard_deviation


# Impute and standardize one numeric feature with train statistics.
def _scale_numeric(raw_values, statistics):
    median, mean, standard_deviation = statistics
    missing_mask = ~np.isfinite(raw_values)
    filled_values = np.where(missing_mask, median, raw_values)
    scaled_values = ((filled_values - mean) / standard_deviation).astype(np.float32)
    return scaled_values, missing_mask.astype(np.float32)


# Count activity by course day and action for one dataset split.
def _build_behavior_features(split_dir, node_count):
    behavior_features = np.zeros(
        (node_count, config.BEHAVIOR_FEATURE_COUNT),
        dtype=np.int32,
    )
    action_indices = {}
    for offset, action in enumerate(config.ACTIONS):
        action_indices[action] = config.ACTION_FEATURE_START + offset

    event_path = split_dir / "events_35d.csv.gz"
    with gzip.open(event_path, "rt", newline="", encoding="utf-8") as source:
        for event in csv.DictReader(source):
            node_id = int(event["node_id"])
            course_day = int(event["course_day"])
            action = event["action"]
            behavior_features[node_id, course_day] += 1
            behavior_features[node_id, action_indices[action]] += 1
    return behavior_features


# Build raw user and course context features for one dataset split.
def _build_context_features(output_dir, nodes):
    users = {}
    for row in read_csv(output_dir / "users.csv"):
        users[int(row["user_id"])] = row
    courses = {}
    for row in read_csv(output_dir / "courses.csv"):
        courses[row["course_id"]] = row

    user_context = np.zeros(
        (len(nodes), config.USER_FEATURE_COUNT),
        dtype=np.float32,
    )
    course_context = np.zeros(
        (len(nodes), config.COURSE_FEATURE_COUNT),
        dtype=np.float32,
    )
    ages = np.full(len(nodes), np.nan, dtype=np.float64)
    durations = np.full(len(nodes), np.nan, dtype=np.float64)

    for node in nodes:
        node_id = int(node["node_id"])
        user = users[int(node["user_id"])]
        course = courses[node["course_id"]]

        gender = user["gender"].strip()
        education = user["education"].strip()
        category = course["category"].strip()
        birth = user["birth"].strip()
        course_end = course["end"].strip()
        if gender.lower() in config.MISSING_VALUES:
            gender = ""
        if education.lower() in config.MISSING_VALUES:
            education = ""
        if category.lower() in config.MISSING_VALUES:
            category = ""
        if birth.lower() in config.MISSING_VALUES:
            birth = ""
        if course_end.lower() in config.MISSING_VALUES:
            course_end = ""

        _set_one_hot(
            user_context,
            node_id,
            gender,
            config.GENDERS,
            config.GENDER_FEATURE_START,
        )
        _set_one_hot(
            user_context,
            node_id,
            education,
            config.EDUCATIONS,
            config.EDUCATION_FEATURE_START,
        )
        _set_one_hot(
            course_context,
            node_id,
            category,
            config.CATEGORIES,
            config.CATEGORY_FEATURE_START,
        )

        course_start = date.fromisoformat(course["start"][:10])
        if course_end:
            course_end_date = date.fromisoformat(course_end[:10])
            duration = (course_end_date - course_start).days
            if duration >= 0:
                durations[node_id] = duration

        if birth:
            age = course_start.year - int(float(birth))
            if 10 <= age <= 100:
                ages[node_id] = age

    return user_context, course_context, ages, durations


# Build all three feature matrices with transforms fitted on train only.
def build_features(output_dir=config.PROCESSED):
    output_dir = Path(output_dir)
    feature_data = {}

    for split_name in config.SPLITS:
        split_dir = output_dir / split_name
        nodes = list(read_csv(split_dir / "nodes.csv"))
        behavior = _build_behavior_features(split_dir, len(nodes))
        user, course, ages, durations = _build_context_features(
            output_dir,
            nodes,
        )
        feature_data[split_name] = {
            "behavior": behavior,
            "user": user,
            "course": course,
            "ages": ages,
            "durations": durations,
        }

    train_behavior = np.log1p(
        feature_data["train"]["behavior"].astype(np.float32)
    )
    behavior_mean = np.mean(train_behavior, axis=0, dtype=np.float64)
    behavior_std = np.std(train_behavior, axis=0, dtype=np.float64)
    behavior_std[behavior_std == 0] = 1.0
    age_statistics = _numeric_statistics(feature_data["train"]["ages"])
    duration_statistics = _numeric_statistics(
        feature_data["train"]["durations"]
    )

    feature_paths = {}
    for split_name in config.SPLITS:
        split_data = feature_data[split_name]
        logged_behavior = np.log1p(
            split_data["behavior"].astype(np.float32)
        )
        normalized_behavior = (
            (logged_behavior - behavior_mean) / behavior_std
        ).astype(np.float32)

        user_context = split_data["user"]
        course_context = split_data["course"]
        scaled_ages, missing_ages = _scale_numeric(
            split_data["ages"],
            age_statistics,
        )
        scaled_durations, missing_durations = _scale_numeric(
            split_data["durations"],
            duration_statistics,
        )
        user_context[:, config.AGE_FEATURE_INDEX] = scaled_ages
        user_context[:, config.AGE_MISSING_FEATURE_INDEX] = missing_ages
        course_context[:, config.DURATION_FEATURE_INDEX] = scaled_durations
        course_context[:, config.DURATION_MISSING_FEATURE_INDEX] = (
            missing_durations
        )

        node_features = np.concatenate(
            (normalized_behavior, user_context, course_context),
            axis=1,
        ).astype(np.float32, copy=False)
        feature_path = output_dir / split_name / "X.npy"
        np.save(feature_path, node_features)
        feature_paths[split_name] = feature_path
        print(f"Saved {feature_path}", flush=True)

    feature_rows = []
    for feature_index, metadata in enumerate(_feature_metadata()):
        feature_name, feature_source = metadata
        feature_rows.append((feature_index, feature_name, feature_source))
    write_csv(
        output_dir / "feature_names.csv",
        ("feature_index", "feature_name", "source"),
        feature_rows,
    )
    return feature_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config.FEATURE_CLI_DESCRIPTION)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()
    build_features(arguments.output_dir)
