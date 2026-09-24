# 3. Build separate train, validation, and test node features.
# Tham khảo từ project/bài báo:
# - SIG-Net, ACM SAC 2024: https://doi.org/10.1145/3605098.3636002
#   Code: https://github.com/Noverse0/SIG-Net
# - MST-GCN, Scientific Reports 2026:
#   https://doi.org/10.1038/s41598-026-40502-w
#   Code: https://github.com/wudongze9/MST-GCN
# - CA-TFHN, ICONIP 2023: https://doi.org/10.1007/978-981-99-8184-7_31
#   Code: https://github.com/codeds27/CA-TFHN
# Các nguồn trên gợi ý cách biểu diễn hành vi theo thời gian và ngữ cảnh học viên.
# Bộ 94 features trong file này là phiên bản đơn giản hóa riêng của project.

import argparse
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
def feature_metadata():
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
def set_one_hot(feature_matrix, row_index, value, vocabulary, feature_start):
    if not value:
        value_index = len(vocabulary)
    else:
        try:
            value_index = vocabulary.index(value)
        except ValueError:
            value_index = len(vocabulary) + 1
    feature_matrix[row_index, feature_start + value_index] = 1.0


# Fit median, mean, and standard deviation on one train numeric feature.
def numeric_statistics(train_values):
    observed_values = train_values[np.isfinite(train_values)]
    median = float(np.median(observed_values))
    filled_values = np.where(np.isfinite(train_values), train_values, median)
    mean = float(np.mean(filled_values, dtype=np.float64))
    standard_deviation = float(np.std(filled_values, dtype=np.float64))
    if standard_deviation == 0:
        standard_deviation = 1.0
    return median, mean, standard_deviation


# Impute and standardize one numeric feature with train statistics.
def scale_numeric(raw_values, statistics):
    median, mean, standard_deviation = statistics
    missing_mask = ~np.isfinite(raw_values)
    filled_values = np.where(missing_mask, median, raw_values)
    scaled_values = ((filled_values - mean) / standard_deviation).astype(np.float32)
    return scaled_values, missing_mask.astype(np.float32)


# Build behavior, user, and course inputs from one complete split CSV.
def build_split_features(data_path):
    action_indices = {}
    for offset, action in enumerate(config.ACTIONS):
        action_indices[action] = config.ACTION_FEATURE_START + offset

    nodes = {}
    behavior_by_node = {}
    for row in read_csv(data_path):
        node_id = int(row["node_id"])
        if node_id not in nodes:
            nodes[node_id] = row
            behavior_by_node[node_id] = np.zeros(
                config.BEHAVIOR_FEATURE_COUNT,
                dtype=np.int32,
            )

        action = row["action"].strip()
        if action:
            course_day = int(row["course_day"])
            behavior_by_node[node_id][course_day] += 1
            behavior_by_node[node_id][action_indices[action]] += 1

    node_count = len(nodes)
    behavior_features = np.zeros(
        (node_count, config.BEHAVIOR_FEATURE_COUNT),
        dtype=np.int32,
    )

    user_context = np.zeros(
        (node_count, config.USER_FEATURE_COUNT),
        dtype=np.float32,
    )
    course_context = np.zeros(
        (node_count, config.COURSE_FEATURE_COUNT),
        dtype=np.float32,
    )
    ages = np.full(node_count, np.nan, dtype=np.float64)
    durations = np.full(node_count, np.nan, dtype=np.float64)

    for node_id in sorted(nodes):
        node = nodes[node_id]
        behavior_features[node_id] = behavior_by_node[node_id]

        gender = node["gender"].strip()
        education = node["education"].strip()
        category = node["category"].strip()
        birth = node["birth"].strip()
        course_end = node["course_end"].strip()
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

        set_one_hot(
            user_context,
            node_id,
            gender,
            config.GENDERS,
            config.GENDER_FEATURE_START,
        )
        set_one_hot(
            user_context,
            node_id,
            education,
            config.EDUCATIONS,
            config.EDUCATION_FEATURE_START,
        )
        set_one_hot(
            course_context,
            node_id,
            category,
            config.CATEGORIES,
            config.CATEGORY_FEATURE_START,
        )

        course_start = date.fromisoformat(node["course_start"][:10])
        if course_end:
            course_end_date = date.fromisoformat(course_end[:10])
            duration = (course_end_date - course_start).days
            if duration >= 0:
                durations[node_id] = duration

        if birth:
            age = course_start.year - int(float(birth))
            if 10 <= age <= 100:
                ages[node_id] = age

    return behavior_features, user_context, course_context, ages, durations


# Build all three feature matrices with transforms fitted on train only.
def build_features(output_dir=config.PROCESSED):
    output_dir = Path(output_dir)
    feature_data = {}

    for split_name in config.SPLITS:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        behavior, user, course, ages, durations = build_split_features(
            output_dir / f"{split_name}.csv"
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
    age_statistics = numeric_statistics(feature_data["train"]["ages"])
    duration_statistics = numeric_statistics(
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
        scaled_ages, missing_ages = scale_numeric(
            split_data["ages"],
            age_statistics,
        )
        scaled_durations, missing_durations = scale_numeric(
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
    for feature_index, metadata in enumerate(feature_metadata()):
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
