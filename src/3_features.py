# 3. Build the node feature matrix X of each split: behavior counts, user context
#    (gender, education) and course context (category). The behavior scaling is
#    fitted on train only; the context columns are one-hot, so nothing is filled in.
# Tham khảo từ project/bài báo:
# - SIG-Net, ACM SAC 2024: https://doi.org/10.1145/3605098.3636002
#   Code: https://github.com/Noverse0/SIG-Net
# - MST-GCN, Scientific Reports 2026:
#   https://doi.org/10.1038/s41598-026-40502-w
#   Code: https://github.com/wudongze9/MST-GCN
# - CA-TFHN, ICONIP 2023: https://doi.org/10.1007/978-981-99-8184-7_31
#   Code: https://github.com/codeds27/CA-TFHN


import argparse
from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
preprocess_module = import_module("2_preprocess")
read_csv = preprocess_module.read_csv
write_csv = preprocess_module.write_csv


# Columns of X used by one feature set (scenario field "features").
# Layout of X (0_config.py):  [0, 58) behavior | [58, 71) user | [71, 90) course
# Input:  "feature", "feature+user", "feature+course" or "full".
# Output: int array of column indices.
def feature_columns(name):
    behavior = np.arange(0, config.BEHAVIOR_FEATURE_COUNT)                       # 35 days + 23 actions
    user = np.arange(config.USER_FEATURE_START, config.COURSE_FEATURE_START)     # gender + education
    course = np.arange(config.COURSE_FEATURE_START, config.TOTAL_FEATURE_COUNT)  # category
    feature_sets = {
        "feature": behavior,
        "feature+user": np.concatenate([behavior, user]),
        "feature+course": np.concatenate([behavior, course]),
        "full": np.concatenate([behavior, user, course]),
    }
    return feature_sets[name]


# Name and description of every X column, in column order.
# Input:  none.
# Output: list of (name, description).
def feature_metadata():
    metadata = []

    # Daily-count columns: total valid events on each observed course day.
    for day_number in range(config.DAY_FEATURE_COUNT):
        metadata.append((
            f"activity_day_{day_number}",
            f"course_day={day_number}",
        ))

    # Action-count columns: total occurrences of each action across all days.
    for family, actions in config.ACTION_GROUPS.items():
        action_number = 1
        for action_name in actions:
            metadata.append((f"action_{family}_{action_number}", action_name))
            action_number += 1

    # User-context columns: one-hot gender, followed by missing and other.
    for gender in config.GENDERS:
        metadata.append((f"gender_{gender}", f"gender={gender}"))
    metadata.extend((
        ("gender_missing", "gender is missing"),
        ("gender_other", "gender outside vocabulary"),
    ))

    # User-context columns: one-hot education, followed by missing and other.
    for education in config.EDUCATIONS:
        clean_education = education.lower().replace("'", "").replace(" ", "_")
        metadata.append((
            f"education_{clean_education}",
            f"education={education}",
        ))
    metadata.extend((
        ("education_missing", "education is missing"),
        ("education_other", "education outside vocabulary"),
    ))

    # Course-context columns: one-hot category, followed by missing and other.
    for category in config.CATEGORIES:
        clean_category = category.replace(" ", "_")
        metadata.append((f"category_{clean_category}", f"category={category}"))
    metadata.extend((
        ("category_missing", "category is missing"),
        ("category_other", "category outside vocabulary"),
    ))
    return metadata


# Set one one-hot cell for a categorical value (known value, missing, or other).
# Input:  feature matrix, row, value ("" = missing), vocabulary, first column.
# Output: none (feature_matrix changed in place).
def set_one_hot(feature_matrix, row_index, value, vocabulary, feature_start):
    if not value:
        value_index = len(vocabulary)
    else:
        try:
            value_index = vocabulary.index(value)
        except ValueError:
            value_index = len(vocabulary) + 1
    feature_matrix[row_index, feature_start + value_index] = 1.0


# Raw features of one split, before scaling.
# Input:  path of the split CSV.
# Output: behavior counts [N, 35 + actions], user one-hot [N, user cols],
#         course one-hot [N, course cols].
def build_split_features(data_path):
    action_indices = {}
    # Column of every action, after the 35 day columns.
    for offset, action in enumerate(config.ACTIONS):
        action_indices[action] = config.ACTION_FEATURE_START + offset

    nodes = {}
    behavior_by_node = {}
    # Count events per day and per action for every node.
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
            behavior_by_node[node_id][course_day] += 1                  # c_v[d] = number of events of v on day d
            behavior_by_node[node_id][action_indices[action]] += 1      # c_v[a] = number of times v did action a

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

    # Copy behavior counts and encode user/course context for every node.
    for node_id in sorted(nodes):
        node = nodes[node_id]
        behavior_features[node_id] = behavior_by_node[node_id]

        gender = node["gender"].strip()
        education = node["education"].strip()
        category = node["category"].strip()
        if gender.lower() in config.MISSING_VALUES:
            gender = ""
        if education.lower() in config.MISSING_VALUES:
            education = ""
        if category.lower() in config.MISSING_VALUES:
            category = ""

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

    return behavior_features, user_context, course_context


# Run step 3: log1p + standardize behavior, add the one-hot context, save X per split.
# Input:  output_dir with the split CSVs.
# Output: dict split -> path of X.npy (also writes feature_names.csv).
def build_features(output_dir=config.PROCESSED):
    output_dir = Path(output_dir)
    feature_data = {}

    for split_name in config.SPLITS:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        behavior, user, course = build_split_features(
            output_dir / f"{split_name}.csv"
        )
        feature_data[split_name] = {
            "behavior": behavior,
            "user": user,
            "course": course,
        }

    train_behavior = np.log1p(
        feature_data["train"]["behavior"].astype(np.float32)
    )
    behavior_mean = np.mean(train_behavior, axis=0, dtype=np.float64)     # μ_train of log(1 + c), per column
    behavior_std = np.std(train_behavior, axis=0, dtype=np.float64)       # σ_train of log(1 + c), per column
    behavior_std[behavior_std == 0] = 1.0

    feature_paths = {}
    # Same train statistics for every split.
    for split_name in config.SPLITS:
        split_data = feature_data[split_name]
        logged_behavior = np.log1p(
            split_data["behavior"].astype(np.float32)
        )
        normalized_behavior = (
            (logged_behavior - behavior_mean) / behavior_std              # x = (log(1 + c) − μ_train) / σ_train
        ).astype(np.float32)

        node_features = np.concatenate(
            (normalized_behavior, split_data["user"], split_data["course"]),
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
    parser = argparse.ArgumentParser(description="Build train, validation, and test node features.")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    arguments = parser.parse_args()
    build_features(arguments.output_dir)
