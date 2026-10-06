# 3. Build the node feature matrix X of each split
# Tham khảo từ project/bài báo:
# - CFIN, AAAI 2019 (Feng et al.), context features (Definition 4) and age:
#   https://ojs.aaai.org/index.php/AAAI/article/view/3825
#   Code: https://github.com/wzfhaha/dropout_prediction (preprocess.py)
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
# Layout of X (0_config.py):  [0, 59) behavior | [59, 71) user | [71, 89) course
# Input:  "feature", "feature+user", "feature+course" or "full".
# Output: int array of column indices.
def feature_columns(name):
    behavior = np.arange(0, config.BEHAVIOR_FEATURE_COUNT)                       # 35 days + total + objects + 22 actions
    user = np.arange(config.USER_FEATURE_START, config.COURSE_FEATURE_START)     # gender + education + age
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

    # Summary columns of MST-GCN: total events and distinct objects over the 35 days.
    metadata.append(("total_events", "events in course days 0-34 (MST-GCN total_actions)"))
    metadata.append(("distinct_objects", "distinct objects used in course days 0-34 (MST-GCN unique_objects)"))

    # Action-count columns: total occurrences of each action across all days.
    for family, actions in config.ACTION_GROUPS.items():
        action_number = 1
        for action_name in actions:
            metadata.append((f"action_{family}_{action_number}", action_name))
            action_number += 1

    # User-context columns: one-hot gender, followed by missing.
    for gender in config.GENDERS:
        metadata.append((f"gender_{gender}", f"gender={gender}"))
    metadata.append(("gender_missing", "gender is missing"))

    # User-context columns: one-hot education, followed by missing.
    for education in config.EDUCATIONS:
        clean_education = education.lower().replace("'", "").replace(" ", "_")
        metadata.append((
            f"education_{clean_education}",
            f"education={education}",
        ))
    metadata.append(("education_missing", "education is missing"))

    # User-context column: age as in CFIN, standardized with train statistics.
    metadata.append((
        "age",
        f"course start year - birth year, {config.AGE_MISSING} if missing or outside "
        f"[{config.AGE_MIN}, {config.AGE_MAX}] (CFIN)",
    ))

    # Course-context columns: one-hot category, followed by missing.
    for category in config.CATEGORIES:
        clean_category = category.replace(" ", "_")
        metadata.append((f"category_{clean_category}", f"category={category}"))
    metadata.append(("category_missing", "category is missing"))
    return metadata


# Set one one-hot cell for a categorical value (known value or missing).
# Missing is its own level, like CFIN's code 0. A value outside the vocabulary
# stops the run instead of going to a silent "other" column (none occurs in XuetangX).
# Input:  feature matrix, row, value ("" = missing), vocabulary, first column.
# Output: none (feature_matrix changed in place).
def set_one_hot(feature_matrix, row_index, value, vocabulary, feature_start):
    if not value:
        value_index = len(vocabulary)                                    # missing column
    elif value in vocabulary:
        value_index = vocabulary.index(value)
    else:
        raise ValueError(f"{value!r} is not in the vocabulary {vocabulary}; add it to 0_config.py")
    feature_matrix[row_index, feature_start + value_index] = 1.0


# Age of a learner when the course starts, as in CFIN (preprocess.py, age_convert).
# CFIN: a = 2018 − birth; a = 0 if birth is missing or a > 70 or a < 10.
# Here the reference year is the course start year, so a is the age while taking the course.
# Input:  birth year text (e.g. "1995.0", "" = missing), course start text ("2015-09-25 08:00:00").
# Output: age a (int), AGE_MISSING when unknown or implausible.
def age_at_course_start(birth, course_start):
    if birth.lower() in config.MISSING_VALUES:
        return config.AGE_MISSING                                        # birth missing -> a = 0
    age = int(course_start[:4]) - int(float(birth))                      # a = year(course start) − birth year
    if age < config.AGE_MIN or age > config.AGE_MAX:
        return config.AGE_MISSING                                        # a outside [10, 70] -> a = 0
    return age


# Raw features of one split, before scaling.
# Input:  path of the split CSV.
# Output: behavior counts [N, 35 + 2 + actions], user context [N, user cols]
#         (one-hot gender, one-hot education, raw age), course one-hot [N, course cols].
def build_split_features(data_path):
    action_indices = {}
    # Column of every action, after the 35 day columns and the 2 summary columns.
    for offset, action in enumerate(config.ACTIONS):
        action_indices[action] = config.ACTION_FEATURE_START + offset

    nodes = {}
    behavior_by_node = {}
    objects_by_node = {}
    # Count events per day and per action, and collect the objects, for every node.
    for row in read_csv(data_path):
        node_id = int(row["node_id"])
        if node_id not in nodes:
            nodes[node_id] = row
            behavior_by_node[node_id] = np.zeros(
                config.BEHAVIOR_FEATURE_COUNT,
                dtype=np.int32,
            )
            objects_by_node[node_id] = set()

        action = row["action"].strip()
        if action:
            course_day = int(row["course_day"])
            behavior_by_node[node_id][course_day] += 1                  # c_v[d] = number of events of v on day d
            behavior_by_node[node_id][config.TOTAL_EVENTS_INDEX] += 1   # c_v = Σ_d c_v[d], all events of v
            behavior_by_node[node_id][action_indices[action]] += 1      # c_v[a] = number of times v did action a
            object_id = row["object_id"].strip()
            if object_id.lower() not in config.MISSING_VALUES:
                objects_by_node[node_id].add(object_id)                 # objects of v, any action (as MST-GCN)

    for node_id, objects in objects_by_node.items():
        behavior_by_node[node_id][config.OBJECT_COUNT_INDEX] = len(objects)   # o_v = |distinct objects of v|

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
        # Raw age a (years, 0 = unknown); standardized in build_features with train statistics.
        user_context[node_id, config.AGE_FEATURE_INDEX] = age_at_course_start(
            node["birth"].strip(),
            node["course_start"].strip(),
        )
        set_one_hot(
            course_context,
            node_id,
            category,
            config.CATEGORIES,
            config.CATEGORY_FEATURE_START,
        )

    return behavior_features, user_context, course_context


# Run step 3 and save X per split. Scaled positions in the final 89-column X are:
#   X[:, 0:59]  = 35 daily counts, total_events, distinct_objects, 22 action counts: log1p + z-score
#   X[:, 70]    = age: z-score
# X[:, 59:70] (except 70) and X[:, 71:89] stay one-hot.
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

    # SCALING REGION 1 -- final X[:, 0:59] (all behavior columns).
    # Counts are non-negative and heavily skewed (most are 0), so as Kloft et al. (2014) they are
    # log-transformed, then standardized per column with train-only statistics:
    #   x = (log(1 + c) - μ_train) / σ_train
    train_counts = np.log1p(feature_data["train"]["behavior"].astype(np.float64))   # log(1 + c)
    count_mean = np.mean(train_counts, axis=0)                           # μ_train of log(1 + c), per column
    count_std = np.std(train_counts, axis=0)                             # σ_train of log(1 + c), per column
    count_std[count_std == 0] = 1.0                                     # constant column -> z = 0

    # SCALING REGION 2 -- final X[:, 70] (user-local column 11).
    # Age uses z = (a - μ_train) / σ_train, like CFIN's StandardScaler but fitted on train
    # only. The statistics include a = 0 rows, so every unknown age maps to the same value.
    train_age = feature_data["train"]["user"][:, config.AGE_FEATURE_INDEX].astype(np.float64)
    age_mean = float(np.mean(train_age))                                  # μ_train of a
    age_std = float(np.std(train_age))                                    # σ_train of a
    print(f"Age: train mean={age_mean:.3f}, std={age_std:.3f}, "
          f"unknown={np.mean(train_age == config.AGE_MISSING):.1%}", flush=True)

    feature_paths = {}
    # Same train statistics for every split.
    for split_name in config.SPLITS:
        split_data = feature_data[split_name]
        behavior = np.log1p(split_data["behavior"].astype(np.float64))   # log(1 + c)
        scaled_behavior = (behavior - count_mean) / count_std             # final X[:, 0:59]: z-score
        user_features = split_data["user"].copy()
        user_features[:, config.AGE_FEATURE_INDEX] = (
            (user_features[:, config.AGE_FEATURE_INDEX] - age_mean) / age_std   # final X[:, 70]: z-score
        )

        node_features = np.concatenate(
            (scaled_behavior, user_features, split_data["course"]),
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
