# 3. Build the node feature matrix X of train / validation / test (OULAD)
# - MST-GCN https://github.com/wudongze9/MST-GCN (daily counts)
# - Wu et al. (2026) code: run_oulad_revision_pipeline.py (clicks, assessment, static fields)


import argparse
from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")
preprocess_module = import_module("2_preprocess")
read_csv = preprocess_module.read_csv
write_csv = preprocess_module.write_csv


# Column indices of X for one feature set ("feature", "feature+user", "feature+course", "full").
# [0, 60) behavior | [60, 97) user | [97, 108) course
# Uses it for experiment
def feature_columns(name):
    behavior = np.arange(0, config.BEHAVIOR_FEATURE_COUNT)                       # clicks + assessment
    user = np.arange(config.USER_FEATURE_START, config.COURSE_FEATURE_START)     # gender ... date_registration
    course = np.arange(config.COURSE_FEATURE_START, config.TOTAL_FEATURE_COUNT)  # module + presentation
    feature_sets = {
        "feature": behavior,
        "feature+user": np.concatenate([behavior, user]),
        "feature+course": np.concatenate([behavior, course]),
        "full": np.concatenate([behavior, user, course]),
    }
    return feature_sets[name]


# Set header for file
def feature_metadata():
    metadata = []

    # Daily-click columns: total sum_click on each observed course day.
    for day_number in range(config.DAY_FEATURE_COUNT):
        metadata.append((f"clicks_day_{day_number}", f"course_day={day_number}"))
    metadata.append(("total_clicks", "clicks in course days 0-34 (Wu total_clicks)"))

    # Activity-type columns: total clicks on each activity type across all days.
    for activity_type in config.ACTIVITY_TYPES:
        metadata.append((f"act_{activity_type}", f"activity_type={activity_type} (Wu act_*)"))

    # Assessment columns (Wu), over assessments due in days 0-34 and submitted before day 35.
    metadata.append(("n_assess", "assessments submitted (Wu n_assess)"))
    metadata.append(("submitted_any", "1 if n_assess > 0 (Wu submitted_any)"))
    metadata.append(("avg_score", "mean of score/100, missing score = 0 (Wu avg_score)"))
    metadata.append(("weighted_score", "sum of score/100 * weight/100 (Wu weighted_score)"))
    metadata.append(("mean_lateness", "mean of date_submitted - due date (Wu mean_lateness)"))
    metadata.append(("max_lateness", "max of date_submitted - due date (Wu max_lateness)"))

    # User-context columns: one-hot gender, age_band, region, highest_education, imd_band (+ missing)
    for gender in config.GENDERS:
        metadata.append((f"gender_{gender}", f"gender={gender}"))
    for age_band in config.AGE_BANDS:
        metadata.append((f"age_{age_band}", f"age_band={age_band}"))
    for region in config.REGIONS:
        metadata.append((f"region_{region.lower().replace(' ', '_')}", f"region={region}"))
    for education in config.EDUCATIONS:
        metadata.append((f"education_{education.lower().replace(' ', '_')}", f"highest_education={education}"))
    for imd_band in config.IMD_BANDS:
        metadata.append((f"imd_{imd_band.replace('%', '')}", f"imd_band={imd_band}"))
    metadata.append(("imd_missing", "imd_band is missing"))

    # User-context columns: numbers, missing = 0
    metadata.append(("studied_credits", "studied_credits"))
    metadata.append(("num_of_prev_attempts", "num_of_prev_attempts"))
    metadata.append(("date_registration", "date_registration, 0 if missing"))

    # Course-context columns: one-hot code_module, code_presentation
    for module in config.MODULES:
        metadata.append((f"module_{module}", f"code_module={module}"))
    for presentation in config.PRESENTATIONS:
        metadata.append((f"presentation_{presentation}", f"code_presentation={presentation}"))
    return metadata


# Set the one-hot cell of a categorical value; has_missing adds a "missing" column after the vocabulary.
def set_one_hot(feature_matrix, row_index, value, vocabulary, feature_start, has_missing=False):
    if not value and has_missing:
        value_index = len(vocabulary)                                    # missing column
    elif value in vocabulary:
        value_index = vocabulary.index(value)
    else:
        raise ValueError(f"'{value}' is not in the vocabulary {vocabulary}; add it to 0_config.py")
    feature_matrix[row_index, feature_start + value_index] = 1.0


# Number of a CSV field; missing -> 0.
def number(value):
    if not value:
        return 0.0
    return float(value)


# Assessment columns of every node, as Wu et al. (build_assessment_features)
def build_assessment_features(assessment_path, node_count):
    assessment = np.zeros((node_count, config.BEHAVIOR_FEATURE_COUNT - config.N_ASSESS_INDEX), dtype=np.float64)
    records = {}
    for row in read_csv(assessment_path):
        node_id = int(row["node_id"])
        if node_id not in records:
            records[node_id] = []
        records[node_id].append(row)

    for node_id, rows in records.items():
        assessment_ids = set()
        scores = []
        weighted_score = 0.0
        lateness = []
        for row in rows:
            assessment_ids.add(row["id_assessment"])
            score = number(row["score"]) / 100.0                              # s = score/100, missing -> 0
            scores.append(score)
            weighted_score += score * number(row["weight"]) / 100.0           # Σ s·(weight/100)
            lateness.append(int(row["date_submitted"]) - int(row["due_date"]))  # l = date_submitted − due
        assessment[node_id, 0] = len(assessment_ids)                          # n_assess
        assessment[node_id, 1] = 1.0                                          # submitted_any
        assessment[node_id, 2] = np.mean(scores)                              # avg_score
        assessment[node_id, 3] = weighted_score                               # weighted_score
        assessment[node_id, 4] = np.mean(lateness)                            # mean_lateness
        assessment[node_id, 5] = np.max(lateness)                             # max_lateness
    return assessment                                                         # no record -> all 0


# Count data for feature
def build_split_features(data_path, assessment_path):
    activity_indices = {}
    # Column of every activity type, after the 35 day columns and total_clicks.
    for offset, activity_type in enumerate(config.ACTIVITY_TYPES):
        activity_indices[activity_type] = config.ACTIVITY_FEATURE_START + offset

    nodes = {}
    clicks_by_node = {}
    # Sum clicks per day and per activity type for every node.
    for row in read_csv(data_path):
        node_id = int(row["node_id"])
        if node_id not in nodes:
            nodes[node_id] = row
            clicks_by_node[node_id] = np.zeros(config.N_ASSESS_INDEX, dtype=np.int64)

        action = row["action"].strip()
        if action:
            if action not in activity_indices:
                raise ValueError(f"activity type '{action}' is not in config.ACTIVITY_TYPES")
            course_day = int(row["course_day"])
            clicks = int(row["clicks"])
            clicks_by_node[node_id][course_day] += clicks
            clicks_by_node[node_id][config.TOTAL_CLICKS_INDEX] += clicks
            clicks_by_node[node_id][activity_indices[action]] += clicks

    node_count = len(nodes)
    if sorted(nodes) != list(range(node_count)):
        raise ValueError(f"{data_path}: node ids are not 0..n-1")
    behavior_features = np.zeros((node_count, config.BEHAVIOR_FEATURE_COUNT), dtype=np.float64)
    behavior_features[:, config.N_ASSESS_INDEX:] = build_assessment_features(assessment_path, node_count)

    user_context = np.zeros((node_count, config.USER_FEATURE_COUNT), dtype=np.float64)
    course_context = np.zeros((node_count, config.COURSE_FEATURE_COUNT), dtype=np.float64)

    # Set click counts and encode user/course context for every node.
    for node_id in range(node_count):
        node = nodes[node_id]
        behavior_features[node_id, :config.N_ASSESS_INDEX] = clicks_by_node[node_id]

        # one-hot gender, age_band, region, highest_education, imd_band (only imd_band has a missing column)
        set_one_hot(user_context, node_id, node["gender"], config.GENDERS, config.GENDER_FEATURE_START)
        set_one_hot(user_context, node_id, node["age_band"], config.AGE_BANDS, config.AGE_FEATURE_START)
        set_one_hot(user_context, node_id, node["region"], config.REGIONS, config.REGION_FEATURE_START)
        set_one_hot(user_context, node_id, node["highest_education"], config.EDUCATIONS,
                    config.EDUCATION_FEATURE_START)
        set_one_hot(user_context, node_id, node["imd_band"], config.IMD_BANDS, config.IMD_FEATURE_START,
                    has_missing=True)
        # numbers, missing -> 0
        user_context[node_id, config.CREDITS_FEATURE_INDEX] = number(node["studied_credits"])
        user_context[node_id, config.PREVIOUS_ATTEMPTS_FEATURE_INDEX] = number(node["num_of_prev_attempts"])
        user_context[node_id, config.REGISTRATION_FEATURE_INDEX] = number(node["date_registration"])
        # one-hot code_module, code_presentation
        set_one_hot(course_context, node_id, node["code_module"], config.MODULES, config.MODULE_FEATURE_START)
        set_one_hot(course_context, node_id, node["code_presentation"], config.PRESENTATIONS,
                    config.PRESENTATION_FEATURE_START)

    return np.concatenate((behavior_features, user_context, course_context), axis=1)


# Run step 3 and save into train-> X.npy / test -> X.npy / validation -> X.npy
def build_features(output_dir=config.PROCESSED):
    output_dir = Path(output_dir)
    raw_features = {}

    for split_name in config.SPLITS:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        raw_features[split_name] = build_split_features(
            output_dir / f"{split_name}.csv",
            output_dir / f"{split_name}_assessment.csv",
        )

    log_columns = np.asarray(config.LOG_Z_COLUMNS)
    z_columns = np.asarray(config.Z_COLUMNS)

    # counts are log-transformed and then standardized
    # x = (log(1 + c) - μ_train) / σ_train
    train_counts = np.log1p(raw_features["train"][:, log_columns])     # log(1 + c)
    count_mean = np.mean(train_counts, axis=0)                          # μ_train of log(1 + c), per column
    count_std = np.std(train_counts, axis=0)                            # σ_train of log(1 + c), per column
    count_std[count_std == 0] = 1.0                                     # constant column -> z = 0

    # other numbers use z = (v - μ_train) / σ_train
    train_values = raw_features["train"][:, z_columns]
    value_mean = np.mean(train_values, axis=0)                          # μ_train, per column
    value_std = np.std(train_values, axis=0)                            # σ_train, per column
    value_std[value_std == 0] = 1.0

    feature_paths = {}
    # Same train statistics for every split.
    for split_name in config.SPLITS:
        node_features = raw_features[split_name].copy()
        node_features[:, log_columns] = (np.log1p(node_features[:, log_columns]) - count_mean) / count_std
        node_features[:, z_columns] = (node_features[:, z_columns] - value_mean) / value_std
        feature_path = output_dir / split_name / "X.npy"
        np.save(feature_path, node_features.astype(np.float32))
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
