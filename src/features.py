# Build node features and fit numeric transforms on train nodes only.

import argparse
import csv
import gzip
import json
import tempfile
from contextlib import ExitStack
from datetime import date
from pathlib import Path

import numpy as np

from preprocess import (
    ACTIONS,
    ACTION_GROUPS,
    OBSERVATION_DAYS,
    PROCESSED,
    read_csv,
    split_nodes,
)


GENDERS = ("female", "male")
EDUCATIONS = (
    "Associate", "Bachelor's", "Doctorate", "High", "Master's", "Middle", "Primary"
)
CATEGORIES = (
    "art", "biology", "business", "chemistry", "computer", "economics",
    "education", "electrical", "engineering", "foreign language", "history",
    "literature", "math", "medicine", "philosophy", "physics", "social science",
)

DAY_FEATURE_START = 0
DAY_FEATURE_COUNT = OBSERVATION_DAYS
ACTION_FEATURE_START = DAY_FEATURE_START + DAY_FEATURE_COUNT
ACTION_FEATURE_COUNT = len(ACTIONS)
SESSION_FEATURE_INDEX = ACTION_FEATURE_START + ACTION_FEATURE_COUNT
OBJECT_FEATURE_INDEX = SESSION_FEATURE_INDEX + 1
BEHAVIOR_FEATURE_COUNT = OBJECT_FEATURE_INDEX + 1

GENDER_FEATURE_START = 0
GENDER_FEATURE_COUNT = len(GENDERS) + 2
EDUCATION_FEATURE_START = GENDER_FEATURE_START + GENDER_FEATURE_COUNT
EDUCATION_FEATURE_COUNT = len(EDUCATIONS) + 2
AGE_FEATURE_INDEX = EDUCATION_FEATURE_START + EDUCATION_FEATURE_COUNT
AGE_MISSING_FEATURE_INDEX = AGE_FEATURE_INDEX + 1
USER_FEATURE_COUNT = AGE_MISSING_FEATURE_INDEX + 1

CATEGORY_FEATURE_START = 0
CATEGORY_FEATURE_COUNT = len(CATEGORIES) + 2
DURATION_FEATURE_INDEX = CATEGORY_FEATURE_START + CATEGORY_FEATURE_COUNT
DURATION_MISSING_FEATURE_INDEX = DURATION_FEATURE_INDEX + 1
COURSE_FEATURE_COUNT = DURATION_MISSING_FEATURE_INDEX + 1

USER_FEATURE_START = BEHAVIOR_FEATURE_COUNT
COURSE_FEATURE_START = USER_FEATURE_START + USER_FEATURE_COUNT
TOTAL_FEATURE_COUNT = COURSE_FEATURE_START + COURSE_FEATURE_COUNT

BEHAVIOR_FEATURE_SLICE = slice(0, BEHAVIOR_FEATURE_COUNT)
USER_FEATURE_SLICE = slice(USER_FEATURE_START, COURSE_FEATURE_START)
COURSE_FEATURE_SLICE = slice(COURSE_FEATURE_START, TOTAL_FEATURE_COUNT)
CLI_DESCRIPTION = "Build XuetangX node features for one experiment seed."

OBJECT_ACTIONS = {}
for family, actions in ACTION_GROUPS.items():
    if family == "web_page":
        continue
    for action in actions:
        OBJECT_ACTIONS[action] = family


# Return feature names in the same order as columns in the saved matrix.
def feature_names():
    names = []
    for day_number in range(DAY_FEATURE_COUNT):
        names.append(f"day_{day_number:02d}")
    for action in ACTIONS:
        names.append(f"action_{action}")
    names.extend(("session_count", "distinct_observed_objects"))

    for gender in GENDERS:
        names.append(f"gender_{gender}")
    names.extend(("gender_missing", "gender_other"))
    for education in EDUCATIONS:
        clean_name = education.lower().replace("'", "").replace(" ", "_")
        names.append(f"education_{clean_name}")
    names.extend((
        "education_missing", "education_other", "age_at_course_start", "age_missing"
    ))

    for category in CATEGORIES:
        names.append(f"category_{category.replace(' ', '_')}")
    names.extend((
        "category_missing", "category_other", "course_duration_days",
        "course_duration_missing",
    ))
    return names


# Return the columns belonging to one experimental feature set.
def feature_columns(feature_set):
    if feature_set == "behavior":
        return BEHAVIOR_FEATURE_SLICE
    if feature_set == "behavior_user":
        return slice(0, COURSE_FEATURE_START)
    if feature_set == "behavior_course":
        return np.r_[
            0:BEHAVIOR_FEATURE_COUNT,
            COURSE_FEATURE_START:TOTAL_FEATURE_COUNT,
        ]
    if feature_set == "full":
        return slice(0, TOTAL_FEATURE_COUNT)
    raise ValueError(f"Unknown feature set: {feature_set}")


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


# Impute and standardize a numeric feature using train nodes only.
def _scale_numeric(raw_values, train_node_ids):
    missing_mask = ~np.isfinite(raw_values)
    train_values = raw_values[train_node_ids]
    observed_train_values = train_values[~missing_mask[train_node_ids]]
    if observed_train_values.size == 0:
        raise ValueError("No observed train values for a numeric context feature")

    median = float(np.median(observed_train_values))
    filled_values = np.where(missing_mask, median, raw_values)
    mean = float(np.mean(filled_values[train_node_ids], dtype=np.float64))
    standard_deviation = float(
        np.std(filled_values[train_node_ids], dtype=np.float64)
    )
    if standard_deviation == 0:
        standard_deviation = 1.0

    scaled_values = ((filled_values - mean) / standard_deviation).astype(np.float32)
    missing_indicator = missing_mask.astype(np.float32)
    statistics = {
        "median": median,
        "mean": mean,
        "std": standard_deviation,
    }
    return scaled_values, missing_indicator, statistics


# Aggregate daily and action counts while distributing events into disk buckets.
def _write_event_buckets(output_dir, nodes, behavior_features, bucket_paths):
    node_by_enrollment = {}
    for node in nodes:
        key = (int(node["enroll_id"]), node["source_partition"])
        if key in node_by_enrollment:
            raise ValueError(f"Repeated enrollment key in nodes.csv: {key}")
        node_by_enrollment[key] = int(node["node_id"])

    action_indices = {}
    for offset, action in enumerate(ACTIONS):
        action_indices[action] = ACTION_FEATURE_START + offset

    event_count = 0
    with ExitStack() as stack:
        writers = []
        for path in bucket_paths:
            handle = stack.enter_context(open(path, "w", newline="", encoding="utf-8"))
            writers.append(csv.writer(handle))

        event_path = output_dir / "events_35d.csv.gz"
        with gzip.open(event_path, "rt", newline="", encoding="utf-8") as source:
            for event in csv.DictReader(source):
                key = (int(event["enroll_id"]), event["source_partition"])
                if key not in node_by_enrollment:
                    raise ValueError(f"Event has no matching node: {key}")

                node_id = node_by_enrollment[key]
                course_day = int(event["course_day"])
                if not 0 <= course_day < DAY_FEATURE_COUNT:
                    raise ValueError(f"Invalid course day: {course_day}")

                action = event["action"]
                if action not in action_indices:
                    raise ValueError(f"Unknown action in clean events: {action}")
                behavior_features[node_id, course_day] += 1
                behavior_features[node_id, action_indices[action]] += 1

                object_type = OBJECT_ACTIONS.get(action, "")
                writers[node_id % len(bucket_paths)].writerow((
                    node_id,
                    event["session_id"],
                    event["course_id"],
                    event["object_id"],
                    object_type,
                ))
                event_count += 1
                if event_count % 5_000_000 == 0:
                    print(f"Aggregated {event_count:,} events", flush=True)
    return event_count


# Count unique sessions and objects, then write object memberships from buckets.
def _summarize_event_buckets(bucket_paths, behavior_features, object_path):
    with gzip.open(
        object_path,
        "wt",
        newline="",
        encoding="utf-8",
        compresslevel=1,
    ) as target:
        writer = csv.writer(target)
        writer.writerow(("node_id", "course_id", "object_id", "object_type"))

        for path in bucket_paths:
            sessions_by_node = {}
            objects_by_node = {}
            memberships = set()

            with open(path, newline="", encoding="utf-8") as source:
                for row in csv.reader(source):
                    node_id = int(row[0])
                    session_id, course_id, object_id, object_type = row[1:]

                    sessions_by_node.setdefault(node_id, set()).add(session_id)
                    if object_id:
                        objects_by_node.setdefault(node_id, set()).add(
                            (course_id, object_id)
                        )
                        if object_type:
                            memberships.add(
                                (node_id, course_id, object_id, object_type)
                            )

            for node_id, session_ids in sessions_by_node.items():
                behavior_features[node_id, SESSION_FEATURE_INDEX] = len(session_ids)
            for node_id, object_ids in objects_by_node.items():
                behavior_features[node_id, OBJECT_FEATURE_INDEX] = len(object_ids)
            writer.writerows(sorted(memberships))


# Build split-independent behavior counts without retaining all events in memory.
def _build_behavior_features(output_dir, nodes, buckets, object_path):
    if buckets <= 0:
        raise ValueError("buckets must be positive")

    behavior_features = np.zeros(
        (len(nodes), BEHAVIOR_FEATURE_COUNT),
        dtype=np.int32,
    )
    with tempfile.TemporaryDirectory(
        dir=output_dir,
        prefix="feature_buckets_",
    ) as temporary:
        temporary_dir = Path(temporary)
        bucket_paths = []
        for bucket_id in range(buckets):
            bucket_paths.append(temporary_dir / f"bucket_{bucket_id:03d}.csv")

        event_count = _write_event_buckets(
            output_dir,
            nodes,
            behavior_features,
            bucket_paths,
        )
        _summarize_event_buckets(bucket_paths, behavior_features, object_path)

    daily_total = int(behavior_features[:, :ACTION_FEATURE_START].sum())
    action_stop = ACTION_FEATURE_START + ACTION_FEATURE_COUNT
    action_total = int(
        behavior_features[:, ACTION_FEATURE_START:action_stop].sum()
    )
    if daily_total != event_count or action_total != event_count:
        raise ValueError("Daily or action feature counts do not match the event count")
    return behavior_features, event_count


# Build raw user and course context features for every node.
def _build_context_features(output_dir, nodes):
    users = {}
    for row in read_csv(output_dir / "users.csv"):
        users[int(row["user_id"])] = row
    courses = {}
    for row in read_csv(output_dir / "courses.csv"):
        courses[row["course_id"]] = row

    user_context = np.zeros((len(nodes), USER_FEATURE_COUNT), dtype=np.float32)
    course_context = np.zeros((len(nodes), COURSE_FEATURE_COUNT), dtype=np.float32)
    ages = np.full(len(nodes), np.nan, dtype=np.float64)
    durations = np.full(len(nodes), np.nan, dtype=np.float64)

    for node in nodes:
        node_id = int(node["node_id"])
        user_id = int(node["user_id"])
        course_id = node["course_id"]
        if user_id not in users or course_id not in courses:
            raise ValueError(f"Missing context for node: {node_id}")

        user = users[user_id]
        course = courses[course_id]
        _set_one_hot(
            user_context,
            node_id,
            user["gender"].strip(),
            GENDERS,
            GENDER_FEATURE_START,
        )
        _set_one_hot(
            user_context,
            node_id,
            user["education"].strip(),
            EDUCATIONS,
            EDUCATION_FEATURE_START,
        )
        _set_one_hot(
            course_context,
            node_id,
            course["category"].strip(),
            CATEGORIES,
            CATEGORY_FEATURE_START,
        )

        course_start = date.fromisoformat(course["start"][:10])
        course_end = date.fromisoformat(course["end"][:10])
        duration = (course_end - course_start).days
        if duration >= 0:
            durations[node_id] = duration

        if user["birth"]:
            age = course_start.year - int(float(user["birth"]))
            if 10 <= age <= 100:
                ages[node_id] = age

    return user_context, course_context, ages, durations


# Build split-independent counts and context without rereading them per seed.
def build_feature_base(output_dir=PROCESSED, *, buckets=64):
    output_dir = Path(output_dir)
    base_path = output_dir / "feature_base.npz"
    object_path = output_dir / "node_objects.csv.gz"
    if base_path.is_file() and object_path.is_file():
        report = {"reused": True, "feature_base": str(base_path)}
        print(report, flush=True)
        return report

    nodes = list(read_csv(output_dir / "nodes.csv"))
    behavior_features, event_count = _build_behavior_features(
        output_dir,
        nodes,
        buckets,
        object_path,
    )
    user_context, course_context, ages, durations = _build_context_features(
        output_dir,
        nodes,
    )
    context = np.concatenate((user_context, course_context), axis=1)
    np.savez_compressed(
        base_path,
        raw=behavior_features,
        context=context,
        ages=ages,
        durations=durations,
    )

    report = {"nodes": len(nodes), "events": event_count, "reused": False}
    print(f"Built split-independent feature base: {report}", flush=True)
    return report


# Normalize log-transformed behavior counts using train nodes only.
def _normalize_behavior_features(raw_behavior_features, train_node_ids):
    logged_behavior = np.log1p(raw_behavior_features.astype(np.float32))
    behavior_mean = np.mean(
        logged_behavior[train_node_ids],
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    behavior_standard_deviation = np.std(
        logged_behavior[train_node_ids],
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    behavior_standard_deviation[behavior_standard_deviation == 0] = 1.0
    normalized_behavior = (
        logged_behavior - behavior_mean
    ) / behavior_standard_deviation
    return normalized_behavior, behavior_mean, behavior_standard_deviation


# Scale age and duration, then place their values and missing flags into context.
def _scale_context_features(context_features, ages, durations, train_node_ids):
    expected_context_columns = USER_FEATURE_COUNT + COURSE_FEATURE_COUNT
    if context_features.shape[1] != expected_context_columns:
        raise ValueError(f"Unexpected context feature shape: {context_features.shape}")

    user_context = context_features[:, :USER_FEATURE_COUNT].copy()
    course_context = context_features[:, USER_FEATURE_COUNT:].copy()
    scaled_ages, missing_ages, age_statistics = _scale_numeric(
        ages,
        train_node_ids,
    )
    scaled_durations, missing_durations, duration_statistics = _scale_numeric(
        durations,
        train_node_ids,
    )
    user_context[:, AGE_FEATURE_INDEX] = scaled_ages
    user_context[:, AGE_MISSING_FEATURE_INDEX] = missing_ages
    course_context[:, DURATION_FEATURE_INDEX] = scaled_durations
    course_context[:, DURATION_MISSING_FEATURE_INDEX] = missing_durations
    return user_context, course_context, age_statistics, duration_statistics


# Combine feature blocks and validate the final matrix before saving it.
def _combine_feature_blocks(normalized_behavior, user_context, course_context, node_count):
    node_features = np.concatenate(
        (normalized_behavior, user_context, course_context),
        axis=1,
    ).astype(np.float32, copy=False)
    if node_features.shape != (node_count, TOTAL_FEATURE_COUNT):
        raise ValueError(f"Unexpected feature shape: {node_features.shape}")
    if not np.isfinite(node_features).all():
        raise ValueError("Node features contain NaN or infinity")
    return node_features


# Normalize one seed's features using only its train nodes.
def build_features(output_dir=PROCESSED, *, seed=1, buckets=64, split=None):
    output_dir = Path(output_dir)
    build_feature_base(output_dir, buckets=buckets)

    nodes = list(read_csv(output_dir / "nodes.csv"))
    if split is None:
        split = split_nodes(nodes, seed)
    if len(split) != len(nodes):
        raise ValueError("Split and node counts differ")

    split_array = np.asarray(split)
    train_node_ids = np.flatnonzero(split_array == "train")
    if len(train_node_ids) == 0:
        raise ValueError("The train split is empty")

    with np.load(output_dir / "feature_base.npz") as feature_base:
        raw_behavior_features = feature_base["raw"]
        context_features = feature_base["context"]
        ages = feature_base["ages"]
        durations = feature_base["durations"]
    if context_features.shape[0] != len(nodes):
        raise ValueError(f"Unexpected context feature shape: {context_features.shape}")

    normalized_behavior, behavior_mean, behavior_standard_deviation = (
        _normalize_behavior_features(raw_behavior_features, train_node_ids)
    )
    user_context, course_context, age_statistics, duration_statistics = (
        _scale_context_features(
            context_features,
            ages,
            durations,
            train_node_ids,
        )
    )
    node_features = _combine_feature_blocks(
        normalized_behavior,
        user_context,
        course_context,
        len(nodes),
    )

    feature_path = output_dir / f"X_seed_{seed}.npy"
    np.save(feature_path, node_features)
    statistics = {
        "seed": seed,
        "names": feature_names(),
        "behavior_mean": behavior_mean.tolist(),
        "behavior_std": behavior_standard_deviation.tolist(),
        "age": age_statistics,
        "duration": duration_statistics,
        "train_nodes": len(train_node_ids),
    }
    statistics_path = output_dir / f"feature_stats_seed_{seed}.json"
    statistics_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    report = {
        "seed": seed,
        "nodes": len(nodes),
        "features": TOTAL_FEATURE_COUNT,
        "output": str(feature_path),
    }
    print(report, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buckets", type=int, default=64)
    arguments = parser.parse_args()
    build_features(
        arguments.output_dir,
        seed=arguments.seed,
        buckets=arguments.buckets,
    )
