"""Build node features and fit numeric transforms on train nodes only."""

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

OBJECT_ACTIONS = {}
for family, actions in ACTION_GROUPS.items():
    if family == "web_page":
        continue
    for action in actions:
        OBJECT_ACTIONS[action] = family


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


def feature_columns(feature_set):
    """Return the columns belonging to one experimental feature set."""
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


def _set_one_hot(matrix, row_index, value, vocabulary, feature_start):
    if not value:
        value_index = len(vocabulary)
    else:
        try:
            value_index = vocabulary.index(value)
        except ValueError:
            value_index = len(vocabulary) + 1
    matrix[row_index, feature_start + value_index] = 1.0


def _scale_numeric(values, train_ids):
    missing = ~np.isfinite(values)
    observed_train_values = values[train_ids][~missing[train_ids]]
    if observed_train_values.size == 0:
        raise ValueError("No observed train values for a numeric context feature")

    median = float(np.median(observed_train_values))
    filled = np.where(missing, median, values)
    mean = float(np.mean(filled[train_ids], dtype=np.float64))
    std = float(np.std(filled[train_ids], dtype=np.float64))
    if std == 0:
        std = 1.0

    scaled = ((filled - mean) / std).astype(np.float32)
    missing_indicator = missing.astype(np.float32)
    statistics = {"median": median, "mean": mean, "std": std}
    return scaled, missing_indicator, statistics


def _write_event_buckets(output_dir, nodes, behavior, bucket_paths):
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
                behavior[node_id, course_day] += 1
                behavior[node_id, action_indices[action]] += 1

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


def _summarize_event_buckets(bucket_paths, behavior, object_path):
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
                behavior[node_id, SESSION_FEATURE_INDEX] = len(session_ids)
            for node_id, object_ids in objects_by_node.items():
                behavior[node_id, OBJECT_FEATURE_INDEX] = len(object_ids)
            writer.writerows(sorted(memberships))


def _build_behavior_features(output_dir, nodes, buckets, object_path):
    if buckets <= 0:
        raise ValueError("buckets must be positive")

    behavior = np.zeros(
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
            behavior,
            bucket_paths,
        )
        _summarize_event_buckets(bucket_paths, behavior, object_path)

    daily_total = int(behavior[:, :ACTION_FEATURE_START].sum())
    action_stop = ACTION_FEATURE_START + ACTION_FEATURE_COUNT
    action_total = int(behavior[:, ACTION_FEATURE_START:action_stop].sum())
    if daily_total != event_count or action_total != event_count:
        raise ValueError("Daily or action feature counts do not match the event count")
    return behavior, event_count


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


def build_feature_base(output_dir=PROCESSED, *, buckets=64):
    """Build split-independent counts and context without rereading them per seed."""
    output_dir = Path(output_dir)
    base_path = output_dir / "feature_base.npz"
    object_path = output_dir / "node_objects.csv.gz"
    if base_path.is_file() and object_path.is_file():
        report = {"reused": True, "feature_base": str(base_path)}
        print(report, flush=True)
        return report

    nodes = list(read_csv(output_dir / "nodes.csv"))
    behavior, event_count = _build_behavior_features(
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
        raw=behavior,
        context=context,
        ages=ages,
        durations=durations,
    )

    report = {"nodes": len(nodes), "events": event_count, "reused": False}
    print(f"Built split-independent feature base: {report}", flush=True)
    return report


def build_features(output_dir=PROCESSED, *, seed=1, buckets=64, split=None):
    """Normalize one seed's features using only its train nodes."""
    output_dir = Path(output_dir)
    build_feature_base(output_dir, buckets=buckets)

    nodes = list(read_csv(output_dir / "nodes.csv"))
    if split is None:
        split = split_nodes(nodes, seed)
    if len(split) != len(nodes):
        raise ValueError("Split and node counts differ")

    split_array = np.asarray(split)
    train_ids = np.flatnonzero(split_array == "train")
    if len(train_ids) == 0:
        raise ValueError("The train split is empty")

    with np.load(output_dir / "feature_base.npz") as base:
        behavior = base["raw"]
        context = base["context"]
        ages = base["ages"]
        durations = base["durations"]
    if context.shape != (len(nodes), USER_FEATURE_COUNT + COURSE_FEATURE_COUNT):
        raise ValueError(f"Unexpected context feature shape: {context.shape}")
    user_context = context[:, :USER_FEATURE_COUNT].copy()
    course_context = context[:, USER_FEATURE_COUNT:].copy()

    logged_behavior = np.log1p(behavior.astype(np.float32))
    behavior_mean = np.mean(
        logged_behavior[train_ids],
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    behavior_std = np.std(
        logged_behavior[train_ids],
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    behavior_std[behavior_std == 0] = 1.0
    normalized_behavior = (logged_behavior - behavior_mean) / behavior_std

    age, age_missing, age_statistics = _scale_numeric(ages, train_ids)
    duration, duration_missing, duration_statistics = _scale_numeric(
        durations,
        train_ids,
    )
    user_context[:, AGE_FEATURE_INDEX] = age
    user_context[:, AGE_MISSING_FEATURE_INDEX] = age_missing
    course_context[:, DURATION_FEATURE_INDEX] = duration
    course_context[:, DURATION_MISSING_FEATURE_INDEX] = duration_missing

    x = np.concatenate(
        (normalized_behavior, user_context, course_context),
        axis=1,
    ).astype(np.float32, copy=False)
    if x.shape != (len(nodes), TOTAL_FEATURE_COUNT):
        raise ValueError(f"Unexpected feature shape: {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Node features contain NaN or infinity")

    np.save(output_dir / f"X_seed_{seed}.npy", x)
    statistics = {
        "seed": seed,
        "names": feature_names(),
        "behavior_mean": behavior_mean.tolist(),
        "behavior_std": behavior_std.tolist(),
        "age": age_statistics,
        "duration": duration_statistics,
        "train_nodes": len(train_ids),
    }
    statistics_path = output_dir / f"feature_stats_seed_{seed}.json"
    statistics_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    report = {
        "seed": seed,
        "nodes": len(nodes),
        "features": TOTAL_FEATURE_COUNT,
        "output": str(output_dir / f"X_seed_{seed}.npy"),
    }
    print(report, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buckets", type=int, default=64)
    arguments = parser.parse_args()
    build_features(
        arguments.output_dir,
        seed=arguments.seed,
        buckets=arguments.buckets,
    )
