# 10. Shared data and protocol for the published baselines (files 11-15).
#
#   python src/10_baseline_data.py      build the cache once (about 10 minutes)
#
# Every baseline uses the same split, threshold rule, and report format as
# 8_train.py, so scripts/collect_results.py and scripts/summarize_results.py
# read them like the main model, and DeLong pairs them on the same test targets.
#
# Cache (data/processed/simple/baselines/), read from the three split CSVs of
# 2_preprocess.py, so every baseline sees exactly the same events (days 0-34):
#   {split}.npz   one row per node (node_id order): enroll/user id, course index,
#                 label, course start/end (day numbers), gender, education, birth,
#                 category; and one row per event: node, action, course day, object
#   vocab.npz     course ids, object keys "course|family|object" (as 4_hypergraph.py),
#                 object family (video, assignment, forum) and course of each object
#
# Protocol shared by all baselines (the same rules as the main model M0):
#   1. Labels: only train labels are used, for the loss and, in SIG-Net, as edge
#      types of train neighbors. A target never sees its own label.
#   2. Neighbors and aggregates come only from train enrollments, never from
#      other validation/test targets (the main model's local graphs do the same).
#   3. Enrollments of the same learner, by --user-rule (config.USER_RULES):
#        temporal  only the learner's train enrollments in courses that started no
#                  later than the target's course: nothing from after the target's
#                  35-day window (the rule of M0, scripts/check_leakage.py)
#        any       all of the learner's train enrollments, also in courses that
#                  start later (like the baselines as published; compare with M-any)
#   4. Scalers, clusterings, and per-course statistics are fitted on train only.
#   5. The checkpoint (epoch) is chosen on validation AUPRC, the threshold t* on
#      the full validation split, and test is scored once with that t*.

import argparse
import importlib.util
import json
import random
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

config = import_module("0_config")

CACHE_DIR_NAME = "baselines"
OBJECT_FAMILIES = ("video", "assignment", "forum")
BASELINE_ROOT = config.ROOT / "baseline"
CSV_COLUMNS = ("node_id", "enroll_id", "user_id", "course_id", "label", "gender", "education", "birth",
               "course_start", "course_end", "category", "action", "object_id", "course_day")
NODE_FIELDS = ("enroll_id", "user_id", "course", "label", "start", "end",
               "gender", "education", "birth", "category")
EVENT_FIELDS = ("event_node", "event_action", "event_day", "event_object")
ACTION_INDEX = {action: index for index, action in enumerate(config.ACTIONS)}


def log(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


def cache_dir(output_dir=config.PROCESSED):
    return Path(output_dir) / CACHE_DIR_NAME


# "2016-11-16 08:00:00" -> day number (the same as 4_hypergraph.start_day).
def day_numbers(values):
    days = np.asarray(values, dtype="U10").astype("datetime64[D]").astype(np.int64)
    return days + 719163  # 1970-01-01 as a proleptic ordinal, so days equal date.toordinal()


# ---------------------------------------------------------------------------
# Building the cache
# ---------------------------------------------------------------------------

def read_split(path, courses, objects, chunk_size):
    import pandas as pd

    node_parts = []
    event_parts = {name: [] for name in EVENT_FIELDS}
    dtypes = {name: str for name in CSV_COLUMNS}
    dtypes.update(node_id=np.int64, enroll_id=np.int64, user_id=np.int64, label=np.int8)
    reader = pd.read_csv(path, usecols=list(CSV_COLUMNS), dtype=dtypes, keep_default_na=False,
                         chunksize=chunk_size)
    for number, chunk in enumerate(reader, start=1):
        node_parts.append(chunk.drop_duplicates("node_id")[[c for c in CSV_COLUMNS[:11]]])
        events = chunk[chunk["action"] != ""]
        action = events["action"].map(ACTION_INDEX)
        if action.isna().any():
            raise ValueError(f"{path}: unknown actions {sorted(set(events['action'][action.isna()]))}")

        # Object key as in 4_hypergraph.read_object_events.
        family = events["action"].map(config.OBJECT_ACTIONS)
        object_id = events["object_id"].str.strip()
        has_object = family.notna() & ~object_id.str.lower().isin(config.MISSING_VALUES)
        keys = events["course_id"][has_object] + "|" + family[has_object] + "|" + object_id[has_object]
        for key in pd.unique(keys):
            if key not in objects:
                objects[key] = len(objects)
        event_object = np.full(len(events), -1, dtype=np.int32)
        event_object[has_object.to_numpy()] = keys.map(objects).to_numpy(dtype=np.int32)

        event_parts["event_node"].append(events["node_id"].to_numpy(dtype=np.int32))
        event_parts["event_action"].append(action.to_numpy(dtype=np.int8))
        event_parts["event_day"].append(events["course_day"].astype(np.int8).to_numpy())
        event_parts["event_object"].append(event_object)
        log("cache", f"{path.name}: chunk {number}, {sum(map(len, event_parts['event_node'])):,} events")

    nodes = pd.concat(node_parts).drop_duplicates("node_id").sort_values("node_id")
    if not np.array_equal(nodes["node_id"].to_numpy(), np.arange(len(nodes))):
        raise ValueError(f"{path}: node ids are not 0..n-1")
    for course_id in nodes["course_id"]:
        courses.setdefault(course_id, len(courses))
    split = {
        "enroll_id": nodes["enroll_id"].to_numpy(np.int64),
        "user_id": nodes["user_id"].to_numpy(np.int64),
        "course": nodes["course_id"].map(courses).to_numpy(np.int32),
        "label": nodes["label"].to_numpy(np.int8),
        "start": day_numbers(nodes["course_start"]),
        "end": day_numbers(nodes["course_end"]),
        "gender": nodes["gender"].to_numpy(dtype=str),
        "education": nodes["education"].to_numpy(dtype=str),
        "birth": pd.to_numeric(nodes["birth"], errors="coerce").to_numpy(np.float64),
        "category": nodes["category"].to_numpy(dtype=str),
    }
    split.update({name: np.concatenate(parts) for name, parts in event_parts.items()})
    return split


def build_cache(output_dir=config.PROCESSED, chunk_size=2_000_000):
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    target = cache_dir(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    courses, objects = {}, {}
    for split_name in config.SPLITS:
        split = read_split(output_dir / f"{split_name}.csv", courses, objects, chunk_size)
        features = np.load(output_dir / split_name / "X.npy", mmap_mode="r")
        if len(features) != len(split["label"]):
            raise ValueError(f"{split_name}.csv and {split_name}/X.npy do not align")
        np.savez(target / f"{split_name}.npz", **split)
        log("cache", f"{split_name}: {len(split['label']):,} nodes, {len(split['event_node']):,} events")

    object_keys = np.asarray(sorted(objects, key=objects.get))
    course_ids = np.asarray(sorted(courses, key=courses.get))
    course_of_key = {course_id: index for index, course_id in enumerate(course_ids)}
    parts = [key.split("|") for key in object_keys]
    np.savez(
        target / "vocab.npz",
        course_ids=course_ids,
        object_keys=object_keys,
        object_family=np.asarray([OBJECT_FAMILIES.index(p[1]) for p in parts], dtype=np.int8),
        object_course=np.asarray([course_of_key[p[0]] for p in parts], dtype=np.int32),
    )
    log("cache", f"saved {target}: {len(course_ids)} courses, {len(object_keys):,} objects, "
        f"{time.perf_counter() - started_at:.0f}s")


# ---------------------------------------------------------------------------
# Loading and the shared rules
# ---------------------------------------------------------------------------

def load_cache(output_dir=config.PROCESSED, splits=config.SPLITS):
    source = cache_dir(output_dir)
    if not (source / "vocab.npz").exists():
        raise FileNotFoundError(f"{source} is missing; run python src/10_baseline_data.py once")
    cache = {}
    for split_name in splits:
        with np.load(source / f"{split_name}.npz") as saved:
            cache[split_name] = {name: saved[name] for name in saved.files}
            cache[split_name]["name"] = split_name
    with np.load(source / "vocab.npz") as saved:
        cache.update({name: saved[name] for name in saved.files})
    return cache


# Smoke runs: a fixed random subset of nodes, events renumbered to match.
def take_nodes(split, keep):
    keep = np.sort(np.asarray(keep))
    new_id = np.full(len(split["label"]), -1, dtype=np.int64)
    new_id[keep] = np.arange(len(keep))
    kept_events = new_id[split["event_node"]] >= 0
    taken = {name: split[name][keep] for name in NODE_FIELDS}
    taken.update({name: split[name][kept_events] for name in EVENT_FIELDS})
    taken["event_node"] = new_id[taken["event_node"]].astype(np.int32)
    taken["name"] = split["name"]
    taken["original_ids"] = keep
    return taken


def limit_cache(cache, limit, seed=config.SPLIT_SEED):
    if not limit:
        return cache
    rng = np.random.default_rng(seed)
    limited = dict(cache)
    for split_name in config.SPLITS:
        count = len(cache[split_name]["label"])
        limited[split_name] = take_nodes(cache[split_name], rng.choice(count, min(limit, count), replace=False))
    return limited


# Train enrollments of the same learner that a target may use, as CSR arrays:
# indices[indptr[t]:indptr[t + 1]] for target t of `split` (never t itself).
def same_user_train(cache, split, rule):
    if rule not in config.USER_RULES:
        raise ValueError(f"rule must be one of {config.USER_RULES}")
    train = cache["train"]
    order = np.argsort(train["user_id"], kind="stable")
    users = train["user_id"][order]
    first = np.searchsorted(users, split["user_id"], side="left")
    last = np.searchsorted(users, split["user_id"], side="right")
    is_train = split["name"] == "train"
    rows = []
    for target in range(len(split["user_id"])):
        members = order[first[target]:last[target]]
        if is_train:
            members = members[members != target]
        if rule == "temporal":
            members = members[train["start"][members] <= split["start"][target]]
        rows.append(np.sort(members))
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(row) for row in rows])
    indices = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, dtype=np.int64)
    check_same_user(cache, split, rule, indptr, indices)
    return indptr, indices


# The same checks as scripts/check_leakage.py, on the lists actually used.
def check_same_user(cache, split, rule, indptr, indices):
    train = cache["train"]
    owner = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    if not np.array_equal(train["user_id"][indices], split["user_id"][owner]):
        raise AssertionError("same_user_train: a neighbor belongs to another learner")
    if split["name"] == "train" and np.any(indices == owner):
        raise AssertionError("same_user_train: a train target is its own neighbor")
    later = train["start"][indices] > split["start"][owner]
    if rule == "temporal" and later.any():
        raise AssertionError("same_user_train: a neighbor's course starts after the target's")
    log("protocol", f"{split['name']}: {len(indices):,} same-learner train neighbors (rule {rule}), "
        f"{int(later.sum()):,} from later courses")


def csr_rows(indptr, indices, row):
    return indices[indptr[row]:indptr[row + 1]]


# counts[node, day] of all events.
def daily_counts(split, days=config.OBSERVATION_DAYS):
    flat = split["event_node"].astype(np.int64) * days + split["event_day"]
    return np.bincount(flat, minlength=len(split["label"]) * days).reshape(-1, days)


# counts[node, action] over the 35 days, actions in config.ACTIONS order.
def action_counts(split):
    count = len(config.ACTIONS)
    flat = split["event_node"].astype(np.int64) * count + split["event_action"]
    return np.bincount(flat, minlength=len(split["label"]) * count).reshape(-1, count)


# counts[node, day, a] for the actions in `actions` (others are ignored).
def action_day_counts(split, actions, days=config.OBSERVATION_DAYS):
    column = np.full(len(config.ACTIONS), -1)
    for position, action in enumerate(actions):
        column[ACTION_INDEX[action]] = position
    picked = column[split["event_action"]]
    keep = picked >= 0
    flat = (split["event_node"][keep].astype(np.int64) * days + split["event_day"][keep]) * len(actions)
    flat += picked[keep]
    return np.bincount(flat, minlength=len(split["label"]) * days * len(actions)).reshape(
        -1, days, len(actions))


# Mean and std fitted on train; std 0 -> 1, as sklearn's StandardScaler.
def fit_scaler(train_values):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    return mean, np.where(std > 0, std, 1.0)


def scale(values, scaler):
    mean, std = scaler
    return ((values - mean) / std).astype(np.float32)


# Number of train enrollments of every course index.
def train_course_sizes(cache):
    return np.bincount(cache["train"]["course"], minlength=len(cache["course_ids"]))


# ---------------------------------------------------------------------------
# Runs: seeds, original code, reports
# ---------------------------------------------------------------------------

def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# Import a file of an original repository in baseline/<repo> (unchanged), with its
# folder on sys.path so its own imports resolve.
def import_original(repo, relative_path, module_name):
    path = BASELINE_ROOT / repo / relative_path
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run bash scripts/setup_baselines.sh")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run_name(settings, seed):
    tag = f"_{settings['tag']}" if settings.get("tag") else ""
    return f"{settings['model']}_{settings['user_rule']}{tag}_seed_{seed}"


# Command-line options shared by files 11-15; run_all.sh passes --mode both --seeds S.
def base_parser(description, defaults):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=("both",), default="both",
                        help="train, select on validation, then test once (the only mode)")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--seeds", type=int, nargs="+", choices=config.SEEDS, default=[1])
    parser.add_argument("--user-rule", choices=config.USER_RULES, default="temporal")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tag", default="", help="suffix for the run name (scenario code)")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke test: this many random nodes per split (0 = all)")
    for name, value in defaults.items():
        if name in ("model", "user_rule", "tag"):
            continue
        flag = "--" + name.replace("_", "-")
        parser.add_argument(flag, type=type(value), default=value)
    return parser


def settings_from(arguments, defaults):
    values = vars(arguments)
    settings = {name: values.get(name, value) for name, value in defaults.items()}
    settings.update(user_rule=arguments.user_rule, tag=arguments.tag, limit=arguments.limit)
    return settings


# Reports in the format of 8_train.py. `validation` and `test` are (labels,
# probabilities) of the selected checkpoint on the full splits, in node order.
def write_reports(run_name, settings, seed, *, history, best_epoch, checkpoint_path, validation, test):
    train_module = import_module("8_train")
    reports = Path(config.REPORTS)
    reports.mkdir(parents=True, exist_ok=True)
    threshold = train_module.best_threshold(*validation)
    full_validation = train_module.classification_metrics(*validation, threshold)
    train_module.log("validation", f"full split at t*: {train_module.format_metrics(full_validation)}")
    train_module.save_probabilities(run_name, "validation", *validation, threshold)
    selection = next((r.get("validation") for r in history if r["epoch"] == best_epoch), None)
    report = {"checkpoint": str(checkpoint_path), "best_epoch": best_epoch, "select_metric": "auprc",
              "threshold": threshold, "best_validation": full_validation,
              "selection_validation": selection, "settings": settings, "seed": seed,
              "user_rule": settings["user_rule"], "history": history}
    train_path = reports / f"{run_name}_train.json"
    train_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    train_module.log("train", f"best epoch={best_epoch}, t*={threshold:.4f}; report={train_path}")

    metrics = train_module.classification_metrics(*test, threshold)
    train_module.log("test", train_module.format_metrics(metrics))
    train_module.save_probabilities(run_name, "test", *test, threshold)
    test_report = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": best_epoch,
                   "checkpoint_validation": full_validation, "threshold": threshold,
                   "threshold_source": "checkpoint", "test": metrics, "branches": {}}
    test_path = reports / f"{run_name}_test.json"
    test_path.write_text(json.dumps(test_report, indent=2), encoding="utf-8")
    train_module.log("test", f"report={test_path}")


# Validation metrics of one epoch (t* of this epoch, as 8_train.py logs them).
def epoch_validation(labels, probabilities):
    train_module = import_module("8_train")
    metrics = train_module.classification_metrics(
        labels, probabilities, train_module.best_threshold(labels, probabilities))
    train_module.log("validation", train_module.format_metrics(metrics))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the event cache shared by the baselines.")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    arguments = parser.parse_args()
    build_cache(arguments.output_dir, arguments.chunk_size)
