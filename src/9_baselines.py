# 9. Baselines without a graph, on the same node features, split, threshold
#    rule, and report format as 8_train.py 

import argparse
import json
import time
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

config = import_module("0_config")
feature_columns = import_module("3_features").feature_columns
load_nodes = import_module("2_preprocess").load_nodes
train_module = import_module("8_train")
log = train_module.log
best_threshold = train_module.best_threshold
classification_metrics = train_module.classification_metrics
format_metrics = train_module.format_metrics
save_probabilities = train_module.save_probabilities

DEFAULT_SETTINGS = {"model": "gbdt", "feature_set": "full", "hsl": False, "tag": ""}


def load_split(output_dir, split_name, feature_set):
    features = np.load(Path(output_dir) / split_name / "X.npy")[:, feature_columns(feature_set)]
    labels = np.asarray([int(node["label"]) for node in load_nodes(Path(output_dir) / f"{split_name}.csv")])
    return features, labels


def make_model(name, seed):
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=1000, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=seed,
    )


def train(settings, *, seed, output_dir=config.PROCESSED):
    tag = f"_{settings['tag']}" if settings["tag"] else ""
    run_name = f"{settings['model']}_{settings['feature_set']}{tag}_seed_{seed}"
    log("setup", f"run={run_name}, settings={settings}")
    started_at = time.perf_counter()
    x, y = load_split(output_dir, "train", settings["feature_set"])
    model = make_model(settings["model"], seed).fit(x, y)
    iterations = int(getattr(model, "n_iter_", [0])[0] if settings["model"] == "logreg" else model.n_iter_)
    log("train", f"fit in {time.perf_counter() - started_at:.0f}s, iterations={iterations}")

    x_validation, y_validation = load_split(output_dir, "validation", settings["feature_set"])
    probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = best_threshold(y_validation, probabilities)
    validation = classification_metrics(y_validation, probabilities, threshold)
    log("validation", format_metrics(validation))
    Path(config.REPORTS).mkdir(parents=True, exist_ok=True)
    save_probabilities(run_name, "validation", y_validation, probabilities, threshold)

    Path(config.RUNS).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.RUNS) / f"{run_name}.joblib"
    joblib.dump({"model": model, "settings": settings, "seed": seed, "threshold": threshold,
                 "full_validation": validation}, checkpoint_path)

    report = {"checkpoint": str(checkpoint_path), "best_epoch": iterations, "select_metric": None,
              "threshold": threshold, "best_validation": validation, "selection_validation": validation,
              "settings": settings, "seed": seed, "history": []}
    report_path = Path(config.REPORTS) / f"{run_name}_train.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("train", f"t*={threshold:.4f}; report={report_path}")
    return report


def test(checkpoint_path, *, output_dir=config.PROCESSED):
    saved = joblib.load(checkpoint_path)
    run_name = Path(checkpoint_path).stem
    x_test, y_test = load_split(output_dir, "test", saved["settings"]["feature_set"])
    probabilities = saved["model"].predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, probabilities, saved["threshold"])
    log("test", format_metrics(metrics))
    save_probabilities(run_name, "test", y_test, probabilities, saved["threshold"])

    report = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": None,
              "checkpoint_validation": saved["full_validation"], "threshold": saved["threshold"],
              "threshold_source": "checkpoint", "test": metrics}
    report_path = Path(config.REPORTS) / f"{run_name}_test.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("test", f"report={report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test graph-free baselines.")
    parser.add_argument("--mode", choices=("train", "test", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--seeds", type=int, nargs="+", choices=config.SEEDS, default=[1])
    parser.add_argument("--model", choices=("gbdt", "logreg"), default=DEFAULT_SETTINGS["model"])
    parser.add_argument("--feature-set", choices=("behavior", "behavior_user", "behavior_course", "full"),
                        default=DEFAULT_SETTINGS["feature_set"])
    parser.add_argument("--tag", default="", help="suffix for the run name (scenario code)")
    arguments = parser.parse_args()

    if arguments.mode == "test":
        if arguments.checkpoint is None:
            parser.error("--checkpoint is required for --mode test")
        test(arguments.checkpoint, output_dir=arguments.output_dir)
    else:
        settings = {**DEFAULT_SETTINGS, "model": arguments.model, "feature_set": arguments.feature_set,
                    "tag": arguments.tag}
        for seed in arguments.seeds:
            report = train(settings, seed=seed, output_dir=arguments.output_dir)
            if arguments.mode == "both":
                test(report["checkpoint"], output_dir=arguments.output_dir)
