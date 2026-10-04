# 6. Train the model on the train hypergraph, pick the checkpoint and the decision
#    threshold on validation, then evaluate once on test.
# Tham khảo từ project/bài báo:
# - HGNN, AAAI 2019 (Feng et al.): hidden size, dropout, optimizer settings
#   Code: https://github.com/iMoonLab/HGNN
# - Lipton et al., ECML PKDD 2014: "Thresholding Classifiers to Maximize F1 Score"

import argparse
import json
import random
import time
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score,
)
from torch.nn import functional as F

config = import_module("0_config")
hypergraph_module = import_module("4_hypergraph")
model_module = import_module("5_model")

resolve_device = hypergraph_module.resolve_device
load_train_graph = hypergraph_module.load_train_graph
load_evaluation_split = hypergraph_module.load_evaluation_split
build_local_graph = hypergraph_module.build_local_graph
merge_local_graphs = hypergraph_module.merge_local_graphs
DropoutModel = model_module.DropoutModel
prepare_graph = model_module.prepare_graph

# All settings of a run; the command line can override each of them.
DEFAULT_SETTINGS = {
    # Graph
    "families": list(config.GRAPH_FAMILIES),  # hyperedge families; ["self_loop"] = MLP
    "hypergraph": "hypergraph.npz",  # bundle from 4_hypergraph.py inside --output-dir
    "causal": False,            # receive only from hyperedges with no later-starting member
    "shuffle_graph": None,      # control: seed for random hyperedge members (None = real graph)
    # Model
    "hidden_dim": 128,
    "dropout": 0.5,             # as in HGNN (Feng et al., 2019)
    "skip_connection": False,   # add the MLP(X) branch
    "family_weights": False,    # one learned weight per hyperedge family (HGNN's W)
    # Training
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "family_weight_lr": 0.05,   # own lr, no weight decay (the main lr is too small)
    "epochs": 600,
    "eval_every": 5,            # validate every N epochs
    "patience": 20,             # stop after N validations without improvement
    "validation_limit": 5000,   # fixed random validation subset during training (0 = all)
    "eval_batch_size": 8,       # local graphs per forward pass
    "tag": "",                  # appended to the run name
}


# Print a message with the time and a scope, e.g. [train].
def log(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


# Seed Python, NumPy and PyTorch.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Threshold t* with the highest F1 of the dropout class.
# Input:  labels [N], probabilities [N].
# Output: float threshold.
def best_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])  # the last point has no threshold


# Evaluation metrics; positive class = dropout, "negative" = non-dropout class.
# Input:  labels [N], probabilities [N], threshold.
# Output: dict auc, auprc, f1, precision, recall, ... at the threshold.
def classification_metrics(labels, probabilities, threshold=0.5):
    def scores_at(cutoff):
        return precision_recall_fscore_support(
            labels, probabilities >= cutoff, labels=[0, 1], zero_division=0
        )

    precision, recall, f1, _ = scores_at(threshold)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(np.mean((probabilities >= threshold) == labels)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "auprc_negative": float(average_precision_score(1 - labels, 1 - probabilities)),
        "threshold": float(threshold),
        "f1": float(f1[1]),
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "f1_negative": float(f1[0]),
        "macro_f1": float(f1.mean()),
        "f1_at_0.5": float(scores_at(0.5)[2][1]),
    }


# Metrics dict -> "name=value, ..." for the log.
def format_metrics(metrics):
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())


# Run name built from the settings (same format as before the clean-up).
# Input:  settings, seed.
# Output: str, e.g. "hgnn_course-object-user_temporal_causal_skip_fw_full_i-cou_seed_1".
def make_run_name(settings, seed):
    families = [family for family in settings["families"] if family != "self_loop"]
    if not families:
        name = "mlp"
    elif families == list(config.GRAPH_FAMILIES):
        name = "hgnn"
    else:
        name = "hgnn_" + "-".join(families)
    bundle = Path(settings["hypergraph"]).stem.removeprefix("hypergraph").strip("_")
    if bundle:  # e.g. hypergraph_temporal.npz -> "_temporal"
        name += f"_{bundle}"
    if settings["causal"]:
        name += "_causal"
    if settings["shuffle_graph"] is not None:
        name += f"_shuffled{settings['shuffle_graph']}"
    if settings["skip_connection"]:
        name += "_skip"
    if settings["family_weights"]:
        name += "_fw"
    name += "_full"  # feature set: always all columns of X
    if settings["tag"]:
        name += f"_{settings['tag']}"
    return f"{name}_seed_{seed}"


# Input:  number of input features, settings.
# Output: DropoutModel built from the settings.
def make_model(input_dim, settings):
    return DropoutModel(input_dim, settings["hidden_dim"], settings["dropout"],
                        skip_connection=settings["skip_connection"],
                        family_weights=settings["family_weights"])


# Adam; the family weights get their own lr and no weight decay.
# Input:  model, settings.
# Output: torch optimizer.
def make_optimizer(model, settings):
    groups = [{"params": [p for name, p in model.named_parameters() if name != "family_logits"]}]
    if model.family_logits is not None:
        groups.append({"params": [model.family_logits],
                       "lr": settings["family_weight_lr"], "weight_decay": 0.0})
    return torch.optim.Adam(groups, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])


# load_evaluation_split with the options in settings.
def load_split(output_dir, split_name, settings):
    return load_evaluation_split(
        output_dir, split_name=split_name, families=settings["families"],
        hypergraph_file=settings["hypergraph"], shuffle_seed=settings["shuffle_graph"],
    )


# Dropout probabilities of validation/test targets, each on its own local graph.
# Input:  model, split data, settings, device, limit (0 = all targets, else a
#         fixed random subset), seed of that subset.
# Output: labels [n], probabilities [n], parts {logit_graph, logit_self}
#         (empty without the skip branch).
@torch.no_grad()
def predict(model, split_data, settings, device, *, limit=0, seed=0):
    target_ids = np.arange(len(split_data["nodes"]))
    if limit:
        chosen = np.random.default_rng(seed).choice(target_ids, min(limit, len(target_ids)), replace=False)
        target_ids = np.sort(chosen)

    model.eval()
    batch_size = settings["eval_batch_size"]
    all_labels, all_probabilities, all_parts = [], [], defaultdict(list)
    started_at = last_log = time.perf_counter()
    for start in range(0, len(target_ids), batch_size):
        local_graphs = [build_local_graph(split_data, int(t)) for t in target_ids[start:start + batch_size]]
        features, graph, target_rows, labels = merge_local_graphs(local_graphs)
        output = model(torch.as_tensor(features, device=device),
                       prepare_graph(graph, device, settings["causal"]))

        rows = torch.as_tensor(target_rows, device=device)
        all_probabilities.extend(torch.sigmoid(output["logits"][rows]).cpu().tolist())
        all_labels.extend(labels.tolist())
        for name in ("logit_graph", "logit_self"):
            if name in output:
                all_parts[name].extend(output[name][rows].cpu().tolist())

        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            log(split_data["split_name"], f"{start + len(local_graphs):,}/{len(target_ids):,} targets")

    log(split_data["split_name"], f"{len(target_ids):,} targets in {time.perf_counter() - started_at:.0f}s")
    parts = {name: np.asarray(values) for name, values in all_parts.items()}
    return np.asarray(all_labels), np.asarray(all_probabilities), parts


# AUC of the graph branch and of the MLP branch alone (skip models).
# Input:  labels [N], parts from predict (empty = no skip branch).
# Output: dict auc_self_branch, auc_graph_branch, std and correlation of the logits.
def branch_metrics(labels, parts):
    if not parts:
        return {}
    graph, own = parts["logit_graph"], parts["logit_self"]
    return {
        "auc_self_branch": float(roc_auc_score(labels, own)),
        "auc_graph_branch": float(roc_auc_score(labels, graph)),
        "std_logit_self": float(np.std(own)),
        "std_logit_graph": float(np.std(graph)),
        "corr_branches": float(np.corrcoef(own, graph)[0, 1]),
    }


# Save labels, probabilities and threshold for plots and DeLong tests.
# Input:  run name, split name, labels, probabilities, threshold, parts.
# Output: none (writes outputs/reports/<run>_<split>_probs.npz).
def save_probabilities(run_name, split_name, labels, probabilities, threshold, parts):
    path = Path(config.REPORTS) / f"{run_name}_{split_name}_probs.npz"
    np.savez_compressed(path, labels=labels, probabilities=probabilities, threshold=threshold, **parts)
    log(split_name, f"probabilities={path}")


# Write a report dict as JSON to outputs/reports/<run>_<kind>.json.
def save_report(report, run_name, kind):
    path = Path(config.REPORTS) / f"{run_name}_{kind}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(kind, f"report={path}")


# Train one seed: full-batch epochs on H0, validation every eval_every epochs,
# keep the best checkpoint (validation AUPRC), early stopping, then t* on the
# full validation split.
# Input:  settings, seed, output_dir, device name.
# Output: path of the checkpoint (report saved as <run>_train.json).
def train(settings, *, seed, output_dir=config.PROCESSED, device_name="auto"):
    set_seed(seed)
    device = resolve_device(device_name)
    run_name = make_run_name(settings, seed)
    log("setup", f"run={run_name}, device={device}, settings={settings}")
    Path(config.RUNS).mkdir(parents=True, exist_ok=True)
    Path(config.REPORTS).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.RUNS) / f"{run_name}.pt"

    train_data = load_train_graph(
        output_dir, families=settings["families"], hypergraph_file=settings["hypergraph"],
        shuffle_seed=settings["shuffle_graph"],
    )
    validation_data = load_split(output_dir, "validation", settings)
    x = torch.as_tensor(train_data["features"], device=device)
    labels = torch.as_tensor(train_data["labels"], device=device)
    graph = prepare_graph(train_data["graph"], device, settings["causal"])
    log("setup", f"X={tuple(x.shape)}, hyperedges={len(graph['edge_family']):,}, "
        f"memberships={len(train_data['graph']['node_ids']):,}")

    model = make_model(x.shape[1], settings).to(device)
    optimizer = make_optimizer(model, settings)
    history = []
    best = {"auprc": -1.0, "epoch": 0, "validation": None}
    stale_validations = 0

    for epoch in range(1, settings["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        logits = model(x, graph)["logits"]
        weights = model.weight_summary()  # before the step, as logged so far
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_auc = roc_auc_score(train_data["labels"], logits.detach().cpu().numpy())
        record = {"epoch": epoch, "loss": loss.item(), "train_auc": float(train_auc), **weights,
                  "seconds": time.perf_counter() - epoch_started}
        log("train", f"epoch {epoch}: " + ", ".join(
            f"{name}={value:.4f}" for name, value in record.items() if name != "epoch"))

        if epoch % settings["eval_every"] == 0 or epoch == settings["epochs"]:
            # Same validation subset for every seed and run; metrics at its best threshold.
            validation_labels, validation_probabilities, _ = predict(
                model, validation_data, settings, device,
                limit=settings["validation_limit"], seed=config.SPLIT_SEED,
            )
            validation = classification_metrics(
                validation_labels, validation_probabilities,
                best_threshold(validation_labels, validation_probabilities),
            )
            record["validation"] = validation
            log("validation", format_metrics(validation))
            if validation["auprc"] > best["auprc"]:
                best = {"auprc": validation["auprc"], "epoch": epoch, "validation": validation}
                stale_validations = 0
                torch.save({"state_dict": model.state_dict(), "input_dim": x.shape[1], "settings": settings,
                            "seed": seed, "epoch": epoch, "validation": validation}, checkpoint_path)
                log("checkpoint", f"new best val_auprc={validation['auprc']:.4f} at epoch {epoch}")
            else:
                stale_validations += 1
        history.append(record)
        if stale_validations >= settings["patience"]:
            log("train", f"early stopping after {stale_validations} validations without improvement")
            break

    # Decision threshold t* of the best checkpoint, on the full validation split.
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    validation_labels, validation_probabilities, parts = predict(model, validation_data, settings, device)
    threshold = best_threshold(validation_labels, validation_probabilities)
    full_validation = classification_metrics(validation_labels, validation_probabilities, threshold)
    log("validation", f"full split at t*: {format_metrics(full_validation)}")
    save_probabilities(run_name, "validation", validation_labels, validation_probabilities, threshold, parts)
    torch.save({**saved, "threshold": threshold, "full_validation": full_validation}, checkpoint_path)

    save_report({"checkpoint": str(checkpoint_path), "best_epoch": best["epoch"], "select_metric": "auprc",
                 "threshold": threshold, "best_validation": full_validation,
                 "selection_validation": best["validation"], "settings": settings, "seed": seed,
                 "user_rule": validation_data["user_rule"], "history": history}, run_name, "train")
    log("train", f"best epoch={best['epoch']}, val_auprc={best['auprc']:.4f}, t*={threshold:.4f}")
    return checkpoint_path


# Evaluate a selected checkpoint once on test, at the validation threshold t*.
# Input:  checkpoint path, output_dir, device name, limit (0 = all test targets).
# Output: test metrics (report saved as <run>_test.json).
def test(checkpoint_path, *, output_dir=config.PROCESSED, device_name="auto", limit=0):
    device = resolve_device(device_name)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if saved["settings"].get("hsl") or saved["settings"].get("edge_weights"):
        raise ValueError("This checkpoint uses HSL or edge weights; test it with git tag full-hsl")
    settings = {name: saved["settings"].get(name, default) for name, default in DEFAULT_SETTINGS.items()}
    model = make_model(saved["input_dim"], settings).to(device)
    model.load_state_dict(saved["state_dict"])
    run_name = Path(checkpoint_path).stem
    threshold = saved["threshold"]
    log("test", f"threshold t*={threshold:.4f}")

    test_data = load_split(output_dir, "test", settings)
    labels, probabilities, parts = predict(model, test_data, settings, device,
                                           limit=limit, seed=saved["seed"])
    metrics = classification_metrics(labels, probabilities, threshold)
    branches = branch_metrics(labels, parts)
    log("test", format_metrics(metrics))
    if branches:
        log("test", "branches: " + format_metrics(branches))
    Path(config.REPORTS).mkdir(parents=True, exist_ok=True)
    save_probabilities(run_name, "test", labels, probabilities, threshold, parts)
    save_report({"checkpoint": str(checkpoint_path), "checkpoint_epoch": saved["epoch"],
                 "checkpoint_validation": saved["full_validation"], "threshold": threshold,
                 "threshold_source": "checkpoint", "test": metrics, "branches": branches},
                run_name, "test")
    return metrics


# --families: "course,object" -> ["course", "object"]; "self_loop" alone = MLP.
def families_argument(value):
    families = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in families if name not in config.EDGE_FAMILIES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown families {unknown}; choose from {config.EDGE_FAMILIES}")
    return families


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config.TRAIN_CLI_DESCRIPTION)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="train")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint for --mode test")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", choices=config.SEEDS, default=[1])
    parser.add_argument("--test-limit", type=int, default=0, help="score only N test targets (0 = all)")
    # Settings (default: DEFAULT_SETTINGS)
    parser.add_argument("--families", type=families_argument, help="e.g. course,object,user; self_loop = MLP")
    parser.add_argument("--hypergraph", help="bundle inside --output-dir, e.g. hypergraph_temporal.npz")
    parser.add_argument("--causal", action="store_true", default=None)
    parser.add_argument("--shuffle-graph", type=int, metavar="SEED")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--skip-connection", action="store_true", default=None)
    parser.add_argument("--family-weights", action="store_true", default=None)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--family-weight-lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--tag", help="suffix for the run name, e.g. i-cou")
    arguments = vars(parser.parse_args())

    if arguments["mode"] == "test":
        if arguments["checkpoint"] is None:
            parser.error("--checkpoint is required for --mode test")
        test(arguments["checkpoint"], output_dir=arguments["output_dir"],
             device_name=arguments["device"], limit=arguments["test_limit"])
    else:
        settings = {name: arguments[name] if arguments.get(name) is not None else default
                    for name, default in DEFAULT_SETTINGS.items()}
        for seed in arguments["seeds"]:
            checkpoint_path = train(settings, seed=seed, output_dir=arguments["output_dir"],
                                    device_name=arguments["device"])
            if arguments["mode"] == "both":
                test(checkpoint_path, output_dir=arguments["output_dir"],
                     device_name=arguments["device"], limit=arguments["test_limit"])
