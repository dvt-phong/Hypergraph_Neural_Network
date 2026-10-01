# 8. Train on the train hypergraph, select the checkpoint on validation, pick
#    the decision threshold on validation, then evaluate the selected checkpoint
#    once on the official test split with that threshold.
#
# One epoch = one full-batch step on the whole train hypergraph:
#     Z0, H*, Z*, logits = model(X, H0)
#     loss = BCE + λ · intra-hyperedge contrastive(Z0, Z*)
# Validation/test targets are scored on their own local graphs (4_hypergraph.py).
#
# Threshold: F1 is reported at the threshold t* that maximizes F1 on the full
# validation split (Lipton et al., 2014), not at a fixed 0.5. t* is stored in
# the checkpoint and reused for test. F1 at 0.5 is still reported for comparison.

import argparse
import csv
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

config = import_module("0_config")
hypergraph_module = import_module("4_hypergraph")
model_module = import_module("5_model")
loss_module = import_module("7_losses")

resolve_device = hypergraph_module.resolve_device
load_train_graph = hypergraph_module.load_train_graph
load_evaluation_split = hypergraph_module.load_evaluation_split
build_local_graph = hypergraph_module.build_local_graph
merge_local_graphs = hypergraph_module.merge_local_graphs
HSLModel = model_module.HSLModel
graph_to_device = model_module.graph_to_device
positive_class_weight = loss_module.positive_class_weight
build_neighbor_sampler = loss_module.build_neighbor_sampler
sample_hyperedge_neighbors = loss_module.sample_hyperedge_neighbors
total_loss = loss_module.total_loss

# All experiment settings in one place; the command line can override them.
DEFAULT_SETTINGS = {
    "feature_set": "full",
    "families": list(config.GRAPH_FAMILIES),  # hyperedge families; ["self_loop"] = MLP
    "hypergraph": "hypergraph.npz",  # bundle from 4_hypergraph.py inside --output-dir
    "hidden_dim": 128,
    "dropout": 0.5,             # as in HGNN (Feng et al., 2019)
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "lr_schedule": "none",      # "multistep": lr × 0.9 at epoch 100, as in HGNN
    "epochs": 600,
    "eval_every": 5,            # validate every N epochs
    "patience": 20,             # stop after N validations without improvement
    "validation_limit": 5000,   # fixed random validation subset during training (0 = all)
    "select_metric": "auprc",   # checkpoint selection: "auprc" or "auc"
    "pos_weight": 1.0,          # BCE weight of the dropout class; "balanced" = #neg / #pos
    "eval_batch_size": 8,       # local graphs per forward pass
    "tag": "",                  # appended to the run name to keep sweep runs apart
    # Encoder (5_model.py)
    "skip_connection": False,   # classifier also sees MLP(X), the node's own features
    "family_weights": False,    # one learned weight per hyperedge family (HGNN's W)
    # Adam moves each parameter by about lr per step, so with the main lr the
    # family weights could only change by ~0.6 in 600 epochs, and weight decay
    # pulled them all toward softplus(0). They get their own lr and no decay.
    "family_weight_lr": 0.05,
    # One weight α_e per hyperedge on top of w_f, from a small MLP of the
    # hyperedge (5_model.py). Its own lr, no weight decay, for the same reason.
    "edge_weights": False,
    "edge_weight_lr": 0.005,
    # A node only receives from hyperedges whose members all started their
    # course no later than it, so nothing comes from after its 35 days (5_model.py).
    "causal": False,
    # Control experiment: hyperedges keep their sizes but get random train
    # members (permutation seed; None = the real hypergraph). 4_hypergraph.py.
    "shuffle_graph": None,
    # HSL
    "hsl": True,                # False = plain HGNN baseline
    "edge_sampling": True,      # Me
    "node_sampling": True,      # Mv
    "add_per_edge": 2,          # ΔH additions per Behavioral hyperedge (0 = none)
    # Contrastive loss
    "lambda_cl": 0.1,
    "temperature": 0.07,
    "contrastive_anchors": 1024,
    "contrastive_neighbors": 32,
}


def log(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Threshold with the highest F1 of the dropout class on these labels.
def best_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])  # the last point has no threshold


# Positive class = dropout. "negative" metrics are for the non-dropout class.
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


def format_metrics(metrics):
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())


# e.g. "hsl_full_seed_1", "hsl_no-cl_full_seed_1", "hgnn_course-object_full_seed_1",
# "mlp_full_seed_1", "hsl_full_lr3e-3_seed_1" (with --tag lr3e-3).
def make_run_name(settings, seed):
    name = "hsl" if settings["hsl"] else "hgnn"
    if settings["hsl"]:
        if not settings["edge_sampling"]:
            name += "_no-edge"
        if not settings["node_sampling"]:
            name += "_no-node"
        if settings["add_per_edge"] == 0:
            name += "_no-add"
        if settings["lambda_cl"] == 0:
            name += "_no-cl"
    families = [family for family in settings["families"] if family != "self_loop"]
    if not families:
        name = "mlp" if not settings["hsl"] else name + "_no-graph"
    elif families != list(config.GRAPH_FAMILIES):
        name += "_" + "-".join(families)
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
    if settings["edge_weights"]:
        name += "_ew"
    name += f"_{settings['feature_set']}"
    if settings["tag"]:
        name += f"_{settings['tag']}"
    return f"{name}_seed_{seed}"


def make_model(input_dim, settings):
    hsl_options = None
    if settings["hsl"]:
        hsl_options = {
            "add_per_edge": settings["add_per_edge"],
            "edge_sampling": settings["edge_sampling"],
            "node_sampling": settings["node_sampling"],
        }
    return HSLModel(input_dim, settings["hidden_dim"], settings["dropout"], hsl_options,
                    skip_connection=settings["skip_connection"],
                    family_weights=settings["family_weights"],
                    edge_weights=settings["edge_weights"],
                    causal=settings["causal"])


def load_split(output_dir, split_name, settings):
    return load_evaluation_split(
        output_dir, split_name=split_name, feature_set=settings["feature_set"],
        families=settings["families"], hypergraph_file=settings["hypergraph"],
        shuffle_seed=settings["shuffle_graph"],
    )


# Dropout probabilities of validation/test targets; each target only sees
# train enrollments. `limit` scores a fixed random subset (same for every call).
# With return_parts, also the logit split {logit_graph, logit_self} (skip models).
@torch.no_grad()
def predict(model, split_data, settings, device, *, limit=0, seed=0, return_parts=False):
    target_ids = np.arange(len(split_data["nodes"]))
    if limit:
        chosen = np.random.default_rng(seed).choice(target_ids, min(limit, len(target_ids)), replace=False)
        target_ids = np.sort(chosen)

    model.eval()
    batch_size = settings["eval_batch_size"]
    all_labels = []
    all_probabilities = []
    all_parts = defaultdict(list)
    started_at = last_log = time.perf_counter()
    for start in range(0, len(target_ids), batch_size):
        batch_ids = target_ids[start:start + batch_size]
        local_graphs = [build_local_graph(split_data, int(target_id)) for target_id in batch_ids]
        features, graph, target_rows, labels = merge_local_graphs(local_graphs)

        output = model(torch.as_tensor(features, device=device), graph_to_device(graph, device))
        rows = torch.as_tensor(target_rows, device=device)
        all_probabilities.extend(torch.sigmoid(output["logits"][rows]).cpu().tolist())
        all_labels.extend(labels.tolist())
        for name in ("logit_graph", "logit_self"):
            if name in output:
                all_parts[name].extend(output[name][rows].cpu().tolist())

        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            done = start + len(batch_ids)
            log(split_data["split_name"], f"{done:,}/{len(target_ids):,} targets, {last_log - started_at:.0f}s")

    log(split_data["split_name"], f"{len(target_ids):,} targets in {time.perf_counter() - started_at:.0f}s")
    if return_parts:
        parts = {name: np.asarray(values) for name, values in all_parts.items()}
        return np.asarray(all_labels), np.asarray(all_probabilities), parts
    return np.asarray(all_labels), np.asarray(all_probabilities)


# Which branch carries the prediction of a skip model. The logit is
# logit_graph + logit_self + bias, so the AUC of one part alone is the AUC of
# the model with the other branch held constant (AUC ignores the shift).
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


# Labels and probabilities for plots (scripts/plot_results.py); `parts` adds
# the logit split of skip models.
def save_probabilities(run_name, split_name, labels, probabilities, threshold, parts=None):
    path = Path(config.REPORTS) / f"{run_name}_{split_name}_probs.npz"
    np.savez_compressed(path, labels=labels, probabilities=probabilities, threshold=threshold,
                        **(parts or {}))
    log(split_name, f"probabilities={path}")


# Learned weight of every train hyperedge (--edge-weights), one CSV row per
# hyperedge without the self-loops: family, key, size, α_e, W_ee = w_f · α_e,
# and the dropout rate of its members. Labels only describe the hyperedges for
# analysis (docs/PHAN_TICH_HSL.md, section 7); the model never reads them.
@torch.no_grad()
def save_edge_weights(model, x, graph, train_data, run_name):
    model.eval()
    alpha = model.edge_alpha(x, graph)
    weight = model.hyperedge_weights(graph, alpha).cpu().numpy()
    alpha = alpha.cpu().numpy()
    node_ids = train_data["graph"]["node_ids"]
    edge_ids = train_data["graph"]["edge_ids"]
    edge_family = train_data["graph"]["edge_family"]
    size = np.bincount(edge_ids, minlength=len(edge_family))
    dropouts = np.bincount(edge_ids, weights=train_data["labels"][node_ids], minlength=len(edge_family))

    path = Path(config.REPORTS) / f"{run_name}_edge_weights.csv"
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["edge_id", "family", "key", "size", "alpha", "weight", "dropout_rate"])
        for edge_id, key in enumerate(train_data["edge_keys"]):  # self-loops come after these
            writer.writerow([
                edge_id, config.EDGE_FAMILIES[edge_family[edge_id]], key, size[edge_id],
                f"{alpha[edge_id]:.4f}", f"{weight[edge_id]:.4f}",
                f"{dropouts[edge_id] / size[edge_id]:.4f}",
            ])
    log("train", f"hyperedge weights={path}")


def train(settings, *, seed, output_dir=config.PROCESSED, device_name="auto"):
    set_seed(seed)
    device = resolve_device(device_name)
    run_name = make_run_name(settings, seed)
    log("setup", f"run={run_name}, device={device}, settings={settings}")

    # Data: X, labels, H0 of the train split; validation targets.
    train_data = load_train_graph(
        output_dir, feature_set=settings["feature_set"], families=settings["families"],
        hypergraph_file=settings["hypergraph"], shuffle_seed=settings["shuffle_graph"],
    )
    validation_data = load_split(output_dir, "validation", settings)
    x = torch.as_tensor(train_data["features"], device=device)
    labels = torch.as_tensor(train_data["labels"], device=device)
    graph = graph_to_device(train_data["graph"], device)
    sampler = build_neighbor_sampler(train_data["graph"])
    positive_weight = positive_class_weight(labels, settings["pos_weight"])
    rng = np.random.default_rng(seed)
    log("setup", f"X={tuple(x.shape)}, hyperedges={graph['num_edges']:,}, "
        f"memberships={len(graph['node_ids']):,}, pos_weight={float(positive_weight):.3f}")

    model = make_model(x.shape[1], settings).to(device)
    own_groups = ("family_logits", "edge_weight_scorer.")
    parameter_groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith(own_groups)]}]
    if model.family_logits is not None:
        parameter_groups.append({"params": [model.family_logits],
                                 "lr": settings["family_weight_lr"], "weight_decay": 0.0})
    if model.edge_weight_scorer is not None:
        parameter_groups.append({"params": list(model.edge_weight_scorer.parameters()),
                                 "lr": settings["edge_weight_lr"], "weight_decay": 0.0})
    optimizer = torch.optim.Adam(
        parameter_groups, lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    scheduler = None
    if settings["lr_schedule"] == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.9)
    lambda_cl = settings["lambda_cl"] if settings["hsl"] else 0.0

    Path(config.RUNS).mkdir(parents=True, exist_ok=True)
    Path(config.REPORTS).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.RUNS) / f"{run_name}.pt"
    select_metric = settings["select_metric"]
    history = []
    best = {"score": -1.0, "epoch": 0, "validation": None}
    stale_validations = 0

    for epoch in range(1, settings["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        output = model(x, graph)

        anchor_count = min(settings["contrastive_anchors"], len(sampler["anchor_pool"]))
        anchors = rng.choice(sampler["anchor_pool"], anchor_count, replace=False)
        neighbors = sample_hyperedge_neighbors(sampler, anchors, settings["contrastive_neighbors"], rng)
        loss, parts = total_loss(
            output, labels, positive_weight,
            torch.as_tensor(anchors, device=device), torch.as_tensor(neighbors, device=device),
            lambda_cl=lambda_cl, temperature=settings["temperature"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        train_auc = roc_auc_score(train_data["labels"], output["logits"].detach().cpu().numpy())
        record = {"epoch": epoch, "loss": loss.item(), **parts, "train_auc": float(train_auc),
                  **output["structure"], "seconds": time.perf_counter() - epoch_started}
        log("train", f"epoch {epoch}: " + ", ".join(
            f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
            for name, value in record.items() if name != "epoch"
        ))

        if epoch % settings["eval_every"] == 0 or epoch == settings["epochs"]:
            validation_started = time.perf_counter()
            # The same validation subset for every seed and run, so curves compare.
            validation_labels, validation_probabilities = predict(
                model, validation_data, settings, device,
                limit=settings["validation_limit"], seed=config.SPLIT_SEED,
            )
            # Metrics at the best threshold of this validation subset: shows
            # where the threshold drifts while training.
            validation = classification_metrics(
                validation_labels, validation_probabilities,
                best_threshold(validation_labels, validation_probabilities),
            )
            record["validation"] = validation
            record["validation_seconds"] = time.perf_counter() - validation_started
            log("validation", format_metrics(validation))
            if validation[select_metric] > best["score"]:
                best = {"score": validation[select_metric], "epoch": epoch, "validation": validation}
                stale_validations = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "input_dim": x.shape[1],
                    "settings": settings,
                    "seed": seed,
                    "epoch": epoch,
                    "validation": validation,
                }, checkpoint_path)
                log("checkpoint", f"new best val_{select_metric}={validation[select_metric]:.4f} at epoch {epoch}")
            else:
                stale_validations += 1
                if stale_validations >= settings["patience"]:
                    history.append(record)
                    log("train", f"early stopping after {stale_validations} validations without improvement")
                    break
        history.append(record)

    # Decision threshold t* of the best checkpoint, on the full validation split.
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    validation_labels, validation_probabilities = predict(model, validation_data, settings, device)
    threshold = best_threshold(validation_labels, validation_probabilities)
    full_validation = classification_metrics(validation_labels, validation_probabilities, threshold)
    log("validation", f"full split at t*: {format_metrics(full_validation)}")
    save_probabilities(run_name, "validation", validation_labels, validation_probabilities, threshold)
    torch.save({**saved, "threshold": threshold, "full_validation": full_validation}, checkpoint_path)
    if model.edge_weight_scorer is not None:
        save_edge_weights(model, x, graph, train_data, run_name)

    report = {"checkpoint": str(checkpoint_path), "best_epoch": best["epoch"],
              "select_metric": select_metric, "threshold": threshold,
              "best_validation": full_validation, "selection_validation": best["validation"],
              "settings": settings, "seed": seed, "history": history}
    report_path = Path(config.REPORTS) / f"{run_name}_train.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("train", f"best epoch={best['epoch']}, val_{select_metric}={best['score']:.4f}, "
        f"t*={threshold:.4f}; report={report_path}")
    return report


# Checkpoints saved before the User family existed have one family fewer in the
# family weights and in the family one-hot of the HSL hyperedge scorer. Insert
# the User entry (initial weight / zero column), so those checkpoints still load.
def upgrade_state_dict(state, model):
    user = config.EDGE_FAMILIES.index("user")
    current = model.state_dict()
    state = dict(state)
    if "family_logits" in state and state["family_logits"].shape != current["family_logits"].shape:
        old = state["family_logits"]
        state["family_logits"] = torch.cat([old[:user], current["family_logits"][user:user + 1], old[user:]])
    name = "structure_learner.edge_scorer.0.weight"
    if name in state and state[name].shape != current[name].shape:
        old = state[name]
        column = old.shape[1] - (len(config.EDGE_FAMILIES) - 1) + user
        state[name] = torch.cat([old[:, :column], torch.zeros_like(old[:, :1]), old[:, column:]], dim=1)
    return state


# Test runs only on a checkpoint that validation has already selected, with the
# threshold chosen on validation.
def test(checkpoint_path, *, output_dir=config.PROCESSED, device_name="auto", limit=0):
    device = resolve_device(device_name)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Older checkpoints lack newer settings; their defaults match the old behavior,
    # except families: before User hyperedges, "all families" had no User.
    settings = {**DEFAULT_SETTINGS, **saved["settings"]}
    if "families" not in saved["settings"]:
        settings["families"] = ["course", "object", "behavioral"]
    model = make_model(saved["input_dim"], settings).to(device)
    model.load_state_dict(upgrade_state_dict(saved["state_dict"], model))
    run_name = Path(checkpoint_path).stem

    threshold = saved.get("threshold")
    threshold_source = "checkpoint"
    if threshold is None:
        # Checkpoints saved before thresholds existed: choose t* on validation now.
        validation_data = load_split(output_dir, "validation", settings)
        validation_labels, validation_probabilities = predict(model, validation_data, settings, device)
        threshold = best_threshold(validation_labels, validation_probabilities)
        threshold_source = "validation, computed at test time"
        save_probabilities(run_name, "validation", validation_labels, validation_probabilities, threshold)
    log("test", f"threshold t*={threshold:.4f} ({threshold_source})")

    test_data = load_split(output_dir, "test", settings)
    labels, probabilities, parts = predict(model, test_data, settings, device, limit=limit,
                                           seed=saved["seed"], return_parts=True)
    metrics = classification_metrics(labels, probabilities, threshold)
    log("test", format_metrics(metrics))
    branches = branch_metrics(labels, parts)
    if branches:
        log("test", "branches: " + format_metrics(branches))
    save_probabilities(run_name, "test", labels, probabilities, threshold, parts)

    report = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": saved["epoch"],
              "checkpoint_validation": saved.get("full_validation", saved["validation"]),
              "threshold": threshold, "threshold_source": threshold_source, "test": metrics,
              "branches": branches}
    report_path = Path(config.REPORTS) / f"{run_name}_test.json"
    Path(config.REPORTS).mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("test", f"report={report_path}")
    return report


def pos_weight_argument(value):
    return value if value == "balanced" else float(value)


# "course,object" -> ["course", "object"]; "self_loop" alone keeps no family.
def families_argument(value):
    families = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in families if name not in config.EDGE_FAMILIES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown families {unknown}; choose from {config.EDGE_FAMILIES}")
    return families


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config.TRAIN_CLI_DESCRIPTION)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", choices=config.SEEDS, default=[1])
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--feature-set", choices=("behavior", "behavior_user", "behavior_course", "full"))
    parser.add_argument("--families", type=families_argument,
                        help="comma-separated, e.g. course,object,user; self_loop alone = MLP")
    parser.add_argument("--hypergraph", help="bundle inside --output-dir, e.g. hypergraph_temporal.npz")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr-schedule", choices=("none", "multistep"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--select-metric", choices=("auprc", "auc"))
    parser.add_argument("--pos-weight", type=pos_weight_argument, help='a number, or "balanced"')
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--tag", help="suffix for the run name, e.g. lr3e-3")
    parser.add_argument("--skip-connection", action="store_true", default=None,
                        help="classifier also sees MLP(X), the node's own features")
    parser.add_argument("--family-weights", action="store_true", default=None,
                        help="learn one weight per hyperedge family")
    parser.add_argument("--family-weight-lr", type=float,
                        help="Adam lr of the family weights (no weight decay)")
    parser.add_argument("--edge-weights", action="store_true", default=None,
                        help="learn one weight per hyperedge on top of the family weights")
    parser.add_argument("--edge-weight-lr", type=float,
                        help="Adam lr of the per-hyperedge weight scorer (no weight decay)")
    parser.add_argument("--causal", action="store_true", default=None,
                        help="a node only receives from hyperedges whose members started no later")
    parser.add_argument("--shuffle-graph", type=int, metavar="SEED",
                        help="control: hyperedges keep their sizes but get random train members")
    parser.add_argument("--lambda-cl", type=float)
    parser.add_argument("--add-per-edge", type=int)
    # Ablations (HSL paper, Fig. 3)
    parser.add_argument("--no-hsl", dest="hsl", action="store_false", default=None)
    parser.add_argument("--no-edge-sampling", dest="edge_sampling", action="store_false", default=None)
    parser.add_argument("--no-node-sampling", dest="node_sampling", action="store_false", default=None)
    arguments = vars(parser.parse_args())

    if arguments["mode"] == "test":
        if arguments["checkpoint"] is None:
            parser.error("--checkpoint is required for --mode test")
        test(arguments["checkpoint"], output_dir=arguments["output_dir"],
             device_name=arguments["device"], limit=arguments["test_limit"])
    else:
        # Command-line values that were given replace the defaults.
        settings = dict(DEFAULT_SETTINGS)
        for name in DEFAULT_SETTINGS:
            if arguments.get(name) is not None:
                settings[name] = arguments[name]
        for seed in arguments["seeds"]:
            report = train(settings, seed=seed, output_dir=arguments["output_dir"],
                           device_name=arguments["device"])
            if arguments["mode"] == "both":
                test(report["checkpoint"], output_dir=arguments["output_dir"],
                     device_name=arguments["device"], limit=arguments["test_limit"])
