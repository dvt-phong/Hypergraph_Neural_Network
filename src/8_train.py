# 8. Train on the train hypergraph, select the checkpoint on validation AUC,
#    then evaluate the selected checkpoint once on the official test split.
#
# One epoch = one full-batch step on the whole train hypergraph:
#     Z0, H*, Z*, logits = model(X, H0)
#     loss = weighted BCE + λ · intra-hyperedge contrastive(Z0, Z*)
# Validation/test targets are scored on their own local graphs (4_hypergraph.py).

import argparse
import json
import random
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

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
    "hidden_dim": 128,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "epochs": 200,
    "eval_every": 5,            # validate every N epochs
    "patience": 5,              # stop after N validations without improvement
    "validation_limit": 0,      # 0 = all validation targets
    "eval_batch_size": 8,       # local graphs per forward pass
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


def classification_metrics(labels, probabilities):
    predicted = probabilities >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


# e.g. "hsl_full_seed_1", "hsl_no-cl_full_seed_1", "hgnn_full_seed_1".
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
    return f"{name}_{settings['feature_set']}_seed_{seed}"


def make_model(input_dim, settings):
    hsl_options = None
    if settings["hsl"]:
        hsl_options = {
            "add_per_edge": settings["add_per_edge"],
            "edge_sampling": settings["edge_sampling"],
            "node_sampling": settings["node_sampling"],
        }
    return HSLModel(input_dim, settings["hidden_dim"], settings["dropout"], hsl_options)


# Score validation/test targets; each target only sees train enrollments.
@torch.no_grad()
def evaluate(model, split_data, settings, device, *, limit=0, seed=0):
    target_ids = np.arange(len(split_data["nodes"]))
    if limit:
        chosen = np.random.default_rng(seed).choice(target_ids, min(limit, len(target_ids)), replace=False)
        target_ids = np.sort(chosen)

    model.eval()
    batch_size = settings["eval_batch_size"]
    all_labels = []
    all_probabilities = []
    started_at = last_log = time.perf_counter()
    for start in range(0, len(target_ids), batch_size):
        batch_ids = target_ids[start:start + batch_size]
        local_graphs = [build_local_graph(split_data, int(target_id)) for target_id in batch_ids]
        features, graph, target_rows, labels = merge_local_graphs(local_graphs)

        output = model(torch.as_tensor(features, device=device), graph_to_device(graph, device))
        target_logits = output["logits"][torch.as_tensor(target_rows, device=device)]
        all_probabilities.extend(torch.sigmoid(target_logits).cpu().tolist())
        all_labels.extend(labels.tolist())

        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            done = start + len(batch_ids)
            log(split_data["split_name"], f"{done:,}/{len(target_ids):,} targets, {last_log - started_at:.0f}s")

    metrics = classification_metrics(np.asarray(all_labels), np.asarray(all_probabilities))
    log(split_data["split_name"], f"{len(target_ids):,} targets in {time.perf_counter() - started_at:.0f}s: "
        + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
    return metrics


def train(settings, *, seed, output_dir=config.PROCESSED, device_name="auto"):
    set_seed(seed)
    device = resolve_device(device_name)
    run_name = make_run_name(settings, seed)
    log("setup", f"run={run_name}, device={device}, settings={settings}")

    # Data: X, labels, H0 of the train split; validation targets.
    train_data = load_train_graph(output_dir, feature_set=settings["feature_set"])
    validation_data = load_evaluation_split(
        output_dir, split_name="validation", feature_set=settings["feature_set"]
    )
    x = torch.as_tensor(train_data["features"], device=device)
    labels = torch.as_tensor(train_data["labels"], device=device)
    graph = graph_to_device(train_data["graph"], device)
    sampler = build_neighbor_sampler(train_data["graph"])
    positive_weight = positive_class_weight(labels)
    rng = np.random.default_rng(seed)
    log("setup", f"X={tuple(x.shape)}, hyperedges={graph['num_edges']:,}, "
        f"memberships={len(graph['node_ids']):,}, pos_weight={float(positive_weight):.3f}")

    model = make_model(x.shape[1], settings).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    lambda_cl = settings["lambda_cl"] if settings["hsl"] else 0.0

    Path(config.RUNS).mkdir(parents=True, exist_ok=True)
    Path(config.REPORTS).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.RUNS) / f"{run_name}.pt"
    history = []
    best = {"auc": -1.0, "epoch": 0, "validation": None}
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

        train_auc = roc_auc_score(train_data["labels"], output["logits"].detach().cpu().numpy())
        record = {"epoch": epoch, "loss": loss.item(), **parts, "train_auc": float(train_auc),
                  **output["structure"]}
        log("train", f"epoch {epoch}: " + ", ".join(
            f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
            for name, value in record.items() if name != "epoch"
        ) + f" ({time.perf_counter() - epoch_started:.1f}s)")

        if epoch % settings["eval_every"] == 0 or epoch == settings["epochs"]:
            validation = evaluate(model, validation_data, settings, device,
                                  limit=settings["validation_limit"], seed=seed)
            record["validation"] = validation
            if validation["auc"] > best["auc"]:
                best = {"auc": validation["auc"], "epoch": epoch, "validation": validation}
                stale_validations = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "input_dim": x.shape[1],
                    "settings": settings,
                    "seed": seed,
                    "epoch": epoch,
                    "validation": validation,
                }, checkpoint_path)
                log("checkpoint", f"new best val_auc={validation['auc']:.4f} at epoch {epoch}")
            else:
                stale_validations += 1
                if stale_validations >= settings["patience"]:
                    history.append(record)
                    log("train", f"early stopping after {stale_validations} validations without improvement")
                    break
        history.append(record)

    report = {"checkpoint": str(checkpoint_path), "best_epoch": best["epoch"],
              "best_validation": best["validation"], "settings": settings, "seed": seed,
              "history": history}
    report_path = Path(config.REPORTS) / f"{run_name}_train.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("train", f"best epoch={best['epoch']}, val_auc={best['auc']:.4f}; report={report_path}")
    return report


# Test runs only on a checkpoint that validation has already selected.
def test(checkpoint_path, *, output_dir=config.PROCESSED, device_name="auto", limit=0):
    device = resolve_device(device_name)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    settings = saved["settings"]
    model = make_model(saved["input_dim"], settings).to(device)
    model.load_state_dict(saved["state_dict"])

    test_data = load_evaluation_split(output_dir, split_name="test", feature_set=settings["feature_set"])
    metrics = evaluate(model, test_data, settings, device, limit=limit, seed=saved["seed"])
    report = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": saved["epoch"],
              "checkpoint_validation": saved["validation"], "test": metrics}
    report_path = Path(config.REPORTS) / f"{Path(checkpoint_path).stem}_test.json"
    Path(config.REPORTS).mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("test", f"report={report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config.TRAIN_CLI_DESCRIPTION)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", choices=config.SEEDS, default=[1])
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--feature-set", choices=("behavior", "behavior_user", "behavior_course", "full"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
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
