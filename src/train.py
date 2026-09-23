"""Train HGSL, select a checkpoint on validation, then evaluate on test."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from hypergraph import build_local_graph, load_evaluation_data, load_train_graph
from losses import total_loss, train_pos_weight
from model import HGSLModel
from preprocess import PROCESSED, ROOT, SEEDS


RUNS = ROOT / "outputs" / "runs"
REPORTS = ROOT / "outputs" / "reports"


def metrics(labels, scores):
    """Compute binary classification metrics without an extra dependency."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores):
        raise ValueError("Labels and predictions have different lengths")
    if len(np.unique(labels)) != 2:
        raise ValueError("Metrics require both label classes")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    stops = np.r_[starts[1:], len(scores)]
    for start, stop in zip(starts, stops):
        average_rank = (start + 1 + stop) / 2
        ranks[order[start:stop]] = average_rank

    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    positive_rank_sum = ranks[labels == 1].sum()
    minimum_positive_rank_sum = positive_count * (positive_count + 1) / 2
    auc = (
        positive_rank_sum - minimum_positive_rank_sum
    ) / (positive_count * negative_count)

    descending = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[descending]
    ordered_scores = scores[descending]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), len(scores) - 1]
    true_positives = np.cumsum(ordered_labels)[ends]
    precision_curve = true_positives / (ends + 1)
    recall_curve = true_positives / positive_count
    recall_steps = np.diff(np.r_[0.0, recall_curve])
    auprc = np.sum(recall_steps * precision_curve)

    predicted = scores >= 0.5
    true_positive = int(np.count_nonzero(predicted & (labels == 1)))
    false_positive = int(np.count_nonzero(predicted & (labels == 0)))
    false_negative = int(np.count_nonzero(~predicted & (labels == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def _batch_graphs(graphs):
    feature_blocks = []
    incidence_blocks = []
    family_blocks = []
    size_blocks = []
    target_positions = []
    node_group_blocks = []
    edge_group_blocks = []
    labels = []
    node_offset = 0

    for graph_id, graph in enumerate(graphs):
        x, h0, families, sizes, label = graph
        feature_blocks.append(x)
        incidence_blocks.append(h0)
        family_blocks.append(families)
        size_blocks.append(sizes)
        target_positions.append(node_offset)
        node_group_blocks.append(np.full(len(x), graph_id, dtype=np.int64))
        edge_group_blocks.append(np.full(h0.shape[1], graph_id, dtype=np.int64))
        labels.append(label)
        node_offset += len(x)

    return (
        np.concatenate(feature_blocks),
        sparse.block_diag(incidence_blocks, format="csr"),
        np.concatenate(family_blocks),
        np.concatenate(size_blocks),
        np.asarray(target_positions, dtype=np.int64),
        np.concatenate(node_group_blocks),
        np.concatenate(edge_group_blocks),
        np.asarray(labels, dtype=np.int8),
    )


def _evaluation_targets(data, split_name, seed, limit):
    targets = []
    for node_id, node_split in enumerate(data["split"]):
        if node_split == split_name:
            targets.append(node_id)
    targets = np.asarray(targets, dtype=np.int64)

    if not limit:
        return targets
    if limit < 2:
        raise ValueError("Evaluation limit must be zero or at least two")

    negative = []
    positive = []
    for node_id in targets:
        if data["nodes"][node_id]["label"] == "1":
            positive.append(node_id)
        else:
            negative.append(node_id)
    if not negative or not positive:
        raise ValueError("Limited evaluation requires both label classes")

    rng = np.random.default_rng(seed)
    selected = [int(rng.choice(negative)), int(rng.choice(positive))]
    remaining = np.setdiff1d(targets, selected)
    extra_count = min(limit - 2, len(remaining))
    if extra_count:
        extra = rng.choice(remaining, extra_count, replace=False)
        selected.extend(int(node_id) for node_id in extra)
    return np.sort(np.asarray(selected, dtype=np.int64))


def evaluate(
    model,
    data,
    split_name,
    *,
    seed,
    device,
    batch_size=4,
    limit=0,
    hsl=True,
    refinement=None,
):
    targets = _evaluation_targets(data, split_name, seed, limit)
    probabilities = []
    labels = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            graphs = []
            for target in targets[start:start + batch_size]:
                graphs.append(build_local_graph(data, int(target)))

            (
                x,
                h0,
                families,
                sizes,
                positions,
                node_groups,
                edge_groups,
                batch_labels,
            ) = _batch_graphs(graphs)
            output = model(
                torch.as_tensor(x, device=device),
                h0,
                families,
                sizes,
                np.random.default_rng(seed),
                hsl=hsl,
                deterministic=True,
                refinement=refinement,
                node_groups=node_groups,
                edge_groups=edge_groups,
            )
            batch_probabilities = torch.sigmoid(output["logits"][positions])
            probabilities.extend(batch_probabilities.cpu().tolist())
            labels.extend(batch_labels.tolist())
    return metrics(labels, probabilities), len(targets)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _check_hsl_gradients(model):
    parameters = (
        model.node_projection.weight,
        model.edge_projection.weight,
        model.membership_bias,
    )
    squared_norm = torch.zeros((), device=model.membership_bias.device)
    for parameter in parameters:
        if parameter.grad is None:
            raise ValueError("No gradient reached the HSL membership scorer")
        if not torch.isfinite(parameter.grad).all():
            raise ValueError("HSL membership scorer has a non-finite gradient")
        squared_norm = squared_norm + parameter.grad.square().sum()
    if squared_norm <= 0:
        raise ValueError("HSL membership scorer has a zero gradient")


def train(
    *,
    seed=1,
    feature_set="behavior",
    epochs=50,
    patience=5,
    output_dir=PROCESSED,
    device_name="auto",
    validation_limit=0,
    validation_batch_size=4,
    hidden_dim=64,
    dropout=0.5,
    learning_rate=0.001,
    weight_decay=0.0005,
    lambda_cl=0.1,
    temperature=0.2,
    contrastive_nodes=512,
    hsl=True,
    refinement=None,
    runs_dir=RUNS,
    reports_dir=REPORTS,
):
    """Run full-batch training and select the best validation-AUC checkpoint."""
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")

    output_dir = Path(output_dir)
    device = _device(device_name)
    _set_seed(seed)
    x, h0, y, families, sizes = load_train_graph(
        output_dir,
        seed=seed,
        feature_set=feature_set,
    )
    evaluation_data = load_evaluation_data(
        output_dir,
        seed=seed,
        feature_set=feature_set,
    )

    x_tensor = torch.as_tensor(x, device=device)
    labels = torch.as_tensor(y, device=device)
    positive_weight = train_pos_weight(labels)
    model = HGSLModel(x.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    refinement = {
        "sampled_hyperedges": 96,
        "positive_nodes": 16,
        "negative_nodes": 16,
        "top_r": 8,
        "threshold": None,
        **(refinement or {}),
    }

    settings = {
        "seed": seed,
        "feature_set": feature_set,
        "epochs": epochs,
        "patience": patience,
        "validation_limit": validation_limit,
        "validation_batch_size": validation_batch_size,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lambda_cl": lambda_cl,
        "temperature": temperature,
        "contrastive_nodes": contrastive_nodes,
        "hsl": hsl,
        "refinement": refinement,
    }
    runs_dir = Path(runs_dir)
    reports_dir = Path(reports_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_name = "hgsl" if hsl else "hgnn"
    run_name = f"simple_{model_name}_{feature_set}_seed_{seed}"
    checkpoint = runs_dir / f"{run_name}.pt"

    history = []
    best_auc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    validation_count = 0
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(seed * 100_003 + epoch)
        sample_count = min(contrastive_nodes, len(y))
        sampled = rng.choice(len(y), sample_count, replace=False)
        sampled = torch.as_tensor(sampled, dtype=torch.int64, device=device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            x_tensor,
            h0,
            families,
            sizes,
            rng,
            hsl=hsl,
            refinement=refinement,
        )
        contrastive_weight = lambda_cl if hsl else 0.0
        loss, loss_parts = total_loss(
            output,
            labels,
            positive_weight,
            sampled,
            lambda_cl=contrastive_weight,
            temperature=temperature,
        )
        if not torch.isfinite(loss):
            raise ValueError("Training loss is not finite")
        loss.backward()
        if hsl:
            _check_hsl_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        validation, validation_count = evaluate(
            model,
            evaluation_data,
            "validation",
            seed=seed,
            device=device,
            batch_size=validation_batch_size,
            limit=validation_limit,
            hsl=hsl,
            refinement=refinement,
        )
        epoch_record = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            **loss_parts,
            "validation": validation,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch}: loss={float(loss.detach()):.4f} "
            f"val_auc={validation['auc']:.4f}",
            flush=True,
        )

        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "validation": validation,
                "input_dim": x.shape[1],
                "settings": settings,
            }, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    report = {
        "checkpoint": str(checkpoint),
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "validation_targets": validation_count,
        "train_nodes": len(y),
        "positive_weight": float(positive_weight),
        "history": history,
        "settings": settings,
    }
    report_path = reports_dir / f"{run_name}_train.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved validation-selected checkpoint: {checkpoint}", flush=True)
    return report


def test(
    checkpoint,
    *,
    output_dir=PROCESSED,
    device_name="auto",
    test_limit=0,
    batch_size=4,
    reports_dir=REPORTS,
):
    """Evaluate a validation-selected checkpoint on the held-out test split."""
    device = _device(device_name)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    settings = saved["settings"]
    model = HGSLModel(
        saved["input_dim"],
        settings["hidden_dim"],
        settings["dropout"],
    ).to(device)
    model.load_state_dict(saved["state_dict"])

    data = load_evaluation_data(
        output_dir,
        seed=settings["seed"],
        feature_set=settings["feature_set"],
    )
    results, target_count = evaluate(
        model,
        data,
        "test",
        seed=settings["seed"],
        device=device,
        batch_size=batch_size,
        limit=test_limit,
        hsl=settings["hsl"],
        refinement=settings["refinement"],
    )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": saved["epoch"],
        "checkpoint_validation": saved["validation"],
        "test_targets": target_count,
        "test_metrics": results,
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{Path(checkpoint).stem}_test.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS)
    parser.add_argument(
        "--feature-set",
        choices=("behavior", "behavior_user", "behavior_course", "full"),
        default="behavior",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-hsl", action="store_true")
    arguments = parser.parse_args()

    if arguments.mode == "test":
        if arguments.checkpoint is None:
            parser.error("--checkpoint is required for --mode test")
        print(test(
            arguments.checkpoint,
            output_dir=arguments.output_dir,
            device_name=arguments.device,
            test_limit=arguments.test_limit,
        ), flush=True)
    else:
        seeds = arguments.seeds or [arguments.seed]
        for experiment_seed in seeds:
            result = train(
                seed=experiment_seed,
                feature_set=arguments.feature_set,
                epochs=arguments.epochs,
                patience=arguments.patience,
                output_dir=arguments.output_dir,
                device_name=arguments.device,
                validation_limit=arguments.validation_limit,
                hsl=not arguments.no_hsl,
            )
            if arguments.mode == "both":
                print(test(
                    result["checkpoint"],
                    output_dir=arguments.output_dir,
                    device_name=arguments.device,
                    test_limit=arguments.test_limit,
                ), flush=True)
