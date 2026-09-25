# 8. Train on the full train hypergraph, select on validation, then test.

import argparse
import json
import random
import time
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

config = import_module("0_config")
hypergraph_module = import_module("4_hypergraph")
model_module = import_module("5_model")
loss_module = import_module("7_losses")

build_local_graph = hypergraph_module.build_local_graph
load_evaluation_data = hypergraph_module.load_evaluation_data
load_train_graph = hypergraph_module.load_train_graph
HGSLModel = model_module.HGSLModel
total_loss = loss_module.total_loss
train_pos_weight = loss_module.train_pos_weight


def log_event(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


def memory_summary(device):
    if device.type != "cuda":
        return "device=cpu"
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    return f"cuda_allocated={allocated:.2f}GiB, cuda_reserved={reserved:.2f}GiB"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(device_name)


# AUC, AUPRC, Precision, Recall, and F1 without another dependency.
def classification_metrics(labels, prediction_scores):
    labels = np.asarray(labels, dtype=np.int8)
    prediction_scores = np.asarray(prediction_scores, dtype=np.float64)
    if len(labels) != len(prediction_scores):
        raise ValueError("Labels and predictions have different lengths")
    if len(np.unique(labels)) != 2:
        raise ValueError("Metrics require both label classes")

    # ROC-AUC from average ranks. Equal scores receive the same average rank.
    ascending_order = np.argsort(prediction_scores, kind="mergesort")
    sorted_scores = prediction_scores[ascending_order]
    ranks = np.empty(len(prediction_scores), dtype=np.float64)
    tie_starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    tie_stops = np.r_[tie_starts[1:], len(prediction_scores)]
    for tie_start, tie_stop in zip(tie_starts, tie_stops):
        average_rank = (tie_start + 1 + tie_stop) / 2
        ranks[ascending_order[tie_start:tie_stop]] = average_rank

    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    positive_rank_sum = ranks[labels == 1].sum()
    minimum_positive_rank_sum = positive_count * (positive_count + 1) / 2
    auc = (
        positive_rank_sum - minimum_positive_rank_sum
    ) / (positive_count * negative_count)

    # AUPRC from precision at each distinct score threshold.
    descending_order = np.argsort(-prediction_scores, kind="mergesort")
    ordered_labels = labels[descending_order]
    ordered_scores = prediction_scores[descending_order]
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(ordered_scores) != 0),
        len(prediction_scores) - 1,
    ]
    true_positives = np.cumsum(ordered_labels)[threshold_ends]
    precision_curve = true_positives / (threshold_ends + 1)
    recall_curve = true_positives / positive_count
    recall_steps = np.diff(np.r_[0.0, recall_curve])
    auprc = np.sum(recall_steps * precision_curve)

    # Classification metrics use the fixed probability threshold 0.5.
    predicted_positive = prediction_scores >= 0.5
    true_positive_count = int(
        np.count_nonzero(predicted_positive & (labels == 1))
    )
    false_positive_count = int(
        np.count_nonzero(predicted_positive & (labels == 0))
    )
    false_negative_count = int(
        np.count_nonzero(~predicted_positive & (labels == 1))
    )
    precision = true_positive_count / max(
        true_positive_count + false_positive_count,
        1,
    )
    recall = true_positive_count / max(
        true_positive_count + false_negative_count,
        1,
    )
    f1_score = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "f1": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
    }


# Batch independent local graphs without creating edges between target nodes.
def batch_local_graphs(local_graphs):
    feature_blocks = []
    incidence_blocks = []
    family_blocks = []
    size_blocks = []
    target_positions = []
    node_group_blocks = []
    edge_group_blocks = []
    labels = []
    node_offset = 0

    for local_graph_id, local_graph in enumerate(local_graphs):
        node_features = local_graph["features"]
        incidence_matrix = local_graph["incidence_matrix"]

        feature_blocks.append(node_features)
        incidence_blocks.append(incidence_matrix)
        family_blocks.append(local_graph["families"])
        size_blocks.append(local_graph["sizes"])
        target_positions.append(node_offset)
        node_group_blocks.append(
            np.full(len(node_features), local_graph_id, dtype=np.int64)
        )
        edge_group_blocks.append(
            np.full(
                incidence_matrix.shape[1],
                local_graph_id,
                dtype=np.int64,
            )
        )
        labels.append(local_graph["label"])
        node_offset += len(node_features)

    return {
        "features": np.concatenate(feature_blocks),
        "incidence_matrix": sparse.block_diag(
            incidence_blocks,
            format="csr",
        ),
        "families": np.concatenate(family_blocks),
        "sizes": np.concatenate(size_blocks),
        "target_positions": np.asarray(target_positions, dtype=np.int64),
        "node_groups": np.concatenate(node_group_blocks),
        "edge_groups": np.concatenate(edge_group_blocks),
        "labels": np.asarray(labels, dtype=np.int8),
    }


# Each prediction graph contains one target node and train reference nodes only.
def evaluate(
    model,
    evaluation_data,
    *,
    seed,
    device,
    batch_size=4,
    limit=0,
    hsl=True,
    refinement=None,
):
    started_at = time.perf_counter()
    split_name = evaluation_data.get("split_name", "evaluation")
    target_ids = np.arange(len(evaluation_data["nodes"]), dtype=np.int64)

    if limit:
        if limit < 2:
            raise ValueError("Evaluation limit must be zero or at least two")
        negative_target_ids = []
        positive_target_ids = []
        for target_id in target_ids:
            if evaluation_data["nodes"][target_id]["label"] == "1":
                positive_target_ids.append(target_id)
            else:
                negative_target_ids.append(target_id)
        if not negative_target_ids or not positive_target_ids:
            raise ValueError("Limited evaluation requires both label classes")

        random_generator = np.random.default_rng(seed)
        selected_target_ids = [
            int(random_generator.choice(negative_target_ids)),
            int(random_generator.choice(positive_target_ids)),
        ]
        remaining_target_ids = np.setdiff1d(target_ids, selected_target_ids)
        extra_count = min(limit - 2, len(remaining_target_ids))
        if extra_count > 0:
            extra_target_ids = random_generator.choice(
                remaining_target_ids,
                extra_count,
                replace=False,
            )
            for target_id in extra_target_ids:
                selected_target_ids.append(int(target_id))
        target_ids = np.sort(np.asarray(selected_target_ids, dtype=np.int64))

    total_batches = (len(target_ids) + batch_size - 1) // batch_size
    log_event(
        split_name,
        f"started: targets={len(target_ids):,}, batches={total_batches:,}, "
        f"batch_size={batch_size}, hsl={hsl}",
    )

    all_probabilities = []
    all_labels = []
    last_progress_at = started_at
    model.eval()

    with torch.no_grad():
        for batch_number, batch_start in enumerate(
            range(0, len(target_ids), batch_size),
            start=1,
        ):
            local_graphs = []
            batch_target_ids = target_ids[
                batch_start:batch_start + batch_size
            ]
            for target_id in batch_target_ids:
                local_graphs.append(
                    build_local_graph(evaluation_data, int(target_id))
                )
            batch = batch_local_graphs(local_graphs)

            model_output = model(
                torch.as_tensor(batch["features"], device=device),
                batch["incidence_matrix"],
                batch["families"],
                batch["sizes"],
                np.random.default_rng(seed),
                hsl=hsl,
                deterministic=True,
                refinement=refinement,
                node_groups=batch["node_groups"],
                edge_groups=batch["edge_groups"],
            )
            target_positions = torch.as_tensor(
                batch["target_positions"],
                dtype=torch.int64,
                device=device,
            )
            probabilities = torch.sigmoid(
                model_output["logits"][target_positions]
            )
            all_probabilities.extend(probabilities.cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

            now = time.perf_counter()
            if (
                batch_number == 1
                or batch_number == total_batches
                or now - last_progress_at >= 10
            ):
                completed = min(batch_start + batch_size, len(target_ids))
                log_event(
                    split_name,
                    f"{completed:,}/{len(target_ids):,} targets "
                    f"({100 * completed / max(len(target_ids), 1):.1f}%), "
                    f"local_nodes={len(batch['features']):,}, "
                    f"elapsed={now - started_at:.1f}s, {memory_summary(device)}",
                )
                last_progress_at = now

    evaluation_metrics = classification_metrics(
        all_labels,
        all_probabilities,
    )
    log_event(
        split_name,
        f"completed in {time.perf_counter() - started_at:.1f}s: "
        f"{evaluation_metrics}",
    )
    return evaluation_metrics, len(target_ids)


def train_one_epoch(
    model,
    optimizer,
    train_graph,
    node_feature_tensor,
    label_tensor,
    positive_weight,
    random_generator,
    settings,
    device,
):
    epoch_started_at = time.perf_counter()
    sampled_node_count = min(
        settings["contrastive_nodes"],
        len(label_tensor),
    )
    sampled_node_ids = random_generator.choice(
        len(label_tensor),
        sampled_node_count,
        replace=False,
    )
    sampled_node_ids = torch.as_tensor(
        sampled_node_ids,
        dtype=torch.int64,
        device=device,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_event("train", f"forward started; {memory_summary(device)}")
    forward_started_at = time.perf_counter()
    model_output = model(
        node_feature_tensor,
        train_graph["incidence_matrix"],
        train_graph["families"],
        train_graph["sizes"],
        random_generator,
        hsl=settings["hsl"],
        refinement=settings["refinement"],
    )
    log_event(
        "train",
        f"forward completed in {time.perf_counter() - forward_started_at:.1f}s; "
        f"{memory_summary(device)}",
    )

    contrastive_weight = settings["lambda_cl"] if settings["hsl"] else 0.0
    loss, loss_components = total_loss(
        model_output,
        label_tensor,
        positive_weight,
        sampled_node_ids,
        lambda_cl=contrastive_weight,
        temperature=settings["temperature"],
    )
    if not torch.isfinite(loss):
        raise ValueError("Training loss is not finite")

    log_event(
        "train",
        f"backward started: loss={float(loss.detach()):.4f}, "
        f"bce={loss_components['bce']:.4f}, "
        f"contrastive={loss_components['contrastive']:.4f}",
    )
    backward_started_at = time.perf_counter()
    loss.backward()
    log_event(
        "train",
        f"backward completed in {time.perf_counter() - backward_started_at:.1f}s; "
        f"{memory_summary(device)}",
    )

    if settings["hsl"]:
        hsl_parameters = (
            model.node_projection.weight,
            model.edge_projection.weight,
            model.membership_bias,
        )
        hsl_gradient_norm = torch.zeros(
            (),
            device=model.membership_bias.device,
        )
        for parameter in hsl_parameters:
            if parameter.grad is None:
                raise ValueError("No gradient reached the HSL membership scorer")
            if not torch.isfinite(parameter.grad).all():
                raise ValueError("HSL membership scorer has a non-finite gradient")
            hsl_gradient_norm = hsl_gradient_norm + parameter.grad.square().sum()
        if hsl_gradient_norm <= 0:
            raise ValueError("HSL membership scorer has a zero gradient")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    log_event(
        "train",
        f"optimizer step completed; epoch compute time="
        f"{time.perf_counter() - epoch_started_at:.1f}s",
    )
    return float(loss.detach()), loss_components


def train(
    *,
    seed=1,
    feature_set="behavior",
    epochs=50,
    patience=5,
    output_dir=config.PROCESSED,
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
    runs_dir=config.RUNS,
    reports_dir=config.REPORTS,
):
    started_at = time.perf_counter()
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")

    device = resolve_device(device_name)
    set_seed(seed)
    output_dir = Path(output_dir)
    log_event(
        "train",
        f"run started: seed={seed}, feature_set={feature_set}, epochs={epochs}, "
        f"patience={patience}, hsl={hsl}, device={device}",
    )

    # Load X_train, H0_train, labels, and validation references.
    train_graph = load_train_graph(output_dir, feature_set=feature_set)
    validation_data = load_evaluation_data(
        output_dir,
        split_name="validation",
        feature_set=feature_set,
    )
    node_feature_tensor = torch.as_tensor(
        train_graph["features"],
        device=device,
    )
    label_tensor = torch.as_tensor(train_graph["labels"], device=device)
    positive_weight = train_pos_weight(label_tensor)

    model = HGSLModel(
        train_graph["features"].shape[1],
        hidden_dim,
        dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    refinement_settings = {
        "sampled_hyperedges": 96,
        "positive_nodes": 16,
        "negative_nodes": 16,
        "top_r": 8,
    }
    if refinement is not None:
        unknown_settings = set(refinement) - set(refinement_settings)
        if unknown_settings:
            raise ValueError(
                f"Unknown refinement settings: {sorted(unknown_settings)}"
            )
        refinement_settings.update(refinement)

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
        "refinement": refinement_settings,
    }

    runs_dir = Path(runs_dir)
    reports_dir = Path(reports_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_name = "hgsl" if hsl else "hgnn"
    run_name = f"simple_{model_name}_{feature_set}_seed_{seed}"
    checkpoint_path = runs_dir / f"{run_name}.pt"

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log_event(
        "setup",
        f"X={train_graph['features'].shape}, "
        f"H0={train_graph['incidence_matrix'].shape}, "
        f"nnz={train_graph['incidence_matrix'].nnz:,}, "
        f"parameters={parameter_count:,}, "
        f"positive_weight={float(positive_weight):.4f}, {memory_summary(device)}",
    )

    # Train -> validate -> save the best validation-AUC checkpoint.
    history = []
    best_auc = -float("inf")
    best_epoch = 0
    best_validation = None
    stale_epoch_count = 0
    validation_target_count = 0

    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        log_event(
            "train",
            f"epoch {epoch}/{epochs} started; best_auc={best_auc:.4f}, "
            f"stale={stale_epoch_count}/{patience}",
        )
        random_generator = np.random.default_rng(seed * 100_003 + epoch)
        train_loss, loss_components = train_one_epoch(
            model,
            optimizer,
            train_graph,
            node_feature_tensor,
            label_tensor,
            positive_weight,
            random_generator,
            settings,
            device,
        )

        validation_metrics, validation_target_count = evaluate(
            model,
            validation_data,
            seed=seed,
            device=device,
            batch_size=validation_batch_size,
            limit=validation_limit,
            hsl=hsl,
            refinement=refinement_settings,
        )
        history.append({
            "epoch": epoch,
            "loss": train_loss,
            **loss_components,
            "validation": validation_metrics,
        })
        log_event(
            "train",
            f"epoch {epoch} completed in "
            f"{time.perf_counter() - epoch_started_at:.1f}s: "
            f"loss={train_loss:.4f}, val_auc={validation_metrics['auc']:.4f}",
        )

        if validation_metrics["auc"] > best_auc:
            best_auc = validation_metrics["auc"]
            best_epoch = epoch
            best_validation = validation_metrics
            stale_epoch_count = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "validation": validation_metrics,
                "input_dim": train_graph["features"].shape[1],
                "settings": settings,
            }, checkpoint_path)
            log_event(
                "checkpoint",
                f"new best epoch={epoch}, val_auc={best_auc:.4f}; "
                f"saved to {checkpoint_path}",
            )
        else:
            stale_epoch_count += 1
            if stale_epoch_count >= patience:
                log_event(
                    "train",
                    f"early stopping after {stale_epoch_count} stale epochs",
                )
                break

    report = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "validation_targets": validation_target_count,
        "train_nodes": len(train_graph["labels"]),
        "positive_weight": float(positive_weight),
        "history": history,
        "settings": settings,
    }
    report_path = reports_dir / f"{run_name}_train.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_event(
        "train",
        f"run completed in {time.perf_counter() - started_at:.1f}s; "
        f"best_epoch={best_epoch}, report={report_path}",
    )
    return report


# Test is called only after validation has selected the checkpoint.
def test(
    checkpoint,
    *,
    output_dir=config.PROCESSED,
    device_name="auto",
    test_limit=0,
    batch_size=4,
    reports_dir=config.REPORTS,
):
    started_at = time.perf_counter()
    device = resolve_device(device_name)
    log_event("test", f"loading checkpoint {checkpoint} on device={device}")
    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    settings = checkpoint_data["settings"]

    model = HGSLModel(
        checkpoint_data["input_dim"],
        settings["hidden_dim"],
        settings["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint_data["state_dict"])

    test_data = load_evaluation_data(
        output_dir,
        split_name="test",
        feature_set=settings["feature_set"],
    )
    test_metrics, target_count = evaluate(
        model,
        test_data,
        seed=settings["seed"],
        device=device,
        batch_size=batch_size,
        limit=test_limit,
        hsl=settings["hsl"],
        refinement=settings["refinement"],
    )

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_data["epoch"],
        "checkpoint_validation": checkpoint_data["validation"],
        "test_targets": target_count,
        "test_metrics": test_metrics,
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{Path(checkpoint).stem}_test.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_event(
        "test",
        f"completed in {time.perf_counter() - started_at:.1f}s; "
        f"report={report_path}, metrics={test_metrics}",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=config.TRAIN_CLI_DESCRIPTION)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
    parser.add_argument("--seed", type=int, choices=config.SEEDS, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", choices=config.SEEDS)
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
        experiment_seeds = arguments.seeds
        if experiment_seeds is None:
            experiment_seeds = [arguments.seed]

        for experiment_seed in experiment_seeds:
            train_report = train(
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
                    train_report["checkpoint"],
                    output_dir=arguments.output_dir,
                    device_name=arguments.device,
                    test_limit=arguments.test_limit,
                ), flush=True)
