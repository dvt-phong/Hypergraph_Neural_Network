# 8. Train HGSL, select a checkpoint on validation, then evaluate on test.

import argparse
import json
import random
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


# Assign average ranks to tied prediction scores.
def _average_ranks(prediction_scores):
    ascending_order = np.argsort(prediction_scores, kind="mergesort")
    ranks = np.empty(len(prediction_scores), dtype=np.float64)
    sorted_scores = prediction_scores[ascending_order]
    tie_starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    tie_stops = np.r_[tie_starts[1:], len(prediction_scores)]
    for tie_start, tie_stop in zip(tie_starts, tie_stops):
        average_rank = (tie_start + 1 + tie_stop) / 2
        ranks[ascending_order[tie_start:tie_stop]] = average_rank
    return ranks


# Compute ROC AUC from binary labels and prediction ranks.
def _roc_auc(binary_labels, prediction_scores):
    ranks = _average_ranks(prediction_scores)
    positive_count = int(binary_labels.sum())
    negative_count = len(binary_labels) - positive_count
    positive_rank_sum = ranks[binary_labels == 1].sum()
    minimum_positive_rank_sum = positive_count * (positive_count + 1) / 2
    return (
        positive_rank_sum - minimum_positive_rank_sum
    ) / (positive_count * negative_count)


# Compute area under the precision-recall curve from ranked predictions.
def _average_precision(binary_labels, prediction_scores):
    descending_order = np.argsort(-prediction_scores, kind="mergesort")
    ordered_labels = binary_labels[descending_order]
    ordered_scores = prediction_scores[descending_order]
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(ordered_scores) != 0),
        len(prediction_scores) - 1,
    ]
    true_positives = np.cumsum(ordered_labels)[threshold_ends]
    precision_curve = true_positives / (threshold_ends + 1)
    positive_count = int(binary_labels.sum())
    recall_curve = true_positives / positive_count
    recall_steps = np.diff(np.r_[0.0, recall_curve])
    return np.sum(recall_steps * precision_curve)


# Compute precision, recall, and F1 at the fixed probability threshold.
def _threshold_metrics(binary_labels, prediction_scores):
    predicted_positive = prediction_scores >= 0.5
    true_positive_count = int(
        np.count_nonzero(predicted_positive & (binary_labels == 1))
    )
    false_positive_count = int(
        np.count_nonzero(predicted_positive & (binary_labels == 0))
    )
    false_negative_count = int(
        np.count_nonzero(~predicted_positive & (binary_labels == 1))
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
    return precision, recall, f1_score


# Compute binary classification metrics without an extra dependency.
def metrics(labels, scores):
    binary_labels = np.asarray(labels, dtype=np.int8)
    prediction_scores = np.asarray(scores, dtype=np.float64)
    if len(binary_labels) != len(prediction_scores):
        raise ValueError("Labels and predictions have different lengths")
    if len(np.unique(binary_labels)) != 2:
        raise ValueError("Metrics require both label classes")

    auc = _roc_auc(binary_labels, prediction_scores)
    auprc = _average_precision(binary_labels, prediction_scores)
    precision, recall, f1_score = _threshold_metrics(
        binary_labels,
        prediction_scores,
    )
    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "f1": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
    }


# Combine independent local graphs into one block-diagonal evaluation batch.
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
        node_features, initial_incidence_matrix, families, sizes, label = graph
        feature_blocks.append(node_features)
        incidence_blocks.append(initial_incidence_matrix)
        family_blocks.append(families)
        size_blocks.append(sizes)
        target_positions.append(node_offset)
        node_group_blocks.append(
            np.full(len(node_features), graph_id, dtype=np.int64)
        )
        edge_group_blocks.append(
            np.full(initial_incidence_matrix.shape[1], graph_id, dtype=np.int64)
        )
        labels.append(label)
        node_offset += len(node_features)

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


# Select deterministic evaluation targets and preserve both label classes.
def _evaluation_targets(evaluation_data, seed, limit):
    targets = np.arange(len(evaluation_data["nodes"]), dtype=np.int64)

    if not limit:
        return targets
    if limit < 2:
        raise ValueError("Evaluation limit must be zero or at least two")

    negative = []
    positive = []
    for node_id in targets:
        if evaluation_data["nodes"][node_id]["label"] == "1":
            positive.append(node_id)
        else:
            negative.append(node_id)
    if not negative or not positive:
        raise ValueError("Limited evaluation requires both label classes")

    random_generator = np.random.default_rng(seed)
    selected = [
        int(random_generator.choice(negative)),
        int(random_generator.choice(positive)),
    ]
    remaining = np.setdiff1d(targets, selected)
    extra_count = min(limit - 2, len(remaining))
    if extra_count:
        extra = random_generator.choice(remaining, extra_count, replace=False)
        for node_id in extra:
            selected.append(int(node_id))
    return np.sort(np.asarray(selected, dtype=np.int64))


# Evaluate a model on local leakage-free graphs for one dataset split.
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
    targets = _evaluation_targets(evaluation_data, seed, limit)
    probabilities = []
    labels = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            graphs = []
            for target in targets[start:start + batch_size]:
                graphs.append(build_local_graph(evaluation_data, int(target)))

            (
                node_features,
                initial_incidence_matrix,
                families,
                sizes,
                positions,
                node_groups,
                edge_groups,
                batch_labels,
            ) = _batch_graphs(graphs)
            model_output = model(
                torch.as_tensor(node_features, device=device),
                initial_incidence_matrix,
                families,
                sizes,
                np.random.default_rng(seed),
                hsl=hsl,
                deterministic=True,
                refinement=refinement,
                node_groups=node_groups,
                edge_groups=edge_groups,
            )
            batch_probabilities = torch.sigmoid(
                model_output["logits"][positions]
            )
            probabilities.extend(batch_probabilities.cpu().tolist())
            labels.extend(batch_labels.tolist())
    return metrics(labels, probabilities), len(targets)


# Seed Python, NumPy, and PyTorch random number generators.
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Resolve the requested training device, including automatic CUDA selection.
def _device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


# Verify that finite nonzero gradients reach the HSL membership scorer.
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


# Merge caller overrides into the default HSL refinement settings.
def _refinement_settings(overrides):
    settings = {
        "sampled_hyperedges": 96,
        "positive_nodes": 16,
        "negative_nodes": 16,
        "top_r": 8,
        "threshold": None,
    }
    if overrides is not None:
        settings.update(overrides)
    return settings


# Return the stable model name used in checkpoint and report filenames.
def _model_variant_name(hsl_enabled):
    if hsl_enabled:
        return "hgsl"
    return "hgnn"


# Train the model for one full-batch epoch and return detached loss details.
def _train_one_epoch(
    model,
    optimizer,
    node_feature_tensor,
    initial_incidence_matrix,
    edge_families,
    edge_sizes,
    label_tensor,
    positive_weight,
    random_generator,
    contrastive_nodes,
    hsl_enabled,
    refinement_settings,
    contrastive_weight,
    temperature,
    device,
):
    sampled_node_count = min(contrastive_nodes, len(label_tensor))
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
    model_output = model(
        node_feature_tensor,
        initial_incidence_matrix,
        edge_families,
        edge_sizes,
        random_generator,
        hsl=hsl_enabled,
        refinement=refinement_settings,
    )
    active_contrastive_weight = 0.0
    if hsl_enabled:
        active_contrastive_weight = contrastive_weight
    loss, loss_components = total_loss(
        model_output,
        label_tensor,
        positive_weight,
        sampled_node_ids,
        lambda_cl=active_contrastive_weight,
        temperature=temperature,
    )
    if not torch.isfinite(loss):
        raise ValueError("Training loss is not finite")

    loss.backward()
    if hsl_enabled:
        _check_hsl_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss, loss_components


# Save one validation-selected checkpoint without changing its file schema.
def _save_checkpoint(
    checkpoint_path,
    model,
    epoch,
    validation_metrics,
    input_dimension,
    settings,
):
    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "validation": validation_metrics,
        "input_dim": input_dimension,
        "settings": settings,
    }, checkpoint_path)


# Collect serializable experiment settings for checkpoints and reports.
def _build_training_settings(
    seed,
    feature_set,
    epochs,
    patience,
    validation_limit,
    validation_batch_size,
    hidden_dim,
    dropout,
    learning_rate,
    weight_decay,
    lambda_cl,
    temperature,
    contrastive_nodes,
    hsl_enabled,
    refinement_settings,
):
    return {
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
        "hsl": hsl_enabled,
        "refinement": refinement_settings,
    }


# Run training epochs, validate each epoch, and save every new best checkpoint.
def _run_training_epochs(
    *,
    model,
    optimizer,
    train_node_features,
    node_feature_tensor,
    initial_incidence_matrix,
    edge_families,
    edge_sizes,
    label_tensor,
    positive_weight,
    evaluation_data,
    checkpoint_path,
    settings,
    device,
):
    history = []
    best_auc = -float("inf")
    best_epoch = 0
    stale_epoch_count = 0
    validation_target_count = 0

    for epoch in range(1, settings["epochs"] + 1):
        random_generator = np.random.default_rng(
            settings["seed"] * 100_003 + epoch
        )
        loss, loss_components = _train_one_epoch(
            model,
            optimizer,
            node_feature_tensor,
            initial_incidence_matrix,
            edge_families,
            edge_sizes,
            label_tensor,
            positive_weight,
            random_generator,
            settings["contrastive_nodes"],
            settings["hsl"],
            settings["refinement"],
            settings["lambda_cl"],
            settings["temperature"],
            device,
        )
        validation_metrics, validation_target_count = evaluate(
            model,
            evaluation_data,
            seed=settings["seed"],
            device=device,
            batch_size=settings["validation_batch_size"],
            limit=settings["validation_limit"],
            hsl=settings["hsl"],
            refinement=settings["refinement"],
        )
        epoch_record = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            **loss_components,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch}: loss={float(loss.detach()):.4f} "
            f"val_auc={validation_metrics['auc']:.4f}",
            flush=True,
        )

        if validation_metrics["auc"] > best_auc:
            best_auc = validation_metrics["auc"]
            best_epoch = epoch
            stale_epoch_count = 0
            _save_checkpoint(
                checkpoint_path,
                model,
                epoch,
                validation_metrics,
                train_node_features.shape[1],
                settings,
            )
        else:
            stale_epoch_count += 1
            if stale_epoch_count >= settings["patience"]:
                break

    return history, best_epoch, validation_target_count


# Load graph data and initialize tensors, model, optimizer, and class weight.
def _initialize_training_components(
    output_dir,
    seed,
    feature_set,
    hidden_dim,
    dropout,
    learning_rate,
    weight_decay,
    device,
):
    (
        train_node_features,
        initial_incidence_matrix,
        train_labels,
        edge_families,
        edge_sizes,
    ) = load_train_graph(output_dir, feature_set=feature_set)
    evaluation_data = load_evaluation_data(
        output_dir,
        split_name="validation",
        feature_set=feature_set,
    )
    node_feature_tensor = torch.as_tensor(train_node_features, device=device)
    label_tensor = torch.as_tensor(train_labels, device=device)
    positive_weight = train_pos_weight(label_tensor)
    model = HGSLModel(train_node_features.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return (
        train_node_features,
        initial_incidence_matrix,
        train_labels,
        edge_families,
        edge_sizes,
        evaluation_data,
        node_feature_tensor,
        label_tensor,
        positive_weight,
        model,
        optimizer,
    )


# Create output directories and resolve the stable run and checkpoint names.
def _prepare_training_paths(runs_dir, reports_dir, hsl_enabled, feature_set, seed):
    runs_dir = Path(runs_dir)
    reports_dir = Path(reports_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_name = _model_variant_name(hsl_enabled)
    run_name = f"simple_{model_name}_{feature_set}_seed_{seed}"
    checkpoint_path = runs_dir / f"{run_name}.pt"
    return reports_dir, run_name, checkpoint_path


# Build and save the final training report without changing its schema.
def _write_training_report(
    reports_dir,
    run_name,
    checkpoint_path,
    best_epoch,
    validation_target_count,
    train_node_count,
    positive_weight,
    history,
    settings,
):
    report = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "validation_targets": validation_target_count,
        "train_nodes": train_node_count,
        "positive_weight": float(positive_weight),
        "history": history,
        "settings": settings,
    }
    report_path = reports_dir / f"{run_name}_train.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved validation-selected checkpoint: {checkpoint_path}", flush=True)
    return report


# Run full-batch training and select the best validation-AUC checkpoint.
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
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")

    output_dir = Path(output_dir)
    device = _device(device_name)
    _set_seed(seed)
    (
        train_node_features,
        initial_incidence_matrix,
        train_labels,
        edge_families,
        edge_sizes,
        evaluation_data,
        node_feature_tensor,
        label_tensor,
        positive_weight,
        model,
        optimizer,
    ) = _initialize_training_components(
        output_dir,
        seed,
        feature_set,
        hidden_dim,
        dropout,
        learning_rate,
        weight_decay,
        device,
    )
    refinement = _refinement_settings(refinement)

    settings = _build_training_settings(
        seed,
        feature_set,
        epochs,
        patience,
        validation_limit,
        validation_batch_size,
        hidden_dim,
        dropout,
        learning_rate,
        weight_decay,
        lambda_cl,
        temperature,
        contrastive_nodes,
        hsl,
        refinement,
    )
    reports_dir, run_name, checkpoint = _prepare_training_paths(
        runs_dir,
        reports_dir,
        hsl,
        feature_set,
        seed,
    )

    history, best_epoch, validation_count = _run_training_epochs(
        model=model,
        optimizer=optimizer,
        train_node_features=train_node_features,
        node_feature_tensor=node_feature_tensor,
        initial_incidence_matrix=initial_incidence_matrix,
        edge_families=edge_families,
        edge_sizes=edge_sizes,
        label_tensor=label_tensor,
        positive_weight=positive_weight,
        evaluation_data=evaluation_data,
        checkpoint_path=checkpoint,
        settings=settings,
        device=device,
    )

    return _write_training_report(
        reports_dir,
        run_name,
        checkpoint,
        best_epoch,
        validation_count,
        len(train_labels),
        positive_weight,
        history,
        settings,
    )


# Evaluate a validation-selected checkpoint on the held-out test split.
def test(
    checkpoint,
    *,
    output_dir=config.PROCESSED,
    device_name="auto",
    test_limit=0,
    batch_size=4,
    reports_dir=config.REPORTS,
):
    device = _device(device_name)
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

    evaluation_data = load_evaluation_data(
        output_dir,
        split_name="test",
        feature_set=settings["feature_set"],
    )
    test_metrics, target_count = evaluate(
        model,
        evaluation_data,
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
        seeds = arguments.seeds
        if seeds is None:
            seeds = [arguments.seed]
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
