"""Phase 6 full-batch smoke training for the HGNN baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import duckdb
import numpy as np
import torch

from data.cache import source_signature, sql_path, utc_now, write_json_atomic
from data.schema import EXPERIMENT_SEEDS
from hypergraph.construction import GRAPH_MANIFEST_ARTIFACT
from hypergraph.io import (
    LocalGraphStore,
    batch_local_hypergraphs,
    load_train_hypergraph,
)
from losses.objective import classification_loss, train_pos_weight
from models.hgnn import HypergraphOperator
from models.model import HGNNBaseline
from paths import PROCESSED_DATA_DIR, REPORTS_DIR, RUNS_DIR, require_project_path
from training.evaluator import binary_metrics


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 1
    hidden_dim: int = 64
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 1
    device: str = "auto"
    validation_limit: int = 128
    validation_batch_size: int = 16
    patience: int = 5

    def validate(self) -> None:
        if self.seed not in EXPERIMENT_SEEDS:
            raise ValueError(f"seed must be one of {EXPERIMENT_SEEDS}")
        if self.hidden_dim <= 0 or self.epochs <= 0:
            raise ValueError("hidden_dim and epochs must be positive")
        if self.validation_limit < 0 or self.validation_limit == 1:
            raise ValueError("validation_limit must be zero or at least two")
        if self.validation_batch_size <= 0 or self.patience <= 0:
            raise ValueError("validation_batch_size and patience must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_for_nodes(nodes_path: Path, node_ids: np.ndarray) -> np.ndarray:
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""SELECT node_id, label
                FROM read_parquet('{sql_path(nodes_path)}') ORDER BY node_id"""
        ).fetchnumpy()
    finally:
        connection.close()
    all_labels = np.empty(rows["node_id"].size, dtype=np.float32)
    all_labels[rows["node_id"].astype(np.int64, copy=False)] = rows["label"]
    labels = all_labels[node_ids]
    if not np.all((labels == 0) | (labels == 1)):
        raise RuntimeError("Loaded labels are not binary")
    return labels


def validation_targets(
    store: LocalGraphStore,
    nodes_path: Path,
    *,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    targets = store.target_node_ids()
    labels = labels_for_nodes(nodes_path, targets)
    if limit and limit < targets.size:
        generator = np.random.default_rng(seed)
        selected = np.sort(generator.choice(targets.size, size=limit, replace=False))
        if np.unique(labels[selected]).size < 2:
            missing_class = 1 - int(labels[selected[0]])
            replacement = np.flatnonzero(labels == missing_class)
            if replacement.size == 0:
                raise RuntimeError("Validation split contains only one label class")
            selected[-1] = replacement[0]
            selected.sort()
        targets = targets[selected]
        labels = labels[selected]
    return targets, labels


def _evaluate_local(
    model: HGNNBaseline,
    store: LocalGraphStore,
    target_node_ids: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, target_node_ids.size, batch_size):
            batch_ids = target_node_ids[start : start + batch_size]
            batch = batch_local_hypergraphs(store.load_many(batch_ids))
            features = torch.as_tensor(
                batch.features, dtype=torch.float32, device=device
            )
            operator = HypergraphOperator.from_scipy(
                batch.incidence, device=device
            )
            logits, _ = model(features, operator)
            target_indices = torch.as_tensor(
                batch.target_indices, dtype=torch.int64, device=device
            )
            probabilities.append(
                torch.sigmoid(logits[target_indices]).cpu().numpy()
            )
    return binary_metrics(labels, np.concatenate(probabilities))


def _write_checkpoint_atomic(
    path: Path,
    model: HGNNBaseline,
    config: BaselineConfig,
    epoch: int,
    validation_metrics: dict[str, float],
    graph_manifest_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    state = {
        "phase": 6,
        "epoch": epoch,
        "config": asdict(config),
        "validation_metrics": validation_metrics,
        "graph_manifest": source_signature(
            graph_manifest_path, include_hash=True
        ),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    torch.save(state, temporary)
    temporary.replace(path)


def run_baseline_smoke(
    config: BaselineConfig,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    """Run a small number of full-batch epochs to validate the Phase 6 stack."""

    config.validate()
    output_dir = require_project_path(output_dir)
    report_dir = require_project_path(report_dir)
    runs_dir = require_project_path(runs_dir)
    set_seed(config.seed)
    device = resolve_device(config.device)
    graph = load_train_hypergraph(config.seed, output_dir)
    labels_numpy = labels_for_nodes(output_dir / "nodes.parquet", graph.node_ids)
    features = torch.as_tensor(graph.features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_numpy, dtype=torch.float32, device=device)
    operator = HypergraphOperator.from_scipy(graph.incidence, device=device)
    model = HGNNBaseline(
        input_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    pos_weight = train_pos_weight(labels)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    losses: list[float] = []
    gradient_norms: list[float] = []
    validation_history: list[dict[str, Any]] = []
    best_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = runs_dir / f"hgnn_baseline_seed_{config.seed}.pt"
    graph_manifest_path = output_dir / GRAPH_MANIFEST_ARTIFACT
    with LocalGraphStore(config.seed, "validation", output_dir) as validation_store:
        validation_node_ids, validation_labels = validation_targets(
            validation_store,
            output_dir / "nodes.parquet",
            limit=config.validation_limit,
            seed=config.seed,
        )
        for epoch in range(1, config.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features, operator)
            loss = classification_loss(logits, labels, pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("HGNN loss is non-finite")
            loss.backward()
            squared_norm = torch.zeros((), dtype=torch.float32, device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("HGNN gradient is non-finite")
                    squared_norm += torch.sum(parameter.grad.square())
            gradient_norm = torch.sqrt(squared_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))

            validation_metrics = _evaluate_local(
                model,
                validation_store,
                validation_node_ids,
                validation_labels,
                batch_size=config.validation_batch_size,
                device=device,
            )
            validation_history.append(
                {"epoch": epoch, **validation_metrics}
            )
            if validation_metrics["auc"] > best_auc:
                best_auc = validation_metrics["auc"]
                best_epoch = epoch
                epochs_without_improvement = 0
                _write_checkpoint_atomic(
                    checkpoint_path,
                    model,
                    config,
                    epoch,
                    validation_metrics,
                    graph_manifest_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(features, operator)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    metrics = binary_metrics(labels_numpy, probabilities)
    report = {
        "phase": 6,
        "scope": (
            "Phase 6 smoke run; validation may be sampled and is not a final result"
        ),
        "generated_utc": utc_now(),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "resolved_device": str(device),
        "nodes": int(graph.incidence.shape[0]),
        "hyperedges": int(graph.incidence.shape[1]),
        "incidences": int(graph.incidence.nnz),
        "feature_dim": int(features.shape[1]),
        "embedding_dim": int(embeddings.shape[1]),
        "positive_labels": int(labels_numpy.sum()),
        "negative_labels": int(labels_numpy.size - labels_numpy.sum()),
        "pos_weight": float(pos_weight.detach().cpu()),
        "loss_by_epoch": losses,
        "gradient_norm_by_epoch": gradient_norms,
        "train_metrics": metrics,
        "validation_targets": int(validation_node_ids.size),
        "validation_sampling": (
            "all" if config.validation_limit == 0 else "fixed random subset"
        ),
        "validation_history": validation_history,
        "best_epoch": best_epoch,
        "best_validation_metrics": validation_history[best_epoch - 1],
        "checkpoint": str(checkpoint_path),
        "checkpoint_graph_manifest_sha256": source_signature(
            graph_manifest_path, include_hash=True
        )["sha256"],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        report_dir / f"phase6_hgnn_seed_{config.seed}.json", report
    )
    return report
