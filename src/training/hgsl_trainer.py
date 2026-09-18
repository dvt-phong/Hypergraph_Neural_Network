"""Phase 8 training protocol for HGSL with inductive local validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.cache import source_signature, utc_now, write_json_atomic
from data.schema import EXPERIMENT_SEEDS
from hypergraph.construction import GRAPH_MANIFEST_ARTIFACT
from hypergraph.io import (
    LocalGraphStore,
    batch_local_hypergraphs,
    load_train_hypergraph,
)
from losses.objective import hgsl_objective, train_pos_weight
from models.hgnn import HypergraphOperator
from models.model import HGSLModel
from models.refinement import RefinementConfig
from paths import PROCESSED_DATA_DIR, REPORTS_DIR, RUNS_DIR, require_project_path
from training.evaluator import binary_metrics
from training.hgsl import hyperedge_metadata
from training.trainer import (
    labels_for_nodes,
    resolve_device,
    set_seed,
    validation_targets,
)


@dataclass(frozen=True)
class HGSLTrainingConfig:
    seed: int = 1
    hidden_dim: int = 64
    dropout: float = 0.5
    sampled_hyperedges: int = 96
    positive_nodes: int = 16
    negative_nodes: int = 16
    mode: str = "top_r"
    top_r: int = 8
    threshold: float = 0.5
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.2
    contrastive_nodes: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    max_gradient_norm: float = 5.0
    epochs: int = 1
    patience: int = 5
    validation_limit: int = 32
    validation_batch_size: int = 4
    device: str = "auto"

    def refinement_config(self) -> RefinementConfig:
        return RefinementConfig(
            sampled_hyperedges=self.sampled_hyperedges,
            positive_nodes=self.positive_nodes,
            negative_nodes=self.negative_nodes,
            mode=self.mode,
            top_r=self.top_r,
            threshold=self.threshold,
        )

    def validate(self) -> None:
        if self.seed not in EXPERIMENT_SEEDS:
            raise ValueError(f"seed must be one of {EXPERIMENT_SEEDS}")
        if self.hidden_dim <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid model dimensions or dropout")
        if self.contrastive_weight < 0 or self.contrastive_temperature <= 0:
            raise ValueError("Invalid contrastive parameters")
        if self.contrastive_nodes < 2:
            raise ValueError("contrastive_nodes must be at least two")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if self.max_gradient_norm <= 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("Training counts and gradient norm must be positive")
        if self.validation_limit < 0 or self.validation_limit == 1:
            raise ValueError("validation_limit must be zero or at least two")
        if self.validation_batch_size <= 0:
            raise ValueError("validation_batch_size must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        self.refinement_config().validate()


def _evaluate_hgsl(
    model: HGSLModel,
    store: LocalGraphStore,
    target_node_ids: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], dict[str, int]]:
    probabilities: list[np.ndarray] = []
    audit_totals = {
        "sampled_hyperedges": 0,
        "membership_candidates": 0,
        "retained_positive_memberships": 0,
        "added_negative_memberships": 0,
        "restored_isolated_nodes": 0,
    }
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
            generator = np.random.default_rng(seed * 1_000_003 + start)
            output = model(
                features,
                operator,
                batch.incidence,
                batch.families,
                batch.sizes,
                generator,
                batch.node_graph_ids,
                batch.edge_graph_ids,
            )
            target_indices = torch.as_tensor(
                batch.target_indices, dtype=torch.int64, device=device
            )
            probabilities.append(
                torch.sigmoid(output.logits[target_indices]).cpu().numpy()
            )
            for key in audit_totals:
                audit_totals[key] += int(output.refinement.audit[key])
    return binary_metrics(labels, np.concatenate(probabilities)), audit_totals


def _checkpoint_atomic(
    path: Path,
    model: HGSLModel,
    optimizer: torch.optim.Optimizer,
    config: HGSLTrainingConfig,
    epoch: int,
    validation_metrics: dict[str, float],
    refinement_audit: dict[str, Any],
    graph_manifest_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    torch.save(
        {
            "phase": 8,
            "epoch": epoch,
            "config": asdict(config),
            "validation_metrics": validation_metrics,
            "refinement_audit": refinement_audit,
            "graph_manifest": source_signature(
                graph_manifest_path, include_hash=True
            ),
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def run_hgsl_training(
    config: HGSLTrainingConfig,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    """Train HGSL and select the checkpoint using validation AUC only."""

    config.validate()
    output_dir = require_project_path(output_dir)
    report_dir = require_project_path(report_dir)
    runs_dir = require_project_path(runs_dir)
    set_seed(config.seed)
    device = resolve_device(config.device)
    graph = load_train_hypergraph(config.seed, output_dir)
    families, sizes = hyperedge_metadata(output_dir, config.seed)
    features = torch.as_tensor(graph.features, dtype=torch.float32, device=device)
    labels_numpy = labels_for_nodes(output_dir / "nodes.parquet", graph.node_ids)
    labels = torch.as_tensor(labels_numpy, dtype=torch.float32, device=device)
    pos_weight = train_pos_weight(labels)
    initial_operator = HypergraphOperator.from_scipy(
        graph.incidence, device=device
    )
    model = HGSLModel(
        input_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        refinement=config.refinement_config(),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    graph_manifest_path = output_dir / GRAPH_MANIFEST_ARTIFACT
    checkpoint_path = runs_dir / f"hgsl_seed_{config.seed}.pt"
    history: list[dict[str, Any]] = []
    best_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    with LocalGraphStore(config.seed, "validation", output_dir) as store:
        validation_node_ids, validation_labels = validation_targets(
            store,
            output_dir / "nodes.parquet",
            limit=config.validation_limit,
            seed=config.seed,
        )
        for epoch in range(1, config.epochs + 1):
            generator = np.random.default_rng(config.seed * 100_003 + epoch)
            contrastive_count = min(
                config.contrastive_nodes, graph.incidence.shape[0]
            )
            contrastive_indices = torch.as_tensor(
                generator.choice(
                    graph.incidence.shape[0],
                    size=contrastive_count,
                    replace=False,
                ),
                dtype=torch.int64,
                device=device,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(
                features,
                initial_operator,
                graph.incidence,
                families,
                sizes,
                generator,
            )
            total_loss, parts = hgsl_objective(
                output.logits,
                labels,
                output.z0,
                output.z_star,
                pos_weight,
                contrastive_weight=config.contrastive_weight,
                temperature=config.contrastive_temperature,
                contrastive_node_indices=contrastive_indices,
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError("HGSL loss is non-finite")
            total_loss.backward()
            unclipped_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_gradient_norm, error_if_nonfinite=True
            )
            scorer_terms = [
                torch.sum(parameter.grad.square())
                for parameter in model.refiner.scorer.parameters()
                if parameter.grad is not None
            ]
            if not scorer_terms:
                raise RuntimeError("No gradient reached the membership scorer")
            scorer_norm = torch.sqrt(torch.stack(scorer_terms).sum())
            if not torch.isfinite(scorer_norm) or scorer_norm <= 0:
                raise RuntimeError("Membership scorer gradient is invalid")
            optimizer.step()

            validation_metrics, validation_refinement = _evaluate_hgsl(
                model,
                store,
                validation_node_ids,
                validation_labels,
                batch_size=config.validation_batch_size,
                device=device,
                seed=config.seed,
            )
            epoch_record = {
                "epoch": epoch,
                "loss": {
                    "total": float(total_loss.detach().cpu()),
                    "bce": float(parts["bce"].detach().cpu()),
                    "contrastive": float(parts["contrastive"].detach().cpu()),
                },
                "gradient_norm_before_clip": float(unclipped_norm.detach().cpu()),
                "scorer_gradient_norm": float(scorer_norm.detach().cpu()),
                "train_refinement": output.refinement.audit,
                "validation_metrics": validation_metrics,
                "validation_refinement": validation_refinement,
            }
            history.append(epoch_record)
            if validation_metrics["auc"] > best_auc:
                best_auc = validation_metrics["auc"]
                best_epoch = epoch
                epochs_without_improvement = 0
                _checkpoint_atomic(
                    checkpoint_path,
                    model,
                    optimizer,
                    config,
                    epoch,
                    validation_metrics,
                    output.refinement.audit,
                    graph_manifest_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

    manifest_signature = source_signature(graph_manifest_path, include_hash=True)
    report = {
        "phase": 8,
        "scope": (
            "HGSL training protocol; sampled validation is not a final experiment"
        ),
        "generated_utc": utc_now(),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "resolved_device": str(device),
        "nodes": int(graph.incidence.shape[0]),
        "hyperedges": int(graph.incidence.shape[1]),
        "pos_weight": float(pos_weight.detach().cpu()),
        "validation_targets": int(validation_node_ids.size),
        "validation_sampling": (
            "all" if config.validation_limit == 0 else "fixed random subset"
        ),
        "selection_metric": "validation_auc",
        "test_split_used": False,
        "local_negative_scope": "same local graph only",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_metrics": history[best_epoch - 1]["validation_metrics"],
        "history": history,
        "checkpoint": str(checkpoint_path),
        "graph_manifest_sha256": manifest_signature["sha256"],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_dir / f"phase8_hgsl_seed_{config.seed}.json", report)
    return report
