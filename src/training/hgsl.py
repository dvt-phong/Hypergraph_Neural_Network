"""End-to-end Phase 7 smoke run for sparse hypergraph structure learning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

from data.cache import sql_path, utc_now, write_json_atomic
from data.schema import EXPERIMENT_SEEDS
from hypergraph.construction import HYPEREDGE_METADATA_ARTIFACT
from hypergraph.io import load_train_hypergraph
from losses.objective import hgsl_objective, train_pos_weight
from models.hgnn import HypergraphOperator
from models.model import HGSLModel
from models.refinement import RefinementConfig
from paths import PROCESSED_DATA_DIR, REPORTS_DIR, require_project_path
from training.trainer import labels_for_nodes, resolve_device, set_seed


@dataclass(frozen=True)
class HGSLSmokeConfig:
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
    device: str = "auto"

    def validate(self) -> None:
        if self.seed not in EXPERIMENT_SEEDS:
            raise ValueError(f"seed must be one of {EXPERIMENT_SEEDS}")
        if self.hidden_dim <= 0 or self.contrastive_nodes < 2:
            raise ValueError("hidden_dim must be positive and contrastive_nodes >= 2")
        if self.contrastive_weight < 0 or self.contrastive_temperature <= 0:
            raise ValueError("Invalid contrastive parameters")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        self.refinement_config().validate()

    def refinement_config(self) -> RefinementConfig:
        return RefinementConfig(
            sampled_hyperedges=self.sampled_hyperedges,
            positive_nodes=self.positive_nodes,
            negative_nodes=self.negative_nodes,
            mode=self.mode,
            top_r=self.top_r,
            threshold=self.threshold,
        )


def _hyperedge_metadata(
    output_dir: Path, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""SELECT family, size
                FROM read_parquet(
                    '{sql_path(output_dir / HYPEREDGE_METADATA_ARTIFACT)}'
                )
                WHERE seed=? ORDER BY hyperedge_id""",
            [seed],
        ).fetchnumpy()
    finally:
        connection.close()
    return rows["family"].astype(str), rows["size"].astype(np.int64)


def run_hgsl_smoke(
    config: HGSLSmokeConfig,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Run one optimizer step through H0 -> Z0 -> H* -> Z*."""

    config.validate()
    output_dir = require_project_path(output_dir)
    report_dir = require_project_path(report_dir)
    set_seed(config.seed)
    device = resolve_device(config.device)
    graph = load_train_hypergraph(config.seed, output_dir)
    families, sizes = _hyperedge_metadata(output_dir, config.seed)
    if families.size != graph.incidence.shape[1]:
        raise RuntimeError("Hyperedge metadata does not match H0 columns")

    features = torch.as_tensor(graph.features, dtype=torch.float32, device=device)
    labels_numpy = labels_for_nodes(output_dir / "nodes.parquet", graph.node_ids)
    labels = torch.as_tensor(labels_numpy, dtype=torch.float32, device=device)
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
    numpy_generator = np.random.default_rng(config.seed)
    contrastive_count = min(config.contrastive_nodes, graph.incidence.shape[0])
    contrastive_indices = torch.as_tensor(
        numpy_generator.choice(
            graph.incidence.shape[0], size=contrastive_count, replace=False
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
        numpy_generator,
    )
    total_loss, parts = hgsl_objective(
        output.logits,
        labels,
        output.z0,
        output.z_star,
        train_pos_weight(labels),
        contrastive_weight=config.contrastive_weight,
        temperature=config.contrastive_temperature,
        contrastive_node_indices=contrastive_indices,
    )
    if not torch.isfinite(total_loss):
        raise RuntimeError("HGSL loss is non-finite")
    total_loss.backward()
    squared_norm = torch.zeros((), dtype=torch.float32, device=device)
    scorer_squared_norm = torch.zeros((), dtype=torch.float32, device=device)
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
        norm = torch.sum(parameter.grad.square())
        squared_norm += norm
        if name.startswith("refiner.scorer"):
            scorer_squared_norm += norm
    gradient_norm = torch.sqrt(squared_norm)
    scorer_gradient_norm = torch.sqrt(scorer_squared_norm)
    if scorer_gradient_norm <= 0:
        raise RuntimeError("No gradient reached the membership scorer")
    optimizer.step()

    report = {
        "phase": 7,
        "scope": "one-step HGSL smoke run; hyperparameters are not selected",
        "generated_utc": utc_now(),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "resolved_device": str(device),
        "nodes": int(graph.incidence.shape[0]),
        "hyperedges": int(graph.incidence.shape[1]),
        "loss": {
            "total": float(total_loss.detach().cpu()),
            "bce": float(parts["bce"].detach().cpu()),
            "contrastive": float(parts["contrastive"].detach().cpu()),
        },
        "gradient_finite": True,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "scorer_gradient_norm": float(scorer_gradient_norm.detach().cpu()),
        "z0_shape": list(output.z0.shape),
        "z_star_shape": list(output.z_star.shape),
        "refinement": output.refinement.audit,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_dir / f"phase7_hgsl_seed_{config.seed}.json", report)
    return report
