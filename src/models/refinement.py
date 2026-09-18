"""Differentiable sparse refinement of memberships in existing hyperedges."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from scipy import sparse
import torch
from torch import nn

from models.hgnn import HypergraphOperator
from models.hyperedge_sampling import (
    sample_balanced_hyperedges,
    sampled_strata_audit,
)
from models.incident_node_sampling import sample_incident_nodes


@dataclass(frozen=True)
class RefinementConfig:
    sampled_hyperedges: int = 96
    positive_nodes: int = 16
    negative_nodes: int = 16
    mode: str = "top_r"
    top_r: int = 8
    threshold: float = 0.5

    def validate(self) -> None:
        if self.sampled_hyperedges <= 0 or self.positive_nodes <= 0:
            raise ValueError("Sampling counts must be positive")
        if self.negative_nodes < 0 or self.top_r <= 0:
            raise ValueError("negative_nodes must be non-negative and top_r positive")
        if self.mode not in {"top_r", "threshold"}:
            raise ValueError("mode must be top_r or threshold")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")


@dataclass(frozen=True)
class RefinementResult:
    operator: HypergraphOperator
    incidence: torch.Tensor
    sampled_edge_ids: np.ndarray
    audit: dict[str, Any]


class MembershipScorer(nn.Module):
    """Bilinear node-edge compatibility scorer."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.node_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = sqrt(embedding_dim)

    def forward(
        self, node_embeddings: torch.Tensor, edge_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if node_embeddings.shape != edge_embeddings.shape:
            raise ValueError("Node and edge embedding batches must have equal shape")
        node_projection = self.node_projection(node_embeddings)
        edge_projection = self.edge_projection(edge_embeddings)
        return (node_projection * edge_projection).sum(dim=1) / self.scale + self.bias


class HypergraphRefiner(nn.Module):
    """Sample candidate memberships and create a sparse weighted H*."""

    def __init__(self, embedding_dim: int, config: RefinementConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.scorer = MembershipScorer(embedding_dim)

    def _select(
        self, logits: torch.Tensor, positive_count: int
    ) -> torch.Tensor:
        if self.config.mode == "top_r":
            count = min(self.config.top_r, logits.numel())
            selected = torch.topk(logits, count).indices
        else:
            selected = torch.flatnonzero(
                torch.sigmoid(logits) >= self.config.threshold
            )
            if selected.numel() == 0:
                selected = torch.argmax(logits).reshape(1)
        if not torch.any(selected < positive_count):
            best_positive = torch.argmax(logits[:positive_count])
            if self.config.mode == "top_r" and selected.numel() >= self.config.top_r:
                selected_logits = logits[selected]
                selected = selected.clone()
                selected[torch.argmin(selected_logits)] = best_positive
            else:
                selected = torch.cat((selected, best_positive.reshape(1)))
        return torch.unique(selected)

    def forward(
        self,
        embeddings: torch.Tensor,
        incidence: sparse.spmatrix,
        families: np.ndarray,
        sizes: np.ndarray,
        generator: np.random.Generator,
    ) -> RefinementResult:
        matrix = incidence.tocsr()
        families = np.asarray(families).astype(str).reshape(-1)
        sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
        if embeddings.ndim != 2 or embeddings.shape[0] != matrix.shape[0]:
            raise ValueError("Embedding rows must match incidence rows")
        if families.size != matrix.shape[1] or sizes.size != matrix.shape[1]:
            raise ValueError("Hyperedge metadata must match incidence columns")

        sampled_edges = sample_balanced_hyperedges(
            families, sizes, self.config.sampled_hyperedges, generator
        )
        sampled = sample_incident_nodes(
            matrix,
            sampled_edges,
            positive_count=self.config.positive_nodes,
            negative_count=self.config.negative_nodes,
            generator=generator,
        )
        positive_tensors = [
            torch.as_tensor(nodes, dtype=torch.int64, device=embeddings.device)
            for nodes in sampled.positive_nodes
        ]
        edge_embeddings = torch.stack(
            [embeddings[nodes].mean(dim=0) for nodes in positive_tensors]
        )

        selected_nodes: list[np.ndarray] = []
        selected_edges: list[np.ndarray] = []
        selected_values: list[torch.Tensor] = []
        retained_positive = 0
        added_negative = 0
        candidate_count = 0
        for position, edge_id in enumerate(sampled.edge_ids):
            positive = sampled.positive_nodes[position]
            negative = sampled.negative_nodes[position]
            candidates = np.concatenate((positive, negative))
            candidate_count += int(candidates.size)
            candidate_tensor = torch.as_tensor(
                candidates, dtype=torch.int64, device=embeddings.device
            )
            repeated_edge = edge_embeddings[position].expand(candidates.size, -1)
            logits = self.scorer(embeddings[candidate_tensor], repeated_edge)
            selected = self._select(logits, positive.size)
            selected_numpy = selected.detach().cpu().numpy()
            selected_nodes.append(candidates[selected_numpy])
            selected_edges.append(
                np.full(selected.numel(), edge_id, dtype=np.int64)
            )
            selected_values.extend(torch.sigmoid(logits[selected]).unbind())
            retained_positive += int(np.count_nonzero(selected_numpy < positive.size))
            added_negative += int(np.count_nonzero(selected_numpy >= positive.size))

        coo = matrix.tocoo(copy=False)
        keep = ~np.isin(coo.col, sampled_edges)
        rows = [coo.row[keep].astype(np.int64, copy=False), *selected_nodes]
        columns = [coo.col[keep].astype(np.int64, copy=False), *selected_edges]
        combined_rows = np.concatenate(rows)
        combined_columns = np.concatenate(columns)

        row_counts = np.bincount(combined_rows, minlength=matrix.shape[0])
        missing_nodes = np.flatnonzero(row_counts == 0)
        fallback_edges = np.empty(missing_nodes.size, dtype=np.int64)
        for index, node in enumerate(missing_nodes):
            start = matrix.indptr[node]
            fallback_edges[index] = matrix.indices[start]
        if missing_nodes.size:
            combined_rows = np.concatenate((combined_rows, missing_nodes))
            combined_columns = np.concatenate((combined_columns, fallback_edges))

        constant_count = int(np.count_nonzero(keep))
        device = embeddings.device
        values = torch.cat(
            (
                torch.ones(constant_count, dtype=embeddings.dtype, device=device),
                torch.stack(selected_values),
                torch.ones(missing_nodes.size, dtype=embeddings.dtype, device=device),
            )
        )
        indices = torch.as_tensor(
            np.vstack((combined_rows, combined_columns)),
            dtype=torch.int64,
            device=device,
        )
        refined_incidence = torch.sparse_coo_tensor(
            indices,
            values,
            size=matrix.shape,
            dtype=embeddings.dtype,
            device=device,
            check_invariants=False,
        ).coalesce()
        operator = HypergraphOperator.from_torch_sparse(refined_incidence)
        audit = {
            "sampled_hyperedges": int(sampled_edges.size),
            "sampled_strata": sampled_strata_audit(
                sampled_edges, families, sizes
            ),
            "membership_candidates": candidate_count,
            "retained_positive_memberships": retained_positive,
            "added_negative_memberships": added_negative,
            "restored_isolated_nodes": int(missing_nodes.size),
            "initial_incidences": int(matrix.nnz),
            "refined_incidences": int(refined_incidence._nnz()),
            "mean_learned_membership": float(
                torch.stack(selected_values).mean().detach().cpu()
            ),
            "mode": self.config.mode,
            "top_r": self.config.top_r,
            "threshold": self.config.threshold,
        }
        return RefinementResult(operator, refined_incidence, sampled_edges, audit)
