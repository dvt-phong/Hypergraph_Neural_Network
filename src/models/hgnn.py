"""Sparse HGNN propagation without materializing a dense graph matrix."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
import torch
from torch import nn


def scipy_to_torch_sparse(
    matrix: sparse.spmatrix,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a coalesced PyTorch COO tensor."""

    coo = matrix.tocoo(copy=False)
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        dtype=torch.float32,
        device=device,
        check_invariants=False,
    ).coalesce()


@dataclass(frozen=True)
class HypergraphOperator:
    """Tensors for Dv^-1/2 H W De^-1 H^T Dv^-1/2 propagation."""

    incidence: torch.Tensor
    incidence_t: torch.Tensor
    node_inverse_sqrt_degree: torch.Tensor
    edge_weight_over_degree: torch.Tensor

    @classmethod
    def from_scipy(
        cls,
        incidence: sparse.spmatrix,
        *,
        device: torch.device,
        edge_weights: np.ndarray | None = None,
    ) -> "HypergraphOperator":
        matrix = incidence.tocsr().astype(np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("incidence must be a non-empty 2-D sparse matrix")
        if matrix.nnz == 0 or not np.all(matrix.data == 1):
            raise ValueError("incidence must be binary with at least one incidence")
        if edge_weights is None:
            weights = np.ones(matrix.shape[1], dtype=np.float32)
        else:
            weights = np.asarray(edge_weights, dtype=np.float32)
        if weights.shape != (matrix.shape[1],) or np.any(weights <= 0):
            raise ValueError("edge_weights must be positive and match edge count")

        edge_degree = np.asarray(matrix.sum(axis=0)).reshape(-1)
        node_degree = np.asarray(matrix @ weights).reshape(-1)
        if np.any(edge_degree <= 0) or np.any(node_degree <= 0):
            raise ValueError("incidence cannot contain empty nodes or hyperedges")
        torch_incidence = scipy_to_torch_sparse(matrix, device=device)
        return cls(
            incidence=torch_incidence,
            incidence_t=torch_incidence.transpose(0, 1).coalesce(),
            node_inverse_sqrt_degree=torch.as_tensor(
                np.power(node_degree, -0.5), dtype=torch.float32, device=device
            ),
            edge_weight_over_degree=torch.as_tensor(
                weights / edge_degree, dtype=torch.float32, device=device
            ),
        )

    def propagate(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("node_features must be a 2-D tensor")
        if node_features.shape[0] != self.incidence.shape[0]:
            raise ValueError("Feature rows must match incidence rows")
        scaled_nodes = node_features * self.node_inverse_sqrt_degree[:, None]
        edge_messages = torch.sparse.mm(self.incidence_t, scaled_nodes)
        edge_messages = edge_messages * self.edge_weight_over_degree[:, None]
        node_messages = torch.sparse.mm(self.incidence, edge_messages)
        return node_messages * self.node_inverse_sqrt_degree[:, None]


class HGNNLayer(nn.Module):
    """Linear projection followed by normalized hypergraph propagation."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self, node_features: torch.Tensor, operator: HypergraphOperator
    ) -> torch.Tensor:
        return operator.propagate(self.linear(node_features)) + self.bias
