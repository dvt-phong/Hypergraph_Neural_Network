"""End-to-end baseline HGNN model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as functional

from models.classifier import DropoutClassifier
from models.hgnn import HGNNLayer, HypergraphOperator
from models.refinement import (
    HypergraphRefiner,
    RefinementConfig,
    RefinementResult,
)


class HGNNBaseline(nn.Module):
    """Two-layer HGNN encoder followed by a binary classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.layer1 = HGNNLayer(input_dim, hidden_dim)
        self.layer2 = HGNNLayer(hidden_dim, hidden_dim)
        self.classifier = DropoutClassifier(hidden_dim)
        self.dropout = dropout

    def encode(
        self, node_features: torch.Tensor, operator: HypergraphOperator
    ) -> torch.Tensor:
        hidden = functional.relu(self.layer1(node_features, operator))
        hidden = functional.dropout(hidden, p=self.dropout, training=self.training)
        return functional.relu(self.layer2(hidden, operator))

    def forward(
        self, node_features: torch.Tensor, operator: HypergraphOperator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(node_features, operator)
        return self.classifier(embeddings), embeddings


@dataclass(frozen=True)
class HGSLForward:
    logits: torch.Tensor
    z0: torch.Tensor
    z_star: torch.Tensor
    refinement: RefinementResult


class HGSLModel(nn.Module):
    """HGNN -> sparse membership refinement -> HGNN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        refinement: RefinementConfig = RefinementConfig(),
    ) -> None:
        super().__init__()
        self.backbone = HGNNBaseline(input_dim, hidden_dim, dropout)
        self.refiner = HypergraphRefiner(hidden_dim, refinement)

    def forward(
        self,
        node_features: torch.Tensor,
        initial_operator: HypergraphOperator,
        initial_incidence: sparse.spmatrix,
        families: np.ndarray,
        sizes: np.ndarray,
        generator: np.random.Generator,
        node_groups: np.ndarray | None = None,
        edge_groups: np.ndarray | None = None,
    ) -> HGSLForward:
        z0 = self.backbone.encode(node_features, initial_operator)
        refinement = self.refiner(
            z0,
            initial_incidence,
            families,
            sizes,
            generator,
            node_groups,
            edge_groups,
        )
        z_star = self.backbone.encode(node_features, refinement.operator)
        logits = self.backbone.classifier(z_star)
        return HGSLForward(logits, z0, z_star, refinement)
