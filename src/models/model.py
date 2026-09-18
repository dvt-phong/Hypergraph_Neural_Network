"""End-to-end baseline HGNN model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from models.classifier import DropoutClassifier
from models.hgnn import HGNNLayer, HypergraphOperator


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
