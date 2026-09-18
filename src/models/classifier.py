"""Binary node classifier used by the HGNN baseline."""
from __future__ import annotations

import torch
from torch import nn


class DropoutClassifier(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.output(embeddings).squeeze(-1)
