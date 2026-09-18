"""Contrastive alignment between initial and refined node embeddings."""
from __future__ import annotations

import torch
from torch.nn import functional as functional


def contrastive_alignment_loss(
    z0: torch.Tensor,
    z_star: torch.Tensor,
    *,
    temperature: float = 0.2,
    node_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric cross-view InfoNCE without an all-node similarity matrix."""

    if z0.shape != z_star.shape or z0.ndim != 2:
        raise ValueError("z0 and z_star must be equal-shape 2-D tensors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if node_indices is not None:
        z0 = z0[node_indices]
        z_star = z_star[node_indices]
    if z0.shape[0] < 2:
        raise ValueError("At least two nodes are required for contrastive loss")
    view0 = functional.normalize(z0, dim=1)
    view_star = functional.normalize(z_star, dim=1)
    similarities = view0 @ view_star.T / temperature
    targets = torch.arange(z0.shape[0], device=z0.device)
    return 0.5 * (
        functional.cross_entropy(similarities, targets)
        + functional.cross_entropy(similarities.T, targets)
    )
