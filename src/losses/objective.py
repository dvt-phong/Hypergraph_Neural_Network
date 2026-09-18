"""Train-only class weighting and binary classification loss."""
from __future__ import annotations

import torch
from torch.nn import functional as functional

from losses.contrastive import contrastive_alignment_loss


def train_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """Return negative/positive count using only the supplied train labels."""

    labels = labels.float().reshape(-1)
    if labels.numel() == 0 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be a non-empty binary tensor")
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Both label classes are required")
    return negatives / positives


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    logits = logits.reshape(-1)
    labels = labels.float().reshape(-1)
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape")
    return functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight
    )


def hgsl_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    z0: torch.Tensor,
    z_star: torch.Tensor,
    pos_weight: torch.Tensor,
    *,
    contrastive_weight: float,
    temperature: float = 0.2,
    contrastive_node_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return BCE + lambda*InfoNCE and its detached-friendly components."""

    if contrastive_weight < 0:
        raise ValueError("contrastive_weight must be non-negative")
    bce = classification_loss(logits, labels, pos_weight)
    if contrastive_weight == 0:
        contrastive = bce.new_zeros(())
        total = bce
    else:
        contrastive = contrastive_alignment_loss(
            z0,
            z_star,
            temperature=temperature,
            node_indices=contrastive_node_indices,
        )
        total = bce + contrastive_weight * contrastive
    return total, {"bce": bce, "contrastive": contrastive}
