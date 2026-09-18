"""Train-only class weighting and binary classification loss."""
from __future__ import annotations

import torch
from torch.nn import functional as functional


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
