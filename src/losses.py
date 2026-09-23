"""Weighted BCE, symmetric contrastive loss, and their sum."""

import torch
from torch.nn import functional as F


def train_pos_weight(labels):
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Both classes are required in the train split")
    return negatives / positives


def contrastive_loss(z0, z_star, indices, temperature=0.2):
    if temperature <= 0 or len(indices) < 2:
        raise ValueError("Invalid contrastive temperature or node sample")
    left = F.normalize(z0[indices], dim=1)
    right = F.normalize(z_star[indices], dim=1)
    similarity = left @ right.T / temperature
    target = torch.arange(len(indices), device=z0.device)
    return (F.cross_entropy(similarity, target) +
            F.cross_entropy(similarity.T, target)) / 2


def total_loss(output, labels, positive_weight, indices, *, lambda_cl=0.1,
               temperature=0.2):
    bce = F.binary_cross_entropy_with_logits(output["logits"], labels,
                                             pos_weight=positive_weight)
    if lambda_cl > 0:
        cl = contrastive_loss(output["z0"], output["z_star"], indices, temperature)
    else:
        cl = bce.new_zeros(())
    return bce + lambda_cl * cl, {"bce": float(bce.detach()),
                                  "contrastive": float(cl.detach())}
