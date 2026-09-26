# 7. Losses: weighted BCE + intra-hyperedge contrastive loss.
#
#     L = BCE(logits, y; pos_weight = #negative / #positive) + λ · L_CL(Z0, Z*)
#
# Intra-hyperedge contrastive loss (HSL, Cai et al., IJCAI 2022, Eq. 10):
#   positive pair   the same enrollment in both views: Z0[i] <-> Z*[i]
#   negatives       enrollments that share a hyperedge with i (T_i),
#                   taken from both views: Z0[j] and Z*[j]
# The loss keeps H* consistent with H0 while keeping learners of the same
# course/object distinguishable (against over-smoothing). It is computed with
# each view as anchor and averaged, like the SimCLR-style loss in HSL's code.

from importlib import import_module

import numpy as np
import torch
from torch.nn import functional as F

config = import_module("0_config")
SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")


def positive_class_weight(labels):
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Both classes are required in the train split")
    return negatives / positives


# Index of H0 memberships (without self-loops) for fast neighbor sampling.
# Arrays follow graph["edge_ids"], which is sorted by hyperedge.
def build_neighbor_sampler(graph):
    keep = graph["edge_family"][graph["edge_ids"]] != SELF_LOOP
    node_ids = graph["node_ids"][keep]
    edge_ids = graph["edge_ids"][keep]
    by_node = np.argsort(node_ids, kind="stable")
    node_start = np.searchsorted(node_ids[by_node], np.arange(graph["num_nodes"] + 1))
    return {
        "node_start": node_start,
        "node_edges": edge_ids[by_node],
        "edge_start": np.searchsorted(edge_ids, np.arange(len(graph["edge_family"]) + 1)),
        "edge_nodes": node_ids,
        "anchor_pool": np.flatnonzero(np.diff(node_start) > 0),
    }


# For each anchor, `count` random members of T_i: pick one of the anchor's
# hyperedges at random, then one member of that hyperedge at random.
def sample_hyperedge_neighbors(sampler, anchors, count, rng):
    start = sampler["node_start"][anchors][:, None]
    degree = sampler["node_start"][anchors + 1][:, None] - start
    random_edge = start + (rng.random((len(anchors), count)) * degree).astype(np.int64)
    edges = sampler["node_edges"][random_edge]

    edge_start = sampler["edge_start"][edges]
    edge_size = sampler["edge_start"][edges + 1] - edge_start
    random_member = edge_start + (rng.random(edges.shape) * edge_size).astype(np.int64)
    return sampler["edge_nodes"][random_member]


# One direction of Eq. 10 with `view` as anchor and `other` as the second view.
def _contrastive_one_direction(view, other, anchors, neighbors, temperature):
    anchor = view[anchors]  # [B, D]
    positive = (anchor * other[anchors]).sum(dim=1, keepdim=True)  # [B, 1]
    same_view = torch.einsum("bd,bkd->bk", anchor, view[neighbors])  # [B, K]
    other_view = torch.einsum("bd,bkd->bk", anchor, other[neighbors])  # [B, K]
    logits = torch.cat([positive, same_view, other_view], dim=1) / temperature

    # A sampled neighbor that is the anchor itself is not a negative.
    is_anchor = neighbors == anchors[:, None]
    never = torch.zeros_like(is_anchor[:, :1])
    logits = logits.masked_fill(torch.cat([never, is_anchor, is_anchor], dim=1), -torch.inf)

    positive_position = torch.zeros(len(anchors), dtype=torch.int64, device=logits.device)
    return F.cross_entropy(logits, positive_position)


def contrastive_loss(z0, z_star, anchors, neighbors, temperature):
    z0 = F.normalize(z0, dim=1)
    z_star = F.normalize(z_star, dim=1)
    return (
        _contrastive_one_direction(z0, z_star, anchors, neighbors, temperature)
        + _contrastive_one_direction(z_star, z0, anchors, neighbors, temperature)
    ) / 2


def total_loss(output, labels, positive_weight, anchors, neighbors, *, lambda_cl, temperature):
    bce = F.binary_cross_entropy_with_logits(output["logits"], labels, pos_weight=positive_weight)
    if lambda_cl > 0:
        cl = contrastive_loss(output["z0"], output["z_star"], anchors, neighbors, temperature)
    else:
        cl = bce.new_zeros(())
    return bce + lambda_cl * cl, {"bce": bce.item(), "contrastive": cl.item()}
