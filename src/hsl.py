"""Learn refined memberships and construct the sparse incidence matrix H*."""

from collections import defaultdict
from math import sqrt

import numpy as np
import torch


def _edge_size_bucket(size):
    if size <= 10:
        return "small"
    if size <= 100:
        return "medium"
    return "large"


def select_hyperedges(families, sizes, budget, rng):
    """Sample across edge families and size buckets in round-robin order."""
    if budget <= 0:
        raise ValueError("sampled_hyperedges must be positive")

    groups = defaultdict(list)
    for edge_id, family in enumerate(families):
        bucket = _edge_size_bucket(sizes[edge_id])
        groups[(family, bucket)].append(edge_id)

    shuffled_groups = {}
    for key in sorted(groups):
        shuffled_groups[key] = list(rng.permutation(groups[key]))

    target_count = min(budget, len(families))
    selected = []
    while len(selected) < target_count:
        for edge_ids in shuffled_groups.values():
            if not edge_ids:
                continue
            selected.append(int(edge_ids.pop()))
            if len(selected) == target_count:
                break
    return np.asarray(selected, dtype=np.int64)


def sample_negative_nodes(
    node_count,
    incident_nodes,
    count,
    rng,
    allowed_nodes=None,
    deterministic=False,
):
    """Choose nodes that do not currently belong to the hyperedge."""
    if count == 0:
        return np.empty(0, dtype=np.int64)

    if allowed_nodes is not None or deterministic:
        if allowed_nodes is None:
            pool = np.arange(node_count)
        else:
            pool = allowed_nodes
        candidates = np.setdiff1d(pool, incident_nodes, assume_unique=False)
        sample_size = min(count, len(candidates))
        if deterministic:
            return candidates[:sample_size]
        sampled = rng.choice(candidates, sample_size, replace=False)
        return np.sort(sampled)

    incident_set = {int(node_id) for node_id in incident_nodes}
    sample_size = min(count, node_count - len(incident_set))
    selected = set()
    while len(selected) < sample_size:
        remaining = sample_size - len(selected)
        proposals = rng.integers(0, node_count, size=max(8, 2 * remaining))
        for node_id in proposals:
            node_id = int(node_id)
            if node_id in incident_set:
                continue
            selected.add(node_id)
            if len(selected) == sample_size:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def _candidate_nodes(
    incident_nodes,
    node_count,
    positive_count,
    negative_count,
    rng,
    allowed_nodes,
    deterministic,
):
    positive_size = min(positive_count, len(incident_nodes))
    if deterministic:
        positive = np.sort(incident_nodes[:positive_size])
    else:
        positive = rng.choice(incident_nodes, positive_size, replace=False)
        positive = np.sort(positive)

    negative = sample_negative_nodes(
        node_count,
        incident_nodes,
        negative_count,
        rng,
        allowed_nodes,
        deterministic,
    )
    candidates = np.concatenate((positive, negative))
    return positive, candidates


def _membership_scores(
    z0,
    positive,
    candidates,
    node_projection,
    edge_projection,
    membership_bias,
):
    positive_tensor = torch.as_tensor(positive, device=z0.device)
    candidate_tensor = torch.as_tensor(candidates, device=z0.device)
    edge_embedding = z0[positive_tensor].mean(dim=0, keepdim=True)
    projected_nodes = node_projection(z0[candidate_tensor])
    projected_edge = edge_projection(edge_embedding)
    scores = (projected_nodes * projected_edge).sum(dim=1)
    scores = scores / sqrt(z0.shape[1])
    return scores + membership_bias


def _select_memberships(scores, positive_count, top_r, threshold):
    if threshold is None:
        selected = torch.topk(scores, min(top_r, len(scores))).indices
    else:
        probabilities = torch.sigmoid(scores)
        selected = torch.flatnonzero(probabilities >= threshold)
        if len(selected) == 0:
            selected = torch.argmax(scores).reshape(1)

    has_positive = torch.any(selected < positive_count)
    if not has_positive:
        best_positive = torch.argmax(scores[:positive_count])
        if threshold is None and len(selected) >= top_r:
            selected = selected.clone()
            lowest_selected = torch.argmin(scores[selected])
            selected[lowest_selected] = best_positive
        else:
            selected = torch.cat((selected, best_positive.reshape(1)))
    return torch.unique(selected)


def _assemble_h_star(h0, selected_edges, rows, columns, learned_values, z0):
    original = h0.tocoo(copy=False)
    untouched = ~np.isin(original.col, selected_edges)
    retained_rows = original.row[untouched]
    retained_columns = original.col[untouched]

    rows = np.concatenate((retained_rows, np.asarray(rows, dtype=np.int64)))
    columns = np.concatenate((
        retained_columns,
        np.asarray(columns, dtype=np.int64),
    ))

    missing_nodes = np.flatnonzero(
        np.bincount(rows, minlength=h0.shape[0]) == 0
    )
    restored_columns = []
    for node_id in missing_nodes:
        restored_columns.append(h0.indices[h0.indptr[node_id]])
    restored_columns = np.asarray(restored_columns, dtype=np.int64)
    rows = np.concatenate((rows, missing_nodes))
    columns = np.concatenate((columns, restored_columns))

    original_values = torch.ones(
        int(untouched.sum()),
        device=z0.device,
        dtype=z0.dtype,
    )
    restored_values = torch.ones(
        len(missing_nodes),
        device=z0.device,
        dtype=z0.dtype,
    )
    values = torch.cat((
        original_values,
        torch.stack(learned_values),
        restored_values,
    ))
    indices = torch.as_tensor(
        np.stack((rows, columns)),
        dtype=torch.int64,
        device=z0.device,
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        h0.shape,
        check_invariants=False,
    ).coalesce()


def refine_hypergraph(
    z0,
    h0,
    families,
    sizes,
    node_projection,
    edge_projection,
    membership_bias,
    rng,
    *,
    sampled_hyperedges=96,
    positive_nodes=16,
    negative_nodes=16,
    top_r=8,
    threshold=None,
    deterministic=False,
    node_groups=None,
    edge_groups=None,
):
    """Score candidate memberships and return the differentiable sparse H*."""
    h0 = h0.tocsr()
    if h0.shape[0] != len(z0):
        raise ValueError("H0 and node embeddings do not align")
    if h0.shape[1] != len(families) or len(sizes) != len(families):
        raise ValueError("H0 and edge metadata do not align")
    if positive_nodes <= 0 or negative_nodes < 0 or top_r <= 0:
        raise ValueError("Invalid node sampling or top-r budget")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Membership threshold must be in [0, 1]")
    if (node_groups is None) != (edge_groups is None):
        raise ValueError("Node and edge graph groups must be provided together")

    if deterministic:
        selected_edges = np.arange(h0.shape[1], dtype=np.int64)
    else:
        selected_edges = select_hyperedges(
            families,
            sizes,
            sampled_hyperedges,
            rng,
        )

    columns = h0.tocsc()
    retained_rows = []
    retained_columns = []
    learned_values = []
    for edge_id in selected_edges:
        start = columns.indptr[edge_id]
        stop = columns.indptr[edge_id + 1]
        incident_nodes = columns.indices[start:stop]
        if len(incident_nodes) == 0:
            raise ValueError("H0 contains an empty hyperedge")

        allowed_nodes = None
        if node_groups is not None:
            graph_id = edge_groups[edge_id]
            allowed_nodes = np.flatnonzero(node_groups == graph_id)

        positive, candidates = _candidate_nodes(
            incident_nodes,
            h0.shape[0],
            positive_nodes,
            negative_nodes,
            rng,
            allowed_nodes,
            deterministic,
        )
        scores = _membership_scores(
            z0,
            positive,
            candidates,
            node_projection,
            edge_projection,
            membership_bias,
        )
        selected = _select_memberships(
            scores,
            len(positive),
            top_r,
            threshold,
        )

        selected_positions = selected.detach().cpu().numpy()
        retained_rows.extend(candidates[selected_positions])
        retained_columns.extend([int(edge_id)] * len(selected_positions))
        learned_values.extend(torch.sigmoid(scores[selected]).unbind())

    return _assemble_h_star(
        h0,
        selected_edges,
        retained_rows,
        retained_columns,
        learned_values,
        z0,
    )
