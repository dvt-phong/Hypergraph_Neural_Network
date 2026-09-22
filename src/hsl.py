"""Sample node-edge pairs and refine the sparse hypergraph incidence matrix."""

from collections import defaultdict
from math import sqrt

import numpy as np
from scipy import sparse
import torch


def sample_edges(families, sizes, budget, rng):
    if budget <= 0:
        raise ValueError("sampled_hyperedges must be positive")
    groups = defaultdict(list)
    for edge_id, (family, size) in enumerate(zip(families, sizes)):
        bucket = "small" if size <= 10 else "medium" if size <= 100 else "large"
        groups[(family, bucket)].append(edge_id)
    groups = {key: list(rng.permutation(ids)) for key, ids in sorted(groups.items())}
    selected = []
    while len(selected) < min(budget, len(families)):
        for ids in groups.values():
            if ids:
                selected.append(int(ids.pop()))
                if len(selected) == min(budget, len(families)):
                    break
    return np.array(selected, dtype=np.int64)


def _negative_nodes(n, incident, count, rng, allowed=None, deterministic=False):
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if allowed is not None or deterministic:
        pool = np.arange(n) if allowed is None else allowed
        candidates = np.setdiff1d(pool, incident, assume_unique=False)
        size = min(count, len(candidates))
        return candidates[:size] if deterministic else np.sort(rng.choice(candidates, size,
                                                                          replace=False))
    incident_set = set(int(node) for node in incident)
    size = min(count, n - len(incident_set))
    selected = set()
    while len(selected) < size:
        for node in rng.integers(0, n, size=max(8, 2 * (size - len(selected)))):
            if int(node) not in incident_set:
                selected.add(int(node))
                if len(selected) == size:
                    break
    return np.array(sorted(selected), dtype=np.int64)


def refine_hypergraph(z0, h0, families, sizes, node_projection, edge_projection, bias,
                      rng, *, sampled_hyperedges=96, positive_nodes=16, negative_nodes=16,
                      top_r=8, threshold=None, deterministic=False,
                      node_groups=None, edge_groups=None):
    """Return weighted H* and an audit; train samples edges, inference refines all."""
    h0 = h0.tocsr()
    if h0.shape[0] != len(z0) or h0.shape[1] != len(families) or len(sizes) != len(families):
        raise ValueError("H0, node embeddings and edge metadata do not align")
    if positive_nodes <= 0 or negative_nodes < 0 or top_r <= 0:
        raise ValueError("Invalid node sampling or top-r budget")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Membership threshold must be in [0, 1]")
    if (node_groups is None) != (edge_groups is None):
        raise ValueError("Node and edge graph groups must be provided together")
    if deterministic:
        selected_edges = np.arange(h0.shape[1], dtype=np.int64)
    else:
        selected_edges = sample_edges(families, sizes, sampled_hyperedges, rng)
    columns = h0.tocsc()
    retained_rows, retained_columns, learned_values = [], [], []
    retained_positive = added_negative = candidate_count = 0
    for edge_id in selected_edges:
        start, stop = columns.indptr[edge_id:edge_id + 2]
        incident = columns.indices[start:stop]
        if len(incident) == 0:
            raise ValueError("H0 contains an empty edge")
        take = min(positive_nodes, len(incident))
        positive = (np.sort(incident[:take]) if deterministic else
                    np.sort(rng.choice(incident, take, replace=False)))
        allowed = None
        if node_groups is not None:
            allowed = np.flatnonzero(node_groups == edge_groups[edge_id])
        negative = _negative_nodes(h0.shape[0], incident, negative_nodes, rng,
                                   allowed, deterministic)
        candidates = np.concatenate((positive, negative))
        candidate_count += len(candidates)
        positive_tensor = torch.as_tensor(positive, device=z0.device)
        candidate_tensor = torch.as_tensor(candidates, device=z0.device)
        edge_embedding = z0[positive_tensor].mean(dim=0, keepdim=True)
        score = ((node_projection(z0[candidate_tensor]) *
                  edge_projection(edge_embedding)).sum(dim=1) / sqrt(z0.shape[1]) + bias)
        if threshold is None:
            keep = torch.topk(score, min(top_r, len(candidates))).indices
        else:
            keep = torch.flatnonzero(torch.sigmoid(score) >= threshold)
            if len(keep) == 0:
                keep = torch.argmax(score).reshape(1)
        if not torch.any(keep < len(positive)):
            best = torch.argmax(score[:len(positive)])
            if threshold is None and len(keep) >= top_r:
                keep = keep.clone()
                keep[torch.argmin(score[keep])] = best
            else:
                keep = torch.cat((keep, best.reshape(1)))
        keep = torch.unique(keep)
        selected = keep.detach().cpu().numpy()
        retained_rows.extend(candidates[selected])
        retained_columns.extend([int(edge_id)] * len(selected))
        learned_values.extend(torch.sigmoid(score[keep]).unbind())
        retained_positive += int(np.count_nonzero(selected < len(positive)))
        added_negative += int(np.count_nonzero(selected >= len(positive)))

    old = h0.tocoo(copy=False)
    untouched = ~np.isin(old.col, selected_edges)
    rows = np.concatenate((old.row[untouched], np.asarray(retained_rows, dtype=np.int64)))
    cols = np.concatenate((old.col[untouched], np.asarray(retained_columns, dtype=np.int64)))
    missing = np.flatnonzero(np.bincount(rows, minlength=h0.shape[0]) == 0)
    restored_columns = np.array([h0.indices[h0.indptr[node]] for node in missing],
                                dtype=np.int64)
    rows = np.concatenate((rows, missing))
    cols = np.concatenate((cols, restored_columns))
    values = torch.cat((torch.ones(int(untouched.sum()), device=z0.device,
                                   dtype=z0.dtype), torch.stack(learned_values),
                        torch.ones(len(missing), device=z0.device, dtype=z0.dtype)))
    indices = torch.as_tensor(np.stack((rows, cols)), dtype=torch.int64, device=z0.device)
    h_star = torch.sparse_coo_tensor(indices, values, h0.shape,
                                     check_invariants=False).coalesce()
    audit = {"sampled_hyperedges": len(selected_edges),
             "membership_candidates": candidate_count,
             "retained_positive_memberships": retained_positive,
             "added_negative_memberships": added_negative,
             "restored_isolated_nodes": len(missing),
             "initial_incidences": h0.nnz, "refined_incidences": h_star._nnz()}
    return h_star, audit
