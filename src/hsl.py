# Learn refined memberships and construct the sparse incidence matrix H*.

from collections import defaultdict
from math import sqrt

import numpy as np
import torch


# Map a hyperedge size to the sampling bucket used by HSL.
def _edge_size_bucket(size):
    if size <= 10:
        return "small"
    if size <= 100:
        return "medium"
    return "large"


# Sample across edge families and size buckets in round-robin order.
def select_hyperedges(families, sizes, budget, random_generator):
    if budget <= 0:
        raise ValueError("sampled_hyperedges must be positive")

    groups = defaultdict(list)
    for edge_id, family in enumerate(families):
        bucket = _edge_size_bucket(sizes[edge_id])
        groups[(family, bucket)].append(edge_id)

    shuffled_groups = {}
    for key in sorted(groups):
        shuffled_groups[key] = list(random_generator.permutation(groups[key]))

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


# Choose nodes that do not currently belong to the hyperedge.
def sample_negative_nodes(
    node_count,
    incident_nodes,
    count,
    random_generator,
    allowed_nodes=None,
    deterministic=False,
):
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
        sampled = random_generator.choice(candidates, sample_size, replace=False)
        return np.sort(sampled)

    incident_set = set()
    for node_id in incident_nodes:
        incident_set.add(int(node_id))
    sample_size = min(count, node_count - len(incident_set))
    selected = set()
    while len(selected) < sample_size:
        remaining = sample_size - len(selected)
        proposals = random_generator.integers(
            0,
            node_count,
            size=max(8, 2 * remaining),
        )
        for node_id in proposals:
            node_id = int(node_id)
            if node_id in incident_set:
                continue
            selected.add(node_id)
            if len(selected) == sample_size:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


# Sample current members and non-members as candidates for one hyperedge.
def _candidate_nodes(
    incident_nodes,
    node_count,
    positive_count,
    negative_count,
    random_generator,
    allowed_nodes,
    deterministic,
):
    positive_size = min(positive_count, len(incident_nodes))
    if deterministic:
        positive_node_ids = np.sort(incident_nodes[:positive_size])
    else:
        positive_node_ids = random_generator.choice(
            incident_nodes,
            positive_size,
            replace=False,
        )
        positive_node_ids = np.sort(positive_node_ids)

    negative_node_ids = sample_negative_nodes(
        node_count,
        incident_nodes,
        negative_count,
        random_generator,
        allowed_nodes,
        deterministic,
    )
    candidate_node_ids = np.concatenate((positive_node_ids, negative_node_ids))
    return positive_node_ids, candidate_node_ids


# Score candidate memberships against an embedding derived from known members.
def _membership_scores(
    initial_node_embeddings,
    positive_node_ids,
    candidate_node_ids,
    node_projection,
    edge_projection,
    membership_bias,
):
    positive_tensor = torch.as_tensor(
        positive_node_ids,
        device=initial_node_embeddings.device,
    )
    candidate_tensor = torch.as_tensor(
        candidate_node_ids,
        device=initial_node_embeddings.device,
    )
    edge_embedding = initial_node_embeddings[positive_tensor].mean(
        dim=0,
        keepdim=True,
    )
    projected_nodes = node_projection(initial_node_embeddings[candidate_tensor])
    projected_edge = edge_projection(edge_embedding)
    scores = (projected_nodes * projected_edge).sum(dim=1)
    scores = scores / sqrt(initial_node_embeddings.shape[1])
    return scores + membership_bias


# Select candidate positions while retaining at least one known member.
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


# Assemble retained and learned memberships into the sparse refined matrix H*.
def _assemble_h_star(
    initial_incidence_matrix,
    selected_edge_ids,
    node_indices,
    edge_indices,
    learned_membership_values,
    initial_node_embeddings,
):
    original = initial_incidence_matrix.tocoo(copy=False)
    untouched = ~np.isin(original.col, selected_edge_ids)
    retained_rows = original.row[untouched]
    retained_columns = original.col[untouched]

    node_indices = np.concatenate((
        retained_rows,
        np.asarray(node_indices, dtype=np.int64),
    ))
    edge_indices = np.concatenate((
        retained_columns,
        np.asarray(edge_indices, dtype=np.int64),
    ))

    missing_nodes = np.flatnonzero(
        np.bincount(node_indices, minlength=initial_incidence_matrix.shape[0]) == 0
    )
    restored_columns = []
    for node_id in missing_nodes:
        restored_columns.append(
            initial_incidence_matrix.indices[initial_incidence_matrix.indptr[node_id]]
        )
    restored_columns = np.asarray(restored_columns, dtype=np.int64)
    node_indices = np.concatenate((node_indices, missing_nodes))
    edge_indices = np.concatenate((edge_indices, restored_columns))

    original_values = torch.ones(
        int(untouched.sum()),
        device=initial_node_embeddings.device,
        dtype=initial_node_embeddings.dtype,
    )
    restored_values = torch.ones(
        len(missing_nodes),
        device=initial_node_embeddings.device,
        dtype=initial_node_embeddings.dtype,
    )
    values = torch.cat((
        original_values,
        torch.stack(learned_membership_values),
        restored_values,
    ))
    indices = torch.as_tensor(
        np.stack((node_indices, edge_indices)),
        dtype=torch.int64,
        device=initial_node_embeddings.device,
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        initial_incidence_matrix.shape,
        check_invariants=False,
    ).coalesce()


# Validate alignment and sampling settings before refining memberships.
def _validate_refinement_inputs(
    initial_node_embeddings,
    initial_incidence_matrix,
    families,
    sizes,
    positive_nodes,
    negative_nodes,
    top_r,
    threshold,
    node_groups,
    edge_groups,
):
    if initial_incidence_matrix.shape[0] != len(initial_node_embeddings):
        raise ValueError("H0 and node embeddings do not align")
    if (
        initial_incidence_matrix.shape[1] != len(families)
        or len(sizes) != len(families)
    ):
        raise ValueError("H0 and edge metadata do not align")
    if positive_nodes <= 0 or negative_nodes < 0 or top_r <= 0:
        raise ValueError("Invalid node sampling or top-r budget")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Membership threshold must be in [0, 1]")
    if (node_groups is None) != (edge_groups is None):
        raise ValueError("Node and edge graph groups must be provided together")


# Refine candidate memberships for one selected hyperedge.
def _refine_one_edge(
    edge_id,
    incidence_by_edge,
    initial_node_embeddings,
    node_projection,
    edge_projection,
    membership_bias,
    random_generator,
    positive_nodes,
    negative_nodes,
    top_r,
    threshold,
    deterministic,
    node_groups,
    edge_groups,
):
    membership_start = incidence_by_edge.indptr[edge_id]
    membership_stop = incidence_by_edge.indptr[edge_id + 1]
    incident_node_ids = incidence_by_edge.indices[
        membership_start:membership_stop
    ]
    if len(incident_node_ids) == 0:
        raise ValueError("H0 contains an empty hyperedge")

    allowed_node_ids = None
    if node_groups is not None:
        graph_id = edge_groups[edge_id]
        allowed_node_ids = np.flatnonzero(node_groups == graph_id)

    positive_node_ids, candidate_node_ids = _candidate_nodes(
        incident_node_ids,
        initial_node_embeddings.shape[0],
        positive_nodes,
        negative_nodes,
        random_generator,
        allowed_node_ids,
        deterministic,
    )
    membership_scores = _membership_scores(
        initial_node_embeddings,
        positive_node_ids,
        candidate_node_ids,
        node_projection,
        edge_projection,
        membership_bias,
    )
    selected_positions = _select_memberships(
        membership_scores,
        len(positive_node_ids),
        top_r,
        threshold,
    )

    selected_numpy_positions = selected_positions.detach().cpu().numpy()
    retained_node_ids = candidate_node_ids[selected_numpy_positions]
    retained_edge_ids = [int(edge_id)] * len(selected_numpy_positions)
    learned_values = torch.sigmoid(
        membership_scores[selected_positions]
    ).unbind()
    return retained_node_ids, retained_edge_ids, learned_values


# Score candidate memberships and return the differentiable sparse H*.
def refine_hypergraph(
    initial_node_embeddings,
    initial_incidence_matrix,
    families,
    sizes,
    node_projection,
    edge_projection,
    membership_bias,
    random_generator,
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
    initial_incidence_matrix = initial_incidence_matrix.tocsr()
    _validate_refinement_inputs(
        initial_node_embeddings,
        initial_incidence_matrix,
        families,
        sizes,
        positive_nodes,
        negative_nodes,
        top_r,
        threshold,
        node_groups,
        edge_groups,
    )

    if deterministic:
        selected_edge_ids = np.arange(
            initial_incidence_matrix.shape[1],
            dtype=np.int64,
        )
    else:
        selected_edge_ids = select_hyperedges(
            families,
            sizes,
            sampled_hyperedges,
            random_generator,
        )

    incidence_by_edge = initial_incidence_matrix.tocsc()
    retained_node_indices = []
    retained_edge_indices = []
    learned_membership_values = []
    for edge_id in selected_edge_ids:
        (
            edge_node_ids,
            edge_ids,
            edge_membership_values,
        ) = _refine_one_edge(
            edge_id,
            incidence_by_edge,
            initial_node_embeddings,
            node_projection,
            edge_projection,
            membership_bias,
            random_generator,
            positive_nodes,
            negative_nodes,
            top_r,
            threshold,
            deterministic,
            node_groups,
            edge_groups,
        )
        retained_node_indices.extend(edge_node_ids)
        retained_edge_indices.extend(edge_ids)
        learned_membership_values.extend(edge_membership_values)

    return _assemble_h_star(
        initial_incidence_matrix,
        selected_edge_ids,
        retained_node_indices,
        retained_edge_indices,
        learned_membership_values,
        initial_node_embeddings,
    )
