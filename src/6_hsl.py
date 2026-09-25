# 6. Learn refined memberships and construct the sparse incidence matrix H*.
# Tham khảo từ project/bài báo:
# - HSL, Cai et al., "Hypergraph Structure Learning for Hypergraph Neural
#   Networks", IJCAI 2022: https://doi.org/10.24963/ijcai.2022/267
#   Code: https://github.com/pkualpha/HSL
# File này là bản điều chỉnh: dùng stratified edge sampling, candidate nodes,
# sigmoid membership và top-r; không tái hiện nguyên bản Gumbel-Softmax hai
# giai đoạn của bài báo.

from collections import defaultdict
from math import sqrt

import numpy as np
import torch


# Map a hyperedge size to the sampling bucket used by HSL.
def _edge_size_bucket(size):
    # Use coarse size ranges to balance sampling across small and large edges.
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
    # Group edges by relation family and size range.
    for edge_id, family in enumerate(families):
        bucket = _edge_size_bucket(sizes[edge_id])
        groups[(family, bucket)].append(edge_id)

    shuffled_groups = {}
    # Shuffle each group independently with the experiment random generator.
    for key in sorted(groups):
        shuffled_groups[key] = list(random_generator.permutation(groups[key]))

    # Never request more refined edges than the graph contains.
    target_count = min(budget, len(families))
    selected = []
    # Round-robin selection prevents one large group from taking the budget.
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
    # Return an empty array when no negative candidates are requested.
    if count == 0:
        return np.empty(0, dtype=np.int64)

    # Build the full valid pool for grouped or deterministic sampling.
    if allowed_nodes is not None or deterministic:
        if allowed_nodes is None:
            pool = np.arange(node_count)
        else:
            pool = allowed_nodes
        # Negative nodes are valid nodes that are not current edge members.
        candidates = np.setdiff1d(pool, incident_nodes, assume_unique=False)
        sample_size = min(count, len(candidates))
        if deterministic:
            return candidates[:sample_size]
        sampled = random_generator.choice(candidates, sample_size, replace=False)
        return np.sort(sampled)

    # Use a set for fast rejection without allocating every graph node.
    incident_set = set()
    for node_id in incident_nodes:
        incident_set.add(int(node_id))
    sample_size = min(count, node_count - len(incident_set))
    selected = set()
    # Draw extra proposals until enough unique non-members are collected.
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
    # Positive candidates are sampled from current members of the edge.
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

    # Negative candidates are sampled from nodes outside the edge.
    negative_node_ids = sample_negative_nodes(
        node_count,
        incident_nodes,
        negative_count,
        random_generator,
        allowed_nodes,
        deterministic,
    )
    # Positives must come first because later code tests position < positive_size.
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
    # Convert NumPy node IDs to tensors on the embedding device.
    positive_tensor = torch.as_tensor(
        positive_node_ids,
        device=initial_node_embeddings.device,
    )
    candidate_tensor = torch.as_tensor(
        candidate_node_ids,
        device=initial_node_embeddings.device,
    )
    # Define edge embedding z_e as the mean embedding of sampled members.
    edge_embedding = initial_node_embeddings[positive_tensor].mean(
        dim=0,
        keepdim=True,
    )
    # Project candidate nodes and the edge into one comparison space.
    projected_nodes = node_projection(initial_node_embeddings[candidate_tensor])
    projected_edge = edge_projection(edge_embedding)
    # Score: s(v,e) = <Wn*h_v, We*z_e> / sqrt(d) + bias.
    scores = (projected_nodes * projected_edge).sum(dim=1)
    scores = scores / sqrt(initial_node_embeddings.shape[1])
    return scores + membership_bias


# Select candidate positions while retaining at least one known member.
def _select_memberships(scores, positive_count, top_r, threshold):
    # Top-r mode keeps a fixed number of candidates with the largest scores.
    if threshold is None:
        selected = torch.topk(scores, min(top_r, len(scores))).indices
    else:
        # Threshold mode keeps candidates with sigmoid(score) >= threshold.
        probabilities = torch.sigmoid(scores)
        selected = torch.flatnonzero(probabilities >= threshold)
        # Keep the best candidate if the threshold removes every candidate.
        if len(selected) == 0:
            selected = torch.argmax(scores).reshape(1)

    # A refined edge must retain at least one known positive member.
    has_positive = torch.any(selected < positive_count)
    if not has_positive:
        best_positive = torch.argmax(scores[:positive_count])
        if threshold is None and len(selected) >= top_r:
            selected = selected.clone()
            lowest_selected = torch.argmin(scores[selected])
            selected[lowest_selected] = best_positive
        else:
            selected = torch.cat((selected, best_positive.reshape(1)))
    # Remove duplicate positions before building the sparse matrix.
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
    # COO exposes all original (node, edge) membership coordinates.
    original = initial_incidence_matrix.tocoo(copy=False)
    # Keep H0 memberships only for edges that were not selected for refinement.
    untouched = ~np.isin(original.col, selected_edge_ids)
    retained_rows = original.row[untouched]
    retained_columns = original.col[untouched]

    # Combine untouched H0 coordinates with learned coordinates.
    node_indices = np.concatenate((
        retained_rows,
        np.asarray(node_indices, dtype=np.int64),
    ))
    edge_indices = np.concatenate((
        retained_columns,
        np.asarray(edge_indices, dtype=np.int64),
    ))

    # Find nodes that lost every membership after edge refinement.
    missing_nodes = np.flatnonzero(
        np.bincount(node_indices, minlength=initial_incidence_matrix.shape[0]) == 0
    )
    restored_columns = []
    # Restore one original edge per isolated node to keep node degree positive.
    for node_id in missing_nodes:
        restored_columns.append(
            initial_incidence_matrix.indices[initial_incidence_matrix.indptr[node_id]]
        )
    restored_columns = np.asarray(restored_columns, dtype=np.int64)
    node_indices = np.concatenate((node_indices, missing_nodes))
    edge_indices = np.concatenate((edge_indices, restored_columns))

    # Untouched and restored H0 memberships keep binary weight 1.
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
    # H* combines fixed weights and differentiable learned probabilities.
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
    # H*[v,e] is the final weight of node v in hyperedge e.
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
    # H0 rows and embedding rows must describe the same nodes.
    if initial_incidence_matrix.shape[0] != len(initial_node_embeddings):
        raise ValueError("H0 and node embeddings do not align")
    # H0 columns, family labels, and edge sizes must describe the same edges.
    if (
        initial_incidence_matrix.shape[1] != len(families)
        or len(sizes) != len(families)
    ):
        raise ValueError("H0 and edge metadata do not align")
    if positive_nodes <= 0 or negative_nodes < 0 or top_r <= 0:
        raise ValueError("Invalid node sampling or top-r budget")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Membership threshold must be in [0, 1]")
    # Graph groups are optional, but node and edge groups must come together.
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
    # CSC column pointers locate all current members of this edge.
    membership_start = incidence_by_edge.indptr[edge_id]
    membership_stop = incidence_by_edge.indptr[edge_id + 1]
    incident_node_ids = incidence_by_edge.indices[
        membership_start:membership_stop
    ]
    if len(incident_node_ids) == 0:
        raise ValueError("H0 contains an empty hyperedge")

    allowed_node_ids = None
    # In a batched graph, sample candidates only from the same local graph.
    if node_groups is not None:
        graph_id = edge_groups[edge_id]
        allowed_node_ids = np.flatnonzero(node_groups == graph_id)

    # Build a small candidate set instead of scoring every graph node.
    positive_node_ids, candidate_node_ids = _candidate_nodes(
        incident_node_ids,
        initial_node_embeddings.shape[0],
        positive_nodes,
        negative_nodes,
        random_generator,
        allowed_node_ids,
        deterministic,
    )
    # Learn how strongly each candidate should belong to this edge.
    membership_scores = _membership_scores(
        initial_node_embeddings,
        positive_node_ids,
        candidate_node_ids,
        node_projection,
        edge_projection,
        membership_bias,
    )
    # Keep candidates by top-r score or probability threshold.
    selected_positions = _select_memberships(
        membership_scores,
        len(positive_node_ids),
        top_r,
        threshold,
    )

    selected_numpy_positions = selected_positions.detach().cpu().numpy()
    retained_node_ids = candidate_node_ids[selected_numpy_positions]
    retained_edge_ids = [int(edge_id)] * len(selected_numpy_positions)
    # Convert logits to soft membership weights in the range (0, 1).
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
    # CSR is efficient for validation and node-based restoration.
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

    # Deterministic mode refines every edge in a fixed order.
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

    # CSC is efficient for reading all member nodes of one edge column.
    incidence_by_edge = initial_incidence_matrix.tocsc()
    retained_node_indices = []
    retained_edge_indices = []
    learned_membership_values = []
    # Refine each sampled edge and collect its learned sparse entries.
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

    # Merge learned entries with untouched H0 entries to form H*.
    return _assemble_h_star(
        initial_incidence_matrix,
        selected_edge_ids,
        retained_node_indices,
        retained_edge_indices,
        learned_membership_values,
        initial_node_embeddings,
    )
