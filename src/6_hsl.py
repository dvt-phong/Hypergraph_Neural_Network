# 6. HSL-inspired structure refinement for the MOOC hypergraph.
#
# This is not the original two-stage Gumbel/Hard-Concrete HSL implementation.
# The adapted flow is intentionally direct:
#   select edges -> sample positive/negative nodes -> score memberships
#   -> keep top-r -> build sparse H*.

import time
from collections import defaultdict
from math import sqrt

import numpy as np
import torch


def log_event(message):
    print(f"[{time.strftime('%H:%M:%S')}][hsl] {message}", flush=True)


# Balance the refinement budget across relation families and edge sizes.
def select_hyperedges(families, sizes, budget, random_generator):
    if budget <= 0:
        raise ValueError("sampled_hyperedges must be positive")

    edge_groups = defaultdict(list)
    for edge_id, family in enumerate(families):
        edge_size = sizes[edge_id]
        if edge_size <= 10:
            size_group = "small"
        elif edge_size <= 100:
            size_group = "medium"
        else:
            size_group = "large"
        edge_groups[(family, size_group)].append(edge_id)

    shuffled_groups = []
    for group_name in sorted(edge_groups):
        shuffled_edge_ids = list(
            random_generator.permutation(edge_groups[group_name])
        )
        shuffled_groups.append(shuffled_edge_ids)

    selected_edge_ids = []
    target_count = min(budget, len(families))
    while len(selected_edge_ids) < target_count:
        for edge_ids in shuffled_groups:
            if not edge_ids:
                continue
            selected_edge_ids.append(int(edge_ids.pop()))
            if len(selected_edge_ids) == target_count:
                break

    return np.asarray(selected_edge_ids, dtype=np.int64)


# Positive candidates are current members; negatives are sampled non-members.
def sample_candidate_nodes(
    incident_node_ids,
    node_count,
    positive_limit,
    negative_limit,
    random_generator,
    *,
    allowed_node_ids=None,
    deterministic=False,
):
    positive_count = min(positive_limit, len(incident_node_ids))
    if deterministic:
        positive_node_ids = np.sort(incident_node_ids[:positive_count])
    else:
        positive_node_ids = random_generator.choice(
            incident_node_ids,
            positive_count,
            replace=False,
        )
        positive_node_ids = np.sort(positive_node_ids)

    if negative_limit == 0:
        negative_node_ids = np.empty(0, dtype=np.int64)
    elif allowed_node_ids is not None or deterministic:
        if allowed_node_ids is None:
            negative_pool = np.arange(node_count)
        else:
            negative_pool = allowed_node_ids
        negative_pool = np.setdiff1d(
            negative_pool,
            incident_node_ids,
            assume_unique=False,
        )
        negative_count = min(negative_limit, len(negative_pool))
        if deterministic:
            negative_node_ids = negative_pool[:negative_count]
        else:
            negative_node_ids = random_generator.choice(
                negative_pool,
                negative_count,
                replace=False,
            )
            negative_node_ids = np.sort(negative_node_ids)
    else:
        # Rejection sampling avoids allocating an array for every graph node.
        incident_set = set(int(node_id) for node_id in incident_node_ids)
        negative_count = min(negative_limit, node_count - len(incident_set))
        selected_negative_ids = set()
        while len(selected_negative_ids) < negative_count:
            remaining = negative_count - len(selected_negative_ids)
            proposals = random_generator.integers(
                0,
                node_count,
                size=max(8, 2 * remaining),
            )
            for node_id in proposals:
                node_id = int(node_id)
                if node_id not in incident_set:
                    selected_negative_ids.add(node_id)
                if len(selected_negative_ids) == negative_count:
                    break
        negative_node_ids = np.asarray(
            sorted(selected_negative_ids),
            dtype=np.int64,
        )

    # Positives stay first so positions < positive_count identify known members.
    candidate_node_ids = np.concatenate(
        (positive_node_ids, negative_node_ids)
    )
    return positive_node_ids, candidate_node_ids


def membership_scores(
    initial_embeddings,
    positive_node_ids,
    candidate_node_ids,
    node_projection,
    edge_projection,
    membership_bias,
):
    positive_node_ids = torch.as_tensor(
        positive_node_ids,
        device=initial_embeddings.device,
    )
    candidate_node_ids = torch.as_tensor(
        candidate_node_ids,
        device=initial_embeddings.device,
    )

    # z_e is the mean embedding of sampled nodes already in the edge.
    edge_embedding = initial_embeddings[positive_node_ids].mean(
        dim=0,
        keepdim=True,
    )
    projected_nodes = node_projection(initial_embeddings[candidate_node_ids])
    projected_edge = edge_projection(edge_embedding)

    # s(v,e) = (Wn z_v)^T (We z_e) / sqrt(d) + b
    scores = (projected_nodes * projected_edge).sum(dim=1)
    scores = scores / sqrt(initial_embeddings.shape[1])
    return scores + membership_bias


# Merge learned memberships with untouched H0 edges and prevent isolated nodes.
def build_refined_incidence(
    initial_incidence_matrix,
    selected_edge_ids,
    learned_node_ids,
    learned_edge_ids,
    learned_values,
    initial_embeddings,
):
    original_memberships = initial_incidence_matrix.tocoo(copy=False)
    untouched_memberships = ~np.isin(
        original_memberships.col,
        selected_edge_ids,
    )

    final_node_ids = np.concatenate((
        original_memberships.row[untouched_memberships],
        np.asarray(learned_node_ids, dtype=np.int64),
    ))
    final_edge_ids = np.concatenate((
        original_memberships.col[untouched_memberships],
        np.asarray(learned_edge_ids, dtype=np.int64),
    ))

    membership_count_by_node = np.bincount(
        final_node_ids,
        minlength=initial_incidence_matrix.shape[0],
    )
    isolated_node_ids = np.flatnonzero(membership_count_by_node == 0)

    restored_edge_ids = []
    for node_id in isolated_node_ids:
        first_membership = initial_incidence_matrix.indptr[node_id]
        restored_edge_ids.append(
            initial_incidence_matrix.indices[first_membership]
        )

    final_node_ids = np.concatenate((final_node_ids, isolated_node_ids))
    final_edge_ids = np.concatenate((
        final_edge_ids,
        np.asarray(restored_edge_ids, dtype=np.int64),
    ))

    fixed_value_count = int(untouched_memberships.sum())
    fixed_values = torch.ones(
        fixed_value_count,
        device=initial_embeddings.device,
        dtype=initial_embeddings.dtype,
    )
    restored_values = torch.ones(
        len(isolated_node_ids),
        device=initial_embeddings.device,
        dtype=initial_embeddings.dtype,
    )
    final_values = torch.cat((
        fixed_values,
        torch.stack(learned_values),
        restored_values,
    ))
    final_indices = torch.as_tensor(
        np.stack((final_node_ids, final_edge_ids)),
        dtype=torch.int64,
        device=initial_embeddings.device,
    )
    return torch.sparse_coo_tensor(
        final_indices,
        final_values,
        initial_incidence_matrix.shape,
        check_invariants=False,
    ).coalesce()


def refine_hypergraph(
    initial_embeddings,
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
    deterministic=False,
    node_groups=None,
    edge_groups=None,
):
    started_at = time.perf_counter()
    initial_incidence_matrix = initial_incidence_matrix.tocsr()

    if initial_incidence_matrix.shape[0] != len(initial_embeddings):
        raise ValueError("H0 and node embeddings do not align")
    if initial_incidence_matrix.shape[1] != len(families):
        raise ValueError("H0 and edge families do not align")
    if len(sizes) != len(families):
        raise ValueError("Edge sizes and edge families do not align")
    if positive_nodes <= 0 or negative_nodes < 0 or top_r <= 0:
        raise ValueError("Invalid candidate sampling settings")
    if (node_groups is None) != (edge_groups is None):
        raise ValueError("node_groups and edge_groups must be provided together")

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
        log_event(
            f"refinement started: selected={len(selected_edge_ids):,}/"
            f"{initial_incidence_matrix.shape[1]:,} edges, "
            f"positive={positive_nodes}, negative={negative_nodes}, top_r={top_r}"
        )

    incidence_by_edge = initial_incidence_matrix.tocsc()
    learned_node_ids = []
    learned_edge_ids = []
    learned_values = []
    progress_step = max(1, len(selected_edge_ids) // 4)

    for edge_position, edge_id in enumerate(selected_edge_ids, start=1):
        membership_start = incidence_by_edge.indptr[edge_id]
        membership_stop = incidence_by_edge.indptr[edge_id + 1]
        incident_node_ids = incidence_by_edge.indices[
            membership_start:membership_stop
        ]
        if len(incident_node_ids) == 0:
            raise ValueError("H0 contains an empty hyperedge")

        allowed_node_ids = None
        if node_groups is not None:
            local_graph_id = edge_groups[edge_id]
            allowed_node_ids = np.flatnonzero(
                node_groups == local_graph_id
            )

        positive_node_ids, candidate_node_ids = sample_candidate_nodes(
            incident_node_ids,
            initial_embeddings.shape[0],
            positive_nodes,
            negative_nodes,
            random_generator,
            allowed_node_ids=allowed_node_ids,
            deterministic=deterministic,
        )
        candidate_scores = membership_scores(
            initial_embeddings,
            positive_node_ids,
            candidate_node_ids,
            node_projection,
            edge_projection,
            membership_bias,
        )

        selected_positions = torch.topk(
            candidate_scores,
            min(top_r, len(candidate_scores)),
        ).indices

        # Every refined edge keeps at least one node from the original edge.
        positive_count = len(positive_node_ids)
        if not torch.any(selected_positions < positive_count):
            best_positive = torch.argmax(candidate_scores[:positive_count])
            lowest_selected = torch.argmin(candidate_scores[selected_positions])
            selected_positions = selected_positions.clone()
            selected_positions[lowest_selected] = best_positive
        selected_positions = torch.unique(selected_positions)

        selected_numpy_positions = selected_positions.detach().cpu().numpy()
        selected_node_ids = candidate_node_ids[selected_numpy_positions]
        learned_node_ids.extend(selected_node_ids)
        learned_edge_ids.extend([int(edge_id)] * len(selected_node_ids))

        membership_probabilities = torch.sigmoid(
            candidate_scores[selected_positions]
        )
        learned_values.extend(membership_probabilities.unbind())

        if (
            not deterministic
            and (
                edge_position % progress_step == 0
                or edge_position == len(selected_edge_ids)
            )
        ):
            log_event(
                f"refined {edge_position:,}/{len(selected_edge_ids):,} edges "
                f"in {time.perf_counter() - started_at:.1f}s"
            )

    refined_incidence_matrix = build_refined_incidence(
        initial_incidence_matrix,
        selected_edge_ids,
        learned_node_ids,
        learned_edge_ids,
        learned_values,
        initial_embeddings,
    )
    if not deterministic:
        log_event(
            f"refinement completed: H* nnz="
            f"{refined_incidence_matrix._nnz():,}, "
            f"elapsed={time.perf_counter() - started_at:.1f}s"
        )
    return refined_incidence_matrix
