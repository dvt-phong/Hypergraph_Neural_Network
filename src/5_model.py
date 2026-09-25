# 5. Define HGNN propagation and the two-pass dropout model.
#
# HGNN follows Feng et al., AAAI 2019:
# https://github.com/iMoonLab/HGNN
#
# The structure-refinement step is inspired by HSL, IJCAI 2022, but is adapted
# for the MOOC graph. It is not a reproduction of the original HSL algorithm.

import time
from importlib import import_module

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

hsl_module = import_module("6_hsl")
refine_hypergraph = hsl_module.refine_hypergraph

MESSAGE_CHUNK_SIZE = 250_000


def log_event(message):
    print(f"[{time.strftime('%H:%M:%S')}][model] {message}", flush=True)


def to_torch_sparse(scipy_matrix, device):
    coordinate_matrix = scipy_matrix.tocoo(copy=False)
    indices = torch.as_tensor(
        np.stack((coordinate_matrix.row, coordinate_matrix.col)),
        dtype=torch.int64,
        device=device,
    )
    values = torch.as_tensor(
        coordinate_matrix.data.astype(np.float32),
        device=device,
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        coordinate_matrix.shape,
        check_invariants=False,
    ).coalesce()


# HGNN: Dv^(-1/2) H De^(-1) H^T Dv^(-1/2) X.
def hypergraph_propagation(node_features, incidence_matrix):
    incidence_matrix = incidence_matrix.coalesce()
    node_indices, edge_indices = incidence_matrix.indices()
    membership_values = incidence_matrix.values()

    node_count, edge_count = incidence_matrix.shape
    node_degree = torch.zeros(
        node_count,
        device=membership_values.device,
        dtype=membership_values.dtype,
    )
    edge_degree = torch.zeros(
        edge_count,
        device=membership_values.device,
        dtype=membership_values.dtype,
    )
    node_degree.scatter_add_(0, node_indices, membership_values)
    edge_degree.scatter_add_(0, edge_indices, membership_values)
    if torch.any(node_degree <= 0) or torch.any(edge_degree <= 0):
        raise ValueError("H contains an isolated node or an empty hyperedge")

    node_scale = node_degree.rsqrt()
    edge_scale = edge_degree.reciprocal()
    scaled_node_features = node_features * node_scale[:, None]

    # H0 has fixed values and can use PyTorch sparse matrix multiplication.
    if not membership_values.requires_grad:
        transposed_incidence = incidence_matrix.transpose(0, 1).coalesce()
        edge_features = torch.sparse.mm(
            transposed_incidence,
            scaled_node_features,
        )
        edge_features = edge_features * edge_scale[:, None]
        propagated_features = torch.sparse.mm(
            incidence_matrix,
            edge_features,
        )
        return propagated_features * node_scale[:, None]

    # H* values need gradients. Chunked indexed reductions avoid the large
    # allocation made by sparse COO backward on the full XuetangX graph.
    started_at = time.perf_counter()
    log_event(
        f"H* propagation started: shape={incidence_matrix.shape}, "
        f"nnz={incidence_matrix._nnz():,}, hidden={node_features.shape[1]}"
    )
    edge_features = weighted_index_add(
        scaled_node_features,
        membership_values,
        node_indices,
        edge_indices,
        edge_count,
    )
    edge_features = edge_features * edge_scale[:, None]
    propagated_features = weighted_index_add(
        edge_features,
        membership_values,
        edge_indices,
        node_indices,
        node_count,
    )
    propagated_features = propagated_features * node_scale[:, None]
    log_event(
        f"H* propagation completed in {time.perf_counter() - started_at:.1f}s"
    )
    return propagated_features


class HGSLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.layer2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias2 = nn.Parameter(torch.zeros(hidden_dim))

        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.membership_bias = nn.Parameter(torch.zeros(()))

        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def encode(self, node_features, incidence_matrix):
        # node_features: [num_nodes, input_dim]
        # incidence_matrix: [num_nodes, num_edges]
        hidden = self.layer1(node_features)
        hidden = hypergraph_propagation(hidden, incidence_matrix) + self.bias1
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        embeddings = self.layer2(hidden)
        embeddings = (
            hypergraph_propagation(embeddings, incidence_matrix) + self.bias2
        )
        return F.relu(embeddings)

    def forward(
        self,
        node_features,
        initial_incidence_matrix,
        edge_families,
        edge_sizes,
        random_generator,
        *,
        hsl=True,
        deterministic=False,
        refinement=None,
        node_groups=None,
        edge_groups=None,
    ):
        if refinement is None:
            refinement = {}

        # X: [num_nodes, input_dim]
        # H0: [num_nodes, num_edges]
        initial_hypergraph = to_torch_sparse(
            initial_incidence_matrix,
            node_features.device,
        )
        initial_embeddings = self.encode(node_features, initial_hypergraph)
        # Z0: [num_nodes, hidden_dim]

        if hsl:
            refined_hypergraph = refine_hypergraph(
                initial_embeddings,
                initial_incidence_matrix,
                edge_families,
                edge_sizes,
                self.node_projection,
                self.edge_projection,
                self.membership_bias,
                random_generator,
                deterministic=deterministic,
                node_groups=node_groups,
                edge_groups=edge_groups,
                **refinement,
            )
            # H*: [num_nodes, num_edges]
            refined_embeddings = self.encode(
                node_features,
                refined_hypergraph,
            )
        else:
            # HGNN baseline: no structure learning.
            refined_hypergraph = initial_hypergraph
            refined_embeddings = initial_embeddings

        # Z*: [num_nodes, hidden_dim]
        # logits: [num_nodes]
        logits = self.classifier(refined_embeddings).squeeze(-1)
        return {
            "logits": logits,
            "z0": initial_embeddings,
            "z_star": refined_embeddings,
            "h_star": refined_hypergraph,
        }


# Technical detail: differentiable message aggregation with bounded memory.
class WeightedIndexAdd(torch.autograd.Function):
    @staticmethod
    def forward(
        context,
        source,
        weights,
        source_indices,
        target_indices,
        target_count,
        chunk_size,
    ):
        output = source.new_zeros((target_count, source.shape[1]))
        for chunk_start in range(0, len(weights), chunk_size):
            chunk_stop = min(chunk_start + chunk_size, len(weights))
            selected_source = source.index_select(
                0,
                source_indices[chunk_start:chunk_stop],
            )
            messages = selected_source * weights[chunk_start:chunk_stop, None]
            output.index_add_(
                0,
                target_indices[chunk_start:chunk_stop],
                messages,
            )

        context.save_for_backward(
            source,
            weights,
            source_indices,
            target_indices,
        )
        context.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(context, output_gradient):
        source, weights, source_indices, target_indices = context.saved_tensors
        source_gradient = None
        weight_gradient = None
        if context.needs_input_grad[0]:
            source_gradient = torch.zeros_like(source)
        if context.needs_input_grad[1]:
            weight_gradient = torch.empty_like(weights)

        chunk_size = context.chunk_size
        for chunk_start in range(0, len(weights), chunk_size):
            chunk_stop = min(chunk_start + chunk_size, len(weights))
            selected_output_gradient = output_gradient.index_select(
                0,
                target_indices[chunk_start:chunk_stop],
            )

            if source_gradient is not None:
                source_messages = (
                    selected_output_gradient
                    * weights[chunk_start:chunk_stop, None]
                )
                source_gradient.index_add_(
                    0,
                    source_indices[chunk_start:chunk_stop],
                    source_messages,
                )

            if weight_gradient is not None:
                selected_source = source.index_select(
                    0,
                    source_indices[chunk_start:chunk_stop],
                )
                weight_gradient[chunk_start:chunk_stop] = (
                    selected_output_gradient * selected_source
                ).sum(dim=1)

        return source_gradient, weight_gradient, None, None, None, None


def weighted_index_add(
    source,
    weights,
    source_indices,
    target_indices,
    target_count,
):
    return WeightedIndexAdd.apply(
        source,
        weights,
        source_indices,
        target_indices,
        target_count,
        MESSAGE_CHUNK_SIZE,
    )
