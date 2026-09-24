# 5. Define HGNN propagation and the two-pass HGSL dropout model.

import numpy as np
import torch
from importlib import import_module
from torch import nn
from torch.nn import functional as F

hsl_module = import_module("6_hsl")
refine_hypergraph = hsl_module.refine_hypergraph


# Convert a SciPy sparse matrix to a coalesced PyTorch sparse tensor.
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


# Precompute incidence tensors and degree scales for HGNN propagation.
def _hypergraph_operator(incidence_matrix):
    incidence_matrix = incidence_matrix.coalesce()
    node_indices, edge_indices = incidence_matrix.indices()
    membership_values = incidence_matrix.values()

    node_degree = torch.zeros(
        incidence_matrix.shape[0],
        device=membership_values.device,
    )
    node_degree = node_degree.scatter_add(0, node_indices, membership_values)
    edge_degree = torch.zeros(
        incidence_matrix.shape[1],
        device=membership_values.device,
    )
    edge_degree = edge_degree.scatter_add(0, edge_indices, membership_values)
    if torch.any(node_degree <= 0) or torch.any(edge_degree <= 0):
        raise ValueError("H has an isolated node or empty hyperedge")

    transposed_incidence = incidence_matrix.transpose(0, 1).coalesce()
    node_scale = node_degree.rsqrt()
    edge_scale = edge_degree.reciprocal()
    return incidence_matrix, transposed_incidence, node_scale, edge_scale


# Propagate node features to hyperedges and back to nodes.
def _propagate(operator, node_features):
    incidence_matrix, transposed_incidence, node_scale, edge_scale = operator
    scaled_nodes = node_features * node_scale[:, None]
    edge_messages = torch.sparse.mm(transposed_incidence, scaled_nodes)
    scaled_edges = edge_messages * edge_scale[:, None]
    node_messages = torch.sparse.mm(incidence_matrix, scaled_edges)
    return node_messages * node_scale[:, None]


# Combine a shared HGNN encoder, learned memberships, and dropout classifier.
class HGSLModel(nn.Module):
    # Initialize encoder, membership scorer, and classifier parameters.
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

    # Encode node features over one sparse incidence matrix.
    def encode(self, node_features, incidence_matrix):
        operator = _hypergraph_operator(incidence_matrix)
        hidden = _propagate(
            operator,
            self.layer1(node_features),
        ) + self.bias1
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        embedding = _propagate(operator, self.layer2(hidden)) + self.bias2
        return F.relu(embedding)

    # Run initial encoding, optional structure learning, and classification.
    def forward(
        self,
        node_features,
        initial_incidence_matrix,
        families,
        sizes,
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
        initial_hypergraph = to_torch_sparse(
            initial_incidence_matrix,
            node_features.device,
        )
        initial_node_embeddings = self.encode(node_features, initial_hypergraph)

        if hsl:
            refined_hypergraph = refine_hypergraph(
                initial_node_embeddings,
                initial_incidence_matrix,
                families,
                sizes,
                self.node_projection,
                self.edge_projection,
                self.membership_bias,
                random_generator,
                deterministic=deterministic,
                node_groups=node_groups,
                edge_groups=edge_groups,
                **refinement,
            )
            refined_node_embeddings = self.encode(
                node_features,
                refined_hypergraph,
            )
        else:
            refined_hypergraph = initial_hypergraph
            refined_node_embeddings = initial_node_embeddings

        logits = self.classifier(refined_node_embeddings).squeeze(-1)
        return {
            "logits": logits,
            "z0": initial_node_embeddings,
            "z_star": refined_node_embeddings,
            "h_star": refined_hypergraph,
        }
