"""HGNN propagation and the two-pass HGSL dropout model."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hsl import refine_hypergraph


def to_torch_sparse(matrix, device):
    coo = matrix.tocoo(copy=False)
    indices = torch.as_tensor(
        np.stack((coo.row, coo.col)),
        dtype=torch.int64,
        device=device,
    )
    values = torch.as_tensor(coo.data.astype(np.float32), device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        coo.shape,
        check_invariants=False,
    ).coalesce()


def _hypergraph_operator(h):
    h = h.coalesce()
    node_indices, edge_indices = h.indices()
    values = h.values()

    node_degree = torch.zeros(h.shape[0], device=values.device)
    node_degree = node_degree.scatter_add(0, node_indices, values)
    edge_degree = torch.zeros(h.shape[1], device=values.device)
    edge_degree = edge_degree.scatter_add(0, edge_indices, values)
    if torch.any(node_degree <= 0) or torch.any(edge_degree <= 0):
        raise ValueError("H has an isolated node or empty hyperedge")

    h_transpose = h.transpose(0, 1).coalesce()
    node_scale = node_degree.rsqrt()
    edge_scale = edge_degree.reciprocal()
    return h, h_transpose, node_scale, edge_scale


def _propagate(operator, x):
    h, h_transpose, node_scale, edge_scale = operator
    scaled_nodes = x * node_scale[:, None]
    edge_messages = torch.sparse.mm(h_transpose, scaled_nodes)
    scaled_edges = edge_messages * edge_scale[:, None]
    node_messages = torch.sparse.mm(h, scaled_edges)
    return node_messages * node_scale[:, None]


class HGSLModel(nn.Module):
    """Shared HGNN encoder, learned memberships, and dropout classifier."""

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

    def encode(self, x, h):
        operator = _hypergraph_operator(h)
        hidden = _propagate(operator, self.layer1(x)) + self.bias1
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        embedding = _propagate(operator, self.layer2(hidden)) + self.bias2
        return F.relu(embedding)

    def forward(
        self,
        x,
        h0,
        families,
        sizes,
        rng,
        *,
        hsl=True,
        deterministic=False,
        refinement=None,
        node_groups=None,
        edge_groups=None,
    ):
        refinement = refinement or {}
        initial_hypergraph = to_torch_sparse(h0, x.device)
        z0 = self.encode(x, initial_hypergraph)

        if hsl:
            refined_hypergraph = refine_hypergraph(
                z0,
                h0,
                families,
                sizes,
                self.node_projection,
                self.edge_projection,
                self.membership_bias,
                rng,
                deterministic=deterministic,
                node_groups=node_groups,
                edge_groups=edge_groups,
                **refinement,
            )
            z_star = self.encode(x, refined_hypergraph)
        else:
            refined_hypergraph = initial_hypergraph
            z_star = z0

        logits = self.classifier(z_star).squeeze(-1)
        return {
            "logits": logits,
            "z0": z0,
            "z_star": z_star,
            "h_star": refined_hypergraph,
        }
