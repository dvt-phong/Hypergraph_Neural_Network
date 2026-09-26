# 5. HGNN encoder and the HSL dropout model.
#
# HGNN layer (Feng et al., AAAI 2019):
#     X' = Dv^-1/2  H  De^-1  H^T  Dv^-1/2  (X Θ + b)
#
# HSL model (Cai et al., IJCAI 2022), one shared HGNN encoder used twice:
#     Z0     = HGNN(X, H0)                         original structure
#     H*     = Me ⊙ Mv ⊙ (H0 + ΔH) + I             learned structure (6_hsl.py)
#     Z*     = HGNN(X, H*)                         same weights as for Z0
#     logits = Linear(Z*)
#
# H* is described by memberships (node_ids, edge_ids) plus one weight per
# membership: 1 for H0, and a 0/1 learned mask for H*.

from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

config = import_module("0_config")
hsl_module = import_module("6_hsl")
StructureLearner = hsl_module.StructureLearner


# Move a NumPy graph from 4_hypergraph.py to PyTorch tensors on `device`.
def graph_to_device(graph, device):
    tensors = {"num_nodes": int(graph["num_nodes"])}
    for name in ("node_ids", "edge_ids", "edge_family", "candidate_edge_ids", "candidate_node_ids"):
        tensors[name] = torch.as_tensor(graph[name], dtype=torch.int64, device=device)
    tensors["num_edges"] = len(tensors["edge_family"])
    return tensors


def _weighted_sum_chunk(source, weights, source_ids, target_ids, target_count):
    messages = source[source_ids] * weights[:, None]
    output = source.new_zeros(target_count, source.shape[1])
    return output.index_add_(0, target_ids, messages)


# output[t] = Σ weights[m] * source[source_ids[m]] over memberships m with
# target_ids[m] = t. Used for both directions: nodes -> hyperedges and back.
# Memberships are processed in chunks; `checkpoint` recomputes a chunk during
# backward instead of keeping every message in memory.
def weighted_sum(source, weights, source_ids, target_ids, target_count):
    output = source.new_zeros(target_count, source.shape[1])
    for start in range(0, len(weights), config.MEMBERSHIP_CHUNK_SIZE):
        part = slice(start, start + config.MEMBERSHIP_CHUNK_SIZE)
        output = output + checkpoint(
            _weighted_sum_chunk,
            source, weights[part], source_ids[part], target_ids[part], target_count,
            use_reentrant=False,
        )
    return output


# One HGNN propagation Dv^-1/2 H De^-1 H^T Dv^-1/2 x with membership weights.
# Degrees are read from the (0/1) weights without gradient, so an emptied
# hyperedge simply sends zero instead of dividing by zero.
def hgnn_propagate(x, graph, weights):
    node_ids, edge_ids = graph["node_ids"], graph["edge_ids"]
    constant_weights = weights.detach()
    node_degree = x.new_zeros(graph["num_nodes"]).index_add_(0, node_ids, constant_weights)
    edge_degree = x.new_zeros(graph["num_edges"]).index_add_(0, edge_ids, constant_weights)
    node_scale = node_degree.clamp_min(1.0).rsqrt()[:, None]
    edge_scale = edge_degree.clamp_min(1.0).reciprocal()[:, None]

    edge_x = weighted_sum(x * node_scale, weights, node_ids, edge_ids, graph["num_edges"])
    node_x = weighted_sum(edge_x * edge_scale, weights, edge_ids, node_ids, graph["num_nodes"])
    return node_x * node_scale


# Hyperedge representation h_e = mean of its members' embeddings in H0.
def hyperedge_means(z, graph):
    ones = z.new_ones(len(graph["node_ids"]))
    edge_size = z.new_zeros(graph["num_edges"]).index_add_(0, graph["edge_ids"], ones)
    total = weighted_sum(z, ones, graph["node_ids"], graph["edge_ids"], graph["num_edges"])
    return total / edge_size[:, None]


class HSLModel(nn.Module):
    # hsl_options=None gives the plain HGNN baseline (H* = H0, Z* = Z0).
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5, hsl_options=None):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.structure_learner = None
        if hsl_options is not None:
            self.structure_learner = StructureLearner(hidden_dim, **hsl_options)

    # Two HGNN layers: Linear -> propagate -> ReLU -> Dropout -> Linear -> propagate -> ReLU.
    def encode(self, x, graph, weights):
        hidden = F.relu(hgnn_propagate(self.layer1(x), graph, weights))
        hidden = F.dropout(hidden, self.dropout, self.training)
        return F.relu(hgnn_propagate(self.layer2(hidden), graph, weights))

    def forward(self, x, graph):
        # x: [num_nodes, input_dim]; graph: output of graph_to_device.
        h0_weights = x.new_ones(len(graph["node_ids"]))
        z0 = self.encode(x, graph, h0_weights)  # [num_nodes, hidden_dim]

        if self.structure_learner is None:
            return {"logits": self.classifier(z0).squeeze(-1), "z0": z0, "z_star": z0, "structure": {}}

        edge_representations = hyperedge_means(z0, graph)
        refined_graph, refined_weights, structure = self.structure_learner(
            z0, edge_representations, graph
        )
        z_star = self.encode(x, refined_graph, refined_weights)  # [num_nodes, hidden_dim]
        return {
            "logits": self.classifier(z_star).squeeze(-1),  # [num_nodes]
            "z0": z0,
            "z_star": z_star,
            "structure": structure,
        }
