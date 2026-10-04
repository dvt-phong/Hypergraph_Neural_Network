# 5. Dropout prediction model: 2-layer HGNN encoder (graph branch), optional MLP
#    branch on the node's own features (skip), one learned weight per hyperedge family.
# Tham khảo từ project/bài báo:
# - HGNN, AAAI 2019 (Feng et al.): https://doi.org/10.1609/aaai.v33i01.33013558
#   Code: https://github.com/iMoonLab/HGNN
# - HGNN+, IEEE TPAMI 2023 (Gao et al.): hyperedge groups with learned weights
#   Code: https://github.com/iMoonLab/DeepHypergraph
# - UniGNN / UniGCNII, IJCAI 2021 (Huang & Yang): skip (initial residual) idea

import math
from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")

# softplus(INITIAL_FAMILY_LOGIT) = 1: family weights start at W = I.
INITIAL_FAMILY_LOGIT = math.log(math.e - 1)


# Turn a graph from 4_hypergraph.py into sparse incidence matrices.
# With causal, node v only receives from hyperedge e when no member of e started
# its course later than v.
# Input:  NumPy graph dict (num_nodes, node_ids, edge_ids, edge_family, node_start),
#         device, causal.
# Output: dict H_T [E, N] (node -> hyperedge), H_recv [N, E] (hyperedge -> node,
#         causal mask applied), edge_size [E], edge_family [E].
def prepare_graph(graph, device, causal=False):
    node_ids = torch.as_tensor(graph["node_ids"], dtype=torch.int64, device=device)
    edge_ids = torch.as_tensor(graph["edge_ids"], dtype=torch.int64, device=device)
    edge_family = torch.as_tensor(graph["edge_family"], dtype=torch.int64, device=device)
    num_nodes, num_edges = int(graph["num_nodes"]), len(edge_family)

    ones = torch.ones(len(node_ids), device=device)
    receive = ones
    if causal:
        start = torch.as_tensor(graph["node_start"], dtype=torch.int64, device=device)[node_ids]
        latest = torch.full((num_edges,), torch.iinfo(torch.int64).min, dtype=torch.int64, device=device)
        latest = latest.scatter_reduce(0, edge_ids, start, reduce="amax")  # latest start per hyperedge
        receive = (start >= latest[edge_ids]).float()

    return {
        "H_T": torch.sparse_coo_tensor(torch.stack([edge_ids, node_ids]), ones,
                                       (num_edges, num_nodes), check_invariants=True).coalesce(),
        "H_recv": torch.sparse_coo_tensor(torch.stack([node_ids, edge_ids]), receive,
                                          (num_nodes, num_edges), check_invariants=True).coalesce(),
        "edge_size": torch.bincount(edge_ids, minlength=num_edges).clamp_min(1).float(),
        "edge_family": edge_family,
    }


# Main model (M0): skip_connection=True, family_weights=True, graph prepared with causal=True.
class DropoutModel(nn.Module):
    # Input:  input_dim, hidden_dim, dropout; skip_connection: add the MLP(X)
    #         branch; family_weights: learn one weight per hyperedge family.
    def __init__(self, input_dim, hidden_dim=128, dropout=0.5, *,
                 skip_connection=False, family_weights=False):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

        self.self_encoder = None
        if skip_connection:
            # MLP(X): the HGNN encoder without propagation.
            self.self_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
        self.classifier = nn.Linear(hidden_dim * (2 if skip_connection else 1), 1)

        self.family_logits = None
        if family_weights:
            self.family_logits = nn.Parameter(
                torch.full((len(config.EDGE_FAMILIES),), INITIAL_FAMILY_LOGIT)
            )

    # Diagonal of W: the weight of each hyperedge's family (all 1 without family weights).
    # Input:  graph from prepare_graph.
    # Output: [E].
    def edge_weight(self, graph):
        if self.family_logits is None:
            return torch.ones(len(graph["edge_family"]), device=graph["edge_family"].device)
        return F.softplus(self.family_logits)[graph["edge_family"]]

    # One HGNN propagation X' = Dv^-1/2 H_recv W De^-1 H^T Dv^-1/2 X:
    # node -> hyperedge (mean of members, times w_e), then hyperedge -> node.
    # Input:  x [N, D], graph from prepare_graph, w [E] from edge_weight.
    # Output: [N, D].
    def propagate(self, x, graph, w):
        node_degree = torch.sparse.mm(graph["H_recv"], w[:, None])     # Dv [N, 1]
        node_scale = node_degree.clamp_min(1e-6).rsqrt()
        edge_x = torch.sparse.mm(graph["H_T"], x * node_scale)          # node -> hyperedge
        edge_x = edge_x * (w / graph["edge_size"])[:, None]             # W De^-1
        return torch.sparse.mm(graph["H_recv"], edge_x) * node_scale    # hyperedge -> node

    # Input:  x [N, input_dim], graph from prepare_graph.
    # Output: dict logits [N]; with the skip branch also logit_graph and
    #         logit_self, where logits = logit_graph + logit_self + bias.
    def forward(self, x, graph):
        w = self.edge_weight(graph)
        h = F.relu(self.propagate(self.layer1(x), graph, w))
        h = F.dropout(h, self.dropout, self.training)
        z = F.relu(self.propagate(self.layer2(h), graph, w))            # graph branch
        if self.self_encoder is None:
            return {"logits": self.classifier(z).squeeze(-1)}

        z_self = self.self_encoder(x)                                   # MLP branch
        logits = self.classifier(torch.cat([z, z_self], dim=1)).squeeze(-1)
        weight = self.classifier.weight[0]
        return {
            "logits": logits,
            "logit_graph": z @ weight[:z.shape[1]],
            "logit_self": z_self @ weight[z.shape[1]:],
        }

    # Learned family weights for the training log.
    # Output: dict w_<family> (empty without family weights).
    @torch.no_grad()
    def weight_summary(self):
        if self.family_logits is None:
            return {}
        weights = F.softplus(self.family_logits).tolist()
        return {f"w_{name}": weight for name, weight in zip(config.EDGE_FAMILIES, weights)}
