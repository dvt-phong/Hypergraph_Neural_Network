# 6. Graph branch: an HGNN encoder 
# - HGNN https://github.com/iMoonLab/HGNN
# - HGNN+ https://github.com/iMoonLab/DeepHypergraph

import math
from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")


INITIAL_FAMILY_LOGIT = math.log(math.e - 1)


# Get graph from graph_data and apply scenarion for experiment
def prepare_graph(graph, device):
    node_ids = torch.as_tensor(graph["node_ids"], dtype=torch.int64, device=device)
    edge_ids = torch.as_tensor(graph["edge_ids"], dtype=torch.int64, device=device)
    edge_family = torch.as_tensor(graph["edge_family"], dtype=torch.int64, device=device)
    num_nodes = int(graph["num_nodes"])
    num_edges = len(edge_family)
    ones = torch.ones(len(node_ids), device=device)                       
    return {
        "H": torch.sparse_coo_tensor(torch.stack([node_ids, edge_ids]), ones,
                                     (num_nodes, num_edges), check_invariants=True).coalesce(),
        "H_T": torch.sparse_coo_tensor(torch.stack([edge_ids, node_ids]), ones,
                                       (num_edges, num_nodes), check_invariants=True).coalesce(),
        "edge_size": torch.bincount(edge_ids, minlength=num_edges).float(),  
        "edge_family": edge_family,
    }


# HGNN
class HGNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, layers=2, learn_w=True):
        super().__init__()
        if layers not in (1, 2):
            raise ValueError("layers must be 1 or 2")
        self.layer1 = nn.Linear(input_dim, hidden_dim)                       # Θ1, b1
        if layers == 2:
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)                  # Θ2, b2 (2 layers only)
        else:
            self.layer2 = None
        self.dropout = dropout
        self.learn_w = learn_w
        logits = torch.full((len(config.EDGE_FAMILIES),), INITIAL_FAMILY_LOGIT)   # θ_f, one per hyperedge family
        if learn_w:
            self.family_logits = nn.Parameter(logits)                        # learned: W = diag(softplus(θ_f(e)))
        else:
            self.register_buffer("family_logits", logits)                    # fixed: softplus(θ) = 1, so W = I

    # Weight of every hyperedge family: w_f = softplus(θ_f) > 0.
    def family_weights(self):
        return F.softplus(self.family_logits)

    def propagate(self, x, graph, w):
        # x'_v = d(v)^-1/2 · Σ_e h(v,e) · (w_e/δ(e)) · Σ_{u∈e} d(u)^-1/2 · x_u
        node_degree = torch.sparse.mm(graph["H"], w.unsqueeze(1))      # d(v) = Σ_e h(v,e)·w_e, [N, 1]
        node_scale = node_degree.clamp_min(1e-6).rsqrt()                 # d(v)^-1/2
        edge_x = torch.sparse.mm(graph["H_T"], x * node_scale)           # Σ_{u∈e} d(u)^-1/2·x_u
        edge_scale = (w / graph["edge_size"]).unsqueeze(1)               # w_e/δ(e), [E] -> [E, 1]
        edge_x = edge_x * edge_scale                                     # m_e = (w_e/δ(e))·Σ_{u∈e} d(u)^-1/2·x_u
        return torch.sparse.mm(graph["H"], edge_x) * node_scale          # x'_v = d(v)^-1/2·Σ_e h(v,e)·m_e

    def forward(self, x, graph):
        w = self.family_weights()[graph["edge_family"]]                   # w_e = w_f(e)
        h = self.layer1(x)                                               # X·Θ1 + b1
        z = F.relu(self.propagate(h, graph, w))                          # Z1 = ReLU(G·(X·Θ1 + b1))
        if self.layer2 is None:
            return z                                                     # 1 layer: Z_g = Z1
        z = F.dropout(z, self.dropout, self.training)                    # only while training
        h = self.layer2(z)                                               # Z1·Θ2 + b2
        return F.relu(self.propagate(h, graph, w))                       # Z_g = ReLU(G·(Z1·Θ2 + b2))
