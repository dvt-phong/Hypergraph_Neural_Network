# 6. Graph branch: an HGNN encoder 
# - HGNN https://github.com/iMoonLab/HGNN
# - HGNN+ https://github.com/iMoonLab/DeepHypergraph

import math
from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")
USER = config.EDGE_FAMILIES.index("user")
SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")


INITIAL_FAMILY_LOGIT = math.log(math.e - 1)


# Get graph from graph_data and apply scenarion for experiment
def prepare_graph(graph, device):
    node_ids = torch.as_tensor(graph["node_ids"], dtype=torch.int64, device=device)
    edge_ids = torch.as_tensor(graph["edge_ids"], dtype=torch.int64, device=device)
    edge_family = torch.as_tensor(graph["edge_family"], dtype=torch.int64, device=device)
    num_nodes = int(graph["num_nodes"])
    num_edges = len(edge_family)
    ones = torch.ones(len(node_ids), device=device)                         # h(v,e) = 1 for every membership
    return {
        "H": torch.sparse_coo_tensor(torch.stack([node_ids, edge_ids]), ones,
                                     (num_nodes, num_edges), check_invariants=True).coalesce(),
        "H_T": torch.sparse_coo_tensor(torch.stack([edge_ids, node_ids]), ones,
                                       (num_edges, num_nodes), check_invariants=True).coalesce(),
        "edge_size": torch.bincount(edge_ids, minlength=num_edges).float(),   # δ(e) = |e|
        "edge_family": edge_family,
        "self_loop": bool(graph["self_loop"]),
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

    def cache_train_states(self, x, graph):
        family_w = self.family_weights()
        w = family_w[graph["edge_family"]]                                # w_e
        node_degree = torch.sparse.mm(graph["H"], w.unsqueeze(1))        # d(u) = Σ_e h(u,e)·w_e, [N, 1]
        node_scale = node_degree.clamp_min(1e-6).rsqrt()                 # d(u)^-1/2
        edge_scale = (w / graph["edge_size"]).unsqueeze(1)               # w_e/δ(e), [E] -> [E, 1]

        send1 = self.layer1(x) * node_scale                              # d(u)^-1/2·(x_u·Θ1 + b1)
        S1 = torch.sparse.mm(graph["H_T"], send1)                        # S1_e = Σ_{u∈e} d(u)^-1/2·(x_u·Θ1 + b1)
        cache = {
            "family_w": family_w,
            "edge_w": w,
            "edge_size": graph["edge_size"],
            "self_loop": graph["self_loop"],
            "send1": send1,
            "S1": S1,
        }
        if self.layer2 is not None:
            node_sum = torch.sparse.mm(graph["H"], S1 * edge_scale)      # Σ_e h(v,e)·(w_e/δ(e))·S1_e
            z1 = F.relu(node_sum * node_scale)                           # Z1 = ReLU(G·(X·Θ1 + b1))
            cache["send2"] = self.layer2(z1) * node_scale                # d(u)^-1/2·(Z1_u·Θ2 + b2)
            cache["S2"] = torch.sparse.mm(graph["H_T"], cache["send2"])  # S2_e = Σ_{u∈e} d(u)^-1/2·(Z1_u·Θ2 + b2)
        return cache

    def forward_targets(self, x, rows, edges, single_user, cache):
        w_user = cache["family_w"][USER]
        if cache["self_loop"]:
            w_self = cache["family_w"][SELF_LOOP]
        else:
            w_self = 0.0                                                  # no self-loop: no {t} term
        w = cache["edge_w"][edges]                                        # w_e, e ∈ E(t) ∩ H0
        edge_scale = (w / (cache["edge_size"][edges] + 1)).unsqueeze(1)   # w_e/(δ(e) + 1), [K] -> [K, 1]
        has_single = single_user >= 0
        single_nodes = single_user[has_single]

        # d(t) = Σ_{e∈E(t)∩H0} w_e + w_user·[t has {t,u}] + w_self
        degree = torch.zeros(len(x), device=x.device)
        degree.index_add_(0, rows, w)                                     # Σ_{e∈H0} w_e
        degree = degree + w_user * has_single + w_self                    # d(t) = Σ_{e∈E(t)} w_e
        node_scale = degree.clamp_min(1e-6).rsqrt().unsqueeze(1)          # d(t)^-1/2, [B] -> [B, 1]

        # Each layer, with own = x_t·Θ + b and S, send of the train nodes from step A:
        #   z_t = ReLU( d(t)^-1/2 · [ Σ_{e∈E(t)∩H0} (w_e/(δ(e)+1))·(S_e + o_t)
        #                            + (w_user/2)·(send_u + o_t) + w_self·o_t ] ),  o_t = d(t)^-1/2·own
        # Layer 1
        own = self.layer1(x) * node_scale                                 # d(t)^-1/2·(x_t·Θ1 + b1)
        total = torch.zeros_like(own)
        total.index_add_(0, rows, edge_scale * (cache["S1"][edges] + own[rows]))   # Σ_{e∈H0} (w_e/(δ(e)+1))·(S1_e + own)
        total[has_single] += (w_user / 2) * (cache["send1"][single_nodes] + own[has_single])   # {t,u}: (w_user/2)·(send_u + own)
        total = total + w_self * own                                      # {t}: (w_self/1)·own  (0 without self-loops)
        z = F.relu(total * node_scale)                                    # z1_t
        if self.layer2 is None:
            return z                                                      # 1 layer: z_g = z1_t

        # Layer 2: the same with Θ2, b2, S2, send2
        own = self.layer2(z) * node_scale                                 # d(t)^-1/2·(z1_t·Θ2 + b2)
        total = torch.zeros_like(own)
        total.index_add_(0, rows, edge_scale * (cache["S2"][edges] + own[rows]))   # Σ_{e∈H0} (w_e/(δ(e)+1))·(S2_e + own)
        total[has_single] += (w_user / 2) * (cache["send2"][single_nodes] + own[has_single])   # {t,u}: (w_user/2)·(send_u + own)
        total = total + w_self * own                                      # {t}: (w_self/1)·own  (0 without self-loops)
        return F.relu(total * node_scale)                                 # z_g of t
