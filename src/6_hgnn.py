# 6. Graph branch: an HGNN encoder with 1 or 2 layers, as HGNN_embedding of the
#    original code (conv1 -> ReLU -> dropout -> conv2 -> ReLU), with one learned
#    weight per hyperedge family instead of the fixed W = I of the original code.
# Tham khảo từ project/bài báo:
# - HGNN, AAAI 2019 (Feng et al.), Eq. 10: https://doi.org/10.1609/aaai.v33i01.33013558
#   Code: https://github.com/iMoonLab/HGNN (models/layers.py: HGNN_conv, HGNN_embedding;
#   utils/hypergraph_utils.py: _generate_G_from_H)
# - HGNN+, IEEE TPAMI 2023 (Gao et al.): hyperedge groups with their own weights
#   Code: https://github.com/iMoonLab/DeepHypergraph
#
# Notation (N train nodes, E hyperedges kept by the scenario, self-loops included if kept):
#   H ∈ {0,1}^{N×E}  h(v,e) = 1 when node v belongs to hyperedge e
#   W = diag(w_e)     w_e = softplus(θ_f(e)), one θ per family f: course, object, user, self_loop
#   De = diag(δ(e))   δ(e) = |e|, the number of members of e
#   Dv = diag(d(v))   d(v) = Σ_e h(v,e)·w_e   (> 0: every node is in its Course hyperedge
#                                               or in its self-loop, in every scenario)
#   G = Dv^-1/2 · H · W · De^-1 · Hᵀ · Dv^-1/2                   (Feng et al., 2019, Eq. 10)
#   1 layer:  Z_g = ReLU( G · (X·Θ1 + b1) )
#   2 layers: Z1  = ReLU( G · (X·Θ1 + b1) ),  Z_g = ReLU( G · (Dropout(Z1)·Θ2 + b2) )
# The bias is added before G, as in HGNN_conv (x = x·Θ + b, then G·x).

import math
from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")
USER = config.EDGE_FAMILIES.index("user")
SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")

# softplus(INITIAL_FAMILY_LOGIT) = 1: training starts from W = I, as in the original code.
INITIAL_FAMILY_LOGIT = math.log(math.e - 1)


# Turn a graph from 5_graph_data.apply_scenario into sparse incidence matrices.
# Input:  graph dict (num_nodes, node_ids, edge_ids, edge_family, self_loop), device.
# Output: dict H [N, E] (hyperedge -> node), H_T [E, N] (node -> hyperedge),
#         edge_size [E] = δ(e), edge_family [E], self_loop (are self-loops in the graph?).
def prepare_graph(graph, device):
    node_ids = torch.as_tensor(graph["node_ids"], dtype=torch.int64, device=device)
    edge_ids = torch.as_tensor(graph["edge_ids"], dtype=torch.int64, device=device)
    edge_family = torch.as_tensor(graph["edge_family"], dtype=torch.int64, device=device)
    num_nodes, num_edges = int(graph["num_nodes"]), len(edge_family)
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


class HGNNEncoder(nn.Module):
    # Input:  input_dim (columns of X), hidden_dim, dropout rate, layers (1 or 2).
    def __init__(self, input_dim, hidden_dim, dropout, layers=2):
        super().__init__()
        if layers not in (1, 2):
            raise ValueError("layers must be 1 or 2")
        self.layer1 = nn.Linear(input_dim, hidden_dim)                       # Θ1, b1
        self.layer2 = nn.Linear(hidden_dim, hidden_dim) if layers == 2 else None   # Θ2, b2 (2 layers only)
        self.dropout = dropout
        self.family_logits = nn.Parameter(                                   # θ_f, one per hyperedge family
            torch.full((len(config.EDGE_FAMILIES),), INITIAL_FAMILY_LOGIT)
        )

    # Output: [families], w_f = softplus(θ_f) > 0.
    def family_weights(self):
        return F.softplus(self.family_logits)

    # One propagation G·x, written node by node.
    # Input:  x [N, D], graph from prepare_graph, w [E] = w_e of every hyperedge.
    # Output: [N, D].
    def propagate(self, x, graph, w):
        node_degree = torch.sparse.mm(graph["H"], w[:, None])          # d(v) = Σ_e h(v,e)·w_e
        node_scale = node_degree.clamp_min(1e-6).rsqrt()                 # d(v)^-1/2
        edge_x = torch.sparse.mm(graph["H_T"], x * node_scale)           # Σ_{u∈e} d(u)^-1/2·x_u
        edge_x = edge_x * (w / graph["edge_size"])[:, None]              # m_e = (w_e/δ(e))·Σ_{u∈e} d(u)^-1/2·x_u
        return torch.sparse.mm(graph["H"], edge_x) * node_scale          # x'_v = d(v)^-1/2·Σ_e h(v,e)·m_e

    # Training: the HGNN layers on the whole train hypergraph (full batch).
    # Input:  x [N, D], graph from prepare_graph.
    # Output: Z_g [N, hidden].
    def forward(self, x, graph):
        w = self.family_weights()[graph["edge_family"]]                   # w_e = w_f(e)
        z = F.relu(self.propagate(self.layer1(x), graph, w))             # Z1 = ReLU(G·(X·Θ1 + b1))
        if self.layer2 is None:
            return z                                                     # 1 layer: Z_g = Z1
        z = F.dropout(z, self.dropout, self.training)                    # only while training
        return F.relu(self.propagate(self.layer2(z), graph, w))          # Z_g = ReLU(G·(Z1·Θ2 + b2))

    # Evaluation, step A: the train-node quantities a new target reads, computed once
    # per evaluation on H0 (call under model.eval() and torch.no_grad()).
    # Input:  x [N, D] train features, graph from prepare_graph.
    # Output: dict family_w [families], edge_w [E], edge_size [E], self_loop,
    #         send1 [N, hidden]: what each train node sends at layer 1,
    #         S1 [E, hidden]: what each H0 hyperedge collects at layer 1,
    #         send2 / S2: the same at layer 2 (2 layers only).
    def cache_train_states(self, x, graph):
        family_w = self.family_weights()
        w = family_w[graph["edge_family"]]                                # w_e
        node_scale = torch.sparse.mm(graph["H"], w[:, None]).clamp_min(1e-6).rsqrt()   # d(u)^-1/2
        edge_scale = (w / graph["edge_size"])[:, None]                   # w_e/δ(e)

        send1 = self.layer1(x) * node_scale                              # d(u)^-1/2·(x_u·Θ1 + b1)
        S1 = torch.sparse.mm(graph["H_T"], send1)                        # S1_e = Σ_{u∈e} d(u)^-1/2·(x_u·Θ1 + b1)
        cache = {"family_w": family_w, "edge_w": w, "edge_size": graph["edge_size"],
                 "self_loop": graph["self_loop"], "send1": send1, "S1": S1}
        if self.layer2 is not None:
            z1 = F.relu(torch.sparse.mm(graph["H"], S1 * edge_scale) * node_scale)   # Z1 = ReLU(G·(X·Θ1 + b1))
            cache["send2"] = self.layer2(z1) * node_scale                # d(u)^-1/2·(Z1_u·Θ2 + b2)
            cache["S2"] = torch.sparse.mm(graph["H_T"], cache["send2"])  # S2_e = Σ_{u∈e} d(u)^-1/2·(Z1_u·Θ2 + b2)
        return cache

    # Evaluation, step B: the HGNN layers for a batch of new targets. Target t joins its
    # hyperedges E(t); the train nodes keep their H0 states (step A). Each H0 hyperedge
    # e ∈ E(t) then has δ(e) + 1 members; {t, u} (single_user) has 2; the self-loop {t},
    # when the scenario keeps self-loops, has 1.
    # Input:  x [B, D] target features,
    #         rows [K], edges [K]: target t = rows[k] joins H0 hyperedge edges[k],
    #         single_user [B]: train node u of the hyperedge {t, u}, or -1,
    #         cache from cache_train_states.
    # Output: z_g [B, hidden].
    def forward_targets(self, x, rows, edges, single_user, cache):
        w_user = cache["family_w"][USER]
        w_self = cache["family_w"][SELF_LOOP] if cache["self_loop"] else 0.0   # no self-loop: no {t} term
        w = cache["edge_w"][edges]                                        # w_e, e ∈ E(t) ∩ H0
        edge_scale = (w / (cache["edge_size"][edges] + 1))[:, None]      # w_e/(δ(e) + 1)
        has_single = single_user >= 0
        single_nodes = single_user[has_single]

        degree = torch.zeros(len(x), device=x.device).index_add_(0, rows, w)   # Σ_{e∈H0} w_e
        degree = degree + w_user * has_single + w_self                    # d(t) = Σ_{e∈E(t)} w_e
        node_scale = degree.clamp_min(1e-6).rsqrt()[:, None]              # d(t)^-1/2

        # One layer for the targets; own = x_t·Θ + b, S / send from step A.
        def layer(own, S, send):
            own = own * node_scale                                        # d(t)^-1/2·(x_t·Θ + b)
            total = torch.zeros_like(own).index_add_(                     # Σ_{e∈H0} (w_e/(δ(e)+1))·(S_e + own)
                0, rows, edge_scale * (S[edges] + own[rows]))
            total[has_single] += (w_user / 2) * (send[single_nodes] + own[has_single])   # {t,u}: (w_user/2)·(send_u + own)
            total = total + w_self * own                                  # {t}: (w_self/1)·own  (0 without self-loops)
            return F.relu(total * node_scale)                             # ReLU(d(t)^-1/2·Σ_{e∈E(t)} m_e)

        z = layer(self.layer1(x), cache["S1"], cache["send1"])           # layer 1: z1_t
        if self.layer2 is None:
            return z                                                     # 1 layer: z_g = z1_t
        return layer(self.layer2(z), cache["S2"], cache["send2"])        # layer 2: z_g of t
