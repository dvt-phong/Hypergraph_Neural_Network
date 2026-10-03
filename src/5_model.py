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

import math
from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

config = import_module("0_config")
hsl_module = import_module("6_hsl")
StructureLearner = hsl_module.StructureLearner
SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")


# Move a NumPy graph from 4_hypergraph.py to PyTorch tensors on `device`.
def graph_to_device(graph, device):
    tensors = {"num_nodes": int(graph["num_nodes"])}
    for name in ("node_ids", "edge_ids", "edge_family", "candidate_edge_ids", "candidate_node_ids",
                 "node_start"):
        tensors[name] = torch.as_tensor(graph[name], dtype=torch.int64, device=device)
    tensors["num_edges"] = len(tensors["edge_family"])
    return tensors


def weighted_sum_chunk(source, weights, source_ids, target_ids, target_count):
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
            weighted_sum_chunk,
            source, weights[part], source_ids[part], target_ids[part], target_count,
            use_reentrant=False,
        )
    return output


# 1 for memberships (v, e) where v may receive from e: no active member of e
# started its course later than v (option causal).
@torch.no_grad()
def causal_receive(graph, weights):
    member_start = graph["node_start"][graph["node_ids"]]
    active = weights > 0
    latest = torch.full((graph["num_edges"],), torch.iinfo(torch.int64).min,
                        dtype=torch.int64, device=member_start.device)
    latest = latest.scatter_reduce(0, graph["edge_ids"][active], member_start[active], reduce="amax")
    return (member_start >= latest[graph["edge_ids"]]).to(weights.dtype)


# One HGNN propagation Dv^-1/2 H W De^-1 H^T Dv^-1/2 x with membership weights.
# Degrees are read from the (0/1) weights without gradient, so an emptied
# hyperedge simply sends zero instead of dividing by zero. edge_weight is the
# diagonal of W (None = identity); it enters Dv but not De, as in HGNN.
# receive (None = all 1) masks only the hyperedge -> node step and Dv: every
# member still counts in De and sends to the hyperedge.
def hgnn_propagate(x, graph, weights, edge_weight=None, receive=None):
    node_ids, edge_ids = graph["node_ids"], graph["edge_ids"]
    constant_weights = weights.detach()
    receive_weights = weights if receive is None else weights * receive
    node_weights = constant_weights if receive is None else constant_weights * receive
    if edge_weight is not None:
        node_weights = node_weights * edge_weight[edge_ids]
    node_degree = x.new_zeros(graph["num_nodes"]).index_add_(0, node_ids, node_weights)
    edge_degree = x.new_zeros(graph["num_edges"]).index_add_(0, edge_ids, constant_weights)
    # Every node keeps its self-loop, so its degree is > 0; unweighted degrees are
    # whole numbers, weighted ones may be below 1.
    node_scale = node_degree.clamp_min(1.0 if edge_weight is None else 1e-6).rsqrt()[:, None]
    edge_scale = edge_degree.clamp_min(1.0).reciprocal()[:, None]

    edge_x = weighted_sum(x * node_scale, weights, node_ids, edge_ids, graph["num_edges"])
    if edge_weight is not None:
        edge_x = edge_x * edge_weight[:, None]
    node_x = weighted_sum(edge_x * edge_scale, receive_weights, edge_ids, node_ids, graph["num_nodes"])
    return node_x * node_scale


# Hyperedge representation h_e = mean of its members' embeddings in H0.
def hyperedge_means(z, graph):
    ones = z.new_ones(len(graph["node_ids"]))
    edge_size = z.new_zeros(graph["num_edges"]).index_add_(0, graph["edge_ids"], ones)
    total = weighted_sum(z, ones, graph["node_ids"], graph["edge_ids"], graph["num_edges"])
    return total / edge_size[:, None]


# Input of the per-hyperedge weight scorer: [mean X of members ‖ one-hot family ‖ log |e|].
@torch.no_grad()
def hyperedge_descriptors(x, graph):
    ones = x.new_ones(len(graph["node_ids"]))
    edge_size = x.new_zeros(graph["num_edges"]).index_add_(0, graph["edge_ids"], ones)
    family = F.one_hot(graph["edge_family"], len(config.EDGE_FAMILIES)).to(x.dtype)
    return torch.cat([hyperedge_means(x, graph), family, edge_size.log()[:, None]], dim=1)


# softplus(INITIAL_FAMILY_LOGIT) = 1: family weights start at W = I.
INITIAL_FAMILY_LOGIT = math.log(math.e - 1)
EDGE_SCORER_DIM = 32


class HSLModel(nn.Module):
    # hsl_options=None gives the plain HGNN baseline (H* = H0, Z* = Z0).
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5, hsl_options=None, *,
                 skip_connection=False, family_weights=False, edge_weights=False, causal=False):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.causal = causal
        self.structure_learner = None
        if hsl_options is not None:
            self.structure_learner = StructureLearner(hidden_dim, **hsl_options)

        self.self_encoder = None
        if skip_connection:
            # Same shape as the HGNN encoder, without propagation.
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

        self.edge_weight_scorer = None
        if edge_weights:
            self.edge_weight_scorer = nn.Sequential(
                nn.Linear(input_dim + len(config.EDGE_FAMILIES) + 1, EDGE_SCORER_DIM), nn.ReLU(),
                nn.Linear(EDGE_SCORER_DIM, 1),
            )
            nn.init.zeros_(self.edge_weight_scorer[-1].weight)
            nn.init.zeros_(self.edge_weight_scorer[-1].bias)

    # α_e of every hyperedge of `graph` (1 for self-loops).
    def edge_alpha(self, x, graph):
        logits = self.edge_weight_scorer(hyperedge_descriptors(x, graph)).squeeze(-1)
        return torch.where(graph["edge_family"] == SELF_LOOP, 1.0, 2 * torch.sigmoid(logits))

    # Diagonal of W for `graph`: w_f(e), times α_e with edge_weights; None = identity.
    def hyperedge_weights(self, graph, alpha):
        weight = None
        if self.family_logits is not None:
            weight = F.softplus(self.family_logits)[graph["edge_family"]]
        if alpha is not None:
            weight = alpha if weight is None else weight * alpha
        return weight

    # Two HGNN layers: Linear -> propagate -> ReLU -> Dropout -> Linear -> propagate -> ReLU.
    def encode(self, x, graph, weights, edge_weight=None):
        receive = causal_receive(graph, weights) if self.causal else None
        hidden = F.relu(hgnn_propagate(self.layer1(x), graph, weights, edge_weight, receive))
        hidden = F.dropout(hidden, self.dropout, self.training)
        return F.relu(hgnn_propagate(self.layer2(hidden), graph, weights, edge_weight, receive))

    # Logits [num_nodes], and with the skip branch the exact split
    # logits = logit_graph + logit_self + bias.
    def classify(self, x, z):
        if self.self_encoder is None:
            return self.classifier(z).squeeze(-1), {}
        z_self = self.self_encoder(x)
        logits = self.classifier(torch.cat([z, z_self], dim=1)).squeeze(-1)
        weight = self.classifier.weight[0]
        parts = {"logit_graph": z @ weight[:z.shape[1]], "logit_self": z_self @ weight[z.shape[1]:]}
        return logits, parts

    # Learned family weights w_f and mean α_e per family, logged next to the HSL statistics.
    @torch.no_grad()
    def weight_summary(self, graph, alpha):
        summary = {}
        if self.family_logits is not None:
            weights = F.softplus(self.family_logits).tolist()
            summary.update({f"w_{name}": weight for name, weight in zip(config.EDGE_FAMILIES, weights)})
        if alpha is not None:
            for family, name in enumerate(config.EDGE_FAMILIES):
                in_family = graph["edge_family"] == family
                if family != SELF_LOOP and bool(in_family.any()):
                    summary[f"alpha_{name}"] = float(alpha[in_family].mean())
        return summary

    def forward(self, x, graph):
        # x: [num_nodes, input_dim]; graph: output of graph_to_device.
        h0_weights = x.new_ones(len(graph["node_ids"]))
        alpha = None if self.edge_weight_scorer is None else self.edge_alpha(x, graph)
        edge_weight = self.hyperedge_weights(graph, alpha)  # [num_edges] or None
        z0 = self.encode(x, graph, h0_weights, edge_weight)  # [num_nodes, hidden_dim]
        summary = self.weight_summary(graph, alpha) if self.training else {}

        if self.structure_learner is None:
            logits, parts = self.classify(x, z0)
            return {"logits": logits, **parts, "z0": z0, "z_star": z0, "structure": summary}

        edge_representations = hyperedge_means(z0, graph)
        refined_graph, refined_weights, structure = self.structure_learner(
            z0, edge_representations, graph
        )
        z_star = self.encode(x, refined_graph, refined_weights, edge_weight)  # [num_nodes, hidden_dim]
        logits, parts = self.classify(x, z_star)
        return {
            "logits": logits,
            **parts,
            "z0": z0,
            "z_star": z_star,
            "structure": {**structure, **summary},
        }
