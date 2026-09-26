# 6. Hypergraph Structure Learning (HSL), following Cai et al., IJCAI 2022:
#
#     H* = Me ⊙ Mv ⊙ (H0 + ΔH) + I
#
#   ΔH  implicit connections (Eq. 4-5): each Behavioral hyperedge recruits the
#       `add_per_edge` candidates whose Z0 is most cosine-similar to the
#       hyperedge. Candidates are the anchor's neighbors k..k_max-1.
#   Me  hyperedge sampling (Eq. 2-3): keep hyperedge e with probability
#       σ(MLP([h_e ‖ family_e])).
#   Mv  incident node sampling (Eq. 6-7): keep membership (v, e) with
#       probability σ(MLP([z_v ‖ h_e])). This also decides whether a ΔH
#       addition stays.
#   I   self-loops are never removed, so no node becomes isolated (Eq. 9).
#
# Training draws 0/1 masks with the straight-through Gumbel trick (hard 0/1
# forward, smooth gradient backward). Validation/test keep a membership when
# its probability is above 0.5, so the result is deterministic.
#
# Adaptation to MOOC data (the paper used transductive node classification):
#   * Me is an MLP of the hyperedge representation instead of one free
#     parameter per hyperedge, so it also works on unseen validation/test graphs.
#   * ΔH only adds nodes to Behavioral hyperedges. Adding a learner of another
#     course to a Course hyperedge would contradict its meaning.

from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

config = import_module("0_config")
SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")

# σ(3) ≈ 0.95: start by keeping almost every hyperedge and membership.
INITIAL_KEEP_LOGIT = 3.0


# Bernoulli(σ(logits)) sample as 0/1 values that still pass a gradient.
def keep_mask(logits, temperature, training):
    if not training:
        return (logits > 0).float()
    uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    logistic_noise = torch.log(uniform) - torch.log1p(-uniform)  # = Gumbel - Gumbel
    soft = torch.sigmoid((logits + logistic_noise) / temperature)
    hard = (soft > 0.5).float()
    return hard + soft - soft.detach()  # value = hard, gradient = d soft


class StructureLearner(nn.Module):
    def __init__(
        self,
        hidden_dim,
        *,
        scorer_dim=32,
        temperature=0.4,
        add_per_edge=2,
        edge_sampling=True,
        node_sampling=True,
    ):
        super().__init__()
        self.temperature = temperature
        self.add_per_edge = add_per_edge
        self.edge_sampling = edge_sampling
        self.node_sampling = node_sampling
        family_count = len(config.EDGE_FAMILIES)

        # Me scorer: MLP([h_e ‖ one-hot family]) -> logit.
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim + family_count, scorer_dim),
            nn.ReLU(),
            nn.Linear(scorer_dim, 1),
        )
        # Mv scorer: MLP([z_v ‖ h_e]). Its first layer W [z_v ‖ h_e] is split
        # into W_v z_v + W_e h_e so each part is computed once per node/edge.
        self.membership_node = nn.Linear(hidden_dim, scorer_dim)
        self.membership_edge = nn.Linear(hidden_dim, scorer_dim, bias=False)
        self.membership_output = nn.Linear(scorer_dim, 1)

        nn.init.constant_(self.edge_scorer[-1].bias, INITIAL_KEEP_LOGIT)
        nn.init.constant_(self.membership_output.bias, INITIAL_KEEP_LOGIT)

    def forward(self, z0, edge_representations, graph):
        h0_count = len(graph["node_ids"])
        is_self_loop = graph["edge_family"] == SELF_LOOP

        # H0 + ΔH
        added_nodes, added_edges = self.implicit_connections(z0, edge_representations, graph)
        node_ids = torch.cat([graph["node_ids"], added_nodes])
        edge_ids = torch.cat([graph["edge_ids"], added_edges])

        # Me: one keep/drop decision per hyperedge.
        edge_keep = z0.new_ones(graph["num_edges"])
        if self.edge_sampling:
            family = F.one_hot(graph["edge_family"], len(config.EDGE_FAMILIES)).to(z0.dtype)
            edge_logits = self.edge_scorer(torch.cat([edge_representations, family], dim=1))
            edge_keep = keep_mask(edge_logits.squeeze(-1), self.temperature, self.training)
            edge_keep = torch.where(is_self_loop, 1.0, edge_keep)

        # Mv: one keep/drop decision per membership (v, e).
        membership_keep = z0.new_ones(len(node_ids))
        if self.node_sampling:
            membership_logits = self.membership_logits(z0, edge_representations, node_ids, edge_ids)
            membership_keep = keep_mask(membership_logits, self.temperature, self.training)
            membership_keep = torch.where(is_self_loop[edge_ids], 1.0, membership_keep)

        weights = edge_keep[edge_ids] * membership_keep
        refined_graph = {**graph, "node_ids": node_ids, "edge_ids": edge_ids}
        return refined_graph, weights, self.summary(graph, weights, h0_count)

    # ΔH: for each Behavioral hyperedge, pick the `add_per_edge` candidates with
    # the highest cosine similarity cos(z_v, h_e). No gradient, as in HSL.
    @torch.no_grad()
    def implicit_connections(self, z0, edge_representations, graph):
        candidate_nodes = graph["candidate_node_ids"]  # [behavioral edges, candidates]
        candidate_edges = graph["candidate_edge_ids"]  # [behavioral edges]
        count = min(self.add_per_edge, candidate_nodes.shape[1])
        if count == 0 or len(candidate_edges) == 0:
            empty = candidate_edges.new_empty(0)
            return empty, empty

        similarity = F.cosine_similarity(
            z0[candidate_nodes],
            edge_representations[candidate_edges][:, None, :],
            dim=-1,
        )
        best = similarity.topk(count, dim=1).indices
        added_nodes = candidate_nodes.gather(1, best).flatten()
        added_edges = candidate_edges[:, None].expand(-1, count).flatten()
        return added_nodes, added_edges

    # Mv logits for every membership, computed in chunks to bound memory.
    def membership_logits(self, z0, edge_representations, node_ids, edge_ids):
        node_part = self.membership_node(z0)
        edge_part = self.membership_edge(edge_representations)
        logits = []
        for start in range(0, len(node_ids), config.MEMBERSHIP_CHUNK_SIZE):
            part = slice(start, start + config.MEMBERSHIP_CHUNK_SIZE)
            logits.append(checkpoint(
                self._membership_chunk,
                node_part, edge_part, node_ids[part], edge_ids[part],
                use_reentrant=False,
            ))
        return torch.cat(logits)

    def _membership_chunk(self, node_part, edge_part, node_ids, edge_ids):
        hidden = F.relu(node_part[node_ids] + edge_part[edge_ids])
        return self.membership_output(hidden).squeeze(-1)

    # Share of H0 memberships kept per family, and ΔH additions kept.
    @torch.no_grad()
    def summary(self, graph, weights, h0_count):
        membership_family = graph["edge_family"][graph["edge_ids"]]
        h0_weights = weights[:h0_count]
        result = {}
        for family, name in enumerate(config.EDGE_FAMILIES):
            in_family = membership_family == family
            if family != SELF_LOOP and bool(in_family.any()):
                result[f"kept_{name}"] = float(h0_weights[in_family].mean())
        result["added"] = round(float(weights[h0_count:].sum()))
        return result
