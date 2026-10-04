# 8. Dropout prediction model: the HGNN branch (6_hgnn.py) and, when the scenario keeps
#    it, the MLP branch (7_mlp.py), read by one linear classifier (as HGNN_classifier of
#    the original code, with the MLP output added):
#      with MLP:     logit = [z_g ‖ z_s]·u + b
#      without MLP:  logit = z_g·u + b            (scenario B1)
#      p = σ(logit) = P(dropout)

from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")
HGNNEncoder = import_module("6_hgnn").HGNNEncoder
MLPEncoder = import_module("7_mlp").MLPEncoder


class DropoutModel(nn.Module):
    # Input:  input_dim (columns of X), hidden_dim, dropout rate,
    #         hgnn_layers (1 or 2), use_mlp (keep the MLP branch), learn_w (learn W).
    def __init__(self, input_dim, hidden_dim, dropout, *, hgnn_layers=2, use_mlp=True, learn_w=True):
        super().__init__()
        self.hgnn = HGNNEncoder(input_dim, hidden_dim, dropout, hgnn_layers, learn_w)   # graph branch
        self.mlp = MLPEncoder(input_dim, hidden_dim, dropout) if use_mlp else None  # own-feature branch
        self.classifier = nn.Linear(hidden_dim * (2 if use_mlp else 1), 1)          # u, b

    # [z_g ‖ z_s] with the MLP branch, z_g alone without it.
    def join(self, z_graph, x):
        if self.mlp is None:
            return z_graph
        return torch.cat([z_graph, self.mlp(x)], dim=1)

    # Training: logits of every train node on H0.
    # Input:  x [N, D], graph from prepare_graph.
    # Output: logits [N].
    def forward(self, x, graph):
        return self.classifier(self.join(self.hgnn(x, graph), x)).squeeze(-1)   # logit = z·u + b

    # Evaluation, step A (see 6_hgnn.py): call once per evaluation, in eval mode.
    def cache_train_states(self, x, graph):
        return self.hgnn.cache_train_states(x, graph)

    # Evaluation, step B: logits of a batch of validation/test targets.
    # Input:  as HGNNEncoder.forward_targets.
    # Output: logits [B].
    def forward_targets(self, x, rows, edges, single_user, cache):
        z_graph = self.hgnn.forward_targets(x, rows, edges, single_user, cache)
        return self.classifier(self.join(z_graph, x)).squeeze(-1)

    # Learned family weights for the results table; reported as "" when W is not learned
    # (scenario W1) or when the scenario does not use the family (it stays at its start value).
    # Input:  families kept by the scenario.
    # Output: dict w_course, w_object, w_user, w_self_loop.
    @torch.no_grad()
    def weight_summary(self, families):
        learned = isinstance(self.hgnn.family_logits, nn.Parameter)
        weights = F.softplus(self.hgnn.family_logits).tolist()     # w_f = softplus(θ_f)
        return {f"w_{name}": (weight if learned and name in families else "")
                for name, weight in zip(config.EDGE_FAMILIES, weights)}
