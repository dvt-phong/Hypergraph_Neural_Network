# 8. Dropout prediction model

from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")
HGNNEncoder = import_module("6_hgnn").HGNNEncoder
MLPEncoder = import_module("7_mlp").MLPEncoder


# Full model: HGNN branch, optional MLP branch, linear classifier.
class DropoutModel(nn.Module):
    # Both branches and the classifier u, b; use_mlp=False drops the MLP branch (scenario H).
    def __init__(self, input_dim, hidden_dim, dropout, hgnn_layers=2, use_mlp=True, learn_w=True):
        super().__init__()
        self.hgnn = HGNNEncoder(input_dim, hidden_dim, dropout, hgnn_layers, learn_w)   # graph branch
        if use_mlp:
            self.mlp = MLPEncoder(input_dim, hidden_dim, dropout)                   # own-feature branch
            classifier_dim = 2 * hidden_dim                                          # [z_g ‖ z_s]
        else:
            self.mlp = None
            classifier_dim = hidden_dim                                              # z_g
        self.classifier = nn.Linear(classifier_dim, 1)                              # u, b

    # [z_g ‖ z_s] with the MLP branch, z_g alone without it.
    def join(self, z_graph, x):
        if self.mlp is None:
            return z_graph
        z_self = self.mlp(x)
        return torch.cat([z_graph, z_self], dim=1)

    # Training: logits of every train node on H0.
    # Shapes: x [N, D] -> logits [N].
    def forward(self, x, graph):
        z_graph = self.hgnn(x, graph)
        z = self.join(z_graph, x)
        logits = self.classifier(z)                                     # logit = z·u + b, [N, 1]
        return logits.squeeze(-1)                                       # [N]

    # Evaluation, step A (see 6_hgnn.py): call once per evaluation, in eval mode.
    def cache_train_states(self, x, graph):
        return self.hgnn.cache_train_states(x, graph)

    def forward_targets(self, x, rows, edges, single_user, cache):
        z_graph = self.hgnn.forward_targets(x, rows, edges, single_user, cache)
        z = self.join(z_graph, x)
        logits = self.classifier(z)                                     # logit = [z_g ‖ z_s]·u + b, [B, 1]
        return logits.squeeze(-1)                                       # [B]

    # Learned family weights for the results table
    def weight_summary(self, families):
        with torch.no_grad():
            weights = F.softplus(self.hgnn.family_logits).tolist()  # w_f = softplus(θ_f)
        summary = {}
        for index in range(len(config.EDGE_FAMILIES)):
            name = config.EDGE_FAMILIES[index]
            if self.hgnn.learn_w and name in families:
                summary[f"w_{name}"] = weights[index]
            else:
                summary[f"w_{name}"] = ""                           # W not learned, or family not used
        return summary
