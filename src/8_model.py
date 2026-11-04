# 8. Dropout prediction model

from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")
HGNNEncoder = import_module("6_hgnn").HGNNEncoder
MLPEncoder = import_module("7_mlp").MLPEncoder


# Full model: optional HGNN branch, optional MLP branch, linear classifier.
class DropoutModel(nn.Module):
    # Both branches and the classifier u, b; use_mlp=False drops the MLP branch (scenario H),
    # use_hgnn=False drops the HGNN branch (scenario P0, MLP-only).
    def __init__(self, input_dim, hidden_dim, dropout, hgnn_layers=2, use_mlp=True, learn_w=True,
                 use_hgnn=True):
        super().__init__()
        if not use_hgnn and not use_mlp:
            raise ValueError("the model needs the HGNN branch, the MLP branch or both")
        classifier_dim = 0
        if use_hgnn:
            self.hgnn = HGNNEncoder(input_dim, hidden_dim, dropout, hgnn_layers, learn_w)   # graph branch
            classifier_dim += hidden_dim                                                 # z_g
        else:
            self.hgnn = None
        if use_mlp:
            self.mlp = MLPEncoder(input_dim, hidden_dim, dropout)                       # own-feature branch
            classifier_dim += hidden_dim                                                 # z_s
        else:
            self.mlp = None
        self.classifier = nn.Linear(classifier_dim, 1)                                  # u, b

    # [z_g ‖ z_s] with both branches, z_g or z_s alone with one branch.
    def join(self, x, graph):
        if self.hgnn is None:
            return self.mlp(x)
        z_graph = self.hgnn(x, graph)
        if self.mlp is None:
            return z_graph
        z_self = self.mlp(x)
        return torch.cat([z_graph, z_self], dim=1)

    # Logits of every node (train + validation + test) on H.
    # Shapes: x [N, D] -> logits [N].
    def forward(self, x, graph):
        z = self.join(x, graph)
        logits = self.classifier(z)                                     # logit = [z_g ‖ z_s]·u + b, [N, 1]
        return logits.squeeze(-1)                                       # [N]

    # Learned family weights for the results table ("" when W is not learned or not used)
    def weight_summary(self, families):
        summary = {}
        if self.hgnn is None:
            for name in config.EDGE_FAMILIES:
                summary[f"w_{name}"] = ""                           # no HGNN branch, no W
            return summary
        with torch.no_grad():
            weights = F.softplus(self.hgnn.family_logits).tolist()  # w_f = softplus(θ_f)
        for index in range(len(config.EDGE_FAMILIES)):
            name = config.EDGE_FAMILIES[index]
            if self.hgnn.learn_w and name in families:
                summary[f"w_{name}"] = weights[index]
            else:
                summary[f"w_{name}"] = ""                           # W not learned, or family not used
        return summary
