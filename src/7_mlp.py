# 7. MLP branch: the node's own features only, no propagation over the hypergraph.
#    Same depth and hidden size as the HGNN branch: it is the HGNN encoder without G.
# Tham khảo từ project/bài báo:
# - UniGNN / UniGCNII, IJCAI 2021 (Huang & Yang): keeping the node's own signal
#   next to the propagated one (initial residual / skip)
#   Code: https://github.com/OneForward/UniGNN

from torch import nn


class MLPEncoder(nn.Module):
    # z_s = ReLU( Dropout(ReLU(x·A1 + c1))·A2 + c2 )
    # Input:  input_dim (columns of X), hidden_dim, dropout rate.
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # x·A1 + c1
            nn.ReLU(),
            nn.Dropout(dropout),                  # only while training
            nn.Linear(hidden_dim, hidden_dim),    # ·A2 + c2
            nn.ReLU(),
        )

    # Input:  x [B, D].
    # Output: z_s [B, hidden].
    def forward(self, x):
        return self.layers(x)
