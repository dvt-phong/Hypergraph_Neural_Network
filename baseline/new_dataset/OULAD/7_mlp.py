# 7. MLP branch
# - UniGNN / UniGCNII https://github.com/OneForward/UniGNN

from torch import nn

# MLP branch: z_s from the node's own features.
class MLPEncoder(nn.Module):
    # Two layers A1, c1 and A2, c2:
    #   z_s = ReLU( Dropout(ReLU(x·A1 + c1))·A2 + c2 )
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # x·A1 + c1
            nn.ReLU(),
            nn.Dropout(dropout),                  # only while training
            nn.Linear(hidden_dim, hidden_dim),    # ·A2 + c2
            nn.ReLU(),
        )

    # Representation from the node's own features.
    # Shapes: x [B, D] -> z_s [B, hidden].
    def forward(self, x):
        return self.layers(x)
