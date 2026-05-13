"""
models/modify_net.py — SL/TP modification model: 1008 → 256 → 128 → 3 outputs.
Outputs: [move_sl_to_be, trail_sl_pips, close_now]
ONNX I/O: float[1, 1008] → float[1, 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MOD_H1, MOD_H2, DROPOUT, FEATURE_DIM_MOD


class ModifyNet(nn.Module):
    """
    Position modification model.
    move_sl_to_be : sigmoid [0,1] — move SL to breakeven
    trail_sl_pips : softplus — trailing distance pips (0=no change)
    close_now     : sigmoid [0,1] — early exit gate
    """

    def __init__(self, in_dim: int = FEATURE_DIM_MOD,
                 h1: int = MOD_H1,
                 h2: int = MOD_H2,
                 dropout: float = DROPOUT):
        super().__init__()
        # BatchNorm1d normalizes per-feature across batch (correct for tabular inputs)
        self.norm = nn.BatchNorm1d(in_dim)
        self.fc1  = nn.Linear(in_dim, h1)
        self.fc2  = nn.Linear(h1, h2)
        self.drop = nn.Dropout(dropout)

        self.fc_be    = nn.Linear(h2, 1)   # move_sl_to_be
        self.fc_trail = nn.Linear(h2, 1)   # trail_sl_pips
        self.fc_close = nn.Linear(h2, 1)   # close_now

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [batch, 3]: [move_sl_to_be, trail_sl_pips, close_now]"""
        x = self.norm(x)
        z = F.relu(self.fc1(x))
        z = self.drop(z)
        z = F.relu(self.fc2(z))
        z = self.drop(z)

        be    = torch.sigmoid(self.fc_be(z))    # [B,1]
        trail = F.softplus(self.fc_trail(z))    # [B,1]
        close = torch.sigmoid(self.fc_close(z)) # [B,1]

        return torch.cat([be, trail, close], dim=1)  # [B,3]


def create_modify_net(in_dim: int = FEATURE_DIM_MOD,
                      h1: int = MOD_H1,
                      h2: int = MOD_H2,
                      dropout: float = DROPOUT) -> ModifyNet:
    return ModifyNet(in_dim, h1, h2, dropout)
