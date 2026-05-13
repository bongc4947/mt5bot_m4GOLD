"""
models/exec_net.py — Execution model: 1120 → 512 → 256 → 5 outputs (multi-task).
Outputs: [timing, sl_pips, tp_pips, vol_mult, session_gate]
ONNX I/O: float[1, 1120] → float[1, 5]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EXEC_H1, EXEC_H2, DROPOUT, FEATURE_DIM_EXEC


class ExecNet(nn.Module):
    """
    Multi-task execution model.
    Shared trunk + 5 independent output heads.
    """

    def __init__(self, in_dim: int = FEATURE_DIM_EXEC,
                 h1: int = EXEC_H1,
                 h2: int = EXEC_H2,
                 dropout: float = DROPOUT):
        super().__init__()
        # Shared trunk — BatchNorm1d normalizes per-feature across batch (correct
        # for mixed-scale tabular inputs; bakes running stats into ONNX for inference)
        self.norm = nn.BatchNorm1d(in_dim)
        self.fc1  = nn.Linear(in_dim, h1)
        self.fc2  = nn.Linear(h1, h2)
        self.drop = nn.Dropout(dropout)

        # Output heads
        self.fc_timing  = nn.Linear(h2, 1)   # sigmoid → [0,1]
        self.fc_sl      = nn.Linear(h2, 1)   # softplus → (0,∞) pips
        self.fc_tp      = nn.Linear(h2, 1)   # softplus → (0,∞) pips
        self.fc_vol     = nn.Linear(h2, 1)   # sigmoid → [0,1] → mapped to [0.5, 2.0]
        self.fc_session = nn.Linear(h2, 1)   # sigmoid → [0,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns [batch, 5]: [timing, sl_pips, tp_pips, vol_mult, session_gate]
        All outputs are in activation-bounded range for numerical stability.
        EA applies thresholds on top.
        """
        x = self.norm(x)
        z = F.relu(self.fc1(x))
        z = self.drop(z)
        z = F.relu(self.fc2(z))
        z = self.drop(z)

        timing  = torch.sigmoid(self.fc_timing(z))         # [B,1]
        sl_pips = F.softplus(self.fc_sl(z))                # [B,1]
        tp_pips = F.softplus(self.fc_tp(z))                # [B,1]
        vol_mult = torch.sigmoid(self.fc_vol(z))           # [B,1]  → EA maps to [0.5, 2.0]
        session = torch.sigmoid(self.fc_session(z))        # [B,1]

        return torch.cat([timing, sl_pips, tp_pips, vol_mult, session], dim=1)  # [B,5]


def create_exec_net(in_dim: int = FEATURE_DIM_EXEC,
                   h1: int = EXEC_H1,
                   h2: int = EXEC_H2,
                   dropout: float = DROPOUT) -> ExecNet:
    return ExecNet(in_dim, h1, h2, dropout)
