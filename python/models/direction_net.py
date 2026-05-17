"""
models/direction_net.py — single-symbol GOLD direction head.

A 200 -> 256 -> 128 -> 1 MLP with BatchNorm and Dropout, trained with binary
focal BCE inside trainer.train_direction. Replaces the multi-symbol PRISM /
APEX / GNN / CE heads from mk4 — at GNN_NODES == 1 those degenerate to MLPs
anyway, so the explicit, single-purpose direction MLP is cleaner.

Input  : float[B, FEATURE_DIM_DIR]  (200 dims — M5 raw/mean20/std20/delta20)
Output : float[B, 1]                 (logit; trainer applies sigmoid via BCEWithLogits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FEATURE_DIM_DIR, DROPOUT


class DirectionNet(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM_DIR,
                 h0: int = 256, h1: int = 128, h2: int = 64,
                 dropout: float = DROPOUT):
        super().__init__()
        # BatchNorm normalizes per-feature across the batch (mixed-scale
        # tabular inputs) and bakes running stats into the exported ONNX so
        # inference matches training without a separate scaler.
        self.norm = nn.BatchNorm1d(in_dim)
        self.fc0 = nn.Linear(in_dim, h0)
        self.fc1 = nn.Linear(h0, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        z = F.relu(self.fc0(x)); z = self.drop(z)
        z = F.relu(self.fc1(z)); z = self.drop(z)
        z = F.relu(self.fc2(z))
        return self.head(z)   # [B, 1] logit


def create_direction_net(in_dim: int = FEATURE_DIM_DIR,
                          dropout: float = DROPOUT) -> DirectionNet:
    return DirectionNet(in_dim=in_dim, dropout=dropout)
