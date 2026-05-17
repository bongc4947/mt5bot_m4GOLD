"""
aurum/heads.py — L3 multi-task output heads.

A single fused embedding feeds four heads. Multi-task learning regularises
the shared trunk: the auxiliary tasks (quantile, exec, regime) inject
gradient signal that a lone direction classifier never sees, which the
Temporal Fusion Transformer (Lim et al., 2021) showed improves the primary
forecast.

Heads (concatenated into the deployed [1, 13] output):
  direction  3 logits   short / flat / long
  quantile   3 values   forward-return quantiles {0.1, 0.5, 0.9}
  exec       3 values   sl_atr, tp_atr, timing  (raw — EA bounds them)
  regime     4 logits   trend-up / trend-down / range / high-vol
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurum.aurum_config import (
    D_MODEL, DROPOUT, N_DIRECTION_CLASSES, N_QUANTILES, N_EXEC, N_REGIME,
)


def _mlp_head(out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(D_MODEL, D_MODEL // 2), nn.GELU(),
        nn.Dropout(DROPOUT), nn.Linear(D_MODEL // 2, out_dim),
    )


class MultiTaskHeads(nn.Module):
    """Fused embedding [B, D_MODEL] -> dict of per-task tensors."""

    def __init__(self):
        super().__init__()
        self.direction = _mlp_head(N_DIRECTION_CLASSES)
        self.quantile = _mlp_head(N_QUANTILES)
        self.exec = _mlp_head(N_EXEC)
        self.regime = _mlp_head(N_REGIME)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "direction": self.direction(h),     # logits
            "quantile":  self.quantile(h),       # raw quantile values
            "exec":      self.exec(h),           # raw
            "regime":    self.regime(h),         # logits
        }


def create_heads() -> MultiTaskHeads:
    return MultiTaskHeads()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class PinballLoss(nn.Module):
    """Quantile (pinball) loss — the standard objective for quantile heads."""

    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles).float())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, Q]  target: [B] (broadcast over Q)
        t = target.unsqueeze(1)
        err = t - pred
        loss = torch.maximum(self.q * err, (self.q - 1.0) * err)
        return loss.mean()
