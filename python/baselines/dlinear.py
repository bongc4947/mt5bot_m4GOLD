"""
baselines/dlinear.py — the honest linear control AURUM must beat.

DLinear (Zeng et al., AAAI 2023, "Are Transformers Effective for Time
Series Forecasting?") decomposes each channel series into a moving-average
trend and a remainder, then applies a *linear* map to each. Zeng et al.
showed this beats every transformer of its day on noisy series — so it is
the right control: if AURUM cannot beat DLinear on purged CV, the
transformer machinery is not earning its keep and should not deploy.

This variant is a classifier (3-class direction) rather than a forecaster.
It takes the same flat float[N, FLAT_INPUT_DIM] contract as AurumNet so the
CV harness and ONNX export path are uniform across models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurum.aurum_config import (
    TIMEFRAMES, N_CHANNELS, FLAT_INPUT_DIM, N_DIRECTION_CLASSES,
)


class _MovingAvg(nn.Module):
    """Causal moving average for trend extraction."""

    def __init__(self, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, L]
        pad = self.kernel - 1
        # left-pad only — strictly causal, no future leak
        xp = torch.cat([x[:, :, :1].repeat(1, 1, pad), x], dim=-1)
        return self.pool(xp)


class DLinearClassifier(nn.Module):
    """
    Per-timeframe series decomposition + linear projection, concatenated
    and mapped to 3 direction logits.
    """

    def __init__(self, kernel: int = 25, proj: int = 32):
        super().__init__()
        self.decomp = _MovingAvg(kernel)
        self.tf_specs = list(TIMEFRAMES.items())   # [(name, L), ...]
        # One linear per (timeframe) for trend and for remainder; input is
        # the flattened L*C series, output `proj` features.
        self.lin_trend = nn.ModuleDict()
        self.lin_resid = nn.ModuleDict()
        for name, L in self.tf_specs:
            self.lin_trend[name] = nn.Linear(L * N_CHANNELS, proj)
            self.lin_resid[name] = nn.Linear(L * N_CHANNELS, proj)
        self.head = nn.Linear(proj * 2 * len(self.tf_specs), N_DIRECTION_CLASSES)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """x_flat: [B, FLAT_INPUT_DIM] -> logits [B, 3]"""
        feats = []
        off = 0
        for name, L in self.tf_specs:
            block = L * N_CHANNELS
            seg = x_flat[:, off:off + block]                     # [B, L*C]
            off += block
            seg2d = seg.reshape(-1, L, N_CHANNELS).transpose(1, 2)  # [B, C, L]
            trend = self.decomp(seg2d)
            resid = seg2d - trend
            t = self.lin_trend[name](trend.reshape(seg.shape[0], -1))
            r = self.lin_resid[name](resid.reshape(seg.shape[0], -1))
            feats.append(t)
            feats.append(r)
        return self.head(torch.cat(feats, dim=1))


def create_dlinear() -> DLinearClassifier:
    return DLinearClassifier()


if __name__ == "__main__":
    m = create_dlinear()
    x = torch.randn(4, FLAT_INPUT_DIM)
    print("DLinear out:", m(x).shape)   # expect [4, 3]
