"""
aurum/fusion.py — L2 cross-timeframe fusion.

The three per-timeframe encoders (M5 / M15 / H1) each emit a pooled
embedding. A decision on the M5 bar should be able to *consult* the H1
and M15 context — a fast signal inside a higher-timeframe trend is worth
more than the same signal against it.

Crossformer (Zhang & Yan, ICLR 2023) introduced explicit cross-dimension
attention for multivariate series; here the "dimensions" are timeframes.
The M5 embedding is the query; M5/M15/H1 embeddings are keys/values.
Output is a single fused decision embedding.

All ops are Linear / LayerNorm / MultiheadAttention — ONNX-clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurum.aurum_config import TIMEFRAMES, D_MODEL, N_HEADS, DROPOUT


class CrossTimeframeFusion(nn.Module):
    """
    Input  : dict {tf: [B, D_MODEL]} pooled embeddings.
    Output : [B, D_MODEL] fused decision embedding.
    """

    def __init__(self):
        super().__init__()
        self.tf_order = list(TIMEFRAMES.keys())
        # learned per-timeframe identity embedding so attention can tell
        # which timeframe a token came from
        self.tf_embed = nn.Parameter(
            torch.randn(len(self.tf_order), D_MODEL) * 0.02)
        self.norm_in = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS,
                                          dropout=DROPOUT, batch_first=True)
        self.norm_out = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(D_MODEL * 2, D_MODEL),
        )

    def forward(self, embeds: dict[str, torch.Tensor]) -> torch.Tensor:
        # Stack into [B, n_tf, D] with timeframe-identity added.
        toks = []
        for i, tf in enumerate(self.tf_order):
            toks.append(embeds[tf] + self.tf_embed[i])
        kv = self.norm_in(torch.stack(toks, dim=1))     # [B, n_tf, D]
        # M5 (index 0) is the decision query.
        q = kv[:, :1, :]
        a, _ = self.attn(q, kv, kv, need_weights=False)
        fused = (q + a).squeeze(1)                      # [B, D]
        fused = fused + self.ffn(self.norm_out(fused))
        return fused


def create_fusion() -> CrossTimeframeFusion:
    return CrossTimeframeFusion()
