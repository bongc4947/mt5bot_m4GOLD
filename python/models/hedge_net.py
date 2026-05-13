"""
hedge_net.py — Phase 3 hedge model architecture.

Dual-input model: takes feature vectors of legA and legB at the same
timestamp plus their spread state, and outputs:

  revert_logit    — sigmoid(...) > threshold means "the spread will
                    mean-revert in the next horizon_bars"

Architecture:
    legA features  ->  shared encoder MLP  ->  zA
    legB features  ->  shared encoder MLP  ->  zB    (weight-tied)
    spread_state   ->  spread MLP         ->  zS
    [zA, zB, zS, zA - zB]  ->  trunk      ->  revert_head

Weight-tying the leg encoder is important: a hedge model shouldn't
learn different feature transformations for "leg A" vs "leg B" — the
architectural symmetry mirrors the problem (the spread is symmetric in
A and B up to sign).

The model never predicts price direction — only whether the spread
mean-reverts. This is what makes hedge profitable on bull-trending
assets where direction prediction would just ride the trend.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HedgeNet(nn.Module):
    """
    Forward:
      leg_a_feat: (B, F)     leg A features (per-bar)
      leg_b_feat: (B, F)     leg B features
      spread_state: (B, S)   spread features (z-score, vol, etc.)
      returns: revert_logit (B,) — single logit per sample
    """

    def __init__(self, feature_dim: int, spread_state_dim: int = 4,
                 leg_hidden: int = 32, trunk_hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.spread_state_dim = int(spread_state_dim)
        # Shared (weight-tied) leg encoder
        self.leg_enc = nn.Sequential(
            nn.Linear(feature_dim, leg_hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Spread-state encoder
        self.spread_enc = nn.Sequential(
            nn.Linear(spread_state_dim, leg_hidden), nn.ReLU(),
        )
        # Trunk consumes [zA, zB, zS, zA - zB] = 4 * leg_hidden
        trunk_in = 4 * leg_hidden
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, trunk_hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.revert_head = nn.Linear(trunk_hidden, 1)

    def forward(self, leg_a_feat: torch.Tensor,
                leg_b_feat: torch.Tensor,
                spread_state: torch.Tensor) -> torch.Tensor:
        za = self.leg_enc(leg_a_feat)
        zb = self.leg_enc(leg_b_feat)         # weight-tied
        zs = self.spread_enc(spread_state)
        z  = torch.cat([za, zb, zs, za - zb], dim=-1)
        z  = self.trunk(z)
        return self.revert_head(z).squeeze(-1)


class HedgeNetExportWrapper(nn.Module):
    """
    Concatenates the three inputs into one (B, 2F + S) tensor on the
    MQL5 side so the EA does one OnnxRun() call. The wrapper splits
    them internally.
    """

    def __init__(self, model: HedgeNet, t_revert: float = 1.0):
        super().__init__()
        self.model = model
        self.t_revert = float(t_revert)
        self.feature_dim = model.feature_dim
        self.spread_state_dim = model.spread_state_dim

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """
        packed: (B, 2*F + S) — [legA features | legB features | spread_state]
        returns: (B,) revert logit (calibrated)
        """
        F = self.feature_dim
        leg_a = packed[:, :F]
        leg_b = packed[:, F:2*F]
        sp    = packed[:, 2*F:]
        out   = self.model(leg_a, leg_b, sp)
        return out / self.t_revert


def create_hedge_net(feature_dim: int, spread_state_dim: int = 4,
                      leg_hidden: int = 32, trunk_hidden: int = 32,
                      dropout: float = 0.1) -> HedgeNet:
    return HedgeNet(feature_dim=feature_dim,
                     spread_state_dim=spread_state_dim,
                     leg_hidden=leg_hidden, trunk_hidden=trunk_hidden,
                     dropout=dropout)
