"""
scalp_net.py — Phase 2 scalp model architecture.

Takes (B, T, F) sequential input and produces TWO outputs per sample:

  direction_logit  — sigmoid(...) > 0.5 → predict LONG, < 0.5 → predict SHORT
  should_trade_logit — sigmoid(...) > should_trade_threshold → fire the trade

The should_trade head is what makes the model selective. It's trained
against a label that's positive only when the directional prediction
would have made positive PnL after costs (set by labeler_scalp.py).
The two-head split gives selectivity by construction:

  predict()    -> if sigmoid(should_trade) < T  -> NO TRADE
                  else if direction > 0.5       -> LONG
                  else                           -> SHORT

Architecture: 1-layer GRU(F → hidden) → MLP(hidden → 32) → 2 heads.
Total ~25K params for F=200, hidden=64. Cheap on CPU per-bar in MT5.

ONNX export: a separate ScalpNetExportWrapper combines both heads' raw
logits into a (B, 2) output tensor (direction_logit, should_trade_logit)
so MQL5 reads one OnnxRun() with one output array.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ScalpNet(nn.Module):
    """
    Two-head sequential model: direction + should-trade.

    Forward:
        x: (B, T, F) float32
        returns (direction_logit, should_trade_logit) both (B,)
    """

    def __init__(self, feature_dim: int, hidden: int = 64,
                 mlp_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)

        self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden,
                          num_layers=1, batch_first=True)
        self.trunk = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.direction_head    = nn.Linear(mlp_hidden, 1)
        self.should_trade_head = nn.Linear(mlp_hidden, 1)

    def forward(self, x: torch.Tensor):
        # GRU on (B, T, F) → take final hidden state (B, hidden).
        _, h_n = self.gru(x)
        h = h_n[-1]
        z = self.trunk(h)
        return (self.direction_head(z).squeeze(-1),
                self.should_trade_head(z).squeeze(-1))


class ScalpNetExportWrapper(nn.Module):
    """
    Wraps ScalpNet for ONNX export with two outputs combined into one
    (B, 2) tensor: column 0 = direction_logit, column 1 = should_trade_logit.
    Optionally divides logits by per-head temperature scalars baked at
    export time (so MQL5 doesn't need to know about calibration — just
    reads the calibrated logit and applies sigmoid + threshold).
    """

    def __init__(self, model: ScalpNet,
                 t_dir: float = 1.0, t_trade: float = 1.0):
        super().__init__()
        self.model   = model
        self.t_dir   = float(t_dir)
        self.t_trade = float(t_trade)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d, s = self.model(x)
        d = d / self.t_dir
        s = s / self.t_trade
        return torch.stack([d, s], dim=-1)


def create_scalp_net(feature_dim: int, hidden: int = 64,
                      mlp_hidden: int = 32, dropout: float = 0.1) -> ScalpNet:
    return ScalpNet(feature_dim=feature_dim, hidden=hidden,
                     mlp_hidden=mlp_hidden, dropout=dropout)
