"""
loss_pnl.py — Cost-aware expected-PnL loss for the scalp model's
should-trade head.

Standard BCE loss treats every label as equally important. PnL loss
weights each sample by the *amount* of money it would make or lose:

  L = -mean( pred_signed * fwd_return ) + cost * mean( |pred_signed| )

Translated to the should-trade head:
  - High-conviction-correct trades reduce loss most.
  - Wrong trades are penalised proportionally to their loss size.
  - Skipping a trade has zero contribution (the |pred_signed| term
    rewards the model for abstaining when uncertain).

On synthetic random data, this loss naturally drives all should_trade
predictions to 0 because E[fwd_return] = 0 → only the |pred| cost
remains → optimal is no-trade.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ExpectedPnLLoss(nn.Module):
    """
    Differentiable expected-PnL loss for a should-trade gate combined
    with a direction prediction.

    Inputs to forward:
      direction_logit : (B,) — model's direction prediction (will be
                              passed through sigmoid → ±1 mapping)
      should_trade_logit : (B,) — model's should-trade gate
      forward_return : (B,) — actual realised forward return per sample
      cost : float — round-trip cost (commission + spread); same units
                     as forward_return

    Loss = -mean( gate * signed * fwd ) + cost * mean( gate * |signed| )
         where gate   = sigmoid(should_trade_logit)
               signed = 2 * sigmoid(direction_logit) - 1   in (-1, +1)

    Gradient flows through both heads. The gate term lets the model
    learn to ABSTAIN when no edge exists.
    """

    def __init__(self, cost: float = 0.0001):
        super().__init__()
        self.cost = float(cost)

    def forward(self, direction_logit: torch.Tensor,
                should_trade_logit:    torch.Tensor,
                forward_return:        torch.Tensor) -> torch.Tensor:
        signed = 2.0 * torch.sigmoid(direction_logit) - 1.0
        gate   = torch.sigmoid(should_trade_logit)
        gross  = gate * signed * forward_return
        cost_term = gate * torch.abs(signed) * self.cost
        return -gross.mean() + cost_term.mean()
