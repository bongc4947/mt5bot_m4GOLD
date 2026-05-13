"""
profit_metric.py — Profitability metrics computed from direction model predictions.

Used during training validation, live monitoring, dashboard, and EA quality gating.

RR_RATIO = 2.0 means TP = 2 × SL. This matches the labeler's ATR-threshold logic:
a label of +1 means price moved at least LABEL_ATR_THRESH × ATR in the long
direction within LABEL_FORWARD_BARS bars — the 2:1 assumption is conservative.

Metrics returned:
  profit_factor  — gross_profit / gross_loss  (>1.0 = profitable; >1.5 = good; >2.0 = excellent)
  win_rate       — fraction of predicted trades that were correct  (breakeven at 1/(1+RR) = 33.3%)
  expected_value — EV per trade in R-multiples  (>0 = profitable per trade)
  n_trades       — number of non-flat predictions used in the calculation
"""

import torch
import numpy as np
from typing import Dict, Union

RR_RATIO = 2.0   # reward:risk — must match labeler ATR-threshold intent


def compute_profit_metrics(
        logits: torch.Tensor,
        y_binary: torch.Tensor,
        rr: float = RR_RATIO,
) -> Dict[str, float]:
    """
    Compute profitability metrics from binary direction model output.

    Args:
        logits    : raw model output (before sigmoid), shape (B,)
        y_binary  : ground truth  0=SHORT won, 1=LONG won, shape (B,)
        rr        : reward-to-risk ratio

    Returns dict with keys: profit_factor, win_rate, expected_value, n_trades
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits.float()) > 0.5).float()
        y     = y_binary.float()

        wins       = (preds == y).float()
        n          = max(len(wins), 1)
        win_rate   = wins.mean().item()

        gross_profit = wins.sum().item() * rr
        gross_loss   = (1.0 - wins).sum().item()

        if gross_loss < 1e-8:
            pf = float("inf") if gross_profit > 0 else 1.0
        else:
            pf = gross_profit / gross_loss

        ev = win_rate * rr - (1.0 - win_rate)   # EV in R-multiples per trade

    return {
        "profit_factor":  round(pf, 4),
        "win_rate":       round(win_rate, 4),
        "expected_value": round(ev, 4),
        "n_trades":       int(n),
    }


def aggregate_profit_metrics(batch_metrics: list) -> Dict[str, float]:
    """Weighted average of per-batch profit metrics dicts."""
    if not batch_metrics:
        return {"profit_factor": 1.0, "win_rate": 0.5, "expected_value": 0.0, "n_trades": 0}

    total_n = sum(m["n_trades"] for m in batch_metrics)
    if total_n == 0:
        return {"profit_factor": 1.0, "win_rate": 0.5, "expected_value": 0.0, "n_trades": 0}

    def _wavg(key):
        return sum(m[key] * m["n_trades"] for m in batch_metrics) / total_n

    gross_profit = sum(m["win_rate"] * m["n_trades"] * RR_RATIO for m in batch_metrics)
    gross_loss   = sum((1 - m["win_rate"]) * m["n_trades"] for m in batch_metrics)
    pf = gross_profit / gross_loss if gross_loss > 1e-8 else float("inf")
    wr = _wavg("win_rate")
    ev = wr * RR_RATIO - (1.0 - wr)

    return {
        "profit_factor":  round(pf, 4),
        "win_rate":       round(wr, 4),
        "expected_value": round(ev, 4),
        "n_trades":       total_n,
    }


def profit_grade(pf: float) -> str:
    """Human-readable grade for a profit factor value."""
    if pf == float("inf") or pf >= 3.0:  return "EXCELLENT"
    if pf >= 2.0:                         return "GOOD"
    if pf >= 1.5:                         return "ACCEPTABLE"
    if pf >= 1.0:                         return "MARGINAL"
    return "UNPROFITABLE"
