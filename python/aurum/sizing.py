"""
aurum/sizing.py — L6 position sizing.

Auditable, deterministic sizing — chosen over an RL policy because RL
position sizers are hard to validate and the project's risk discipline
favours sizing you can reason about line by line.

Two signals combine into one lot multiplier:

  1. Quantile-Kelly. The quantile head predicts forward-return quantiles
     {q10, q50, q90}. Estimate a win probability and a payoff ratio from
     them, then apply FRACTIONAL Kelly (Kelly 1956) — never full Kelly,
     which is famously over-levered under parameter uncertainty.

  2. Volatility targeting (Moskowitz, Ooi & Pedersen 2012). Scale inversely
     with recent realised volatility so the position's risk contribution
     is roughly constant across calm and turbulent regimes.

Output is a multiplier in [SIZING_MIN_LOT_MULT, SIZING_MAX_LOT_MULT] that
the EA multiplies onto its base lot. The function is pure arithmetic and
is mirrored in AurumAgent.mqh — no ONNX needed.
"""

from __future__ import annotations

import numpy as np

from aurum.aurum_config import (
    SIZING_KELLY_FRACTION, SIZING_VOL_TARGET, SIZING_MAX_LOT_MULT,
    SIZING_MIN_LOT_MULT,
)


def _kelly_fraction(q10: float, q50: float, q90: float) -> float:
    """
    Estimate a Kelly fraction from forward-return quantiles.

    Treat the trade as a bet whose upside is the q90 excursion and whose
    downside is the |q10| excursion. Win probability is inferred from where
    the median sits between them. Kelly: f* = p/loss - (1-p)/win  ... in
    the standard win/loss-ratio form  f* = p - (1-p)/R  with R = win/loss.
    """
    win = max(1e-6, q90)
    loss = max(1e-6, -q10)
    R = win / loss
    # p: median return mapped through the [q10, q90] span to a probability.
    span = max(1e-9, q90 - q10)
    p = float(np.clip((q50 - q10) / span, 0.05, 0.95))
    f = p - (1.0 - p) / R
    return max(0.0, f)


def lot_multiplier(q10: float, q50: float, q90: float,
                   realized_vol: float) -> float:
    """
    Combined quantile-Kelly + vol-target lot multiplier.

    realized_vol : recent annualised realised volatility of GOLD.
    """
    kelly = _kelly_fraction(q10, q50, q90) * SIZING_KELLY_FRACTION
    vol_scale = SIZING_VOL_TARGET / max(1e-6, realized_vol)
    mult = kelly * vol_scale
    # Kelly alone can be ~0 on a marginal edge; floor keeps a deployed
    # signal from sizing to nothing, cap prevents over-leverage.
    return float(np.clip(mult, SIZING_MIN_LOT_MULT, SIZING_MAX_LOT_MULT))


def sizing_params() -> dict:
    """The sizing constants, for embedding in the spec JSON / EA parity."""
    return {
        "kelly_fraction": SIZING_KELLY_FRACTION,
        "vol_target": SIZING_VOL_TARGET,
        "max_lot_mult": SIZING_MAX_LOT_MULT,
        "min_lot_mult": SIZING_MIN_LOT_MULT,
    }
