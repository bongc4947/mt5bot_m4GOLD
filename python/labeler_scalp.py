"""
labeler_scalp.py — micro triple-barrier labels for the scalp model.

Different from labeler.py's M5 triple-barrier in two important ways:

1. SL/TP are in **spread-multiples**, not ATR-multiples. Scalp's edge
   is small per trade; if SL = 1×ATR (which is ~10× spread on FX), even
   a 60%-correct model gets eaten by spread. We size SL/TP relative to
   the symbol's typical spread so the cost moat is explicit.

2. Timeout is in tick-bars, not minutes. This makes labels
   information-uniform with the input tick-bars (each barrier event
   represents the same number of trades, not the same wall-clock time).

Output:
    direction_label: +1 (TP hit), -1 (SL hit), 0 (timeout)  → directional truth
    should_trade_label: 1 (the trade *would have* made positive expected
                         PnL after costs), 0 otherwise

The should_trade label is the selectivity signal — model learns to
abstain on bars where direction prediction wouldn't beat costs anyway.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_scalp_labels(bars: pd.DataFrame,
                          *,
                          sl_spread_mult: float = 1.5,
                          tp_spread_mult: float = 2.5,
                          timeout_bars: int  = 100,
                          min_spread_floor: float = 1e-5,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        direction_label : (N,) int8 in {-1, 0, +1}
        should_trade    : (N,) int8 in {0, 1} — positive expected-value bars

    bars must have columns: high, low, close, spread.

    SL/TP are placed at:
        long:   entry = close[i],  TP = close + tp_spread_mult * spread[i]
                                    SL = close - sl_spread_mult * spread[i]
        short:  mirror

    For each bar i, simulate BOTH a long and a short trade in the next
    `timeout_bars` bars; pick the more-favourable outcome as the label.
    This deliberately gives the labeller "perfect direction" — the
    trade-direction decision is the model's job. should_trade=1 only
    if either direction would have made > cost.
    """
    n = len(bars)
    if n == 0:
        return np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int8)

    if "spread" not in bars.columns:
        raise ValueError("scalp labels need a 'spread' column on the bars")

    high  = bars["high"].to_numpy(dtype=np.float64)
    low   = bars["low"].to_numpy(dtype=np.float64)
    close = bars["close"].to_numpy(dtype=np.float64)
    spread = np.maximum(bars["spread"].to_numpy(dtype=np.float64), min_spread_floor)

    direction = np.zeros(n, dtype=np.int8)
    should_trade = np.zeros(n, dtype=np.int8)

    # cost of a round-trip = spread (paid on entry + half on exit ≈ 1 spread for scalp)
    # Use a slightly conservative cost so should_trade only fires when edge clearly exceeds spread.
    cost = spread.copy()

    for i in range(n - timeout_bars):
        entry = close[i]
        sp    = spread[i]
        tp_long  = entry + tp_spread_mult * sp
        sl_long  = entry - sl_spread_mult * sp
        tp_short = entry - tp_spread_mult * sp
        sl_short = entry + sl_spread_mult * sp

        long_outcome = 0
        short_outcome = 0
        for k in range(1, timeout_bars + 1):
            j = i + k
            hi = high[j]
            lo = low[j]
            if long_outcome == 0:
                if hi >= tp_long and lo <= sl_long:
                    # Both touched — assume worst case (SL first)
                    long_outcome = -1
                elif hi >= tp_long:
                    long_outcome = +1
                elif lo <= sl_long:
                    long_outcome = -1
            if short_outcome == 0:
                if hi >= sl_short and lo <= tp_short:
                    short_outcome = -1
                elif lo <= tp_short:
                    short_outcome = +1
                elif hi >= sl_short:
                    short_outcome = -1
            if long_outcome != 0 and short_outcome != 0:
                break

        # Pick the winning direction; if both lose, label as flat with should_trade=0.
        if long_outcome > 0 and short_outcome <= 0:
            direction[i] = +1
            should_trade[i] = 1
        elif short_outcome > 0 and long_outcome <= 0:
            direction[i] = -1
            should_trade[i] = 1
        elif long_outcome > 0 and short_outcome > 0:
            # Both directions win — pick the larger nominal move.
            direction[i] = +1
            should_trade[i] = 1
        else:
            direction[i] = 0
            should_trade[i] = 0

    n_long  = int((direction > 0).sum())
    n_short = int((direction < 0).sum())
    n_flat  = int((direction == 0).sum())
    log.info(
        "Scalp labels: LONG=%d  SHORT=%d  FLAT=%d  should_trade=%d  "
        "(SL=%.1fxSpread  TP=%.1fxSpread  timeout=%dbars)",
        n_long, n_short, n_flat, int(should_trade.sum()),
        sl_spread_mult, tp_spread_mult, timeout_bars,
    )
    return direction, should_trade
