"""
labeler_modify.py — Build modification labels from open trade history:
  [move_sl_to_be, trail_sl_pips, close_now]

Derived from MFE/MAE trajectory and direction model confidence changes.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    MOD_BE_MFE_RATIO, MOD_CLOSE_CONF, MOD_CLOSE_MAE_FRAC,
    LABEL_FORWARD_BARS,
)

log = logging.getLogger(__name__)
EPS = 1e-10


def _atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    tr = np.zeros(len(close))
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.zeros(len(close))
    if period < len(tr):
        atr[period] = float(np.mean(tr[1:period + 1]))
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_modify_labels(df: pd.DataFrame,
                           direction_labels: np.ndarray,
                           pip_size: float = 0.0001,
                           exec_sl_labels: Optional[np.ndarray] = None,
                           dir_confidence: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns [N, 3] float32: [move_sl_to_be, trail_sl_pips, close_now]

    For each bar where a trade would be open (direction != 0), simulates
    what the optimal management action would have been.

    exec_sl_labels: optional [N] SL pips from labeler_exec (used as initial SL reference)
    dir_confidence: optional [N] direction confidence from direction model
    """
    close  = df["close"].to_numpy(dtype=np.float64)
    high   = df["high"].to_numpy(dtype=np.float64)
    low    = df["low"].to_numpy(dtype=np.float64)
    N      = len(close)

    atr14 = _atr_series(high, low, close, 14)
    labels = np.zeros((N, 3), dtype=np.float32)

    for i in range(N - LABEL_FORWARD_BARS):
        dir_i = int(direction_labels[i]) if direction_labels is not None else 0
        if dir_i == 0:
            continue

        atr_i = atr14[i] if atr14[i] > 0 else pip_size
        atr_pips = atr_i / pip_size

        # Reconstruct initial SL from exec labels or fallback
        initial_sl_pips = (float(exec_sl_labels[i])
                           if exec_sl_labels is not None and exec_sl_labels[i] > 0
                           else 1.5 * atr_pips)

        fwd_high  = high[i + 1: i + 1 + LABEL_FORWARD_BARS]
        fwd_low   = low[i + 1:  i + 1 + LABEL_FORWARD_BARS]
        fwd_close = close[i + 1: i + 1 + LABEL_FORWARD_BARS]

        if dir_i == 1:
            mfe_pips = max(0.0, (np.max(fwd_high) - close[i]) / pip_size)
            mae_pips = max(0.0, (close[i] - np.min(fwd_low)) / pip_size)
            # Find bar of max favorable excursion
            mfe_bar = int(np.argmax(fwd_high))
            # Price at MFE peak
            mfe_peak = float(np.max(fwd_high))
        else:
            mfe_pips = max(0.0, (close[i] - np.min(fwd_low)) / pip_size)
            mae_pips = max(0.0, (np.max(fwd_high) - close[i]) / pip_size)
            mfe_bar  = int(np.argmin(fwd_low))
            mfe_peak = float(np.min(fwd_low))

        # --- MOVE_SL_TO_BE ---
        # Trigger when MFE >= initial_sl_pips (trade is in profit >= risk)
        # AND not within 3 bars of expected TP (last 15% of forward window)
        be_threshold = MOD_BE_MFE_RATIO * initial_sl_pips
        close_to_tp = mfe_bar >= int(LABEL_FORWARD_BARS * 0.85)
        move_be = 1.0 if (mfe_pips >= be_threshold and not close_to_tp) else 0.0
        labels[i, 0] = move_be

        # --- TRAIL_SL_PIPS ---
        # Optimal trail = price at MFE peak - 0.5×ATR (for long)
        # Normalize to pips; 0 = no trail (haven't reached trail activation)
        if mfe_pips >= be_threshold and atr_pips > 0:
            trail_pips = max(0.0, 0.5 * atr_pips)
        else:
            trail_pips = 0.0
        labels[i, 1] = float(trail_pips)

        # --- CLOSE_NOW ---
        # Trigger if:
        #   - Direction model flips sign with confidence >= threshold, OR
        #   - Floating PnL < -0.7 × initial_sl_pips (approaching SL)
        conf_i = float(dir_confidence[i]) if dir_confidence is not None else 0.0
        # Simulate a forward direction flip: if the last quarter of fwd window reverses
        if len(fwd_close) >= 4:
            late_return = (fwd_close[-1] - fwd_close[-4]) / (fwd_close[-4] + EPS)
            direction_flipped = (dir_i == 1 and late_return < 0) or (dir_i == -1 and late_return > 0)
        else:
            direction_flipped = False

        approaching_sl = (mae_pips > MOD_CLOSE_MAE_FRAC * initial_sl_pips)
        close_now = 1.0 if (direction_flipped and conf_i >= MOD_CLOSE_CONF) or approaching_sl else 0.0
        labels[i, 2] = close_now

    n_be    = int(np.sum(labels[:, 0] > 0.5))
    n_close = int(np.sum(labels[:, 2] > 0.5))
    n_trail = int(np.sum(labels[:, 1] > 0))
    log.info("Modify labels: move_to_be=%d  trail=%d  close_now=%d  (N=%d)",
             n_be, n_trail, n_close, N)

    return labels
