"""
labeler.py — Direction labels with mk4 enhancements:
  - Sharpe-adjusted (|return| > 0.5×ATR AND forward Sharpe > 0.3)
  - Risk-penalized (max DD within forward window > 2×ATR → flat)
  - Regime-conditional resampling (BULL/BEAR/SIDEWAYS)
  - Trade outcome injection (override with actual EA trade results)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    LABEL_FORWARD_BARS, LABEL_SHARPE_MIN, LABEL_ATR_THRESH,
    LABEL_DD_PENALIZE, LABEL_ATR_PERIOD,
    LABEL_EARLY_BARS, LABEL_EARLY_ATR_MIN,
    LABEL_TB_SL_ATR, LABEL_TB_TP_ATR, LABEL_TB_USE,
)

log = logging.getLogger(__name__)

EPS = 1e-10


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HMM-based regime detection (simple 3-state Gaussian HMM)
# ---------------------------------------------------------------------------

def detect_regime(close: np.ndarray, window: int = 20,
                  z_thresh: float = 0.3) -> np.ndarray:
    """
    Returns regime array: 0=BULL, 1=SIDEWAYS, 2=BEAR for each bar.

    Uses the z-score of the rolling 20-bar return (mean / std) to detect
    direction-of-recent-trend. mk4.3.2: threshold lowered from 0.5 to 0.3
    so that ~10-15% of bars get classified as BULL or BEAR (was ~0.7%
    each — too restrictive to be a useful diagnostic). The label
    distribution itself is unaffected; this only changes the regime
    breakdown print and any feature that uses regime as input.

    Note: this detector classifies *what just happened*. On EURUSD M5,
    forward direction tends to mean-revert against the recent trend
    (statistically ~9-12 sigma). That's a feature the model exploits, not
    a bug.
    """
    N = len(close)
    lr = np.zeros(N)
    for i in range(1, N):
        lr[i] = (close[i] - close[i - 1]) / (close[i - 1] + EPS)

    regime = np.ones(N, dtype=np.int32)  # default SIDEWAYS

    roll_ret = pd.Series(lr).rolling(window, min_periods=window // 2).mean().to_numpy()
    roll_std = pd.Series(lr).rolling(window, min_periods=window // 2).std().to_numpy()
    roll_std = np.where(roll_std < EPS, EPS, roll_std)
    z = roll_ret / roll_std

    for i in range(N):
        if np.isnan(z[i]):
            regime[i] = 1
        elif z[i] > z_thresh:
            regime[i] = 0   # BULL
        elif z[i] < -z_thresh:
            regime[i] = 2   # BEAR
        else:
            regime[i] = 1   # SIDEWAYS

    return regime


# ---------------------------------------------------------------------------
# Triple-barrier labelling (Lopez de Prado AFML Ch.3) — mk4.4
# ---------------------------------------------------------------------------

def _compute_triple_barrier_labels(
        close: np.ndarray, high: np.ndarray, low: np.ndarray,
        atr: np.ndarray, forward_bars: int,
        sl_atr_mult: float, tp_atr_mult: float,
) -> np.ndarray:
    """
    Triple-barrier labelling.

    For each bar i, simulate opening BOTH a long and a short with:
        SL = entry +/- sl_atr_mult * atr[i]
        TP = entry +/- tp_atr_mult * atr[i]
    Walk forward up to forward_bars and check which barrier is hit first
    (using high/low intraday-extremes, not close).

    Label:
        +1  if the long side hit TP before SL AND the short side did not
        -1  if the short side hit TP before SL AND the long side did not
         0  otherwise (timeout, both sides win, or both sides lose)

    With sensible RR (e.g. 2:1) the "both win" case is mathematically
    impossible within the same window, so 0 mostly means "no clean signal".

    Implementation note: the inner per-bar walk is vectorised via
    `argmax` against boolean barrier-hit masks, which is ~50x faster
    than a pure Python loop while remaining easy to read.
    """
    N = len(close)
    labels = np.zeros(N, dtype=np.int8)

    for i in range(N - forward_bars):
        if atr[i] <= 0:
            continue
        sl_dist = sl_atr_mult * atr[i]
        tp_dist = tp_atr_mult * atr[i]

        h_win = high[i + 1: i + 1 + forward_bars]
        l_win = low [i + 1: i + 1 + forward_bars]

        entry = close[i]

        # --- LONG ---
        long_tp_hits = h_win >= (entry + tp_dist)
        long_sl_hits = l_win <= (entry - sl_dist)
        # argmax on a bool array returns 0 if no True, so guard via .any()
        long_tp_idx = long_tp_hits.argmax() if long_tp_hits.any() else forward_bars
        long_sl_idx = long_sl_hits.argmax() if long_sl_hits.any() else forward_bars
        long_outcome = (+1 if long_tp_idx < long_sl_idx else
                        -1 if long_sl_idx < long_tp_idx else 0)

        # --- SHORT ---
        short_tp_hits = l_win <= (entry - tp_dist)
        short_sl_hits = h_win >= (entry + sl_dist)
        short_tp_idx = short_tp_hits.argmax() if short_tp_hits.any() else forward_bars
        short_sl_idx = short_sl_hits.argmax() if short_sl_hits.any() else forward_bars
        short_outcome = (+1 if short_tp_idx < short_sl_idx else
                         -1 if short_sl_idx < short_tp_idx else 0)

        if long_outcome == +1 and short_outcome != +1:
            labels[i] = +1
        elif short_outcome == +1 and long_outcome != +1:
            labels[i] = -1
        # else: leave 0 (no clean directional signal)

    return labels


# ---------------------------------------------------------------------------
# Core label computation
# ---------------------------------------------------------------------------

def compute_direction_labels(df: pd.DataFrame,
                              forward_bars: int = LABEL_FORWARD_BARS,
                              atr_period: int = LABEL_ATR_PERIOD,
                              sharpe_min: float = LABEL_SHARPE_MIN,
                              atr_thresh: float = LABEL_ATR_THRESH,
                              dd_penalize: float = LABEL_DD_PENALIZE,
                              signal_log_df: Optional[pd.DataFrame] = None,
                              regime_resample: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (labels [N], regime [N]) arrays.
    labels: +1=LONG, -1=SHORT, 0=FLAT

    mk4.4: when LABEL_TB_USE is True (default), uses triple-barrier labels
    that mirror what the trader actually does at runtime: simulate a
    SL/TP exit on every bar, label = which barrier was hit first. This
    eliminates the train/live mismatch that the legacy ATR-Sharpe filter
    had. Fall back to the legacy labeler if LABEL_TB_USE=False.
    """
    close  = df["close"].to_numpy(dtype=np.float64)
    high   = df["high"].to_numpy(dtype=np.float64)
    low    = df["low"].to_numpy(dtype=np.float64)
    N      = len(close)

    atr    = _atr_series(high, low, close, atr_period)
    regime = detect_regime(close)

    # mk4.4 #1: route to triple-barrier when configured.
    if LABEL_TB_USE:
        labels = _compute_triple_barrier_labels(
            close, high, low, atr,
            forward_bars=forward_bars,
            sl_atr_mult=LABEL_TB_SL_ATR,
            tp_atr_mult=LABEL_TB_TP_ATR,
        )
        # Trade-outcome injection (no-op since signal_log_df=None in train_agent)
        if signal_log_df is not None and len(signal_log_df) > 0:
            times = df.get("time", pd.Series([None] * N))
            _inject_trade_outcomes(labels, times, signal_log_df, close)
        if regime_resample:
            labels = _resample_by_regime(labels, regime)
        label_counts = {v: int(np.sum(labels == v)) for v in [-1, 0, 1]}
        log.info("Triple-barrier labels: LONG=%d  FLAT=%d  SHORT=%d  "
                 "(SL=%gxATR  TP=%gxATR  RR=%g  N=%d)",
                 label_counts[1], label_counts[0], label_counts[-1],
                 LABEL_TB_SL_ATR, LABEL_TB_TP_ATR,
                 LABEL_TB_TP_ATR / LABEL_TB_SL_ATR, N)
        return labels.astype(np.float32), regime

    # --- Legacy ATR-Sharpe labeller (LABEL_TB_USE=False) ---
    labels = np.zeros(N, dtype=np.int8)

    for i in range(N - forward_bars):
        fwd_window_c = close[i + 1: i + 1 + forward_bars]
        fwd_window_h = high[i + 1: i + 1 + forward_bars]
        fwd_window_l = low[i + 1:  i + 1 + forward_bars]

        fwd_return = (fwd_window_c[-1] - close[i]) / (close[i] + EPS)
        fwd_returns = np.diff(np.concatenate([[close[i]], fwd_window_c])) / (close[i] + EPS)

        atr_i = atr[i] if atr[i] > 0 else (abs(fwd_return) + EPS)

        # --- Sharpe-adjusted filter ---
        if abs(fwd_return) < atr_thresh * atr_i / close[i]:
            labels[i] = 0
            continue

        fwd_std = float(np.std(fwd_returns)) if len(fwd_returns) > 1 else EPS
        fwd_sharpe = float(np.mean(fwd_returns)) / (fwd_std + EPS)
        if abs(fwd_sharpe) < sharpe_min:
            labels[i] = 0
            continue

        # --- Risk-penalty: max adverse drawdown within forward window ---
        direction = 1 if fwd_return > 0 else -1
        if direction == 1:
            # for long: adverse = close[i] - min(low)
            mae = (close[i] - np.min(fwd_window_l)) / (close[i] + EPS)
        else:
            mae = (np.max(fwd_window_h) - close[i]) / (close[i] + EPS)

        if mae > dd_penalize * atr_i / close[i]:
            labels[i] = 0
            continue

        # --- Early-onset momentum filter ---
        # The move must START within the first LABEL_EARLY_BARS bars.
        # Without this, bars labelled LONG/SHORT at the beginning of a
        # consolidation zone qualify (the eventual move is real, but price
        # wanders flat or against the trade for the first 5-15 bars, hitting
        # the SL before the actual leg begins).
        # Require: mean(close[i+1 … i+LABEL_EARLY_BARS]) is at least
        #          LABEL_EARLY_ATR_MIN × ATR away from close[i] in the
        #          expected direction.
        early_end = min(i + 1 + LABEL_EARLY_BARS, len(close))
        early_mean = float(np.mean(close[i + 1: early_end]))
        early_drift = (early_mean - close[i]) * direction   # positive = correct direction
        if early_drift < LABEL_EARLY_ATR_MIN * atr_i:
            labels[i] = 0
            continue

        labels[i] = direction

    # --- Trade outcome injection ---
    if signal_log_df is not None and len(signal_log_df) > 0:
        times = df.get("time", pd.Series([None] * N))
        _inject_trade_outcomes(labels, times, signal_log_df, close)

    # --- Regime-conditional resampling ---
    if regime_resample:
        labels = _resample_by_regime(labels, regime)

    label_counts = {v: int(np.sum(labels == v)) for v in [-1, 0, 1]}
    log.info("Labels: LONG=%d  FLAT=%d  SHORT=%d  (N=%d)",
             label_counts[1], label_counts[0], label_counts[-1], N)

    return labels.astype(np.float32), regime


def _inject_trade_outcomes(labels: np.ndarray,
                           times: pd.Series,
                           resolved_df: pd.DataFrame,
                           close: np.ndarray):
    """
    Override synthetic direction labels with actual EA trade outcomes.

    Source: HYDRA4_resolved_signals.csv (parse_resolved_signals_log).
    Each row links a model-signal timestamp to the actual pips outcome.
    Bars that match (within 1 bar = 5 min) have their label replaced with the
    real win/loss direction, giving the model feedback from live trading.

    Previously read from the signal log which has no 'actual_pips' column —
    now correctly reads from the resolved signals file.
    """
    ts_col = "timestamp_signal" if "timestamp_signal" in resolved_df.columns else "timestamp"
    if ts_col not in resolved_df.columns:
        return
    if "actual_pips" not in resolved_df.columns:
        return

    # Build lookup: timestamp → outcome direction (+1 / -1 / 0)
    trade_outcomes = {}
    closed = resolved_df.dropna(subset=["actual_pips", ts_col])
    for _, row in closed.iterrows():
        ts = row[ts_col]
        pips = float(row["actual_pips"])
        trade_outcomes[ts] = 1 if pips > 0 else (-1 if pips < 0 else 0)

    if not trade_outcomes:
        return

    injected = 0
    for i, t in enumerate(times):
        if t is None:
            continue
        if hasattr(t, "to_pydatetime"):
            t = t.to_pydatetime()
        for ts_key, outcome in trade_outcomes.items():
            try:
                diff = abs((t - ts_key).total_seconds())
                if diff <= 300:  # within one M5 bar
                    labels[i] = outcome
                    injected += 1
                    break
            except Exception:
                pass

    log.info("Injected %d trade outcomes from resolved signals log", injected)


def _resample_by_regime(labels: np.ndarray, regime: np.ndarray) -> np.ndarray:
    """
    Log label distribution per regime (BULL/SIDEWAYS/BEAR).
    Actual class balancing is handled by WeightedRandomSampler in dataset.py.
    """
    regime_names = {0: "BULL", 1: "SIDEWAYS", 2: "BEAR"}
    for r, rname in regime_names.items():
        mask = regime == r
        sub  = labels[mask]
        if len(sub) == 0:
            continue
        n_long  = int(np.sum(sub == 1))
        n_short = int(np.sum(sub == -1))
        n_flat  = int(np.sum(sub == 0))
        log.info("  Regime %-10s  LONG=%d  SHORT=%d  FLAT=%d",
                 rname, n_long, n_short, n_flat)
    return labels
