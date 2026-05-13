"""
labeler_exec.py — Build execution labels from bar history:
  [timing, sl_pips, tp_pips, vol_mult, session_gate]

All derived from MAE/MFE analysis over historical price paths.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    EXEC_TIMING_BARS, EXEC_TIMING_THRESH,
    EXEC_SL_SAFETY, EXEC_SL_MIN_ATR, EXEC_SL_MAX_ATR, EXEC_SL_MAX_ATR_BY_CLASS,
    EXEC_TP_CONSERVATIVE, EXEC_TP_MIN_ATR, EXEC_TP_MAX_ATR, EXEC_TP_RR_FLOOR,
    EXEC_VOL_CLAMP_LO, EXEC_VOL_CLAMP_HI,
    EXEC_SESSION_SPREAD_MAX, EXEC_ROLLOVER_MIN,
    MAX_RISK_PER_TRADE, LABEL_FORWARD_BARS,
)

_SYMBOL_ASSET_CLASS = {
    "BTCUSD": "crypto", "ETHUSD": "crypto", "LTCUSD": "crypto",
    "CrudeOIL": "energy", "BRENT_OIL": "energy", "NATURAL_GAS": "energy",
    "US_500": "indices", "UK_100": "indices",
    "GOLD": "metals", "SILVER": "metals", "PLATINUM": "metals", "COPPER": "metals",
}

def _sl_max_atr(symbol: str) -> float:
    ac = _SYMBOL_ASSET_CLASS.get(symbol, "forex")
    return EXEC_SL_MAX_ATR_BY_CLASS.get(ac, EXEC_SL_MAX_ATR)
from market_hours import MarketHoursEncoder

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


def compute_exec_labels(df: pd.DataFrame,
                        direction_labels: np.ndarray,
                        pip_size: float = 0.0001,
                        median_spread_pips: float = 1.0,
                        mhe: Optional[MarketHoursEncoder] = None,
                        symbol: str = "") -> np.ndarray:
    """
    Returns [N, 5] float32 array: [timing, sl_pips, tp_pips, vol_mult, session_gate]
    Only non-flat direction bars get meaningful SL/TP labels; flat bars get zeros.
    """
    close  = df["close"].to_numpy(dtype=np.float64)
    high   = df["high"].to_numpy(dtype=np.float64)
    low    = df["low"].to_numpy(dtype=np.float64)
    spread = df.get("spread", pd.Series(np.zeros(len(df)))).to_numpy(np.float64)
    N = len(close)

    atr14 = _atr_series(high, low, close, 14)
    labels = np.zeros((N, 5), dtype=np.float32)
    _mhe = mhe or MarketHoursEncoder()

    times = df.get("time", pd.Series([None] * N))

    for i in range(N - max(EXEC_TIMING_BARS, LABEL_FORWARD_BARS)):
        atr_i = atr14[i] if atr14[i] > 0 else pip_size
        dir_i = int(direction_labels[i]) if direction_labels is not None else 0

        # --- SESSION GATE ---
        ts = times.iloc[i]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        live_spread_pips = float(spread[i]) * close[i] / pip_size if spread[i] > 0 else 0.0
        mh = _mhe.encode(ts, symbol, live_spread_pips=live_spread_pips) if ts else {}

        session_ok = bool(
            (mh.get("session_london") or mh.get("session_ny"))
            and live_spread_pips <= EXEC_SESSION_SPREAD_MAX * median_spread_pips
            and not mh.get("is_holiday_risk", False)
            and mh.get("minutes_to_rollover", 1.0) > (EXEC_ROLLOVER_MIN / 1440.0)
        )
        session_gate = 1.0 if session_ok else 0.0
        labels[i, 4] = session_gate

        # --- TIMING ---
        # Compare entry at bar i vs i+1, i+2, i+3
        if dir_i != 0 and i + EXEC_TIMING_BARS < N:
            if dir_i == 1:
                # Long: best entry = lowest close in window
                entry_now  = close[i]
                best_entry = min(close[i + 1: i + 1 + EXEC_TIMING_BARS])
                improvement = (entry_now - best_entry) / (atr_i + EPS)
            else:
                entry_now  = close[i]
                best_entry = max(close[i + 1: i + 1 + EXEC_TIMING_BARS])
                improvement = (best_entry - entry_now) / (atr_i + EPS)

            if improvement <= 0:
                timing = 1.0   # already at best price
            elif improvement < EXEC_TIMING_THRESH:
                timing = 0.8   # minor improvement possible — strong momentum
            elif improvement < EXEC_TIMING_THRESH * 2:
                timing = 0.5   # limit order worthwhile
            else:
                timing = 0.2   # significantly better entry available — wait
            labels[i, 0] = timing
        else:
            labels[i, 0] = 0.5   # neutral for flat bars

        # --- SL / TP (only for non-flat) ---
        if dir_i == 0 or i + LABEL_FORWARD_BARS >= N:
            labels[i, 1] = 1.0   # fallback: 1.0 × ATR (neutral SL multiple)
            labels[i, 2] = 1.5   # fallback: 1.5 × ATR (neutral TP multiple)
            labels[i, 3] = 1.0   # vol_mult neutral
            continue

        fwd_high = high[i + 1: i + 1 + LABEL_FORWARD_BARS]
        fwd_low  = low[i + 1:  i + 1 + LABEL_FORWARD_BARS]
        fwd_close = close[i + 1: i + 1 + LABEL_FORWARD_BARS]

        if dir_i == 1:
            mae_price = close[i] - np.min(fwd_low)
            mfe_price = np.max(fwd_high) - close[i]
        else:
            mae_price = np.max(fwd_high) - close[i]
            mfe_price = close[i] - np.min(fwd_low)

        mae_pips = max(0.0, mae_price / pip_size)
        mfe_pips = max(0.0, mfe_price / pip_size)
        atr_pips = atr_i / pip_size

        sl_max_atr = _sl_max_atr(symbol)
        sl_raw = mae_pips * EXEC_SL_SAFETY
        sl_raw = max(sl_raw, EXEC_SL_MIN_ATR * atr_pips)
        sl_raw = min(sl_raw, sl_max_atr * atr_pips)

        tp_raw = mfe_pips * EXEC_TP_CONSERVATIVE
        tp_raw = max(tp_raw, EXEC_TP_MIN_ATR * atr_pips)
        tp_raw = min(tp_raw, EXEC_TP_MAX_ATR * atr_pips)
        tp_raw = max(tp_raw, EXEC_TP_RR_FLOOR * sl_raw)

        # Store as ATR multiples so labels are scale-invariant across symbols.
        # Range: sl ∈ [0.5, 4.0], tp ∈ [1.0, 6.0] — same bounds as training clamps.
        # EA multiplies back by ATR(14) pips after inference.
        labels[i, 1] = float(sl_raw / (atr_pips + EPS))
        labels[i, 2] = float(tp_raw / (atr_pips + EPS))

        # --- VOL MULT ---
        # Base lot: 1% risk / sl_pips × some normalization
        # vol_mult adjusts for current vol vs average vol
        atr_avg = float(np.mean(atr14[max(0, i - 20): i + 1])) if i > 0 else atr_i
        vol_mult = float(atr_avg / (atr_i + EPS))
        vol_mult = max(EXEC_VOL_CLAMP_LO, min(EXEC_VOL_CLAMP_HI, vol_mult))
        # Map [0.5, 2.0] → [0, 1] for model output
        labels[i, 3] = (vol_mult - EXEC_VOL_CLAMP_LO) / (EXEC_VOL_CLAMP_HI - EXEC_VOL_CLAMP_LO)

    n_session_open = int(np.sum(labels[:, 4] > 0.5))
    log.info("Exec labels: session_open=%d/%d  avg_sl=%.1f pips  avg_tp=%.1f pips",
             n_session_open, N,
             float(np.mean(labels[labels[:, 1] > 0, 1])) if np.any(labels[:, 1] > 0) else 0,
             float(np.mean(labels[labels[:, 2] > 0, 2])) if np.any(labels[:, 2] > 0) else 0)

    return labels
