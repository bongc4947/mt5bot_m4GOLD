"""
multi_timeframe.py — align H1 + H4 context features onto tick-bar timestamps.

For each tick-bar at time t, look up the most recent H1 / H4 bar with
time <= t (strictly causal — no lookahead) and append 8 bounded
[-1, 1]-ish features:

    h1_trend, h1_rsi, h1_atr_norm, h1_vwap_rel
    h4_trend, h4_rsi, h4_atr_norm, h4_vwap_rel

Why these 8:
  trend     — sign × magnitude of return over last 20 bars (direction bias)
  rsi       — overbought / oversold (RSI(14) mapped to [-1, 1])
  atr_norm  — ATR(14) / price (volatility regime)
  vwap_rel  — (close - rolling_vwap) / vwap (mean-reversion proxy)

The idea: the scalp model conditions on "what regime is the broader
market in right now?" without owning multi-timeframe history itself.
Live, the EA refreshes these on every new H1/H4 close — same alignment,
same features.

Used by `data_pipeline.run_tick_pipeline` when MT5 returns valid H1 / H4
bars; falls back to zero-filled columns if either timeframe is unavailable
so downstream consumers always see the same schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-timeframe primitives
# ---------------------------------------------------------------------------

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI(14), simple-moving-average flavour. Returns values in [0, 100]."""
    delta = np.diff(close, prepend=close[0])
    up = pd.Series(np.where(delta > 0, delta, 0.0)).rolling(period, min_periods=1).mean().to_numpy()
    dn = pd.Series(np.where(delta < 0, -delta, 0.0)).rolling(period, min_periods=1).mean().to_numpy()
    rs = up / (dn + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> np.ndarray:
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close),
    ])
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def _compute_tf_features(bars: pd.DataFrame,
                          trend_lookback: int = 20,
                          vwap_window: int   = 50) -> pd.DataFrame:
    """
    Compute the 4 standard features (trend, rsi, atr_norm, vwap_rel) for
    one timeframe's bars. All outputs bounded to roughly [-1, 1] so the
    model doesn't see wildly different magnitudes between H1 and H4.
    """
    close = bars["close"].to_numpy(dtype=np.float64)
    high  = bars["high"].to_numpy(dtype=np.float64)
    low   = bars["low"].to_numpy(dtype=np.float64)
    n = len(close)

    # 1. Trend — signed normalized return over last `trend_lookback` bars.
    #    tanh squashes large moves; multiplier picks the saturation knee.
    if n > trend_lookback:
        prior = np.concatenate([np.full(trend_lookback, close[0]),
                                close[:-trend_lookback]])
    else:
        prior = np.full(n, close[0])
    trend = np.tanh(((close - prior) / (prior + 1e-12)) * 50.0)

    # 2. RSI normalized to [-1, 1] (50 → 0, 100 → +1, 0 → -1)
    rsi_raw = _rsi(close, 14)
    rsi = (rsi_raw - 50.0) / 50.0

    # 3. ATR / price; clipped & rescaled so 0..0.05 → 0..1
    atr_raw = _atr(high, low, close, 14)
    atr_norm = np.clip(atr_raw / (close + 1e-12), 0.0, 0.05) / 0.05

    # 4. VWAP-relative position: (close - vwap) / vwap, ×100 for FX-scale,
    #    clipped so a 1% deviation maps to 1.0.
    if "tick_volume" in bars.columns:
        vol = bars["tick_volume"].to_numpy(dtype=np.float64)
    else:
        vol = np.ones_like(close)
    vol = np.where(vol <= 0, 1.0, vol)        # broker quirk: zero-volume rows
    pv = close * vol
    sum_pv  = pd.Series(pv).rolling(vwap_window, min_periods=1).sum().to_numpy()
    sum_vol = pd.Series(vol).rolling(vwap_window, min_periods=1).sum().to_numpy()
    vwap = sum_pv / np.clip(sum_vol, 1e-12, None)
    vwap_rel = np.clip((close - vwap) / (vwap + 1e-12) * 100.0, -1.0, 1.0)

    return pd.DataFrame({
        "trend":    trend.astype(np.float32),
        "rsi":      rsi.astype(np.float32),
        "atr_norm": atr_norm.astype(np.float32),
        "vwap_rel": vwap_rel.astype(np.float32),
    })


# ---------------------------------------------------------------------------
# Public alignment function
# ---------------------------------------------------------------------------

MTF_FEATURE_COLUMNS = [
    "h1_trend", "h1_rsi", "h1_atr_norm", "h1_vwap_rel",
    "h4_trend", "h4_rsi", "h4_atr_norm", "h4_vwap_rel",
]
MTF_FEATURE_DIM = len(MTF_FEATURE_COLUMNS)


def build_mtf_block(bars: pd.DataFrame) -> np.ndarray:
    """
    Read the MTF feature columns from a tick-bar DataFrame. Returns an
    (N, 8) float32 matrix in the canonical column order. Missing columns
    are zero-filled (so old parquets without MTF still work — schema
    stable, values neutral).

    Sub-phase 1a.3: this is what the new scalp/hedge model architectures
    consume. The 200-dim parity-floored direction feature stays
    untouched (preserves MQL5 EA contract); MTF flows as a separate
    block that scalp/hedge models concatenate on the input side.
    """
    n = len(bars)
    out = np.zeros((n, MTF_FEATURE_DIM), dtype=np.float32)
    for i, c in enumerate(MTF_FEATURE_COLUMNS):
        if c in bars.columns:
            out[:, i] = bars[c].to_numpy(dtype=np.float32)
    return out


def align_mtf_features(tick_bars: pd.DataFrame,
                        h1_bars: pd.DataFrame | None = None,
                        h4_bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Append H1 + H4 context features to a tick-bar DataFrame.

    For every row in `tick_bars`, the most recent H1 / H4 bar with
    `time <= row.time` is found via `pd.merge_asof(direction='backward')`
    and its 4 timeframe features are copied across. **No lookahead** —
    the EA can reproduce this exactly at runtime by reading the most
    recent closed H1 / H4 bar.

    If `h1_bars` (or `h4_bars`) is None or empty, the corresponding 4
    columns are filled with zeros. The schema (8 columns,
    MTF_FEATURE_COLUMNS) is always the same — downstream code never has
    to branch on whether MTF was available.

    Returns a new DataFrame; does not mutate the input.
    """
    if "time" not in tick_bars.columns:
        raise ValueError("tick_bars must have a 'time' column for merge_asof")

    out = tick_bars.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.sort_values("time").reset_index(drop=True)

    for prefix, mtf in (("h1", h1_bars), ("h4", h4_bars)):
        if mtf is None or len(mtf) == 0:
            for suffix in ("trend", "rsi", "atr_norm", "vwap_rel"):
                out[f"{prefix}_{suffix}"] = np.float32(0.0)
            continue

        mtf = mtf.copy()
        if "time" not in mtf.columns:
            raise ValueError(f"{prefix} bars must have a 'time' column")
        mtf["time"] = pd.to_datetime(mtf["time"], utc=True)
        mtf = mtf.sort_values("time").reset_index(drop=True)

        feats = _compute_tf_features(mtf)
        feats.columns = [f"{prefix}_{c}" for c in feats.columns]
        feats["time"] = mtf["time"].to_numpy()

        # Backward as-of merge: each tick-bar gets the most recent MTF
        # feature row whose time is <= the tick-bar's time.
        out = pd.merge_asof(
            out, feats,
            on="time",
            direction="backward",
        )

    # merge_asof leaves NaN where the tick-bar is *earlier* than the
    # first available MTF bar (warmup edge). Fill with zeros — same
    # behaviour as the missing-MTF branch above.
    for col in MTF_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna(0.0).astype(np.float32)

    return out
