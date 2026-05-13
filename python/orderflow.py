"""
orderflow.py — order-flow microstructure features derived from raw ticks
*without* needing Level-2 depth-of-book.

For each tick we infer trade direction (buyer-initiated vs seller-initiated)
via a Lee-Ready-style rule:
    - if last >= ask  (or last closer to ask)  -> +1  (buyer-initiated)
    - if last <= bid  (or last closer to bid)  -> -1  (seller-initiated)
    - otherwise                                ->  0  (mid-tick / unknown)

From the signed tick stream we aggregate per tick-bar:
    ofi               — sum of signed_volume / sum of |volume| in [-1, 1]
    cvd               — cumulative volume delta normalized by rolling-1k delta
    taker_ratio       — fraction of ticks classified as buyer- or seller-initiated
    microprice_drift  — last_microprice - first_microprice (volume-weighted mid)
    spread_regime     — current bar mean spread / rolling-1h median spread
    rv_30bars         — backward realised vol over last 30 bars (sqrt of squared returns)
    rv_120bars        — backward realised vol over last 120 bars

Plus 8 MTF context columns from multi_timeframe.py = 15 microstructure +
context features per tick-bar.

Strictly causal — every feature uses only data available *up to and
including* the tick-bar's close. The EA mirrors this exactly: maintain
a rolling buffer of the last K ticks per symbol and compute the same
aggregations on every new bar close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ORDERFLOW_FEATURE_COLUMNS = [
    "ofi", "cvd", "taker_ratio",
    "microprice_drift", "spread_regime",
    "rv_30", "rv_120",
]
ORDERFLOW_FEATURE_DIM = len(ORDERFLOW_FEATURE_COLUMNS)


def _classify_ticks(ticks: pd.DataFrame) -> np.ndarray:
    """
    Lee-Ready trade direction. +1 buyer, -1 seller, 0 unknown.
    Uses (last - mid) sign as the primary rule; tie-breaks fall to 0
    (this differs from textbook Lee-Ready which uses prior-tick test
    on ties — for our purposes 0 is fine since we average over a bar).
    """
    bid = ticks["bid"].to_numpy(dtype=np.float64)
    ask = ticks["ask"].to_numpy(dtype=np.float64)
    last = ticks.get("last", (bid + ask) / 2.0)
    if isinstance(last, pd.Series):
        last = last.to_numpy(dtype=np.float64)
    mid = (bid + ask) / 2.0
    signed = np.zeros(len(ticks), dtype=np.int8)
    signed[last >  mid + 1e-12] = 1
    signed[last <  mid - 1e-12] = -1
    return signed


def aggregate_orderflow_to_bars(ticks: pd.DataFrame,
                                 ticks_per_bar: int = 100,
                                 ) -> pd.DataFrame:
    """
    Aggregate raw ticks to per-tick-bar order-flow features. Returns a
    DataFrame of length N // ticks_per_bar with the columns listed in
    ORDERFLOW_FEATURE_COLUMNS.

    Designed to run alongside aggregate_ticks_to_bars: same bar
    boundaries, same row count. The resulting DataFrame is hstacked
    onto the OHLCV bars by run_tick_pipeline.
    """
    needed = {"bid", "ask"}
    if not needed.issubset(ticks.columns):
        raise ValueError(f"ticks missing {needed - set(ticks.columns)}")

    n_bars = len(ticks) // ticks_per_bar
    if n_bars == 0:
        raise ValueError(f"need >= {ticks_per_bar} ticks; got {len(ticks)}")

    df = ticks.iloc[: n_bars * ticks_per_bar]
    bid = df["bid"].to_numpy(dtype=np.float64)
    ask = df["ask"].to_numpy(dtype=np.float64)
    spread = (ask - bid)
    mid    = (bid + ask) / 2.0
    vol    = (df["volume"].to_numpy(dtype=np.float64)
              if "volume" in df.columns else np.ones(len(df)))
    vol    = np.where(vol <= 0, 1.0, vol)

    signed = _classify_ticks(df).astype(np.float64)

    # Microprice = volume-weighted mid (proxy when no L2 depth).
    micro = mid    # without bid_size/ask_size we just use mid; placeholder

    # Reshape per-tick arrays into (n_bars, ticks_per_bar) for fast aggregation.
    shape = (n_bars, ticks_per_bar)
    signed_t = signed.reshape(shape)
    spread_t = spread.reshape(shape)
    vol_t    = vol.reshape(shape)
    micro_t  = micro.reshape(shape)
    mid_t    = mid.reshape(shape)

    # OFI: signed-volume sum / |volume| sum, clipped to [-1, 1].
    sv = (signed_t * vol_t).sum(axis=1)
    av = vol_t.sum(axis=1)
    ofi = np.clip(sv / np.where(av > 0, av, 1.0), -1.0, 1.0)

    # Taker ratio: fraction of ticks where direction was non-zero.
    taker_ratio = (signed_t != 0).mean(axis=1).astype(np.float64)

    # Cumulative volume delta — bar's signed volume, then cumulative
    # over bars normalized by a rolling 1000-bar absolute-delta sum.
    bar_delta = sv
    cum_delta = np.cumsum(bar_delta)
    norm = pd.Series(np.abs(bar_delta)).rolling(1000, min_periods=1).sum().to_numpy()
    cvd = np.clip(cum_delta / np.where(norm > 0, norm, 1.0), -10.0, 10.0) / 10.0

    # Microprice drift over the bar — last - first.
    microprice_drift = (micro_t[:, -1] - micro_t[:, 0]) / np.where(
        micro_t[:, 0] > 0, micro_t[:, 0], 1.0)
    microprice_drift = np.clip(microprice_drift * 1e4, -10.0, 10.0) / 10.0  # scale ~ pips

    # Spread regime — bar-mean / 1-hour rolling median (proxy: 60 bars
    # if ticks_per_bar = 100; close enough — exact mapping happens at
    # the model side via additional rescaling if needed).
    bar_spread = spread_t.mean(axis=1)
    rolling_median = pd.Series(bar_spread).rolling(60, min_periods=5).median().to_numpy()
    spread_regime = np.clip(bar_spread / np.where(rolling_median > 1e-12,
                                                    rolling_median, 1.0),
                              0.0, 5.0) / 5.0

    # Backward realised vol — sqrt of sum of squared close-to-close
    # returns over the last 30 / 120 bars. Closes here = mid at bar end.
    bar_close = mid_t[:, -1]
    log_ret = np.diff(np.log(np.where(bar_close > 0, bar_close, 1.0)),
                       prepend=np.log(bar_close[0] if bar_close[0] > 0 else 1.0))
    sq = log_ret ** 2
    rv_30  = np.sqrt(pd.Series(sq).rolling(30,  min_periods=1).sum().to_numpy())
    rv_120 = np.sqrt(pd.Series(sq).rolling(120, min_periods=1).sum().to_numpy())
    # Scale to bounded range — typical FX bar log-vol ~ 1e-4 .. 5e-3
    rv_30  = np.clip(rv_30  * 1e3, 0.0, 10.0) / 10.0
    rv_120 = np.clip(rv_120 * 1e3, 0.0, 10.0) / 10.0

    return pd.DataFrame({
        "ofi":               ofi.astype(np.float32),
        "cvd":               cvd.astype(np.float32),
        "taker_ratio":       taker_ratio.astype(np.float32),
        "microprice_drift":  microprice_drift.astype(np.float32),
        "spread_regime":     spread_regime.astype(np.float32),
        "rv_30":             rv_30.astype(np.float32),
        "rv_120":            rv_120.astype(np.float32),
    })


def build_orderflow_block(bars: pd.DataFrame) -> np.ndarray:
    """Return the (N, 7) order-flow block from a tick-bar DataFrame."""
    n = len(bars)
    out = np.zeros((n, ORDERFLOW_FEATURE_DIM), dtype=np.float32)
    for i, c in enumerate(ORDERFLOW_FEATURE_COLUMNS):
        if c in bars.columns:
            out[:, i] = bars[c].to_numpy(dtype=np.float32)
    return out
