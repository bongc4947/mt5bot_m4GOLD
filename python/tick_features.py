"""
tick_features.py — aggregate raw tick streams into M5-aligned micro features.

Given a tick parquet (from fetch_ticks.py) and an M5 bars parquet, produce
one feature row per M5 bar describing the *intra-bar* tick behaviour.

PROPOSED — NOT YET WIRED INTO TRAINING
--------------------------------------
This module produces a set of tick-derived features that *would* extend
the model input from 200 dims to ~240. We are NOT wiring it into the
trainer in this commit because the EA-side MQL5 implementation does not
yet compute the same features at inference — adding them training-only
would re-introduce the parity gap mk4.3 just closed.

To deploy these:
  1. Implement an MQL5 mirror in FeatureEncoder.mqh that produces the
     same 40 dims at live inference (using OnTick events to maintain
     a per-symbol tick ring).
  2. Bump FEATURE_DIM_DIR from 200 to 240 in config.py + Defines.mqh.
  3. Append these features to dir_feat in feature_engine.py.
  4. Retrain.

For now this file is reference + research. It can also be useful as a
diagnostic — compute on training data and inspect distributions before
committing to live inference.

WHAT IT COMPUTES (40 dims per M5 bar)
-------------------------------------
0  tick_count                  number of ticks in the M5 window
1  trade_tick_ratio            COPY_TICKS_TRADE flagged / total
2  micro_price_open            volume-weighted mid at bar open
3  micro_price_close           volume-weighted mid at bar close
4  micro_price_high            max micro-price during bar
5  micro_price_low             min micro-price
6  micro_range_pips            (high - low) in pips
7  spread_mean_pips            mean(ask - bid) over bar
8  spread_std_pips             stdev of spread within bar
9  spread_max_pips             max spread within bar (vol spike proxy)
10 mid_returns_mean            mean log-return between consecutive ticks
11 mid_returns_std             stdev of tick-to-tick log-returns
12 mid_returns_skew            skew (asymmetry) of tick-return distribution
13 mid_returns_kurt            kurtosis (tail mass)
14 abs_returns_mean            mean(|tick log-return|) — micro-volatility
15 signed_volume_imbalance     trade ticks closer to ask vs bid (proxy for OFI)
16 lee_ready_buy_ratio         Lee-Ready trade-classification: buys / total
17 tick_arrival_mean_ms        mean inter-tick interval (ms)
18 tick_arrival_std_ms         stdev of inter-tick interval
19 effective_spread_pips       2 * |last - mid| averaged
20 quote_revisions_per_sec     count of bid/ask updates per second
21 micro_drift                 sign(close - open) * range/(spread+eps)
22-39 reserved                 future micro-features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TICK_FEATURE_DIM = 40

# 1 pip in price units — caller supplies via pip_size.
EPS = 1e-12


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else 0.0


def aggregate_ticks_to_m5(
    ticks: pd.DataFrame,
    bar_times: pd.DatetimeIndex,
    pip_size: float,
) -> np.ndarray:
    """
    Bucket `ticks` into M5 bars whose open-times are given by `bar_times`.

    Args:
        ticks    : DataFrame with columns ['time', 'bid', 'ask', 'last',
                   'volume', 'flags']. 'time' must be timezone-aware UTC.
        bar_times: M5 bar open times (timezone-aware UTC), ascending.
        pip_size : 0.0001 for most FX, 0.01 for JPY/metals, 1.0 for BTC.

    Returns: float32 array shape (len(bar_times), TICK_FEATURE_DIM).
             Bars with zero ticks are zero-filled.
    """
    if ticks.empty:
        return np.zeros((len(bar_times), TICK_FEATURE_DIM), dtype=np.float32)

    if not pd.api.types.is_datetime64tz_dtype(ticks["time"]):
        raise ValueError("ticks['time'] must be tz-aware UTC")

    ticks = ticks.sort_values("time").reset_index(drop=True)
    times = ticks["time"].values.astype("datetime64[ns]")
    bid   = ticks["bid"].to_numpy(dtype=np.float64)
    ask   = ticks["ask"].to_numpy(dtype=np.float64)
    last  = ticks.get("last", pd.Series(0.0, index=ticks.index)).to_numpy(dtype=np.float64)
    vol   = ticks.get("volume", pd.Series(0.0, index=ticks.index)).to_numpy(dtype=np.float64)
    flags = ticks.get("flags", pd.Series(0, index=ticks.index)).to_numpy(dtype=np.int64)

    mid    = (bid + ask) * 0.5
    spread = (ask - bid)

    out = np.zeros((len(bar_times), TICK_FEATURE_DIM), dtype=np.float32)

    # bucket index per tick (left-edge: tick belongs to bar whose open <= tick.time
    # < open + 5min). np.searchsorted on bar_times gives that.
    bar_arr = bar_times.to_numpy().astype("datetime64[ns]")
    idx = np.searchsorted(bar_arr, times, side="right") - 1
    valid = (idx >= 0) & (idx < len(bar_arr))

    for b in range(len(bar_arr)):
        mask = (idx == b) & valid
        n = int(mask.sum())
        if n == 0:
            continue

        m_mid    = mid[mask]
        m_spread = spread[mask]
        m_last   = last[mask]
        m_vol    = vol[mask]
        m_flags  = flags[mask]
        m_times  = times[mask]

        out[b, 0]  = n
        is_trade   = (m_flags & 0x4).astype(bool)   # COPY_TICKS_TRADE bit
        out[b, 1]  = is_trade.mean() if n > 0 else 0.0

        # micro price (volume weighted) — fall back to plain mean if no volume
        v_pos = m_vol > 0
        if v_pos.any():
            mp = (m_mid[v_pos] * m_vol[v_pos]).sum() / m_vol[v_pos].sum()
        else:
            mp = m_mid.mean()
        out[b, 2]  = m_mid[0]
        out[b, 3]  = m_mid[-1]
        out[b, 4]  = m_mid.max()
        out[b, 5]  = m_mid.min()
        out[b, 6]  = (m_mid.max() - m_mid.min()) / pip_size

        out[b, 7]  = m_spread.mean() / pip_size
        out[b, 8]  = m_spread.std()  / pip_size if n > 1 else 0.0
        out[b, 9]  = m_spread.max()  / pip_size

        if n > 1:
            ret = np.diff(np.log(np.maximum(m_mid, EPS)))
            out[b, 10] = ret.mean()
            out[b, 11] = ret.std()
            if n > 3 and ret.std() > EPS:
                # skew/kurt — guard against zero-variance windows
                z = (ret - ret.mean()) / (ret.std() + EPS)
                out[b, 12] = float((z ** 3).mean())
                out[b, 13] = float((z ** 4).mean()) - 3.0   # excess kurtosis
            out[b, 14] = np.abs(ret).mean()

        # signed volume imbalance: trade ticks closer to ask = +1, closer to bid = -1
        if is_trade.any():
            mid_at_trade = (bid[mask][is_trade] + ask[mask][is_trade]) * 0.5
            last_at_trade = m_last[is_trade]
            sign = np.sign(last_at_trade - mid_at_trade)   # >0 = buy aggressive
            wt   = m_vol[mask][is_trade] if False else m_vol[is_trade]   # noqa
            # simpler: just count +/-
            out[b, 15] = float(sign.sum() / max(is_trade.sum(), 1))
            out[b, 16] = float((sign > 0).mean())          # Lee-Ready buy ratio

        # tick arrival inter-times in ms
        if n > 1:
            dt_ms = np.diff(m_times.astype("int64")) / 1_000_000.0
            out[b, 17] = float(dt_ms.mean())
            out[b, 18] = float(dt_ms.std()) if dt_ms.size > 1 else 0.0

        # effective spread proxy
        if is_trade.any():
            es = 2.0 * np.abs(m_last[is_trade] - (bid[mask][is_trade] + ask[mask][is_trade]) * 0.5)
            out[b, 19] = float(es.mean()) / pip_size

        # quote revisions per second
        if n > 1:
            dt_total_s = (m_times[-1] - m_times[0]).astype("timedelta64[s]").astype(float)
            if dt_total_s > 0:
                out[b, 20] = n / dt_total_s

        # micro drift (signed range / spread)
        if out[b, 7] > EPS:
            sgn = np.sign(out[b, 3] - out[b, 2])
            out[b, 21] = sgn * out[b, 6] / max(out[b, 7], EPS)

    return out


# ---------------------------------------------------------------------------
# Convenience: combine multiple per-month tick parquets + bars parquet
# ---------------------------------------------------------------------------

def aggregate_directory(
    tick_dir: Path,
    symbol: str,
    bar_times: pd.DatetimeIndex,
    pip_size: float,
) -> np.ndarray:
    """Concat all <symbol>_*.parquet under tick_dir and aggregate to bar_times."""
    files = sorted(tick_dir.glob(f"{symbol}_*.parquet"))
    if not files:
        log.warning("No tick parquet for %s under %s — returning zeros", symbol, tick_dir)
        return np.zeros((len(bar_times), TICK_FEATURE_DIM), dtype=np.float32)
    dfs = [pd.read_parquet(p) for p in files]
    ticks = pd.concat(dfs, ignore_index=True)
    log.info("[%s] aggregating %d ticks across %d files",
             symbol, len(ticks), len(files))
    return aggregate_ticks_to_m5(ticks, bar_times, pip_size)
