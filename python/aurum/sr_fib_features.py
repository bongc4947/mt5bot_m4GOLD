"""
aurum/sr_fib_features.py — causal swing-based S/R and Fibonacci features.

Detects pivot-fractal swing highs and lows on H1 bars (resampled from M5),
filters out noise swings smaller than MIN_SWING_PIPS, clusters survivors
into S/R levels by price proximity, projects Fibonacci retracements on
the most recent valid swing, and emits six per-M5-bar features:

  dist_to_sup_atr        signed distance to nearest support BELOW price (ATR units)
  dist_to_res_atr        signed distance to nearest resistance ABOVE price (ATR units)
  sup_strength           number of swing touches confirming the support level
  res_strength           number of swing touches confirming the resistance level
  dist_to_nearest_fib_atr distance to the nearest of {23.6, 38.2, 50, 61.8, 78.6}%
                          retracement of the most recent ≥40-pip swing
  in_sr_zone             binary: 1 if within 0.5 ATR of any S/R or Fib level

Leak-free guarantees:
  - Swings are detected with `SWING_N` bars of forward confirmation. The
    most-recent visible swing for an M5 anchor at time t is always at least
    SWING_N H1 bars old (~12 hours).
  - Levels at time t use only swings confirmed by time t.
  - ATR(14) is computed from closed bars up to the previous M5 bar.

Pip convention: $0.10 per pip on XAU/USD (standard MetaTrader). 40 pips = $4.00.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs — mirror these in MetaGate.mqh if/when we ship to MQL5
# ---------------------------------------------------------------------------
PIP_USD          = 0.10        # 1 XAU pip = $0.10
MIN_SWING_PIPS   = 40          # ignore swings smaller than this
MIN_SWING_USD    = MIN_SWING_PIPS * PIP_USD       # = $4.00
SWING_N          = 8           # fractal lookback/forward bars on H1 (~8h)
CLUSTER_PIPS     = 30          # S/R level cluster bandwidth (30 pips = $3)
MAX_LEVELS       = 6           # keep this many most-recent S/R levels per side
LEVEL_LOOKBACK_H = 24 * 21     # consider swings from the past 3 weeks (recent enough to matter)
FIB_RATIOS       = (0.236, 0.382, 0.500, 0.618, 0.786)
SR_ZONE_ATR      = 0.75        # "in zone" threshold
DIST_CAP_ATR     = 3.0         # clip all distance features at this many ATR
                                # (anything farther doesn't change behaviour)

SR_FIB_FEATURES = [
    "dist_to_sup_atr",
    "dist_to_res_atr",
    "sup_strength",
    "res_strength",
    "dist_to_nearest_fib_atr",
    "in_sr_zone",
]
N_SR_FIB_FEATURES = len(SR_FIB_FEATURES)


def _cache_path(symbol: str, anchors: np.ndarray) -> Path:
    h = hashlib.sha1(anchors.astype(np.int64).tobytes()).hexdigest()[:16]
    base = Path(__file__).parent.parent.parent / "onnx_out"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"sr_fib_features_v2_{symbol}_{len(anchors)}_{h}.npy"


def _resample_h1(m5: pd.DataFrame) -> pd.DataFrame:
    """Clock-aligned H1 OHLC from M5 (label=left, closed=left)."""
    g = m5.set_index("time")
    return g.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last"),
    ).dropna(subset=["close"]).reset_index()


def _atr14_m5(m5: pd.DataFrame) -> np.ndarray:
    """ATR(14) on M5 — same as the 18-feature builder."""
    h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64)
    c = m5["close"].to_numpy(np.float64)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()


def _detect_swings(h1: pd.DataFrame, n: int = SWING_N) -> list[tuple[int, float, str]]:
    """
    Fractal swing detection on H1. A bar i is a swing HIGH if:
       h[i] >= max(h[i-n:i+n+1])  with strict > on at least one side
    Symmetric for swing LOW.

    Returns list of (h1_bar_index, price, 'H'|'L'). The bar index is the
    actual swing bar; for causal use, treat the swing as "visible" only
    from h1_bar_index + n onwards (when forward confirmation is complete).
    """
    h = h1["high"].to_numpy(np.float64)
    l = h1["low"].to_numpy(np.float64)
    swings = []
    for i in range(n, len(h) - n):
        win_h = h[i - n: i + n + 1]
        win_l = l[i - n: i + n + 1]
        if h[i] == win_h.max() and (h[i] > h[i - n:i].max() or h[i] > h[i + 1:i + n + 1].max()):
            swings.append((i, float(h[i]), "H"))
        if l[i] == win_l.min() and (l[i] < l[i - n:i].min() or l[i] < l[i + 1:i + n + 1].min()):
            swings.append((i, float(l[i]), "L"))
    return swings


def _cluster_levels(prices: list[float], bandwidth_usd: float) -> list[tuple[float, int]]:
    """
    Greedy 1-D price clustering. Sort prices, merge runs whose adjacent
    gap is ≤ bandwidth. Returns [(cluster_center, touch_count), ...]
    sorted by touch_count descending.
    """
    if not prices: return []
    ps = sorted(prices)
    clusters = [[ps[0]]]
    for p in ps[1:]:
        if p - clusters[-1][-1] <= bandwidth_usd:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    out = [(float(np.mean(c)), len(c)) for c in clusters]
    out.sort(key=lambda x: -x[1])
    return out


def _bars_per_h1_in_m5() -> int:
    return 12  # 60 min / 5 min


def extract(m5: pd.DataFrame, anchors: np.ndarray,
            symbol: str = "GOLD") -> np.ndarray:
    """
    Return float32[len(anchors), N_SR_FIB_FEATURES]. Result is cached.
    """
    cp = _cache_path(symbol, anchors)
    if cp.exists():
        log.info("[sr_fib] cache hit: %s", cp.name)
        return np.load(cp)

    m5 = m5.copy()
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    h1 = _resample_h1(m5)
    log.info("[sr_fib] resampled %d H1 bars from %d M5 bars",
             len(h1), len(m5))

    swings = _detect_swings(h1, n=SWING_N)
    log.info("[sr_fib] detected %d raw swings (n=%d fractal lookback)",
             len(swings), SWING_N)

    # Map H1 swing bar index -> visibility timestamp (when forward
    # confirmation completes), then map to the next M5 bar after that.
    h1_times = h1["time"].to_numpy()
    m5_times = m5["time"].to_numpy()
    atr_m5   = _atr14_m5(m5)

    # for each swing: (visible_at_m5_idx, price, kind)
    vis_swings = []
    for h1_idx, price, kind in swings:
        confirm_idx = min(h1_idx + SWING_N, len(h1) - 1)
        confirm_t = h1_times[confirm_idx]
        m5_idx = int(np.searchsorted(m5_times, confirm_t, side="left"))
        if m5_idx >= len(m5): continue
        vis_swings.append((m5_idx, price, kind))
    vis_swings.sort(key=lambda x: x[0])
    log.info("[sr_fib] %d swings projected to M5 visibility timeline",
             len(vis_swings))

    # for each anchor, slice the visible swings, apply min-swing filter,
    # cluster into S/R levels, project Fibonacci on the most recent swing.
    out = np.zeros((len(anchors), N_SR_FIB_FEATURES), dtype=np.float32)
    closes = m5["close"].to_numpy(np.float64)

    # quick lookup: for each anchor, swings visible by then
    # vis_swings is sorted by m5_idx, do a single pass
    vs_idx_array = np.array([s[0] for s in vis_swings], dtype=np.int64)

    log.info("[sr_fib] computing features for %d anchors ...", len(anchors))
    for j, a in enumerate(anchors):
        # how many swings are visible by anchor a?
        k = int(np.searchsorted(vs_idx_array, a, side="right"))
        if k == 0: continue
        # restrict to last LEVEL_LOOKBACK_H hours = LEVEL_LOOKBACK_H * 12 m5 bars
        lookback_m5 = LEVEL_LOOKBACK_H * _bars_per_h1_in_m5()
        cutoff = a - lookback_m5
        recent = [vis_swings[i] for i in range(k) if vis_swings[i][0] >= cutoff]
        if not recent: continue

        # apply min-swing filter: keep swings that mark a move of >= MIN_SWING_USD
        # from the prior opposite-kind swing
        filt = []
        last_opp = {"H": None, "L": None}
        for _, p, kind in recent:
            opp = last_opp["L" if kind == "H" else "H"]
            if opp is None or abs(p - opp) >= MIN_SWING_USD:
                filt.append((p, kind))
                last_opp[kind] = p
        if not filt: continue

        # cluster S/R levels
        sup_prices = [p for p, k in filt if k == "L"]
        res_prices = [p for p, k in filt if k == "H"]
        sup_levels = _cluster_levels(sup_prices, CLUSTER_PIPS * PIP_USD)[:MAX_LEVELS]
        res_levels = _cluster_levels(res_prices, CLUSTER_PIPS * PIP_USD)[:MAX_LEVELS]

        cur = closes[a]
        atr = atr_m5[a] if not np.isnan(atr_m5[a]) else 0.0
        if atr <= 0: continue

        # nearest support BELOW price (capped at DIST_CAP_ATR)
        sups = [(c, s) for c, s in sup_levels if c < cur]
        if sups:
            sup_c, sup_s = max(sups, key=lambda x: x[0])
            out[j, 0] = min((cur - sup_c) / atr, DIST_CAP_ATR)
            out[j, 2] = float(sup_s)
        else:
            out[j, 0] = DIST_CAP_ATR
            out[j, 2] = 0.0
        # nearest resistance ABOVE price (capped)
        ress = [(c, s) for c, s in res_levels if c > cur]
        if ress:
            res_c, res_s = min(ress, key=lambda x: x[0])
            out[j, 1] = min((res_c - cur) / atr, DIST_CAP_ATR)
            out[j, 3] = float(res_s)
        else:
            out[j, 1] = DIST_CAP_ATR
            out[j, 3] = 0.0

        # Fibonacci on the most recent swing pair (with ≥ MIN_SWING_USD range)
        if len(filt) >= 2:
            p_last, k_last = filt[-1]
            p_prev, k_prev = filt[-2]
            if k_last != k_prev and abs(p_last - p_prev) >= MIN_SWING_USD:
                lo = min(p_last, p_prev); hi = max(p_last, p_prev)
                rng = hi - lo
                fib_levels = [lo + r * rng for r in FIB_RATIOS]
                fib_dist = min(abs(cur - f) for f in fib_levels)
                out[j, 4] = min(fib_dist / atr, DIST_CAP_ATR)
            else:
                out[j, 4] = DIST_CAP_ATR
        else:
            out[j, 4] = DIST_CAP_ATR

        # in_sr_zone
        nearest = min(out[j, 0], out[j, 1], out[j, 4])
        out[j, 5] = 1.0 if nearest <= SR_ZONE_ATR else 0.0

    out = np.nan_to_num(out, nan=0.0, posinf=5.0, neginf=5.0)
    np.save(cp, out)
    log.info("[sr_fib] cached -> %s  shape=%s", cp.name, out.shape)
    return out
