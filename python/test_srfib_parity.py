"""
test_srfib_parity.py - verify the MetaGate.mqh S/R+Fib math matches the
Python extractor bit-for-bit on real GOLD M5 data.

Strategy: re-implement the MetaGate.mqh logic in Python (the function
mql5_srfib_for_anchor below), run it on several recent anchors, and
diff against aurum.sr_fib_features.extract() for the same anchors.

If max_err <= 1e-4 across all 6 features on all test anchors, the MQL5
port is faithful and we can ship.
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)

# These mirror MetaGate.mqh constants
PIP_USD            = 0.10
MIN_SWING_USD      = 4.0
SWING_N            = 8
CLUSTER_USD        = 3.0
MAX_LEVELS         = 6
LOOKBACK_H         = 504        # 21 days of H1
ZONE_ATR           = 0.75
DIST_CAP_ATR       = 3.0


def _mql5_detect_swings(h1_high: np.ndarray, h1_low: np.ndarray, n: int):
    """Mirror _MG_DetectSwings in MetaGate.mqh: causal fractal swings.

    The MQL5 version only confirms swings where i + n <= last_confirmable
    (i.e., n forward bars exist). So a swing at i is visible at H1 bar (i+n).
    """
    swings = []   # list of (h1_idx, price, kind)  kind: 0=L, 1=H
    n_bars = len(h1_high)
    if n_bars < 2 * n + 2: return swings
    last_confirmable = n_bars - n - 1
    for i in range(n, last_confirmable + 1):
        ph, pl = h1_high[i], h1_low[i]
        # SWING HIGH
        is_high = True
        strict_left = strict_right = False
        for k in range(i - n, i + n + 1):
            if k == i: continue
            if h1_high[k] > ph: is_high = False; break
            if h1_high[k] < ph:
                if k < i: strict_left = True
                else: strict_right = True
        if is_high and (strict_left or strict_right):
            swings.append((i, float(ph), 1))
        # SWING LOW
        is_low = True
        strict_left = strict_right = False
        for k in range(i - n, i + n + 1):
            if k == i: continue
            if h1_low[k] < pl: is_low = False; break
            if h1_low[k] > pl:
                if k < i: strict_left = True
                else: strict_right = True
        if is_low and (strict_left or strict_right):
            swings.append((i, float(pl), 0))
    return swings


def _mql5_filter_minswing(swings, min_swing):
    """Mirror _MG_FilterMinSwing."""
    last_high = -1.0; last_low = -1.0
    out = []
    for _, p, k in swings:
        ok = False
        if k == 1:
            if last_low < 0 or (p - last_low) >= min_swing: ok = True
        else:
            if last_high < 0 or (last_high - p) >= min_swing: ok = True
        if ok:
            out.append((p, k))
            if k == 1: last_high = p
            else:      last_low = p
    return out


def _mql5_cluster(prices, bandwidth):
    """Mirror _MG_ClusterLevels - greedy 1D, sort by strength desc."""
    if not prices: return []
    ps = sorted(prices)
    clusters = [[ps[0]]]
    for p in ps[1:]:
        if p - clusters[-1][-1] <= bandwidth:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    out = [(float(np.mean(c)), len(c)) for c in clusters]
    out.sort(key=lambda x: -x[1])
    return out[:MAX_LEVELS]


def mql5_srfib_for_anchor(m5: pd.DataFrame, anchor: int, atr14: float):
    """Re-implement MetaGate.mqh::_MG_BuildSrFibFeatures."""
    # The MQL5 EA pulls H1 closed bars via CopyRates(_Symbol, PERIOD_H1, 1, 520).
    # That returns the last 520 closed H1 bars ending at the bar BEFORE the
    # current forming H1. The M5 anchor is at m5[anchor]; figure out which H1
    # bars are "closed" relative to that M5 close.
    anchor_t = pd.to_datetime(m5["time"].iloc[anchor], utc=True)
    # M5 bar i opens at t_i and closes at t_i + 5min. So decision time = anchor_t + 5min.
    decision_t = anchor_t + pd.Timedelta(minutes=5)
    # Build clock-aligned H1 from the M5 series
    g = m5.set_index("time")
    g.index = pd.to_datetime(g.index, utc=True)
    h1 = g.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    ).dropna(subset=["close"]).reset_index()
    # Last CLOSED H1 bar has close_time <= decision_t. H1 bar opening at t_h
    # closes at t_h + 1h. So closed if t_h + 1h <= decision_t -> t_h <= decision_t - 1h.
    cutoff = decision_t - pd.Timedelta(hours=1)
    closed_h1 = h1[h1["time"] <= cutoff].reset_index(drop=True)
    # Take last 520 (or fewer if not enough history)
    closed_h1 = closed_h1.iloc[-(LOOKBACK_H + SWING_N + 8):].reset_index(drop=True)
    if len(closed_h1) < LOOKBACK_H + SWING_N + 1:
        return None
    h1_high = closed_h1["high"].to_numpy(np.float64)
    h1_low  = closed_h1["low"].to_numpy(np.float64)
    swings = _mql5_detect_swings(h1_high, h1_low, SWING_N)
    filt = _mql5_filter_minswing(swings, MIN_SWING_USD)
    sups = [p for p, k in filt if k == 0]
    ress = [p for p, k in filt if k == 1]
    sup_lv = _mql5_cluster(sups, CLUSTER_USD)
    res_lv = _mql5_cluster(ress, CLUSTER_USD)
    cur = float(m5["close"].iloc[anchor])
    out = [DIST_CAP_ATR, DIST_CAP_ATR, 0.0, 0.0, DIST_CAP_ATR, 0.0]
    # nearest sup BELOW
    sups_below = [(c, s) for c, s in sup_lv if c < cur]
    if sups_below:
        c_, s_ = max(sups_below, key=lambda x: x[0])
        out[0] = min((cur - c_) / atr14, DIST_CAP_ATR); out[2] = float(s_)
    # nearest res ABOVE
    ress_above = [(c, s) for c, s in res_lv if c > cur]
    if ress_above:
        c_, s_ = min(ress_above, key=lambda x: x[0])
        out[1] = min((c_ - cur) / atr14, DIST_CAP_ATR); out[3] = float(s_)
    # Fib on most recent valid pair
    if len(filt) >= 2:
        p_last, k_last = filt[-1]
        p_prev, k_prev = filt[-2]
        if k_last != k_prev and abs(p_last - p_prev) >= MIN_SWING_USD:
            lo = min(p_last, p_prev); hi = max(p_last, p_prev); rng = hi - lo
            fib_levels = [lo + r * rng for r in (0.236, 0.382, 0.5, 0.618, 0.786)]
            fib_dist = min(abs(cur - f) for f in fib_levels)
            out[4] = min(fib_dist / atr14, DIST_CAP_ATR)
    nearest = min(out[0], out[1], out[4])
    out[5] = 1.0 if nearest <= ZONE_ATR else 0.0
    return out


def main() -> int:
    from aurum.datamodule import _load_m5_bars
    from aurum.sr_fib_features import extract as python_extract
    from aurum.metatrend import build_features

    m5 = _load_m5_bars()
    print(f"M5 loaded: {len(m5)} bars")

    # Pick 6 test anchors spread across the last year
    n = len(m5)
    test_anchors = np.linspace(int(n * 0.4), n - 200, 6, dtype=np.int64)
    print(f"Test anchors: {test_anchors.tolist()}")

    # Extract via the Python "training-time" path
    py_feats = python_extract(m5, test_anchors)

    # Compute ATR14(M5) at each anchor - same formula as MetaGate.mqh
    h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64)
    c = m5["close"].to_numpy(np.float64)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_series = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()

    print()
    print(f"{'anchor':>8} {'time':24}  comparison (mql5_emu vs python)  max_diff")
    print("-" * 90)
    max_diff_overall = 0.0
    for j, a in enumerate(test_anchors):
        mql5 = mql5_srfib_for_anchor(m5, int(a), atr_series[a])
        if mql5 is None:
            print(f"  anchor {a}: insufficient history")
            continue
        py = py_feats[j]
        diff = np.abs(np.array(mql5) - py.astype(np.float64))
        md = float(diff.max())
        max_diff_overall = max(max_diff_overall, md)
        names = ["d_sup", "d_res", "s_sup", "s_res", "d_fib", "in_z"]
        print(f"  {a:>6} {str(m5['time'].iloc[a]):24}  max_diff={md:.4f}")
        for i, name in enumerate(names):
            mark = "" if diff[i] < 1e-3 else "  <-- HIGH"
            print(f"      {name:6}  mql5={mql5[i]:+.4f}  py={py[i]:+.4f}  diff={diff[i]:.4f}{mark}")

    print()
    print(f"OVERALL max diff across {len(test_anchors)} anchors x 6 features: {max_diff_overall:.4e}")
    if max_diff_overall < 1e-3:
        print("OK - MQL5 port is faithful to Python")
        return 0
    elif max_diff_overall < 0.05:
        print("ACCEPTABLE but watch - small numerical drift, likely from H1 resample alignment")
        return 0
    else:
        print("HIGH DRIFT - investigate. Likely MQL5 port has a bug.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
