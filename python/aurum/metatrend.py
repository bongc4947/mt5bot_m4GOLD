"""
aurum/metatrend.py — the meta-trend strategy core.

The validated GOLD edge (see docs/RESEARCH_FINDINGS.md): a deterministic
SLOW trend rule supplies the direction; an ML meta-gate decides WHEN to
trust it. Net PF ~1.2-1.27 under leak-free purged CV, every fold positive.

This module is the single source of truth for three things, each of
which MUST be reproduced bit-for-bit by the EA (ea/includes/MetaGate.mqh):

  1. PRIMARY_PARAMS   — the trend rule (EMA fast/slow cross).
  2. META_FEATURES    — the 18 causal features fed to the meta-gate.
  3. build_features() — how those features are computed (all backward-
                        looking; no leak).

Keep this list and the MQL5 implementation in lock-step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy constants — mirrored in MetaGate.mqh
# ---------------------------------------------------------------------------
PRIMARY_EMA_FAST = 50          # EMA fast span (M5 bars)
PRIMARY_EMA_SLOW = 200         # EMA slow span
LABEL_HORIZON    = 240         # forward bars the meta-label scores (~20 h)
META_ACT_THRESHOLD = 0.55      # EA acts on the trend only when P(act) >= this
COST_ROUNDTRIP   = 1.8e-4      # spread assumption used in training/eval

# Canonical feature order — the EA must build this vector identically.
META_FEATURES = [
    "ret12", "ret48", "ret96",
    "atr14_norm", "atr48_norm",
    "rv24", "rv96",
    "pos96", "pos288",
    "ema_fast_dist", "ema_slow_dist", "ema_gap",
    "bars_since_hi96", "bars_since_lo96",
    "trend_age", "up_streak",
    "hod_sin", "hod_cos",
]
N_META_FEATURES = len(META_FEATURES)


# ---------------------------------------------------------------------------
# Primary trend signal — deterministic, the EA computes this directly
# ---------------------------------------------------------------------------
def primary_signal(close: np.ndarray) -> np.ndarray:
    """+1 when EMA(fast) > EMA(slow), else -1. The trade direction."""
    ef = pd.Series(close).ewm(span=PRIMARY_EMA_FAST, adjust=False).mean()
    es = pd.Series(close).ewm(span=PRIMARY_EMA_SLOW, adjust=False).mean()
    return np.where(ef.to_numpy() > es.to_numpy(), 1, -1).astype(np.int64)


# ---------------------------------------------------------------------------
# Causal feature builder — every column uses only past/current closed bars
# ---------------------------------------------------------------------------
def build_features(m5: pd.DataFrame) -> np.ndarray:
    """Return float32[N, N_META_FEATURES] in the exact META_FEATURES order."""
    o = m5["open"].to_numpy(np.float64)
    h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64)
    c = m5["close"].to_numpy(np.float64)
    t = pd.to_datetime(m5["time"], utc=True)
    eps = 1e-12
    n = len(c)
    logc = np.log(np.clip(c, eps, None))

    def ret(k):
        r = np.zeros(n)
        r[k:] = logc[k:] - logc[:-k]
        return r

    def roll(a, k, fn):
        return pd.Series(a).rolling(k, min_periods=max(2, k // 2)).agg(fn).to_numpy()

    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = roll(tr, 14, "mean")
    atr48 = roll(tr, 48, "mean")
    r1 = ret(1)

    ef = pd.Series(c).ewm(span=PRIMARY_EMA_FAST, adjust=False).mean().to_numpy()
    es = pd.Series(c).ewm(span=PRIMARY_EMA_SLOW, adjust=False).mean().to_numpy()
    prim = np.where(ef > es, 1, -1)

    hi96 = roll(h, 96, "max")
    lo96 = roll(l, 96, "min")
    hi288 = roll(h, 288, "max")
    lo288 = roll(l, 288, "min")

    # bars since the rolling 96-bar high / low (normalised 0..1)
    def bars_since(extreme_is_high: bool):
        out = np.zeros(n)
        win = 96
        for i in range(n):
            lo = max(0, i - win + 1)
            seg = (h if extreme_is_high else l)[lo:i + 1]
            k = (np.argmax(seg) if extreme_is_high else np.argmin(seg))
            out[i] = (len(seg) - 1 - k) / win
        return out

    # trend age — bars since the last EMA cross (normalised, capped at 1)
    trend_age = np.zeros(n)
    age = 0
    for i in range(1, n):
        age = 0 if prim[i] != prim[i - 1] else age + 1
        trend_age[i] = min(age / 200.0, 1.0)

    # consecutive same-direction closes (capped)
    up_streak = np.zeros(n)
    s = 0
    for i in range(1, n):
        if c[i] > c[i - 1]:
            s = s + 1 if s >= 0 else 1
        elif c[i] < c[i - 1]:
            s = s - 1 if s <= 0 else -1
        else:
            s = 0
        up_streak[i] = np.clip(s / 10.0, -1.0, 1.0)

    hod = t.dt.hour.to_numpy() + t.dt.minute.to_numpy() / 60.0

    cols = {
        "ret12": ret(12), "ret48": ret(48), "ret96": ret(96),
        "atr14_norm": atr14 / np.clip(c, eps, None),
        "atr48_norm": atr48 / np.clip(c, eps, None),
        "rv24": roll(r1, 24, "std"), "rv96": roll(r1, 96, "std"),
        "pos96": (c - lo96) / np.clip(hi96 - lo96, eps, None),
        "pos288": (c - lo288) / np.clip(hi288 - lo288, eps, None),
        "ema_fast_dist": (c - ef) / np.clip(atr14, eps, None),
        "ema_slow_dist": (c - es) / np.clip(atr14, eps, None),
        "ema_gap": (ef - es) / np.clip(atr14, eps, None),
        "bars_since_hi96": bars_since(True),
        "bars_since_lo96": bars_since(False),
        "trend_age": trend_age,
        "up_streak": up_streak,
        "hod_sin": np.sin(2 * np.pi * hod / 24.0),
        "hod_cos": np.cos(2 * np.pi * hod / 24.0),
    }
    X = np.column_stack([cols[name] for name in META_FEATURES]).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Meta-label — "would following the primary over LABEL_HORIZON have won?"
# ---------------------------------------------------------------------------
def build_meta_label(m5: pd.DataFrame) -> dict:
    c = m5["close"].to_numpy(np.float64)
    n = len(c)
    eps = 1e-12
    fwd = np.zeros(n)
    fwd[:n - LABEL_HORIZON] = np.log(
        np.clip(c[LABEL_HORIZON:], eps, None)
        / np.clip(c[:n - LABEL_HORIZON], eps, None))
    prim = primary_signal(c)
    # net of cost — a trade only counts as a win if it beats the spread
    y = ((prim * fwd - COST_ROUNDTRIP) > 0).astype(np.int64)
    return {"y": y, "primary": prim, "fwd_ret": fwd}
