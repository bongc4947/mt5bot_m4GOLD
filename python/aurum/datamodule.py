"""
aurum/datamodule.py — multi-timeframe dataset builder for AURUM.

Single source of truth: load GOLD M5 bars once, resample to M15 and H1 in
Python. This guarantees the three timeframes are perfectly aligned (no
broker-side gap mismatch) — every M15 / H1 bar is an exact aggregate of
the M5 bars beneath it.

Output of `build_dataset()`:
  X        dict {tf: float32[N, L_tf, C]}   windowed feature tensors
  X_flat   float32[N, FLAT_INPUT_DIM]       same data, flattened to the
                                            deployed-model input contract
  y_dir    int64[N]    triple-barrier direction label  (0 short/1 flat/2 long)
  y_ret    float32[N]  realised forward return over LABEL_HORIZON_BARS
  y_regime int64[N]    coarse regime label (0 trend-up/1 trend-dn/2 range/3 hi-vol)
  t_index  int64[N]    M5 bar index of each sample (for purged CV ordering)
  norm     dict        per-channel mean/std baked into the spec JSON

For SSL pretraining only the unlabelled X tensors are needed; pass
`labelled=False` to skip the (slower) triple-barrier pass.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurum.aurum_config import (
    CHANNELS, N_CHANNELS, TIMEFRAMES, FLAT_INPUT_DIM, ATR_PERIOD,
    LABEL_HORIZON_BARS, LABEL_TB_SL_ATR, LABEL_TB_TP_ATR,
)

log = logging.getLogger(__name__)

# M5-bar multiples per timeframe (for warmup sizing) + pandas resample rule.
_TF_M5_MULT = {"M5": 1, "M15": 3, "H1": 12}
_TF_RULE    = {"M5": None, "M15": "15min", "H1": "1h"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_m5_bars() -> pd.DataFrame:
    """
    Load the GOLD M5 bar series.

    Prefers a prebuilt M5 parquet; if none exists (e.g. a fresh Kaggle run
    where only raw ticks are mounted), builds M5 bars from the GOLD tick
    parquet via strategies_common.load_or_build_bars — which caches the
    result so subsequent AURUM phases reuse it.
    """
    from config import PARQUET_DIR, TICKS_DIR
    candidates = [
        "HYDRA4_5MFROMTICKS_GOLD.parquet",
        "HYDRA4_M5FROMTICKS_GOLD.parquet",
        "HYDRA4_TBARS_GOLD_100tpb.parquet",   # tick-bars — acceptable fallback
    ]
    # Search the writable cache dir AND the (possibly read-only) dataset
    # dir — a Kaggle dataset may ship prebuilt bars instead of raw ticks.
    df = None
    for d in (PARQUET_DIR, TICKS_DIR):
        for name in candidates:
            p = d / name
            if p.exists():
                df = pd.read_parquet(p)
                log.info("[datamodule] loaded %s (%d bars) from %s",
                         name, len(df), d)
                break
        if df is not None:
            break
    if df is None:
        # No prebuilt bars anywhere — resample from raw GOLD ticks.
        from strategies_common import load_or_build_bars
        log.info("[datamodule] no M5 parquet — building M5 bars from GOLD "
                 "ticks in %s ...", TICKS_DIR)
        df = load_or_build_bars("GOLD", "5min")
        if df is None:
            raise FileNotFoundError(
                f"No GOLD bars or ticks found.\n"
                f"  searched bar dirs : {PARQUET_DIR} , {TICKS_DIR}\n"
                f"  expected ticks at : {TICKS_DIR / 'HYDRA4_TICKS_GOLD.parquet'}\n"
                f"On Kaggle: confirm the tick dataset is ATTACHED to the "
                f"notebook (Add Input -> your dataset) — it must appear "
                f"under /kaggle/input/. Otherwise set M4GOLD_TICKS_DIR.")
        log.info("[datamodule] built %d M5 bars from ticks", len(df))
    df = df.rename(columns={"tick_volume": "volume"})
    if "volume" not in df.columns and "real_volume" in df.columns:
        df = df.rename(columns={"real_volume": "volume"})
    keep = ["time", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.dropna().sort_values("time").reset_index(drop=True)
    return df


def _resample(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Aggregate M5 bars into CLOCK-ALIGNED higher-timeframe bars.

    `rule` is a pandas offset alias ('15min', '1h'). Clock alignment —
    not positional 12-at-a-time grouping — so the bars line up exactly
    with MT5's native H1/M15 bars the EA reads via CopyRates. Positional
    grouping drifts out of phase across daily/weekend gaps. The bar's
    `time` is its period START (label='left'), matching MT5's convention.
    """
    g = m5.set_index("time")
    agg = g.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return agg.dropna(subset=["close"]).reset_index()


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
def _compute_channels(df: pd.DataFrame) -> np.ndarray:
    """Return float32[len(df), N_CHANNELS] of microstructure channels."""
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    eps = 1e-12

    ret = np.zeros_like(c)
    ret[1:] = np.log(np.clip(c[1:], eps, None) / np.clip(c[:-1], eps, None))
    hl_range = (h - l) / np.clip(c, eps, None)
    body = (c - o) / np.clip(c, eps, None)
    upper = (h - np.maximum(o, c)) / np.clip(c, eps, None)
    lower = (np.minimum(o, c) - l) / np.clip(c, eps, None)

    # Tick-rule signed volume: sign of the close-to-close move × volume,
    # then z-scored over a rolling window.
    sign = np.sign(ret)
    signed_vol_raw = sign * v
    sv = pd.Series(signed_vol_raw)
    sv_z = ((sv - sv.rolling(200, min_periods=20).mean())
            / sv.rolling(200, min_periods=20).std()).to_numpy()

    vol_ma = pd.Series(v).rolling(50, min_periods=10).mean().to_numpy()
    vol_ratio = np.where(vol_ma > eps, v / vol_ma, 1.0)

    # ATR(14) normalised by price.
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().to_numpy()
    atr_norm = np.where(c > eps, atr / c, 0.0)

    out = np.column_stack([ret, hl_range, body, upper, lower,
                           sv_z, vol_ratio, atr_norm]).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    assert out.shape[1] == N_CHANNELS, (out.shape, N_CHANNELS)
    return out


# ---------------------------------------------------------------------------
# Triple-barrier labelling
# ---------------------------------------------------------------------------
def _triple_barrier(m5: pd.DataFrame,
                    atr_m5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each M5 bar, simulate a long over LABEL_HORIZON_BARS:
      direction label — TP (+2 ATR) hit first -> long(2),
                        SL (-1 ATR) hit first -> short(0), neither -> flat(1).
      exec targets    — real per-bar SL/TP/timing supervision so the exec
                        head learns something meaningful instead of a proxy:
        y_exec[:,0] = max adverse excursion in ATR  (SL target)
        y_exec[:,1] = max favourable excursion in ATR (TP target)
        y_exec[:,2] = bars-to-resolution / horizon  (timing target, 0..1)
    """
    c = m5["close"].to_numpy(np.float64)
    h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64)
    n = len(c)
    y = np.full(n, 1, dtype=np.int64)                  # default flat
    y_exec = np.zeros((n, 3), dtype=np.float32)
    H = LABEL_HORIZON_BARS
    for i in range(n - H):
        a = atr_m5[i]
        if not np.isfinite(a) or a <= 0:
            continue
        tp = c[i] + LABEL_TB_TP_ATR * a
        sl = c[i] - LABEL_TB_SL_ATR * a
        win = slice(i + 1, i + 1 + H)
        hw, lw = h[win], l[win]
        hit_tp = np.argmax(hw >= tp) if (hw >= tp).any() else H + 1
        hit_sl = np.argmax(lw <= sl) if (lw <= sl).any() else H + 1
        if hit_tp < hit_sl:
            y[i] = 2
        elif hit_sl < hit_tp:
            y[i] = 0
        # exec supervision — excursions over the window, in ATR units
        mfe = float(np.clip((hw.max() - c[i]) / a, 0.0, 8.0))
        mae = float(np.clip((c[i] - lw.min()) / a, 0.0, 8.0))
        first_hit = min(hit_tp, hit_sl)
        res = (first_hit + 1) / H if first_hit <= H else 1.0
        y_exec[i, 0] = mae
        y_exec[i, 1] = mfe
        y_exec[i, 2] = float(np.clip(res, 0.0, 1.0))
    return y, y_exec


def _regime_label(m5: pd.DataFrame, atr_norm: np.ndarray) -> np.ndarray:
    """Coarse regime: trend-up / trend-down / range / high-vol."""
    c = m5["close"].to_numpy(np.float64)
    ma_fast = pd.Series(c).rolling(50, min_periods=10).mean().to_numpy()
    ma_slow = pd.Series(c).rolling(200, min_periods=20).mean().to_numpy()
    vol_hi = atr_norm > np.nanquantile(atr_norm[atr_norm > 0], 0.85)
    y = np.full(len(c), 2, dtype=np.int64)   # default range
    y[ma_fast > ma_slow] = 0
    y[ma_fast < ma_slow] = 1
    y[vol_hi] = 3
    return y


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def _window_for_tf(channels: np.ndarray, lookback: int,
                    anchor_idx: np.ndarray) -> np.ndarray:
    """
    For each anchor index a, take channels[a-lookback+1 : a+1].
    `channels` is already at the timeframe's resolution; `anchor_idx`
    are indices INTO that timeframe.
    """
    n_samp = len(anchor_idx)
    out = np.zeros((n_samp, lookback, channels.shape[1]), dtype=np.float32)
    for j, a in enumerate(anchor_idx):
        lo = a - lookback + 1
        if lo < 0:
            out[j, -(a + 1):] = channels[:a + 1]
        else:
            out[j] = channels[lo:a + 1]
    return out


def summary_features(X: dict) -> np.ndarray:
    """
    Compact per-(timeframe, channel) summary statistics for the tabular
    XGBoost baseline. Feeding a gradient-boosted model the full 2048-dim
    patch-flattened sequence is both pathologically slow and not how a GBM
    baseline is built — 5 stats per channel is the right tabular view.

    Returns float32[N, 3 timeframes * N_CHANNELS * 5].
    """
    feats = []
    for tf in TIMEFRAMES:
        a = X[tf]                                  # [N, L, C]
        feats.append(a.mean(axis=1))               # mean over the window
        feats.append(a.std(axis=1))                # volatility
        feats.append(a[:, -1, :])                  # latest bar
        feats.append(a.max(axis=1))                # window high
        feats.append(a[:, -1, :] - a[:, 0, :])     # window drift
    return np.concatenate(feats, axis=1).astype(np.float32)


def build_dataset(labelled: bool = True, max_bars: int | None = None) -> dict:
    """Build the full multi-timeframe AURUM dataset. See module docstring."""
    m5 = _load_m5_bars()
    if max_bars is not None and len(m5) > max_bars:
        m5 = m5.iloc[-max_bars:].reset_index(drop=True)

    # Per-timeframe bar frames + channels (clock-aligned HTF bars).
    bars = {}
    for tf in TIMEFRAMES:
        bars[tf] = m5.copy() if tf == "M5" else _resample(m5, _TF_RULE[tf])
    chans = {tf: _compute_channels(bars[tf]) for tf in TIMEFRAMES}

    # Per-channel normalisation stats (from M5 — the densest source).
    norm = {"mean": chans["M5"].mean(axis=0).tolist(),
            "std": (chans["M5"].std(axis=0) + 1e-8).tolist()}

    # Valid M5 anchors: enough history for the longest lookback AND
    # (if labelled) enough forward room for the triple barrier.
    warmup = max(L * _TF_M5_MULT[tf] for tf, L in TIMEFRAMES.items())
    n_m5 = len(m5)
    hi = n_m5 - (LABEL_HORIZON_BARS + 1 if labelled else 0)
    m5_anchors = np.arange(warmup, hi, dtype=np.int64)
    log.info("[datamodule] %d M5 anchors  (warmup=%d, labelled=%s)",
             len(m5_anchors), warmup, labelled)

    # Window each timeframe with CAUSAL anchors. For an M5 anchor, the
    # higher-tf window must end at the last FULLY-CLOSED HTF bar as of
    # that M5 bar's close — never the still-forming current HTF bar,
    # which aggregates M5 bars from the future and leaks the label.
    # tz-aware -> tz-naive datetime64 so numpy timedelta arithmetic works
    m5_time = m5["time"].dt.tz_localize(None).to_numpy()
    # an M5 bar opens at t and closes ~5 min later — that is decision time
    m5_decision = m5_time + np.timedelta64(5, "m")
    X = {}
    for tf, L in TIMEFRAMES.items():
        if tf == "M5":
            tf_anchors = m5_anchors            # the M5 bar itself is closed
        else:
            period = np.timedelta64(_TF_M5_MULT[tf] * 5, "m")
            htf_close = (bars[tf]["time"].dt.tz_localize(None).to_numpy()
                         + period)
            # last HTF bar whose close <= the M5 bar's decision time
            idx = np.searchsorted(htf_close, m5_decision, side="right") - 1
            tf_anchors = np.clip(idx[m5_anchors], 0, len(chans[tf]) - 1)
        X[tf] = _window_for_tf(chans[tf], L, tf_anchors)

    X_flat = np.concatenate(
        [X[tf].reshape(len(m5_anchors), -1) for tf in TIMEFRAMES], axis=1
    ).astype(np.float32)
    assert X_flat.shape[1] == FLAT_INPUT_DIM, (X_flat.shape, FLAT_INPUT_DIM)

    out = {"X": X, "X_flat": X_flat, "t_index": m5_anchors, "norm": norm}

    if labelled:
        prev_c = m5["close"].shift(1).fillna(m5["close"])
        tr = np.maximum(
            m5["high"] - m5["low"],
            np.maximum((m5["high"] - prev_c).abs(), (m5["low"] - prev_c).abs()))
        atr_m5 = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().to_numpy()
        atr_norm_m5 = np.where(m5["close"].to_numpy() > 0,
                               atr_m5 / m5["close"].to_numpy(), 0.0)
        y_dir_full, y_exec_full = _triple_barrier(m5, atr_m5)
        y_reg_full = _regime_label(m5, atr_norm_m5)
        c = m5["close"].to_numpy(np.float64)
        fwd = np.zeros(n_m5, dtype=np.float32)
        H = LABEL_HORIZON_BARS
        fwd[:n_m5 - H] = np.log(
            np.clip(c[H:], 1e-12, None) / np.clip(c[:n_m5 - H], 1e-12, None))
        out["y_dir"] = y_dir_full[m5_anchors]
        out["y_ret"] = fwd[m5_anchors]
        out["y_regime"] = y_reg_full[m5_anchors]
        out["y_exec"] = y_exec_full[m5_anchors]
        log.info("[datamodule] labels  short=%d flat=%d long=%d",
                 int((out["y_dir"] == 0).sum()),
                 int((out["y_dir"] == 1).sum()),
                 int((out["y_dir"] == 2).sum()))
    return out
