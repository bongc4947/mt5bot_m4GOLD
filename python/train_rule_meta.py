"""
train_rule_meta.py — meta-labelling on TOP of the proven rule.

Hypothesis: the z-score fade rule from BACKTEST_RESULTS.md has PF 1.7-4.2
on 12 symbols. An ML classifier trained ONLY on the rule's fired
candidates may be able to predict which candidates will hit TP vs SL,
filtering out the worst ~20-40% and lifting overall PF.

Unlike direct directional ML (which we've now proven doesn't extract
edge above retail spread), meta-labelling solves a much easier problem:
binary classification on ~1500-3000 candidate events per symbol per year,
where the rule has already done the hard "should we even consider this
bar" filtering.

If excess_PF (rule+meta vs rule alone) > +0.30 on at least one symbol,
this is the path. The meta ONNX gets loaded by the Rule EA's optional
InpUseMetaFilter; the EA skips any rule-fired signal where meta_prob
< confidence threshold.

Usage:
    python python/train_rule_meta.py UK_100
    python python/train_rule_meta.py BTCUSD ETHUSD SILVER --epochs 200
    python python/train_rule_meta.py --all   # all LIVE-READY symbols on disk
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)

# LIVE-READY symbols from BACKTEST_RESULTS.md (PF 1.71-4.22).
# mk4.8: NAS100 removed — broker doesn't quote it on the active terminal,
# so no tick parquet is extracted. The list dropped from 12 to 11 as a
# result; data on disk is the source of truth.
LIVE_READY_SYMBOLS = [
    "UK_100", "US_500", "SILVER", "BTCUSD", "PLATINUM",
    "ETHUSD", "LTCUSD", "NATURAL_GAS", "BRENT_OIL", "CrudeOIL", "COPPER",
]


# ---------------------------------------------------------------------------
# Rule signal generation (matches MeanReversionRule.mqh exactly)
# ---------------------------------------------------------------------------

def _z_score(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling z-score of log-returns over `window` bars. Backward-causal."""
    log_ret = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=0.0)
    s = pd.Series(log_ret)
    mean = s.rolling(window, min_periods=2).mean().to_numpy()
    std  = s.rolling(window, min_periods=2).std().to_numpy()
    std  = np.where(std > 1e-12, std, 1.0)
    z = (log_ret - mean) / std
    return np.nan_to_num(z, nan=0.0)


def _vol_regime_ratio(close: np.ndarray, short: int = 20, long: int = 200) -> np.ndarray:
    """std(last short) / std(last long) — high = active regime."""
    log_ret = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=0.0)
    s = pd.Series(log_ret)
    short_std = s.rolling(short, min_periods=2).std().to_numpy()
    long_std  = s.rolling(long,  min_periods=2).std().to_numpy()
    long_std  = np.where(long_std > 1e-12, long_std, 1.0)
    ratio = short_std / long_std
    return np.nan_to_num(ratio, nan=1.0)


def _atr(high, low, close, period=14):
    """Backward-causal ATR(period)."""
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close),
    ])
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def _pip_size(symbol: str) -> float:
    """
    Price units per "pip" for spread / cost display. Round-number conventions
    match what brokers / MT5 typically call a pip:
      - 4-digit FX majors:      0.0001
      - JPY pairs:              0.01
      - Metals (Gold/Silver/etc): 0.01
      - Major indices:          1.0   (1 point = 1 pip in practice)
      - Crude / Brent / NatGas: 0.01
      - COPPER:                 0.001
      - Crypto CFDs:            1.0
    """
    s = symbol.upper()
    if "JPY" in s:                                              return 0.01
    if s in ("GOLD", "XAUUSD", "SILVER", "XAGUSD",
              "PLATINUM", "XPTUSD"):                            return 0.01
    if s in ("UK_100", "US_500", "JP_225", "DE_30"):            return 1.0
    if s in ("CRUDEOIL", "BRENT_OIL", "NATURAL_GAS"):           return 0.01
    if s == "COPPER":                                           return 0.001
    if s in ("BTCUSD", "ETHUSD", "LTCUSD"):                     return 1.0
    return 0.0001   # default — 4-digit FX major


def _ticks_to_m5(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Build honest M5 OHLCV bars from a raw tick parquet (one row per tick).
    Output schema matches HYDRA4_FEAT_*.parquet so the rest of the pipeline
    consumes it unchanged.

    HIGH/LOW capture *every* tick mid-price in the 5-min window (not the
    pre-aggregated tick-bar OHLCs that lose intra-tick-bar extremes).
    This is the closest possible approximation to MT5's TIMEFRAME_M5
    bars when you don't have them directly.
    """
    df = ticks.copy()
    if "time_msc" in df.columns:
        df["time"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    else:
        raise ValueError("ticks must have 'time_msc' or 'time' column")
    if "bid" not in df.columns or "ask" not in df.columns:
        raise ValueError("ticks must have 'bid' and 'ask' columns")

    df["mid"]    = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df = df[["time", "mid", "spread"] + (["volume"] if "volume" in df.columns else [])]
    df = df.set_index("time").sort_index()

    g = df["mid"].resample("5min")
    bars = pd.DataFrame({
        "open":  g.first(),
        "high":  g.max(),
        "low":   g.min(),
        "close": g.last(),
    })
    if "volume" in df.columns:
        bars["tick_volume"] = df["volume"].resample("5min").sum()
    else:
        bars["tick_volume"] = df["mid"].resample("5min").count()
    bars["spread"] = df["spread"].resample("5min").mean()
    return bars.dropna(subset=["close"]).reset_index()


def _load_bars_for_symbol(symbol: str):
    """
    Resolve the best bar source available for `symbol`, in priority order:

      1. HYDRA4_FEAT_<SYM>_*.parquet     — direct MT5 M5 bars (best)
      2. HYDRA4_M5FROMTICKS_<SYM>.parquet — pre-built from raw ticks (cached)
      3. HYDRA4_TICKS_<SYM>.parquet       — raw ticks; build M5 + cache + use
      4. HYDRA4_TBARS_<SYM>_*.parquet     — 100-tick bars; resample to M5
                                            (last resort — rule PF may not reproduce)

    Returns (bars_df, source_label) or (None, "none").
    """
    from config import PARQUET_DIR, TICKS_DIR

    direct = sorted(PARQUET_DIR.glob(f"HYDRA4_FEAT_{symbol}_*.parquet"),
                     key=lambda p: p.stat().st_size, reverse=True)
    if direct:
        log.info("[%s] loading DIRECT M5 cache: %s", symbol, direct[0].name)
        return pd.read_parquet(direct[0]), "direct_m5"

    cached_m5 = PARQUET_DIR / f"HYDRA4_M5FROMTICKS_{symbol}.parquet"
    if cached_m5.exists():
        log.info("[%s] loading cached M5-from-ticks: %s", symbol, cached_m5.name)
        return pd.read_parquet(cached_m5), "m5_from_ticks_cached"

    raw_ticks = TICKS_DIR / f"HYDRA4_TICKS_{symbol}.parquet"
    if raw_ticks.exists():
        sz_mb = raw_ticks.stat().st_size / 1e6
        log.info("[%s] building M5 from raw ticks: %s (%.0f MB)",
                 symbol, raw_ticks.name, sz_mb)
        ticks = pd.read_parquet(raw_ticks)
        # Raw ticks are stored newest-first by fetch_ticks_capped(), so
        # iloc[0] is the most recent tick. Use min/max to display the
        # actual span without ordering ambiguity.
        if "time_msc" in ticks.columns:
            t_min = pd.to_datetime(ticks["time_msc"].min(), unit="ms", utc=True)
            t_max = pd.to_datetime(ticks["time_msc"].max(), unit="ms", utc=True)
            log.info("[%s] read %d ticks  span=%s -> %s",
                     symbol, len(ticks), t_min, t_max)
        else:
            log.info("[%s] read %d ticks", symbol, len(ticks))
        bars = _ticks_to_m5(ticks)
        cached_m5.parent.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(cached_m5, compression="zstd", compression_level=9)
        log.info("[%s] cached %d M5 bars -> %s", symbol, len(bars), cached_m5.name)
        return bars, "m5_from_raw_ticks"

    tbar = sorted(PARQUET_DIR.glob(f"HYDRA4_TBARS_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_size, reverse=True)
    if tbar:
        log.warning("[%s] FALLBACK: only tick-bars available (no direct M5 "
                     "or raw ticks). Resampling to M5 — rule PF may not "
                     "reproduce BACKTEST_RESULTS.md. Re-extract with "
                     "`extract_data.py %s --source ticks --save-raw-ticks` "
                     "or `--source bars` for proper M5.",
                     symbol, symbol)
        return pd.read_parquet(tbar[0]), "tickbar_resampled"

    return None, "none"


def resample_to_m5(bars: pd.DataFrame) -> pd.DataFrame:
    """
    If `bars` is tick-bars (information-uniform, variable wall-clock duration),
    resample to time-uniform M5 bars matching the schema BACKTEST_RESULTS.md
    used. If it's already M5, return unchanged.

    Detection heuristic: median wall-clock gap between consecutive bars.
    M5 bars sit at ~300 s. Tick-bars are much shorter on average (seconds
    or sub-second on liquid symbols, minutes on quiet ones).
    """
    if "time" not in bars.columns:
        log.warning("no 'time' column — cannot resample to M5, using bars as-is")
        return bars
    df = bars.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").set_index("time")

    # Median wall-clock interval between bars.
    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    if len(diffs) == 0:
        return df.reset_index()
    median_dur = float(diffs.median())

    # Already M5? (250-350 s window). Return unchanged.
    if 250 <= median_dur <= 350:
        log.info("  bars already at M5 (median gap %.0fs) — no resample needed",
                 median_dur)
        return df.reset_index()

    log.info("  resampling tick-bars to M5: median input gap=%.1fs  in_rows=%d",
             median_dur, len(df))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "tick_volume" in df.columns: agg["tick_volume"] = "sum"
    if "real_volume" in df.columns: agg["real_volume"] = "sum"
    if "spread"      in df.columns: agg["spread"]      = "mean"
    out = df.resample("5min").agg(agg).dropna(subset=["close"])
    log.info("  resampled to %d M5 bars  span=%s -> %s",
             len(out), out.index.min(), out.index.max())
    return out.reset_index()


def generate_rule_candidates(bars: pd.DataFrame, *,
                              z_thresh: float = 2.5,
                              vol_threshold: float = 0.70,
                              sl_atr_mult: float = 1.0,
                              tp_atr_mult: float = 2.0,
                              timeout_bars: int = 20,
                              atr_period: int = 14) -> pd.DataFrame:
    """
    Return one row per bar where the rule fires. Each row carries:
      bar_idx, direction (+1/-1), entry, sl, tp, atr, z, vol_ratio,
      tp_hit (1 if TP first), sl_hit (1 if SL first), timeout (1 if neither)

    The outcome is filled by simulating the trade forward `timeout_bars`.
    """
    close = bars["close"].to_numpy(dtype=np.float64)
    high  = bars["high"].to_numpy(dtype=np.float64)
    low   = bars["low"].to_numpy(dtype=np.float64)
    n = len(close)

    z = _z_score(close, 20)
    vol_ratio = _vol_regime_ratio(close, 20, 200)
    atr = _atr(high, low, close, atr_period)

    candidates = []
    for i in range(200, n - timeout_bars):
        if atr[i] <= 0:
            continue
        if vol_ratio[i] < vol_threshold:
            continue
        if z[i] >= z_thresh:
            direction = -1   # SHORT (fade up-spike)
        elif z[i] <= -z_thresh:
            direction = +1   # LONG  (fade down-spike)
        else:
            continue

        entry = close[i]
        if direction > 0:
            sl_p = entry - sl_atr_mult * atr[i]
            tp_p = entry + tp_atr_mult * atr[i]
        else:
            sl_p = entry + sl_atr_mult * atr[i]
            tp_p = entry - tp_atr_mult * atr[i]

        tp_hit = sl_hit = 0
        for k in range(1, timeout_bars + 1):
            j = i + k
            hi = high[j]; lo = low[j]
            if direction > 0:
                if hi >= tp_p:    tp_hit = 1; break
                if lo <= sl_p:    sl_hit = 1; break
            else:
                if lo <= tp_p:    tp_hit = 1; break
                if hi >= sl_p:    sl_hit = 1; break
        timeout = 1 if (tp_hit == 0 and sl_hit == 0) else 0

        candidates.append({
            "bar_idx":   i,
            "direction": direction,
            "entry":     entry,
            "sl":        sl_p,
            "tp":        tp_p,
            "atr":       atr[i],
            "z":         z[i],
            "vol_ratio": vol_ratio[i],
            "tp_hit":    tp_hit,
            "sl_hit":    sl_hit,
            "timeout":   timeout,
        })

    return pd.DataFrame(candidates)


# ---------------------------------------------------------------------------
# Feature engineering at the candidate bar
# ---------------------------------------------------------------------------

def build_meta_features(bars: pd.DataFrame, cand: pd.DataFrame) -> np.ndarray:
    """
    For each candidate bar, build a fixed-dim feature vector. Uses the
    rule's own diagnostic numbers + simple microstructure context.

      0-3   :  z, |z|, vol_ratio, atr_normalized
      4-7   :  log returns over last 5, 10, 20, 50 bars
      8-11  :  std of log returns over last 5, 10, 20, 50 bars
      12-15 :  hi-lo range over last 5, 10, 20, 50 bars (normalized)
      16-19 :  count of large moves (|z|>1.5) in last 50, 100, 150, 200 bars
      20-23 :  rolling RSI(14), RSI position, MACD-like signal, volume_z
      24-27 :  sin/cos of hour-of-day + day-of-week (cyclical time)
      28    :  spread / atr ratio if spread available

    Returns (N, ~29) float32 array aligned to cand rows.
    """
    close = bars["close"].to_numpy(dtype=np.float64)
    high  = bars["high"].to_numpy(dtype=np.float64)
    low   = bars["low"].to_numpy(dtype=np.float64)
    spread = bars.get("spread", pd.Series(np.zeros(len(close)))).to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=0.0)

    # Pre-compute rolling stats
    s = pd.Series(log_ret)
    roll_ret = {h: s.rolling(h, min_periods=2).sum().to_numpy() for h in (5, 10, 20, 50)}
    roll_std = {h: s.rolling(h, min_periods=2).std().to_numpy() for h in (5, 10, 20, 50)}
    roll_rng = {}
    for h in (5, 10, 20, 50):
        hh = pd.Series(high).rolling(h, min_periods=2).max().to_numpy()
        ll = pd.Series(low ).rolling(h, min_periods=2).min().to_numpy()
        rng = (hh - ll) / np.where(close > 0, close, 1.0)
        roll_rng[h] = np.nan_to_num(rng, nan=0.0)
    z_abs = np.abs(_z_score(close, 20))
    extremes_lookback = {}
    extreme_flag = (z_abs > 1.5).astype(np.float32)
    for h in (50, 100, 150, 200):
        extremes_lookback[h] = pd.Series(extreme_flag).rolling(h, min_periods=2).sum().to_numpy()

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    up = pd.Series(np.where(delta > 0, delta, 0.0)).rolling(14, min_periods=1).mean().to_numpy()
    dn = pd.Series(np.where(delta < 0, -delta, 0.0)).rolling(14, min_periods=1).mean().to_numpy()
    rs = up / (dn + 1e-12)
    rsi = 100 - (100 / (1 + rs))

    # MACD-like
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().to_numpy()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().to_numpy()
    macd  = (ema12 - ema26) / np.where(close > 0, close, 1.0)

    # Volume z-score (using tick_volume if present)
    vol = bars.get("tick_volume", pd.Series(np.ones(len(close)))).to_numpy(dtype=np.float64)
    vol_mean = pd.Series(vol).rolling(50, min_periods=2).mean().to_numpy()
    vol_std  = pd.Series(vol).rolling(50, min_periods=2).std().to_numpy()
    vol_z    = (vol - vol_mean) / np.where(vol_std > 1e-12, vol_std, 1.0)
    vol_z    = np.clip(np.nan_to_num(vol_z, nan=0.0), -5, 5) / 5

    # Time cyclicals
    if "time" in bars.columns:
        t = pd.to_datetime(bars["time"], utc=True)
        hour = t.dt.hour.to_numpy()
        dow = t.dt.dayofweek.to_numpy()
        sin_h = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        cos_h = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        sin_d = np.sin(2 * np.pi * dow / 7).astype(np.float32)
        cos_d = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    else:
        sin_h = cos_h = sin_d = cos_d = np.zeros(len(close), dtype=np.float32)

    # Pull at candidate indices
    idx = cand["bar_idx"].to_numpy()
    atr = _atr(high, low, close, 14)

    feats = np.column_stack([
        cand["z"].to_numpy(),
        np.abs(cand["z"].to_numpy()),
        cand["vol_ratio"].to_numpy(),
        atr[idx] / np.where(close[idx] > 0, close[idx], 1.0) * 100,
        roll_ret[5][idx]   * 100, roll_ret[10][idx]  * 100,
        roll_ret[20][idx]  * 100, roll_ret[50][idx]  * 100,
        roll_std[5][idx]   * 100, roll_std[10][idx]  * 100,
        roll_std[20][idx]  * 100, roll_std[50][idx]  * 100,
        roll_rng[5][idx]   * 100, roll_rng[10][idx]  * 100,
        roll_rng[20][idx]  * 100, roll_rng[50][idx]  * 100,
        extremes_lookback[50][idx],  extremes_lookback[100][idx],
        extremes_lookback[150][idx], extremes_lookback[200][idx],
        (rsi[idx] - 50) / 50,
        np.where(rsi[idx] > 70, 1.0, np.where(rsi[idx] < 30, -1.0, 0.0)),
        np.clip(macd[idx] * 100, -1, 1),
        vol_z[idx],
        sin_h[idx], cos_h[idx], sin_d[idx], cos_d[idx],
        np.clip(spread[idx] / np.where(atr[idx] > 0, atr[idx], 1.0), 0, 1),
    ]).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


# ---------------------------------------------------------------------------
# Train meta classifier (XGBoost — calibrated probs, fast, ONNX-exportable)
# ---------------------------------------------------------------------------

def train_meta_classifier(X_tr, y_tr, X_va, y_va, *, n_estimators=300, max_depth=4):
    """
    Train a binary classifier predicting P(TP hit | rule fired).
    y in {0, 1}: 1 if the candidate hit TP, 0 if SL or timeout.

    Returns (model, val_probs, T_calibration).
    """
    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost not installed — pip install xgboost")
        raise

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_probs = model.predict_proba(X_va)[:, 1]

    # Temperature calibration on val
    from validation import fit_temperature
    val_logits = np.log(np.clip(val_probs, 1e-6, 1 - 1e-6) / np.clip(1 - val_probs, 1e-6, 1))
    T = fit_temperature(val_logits, y_va.astype(np.float32))
    cal_probs = 1.0 / (1.0 + np.exp(-val_logits / T))

    return model, cal_probs, float(T)


# ---------------------------------------------------------------------------
# Per-symbol training + evaluation
# ---------------------------------------------------------------------------

def train_one_symbol(symbol: str, *,
                      n_estimators: int = 300,
                      max_depth: int = 4,
                      meta_threshold: float = 0.55,
                      cost_pip: float = 2.0,
                      tp_atr_mult: float = 2.0,
                      sl_atr_mult: float = 1.0,
                      ) -> dict:
    from config import ONNX_OUTPUT_DIR
    bars, source_kind = _load_bars_for_symbol(symbol)
    if bars is None:
        return {"symbol": symbol, "ok": False, "reason": "no parquet"}
    log.info("[%s] %d input bars  source=%s", symbol, len(bars), source_kind)

    # Only do the lossy tick-bar -> M5 resample if we genuinely fell back
    # to the HYDRA4_TBARS_* path. Direct M5 (or M5 built from raw ticks)
    # is already at the right timescale.
    if source_kind == "tickbar_resampled":
        bars = resample_to_m5(bars)

    if len(bars) < 5000:
        return {"symbol": symbol, "ok": False,
                "reason": f"only {len(bars)} M5 bars (source={source_kind})"}

    # 1. Generate rule candidates
    cand = generate_rule_candidates(bars,
                                      z_thresh=2.5, vol_threshold=0.70,
                                      sl_atr_mult=sl_atr_mult,
                                      tp_atr_mult=tp_atr_mult,
                                      timeout_bars=20, atr_period=14)
    if len(cand) < 100:
        return {"symbol": symbol, "ok": False,
                "reason": f"only {len(cand)} rule candidates"}

    # Log rule fire-rate so we can compare to BACKTEST_RESULTS.md
    # (~1000-1200 trades/year was typical there). If our rate is <100/year
    # the M5 resample is over-smoothing or the broker's price action is
    # genuinely different.
    if "time" in bars.columns:
        span_yrs = (pd.to_datetime(bars["time"].iloc[-1]) -
                    pd.to_datetime(bars["time"].iloc[0])).total_seconds() / (365.25 * 86400)
        if span_yrs > 0:
            log.info("[%s] rule fire-rate: %.1f candidates/year  (span=%.2f yr)",
                     symbol, len(cand) / span_yrs, span_yrs)

    rule_wr_no_cost = float((cand["tp_hit"] == 1).mean())
    n_total = len(cand)
    n_tp = int((cand["tp_hit"] == 1).sum())
    n_sl = int((cand["sl_hit"] == 1).sum())
    n_to = int(cand["timeout"].sum())
    log.info("[%s] rule candidates: %d total  TP=%d (%.1f%%)  SL=%d (%.1f%%)  TIMEOUT=%d (%.1f%%)",
             symbol, n_total,
             n_tp, 100 * n_tp / n_total,
             n_sl, 100 * n_sl / n_total,
             n_to, 100 * n_to / n_total)

    # 2. Compute per-candidate PnL in price units, then subtract realistic cost.
    pnl_price = np.where(
        cand["tp_hit"] == 1,
        cand["tp"] - cand["entry"],
        np.where(cand["sl_hit"] == 1, cand["sl"] - cand["entry"], 0.0)
    )
    pnl_price = pnl_price * cand["direction"].to_numpy()

    # COST: prefer the actual broker spread at each candidate bar if the
    # `spread` column survived. Round-trip ≈ 2× spread. Pip-size is per
    # symbol (EURUSD = 1e-4, USDJPY = 1e-2, GOLD = 0.01, indices = 1.0).
    pip = _pip_size(symbol)
    if "spread" in bars.columns:
        cand_idx_arr = cand["bar_idx"].to_numpy()
        bar_spreads = bars["spread"].iloc[cand_idx_arr].to_numpy(dtype=np.float64)
        # Fall back to (cost_pip × pip / 2) per-side if a bar's spread is
        # missing / non-positive (rare; broker quote glitches).
        fallback_per_side = (cost_pip * pip) / 2.0
        bar_spreads = np.where(np.isfinite(bar_spreads) & (bar_spreads > 0),
                                bar_spreads, fallback_per_side)
        cost_price = 2.0 * bar_spreads
        # Convert price-unit spread to *pips* by dividing by the per-symbol
        # pip-size (NOT a fixed 1e5 multiplier — that one was wrong for
        # 4-digit FX and JPY pairs alike).
        median_sp_pip = float(np.median(bar_spreads) / pip)
        p95_sp_pip    = float(np.percentile(bar_spreads, 95) / pip)
        log.info("[%s] using actual broker spreads: median=%.2f pips  "
                 "P95=%.2f pips  (round-trip cost = 2×spread, "
                 "pip_size=%.0e)",
                 symbol, median_sp_pip, p95_sp_pip, pip)
    else:
        cost_price = np.full(len(cand), cost_pip * pip)
        log.info("[%s] no spread column — flat cost=%.1f pips/trade "
                 "(pip_size=%.0e -> %.0e per round trip)",
                 symbol, cost_pip, pip, cost_pip * pip)
    pnl_after_cost = pnl_price - cost_price

    rule_alone_pf = float(pnl_after_cost[pnl_after_cost > 0].sum() /
                          max(1e-12, -pnl_after_cost[pnl_after_cost < 0].sum()))
    n_winners_rule = int((pnl_after_cost > 0).sum())
    n_losers_rule  = int((pnl_after_cost < 0).sum())
    log.info("[%s] rule-alone PF: %.3f  (%d winners / %d losers / %d zero)",
             symbol, rule_alone_pf, n_winners_rule, n_losers_rule,
             len(pnl_after_cost) - n_winners_rule - n_losers_rule)

    # 3. Train meta classifier
    feats = build_meta_features(bars, cand)
    y = cand["tp_hit"].to_numpy().astype(np.int8)

    # Chronological split with horizon gap
    n_val = max(50, int(len(cand) * 0.20))
    n_tr = len(cand) - n_val
    X_tr, y_tr = feats[:n_tr], y[:n_tr]
    X_va, y_va = feats[n_tr:], y[n_tr:]
    log.info("[%s] meta train: %d (TP frac=%.3f)  val: %d (TP frac=%.3f)",
             symbol, len(X_tr), float(y_tr.mean()),
             len(X_va), float(y_va.mean()))

    if abs(float(y_va.mean()) - 0.5) > 0.35:
        return {"symbol": symbol, "ok": False,
                "reason": f"degenerate val labels (TP frac={float(y_va.mean()):.3f})"}

    model, cal_probs, T = train_meta_classifier(X_tr, y_tr, X_va, y_va,
                                                  n_estimators=n_estimators,
                                                  max_depth=max_depth)
    # Diagnostic: what does the meta classifier actually output on val?
    # If cal_probs all sit below 0.50, every threshold ≥ 0.50 will mask 0
    # candidates and excess=-1.0 will be reported. Surface that so the
    # user sees the prediction distribution.
    log.info("[%s] meta cal_probs on val:  min=%.3f  med=%.3f  max=%.3f  "
             "frac>=0.50=%.2f  frac>=0.55=%.2f  frac>=0.60=%.2f",
             symbol,
             float(cal_probs.min()), float(np.median(cal_probs)), float(cal_probs.max()),
             float((cal_probs >= 0.50).mean()),
             float((cal_probs >= 0.55).mean()),
             float((cal_probs >= 0.60).mean()))

    # 4. Evaluate rule + meta-filter at multiple thresholds
    pnl_va = pnl_after_cost[n_tr:]
    # Rule-alone PF on the val slice (constant across thresholds — once,
    # used in every result row so all thresholds compare against the same
    # baseline).
    rule_va_pf = float(pnl_va[pnl_va > 0].sum() /
                        max(1e-12, -pnl_va[pnl_va < 0].sum())) if len(pnl_va) else 0.0

    def _empty_result(thr_, mask_):
        # Every key the deploy gate / summary loop / meta JSON references.
        # Keep this synchronised with the populated branch below.
        return {
            "n_trades":       int(mask_.sum()),
            "wr":             0.0,
            "pf":             0.0,
            "rule_va_pf":     rule_va_pf,
            "excess_vs_rule": -1.0,
            "frac_kept":      float(mask_.mean()) if len(mask_) else 0.0,
        }

    results = {}
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
        mask = cal_probs >= thr
        if mask.sum() < 10:
            results[thr] = _empty_result(thr, mask)
            continue
        kept = pnl_va[mask]
        wr = float((kept > 0).mean())
        wins = kept[kept > 0].sum()
        losses = -kept[kept < 0].sum()
        pf = float(wins / max(1e-12, losses))
        results[thr] = {
            "n_trades":       int(mask.sum()),
            "wr":             wr,
            "pf":             pf,
            "rule_va_pf":     rule_va_pf,
            "excess_vs_rule": pf - rule_va_pf,
            "frac_kept":      float(mask.mean()),
        }
        log.info("[%s] @threshold=%.2f  kept=%d/%d (%.1f%%)  WR=%.3f  meta_PF=%.3f  "
                 "rule_PF=%.3f  excess=%+.3f",
                 symbol, thr, mask.sum(), len(mask), 100 * mask.mean(),
                 wr, pf, rule_va_pf, pf - rule_va_pf)

    # 5. Pick best threshold by excess PF (must beat rule alone)
    best_thr = max(results.keys(),
                    key=lambda t: results[t]["excess_vs_rule"]
                    if results[t]["n_trades"] >= 30 else -99)
    best = results[best_thr]
    deploy_pass = bool(best.get("excess_vs_rule", -1) > 0.10 and
                       best.get("pf", 0) > 1.20 and
                       best.get("n_trades", 0) >= 30)

    # 6. Export ONNX. Use onnxmltools' own FloatTensorType (not skl2onnx's).
    # The two classes share a name but are different objects, and
    # onnxmltools.convert_xgboost rejects skl2onnx's class with an opaque
    # message. The fix is just to import from the right module.
    ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_OUTPUT_DIR / f"HYDRA4_RULEMETA_{symbol}.onnx"
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        initial_type = [("features", FloatTensorType([None, feats.shape[1]]))]
        onx = onnxmltools.convert_xgboost(model, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        log.info("[%s] ONNX written: %s", symbol, onnx_path.name)
        onnx_ok = True
    except Exception as e:
        log.warning("[%s] ONNX export failed: %s", symbol, e)
        onnx_ok = False

    meta = {
        "symbol": symbol,
        "feature_dim": int(feats.shape[1]),
        "n_candidates_total": int(n_total),
        "rule_tp_rate_overall": float(rule_wr_no_cost),
        "rule_alone_pf_with_cost": float(rule_alone_pf),
        "cost_pip_assumed": float(cost_pip),
        "best_threshold":       float(best_thr),
        "best_pf":              float(best.get("pf",            0.0)),
        "best_wr":              float(best.get("wr",            0.0)),
        "best_excess_vs_rule":  float(best.get("excess_vs_rule", 0.0)),
        "best_n_trades":          int(best.get("n_trades",      0)),
        "best_frac_kept":       float(best.get("frac_kept",     0.0)),
        "results_by_threshold": {str(k): v for k, v in results.items()},
        "temperature": float(T),
        "onnx_ok": bool(onnx_ok),
        "deploy": bool(deploy_pass),
    }
    meta_path = ONNX_OUTPUT_DIR / f"HYDRA4_RULEMETA_{symbol}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("[%s] %s — best_threshold=%.2f  meta_PF=%.3f  excess_vs_rule=%+.3f",
             symbol,
             "DEPLOY" if deploy_pass else "BLOCKED",
             best_thr, best["pf"], best["excess_vs_rule"])
    return meta


def _preflight():
    """Verify required deps at startup so failures get a clean message."""
    missing = []
    try:
        import xgboost  # noqa: F401
    except ImportError:
        missing.append("xgboost (pip install xgboost)")
    try:
        import onnxmltools  # noqa: F401
    except ImportError:
        missing.append("onnxmltools (pip install onnxmltools)  # for ONNX export only")
    try:
        from skl2onnx.common.data_types import FloatTensorType  # noqa: F401
    except ImportError:
        missing.append("skl2onnx (pip install skl2onnx)  # for ONNX export only")
    if missing:
        print("\n[train_rule_meta] missing dependencies — install them and retry:",
              file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        if any("xgboost" in m for m in missing):
            # xgboost is required; the others are only for ONNX export.
            print("[train_rule_meta] xgboost is required for training — aborting.",
                  file=sys.stderr)
            sys.exit(3)
        else:
            print("[train_rule_meta] continuing; meta JSON will be written even "
                  "if ONNX export fails.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s",
                         stream=sys.stdout, force=True)
    _preflight()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbols", nargs="*",
                   help="symbols to train (default: 12 LIVE-READY)")
    p.add_argument("--all", action="store_true",
                   help="train on all 11 LIVE-READY symbols")
    p.add_argument("--estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--cost-pip", type=float, default=2.0,
                   help="round-trip cost in pip-equivalents. Default 2.0 (retail). "
                        "Use 0.3 for ECN broker.")
    p.add_argument("--meta-threshold", type=float, default=0.55,
                   help="meta-prob threshold for live trading (informational only; "
                        "training scans 0.50-0.70 and picks the best)")
    args = p.parse_args(argv)

    symbols = LIVE_READY_SYMBOLS if (args.all or not args.symbols) else args.symbols

    t0 = time.time()
    results = []
    for sym in symbols:
        try:
            r = train_one_symbol(sym,
                                   n_estimators=args.estimators,
                                   max_depth=args.max_depth,
                                   cost_pip=args.cost_pip)
        except Exception as e:
            log.exception("[%s] failed", sym)
            r = {"symbol": sym, "ok": False, "reason": str(e)}
        results.append(r)

    # Summary
    print(f"\n{'='*78}")
    print(f"  RULE+META training summary  ({time.time() - t0:.0f}s)")
    print(f"  cost_pip assumed: {args.cost_pip}")
    print(f"{'='*78}")
    print(f"  {'Symbol':<12}  {'rule_PF':>8}  {'meta_PF':>8}  {'excess':>8}  "
          f"{'thr':>5}  {'kept':>6}  {'N':>5}  Deploy")
    print(f"  {'-'*78}")
    for r in results:
        sym = r.get("symbol", "?")
        # FAIL row only when we have neither numerical results nor an ONNX —
        # i.e. the run truly couldn't measure anything for this symbol.
        if "best_pf" not in r and not r.get("onnx_ok"):
            print(f"  {sym:<12}  FAIL ({r.get('reason', 'unknown')})")
            continue
        print(f"  {sym:<12}  "
              f"{r.get('rule_alone_pf_with_cost', 0.0):>8.3f}  "
              f"{r.get('best_pf', 0.0):>8.3f}  "
              f"{r.get('best_excess_vs_rule', 0.0):>+8.3f}  "
              f"{r.get('best_threshold', 0.0):>5.2f}  "
              f"{100 * r.get('best_frac_kept', 0.0):>5.0f}%  "
              f"{r.get('best_n_trades', 0):>5d}  "
              f"{'yes' if r.get('deploy') else 'no'}")
    print(f"{'='*78}\n")
    deployed = sum(1 for r in results if r.get("deploy"))
    print(f"  {deployed} / {len(results)} symbols pass the meta gate "
          f"(meta_PF > rule_PF + 0.10 AND meta_PF > 1.20 AND N >= 30)")
    print(f"  Best symbols by excess_vs_rule:")
    sorted_r = sorted([r for r in results if r.get("onnx_ok")],
                       key=lambda x: x.get("best_excess_vs_rule", -99),
                       reverse=True)
    for r in sorted_r[:5]:
        print(f"    {r['symbol']:<12}  excess=+{r['best_excess_vs_rule']:.3f}  "
              f"meta_PF={r['best_pf']:.3f}  vs rule_PF={r['rule_alone_pf_with_cost']:.3f}")
    print()
    # Return 0 regardless of deploy count — "zero symbols pass" is a valid
    # answer, not a script failure. The caller decides what to do with the
    # results (see ONNX outputs + per-symbol meta JSON in onnx_out/).
    # Non-zero only if an *exception* propagated out of every train call.
    n_completed = sum(1 for r in results if "best_pf" in r or r.get("ok") is False)
    return 0 if n_completed > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
