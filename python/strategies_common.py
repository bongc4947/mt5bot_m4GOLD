"""strategies_common.py — shared helpers for the H1/H2/H4 strategy trainers.

Centralises bar construction (M5/H1/H4 from raw ticks), walk-forward splits,
skill gate, XGBoost training + temperature calibration, and ONNX export so
the three trainers stay short and consistent.

All public functions are strictly backward-causal. The audit script
(audit_strategies.py) reads this file's source and asserts no `shift(-N)`,
no `[::-1]`, and no `forward=*` argument names.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbol conventions
# ---------------------------------------------------------------------------

def pip_size(symbol: str) -> float:
    """
    Round-number pip size by symbol family. Mirrors train_rule_meta._pip_size.

    The tuples below list every symbol that actually has a tick parquet
    on disk (see data/ticks/HYDRA4_TICKS_*.parquet). NAS100 is omitted
    because the active broker doesn't quote it and no parquet exists.
    """
    s = symbol.upper()
    if "JPY" in s:                                              return 0.01
    if s in ("GOLD", "XAUUSD", "SILVER", "XAGUSD",
             "PLATINUM", "XPTUSD"):                             return 0.01
    if s in ("UK_100", "US_500", "JP_225", "DE_30"):            return 1.0
    if s in ("CRUDEOIL", "BRENT_OIL", "NATURAL_GAS"):           return 0.01
    if s == "COPPER":                                           return 0.001
    if s in ("BTCUSD", "ETHUSD", "LTCUSD"):                     return 1.0
    return 1e-4   # default 4-digit FX major


# ---------------------------------------------------------------------------
# Tick / bar loaders
# ---------------------------------------------------------------------------

def load_ticks(symbol: str) -> Optional[pd.DataFrame]:
    """
    Read raw ticks from data/ticks/HYDRA4_TICKS_<sym>.parquet, projecting
    to the minimal column set needed for H1 (time, bid, ask, last, volume)
    and downcasting to float32 to keep 50M+ tick streams under ~1 GB.

    `last` and `volume` are optional — synthesised if absent.
    """
    from config import TICKS_DIR
    p = TICKS_DIR / f"HYDRA4_TICKS_{symbol}.parquet"
    if not p.exists():
        log.warning("[%s] no tick parquet at %s", symbol, p); return None
    try:
        cols = ["time_msc", "bid", "ask", "last", "volume"]
        df = pd.read_parquet(p, columns=cols)
    except Exception:
        # Older parquet may lack `last` or `volume`.
        df = pd.read_parquet(p, columns=["time_msc", "bid", "ask"])
    if "time_msc" in df.columns:
        df["time"] = pd.to_datetime(df["time_msc"].to_numpy(), unit="ms", utc=True)
        df = df.drop(columns="time_msc")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    else:
        raise ValueError("ticks need time_msc or time column")
    if not {"bid", "ask"}.issubset(df.columns):
        raise ValueError("ticks need bid/ask columns")
    # Downcast — bid/ask sit comfortably in float32 for FX & metals.
    df["bid"]  = df["bid"].astype(np.float32)
    df["ask"]  = df["ask"].astype(np.float32)
    if "last" in df.columns:
        df["last"] = df["last"].astype(np.float32)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(np.float32)
    df["mid"]    = ((df["bid"] + df["ask"]) * np.float32(0.5)).astype(np.float32)
    df["spread"] = (df["ask"] - df["bid"]).astype(np.float32)
    # Tick parquets are often newest-first; sort ascending without a full
    # DataFrame copy by sorting the underlying time array first.
    t_ns = df["time"].astype("int64").to_numpy()
    if not (t_ns[:-1] <= t_ns[1:]).all():
        order = np.argsort(t_ns)
        df = df.iloc[order].reset_index(drop=True)
    return df


def ticks_to_ohlcv(ticks: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Build time-uniform OHLCV bars at `period` (pandas resample alias:
    '5min', '15min', '1h', '4h', '1d'). Output schema:
        time, open, high, low, close, tick_volume, spread
    """
    df = ticks.set_index("time").sort_index()
    g = df["mid"].resample(period)
    bars = pd.DataFrame({
        "open":  g.first(),
        "high":  g.max(),
        "low":   g.min(),
        "close": g.last(),
    })
    bars["tick_volume"] = df["mid"].resample(period).count()
    bars["spread"]      = df["spread"].resample(period).mean()
    return bars.dropna(subset=["close"]).reset_index()


def ticks_to_tickbars(ticks: pd.DataFrame,
                       ticks_per_bar: int = 100) -> pd.DataFrame:
    """
    Information-uniform tick-bars: every N ticks closes a new bar. Output
    schema includes order-flow microstructure (signed-volume / OFI) which
    is the whole point for H1.
        time, open, high, low, close, mid, spread, ofi, taker_ratio,
        signed_volume, total_volume

    Memory-frugal: every working array is float32 (load_ticks already
    downcasts on read). The prior version cast back to float64 which
    doubled per-worker peak memory and OOM-killed parallel runs on the
    biggest tick parquets (ETHUSD 1.8 GB). All `del + gc.collect` calls
    are intentional — they free intermediate buffers BEFORE the next
    big array is allocated, instead of letting Python's refcount lag.
    """
    import gc
    needed = {"bid", "ask"}
    if not needed.issubset(ticks.columns):
        raise ValueError(f"ticks missing {needed - set(ticks.columns)}")
    n_bars = len(ticks) // ticks_per_bar
    if n_bars == 0:
        raise ValueError(f"need >= {ticks_per_bar} ticks; got {len(ticks)}")
    n_used = n_bars * ticks_per_bar
    # iloc[:].to_numpy() — pull arrays out directly without copy-on-DataFrame,
    # then drop the DataFrame slice so its memory can be reclaimed.
    bid = ticks["bid"].iloc[:n_used].to_numpy(dtype=np.float32, copy=False)
    ask = ticks["ask"].iloc[:n_used].to_numpy(dtype=np.float32, copy=False)
    if "volume" in ticks.columns:
        vol = ticks["volume"].iloc[:n_used].to_numpy(dtype=np.float32, copy=False)
    else:
        vol = np.ones(n_used, dtype=np.float32)
    has_last = "last" in ticks.columns
    if has_last:
        last = ticks["last"].iloc[:n_used].to_numpy(dtype=np.float32, copy=False)
    time_arr = ticks["time"].iloc[:n_used].to_numpy()   # datetime64[ns]; keep native

    mid    = ((bid + ask) * np.float32(0.5)).astype(np.float32)
    spread = (ask - bid).astype(np.float32)
    vol    = np.where(vol > 0, vol, np.float32(1.0)).astype(np.float32)
    if has_last:
        signed = np.zeros(n_used, dtype=np.float32)
        signed[last > mid + 1e-7] = 1.0
        signed[last < mid - 1e-7] = -1.0
        del last
    else:
        signed = np.zeros(n_used, dtype=np.float32)
    del bid, ask
    gc.collect()

    shape = (n_bars, ticks_per_bar)
    mid_t    = mid.reshape(shape)
    spread_t = spread.reshape(shape)
    signed_t = signed.reshape(shape)
    vol_t    = vol.reshape(shape)
    time_t   = time_arr.reshape(shape)

    # Reductions: each output is n_bars-sized (small — a few MB).
    sv = (signed_t * vol_t).sum(axis=1).astype(np.float32)
    av = vol_t.sum(axis=1).astype(np.float32)
    ofi = np.clip(sv / np.where(av > 0, av, np.float32(1.0)),
                  np.float32(-1.0), np.float32(1.0))
    taker = (signed_t != 0).mean(axis=1).astype(np.float32)

    out = pd.DataFrame({
        "time":          time_t[:, -1],           # bar close timestamp
        "open":          mid_t[:, 0].astype(np.float32),
        "high":          mid_t.max(axis=1).astype(np.float32),
        "low":           mid_t.min(axis=1).astype(np.float32),
        "close":         mid_t[:, -1].astype(np.float32),
        "mid":           mid_t[:, -1].astype(np.float32),
        "spread":        spread_t.mean(axis=1).astype(np.float32),
        "ofi":           ofi,
        "taker_ratio":   taker,
        "signed_volume": sv,
        "total_volume":  av,
    })
    out["time"] = pd.to_datetime(out["time"], utc=True)
    # Free the per-tick reshape views + buffers BEFORE returning so the
    # caller's downstream allocations (feature builder, XGBoost matrices)
    # don't race against the old arrays. Saves ~1-2 GB peak on ETHUSD.
    del mid, spread, signed, vol, time_arr
    del mid_t, spread_t, signed_t, vol_t, time_t
    del sv, av, ofi, taker
    gc.collect()
    return out


def _build_bars_lite(symbol: str, period: str) -> Optional[pd.DataFrame]:
    """
    Memory-frugal time-resample. Reads only (time_msc, bid, ask) from the
    tick parquet, casts to float32, then resamples directly off a Series.
    Avoids the multi-column DataFrame consolidation that OOMs on 50M+ tick
    streams. Output schema matches ticks_to_ohlcv (open/high/low/close/
    tick_volume/spread).
    """
    from config import TICKS_DIR
    p = TICKS_DIR / f"HYDRA4_TICKS_{symbol}.parquet"
    if not p.exists():
        log.warning("[%s] no tick parquet at %s", symbol, p); return None
    try:
        raw = pd.read_parquet(p, columns=["time_msc", "bid", "ask"])
    except Exception:
        # Older parquet may not have `time_msc` — fall back to `time`.
        raw = pd.read_parquet(p, columns=["time", "bid", "ask"])
        raw["time_msc"] = (pd.to_datetime(raw["time"], utc=True).astype("int64")
                            // 10**6)
    log.info("[%s] read %d ticks (%.0f MB on disk)  building %s bars ...",
             symbol, len(raw), p.stat().st_size / 1e6, period)
    bid32 = raw["bid"].astype(np.float32).to_numpy()
    ask32 = raw["ask"].astype(np.float32).to_numpy()
    times = pd.to_datetime(raw["time_msc"].to_numpy(), unit="ms", utc=True)
    del raw
    mid    = ((bid32 + ask32) * np.float32(0.5)).astype(np.float32)
    spread = (ask32 - bid32).astype(np.float32)
    del bid32, ask32
    # Sort: many MT5 dumps are newest-first; resample needs monotonic asc.
    order = np.argsort(times.asi8)
    if not (order[:-1] <= order[1:]).all():
        times = times[order]; mid = mid[order]; spread = spread[order]
    mid_s    = pd.Series(mid,    index=times, name="mid",    copy=False)
    spread_s = pd.Series(spread, index=times, name="spread", copy=False)
    del mid, spread, times
    g_mid    = mid_s.resample(period)
    bars = pd.DataFrame({
        "open":  g_mid.first(),
        "high":  g_mid.max(),
        "low":   g_mid.min(),
        "close": g_mid.last(),
    })
    bars["tick_volume"] = mid_s.resample(period).count().astype(np.int32)
    bars["spread"]      = spread_s.resample(period).mean().astype(np.float32)
    return bars.dropna(subset=["close"]).reset_index(names="time")


def load_or_build_bars(symbol: str, period: str) -> Optional[pd.DataFrame]:
    """
    Resolve OHLCV bars for `symbol` at `period`. Cache derivatives under
    data/parquet/HYDRA4_<PERIOD>FROMTICKS_<SYM>.parquet so the second
    call is fast. Uses _build_bars_lite for the construction step so 50M+
    tick streams don't OOM.
    """
    from config import PARQUET_DIR
    cache_tag = period.upper().replace("MIN", "M").replace("H", "H")
    cache = PARQUET_DIR / f"HYDRA4_{cache_tag}FROMTICKS_{symbol}.parquet"
    if cache.exists():
        log.info("[%s] loading %s bar cache: %s", symbol, period, cache.name)
        return pd.read_parquet(cache)
    bars = _build_bars_lite(symbol, period)
    if bars is None:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(cache, compression="zstd", compression_level=9)
    log.info("[%s] cached %d %s bars -> %s",
             symbol, len(bars), period, cache.name)
    return bars


# ---------------------------------------------------------------------------
# Walk-forward + metrics
# ---------------------------------------------------------------------------

def chronological_split(n: int, val_frac: float = 0.30, gap: int = 20):
    """Return (train_idx_slice, val_idx_slice) with `gap` rows reserved
    between train and val so labels with horizon up to `gap` cannot leak."""
    val_size = max(50, int(n * val_frac))
    train_end = max(0, n - val_size - gap)
    return slice(0, train_end), slice(train_end + gap, n)


def profit_factor(pnl: np.ndarray) -> float:
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return float(wins / max(1e-12, losses))


def sharpe(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe ratio. `returns` are per-period (not annualised).
    periods_per_year:
      M5  ≈ 252 * 24 * 12 = 72576
      H1  ≈ 252 * 24 = 6048
      H4  ≈ 252 * 6  = 1512
      D1  ≈ 252
    Caller supplies the right value."""
    r = returns[np.isfinite(returns)]
    if len(r) < 2 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the cumulative-return curve."""
    if len(returns) == 0:
        return 0.0
    eq = np.cumsum(returns)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(-dd.min() / max(1e-12, peak.max() if peak.max() > 0 else 1.0))


def passive_pf(price_returns: np.ndarray, cost_per_bar: float = 0.0):
    """PF of always-long and always-short baselines, net of `cost_per_bar`."""
    long_r  =  price_returns - cost_per_bar
    short_r = -price_returns - cost_per_bar
    return profit_factor(long_r), profit_factor(short_r)


# ---------------------------------------------------------------------------
# Skill gate
# ---------------------------------------------------------------------------

def skill_gate(model_pf: float,
               passive_long_pf: float,
               passive_short_pf: float,
               n_trades: int,
               *,
               excess_min: float = 0.10,
               pf_min: float = 1.20,
               n_min: int = 30) -> tuple[bool, float]:
    """
    Returns (passes, excess_pf). A strategy passes iff:
       excess = model_pf - max(passive_long, passive_short) >= excess_min
       AND model_pf >= pf_min
       AND n_trades >= n_min
    """
    baseline = max(passive_long_pf, passive_short_pf)
    excess = model_pf - baseline
    ok = (excess >= excess_min) and (model_pf >= pf_min) and (n_trades >= n_min)
    return bool(ok), float(excess)


# ---------------------------------------------------------------------------
# XGBoost training + ONNX export
# ---------------------------------------------------------------------------

def fit_xgb_binary(X_tr, y_tr, X_va, y_va, *,
                    n_estimators: int = 300,
                    max_depth: int = 4,
                    lr: float = 0.05,
                    early_stop: int = 20,
                    seed: int = 42,
                    use_gpu: bool = False):
    """Train an XGBoost binary classifier with temperature calibration on val.

    use_gpu=True moves DMatrix + model to CUDA (XGBoost 2.0+ `device="cuda"`).
    Modest RAM relief (~50-200 MB per training is offloaded to VRAM); on
    200K x 16 tabular data the wall-clock is typically the same as CPU due
    to PCIe transfer overhead. Auto-falls back to CPU if CUDA isn't available
    or the local XGBoost build lacks GPU support.

    Returns (model, cal_probs_on_val, temperature_T).
    """
    import xgboost as xgb
    params = dict(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, objective="binary:logistic",
        eval_metric="logloss", early_stopping_rounds=early_stop,
        random_state=seed, tree_method="hist",
    )
    if use_gpu:
        # Sniff CUDA availability cheaply — `torch.cuda.is_available()` is
        # the most reliable check across XGBoost versions; xgboost itself
        # raises a confusing message when device="cuda" is requested
        # without GPU support.
        gpu_ok = False
        try:
            import torch
            gpu_ok = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            gpu_ok = False
        if gpu_ok:
            params["device"] = "cuda"
            log.info("XGBoost: using GPU (device=cuda)")
        else:
            log.info("XGBoost: GPU requested but CUDA unavailable — CPU fallback")
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_probs = model.predict_proba(X_va)[:, 1]
    try:
        from validation import fit_temperature
        logits = np.log(np.clip(val_probs, 1e-6, 1 - 1e-6) /
                        np.clip(1 - val_probs, 1e-6, 1))
        T = fit_temperature(logits, y_va.astype(np.float32))
        cal = 1.0 / (1.0 + np.exp(-logits / T))
    except Exception as e:
        log.warning("temperature calibration failed: %s — using raw probs", e)
        cal = val_probs; T = 1.0
    return model, cal, float(T)


def export_xgb_onnx(model, feature_dim: int, out_path: Path) -> bool:
    """Convert a fitted XGBClassifier to ONNX. Returns True on success."""
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        log.warning("onnxmltools missing — cannot export %s", out_path.name)
        return False
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        initial = [("features", FloatTensorType([None, feature_dim]))]
        onx = onnxmltools.convert_xgboost(model, initial_types=initial)
        out_path.write_bytes(onx.SerializeToString())
        log.info("ONNX written: %s", out_path.name)
        return True
    except Exception as e:
        log.warning("ONNX export failed for %s: %s", out_path.name, e)
        return False


# ---------------------------------------------------------------------------
# Feature engineering primitives shared across strategies (strictly backward)
# ---------------------------------------------------------------------------

def rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Backward-causal rolling z-score over `window` samples."""
    s = pd.Series(x)
    m = s.rolling(window, min_periods=2).mean().to_numpy()
    sd = s.rolling(window, min_periods=2).std().to_numpy()
    sd = np.where(sd > 1e-12, sd, 1.0)
    return np.nan_to_num((x - m) / sd, nan=0.0)


def backward_atr(high, low, close, period: int = 14) -> np.ndarray:
    """Backward-causal ATR."""
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low,
                            np.abs(high - prev),
                            np.abs(low  - prev)])
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def backward_vol_regime_ratio(close, high, low, *,
                                short_window: int = 20,
                                long_window: int = 500) -> np.ndarray:
    """
    Backward-causal vol-regime ratio: ATR(short_window) divided by the mean
    of ATR(short_window) over the trailing `long_window` bars.

    Interpretation:
        ratio > 1.0  : current realised volatility is HIGHER than its trailing
                       baseline — typically a trending or breakout regime
                       where trend-following strategies have edge.
        ratio < 1.0  : current vol is LOWER than baseline — typically chop /
                       range-bound regime where trend-following gets whipsawed.

    Used by train_h4_trend's --vol-filter-ratio knob to zero out positions
    during chop sub-windows. The Sept-Nov 2025 GOLD sub-window that produced
    val Sharpe -2.25 on the fresh-data run was a low-vol chop regime; the
    filter mechanically eliminates exactly that class of period.

    Strictly causal — bar i uses only bars 0..i.
    """
    atr_short = backward_atr(high, low, close, short_window)
    baseline = pd.Series(atr_short).rolling(
        long_window, min_periods=long_window // 4).mean().to_numpy()
    safe = np.where(baseline > 1e-12, baseline, 1.0)
    return atr_short / safe


def backward_donchian(high, low, window: int):
    """Returns (donchian_high, donchian_low) over the PREVIOUS `window` bars
    (excludes current bar — so signals computed from these are causal)."""
    hh = pd.Series(high).shift(1).rolling(window, min_periods=2).max().to_numpy()
    ll = pd.Series(low ).shift(1).rolling(window, min_periods=2).min().to_numpy()
    return hh, ll


def triple_barrier_outcome(high, low, entry_idx: int, *,
                            direction: int, sl: float, tp: float,
                            timeout: int) -> tuple[int, int, int]:
    """Walk forward `timeout` bars from entry_idx and return (tp_hit, sl_hit, timeout_hit).
    Causal w.r.t. the caller: we only look at bars AFTER entry_idx."""
    tp_hit = sl_hit = 0
    n = len(high)
    end = min(entry_idx + timeout, n - 1)
    for k in range(1, timeout + 1):
        j = entry_idx + k
        if j > end:
            break
        hi = high[j]; lo = low[j]
        if direction > 0:
            if hi >= tp: tp_hit = 1; break
            if lo <= sl: sl_hit = 1; break
        else:
            if lo <= tp: tp_hit = 1; break
            if hi >= sl: sl_hit = 1; break
    timeout_hit = 1 if (tp_hit == 0 and sl_hit == 0) else 0
    return tp_hit, sl_hit, timeout_hit


# ---------------------------------------------------------------------------
# Result-card formatting
# ---------------------------------------------------------------------------

def format_result_card(strategy: str, symbol: str, *,
                       n_trades: int,
                       model_pf: float, passive_long: float, passive_short: float,
                       excess: float, wr: float, sharpe_val: float, mdd: float,
                       deploy: bool) -> str:
    return (f"[{strategy}:{symbol}] "
            f"N={n_trades:<5d}  PF={model_pf:.3f}  "
            f"passive(L/S)={passive_long:.2f}/{passive_short:.2f}  "
            f"excess={excess:+.3f}  WR={wr:.3f}  "
            f"Sharpe={sharpe_val:.2f}  MDD={mdd:.1%}  "
            f"-> {'DEPLOY' if deploy else 'BLOCKED'}")


# ---------------------------------------------------------------------------
# Symbol roster (default symbol set for the strategies — anything with
# a tick parquet on disk is fair game)
# ---------------------------------------------------------------------------

def discover_tick_symbols() -> list[str]:
    """List of symbols that have a tick parquet available."""
    from config import TICKS_DIR
    return sorted(p.stem.replace("HYDRA4_TICKS_", "")
                  for p in TICKS_DIR.glob("HYDRA4_TICKS_*.parquet"))
