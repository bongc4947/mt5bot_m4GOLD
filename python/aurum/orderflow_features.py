"""
aurum/orderflow_features.py — microstructure / order-flow features for
the MetaTrend meta-gate.

The HYDRA4_TBARS_GOLD_100tpb.parquet ships 8 nominally-microstructure
columns, but a full audit shows only THREE carry usable signal on this
broker feed:

    microprice_drift   real, ~ +/- 0.65 quantiles
    spread             real, 0.27 .. 1.67 quantiles
    spread_regime      real, 0.13 .. 0.31 (narrow but live)

The other five (`ofi`, `cvd`, `taker_ratio`, `tick_volume`, `real_volume`)
are saturated or constant in the source parquet — the upstream tick-bar
builder never wired trade-classification because MT5 retail ticks do not
carry the buy/sell aggressor flag for most brokers. This is a data-
availability fact, not a code bug.

So the order-flow feature pack here is built from the three live signals
plus an activity proxy (tbar count per M5 bucket). Five features:

    of_micro_drift_z48   rolling z-score of per-M5 micro_drift_sum (48 bars)
    of_micro_drift_cum24 cumulative micro_drift over 24 M5 bars / atr14_norm
    of_spread_z96        rolling z-score of mean spread (96 bars)
    of_sreg_now          current bar's max spread_regime (narrow gauge)
    of_activity_z96      rolling z-score of tbar count (96 bars)

All causal: feature for M5 anchor i uses only tbars whose `time` <= the
close of M5 bar i. Each tbar's `time` is its close timestamp; we floor it
to 5-min, which assigns it to the M5 bar STARTING at that floor — that
M5 bar's decision time is its OWN close, by which point every tbar in
the bucket is already closed. No leak.

Cached on disk like tspulse_features so repeated trainer runs are
instant.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ORDERFLOW_FEATURES = [
    "of_micro_drift_z48",
    "of_micro_drift_cum24",
    "of_spread_z96",
    "of_sreg_now",
    "of_activity_z96",
]
N_ORDERFLOW_FEATURES = len(ORDERFLOW_FEATURES)


def _cache_path(symbol: str, anchors: np.ndarray) -> Path:
    h = hashlib.sha1(anchors.astype(np.int64).tobytes()).hexdigest()[:16]
    base = Path(__file__).parent.parent.parent / "onnx_out"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"orderflow_features_{symbol}_{len(anchors)}_{h}.npy"


def _find_tbar_parquet(symbol: str) -> Path | None:
    """Locate HYDRA4_TBARS_{SYMBOL}_100tpb.parquet across known dirs."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PARQUET_DIR, TICKS_DIR
    name = f"HYDRA4_TBARS_{symbol}_100tpb.parquet"
    for d in (PARQUET_DIR, TICKS_DIR):
        p = d / name
        if p.exists():
            return p
    return None


def _bucket_tbars(m5: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Aggregate tick-bars onto M5 buckets.

    Returns a frame aligned to `m5` (same length, same `time`) with columns:
      micro_drift_sum, spread_mean, sreg_max, tbar_count
    Missing buckets (no tbars) get 0 / NaN; we fill forward conservatively.
    """
    p = _find_tbar_parquet(symbol)
    if p is None:
        raise FileNotFoundError(
            f"HYDRA4_TBARS_{symbol}_100tpb.parquet not found in either "
            "PARQUET_DIR or TICKS_DIR — needed for order-flow features.")
    tb = pd.read_parquet(p, columns=[
        "time", "microprice_drift", "spread", "spread_regime"])
    tb["time"] = pd.to_datetime(tb["time"], utc=True)
    tb["m5_bucket"] = tb["time"].dt.floor("5min")
    agg = tb.groupby("m5_bucket").agg(
        micro_drift_sum=("microprice_drift", "sum"),
        spread_mean=("spread", "mean"),
        sreg_max=("spread_regime", "max"),
        tbar_count=("time", "size"),
    )
    m5 = m5.copy()
    m5["m5_bucket"] = pd.to_datetime(m5["time"], utc=True).dt.floor("5min")
    merged = m5.merge(agg, left_on="m5_bucket", right_index=True, how="left")
    log.info("[orderflow] tbar coverage %d/%d M5 buckets",
             int(merged["tbar_count"].notna().sum()), len(merged))
    # Missing micro_drift means quiet bar → 0 flow. Missing spread/sreg →
    # carry forward the last known value (spread regime is persistent).
    merged["micro_drift_sum"] = merged["micro_drift_sum"].fillna(0.0)
    merged["tbar_count"] = merged["tbar_count"].fillna(0.0)
    merged["spread_mean"] = merged["spread_mean"].ffill().fillna(0.0)
    merged["sreg_max"] = merged["sreg_max"].ffill().fillna(0.0)
    return merged[["micro_drift_sum", "spread_mean", "sreg_max", "tbar_count"]]


def extract(m5: pd.DataFrame, anchors: np.ndarray,
            symbol: str = "GOLD") -> np.ndarray:
    """
    Return float32[len(anchors), N_ORDERFLOW_FEATURES].

    Each anchor i uses only M5 bars 0..i (the bar at i is itself closed
    at decision time, so its own bucket is fair game).
    """
    cp = _cache_path(symbol, anchors)
    if cp.exists():
        log.info("[orderflow] cache hit: %s", cp.name)
        return np.load(cp)

    log.info("[orderflow] building features for %d anchors ...", len(anchors))
    micro = _bucket_tbars(m5, symbol)
    md = micro["micro_drift_sum"].to_numpy(np.float64)
    sp = micro["spread_mean"].to_numpy(np.float64)
    sr = micro["sreg_max"].to_numpy(np.float64)
    ac = micro["tbar_count"].to_numpy(np.float64)
    eps = 1e-9

    def z(a: np.ndarray, w: int) -> np.ndarray:
        s = pd.Series(a)
        mu = s.rolling(w, min_periods=max(8, w // 4)).mean().to_numpy()
        sd = s.rolling(w, min_periods=max(8, w // 4)).std().to_numpy()
        return np.where(sd > eps, (a - mu) / (sd + eps), 0.0)

    md_z48 = z(md, 48)
    md_cum24 = pd.Series(md).rolling(24, min_periods=8).sum().to_numpy()
    # normalise the cumulative-flow by a rolling activity scale so it's
    # comparable across regimes (heavy session vs Asian lull)
    md_scale = pd.Series(np.abs(md)).rolling(96, min_periods=16).mean().to_numpy()
    md_cum24 = md_cum24 / (md_scale * 24.0 + eps)

    sp_z96 = z(sp, 96)
    ac_z96 = z(ac, 96)
    # sreg_now is already on a known narrow scale; centre around the
    # plateau (0.20) so the gate sees deviations
    sreg_now = sr - 0.20

    all_feat = np.column_stack([md_z48, md_cum24, sp_z96, sreg_now, ac_z96])
    all_feat = np.nan_to_num(all_feat, nan=0.0, posinf=0.0, neginf=0.0)
    out = all_feat[anchors].astype(np.float32)

    np.save(cp, out)
    log.info("[orderflow] cached -> %s  shape=%s", cp.name, out.shape)
    return out
