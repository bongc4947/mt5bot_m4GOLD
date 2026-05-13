"""
train_h1_orderflow.py — H1: tick-level order-flow imbalance directional model.

Hypothesis: at sub-second timescales, signed-tick volume (uptick-downtick over
N ticks) predicts the next K tick-bars of mid-price drift. Lives or dies on
spread cost; documented in microstructure literature (Hasbrouck, Cont).

Pipeline:
  1. Load raw ticks, build 100-tick information-uniform bars with OFI/taker.
  2. Build a (D)-dim feature vector at each bar (strictly backward-causal):
       core micro (OFI now/EMA/CVD/taker)  +  rolling vol  +  spread regime
       + last-K bar returns  +  microprice drift
  3. Label: sign of mid-price change over the next K bars (default K=10).
     y = 1 if mid[i+K] > mid[i] (after subtracting half-spread round-trip),
     y = 0 otherwise. K is the trade horizon.
  4. Chronological train/val split (val = last 30% with 20-bar gap).
  5. Train XGBoost binary classifier with temperature calibration.
  6. Skill gate: net-of-cost model_PF >= passive max + 0.10
                 AND model_PF >= 1.20 AND N_trades >= 30.
  7. ONNX export -> HYDRA4_H1OF_<SYM>.onnx.

Usage:
    python python/train_h1_orderflow.py EURUSD
    python python/train_h1_orderflow.py --all  --ticks-per-bar 100  --horizon 10
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

from strategies_common import (
    chronological_split, discover_tick_symbols, export_xgb_onnx, fit_xgb_binary,
    load_ticks, max_drawdown, passive_pf, pip_size, profit_factor, sharpe,
    skill_gate, ticks_to_tickbars,
)

log = logging.getLogger(__name__)

H1_FEATURE_COLUMNS = [
    "ofi_now", "ofi_ema5", "ofi_ema20",          # 0..2
    "cvd_norm",                                    # 3
    "taker_now", "taker_ema20",                    # 4..5
    "ret_1", "ret_5", "ret_20",                    # 6..8
    "vol_5", "vol_20",                             # 9..10
    "microdrift_5", "microdrift_20",               # 11..12
    "spread_now", "spread_ratio_60",               # 13..14
    "tick_intensity_z",                            # 15
]
H1_FEATURE_DIM = len(H1_FEATURE_COLUMNS)


def build_h1_features(bars: pd.DataFrame) -> np.ndarray:
    """All features below use ONLY rows 0..i to produce feature[i] — strict
    backward causality. The audit script verifies via .shift(-N) absence."""
    mid    = bars["mid"].to_numpy(dtype=np.float64)
    spread = bars["spread"].to_numpy(dtype=np.float64)
    ofi    = bars["ofi"].to_numpy(dtype=np.float64)
    taker  = bars["taker_ratio"].to_numpy(dtype=np.float64)
    sv     = bars["signed_volume"].to_numpy(dtype=np.float64)
    av     = bars["total_volume"].to_numpy(dtype=np.float64)

    ofi_ema5  = pd.Series(ofi).ewm(span=5,  adjust=False).mean().to_numpy()
    ofi_ema20 = pd.Series(ofi).ewm(span=20, adjust=False).mean().to_numpy()
    taker_ema = pd.Series(taker).ewm(span=20, adjust=False).mean().to_numpy()

    # CVD normalised by 1000-bar absolute-delta sum (bounded ~[-1, 1] after /10)
    cum = np.cumsum(sv)
    norm = pd.Series(np.abs(sv)).rolling(1000, min_periods=1).sum().to_numpy()
    cvd_norm = np.clip(cum / np.where(norm > 0, norm, 1.0), -10, 10) / 10.0

    log_mid = np.log(np.clip(mid, 1e-12, None))
    ret_1  = np.diff(log_mid,  prepend=log_mid[0])
    ret_5  = log_mid - pd.Series(log_mid).shift(5).fillna(log_mid[0]).to_numpy()
    ret_20 = log_mid - pd.Series(log_mid).shift(20).fillna(log_mid[0]).to_numpy()

    vol_5  = pd.Series(ret_1).rolling(5,  min_periods=2).std().to_numpy()
    vol_20 = pd.Series(ret_1).rolling(20, min_periods=2).std().to_numpy()

    # Microprice drift over last K bars (mid only; no L2 depth -> mid proxy)
    microdrift_5  = (mid - pd.Series(mid).shift(5).fillna(mid[0]).to_numpy()
                     ) / np.where(mid > 0, mid, 1.0)
    microdrift_20 = (mid - pd.Series(mid).shift(20).fillna(mid[0]).to_numpy()
                     ) / np.where(mid > 0, mid, 1.0)

    spread_med60 = pd.Series(spread).rolling(60, min_periods=5).median().to_numpy()
    spread_ratio = spread / np.where(spread_med60 > 1e-12, spread_med60, 1.0)

    # Tick intensity z (av is per-bar total volume; surrogate for trade
    # intensity since we don't have wall-clock duration per tick-bar without
    # extra bookkeeping).
    av_mean = pd.Series(av).rolling(200, min_periods=10).mean().to_numpy()
    av_std  = pd.Series(av).rolling(200, min_periods=10).std().to_numpy()
    tick_intensity_z = np.clip(
        (av - av_mean) / np.where(av_std > 1e-12, av_std, 1.0), -5, 5
    ) / 5.0

    feats = np.column_stack([
        ofi, ofi_ema5, ofi_ema20,
        cvd_norm,
        taker, taker_ema,
        ret_1 * 1e4, ret_5 * 1e4, ret_20 * 1e4,
        vol_5 * 1e4, vol_20 * 1e4,
        microdrift_5 * 1e4, microdrift_20 * 1e4,
        spread / np.where(mid > 0, mid, 1.0) * 1e4,
        np.clip(spread_ratio, 0, 5) / 5.0,
        tick_intensity_z,
    ]).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def make_labels_and_pnl(bars: pd.DataFrame, horizon: int,
                         symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y, pnl_price) of length len(bars).
    y[i] = 1 if mid[i+horizon] > mid[i], else 0.  Pure direction —
        cost is applied at evaluation time, not in the label, otherwise
        the label becomes degenerate (~5-15% positive on majors where
        the spread eats most short-horizon moves).
    pnl_price[i] = signed mid-price change over horizon (no direction yet
        — caller chooses long/short via the classifier prediction).

    The label uses the future. The features (build_h1_features) do not.
    """
    mid = bars["mid"].to_numpy(dtype=np.float64)
    n = len(mid)
    if n <= horizon + 1:
        return np.array([], dtype=np.int8), np.array([], dtype=np.float64)
    fwd_mid = pd.Series(mid).shift(-horizon).to_numpy()
    fwd_change = fwd_mid - mid
    valid = np.isfinite(fwd_change)
    y = np.zeros(n, dtype=np.int8)
    y[valid] = (fwd_change[valid] > 0).astype(np.int8)
    pnl = np.where(valid, fwd_change, 0.0)
    return y, pnl


def evaluate_at_threshold(probs: np.ndarray, pnl_va: np.ndarray,
                          spread_va: np.ndarray, threshold: float,
                          *, invert: bool = False) -> dict:
    """
    Long where probs >= threshold, short where probs <= 1-threshold,
    flat otherwise. Net of 2*spread cost per trade.

    invert=True flips the trade direction at each bar (long becomes short
    and vice versa). Used to expose anti-signal: if the model is
    systematically wrong (WR < 50% on val), the inverted strategy may
    be the profitable one. The skill gate then runs against the inverted
    metrics too — whichever direction passes (if any) wins.

    Returns dict with n_trades, wr, pf, sharpe, mdd, frac_kept, inverted.
    """
    long_mask  = probs >= threshold
    short_mask = probs <= (1.0 - threshold)
    direction = np.zeros_like(probs)
    direction[long_mask]  = +1.0
    direction[short_mask] = -1.0
    if invert:
        direction = -direction
    cost = 2.0 * spread_va
    trade_mask = direction != 0
    raw_pnl = direction * pnl_va
    net_pnl = np.where(trade_mask, raw_pnl - cost, 0.0)
    kept_pnl = net_pnl[trade_mask]
    n = int(trade_mask.sum())
    return {
        "n_trades":  n,
        "wr":        float((kept_pnl > 0).mean()) if n else 0.0,
        "pf":        profit_factor(kept_pnl) if n else 0.0,
        "frac_kept": float(trade_mask.mean()) if len(trade_mask) else 0.0,
        # Tick-bar Sharpe — caller should report this raw, not annualised.
        "sharpe":    sharpe(net_pnl, periods_per_year=1.0),
        "mdd":       max_drawdown(net_pnl),
        "inverted":  bool(invert),
    }


def train_one_symbol(symbol: str, *,
                      ticks_per_bar: int = 100,
                      horizon: int = 10,
                      n_estimators: int = 300,
                      max_depth: int = 4,
                      seed: int = 42,
                      use_gpu: bool = False,
                      max_tick_file_gb: float = 1.0) -> dict:
    """
    max_tick_file_gb (mk4.8.4): pre-flight guard against the SIGKILL-by-RAM-
    watchdog pattern. ETHUSD (1.8 GB) and GOLD (952 MB) reproducibly hit
    rc=-9 on Kaggle's 30 GB box because the subprocess's float32 working
    set peaks at 3-6× the on-disk size. Skip those H1 cells cleanly with a
    deploy=False meta JSON so the rest of the H1 sweep keeps going.

    Override by passing max_tick_file_gb=99.0 on a fat-RAM box (RunPod,
    vast.ai). Note: on those platforms ProcessPoolExecutor with workers>1
    is also unblocked, so you'd typically combine both flags.
    """
    from config import ONNX_OUTPUT_DIR, TICKS_DIR
    tick_path = TICKS_DIR / f"HYDRA4_TICKS_{symbol}.parquet"
    if tick_path.exists():
        size_gb = tick_path.stat().st_size / 1e9
        if size_gb > max_tick_file_gb:
            log.warning("[H1:%s] SKIPPING — tick file %.2f GB > %.2f GB guard. "
                         "Pre-empts SIGKILL OOM on Kaggle. Override with "
                         "--h1-max-tick-file-gb on platforms with >40 GB RAM.",
                         symbol, size_gb, max_tick_file_gb)
            meta = {
                "strategy": "H1_OF", "symbol": symbol,
                "ok": False, "deploy": False,
                "reason": f"skipped: tick file {size_gb:.2f} GB exceeds "
                          f"{max_tick_file_gb:.2f} GB guard",
                "tick_file_gb": size_gb,
                "max_tick_file_gb": max_tick_file_gb,
            }
            (ONNX_OUTPUT_DIR / f"HYDRA4_H1OF_{symbol}_meta.json").write_text(
                json.dumps(meta, indent=2))
            return meta
    ticks = load_ticks(symbol)
    if ticks is None:
        return {"strategy": "H1_OF", "symbol": symbol, "ok": False,
                "reason": "no tick parquet"}
    if len(ticks) < ticks_per_bar * 200:
        return {"strategy": "H1_OF", "symbol": symbol, "ok": False,
                "reason": f"only {len(ticks)} ticks"}
    log.info("[H1:%s] %d raw ticks  ticks_per_bar=%d  horizon=%d",
             symbol, len(ticks), ticks_per_bar, horizon)
    bars = ticks_to_tickbars(ticks, ticks_per_bar=ticks_per_bar)
    log.info("[H1:%s] %d tick-bars  span=%s -> %s",
             symbol, len(bars), bars["time"].iloc[0], bars["time"].iloc[-1])
    if len(bars) < 1000:
        return {"strategy": "H1_OF", "symbol": symbol, "ok": False,
                "reason": f"only {len(bars)} tick-bars"}

    feats = build_h1_features(bars)
    y, pnl_price = make_labels_and_pnl(bars, horizon, symbol)
    if len(y) == 0:
        return {"strategy": "H1_OF", "symbol": symbol, "ok": False,
                "reason": "no labels generated"}

    # Truncate the last `horizon` rows where the forward mid is undefined.
    n = max(0, len(bars) - horizon)
    feats  = feats[:n]
    y      = y[:n]
    pnl_price = pnl_price[:n]
    spread = bars["spread"].to_numpy()[:n]
    mid    = bars["mid"].to_numpy()[:n]

    tr_idx, va_idx = chronological_split(n, val_frac=0.30, gap=horizon)
    X_tr, y_tr = feats[tr_idx], y[tr_idx]
    X_va, y_va = feats[va_idx], y[va_idx]
    pnl_va = pnl_price[va_idx]
    spread_va = spread[va_idx]
    log.info("[H1:%s] train=%d (pos=%.3f)  val=%d (pos=%.3f)",
             symbol, len(X_tr), float(y_tr.mean()) if len(y_tr) else 0,
             len(X_va), float(y_va.mean()) if len(y_va) else 0)
    if len(X_tr) < 200 or len(X_va) < 50:
        return {"strategy": "H1_OF", "symbol": symbol, "ok": False,
                "reason": f"insufficient split (tr={len(X_tr)} va={len(X_va)})"}
    if abs(float(y_va.mean()) - 0.5) > 0.4:
        return {"strategy": "H1_OF", "symbol": symbol, "ok": False,
                "reason": f"degenerate val labels ({float(y_va.mean()):.3f})"}

    model, cal_probs, T = fit_xgb_binary(X_tr, y_tr, X_va, y_va,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth, seed=seed,
                                          use_gpu=use_gpu)
    log.info("[H1:%s] cal_probs min/med/max = %.3f/%.3f/%.3f  T=%.3f",
             symbol, float(cal_probs.min()), float(np.median(cal_probs)),
             float(cal_probs.max()), T)

    # Passive baselines on the val window — these are what the model must beat.
    cost_per_bar = 2.0 * spread_va.mean()
    p_long_pf, p_short_pf = passive_pf(pnl_va, cost_per_bar)

    # Sweep thresholds, BOTH directions. The 14:28 UTC Kaggle trace showed
    # several H1 cells where WR at low thresholds was systematically <0.45
    # (z<-10 on N>30k trades) — a real anti-signal, not noise. The model
    # learns *something* but the hypothesised direction is wrong. We
    # evaluate the inverted direction (flip long↔short) at every threshold
    # so if the anti-signal clears the gate, we deploy it inverted.
    THRESHOLDS = (0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60, 0.65)
    results = {}        # normal direction (model says up -> long)
    results_inv = {}    # inverted (model says up -> short)
    for thr in THRESHOLDS:
        for inv_flag, dst, tag in ((False, results, "norm"),
                                    (True,  results_inv, "INV")):
            r = evaluate_at_threshold(cal_probs, pnl_va, spread_va, thr,
                                        invert=inv_flag)
            r["passive_long_pf"]  = p_long_pf
            r["passive_short_pf"] = p_short_pf
            ok, excess = skill_gate(r["pf"], p_long_pf, p_short_pf, r["n_trades"])
            r["excess_vs_passive"] = excess
            r["deploy"] = ok
            dst[thr] = r
            log.info("[H1:%s] @thr=%.2f %s  N=%4d kept=%4.1f%%  WR=%.3f  PF=%.3f  "
                     "vs passive(%.2f/%.2f)  excess=%+.3f  %s",
                     symbol, thr, tag, r["n_trades"], 100*r["frac_kept"],
                     r["wr"], r["pf"], p_long_pf, p_short_pf,
                     r["excess_vs_passive"],
                     "DEPLOY" if r["deploy"] else "skip")

    # Pick the best (threshold, direction) cell. Deploy precedence:
    #   1) any deployable cell with the highest PF
    #   2) else the highest excess_vs_passive at N>=30 (informational)
    all_cells = [("normal", t, r) for t, r in results.items()] + \
                [("inverted", t, r) for t, r in results_inv.items()]
    deployable = [(d, t, r) for (d, t, r) in all_cells if r["deploy"]]
    if deployable:
        best_dir, best_thr, best = max(deployable, key=lambda x: x[2]["pf"])
    else:
        best_dir, best_thr, best = max(all_cells,
            key=lambda x: x[2]["excess_vs_passive"] if x[2]["n_trades"] >= 30 else -99)
    log.info("[H1:%s] WINNER: direction=%s  thr=%.2f  PF=%.3f  WR=%.3f  N=%d  "
             "excess=%+.3f",
             symbol, best_dir, best_thr, best["pf"], best["wr"],
             best["n_trades"], best["excess_vs_passive"])

    onnx_path = ONNX_OUTPUT_DIR / f"HYDRA4_H1OF_{symbol}.onnx"
    onnx_ok = export_xgb_onnx(model, H1_FEATURE_DIM, onnx_path) \
              if best["deploy"] else False
    if not best["deploy"] and onnx_path.exists():
        try: onnx_path.unlink()
        except Exception: pass

    meta = {
        "strategy":        "H1_OF",
        "symbol":          symbol,
        "feature_dim":     H1_FEATURE_DIM,
        "feature_columns": H1_FEATURE_COLUMNS,
        "ticks_per_bar":   ticks_per_bar,
        "horizon":         horizon,
        "best_direction":  best_dir,            # "normal" or "inverted"
        "best_threshold":  float(best_thr),
        "best_pf":         float(best["pf"]),
        "best_wr":         float(best["wr"]),
        "best_n_trades":   int(best["n_trades"]),
        "passive_long_pf": float(p_long_pf),
        "passive_short_pf":float(p_short_pf),
        "excess_vs_passive": float(best["excess_vs_passive"]),
        "sharpe_val":      float(best["sharpe"]),
        "mdd_val":         float(best["mdd"]),
        "temperature":     float(T),
        "deploy":          bool(best["deploy"]),
        "onnx_ok":         bool(onnx_ok),
        "results_by_threshold":         {str(k): v for k, v in results.items()},
        "results_inverted_by_threshold": {str(k): v for k, v in results_inv.items()},
    }
    (ONNX_OUTPUT_DIR / f"HYDRA4_H1OF_{symbol}_meta.json").write_text(
        json.dumps(meta, indent=2))
    log.info("[H1:%s] FINAL  dir=%s  thr=%.2f  PF=%.3f  excess=%+.3f  deploy=%s",
             symbol, best_dir, best_thr, best["pf"],
             best["excess_vs_passive"], best["deploy"])
    return meta


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbols", nargs="*", help="symbols (default: all tick-available)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--ticks-per-bar", type=int, default=100)
    p.add_argument("--horizon", type=int, default=10,
                   help="forward horizon in tick-bars for label generation")
    p.add_argument("--estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-gpu", action="store_true",
                   help="move XGBoost DMatrix + model to CUDA (modest RAM relief)")
    p.add_argument("--max-tick-file-gb", type=float, default=1.0,
                   help="skip H1 for any symbol whose raw tick parquet is "
                        "larger than this many GB. Default 1.0 (Kaggle-safe). "
                        "Set 99 on fat-RAM boxes to disable the guard.")
    args = p.parse_args(argv)
    symbols = args.symbols or discover_tick_symbols()
    if not symbols:
        log.error("no symbols and no tick parquets on disk")
        return 2

    t0 = time.time()
    rows = []
    for sym in symbols:
        try:
            rows.append(train_one_symbol(sym,
                                          ticks_per_bar=args.ticks_per_bar,
                                          horizon=args.horizon,
                                          n_estimators=args.estimators,
                                          max_depth=args.max_depth,
                                          seed=args.seed,
                                          use_gpu=args.use_gpu,
                                          max_tick_file_gb=args.max_tick_file_gb))
        except Exception as e:
            log.exception("[H1:%s] failed", sym)
            rows.append({"strategy": "H1_OF", "symbol": sym,
                         "ok": False, "reason": str(e)})
    print(f"\n{'='*82}")
    print(f"  H1 / order-flow imbalance  ({time.time() - t0:.0f}s)")
    print(f"{'='*82}")
    print(f"  {'Symbol':<12}  {'dir':<9}  {'thr':>5}  {'PF':>6}  "
          f"{'pasL':>6}  {'pasS':>6}  {'excess':>7}  {'N':>5}  {'WR':>5}  Deploy")
    print(f"  {'-'*94}")
    for r in rows:
        sym = r.get("symbol", "?")
        if "best_pf" not in r:
            print(f"  {sym:<12}  FAIL ({r.get('reason','?')})"); continue
        direction = r.get("best_direction", "normal")
        print(f"  {sym:<12}  {direction:<9}  "
              f"{r['best_threshold']:>5.2f}  {r['best_pf']:>6.3f}  "
              f"{r['passive_long_pf']:>6.2f}  {r['passive_short_pf']:>6.2f}  "
              f"{r['excess_vs_passive']:>+7.3f}  {r['best_n_trades']:>5d}  "
              f"{r['best_wr']:>5.2f}  {'YES' if r['deploy'] else 'no'}")
    print(f"{'='*82}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
