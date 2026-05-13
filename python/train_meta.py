"""
train_meta.py — meta-labeling trainer (Lopez de Prado AFML Ch.3).

mk4.6: the answer to "how do we make ML work on FX where rule + ML alone
both fail?"

CONCEPT
-------
A *primary* rule (z>2.5 mean-reversion fade, same as backtest_meanreversion.py)
generates candidate trades. ~1.5% of bars on EURUSD M5 fire the rule.

A *secondary* binary classifier — trained ONLY on rule-fired bars — predicts
whether each candidate will win or lose. The two stack:

    rule fires + meta says "yes" -> trade
    rule fires + meta says "no"  -> skip
    rule does not fire           -> no candidate

This is the standard quant-fund technique for exactly the EURUSD-style
problem. The rule alone has PF<1; the meta-filter alone is useless;
together they can clear PF>1.3 because the ML solves an *easier* problem
than direction-prediction-at-any-random-bar:

  Rule alone (EURUSD M5):        WR 30%   PF 0.59  (loses to cost)
  Direct direction MLP:          WR 53%   PF 0.78  (loses to cost)
  Meta-filter on rule signals:   target WR 55-65%  expected PF 1.3-1.8

The ML problem is just "of the ~18K extreme-z bars, which 50% will revert
within 20 bars?" — a much sharper question than asking the model to
forecast direction blindly.

USAGE
-----
    python python/train_meta.py EURUSD --skip-extract --estimators 600
    python python/train_meta.py all    --skip-extract     # every symbol
    python python/train_meta.py EURUSD GBPUSD USDJPY \\
                                --z 2.5 --tp-atr 2.0 --sl-atr 1.0

OUTPUT
------
    Per symbol:
      n_candidates   how many bars the rule fired on
      n_train, n_val chronological 80/20 split
      val_acc        accuracy on the meta-label binary task
      base PF        rule alone, no ML
      meta PF        rule + meta filter at 0.55 confidence
      meta PF top10  rule + meta filter at top-10% confidence

Saves an ONNX next to the existing direction model — but this is the
META filter, so its semantic is "trade gate" not "direction predictor".
The EA needs a thin glue: fire rule, ask meta, take only if meta>0.55.

REQUIREMENTS
------------
  pip install xgboost onnxmltools  (already in requirements-train.txt)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FEATURE_DIM_DIR, VAL_SPLIT, LABEL_FORWARD_BARS, MIN_BAR_DATE,
    PARQUET_DIR, ONNX_OUTPUT_DIR, ALL_SYMBOLS,
    FOREX_SYMBOLS, METALS_SYMBOLS, INDICES_SYMBOLS, CE_SYMBOLS,
)

log = logging.getLogger(__name__)
EPS = 1e-12


# ---------------------------------------------------------------------------
# Symbol metadata
# ---------------------------------------------------------------------------

def pip_size(symbol: str) -> float:
    if "JPY" in symbol: return 0.01
    if symbol in ("GOLD", "SILVER", "PLATINUM"): return 0.01
    if symbol == "COPPER": return 0.001
    if symbol == "BTCUSD": return 1.0
    if symbol == "ETHUSD": return 0.1
    if symbol in ("CrudeOIL", "BRENT_OIL"): return 0.01
    if symbol == "NATURAL_GAS": return 0.001
    if symbol in ("US_500", "UK_100"): return 0.01
    return 0.0001


# ---------------------------------------------------------------------------
# Indicators (subset of backtest_meanreversion to keep this self-contained)
# ---------------------------------------------------------------------------

def _atr(high, low, close, period=14):
    n = len(close); tr = np.zeros(n); tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr = np.zeros(n)
    if period < n:
        atr[period] = tr[1:period + 1].mean()
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _zscore_returns(close, window=20):
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(np.maximum(close[1:], EPS) /
                         np.maximum(close[:-1], EPS))
    s = pd.Series(log_ret)
    mu = s.rolling(window, min_periods=window // 2).mean().to_numpy()
    sd = s.rolling(window, min_periods=window // 2).std().to_numpy()
    z = (log_ret - mu) / np.where(sd > EPS, sd, EPS)
    z[~np.isfinite(z)] = 0.0
    return z


def _vol_regime(close, forward=20, baseline=200):
    n = len(close)
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(np.maximum(close[1:], EPS) /
                         np.maximum(close[:-1], EPS))
    sq = log_ret * log_ret
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    rv = np.zeros(n)
    rv[:-forward] = np.sqrt(cs[forward+1:n+1] - cs[1:n-forward+1])
    bl = pd.Series(rv).rolling(baseline, min_periods=baseline // 2).mean().to_numpy()
    bl = np.where(np.isfinite(bl) & (bl > EPS), bl, EPS)
    ratio = rv / bl
    valid = np.zeros(n, dtype=bool); valid[baseline:n - forward] = True
    if not valid.any(): return np.full(n, -1, dtype=np.int8)
    q_lo, q_hi = np.quantile(ratio[valid], [0.33, 0.67])
    out = np.full(n, -1, dtype=np.int8)
    out[valid] = np.where(ratio[valid] < q_lo, 0,
                  np.where(ratio[valid] < q_hi, 1, 2)).astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# Candidate generation + meta-label
# ---------------------------------------------------------------------------

def generate_candidates(bars: pd.DataFrame, *,
                         z_thresh: float, sl_atr: float, tp_atr: float,
                         timeout: int, vol_min: int = 1
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk every bar, fire the z-fade rule, simulate the resulting trade.

    Returns:
        bar_idx     : ndarray of bar indices that fired the rule
        meta_label  : ndarray of {0,1} — 1 if the trade hit TP, 0 if SL or timeout

    The 'meta' label means: would this rule-fired trade have been a winner?
    The downstream ML model's job is to filter candidates by predicting
    this binary outcome from the model's 200-dim feature input.
    """
    close = bars["close"].to_numpy(dtype=np.float64)
    high  = bars["high"].to_numpy(dtype=np.float64)
    low   = bars["low"].to_numpy(dtype=np.float64)
    n = len(close)

    atr = _atr(high, low, close, 14)
    z   = _zscore_returns(close, 20)
    vr  = _vol_regime(close)

    bar_idx = []
    label   = []

    i = 0
    while i < n - timeout - 1:
        if atr[i] <= 0 or vr[i] < vol_min:
            i += 1; continue
        side = 0
        if   z[i] >  z_thresh: side = -1
        elif z[i] < -z_thresh: side = +1
        if side == 0:
            i += 1; continue

        entry  = close[i]
        sl_d   = sl_atr * atr[i]
        tp_d   = tp_atr * atr[i]
        sl_p   = entry - side * sl_d   # if side=+1 (long): sl_p = entry - sl_d
        tp_p   = entry + side * tp_d

        # Walk forward — TP/SL with intra-bar high/low.
        outcome = 0   # 0 = SL or timeout (loss); 1 = TP (win)
        consumed = timeout
        for j in range(1, timeout + 1):
            if i + j >= n: break
            h, l = high[i + j], low[i + j]
            if side == +1:
                if l <= sl_p: outcome = 0; consumed = j; break
                if h >= tp_p: outcome = 1; consumed = j; break
            else:
                if h >= sl_p: outcome = 0; consumed = j; break
                if l <= tp_p: outcome = 1; consumed = j; break

        bar_idx.append(i)
        label.append(outcome)
        i = i + consumed + 1   # don't open overlapping positions

    return np.asarray(bar_idx, dtype=np.int64), np.asarray(label, dtype=np.int8)


# ---------------------------------------------------------------------------
# Train one symbol end-to-end
# ---------------------------------------------------------------------------

def _load_bars(symbol: str) -> pd.DataFrame:
    cands = sorted(PARQUET_DIR.glob(f"HYDRA4_FEAT_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_size, reverse=True)
    if not cands:
        raise SystemExit(f"No cached parquet for {symbol} under {PARQUET_DIR}")
    df = pd.read_parquet(cands[0])
    if MIN_BAR_DATE and "time" in df.columns:
        cutoff = pd.Timestamp(MIN_BAR_DATE, tz="UTC")
        df = df[df["time"] >= cutoff].reset_index(drop=True)
    return df


def train_one(symbol: str, *,
              z_thresh: float, sl_atr: float, tp_atr: float,
              timeout: int,
              estimators: int = 500, max_depth: int = 6, lr: float = 0.05,
              seed: int = 42) -> dict:
    """
    End-to-end:
      load bars -> features -> rule candidates + meta-labels ->
      chronological 80/20 split -> XGB binary classifier -> evaluate ->
      compare base rule PF vs meta-filtered PF.
    """
    from feature_engine     import build_feature_dataframe
    from models.xgb_head    import XGBDirectionHead

    log.info("[%s] loading bars", symbol)
    bars = _load_bars(symbol)
    log.info("[%s] %d bars", symbol, len(bars))

    log.info("[%s] building features", symbol)
    ps = pip_size(symbol)
    dir_feat, _ = build_feature_dataframe(bars, symbol, pip_size=ps)

    log.info("[%s] generating candidates (z>=%.2f, RR=%.2f)",
             symbol, z_thresh, tp_atr / max(sl_atr, EPS))
    bar_idx, meta_y = generate_candidates(
        bars, z_thresh=z_thresh, sl_atr=sl_atr, tp_atr=tp_atr,
        timeout=timeout)
    n_cand = len(bar_idx)
    base_wr = float(meta_y.mean()) if n_cand else 0.0
    base_pf = (base_wr * tp_atr) / max((1 - base_wr) * sl_atr, EPS)
    log.info("[%s] candidates=%d  base_WR=%.3f  base_PF=%.3f  (rule alone)",
             symbol, n_cand, base_wr, base_pf)

    if n_cand < 500:
        log.warning("[%s] too few candidates (%d) — skipping", symbol, n_cand)
        return {"symbol": symbol, "n_cand": n_cand, "skipped": True}

    # Pull features at the candidate bars only.
    X = dir_feat[bar_idx].astype(np.float32)
    y = meta_y.astype(np.int8)

    # Chronological 80/20 split with a forward-bar gap to prevent leakage.
    n_val = max(100, int(n_cand * VAL_SPLIT))
    n_tr  = max(100, n_cand - n_val - timeout)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr + timeout:], y[n_tr + timeout:]
    log.info("[%s] split: train=%d  gap=%d  val=%d",
             symbol, n_tr, timeout, len(y_va))

    # Train XGBoost binary classifier.
    t0 = time.time()
    head = XGBDirectionHead.train(
        X_tr, y_tr, X_va, y_va,
        n_estimators=estimators, max_depth=max_depth, learning_rate=lr,
        seed=seed,
    )
    elapsed = time.time() - t0
    val_acc = head.score(X_va, y_va)
    log.info("[%s] meta XGB val_acc=%.4f  (in %.0fs)", symbol, val_acc, elapsed)

    # Evaluate the meta-filter against base rule PF.
    proba = head.predict_proba(X_va)
    # Meta-PF at threshold 0.55 (must say "yes, this trade will win"):
    for thresh in (0.50, 0.55, 0.60, 0.65):
        keep = proba >= thresh
        n_keep = int(keep.sum())
        if n_keep < 50:
            log.info("[%s]   thresh=%.2f  n_keep=%d  (too few — skip)",
                     symbol, thresh, n_keep)
            continue
        wr_keep = float(y_va[keep].mean())
        pf_keep = (wr_keep * tp_atr) / max((1 - wr_keep) * sl_atr, EPS)
        log.info("[%s]   thresh=%.2f  n_keep=%-5d  WR=%.3f  PF=%.3f",
                 symbol, thresh, n_keep, wr_keep, pf_keep)

    # Top decile.
    top10_thresh = float(np.quantile(proba, 0.90))
    keep = proba >= top10_thresh
    n_keep = int(keep.sum())
    wr_top = float(y_va[keep].mean()) if n_keep else 0.0
    pf_top = (wr_top * tp_atr) / max((1 - wr_top) * sl_atr, EPS)
    log.info("[%s]   top-10%% (thresh=%.3f)  n_keep=%d  WR=%.3f  PF=%.3f",
             symbol, top10_thresh, n_keep, wr_top, pf_top)

    # Top-K feature importance — the diagnostic we've wanted.
    log.info("[%s] top-15 feature importance (gain):", symbol)
    for fi, gain in head.feature_importance(top_k=15):
        log.info("    f%-3d  gain=%.4f", fi, gain)

    # Export ONNX. This file is a META filter, NOT a direction predictor —
    # the EA needs to know which interpretation to apply at inference.
    onnx_out = ONNX_OUTPUT_DIR / f"HYDRA4_META_{symbol}.onnx"
    head.export_onnx(onnx_out, n_features=FEATURE_DIM_DIR)

    meta = {
        "symbol": symbol, "model_class": "xgb_meta_filter",
        "purpose": "binary_trade_quality (1=will hit TP, 0=will hit SL or timeout)",
        "feat_dim": FEATURE_DIM_DIR,
        "rule": f"z>={z_thresh}, SL={sl_atr}xATR, TP={tp_atr}xATR, timeout={timeout}",
        "n_candidates": n_cand, "base_wr": base_wr, "base_pf": base_pf,
        "val_acc": float(val_acc),
    }
    p_meta = ONNX_OUTPUT_DIR / f"HYDRA4_META_{symbol}_meta.json"
    p_meta.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return {"symbol": symbol, "n_cand": n_cand, "val_acc": val_acc,
            "base_pf": base_pf}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbols", nargs="+",
                   help="Symbols, or 'all' / 'forex' / 'metals' / 'indices' / 'ce'")
    p.add_argument("--z",          type=float, default=2.5)
    p.add_argument("--sl-atr",     type=float, default=1.0)
    p.add_argument("--tp-atr",     type=float, default=2.0)
    p.add_argument("--timeout",    type=int,   default=20)
    p.add_argument("--estimators", type=int,   default=500)
    p.add_argument("--max-depth",  type=int,   default=6)
    p.add_argument("--lr",         type=float, default=0.05)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--skip-extract", action="store_true",
                   help="Compatibility shim — bars are loaded from cache regardless.")
    args = p.parse_args(argv)

    expand = {"all": ALL_SYMBOLS, "forex": FOREX_SYMBOLS,
              "metals": METALS_SYMBOLS, "indices": INDICES_SYMBOLS,
              "ce": CE_SYMBOLS}
    syms: list[str] = []
    for s in args.symbols:
        if s.lower() in expand: syms.extend(expand[s.lower()])
        else: syms.append(s)
    syms = list(dict.fromkeys(syms))

    print(f"\nMeta-labeling trainer — z>={args.z}  RR={args.tp_atr/args.sl_atr:.2f}\n")
    failed: list[str] = []
    for sym in syms:
        try:
            train_one(sym, z_thresh=args.z, sl_atr=args.sl_atr,
                      tp_atr=args.tp_atr, timeout=args.timeout,
                      estimators=args.estimators, max_depth=args.max_depth,
                      lr=args.lr, seed=args.seed)
        except Exception as e:
            log.exception("[%s] failed: %s", sym, e)
            failed.append(sym)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
