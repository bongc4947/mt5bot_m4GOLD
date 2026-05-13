"""
train_h5_scalp_gold.py — Trend-aligned intraday scalp on GOLD.

Hypothesis. The validated H4 ma_50_200 strategy on GOLD makes ~25-130
trades per year holding positions for days. Intraday GOLD pulls back
~0.3-0.8% inside the same trend then continues. An M5 entry that:

  1. Only opens trades in the direction of the H4 trend (filter)
  2. Buys pullbacks: when M5 low touches MA(20) - k*ATR(14) inside an
      uptrend (mirror for shorts)
  3. Exits on a fixed RR (TP = 1.5×ATR, SL = 0.7×ATR, timeout = 12 bars)

captures continuation moves WITHIN the H4 trend — much higher trade
count than H4 alone, leveraging GOLD's intraday volatility instead of
just its weekly drift.

Hypothesis is meaningful ONLY when the H4 trend has edge (it does for
GOLD: val Sharpe 1.21, WF 100% consistency). For symbols where H4
failed, this scalp would just trade noise.

Why GOLD only:
  - H4 deploy already validated -> trend filter is real, not regime luck
  - Tick data 2.6 yr -> 200K+ M5 bars, plenty for backtest
  - High intraday range -> spread cost is small fraction of typical move
  - ALL_SYMBOLS roster check at master driver will reject other symbols
    that try to use this — H5 is GOLD-specific by design.

Usage:
  python python/train_h5_scalp_gold.py
  python python/train_h5_scalp_gold.py --pullback-k 1.2 --tp-atr 2.0
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
    backward_atr, chronological_split, load_or_build_bars, max_drawdown,
    passive_pf, pip_size, profit_factor, sharpe, skill_gate,
    triple_barrier_outcome,
)

log = logging.getLogger(__name__)

SYMBOL = "GOLD"      # H5 is hard-bound to GOLD by hypothesis design


def _h4_trend_signal(h1_bars: pd.DataFrame, fast: int = 50,
                      slow: int = 200) -> np.ndarray:
    """
    State-based H4 trend signal computed on H1 closes. Returns int8 array
    aligned to h1_bars: +1 long, -1 short, 0 flat (warmup only).
    Causal: signal at bar i uses bars 0..i.
    """
    close = h1_bars["close"].to_numpy(dtype=np.float64)
    s = pd.Series(close)
    ma_fast = s.rolling(fast, min_periods=fast).mean().to_numpy()
    ma_slow = s.rolling(slow, min_periods=slow).mean().to_numpy()
    signal = np.zeros(len(close), dtype=np.int8)
    valid = np.isfinite(ma_fast) & np.isfinite(ma_slow)
    signal[valid & (ma_fast > ma_slow)] = +1
    signal[valid & (ma_fast < ma_slow)] = -1
    return signal


def _align_h1_to_m5(m5_times: pd.Series, h1_times: pd.Series,
                     h1_signal: np.ndarray) -> np.ndarray:
    """
    For each M5 bar's open time, find the most recent (strictly prior)
    H1 close and return the trend signal that was in force at THAT moment.
    Strictly causal — never peeks at the current H1 bar mid-formation.
    """
    m5_ns = pd.to_datetime(m5_times, utc=True).astype("int64").to_numpy()
    h1_ns = pd.to_datetime(h1_times, utc=True).astype("int64").to_numpy()
    # H1 bar at time T means "close of the H1 ending AT T".
    # An M5 bar at time u sees H1 close at T iff T <= u (and T is the
    # latest such). searchsorted-right gives index past the match.
    idx = np.searchsorted(h1_ns, m5_ns, side="right") - 1
    # Where no H1 bar has yet closed (idx < 0), force flat.
    out = np.zeros(len(m5_ns), dtype=np.int8)
    valid = idx >= 0
    out[valid] = h1_signal[idx[valid]]
    return out


def generate_scalp_candidates(m5_bars: pd.DataFrame, h4_signal: np.ndarray,
                                *, pullback_k: float = 1.0,
                                ma_window: int = 20,
                                sl_atr: float = 0.7,
                                tp_atr: float = 1.5,
                                timeout_bars: int = 12,
                                atr_period: int = 14) -> pd.DataFrame:
    """
    For each M5 bar in a non-flat H4 trend, fire a candidate iff:
      LONG  trend:  low  <= MA(20) - pullback_k * ATR(14)
                    AND close > MA(20) - pullback_k * ATR(14)  (didn't break)
      SHORT trend: high >= MA(20) + pullback_k * ATR(14)
                    AND close < MA(20) + pullback_k * ATR(14)

    Triple barrier outcome over `timeout_bars` M5 bars.
    """
    close = m5_bars["close"].to_numpy(dtype=np.float64)
    high  = m5_bars["high"].to_numpy(dtype=np.float64)
    low   = m5_bars["low"].to_numpy(dtype=np.float64)
    atr   = backward_atr(high, low, close, atr_period)
    ma    = pd.Series(close).rolling(ma_window, min_periods=ma_window).mean().to_numpy()
    n = len(close)

    cands = []
    last = n - timeout_bars - 1
    for i in range(max(ma_window, atr_period) + 1, last):
        trend = int(h4_signal[i])
        if trend == 0:           # warmup or flat
            continue
        if atr[i] <= 0 or not np.isfinite(ma[i]):
            continue
        threshold_long  = ma[i] - pullback_k * atr[i]
        threshold_short = ma[i] + pullback_k * atr[i]
        direction = 0
        if trend > 0 and low[i] <= threshold_long and close[i] > threshold_long:
            direction = +1
        elif trend < 0 and high[i] >= threshold_short and close[i] < threshold_short:
            direction = -1
        else:
            continue
        entry = close[i]
        if direction > 0:
            sl = entry - sl_atr * atr[i]
            tp = entry + tp_atr * atr[i]
        else:
            sl = entry + sl_atr * atr[i]
            tp = entry - tp_atr * atr[i]
        tp_hit, sl_hit, to_hit = triple_barrier_outcome(
            high, low, i, direction=direction, sl=sl, tp=tp,
            timeout=timeout_bars)
        if tp_hit:
            fwd_pnl = (tp - entry) * direction
        elif sl_hit:
            fwd_pnl = (sl - entry) * direction
        else:
            j = min(i + timeout_bars, n - 1)
            fwd_pnl = (close[j] - entry) * direction
        cands.append({
            "bar_idx":      i,
            "direction":    direction,
            "entry":        entry,
            "sl":           sl,
            "tp":           tp,
            "atr":          atr[i],
            "trend":        trend,
            "tp_hit":       tp_hit,
            "sl_hit":       sl_hit,
            "timeout":      to_hit,
            "fwd_pnl_price": fwd_pnl,
        })
    return pd.DataFrame(cands)


def _pnl_with_cost(cand: pd.DataFrame, m5_bars: pd.DataFrame) -> np.ndarray:
    """Per-candidate PnL net of 2 * mean-bar-spread cost."""
    pip = pip_size(SYMBOL)
    idx = cand["bar_idx"].to_numpy()
    if "spread" in m5_bars.columns:
        bar_spread = m5_bars["spread"].iloc[idx].to_numpy(dtype=np.float64)
        bar_spread = np.where(np.isfinite(bar_spread) & (bar_spread > 0),
                               bar_spread, pip)
    else:
        bar_spread = np.full(len(idx), pip)
    return cand["fwd_pnl_price"].to_numpy() - 2.0 * bar_spread


def train_one(symbol: str = SYMBOL, *,
               h4_fast: int = 50, h4_slow: int = 200,
               pullback_k: float = 1.0, ma_window: int = 20,
               sl_atr: float = 0.7, tp_atr: float = 1.5,
               timeout_bars: int = 12) -> dict:
    if symbol.upper() != SYMBOL:
        return {"strategy": "H5_SCALP", "symbol": symbol, "ok": False,
                "deploy": False,
                "reason": f"H5 is GOLD-only by design (got {symbol})"}
    from config import ONNX_OUTPUT_DIR
    # Need both timeframes
    m5_bars = load_or_build_bars(SYMBOL, "5min")
    h1_bars = load_or_build_bars(SYMBOL, "1h")
    if m5_bars is None or h1_bars is None:
        return {"strategy": "H5_SCALP", "symbol": symbol, "ok": False,
                "reason": "missing M5 or H1 bars"}
    if len(m5_bars) < 5000:
        return {"strategy": "H5_SCALP", "symbol": symbol, "ok": False,
                "reason": f"only {len(m5_bars)} M5 bars"}
    log.info("[H5:%s] %d M5 bars + %d H1 bars",
             SYMBOL, len(m5_bars), len(h1_bars))

    # H4 trend signal computed on H1, then projected onto M5 timeline
    h1_signal = _h4_trend_signal(h1_bars, h4_fast, h4_slow)
    h4_on_m5 = _align_h1_to_m5(m5_bars["time"], h1_bars["time"], h1_signal)
    pct_long = float((h4_on_m5 == +1).mean()) * 100
    pct_short = float((h4_on_m5 == -1).mean()) * 100
    log.info("[H5:%s] H4 trend on M5 timeline: long=%.0f%%  short=%.0f%%  flat=%.0f%%",
             SYMBOL, pct_long, pct_short, 100 - pct_long - pct_short)

    cand = generate_scalp_candidates(m5_bars, h4_on_m5,
                                       pullback_k=pullback_k, ma_window=ma_window,
                                       sl_atr=sl_atr, tp_atr=tp_atr,
                                       timeout_bars=timeout_bars)
    if len(cand) < 60:
        return {"strategy": "H5_SCALP", "symbol": symbol, "ok": False,
                "deploy": False,
                "reason": f"only {len(cand)} scalp candidates"}
    n_tp = int(cand["tp_hit"].sum())
    n_sl = int(cand["sl_hit"].sum())
    n_to = int(cand["timeout"].sum())
    log.info("[H5:%s] candidates=%d  TP=%d (%.0f%%)  SL=%d (%.0f%%)  TO=%d (%.0f%%)",
             SYMBOL, len(cand), n_tp, 100*n_tp/len(cand),
             n_sl, 100*n_sl/len(cand), n_to, 100*n_to/len(cand))

    pnl = _pnl_with_cost(cand, m5_bars)

    # Chronological split for honest val PF + walk-forward
    n_cand = len(cand)
    tr, va = chronological_split(n_cand, val_frac=0.30, gap=timeout_bars)
    pnl_va = pnl[va]
    val_pf = profit_factor(pnl_va)
    val_wr = float((pnl_va > 0).mean()) if len(pnl_va) else 0.0
    # Passive baseline = always-long the M5 stream
    m5_returns = np.diff(np.log(np.clip(m5_bars["close"].to_numpy(), 1e-12, None)),
                          prepend=0.0)
    p_long_pf, p_short_pf = passive_pf(m5_returns, cost_per_bar=2 * pip_size(SYMBOL))
    ok, excess = skill_gate(val_pf, p_long_pf, p_short_pf, len(pnl_va))
    log.info("[H5:%s] val:  N=%d  WR=%.3f  PF=%.3f  passive(L/S)=%.2f/%.2f  "
             "excess=%+.3f  -> %s",
             SYMBOL, len(pnl_va), val_wr, val_pf, p_long_pf, p_short_pf,
             excess, "PASS" if ok else "FAIL")

    # Walk-forward consistency: slice val into 4 windows
    edges = np.linspace(0, len(pnl_va), 5, dtype=int)
    sharpes, pfs = [], []
    for k in range(4):
        lo, hi = edges[k], edges[k + 1]
        win = pnl_va[lo:hi]
        if len(win) >= 5:
            sharpes.append(sharpe(win, periods_per_year=1.0))
            pfs.append(profit_factor(win))
        else:
            sharpes.append(0.0); pfs.append(0.0)
    consistency = sum(1 for s in sharpes if s > 0) / 4.0
    log.info("[H5:%s] walk-forward Sharpe per win: %s  consistency=%.2f",
             SYMBOL, [round(s, 3) for s in sharpes], consistency)
    deploy_unstable = ok and consistency < 0.50
    if deploy_unstable:
        log.warning("[H5:%s] passes single-split gate but WF consistency "
                     "is %.0f%% — marking deploy_unstable", SYMBOL, 100*consistency)
        ok = False

    spec = {
        "strategy":         "H5_SCALP",
        "symbol":           SYMBOL,
        "params": {
            "h4_fast":      h4_fast,
            "h4_slow":      h4_slow,
            "ma_window":    ma_window,
            "pullback_k":   pullback_k,
            "sl_atr":       sl_atr,
            "tp_atr":       tp_atr,
            "timeout_bars": timeout_bars,
        },
        "n_candidates":     int(n_cand),
        "val_n":            int(len(pnl_va)),
        "val_pf":           float(val_pf),
        "val_wr":           float(val_wr),
        "excess_vs_passive": float(excess),
        "wf_sharpe_per_win": [round(s, 3) for s in sharpes],
        "wf_consistency":    float(consistency),
        "deploy":            bool(ok),
        "deploy_unstable":   bool(deploy_unstable),
    }
    out = ONNX_OUTPUT_DIR / f"HYDRA4_H5SCALP_{SYMBOL}_spec.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2))
    log.info("[H5:%s] FINAL  val_PF=%.3f  WF=%.2f  -> %s",
             SYMBOL, val_pf, consistency,
             "DEPLOY" if ok else ("UNSTABLE" if deploy_unstable else "BLOCKED"))
    return spec


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pullback-k", type=float, default=1.0,
                   help="entry trigger: low <= MA - k*ATR (long), inverse for short")
    p.add_argument("--ma-window",  type=int,   default=20)
    p.add_argument("--sl-atr",     type=float, default=0.7)
    p.add_argument("--tp-atr",     type=float, default=1.5)
    p.add_argument("--timeout-bars", type=int, default=12)
    p.add_argument("--h4-fast",    type=int,   default=50)
    p.add_argument("--h4-slow",    type=int,   default=200)
    args = p.parse_args(argv)
    t0 = time.time()
    train_one(SYMBOL,
              h4_fast=args.h4_fast, h4_slow=args.h4_slow,
              pullback_k=args.pullback_k, ma_window=args.ma_window,
              sl_atr=args.sl_atr, tp_atr=args.tp_atr,
              timeout_bars=args.timeout_bars)
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
