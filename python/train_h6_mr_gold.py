"""
train_h6_mr_gold.py — GOLD-only intraday mean-reversion (H1, z-score).

Replaces the mk4 GOLD/SILVER cointegration hedge with a single-symbol
strategy that fits the m4Gold mandate: no second leg, no broker silver
dependency, just GOLD's own deviation from its rolling mean.

Hypothesis. Within a single H1 session of GOLD, log-price excursions
beyond ±2σ from a 200-bar rolling mean tend to revert within hours.
This is a classical Ornstein-Uhlenbeck assumption (Lo–MacKinlay 1988);
it holds intra-day for spot metals when no fundamental catalyst flips
the regime. The strategy gives up the hedge that the silver leg
provided — it is directionally exposed to GOLD between entry and exit.

Pipeline:
  1. Load H1 GOLD bars.
  2. Compute rolling mean / std of log close over `z_window` past bars.
  3. Entry: z >=  z_in  -> SHORT (overbought)
            z <= -z_in  -> LONG  (oversold)
  4. Exit:  |z| <= z_out (revert) OR |z| >= z_stop (regime break)
            OR timeout_bars elapsed.
  5. Cost: spread crossings at entry + exit.
  6. Skill gate: PF >= 1.20, excess vs always-long-of-z-cross >= +0.10,
     plus 4-window walk-forward consistency >= 50%.

Usage:
    python python/train_h6_mr_gold.py
    python python/train_h6_mr_gold.py --z-window 200 --z-in 2.0 --z-stop 3.5
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
    chronological_split, load_or_build_bars, pip_size,
    profit_factor, sharpe, skill_gate,
)

log = logging.getLogger(__name__)

SYMBOL = "GOLD"


def generate_mr_signals(bars: pd.DataFrame, *,
                         z_window: int = 200,
                         z_in: float = 2.0,
                         z_out: float = 0.5,
                         z_stop: float = 3.5,
                         timeout_bars: int = 48) -> pd.DataFrame:
    close = bars["close"].to_numpy(dtype=np.float64)
    spread = bars["spread"].to_numpy(dtype=np.float64) if "spread" in bars.columns \
             else np.full(len(bars), pip_size(SYMBOL))
    log_p = np.log(np.clip(close, 1e-12, None))
    n = len(log_p)
    mean = pd.Series(log_p).rolling(z_window, min_periods=z_window).mean().to_numpy()
    sd = pd.Series(log_p).rolling(z_window, min_periods=z_window).std().to_numpy()
    sd = np.where(sd > 1e-12, sd, np.nan)
    z = (log_p - mean) / sd

    trades = []
    open_idx = -1
    open_dir = 0
    open_price = 0.0
    open_spread = 0.0
    for i in range(z_window, n - 1):
        if not np.isfinite(z[i]):
            continue
        if open_idx < 0:
            if z[i] >= z_in:
                open_dir = -1
            elif z[i] <= -z_in:
                open_dir = +1
            else:
                continue
            open_idx = i
            open_price = close[i]
            open_spread = spread[i] if np.isfinite(spread[i]) else pip_size(SYMBOL)
        else:
            bars_open = i - open_idx
            exit_reason = None
            if abs(z[i]) <= z_out:
                exit_reason = "mean_revert"
            elif abs(z[i]) >= z_stop:
                exit_reason = "stop"
            elif bars_open >= timeout_bars:
                exit_reason = "timeout"
            if exit_reason is None:
                continue
            move = (close[i] - open_price) * open_dir
            sp_out = spread[i] if np.isfinite(spread[i]) else pip_size(SYMBOL)
            cost = open_spread + sp_out
            net_pnl = move - cost
            trades.append({
                "entry_idx": open_idx, "exit_idx": i, "bars_open": bars_open,
                "direction": open_dir,
                "entry_z": float(z[open_idx]), "exit_z": float(z[i]),
                "gold_pnl_price": float(move),
                "cost_price":     float(cost),
                "net_pnl_price":  float(net_pnl),
                "exit_reason":    exit_reason,
            })
            open_idx = -1
            open_dir = 0
    return pd.DataFrame(trades)


def train(*, z_window: int = 200, z_in: float = 2.0,
           z_out: float = 0.5, z_stop: float = 3.5,
           timeout_bars: int = 48) -> dict:
    from config import ONNX_OUTPUT_DIR

    bars = load_or_build_bars(SYMBOL, "1h")
    if bars is None or len(bars) < 2000:
        return {"strategy": "H6_MR", "symbol": SYMBOL,
                "ok": False, "deploy": False,
                "reason": f"insufficient bars: {0 if bars is None else len(bars)}"}
    log.info("[H6] %s H1 bars: %d  span=%s -> %s",
             SYMBOL, len(bars), bars["time"].iloc[0], bars["time"].iloc[-1])

    trades = generate_mr_signals(bars, z_window=z_window, z_in=z_in,
                                  z_out=z_out, z_stop=z_stop,
                                  timeout_bars=timeout_bars)
    if len(trades) < 30:
        return {"strategy": "H6_MR", "symbol": SYMBOL,
                "ok": False, "deploy": False,
                "reason": f"only {len(trades)} trades — try z_in lower"}
    n_total = len(trades)
    n_revert = int((trades["exit_reason"] == "mean_revert").sum())
    n_stop = int((trades["exit_reason"] == "stop").sum())
    n_to = int((trades["exit_reason"] == "timeout").sum())
    log.info("[H6] %d trades  revert=%d (%.0f%%)  stop=%d (%.0f%%)  to=%d (%.0f%%)",
             n_total,
             n_revert, 100 * n_revert / n_total,
             n_stop, 100 * n_stop / n_total,
             n_to, 100 * n_to / n_total)

    pnl = trades["net_pnl_price"].to_numpy()
    tr_idx, va_idx = chronological_split(n_total, val_frac=0.30, gap=0)
    pnl_va = pnl[va_idx]
    val_pf = profit_factor(pnl_va)
    val_wr = float((pnl_va > 0).mean()) if len(pnl_va) else 0.0
    p_pnl = np.where(trades["direction"] == 1,
                     pnl,
                     -pnl - 2 * trades["cost_price"].to_numpy())
    passive_long_pf = profit_factor(p_pnl[va_idx])
    passive_short_pf = profit_factor(-p_pnl[va_idx])
    ok, excess = skill_gate(val_pf, passive_long_pf, passive_short_pf, len(pnl_va))
    log.info("[H6] val:  N=%d  WR=%.3f  PF=%.3f  passive(L/S)=%.2f/%.2f  "
             "excess=%+.3f  -> %s",
             len(pnl_va), val_wr, val_pf, passive_long_pf, passive_short_pf,
             excess, "PASS" if ok else "FAIL")

    edges = np.linspace(0, len(pnl_va), 5, dtype=int)
    sharpes = []
    for k in range(4):
        lo, hi = edges[k], edges[k + 1]
        w = pnl_va[lo:hi]
        sharpes.append(sharpe(w, periods_per_year=1.0) if len(w) >= 5 else 0.0)
    consistency = sum(1 for s in sharpes if s > 0) / 4.0
    log.info("[H6] walk-forward Sharpe per win: %s  consistency=%.2f",
             [round(s, 3) for s in sharpes], consistency)
    deploy_unstable = ok and consistency < 0.50
    if deploy_unstable:
        log.warning("[H6] passes single-split but WF=%.0f%% — deploy_unstable",
                    100 * consistency)
        ok = False

    spec = {
        "strategy": "H6_MR",
        "symbol":   SYMBOL,
        "params": {
            "z_window":     z_window,
            "z_in":         z_in,
            "z_out":        z_out,
            "z_stop":       z_stop,
            "timeout_bars": timeout_bars,
        },
        "n_trades_total":    int(n_total),
        "n_revert":          int(n_revert),
        "n_stop":            int(n_stop),
        "n_timeout":         int(n_to),
        "val_n":             int(len(pnl_va)),
        "val_pf":            float(val_pf),
        "val_wr":            float(val_wr),
        "passive_long_pf":   float(passive_long_pf),
        "passive_short_pf":  float(passive_short_pf),
        "excess_vs_passive": float(excess),
        "wf_sharpe_per_win": [round(s, 3) for s in sharpes],
        "wf_consistency":    float(consistency),
        "deploy":            bool(ok),
        "deploy_unstable":   bool(deploy_unstable),
    }
    out = ONNX_OUTPUT_DIR / f"M4GOLD_H6MR_{SYMBOL}_spec.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2))
    log.info("[H6] FINAL  val_PF=%.3f  WF=%.2f  -> %s",
             val_pf, consistency,
             "DEPLOY" if ok else ("UNSTABLE" if deploy_unstable else "BLOCKED"))
    return spec


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Orchestrator passes SYMBOL positionally for uniform CLI; H6 is GOLD-only
    # so the arg is accepted-and-validated rather than acted on.
    p.add_argument("symbol", nargs="?", default=SYMBOL,
                   help=f"symbol (default {SYMBOL}; H6 is GOLD-only)")
    p.add_argument("--z-window",     type=int,   default=200)
    p.add_argument("--z-in",         type=float, default=2.0)
    p.add_argument("--z-out",        type=float, default=0.5)
    p.add_argument("--z-stop",       type=float, default=3.5)
    p.add_argument("--timeout-bars", type=int,   default=48)
    args = p.parse_args(argv)
    if args.symbol.upper() != SYMBOL:
        log.warning("H6 is GOLD-only; got %r — ignoring and training GOLD",
                    args.symbol)
    t0 = time.time()
    train(z_window=args.z_window, z_in=args.z_in, z_out=args.z_out,
          z_stop=args.z_stop, timeout_bars=args.timeout_bars)
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
