"""
train_h4_trend.py — H4: long-horizon trend-following on H1 / H4 bars.

Hypothesis: long-horizon momentum on commodities and FX majors is the most
robust anomaly in finance (Asness-Moskowitz-Pedersen, "Value & Momentum
Everywhere"). Simple MA(50) > MA(200) crossover or 12-period look-back
momentum on H1+ bars has positive expectancy across most regimes. Low Sharpe
by design, but the lowest-frequency premium = friendliest to retail costs.

Pipeline (rule-only — no ML required if Sharpe gate passes):
  1. Build H1 bars from raw ticks (cached).
  2. Two parallel rules:
       MA-cross: state-based long when MA(fast) > MA(slow), flat otherwise.
       MOM:      state-based long when close > close[lookback] (momentum sign).
     For each rule, optionally allow shorts (symmetric).
  3. Walk-forward evaluation:
       train window:  first 70%
       val   window:  last 30%
     Compute net annualised Sharpe + MaxDD on val window.
  4. Skill gate:
       val_Sharpe >= 0.6 AND val_MaxDD <= 0.30 AND val_PF >= 1.10
     If both rules clear the gate, pick the higher val_Sharpe.
  5. No ONNX needed — rule is deterministic and small enough to embed in
     the EA directly. Emit HYDRA4_H4TREND_<SYM>_spec.json with the chosen
     parameters; the MQL5 TrendRule.mqh reads them at attach time.

Usage:
    python python/train_h4_trend.py EURUSD GBPUSD
    python python/train_h4_trend.py --all --timeframe 1h --fast 50 --slow 200
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
    backward_vol_regime_ratio, chronological_split, discover_tick_symbols,
    load_or_build_bars, max_drawdown, pip_size, profit_factor, sharpe,
)

log = logging.getLogger(__name__)

PERIODS_PER_YEAR = {"1h": 252 * 24, "4h": 252 * 6, "1d": 252}


def _ma_cross_positions(close: np.ndarray, fast: int, slow: int,
                         allow_short: bool = True) -> np.ndarray:
    """Returns position vector in {-1, 0, +1} aligned to close.
    Backward-causal: position at i depends only on close[0..i]."""
    s = pd.Series(close)
    ma_fast = s.rolling(fast, min_periods=fast).mean().to_numpy()
    ma_slow = s.rolling(slow, min_periods=slow).mean().to_numpy()
    pos = np.zeros(len(close), dtype=np.float64)
    long_mask = ma_fast > ma_slow
    short_mask = ma_fast < ma_slow
    pos[long_mask] = 1.0
    if allow_short:
        pos[short_mask] = -1.0
    # Trade decision at bar i is applied to bar i+1's return (no look-ahead)
    pos = np.concatenate([[0.0], pos[:-1]])
    return pos


def _momentum_positions(close: np.ndarray, lookback: int,
                         allow_short: bool = True) -> np.ndarray:
    s = pd.Series(close)
    shifted = s.shift(lookback).to_numpy()
    pos = np.zeros(len(close), dtype=np.float64)
    pos[close > shifted] = 1.0
    if allow_short:
        pos[close < shifted] = -1.0
    pos = np.concatenate([[0.0], pos[:-1]])
    return pos


def _backtest_positions(close: np.ndarray, positions: np.ndarray,
                         cost_per_turn: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (per-bar net returns, per-bar position changes).
    cost_per_turn is in price units (round-trip spread)."""
    log_ret = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=0.0)
    pos_change = np.abs(np.diff(positions, prepend=0.0))
    # Half-spread per side, full spread per flip — pos_change counts each
    # transition once, so cost_per_turn IS one round-trip charged on entries.
    # When flipping long->short pos_change=2 -> double-cost.
    cost = pos_change * cost_per_turn
    bar_ret = positions * log_ret - cost
    return bar_ret, pos_change


def _eval_strategy(close: np.ndarray, positions: np.ndarray, *,
                    cost_per_turn: float, period: str) -> dict:
    bar_ret, pos_change = _backtest_positions(close, positions, cost_per_turn)
    eq = np.cumsum(bar_ret)
    n_trades = int((pos_change > 0).sum())
    return {
        "n_trades":   n_trades,
        "n_bars":     int(len(bar_ret)),
        "total_ret":  float(eq[-1]) if len(eq) else 0.0,
        "ann_ret":    float(np.mean(bar_ret) * PERIODS_PER_YEAR[period]),
        "sharpe":     sharpe(bar_ret, PERIODS_PER_YEAR[period]),
        "mdd":        max_drawdown(bar_ret),
        "pf":         profit_factor(bar_ret),
        "wr_bars":    float((bar_ret > 0).mean()) if len(bar_ret) else 0.0,
    }


def walk_forward_robustness(close: np.ndarray, positions: np.ndarray, *,
                              cost_per_turn: float, period: str,
                              n_windows: int = 4) -> dict:
    """
    Slice the val series into K consecutive equal-length windows and run
    the same strategy on each. Returns a stability score = fraction of
    windows where Sharpe > 0. A genuinely robust trend strategy should
    work in most regimes; a regime-luck artifact will show wildly varying
    per-window Sharpes.

    Required reads BEFORE deploy:
      * `consistency`     — fraction of windows with Sharpe > 0
      * `sharpe_per_win`  — list of per-window Sharpes
      * `min_sharpe`      — worst window's Sharpe (downside scenario)
      * `mean_sharpe`     — across windows (closer to a true expected Sharpe)
    """
    n = len(close)
    if n < n_windows * 50:
        return {"n_windows": 0, "consistency": float("nan"),
                "sharpe_per_win": [], "min_sharpe": float("nan"),
                "mean_sharpe": float("nan"),
                "note": "too few bars for walk-forward"}
    edges = np.linspace(0, n, n_windows + 1, dtype=int)
    sharpes, pfs, mdds = [], [], []
    for k in range(n_windows):
        lo, hi = edges[k], edges[k + 1]
        win_close = close[lo:hi]
        win_pos   = positions[lo:hi]
        win = _eval_strategy(win_close, win_pos, cost_per_turn=cost_per_turn,
                              period=period)
        sharpes.append(win["sharpe"])
        pfs.append(win["pf"])
        mdds.append(win["mdd"])
    pos_windows = sum(1 for s in sharpes if s > 0)
    return {
        "n_windows":      n_windows,
        "consistency":    pos_windows / n_windows,
        "sharpe_per_win": [round(s, 3) for s in sharpes],
        "pf_per_win":     [round(p, 3) for p in pfs],
        "mdd_per_win":    [round(m, 3) for m in mdds],
        "min_sharpe":     float(min(sharpes)),
        "mean_sharpe":    float(np.mean(sharpes)),
        "robust":         pos_windows >= (n_windows + 1) // 2,   # majority
    }


def train_one_symbol(symbol: str, *,
                      timeframe: str = "1h",
                      fast: int = 50, slow: int = 200,
                      mom_lookback: int = 240,    # legacy single-lookback arg
                      mom_lookbacks: list[int] | None = None,
                      allow_short: bool = True,
                      vol_filter_ratio: float = 0.0,
                      vol_filter_short: int = 20,
                      vol_filter_long: int = 500,
                      seed: int = 42) -> dict:
    """
    mom_lookbacks (mk4.8.3): list of momentum lookback windows to test in
    parallel. Defaults to [120, 240] which covers ~5-day and ~10-day
    horizons on H1 bars. Each lookback becomes its own rule in the
    summary (`mom_120`, `mom_240` etc.) and goes through the same
    skill gate + walk-forward check. Wider rule coverage catches symbols
    where a different trend horizon is needed (e.g., faster-moving
    crypto may need mom_60, slower commodities mom_480).
    """
    if mom_lookbacks is None:
        # Honour the legacy single-lookback param if no list is supplied.
        mom_lookbacks = [mom_lookback]
    bars = load_or_build_bars(symbol, timeframe)
    if bars is None:
        return {"strategy": "H4_TREND", "symbol": symbol, "ok": False,
                "reason": "no bars"}
    n = len(bars)
    if n < max(slow, mom_lookback) * 3:
        return {"strategy": "H4_TREND", "symbol": symbol, "ok": False,
                "reason": f"only {n} bars at {timeframe}"}
    log.info("[H4:%s] %d %s bars  span=%s -> %s",
             symbol, n, timeframe, bars["time"].iloc[0], bars["time"].iloc[-1])
    close = bars["close"].to_numpy(dtype=np.float64)
    spread = bars.get("spread", pd.Series(np.zeros(n))).to_numpy(dtype=np.float64)
    # cost_per_turn in log-return units: ~spread / mid * direction
    cost_logret = float(np.nanmean(spread / np.where(close > 0, close, 1.0)))
    if not np.isfinite(cost_logret) or cost_logret <= 0:
        cost_logret = pip_size(symbol) / float(np.nanmedian(close))
    log.info("[H4:%s] mean cost per turn = %.6f (%.1f bps)",
             symbol, cost_logret, cost_logret * 1e4)

    # Build the rule family. One MA-cross + N momentum lookbacks
    # (mk4.8.3 expanded the momentum family from a single 240 to a list).
    rules = {f"ma_{fast}_{slow}":
                _ma_cross_positions(close, fast, slow, allow_short)}
    for lb in mom_lookbacks:
        if lb <= 0 or lb >= n:
            continue
        rules[f"mom_{lb}"] = _momentum_positions(close, lb, allow_short)

    # mk4.8.6: vol-regime filter. Zero out positions during bars where
    # current ATR is materially below its trailing baseline ('chop'
    # regime). Motivation: the fresh-data GOLD H4 run produced WF
    # sub-windows [-2.25, +2.29, +2.29, +2.30] — one catastrophic chop
    # sub-window in late 2025 dragged the aggregate val Sharpe to -1.10
    # despite the other 3 quarters working well. The filter mechanically
    # eliminates that class of period.
    high = bars["high"].to_numpy(dtype=np.float64)
    low  = bars["low"].to_numpy(dtype=np.float64)
    vol_filter_active = vol_filter_ratio > 0.0
    if vol_filter_active:
        vol_ratio = backward_vol_regime_ratio(
            close, high, low,
            short_window=vol_filter_short,
            long_window=vol_filter_long,
        )
        regime_mask = (vol_ratio >= vol_filter_ratio).astype(np.float64)
        # Don't zero out the warmup region — that's NaN territory anyway.
        valid_filter = np.isfinite(vol_ratio)
        regime_mask = np.where(valid_filter, regime_mask, 1.0)
        masked_pct = float(100 * (regime_mask == 0).mean())
        log.info("[H4:%s] vol filter active  short=%d  long=%d  ratio>=%.2f  "
                 "-> %.0f%% of bars masked OUT",
                 symbol, vol_filter_short, vol_filter_long, vol_filter_ratio,
                 masked_pct)
        for name in list(rules.keys()):
            rules[name] = rules[name] * regime_mask

    # Walk-forward: train metrics use bars 0 .. 0.7n, val uses last 0.3n
    tr, va = chronological_split(n, val_frac=0.30, gap=slow)
    summary = {}
    for name, pos in rules.items():
        tr_metrics = _eval_strategy(close[tr], pos[tr],
                                     cost_per_turn=cost_logret, period=timeframe)
        va_metrics = _eval_strategy(close[va], pos[va],
                                     cost_per_turn=cost_logret, period=timeframe)
        gate_ok = (va_metrics["sharpe"] >= 0.6 and
                   va_metrics["mdd"]    <= 0.30 and
                   va_metrics["pf"]     >= 1.10 and
                   va_metrics["n_trades"] >= 20)
        summary[name] = {"train": tr_metrics, "val": va_metrics, "deploy": gate_ok}
        log.info("[H4:%s] %s  TRAIN Sharpe=%.2f PF=%.2f MDD=%.1f%% N=%d  "
                 "| VAL Sharpe=%.2f PF=%.2f MDD=%.1f%% N=%d  -> %s",
                 symbol, name,
                 tr_metrics["sharpe"], tr_metrics["pf"], 100*tr_metrics["mdd"],
                 tr_metrics["n_trades"],
                 va_metrics["sharpe"], va_metrics["pf"], 100*va_metrics["mdd"],
                 va_metrics["n_trades"],
                 "DEPLOY" if gate_ok else "skip")

    deployable = {k: v for k, v in summary.items() if v["deploy"]}
    if deployable:
        best_rule = max(deployable, key=lambda k: deployable[k]["val"]["sharpe"])
        deploy = True
    else:
        best_rule = max(summary, key=lambda k: summary[k]["val"]["sharpe"])
        deploy = False

    # Walk-forward robustness check on the chosen rule. Splits the VAL
    # window into 4 consecutive sub-windows and re-runs. A "regime-lucky"
    # deploy will have wildly varying per-window Sharpes (e.g. +2 in one,
    # -1 in another). A genuine edge clears Sharpe>0 in majority of windows.
    # This catches the train≈0 / val>1 pattern that the 15:30 UTC Kaggle
    # run produced — train was a chop regime, val was a trend regime, so
    # naive train/val confirmed an edge that may not generalise.
    wf = walk_forward_robustness(
        close[va], rules[best_rule][va],
        cost_per_turn=cost_logret, period=timeframe, n_windows=4,
    )
    log.info("[H4:%s] walk-forward (4 sub-windows of val): "
             "Sharpe_per_win=%s  consistency=%.2f  min=%.2f  robust=%s",
             symbol, wf.get("sharpe_per_win"),
             wf.get("consistency", float("nan")),
             wf.get("min_sharpe", float("nan")),
             wf.get("robust"))
    # Tighten the deploy decision: a cell only ships if it both passes the
    # original gate AND has WF consistency >= 50%. A pass-but-fragile cell
    # is downgraded to "deploy_unstable" — visible in the spec but not
    # auto-deployed by the EA dispatcher.
    deploy_unstable = deploy and not wf.get("robust", False)
    if deploy_unstable:
        log.warning("[H4:%s] cell passes single-split gate but FAILS "
                     "walk-forward (only %d/%d windows positive). Marking "
                     "deploy_unstable; do not put live capital on this without "
                     "extra evidence.", symbol,
                     int(wf.get("consistency", 0) * wf.get("n_windows", 0)),
                     wf.get("n_windows", 0))
        deploy = False

    from config import ONNX_OUTPUT_DIR
    spec = {
        "strategy":   "H4_TREND",
        "symbol":     symbol,
        "timeframe":  timeframe,
        "best_rule":  best_rule,
        "rule_kind":  "ma_cross" if best_rule.startswith("ma_") else "momentum",
        "params":     ({"fast": fast, "slow": slow}
                       if best_rule.startswith("ma_") else
                       # mk4.8.3: extract the lookback from the rule name
                       # (e.g. "mom_120" -> 120) so multiple momentum
                       # rules can coexist and each writes its own spec.
                       {"lookback": int(best_rule.split("_")[1])}),
        "allow_short": bool(allow_short),
        "cost_logret_per_turn": float(cost_logret),
        # mk4.8.6: vol-regime filter params (LIVE EA must mirror these)
        "vol_filter": {
            "active":   bool(vol_filter_active),
            "ratio":    float(vol_filter_ratio),
            "short":    int(vol_filter_short),
            "long":     int(vol_filter_long),
        },
        "deploy":           bool(deploy),
        "deploy_unstable":  bool(deploy_unstable),
        "walk_forward":     wf,
        "summary":          summary,
    }
    out_path = ONNX_OUTPUT_DIR / f"HYDRA4_H4TREND_{symbol}_spec.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2, default=float))
    log.info("[H4:%s] FINAL  best_rule=%s  val_Sharpe=%.2f  val_MDD=%.1f%%  "
             "WF=%.2f  -> %s",
             symbol, best_rule, summary[best_rule]["val"]["sharpe"],
             100 * summary[best_rule]["val"]["mdd"],
             wf.get("consistency", float("nan")),
             "DEPLOY" if deploy else
             ("UNSTABLE" if deploy_unstable else "BLOCKED"))
    return spec


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbols", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--timeframe", choices=("1h", "4h", "1d"), default="1h")
    p.add_argument("--fast", type=int, default=50)
    p.add_argument("--mom-lookbacks", type=str, default="120,240",
                   help="comma-separated momentum lookbacks (default 120,240). "
                        "Each becomes a separate mom_<lb> rule.")
    p.add_argument("--slow", type=int, default=200)
    p.add_argument("--mom-lookback", type=int, default=240)
    p.add_argument("--no-short", action="store_true",
                   help="long-only — skip the short side of every rule")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vol-filter-ratio", type=float, default=0.0,
                   help="vol-regime filter threshold. 0.0=off (default); try "
                        "0.85-0.95 to skip trend trades during chop regimes "
                        "(ATR(short)/baseline(long) < ratio = masked out)")
    p.add_argument("--vol-filter-short", type=int, default=20)
    p.add_argument("--vol-filter-long",  type=int, default=500)
    args = p.parse_args(argv)
    symbols = args.symbols or discover_tick_symbols()
    if not symbols:
        log.error("no symbols / tick parquets available")
        return 2
    # Parse --mom-lookbacks "120,240" -> [120, 240]
    mom_lookbacks = [int(s) for s in args.mom_lookbacks.split(",") if s.strip()]

    t0 = time.time()
    rows = []
    for sym in symbols:
        try:
            rows.append(train_one_symbol(sym,
                                          timeframe=args.timeframe,
                                          fast=args.fast, slow=args.slow,
                                          mom_lookbacks=mom_lookbacks,
                                          allow_short=not args.no_short,
                                          vol_filter_ratio=args.vol_filter_ratio,
                                          vol_filter_short=args.vol_filter_short,
                                          vol_filter_long=args.vol_filter_long,
                                          seed=args.seed))
        except Exception as e:
            log.exception("[H4:%s] failed", sym)
            rows.append({"strategy": "H4_TREND", "symbol": sym, "ok": False,
                          "reason": str(e)})
    print(f"\n{'='*82}")
    print(f"  H4 / trend-following  tf={args.timeframe}  ({time.time() - t0:.0f}s)")
    print(f"{'='*82}")
    print(f"  {'Symbol':<12}  {'best':<14}  {'valSharpe':>9}  {'valPF':>6}  "
          f"{'valMDD':>7}  {'N':>5}  Deploy")
    print(f"  {'-'*82}")
    for r in rows:
        sym = r.get("symbol", "?")
        if "best_rule" not in r:
            print(f"  {sym:<12}  FAIL ({r.get('reason','?')})"); continue
        best = r["summary"][r["best_rule"]]["val"]
        print(f"  {sym:<12}  {r['best_rule']:<14}  "
              f"{best['sharpe']:>9.2f}  {best['pf']:>6.2f}  "
              f"{100*best['mdd']:>6.1f}%  {best['n_trades']:>5d}  "
              f"{'YES' if r['deploy'] else 'no'}")
    print(f"{'='*82}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
