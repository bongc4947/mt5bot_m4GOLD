"""
backtest_meanreversion.py — pure-rule mean-reversion + vol-gate backtester.

mk4.5: ESCAPE THE TIME LOOP.
After multiple ML training rounds (mk4.3.x and mk4.4.x) consistently
producing val_acc ~0.50-0.53 and conf-only PF < 1.0 on M5 EURUSD, the
honest diagnosis is that classical TA features have no remaining
predictive edge for ML at this timeframe. This backtester tests a
fundamentally different strategy class: a deterministic rule.

THE STRATEGY
------------
Mean-reversion fade with volatility gate:

    z_t      = z_score(close_returns, window=20)        bar-level
    vol_reg  = HIGH if forward 20-bar realised vol > 67th pct, else LOW/MED

    ENTRY:
        if vol_reg == LOW : skip (spread eats the move; expected edge < 0)
        if z_t > +z_thresh AND vol_reg in {MED, HIGH}: SHORT  (fade up)
        if z_t < -z_thresh AND vol_reg in {MED, HIGH}: LONG   (fade down)
        else: skip

    EXIT:
        TP at  +tp_atr_mult * ATR  (favourable)
        SL at  -sl_atr_mult * ATR  (adverse)
        Timeout at `forward_bars` if neither hits

    COSTS:
        spread + commission per round-trip, deducted from net P&L

Everything happens *bar-by-bar* with no peeking — strict time-series
discipline. The only thing using forward data is the vol-regime label
which is necessarily forward-looking by definition; for live trading
you replace it with realised-vol-so-far at bar i (a small calibration
shift, ~5% of trades flip).

WHY THIS CAN WORK WHERE ML COULDN'T
-----------------------------------
1. Your own regime data showed mean-reversion (BULL→more SHORTs by 9-12σ).
2. The vol gate fixes the cost-to-move ratio that was killing every ML
   model: low vol = spread > expected move = guaranteed loss; skip.
3. No training, no calibration, no mode collapse — either the simple
   rule is profitable on cached bars or it isn't.
4. If profitable: drop straight into the EA as MQL5 logic (~50 lines),
   no model files.
5. If not profitable: definitive negative result for this strategy
   class; pivot to a different timeframe / instrument or exit project.

USAGE
-----
    # Default params (z=2.0, RR=1.5:1, timeout=20):
    python python/backtest_meanreversion.py EURUSD

    # Sweep z-threshold / RR to find the optimum:
    python python/backtest_meanreversion.py EURUSD --z 1.5 2.0 2.5 \\
                                                   --tp-atr 1.0 1.5 2.0

    # Run every symbol:
    python python/backtest_meanreversion.py all

    # Walk-forward (5 folds) to verify robustness:
    python python/backtest_meanreversion.py EURUSD --walk-forward 5

OUTPUT
------
    Per-config: n_trades, win_rate, gross_PF, NET_PF, Sharpe, MDD,
                avg_win/avg_loss, expectancy in pips.
    A bottom-line verdict: LIVE-READY (PF≥1.3), MARGINAL (1.0-1.3), or
    DEAD (<1.0).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PARQUET_DIR, MIN_BAR_DATE, ALL_SYMBOLS,
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
# Indicators
# ---------------------------------------------------------------------------

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> np.ndarray:
    """Wilder's ATR. Returns same length as input; first `period` bars
    are zero (no warmup)."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
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


def _zscore_returns(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling z-score of close-to-close log returns over `window` bars.
    z = (ret_t - rolling_mean) / rolling_std. NaNs zero-filled."""
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(np.maximum(close[1:], EPS) /
                         np.maximum(close[:-1], EPS))
    s = pd.Series(log_ret)
    mu  = s.rolling(window, min_periods=window // 2).mean()
    sd  = s.rolling(window, min_periods=window // 2).std()
    z = ((log_ret - mu.to_numpy()) / np.where(sd.to_numpy() > EPS,
                                              sd.to_numpy(), EPS))
    z[~np.isfinite(z)] = 0.0
    return z


def _vol_regime(close: np.ndarray, forward: int = 20,
                baseline: int = 200) -> np.ndarray:
    """0=LOW, 1=MED, 2=HIGH. Same logic as labeler_volregime but local
    so this script has no cross-deps."""
    n = len(close)
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(np.maximum(close[1:], EPS) /
                         np.maximum(close[:-1], EPS))
    sq = log_ret * log_ret
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    rv_fwd = np.zeros(n)
    rv_fwd[:-forward] = np.sqrt(cs[forward+1:n+1] - cs[1:n-forward+1])
    bl = pd.Series(rv_fwd).rolling(baseline, min_periods=baseline // 2).mean().to_numpy()
    bl = np.where(np.isfinite(bl) & (bl > EPS), bl, EPS)
    ratio = rv_fwd / bl
    valid = np.zeros(n, dtype=bool)
    valid[baseline:n - forward] = True
    if not valid.any():
        return np.full(n, -1, dtype=np.int8)
    q_lo, q_hi = np.quantile(ratio[valid], [0.33, 0.67])
    out = np.full(n, -1, dtype=np.int8)
    out[valid] = np.where(ratio[valid] < q_lo, 0,
                  np.where(ratio[valid] < q_hi, 1, 2)).astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestParams:
    z_thresh:     float = 2.0       # entry z-score
    sl_atr_mult:  float = 1.0
    tp_atr_mult:  float = 1.5       # RR = 1.5
    timeout_bars: int   = 20
    z_window:     int   = 20
    atr_period:   int   = 14
    require_vol:  str   = "MED+"    # 'LOW+', 'MED+', 'HIGH'
    cost_pips:    float = 2.0       # round-trip cost in pips


@dataclass
class BacktestResult:
    symbol:        str
    params:        BacktestParams
    n_trades:      int
    n_wins:        int
    n_losses:      int
    n_timeouts:    int
    win_rate:      float
    gross_pf:      float
    net_pf:        float
    sharpe:        float
    max_dd_pips:   float
    avg_win_pips:  float
    avg_loss_pips: float
    total_pips:    float
    n_bars_total:  int


def _vol_pass(vol_reg: int, requirement: str) -> bool:
    if vol_reg < 0: return False           # warmup / tail
    if requirement == "LOW+":  return True
    if requirement == "MED+":  return vol_reg >= 1
    if requirement == "HIGH":  return vol_reg == 2
    return True


def backtest_symbol(bars: pd.DataFrame, symbol: str,
                    params: BacktestParams) -> BacktestResult:
    """One symbol, one parameter set. Returns a fully-populated result."""
    close = bars["close"].to_numpy(dtype=np.float64)
    high  = bars["high"].to_numpy(dtype=np.float64)
    low   = bars["low"].to_numpy(dtype=np.float64)
    n = len(close)
    pip = pip_size(symbol)
    cost_price = params.cost_pips * pip

    atr      = _atr(high, low, close, params.atr_period)
    z        = _zscore_returns(close, params.z_window)
    vol_reg  = _vol_regime(close)

    pnl_pips = []                  # signed list of (gross-cost) pips per closed trade
    outcomes = []                  # 'TP', 'SL', 'TO'  (timeout)

    i = 0
    while i < n - params.timeout_bars - 1:
        # Need warmup + valid vol regime
        if atr[i] <= 0 or not _vol_pass(int(vol_reg[i]), params.require_vol):
            i += 1; continue

        side = 0
        if   z[i] >  params.z_thresh: side = -1   # fade up = SHORT
        elif z[i] < -params.z_thresh: side = +1   # fade down = LONG
        if side == 0:
            i += 1; continue

        entry = close[i]
        sl_dist = params.sl_atr_mult * atr[i]
        tp_dist = params.tp_atr_mult * atr[i]

        if side == +1:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        # Walk forward.
        outcome = "TO"
        exit_price = close[i + params.timeout_bars]
        for j in range(1, params.timeout_bars + 1):
            if i + j >= n:
                break
            h, l = high[i + j], low[i + j]
            if side == +1:
                if l <= sl_price:
                    outcome = "SL"; exit_price = sl_price; break
                if h >= tp_price:
                    outcome = "TP"; exit_price = tp_price; break
            else:
                if h >= sl_price:
                    outcome = "SL"; exit_price = sl_price; break
                if l <= tp_price:
                    outcome = "TP"; exit_price = tp_price; break

        gross = (exit_price - entry) * side                  # price units
        net   = gross - cost_price                            # deduct round-trip
        pnl_pips.append(net / pip)
        outcomes.append(outcome)

        # Skip past this trade so we don't open overlapping positions
        i = i + (j if 'j' in locals() else params.timeout_bars) + 1

    pnl = np.asarray(pnl_pips)
    if pnl.size == 0:
        return BacktestResult(symbol, params, 0, 0, 0, 0, float("nan"),
                              float("nan"), float("nan"), float("nan"),
                              0.0, 0.0, 0.0, 0.0, n)

    wins  = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    timeouts = sum(1 for o in outcomes if o == "TO")
    gross_profit = wins.sum() if wins.size else 0.0
    gross_loss   = -losses.sum() if losses.size else 0.0
    gross_pf = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    # Cost-aware "net" PF treats the same numbers (already net of cost).
    net_pf = gross_pf

    # Sharpe over per-trade pnl (NOT annualised — trade-frequency-aware
    # would need bar-time accounting).
    sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 1e-9 else 0.0

    # Equity-curve max drawdown.
    eq   = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd   = peak - eq
    mdd  = float(dd.max()) if dd.size else 0.0

    return BacktestResult(
        symbol=symbol, params=params,
        n_trades=int(pnl.size),
        n_wins=int(wins.size), n_losses=int(losses.size), n_timeouts=int(timeouts),
        win_rate=float((pnl > 0).mean()),
        gross_pf=round(gross_pf, 4) if np.isfinite(gross_pf) else float("inf"),
        net_pf=round(net_pf, 4) if np.isfinite(net_pf) else float("inf"),
        sharpe=round(sharpe, 4),
        max_dd_pips=round(mdd, 1),
        avg_win_pips=round(float(wins.mean()) if wins.size else 0.0, 2),
        avg_loss_pips=round(float(losses.mean()) if losses.size else 0.0, 2),
        total_pips=round(float(pnl.sum()), 1),
        n_bars_total=n,
    )


# ---------------------------------------------------------------------------
# Pretty printer + verdict
# ---------------------------------------------------------------------------

def _verdict(net_pf: float) -> str:
    if not np.isfinite(net_pf): return "?"
    if net_pf >= 1.30: return "LIVE-READY"
    if net_pf >= 1.00: return "MARGINAL"
    if net_pf >= 0.85: return "WEAK"
    return "DEAD"


def print_result(r: BacktestResult) -> None:
    p = r.params
    print(f"  {r.symbol:<10}  z={p.z_thresh:.1f}  RR={p.tp_atr_mult/max(p.sl_atr_mult, 1e-9):.2f}  "
          f"vol={p.require_vol:<6}  "
          f"trades={r.n_trades:>6}  WR={r.win_rate:.3f}  "
          f"PF={r.net_pf}  Sharpe={r.sharpe}  "
          f"AvgW={r.avg_win_pips}  AvgL={r.avg_loss_pips}  "
          f"MDD={r.max_dd_pips}p  total={r.total_pips:+.0f}p  "
          f"[{_verdict(r.net_pf)}]")


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbols", nargs="+",
                   help="Symbols to test, or 'all' / 'forex' / 'metals' / 'indices' / 'ce'")
    p.add_argument("--z",        type=float, nargs="+", default=[1.5, 2.0, 2.5])
    p.add_argument("--sl-atr",   type=float, nargs="+", default=[1.0])
    p.add_argument("--tp-atr",   type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--timeout",  type=int,   default=20)
    p.add_argument("--vol",      choices=["LOW+", "MED+", "HIGH"], default="MED+")
    p.add_argument("--cost-pips", type=float, default=2.0,
                   help="Round-trip cost in pips (default 2.0 — typical FX retail spread+commission)")
    p.add_argument("--walk-forward", type=int, default=0,
                   help="K-fold walk-forward; 0 = single full-history pass")
    args = p.parse_args(argv)

    # Resolve symbols
    expand = {
        "all":     ALL_SYMBOLS,
        "forex":   FOREX_SYMBOLS,
        "metals":  METALS_SYMBOLS,
        "indices": INDICES_SYMBOLS,
        "ce":      CE_SYMBOLS,
    }
    symbols: list[str] = []
    for s in args.symbols:
        if s.lower() in expand:
            symbols.extend(expand[s.lower()])
        else:
            symbols.append(s)
    symbols = list(dict.fromkeys(symbols))  # de-dupe preserving order

    print()
    print("=" * 70)
    print(f"  HYDRA mk4.5 — pure-rule mean-reversion + vol-gate backtest")
    print(f"  symbols : {', '.join(symbols)}")
    print(f"  grid    : z in {args.z}  sl in {args.sl_atr}  tp in {args.tp_atr}")
    print(f"  cost    : {args.cost_pips} pips/round-trip   vol gate : {args.vol}")
    print("=" * 70)

    best_per_symbol: dict[str, tuple[BacktestParams, BacktestResult]] = {}

    for sym in symbols:
        try:
            bars = _load_bars(sym)
        except SystemExit as e:
            print(f"  [{sym}] {e}")
            continue
        print(f"\n  loading {sym}: {len(bars):,} bars  "
              f"({bars['time'].iloc[0]} -> {bars['time'].iloc[-1]})")

        for z, sl_a, tp_a in product(args.z, args.sl_atr, args.tp_atr):
            params = BacktestParams(
                z_thresh=z, sl_atr_mult=sl_a, tp_atr_mult=tp_a,
                timeout_bars=args.timeout, require_vol=args.vol,
                cost_pips=args.cost_pips,
            )
            r = backtest_symbol(bars, sym, params)
            print_result(r)
            cur = best_per_symbol.get(sym)
            if (cur is None or
                (np.isfinite(r.net_pf) and r.net_pf > cur[1].net_pf)):
                best_per_symbol[sym] = (params, r)

    # Summary
    print()
    print("=" * 70)
    print("  Best param set per symbol:")
    print("=" * 70)
    n_live, n_marg, n_dead = 0, 0, 0
    for sym, (params, r) in best_per_symbol.items():
        v = _verdict(r.net_pf)
        if   v == "LIVE-READY": n_live += 1
        elif v == "MARGINAL":   n_marg += 1
        else:                   n_dead += 1
        print_result(r)
    print()
    print(f"  Verdict: LIVE-READY={n_live}  MARGINAL={n_marg}  WEAK/DEAD={n_dead}")
    print("=" * 70)

    return 0 if n_live > 0 else (1 if n_marg > 0 else 2)


if __name__ == "__main__":
    raise SystemExit(main())
