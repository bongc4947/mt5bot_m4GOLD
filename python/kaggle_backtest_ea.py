"""
kaggle_backtest_ea.py - faithful Python port of the v1.30 MetaTrend EA.

Runs the SAME ONNX meta-gate and SAME sr_fib feature extractor as the live
MT5 EA, replays the EA's decision tree on a pandas M5 OHLC bar series, and
reports equity / PF / max-DD / per-trade ledger. Designed for Kaggle
out-of-sample testing on the comprehensive XAUUSD historical dataset.

EA logic ported (mirrors ea/includes/MetaGate.mqh + ea/MT5bot_m4Gold_MetaTrend.mq5):
  - Direction: EMA(50) cross EMA(200) on closed M5 bars
  - Consent: XGBoost meta-gate ONNX P(act) >= act_threshold (from spec)
  - Entry: open at the M5 close (next-bar fill), initial SL at SlAtr * ATR(14)
  - Exit engine (in order, per-bar resolution):
      1) breakeven move: at +BreakevenAtr, ratchet SL to entry + buffer
      2) ATR trailing stop: at +TrailStartAtr profit, trail SlBehind = TrailAtr * ATR
      3) trend flip: close on next bar if EMA cross direction reversed
      4) timeout: close after MaxHoldBars
      5) SL hit: any time price touches SL, close at SL price

Bar resolution: we check SL hit using bar high/low intra-bar. Trail/breakeven
updates use the bar's close. This is a slightly LESS pessimistic SL model
than tick-level (we don't simulate wick-then-recovery), but matches MT5's
M1-OHLC tick generation model the EA was Tester-validated against.

Cost model: a fixed round-trip cost in price units (default $0.40, ~ AvaTrade
median 0.27 USD spread). Pass --spread-usd to change.

Usage:
    python python/kaggle_backtest_ea.py \\
        --data /kaggle/input/comprehensive-xauusd-historical-price-data/XAUUSD_M5.csv \\
        --onnx onnx_out/M4GOLD_METATREND_GOLD.onnx \\
        --spec onnx_out/M4GOLD_METATREND_GOLD_spec.json \\
        --deposit 10000 --lot 0.01 --output results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# These MUST match python/aurum/metatrend.py + MetaGate.mqh
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
N_BASE_FEATURES = 18

# Default EA inputs (override via CLI / Kaggle notebook parameters)
DEFAULT_INPUTS = dict(
    base_lot           = 0.01,
    sl_atr             = 3.0,
    tp_atr             = 0.0,        # 0 = disabled
    use_breakeven      = True,
    breakeven_atr      = 1.0,
    breakeven_buffer   = 0.05,
    use_trailing       = True,
    trail_start_atr    = 2.0,
    trail_atr          = 3.0,
    max_hold_bars      = 288,        # ~24 h
    exit_on_flip       = True,
    max_stack          = 1,          # 1 = no pyramiding (Tester-validated)
    stack_step_atr     = 1.0,
    spread_usd         = 0.40,       # round-trip cost in price units
    contract_size      = 100.0,      # 1 lot = 100 oz GOLD on most brokers
)


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------
@dataclass
class Position:
    open_idx:    int
    open_time:   pd.Timestamp
    open_price:  float
    side:        int            # +1 long, -1 short
    lot:         float
    sl:          float
    atr_at_open: float
    spread_pts_at_open: float = -1.0   # -1 = use flat fallback
    breakeven_done: bool = False


@dataclass
class ClosedTrade:
    open_time:   pd.Timestamp
    close_time:  pd.Timestamp
    side:        str            # "long" or "short"
    open_price:  float
    close_price: float
    lot:         float
    raw_pnl:     float          # before cost
    cost:        float
    pnl:         float          # net
    exit_reason: str            # "sl" | "trail" | "flip" | "timeout" | "tp"


# ---------------------------------------------------------------------------
# Feature builder + meta-gate
# ---------------------------------------------------------------------------
def build_features_at(m5: pd.DataFrame, n_features: int, sr_extract=None,
                       use_sr: bool = False) -> np.ndarray:
    """
    Build the full per-anchor feature matrix.

    `m5` must have columns: time, open, high, low, close. Time must be
    parseable to UTC.

    Returns float32[N, n_features] where row i corresponds to the closed
    M5 bar at index i (decision time = i's close).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from aurum.metatrend import build_features  # the 18-feat builder
    X_base = build_features(m5)
    if not use_sr:
        return X_base
    if sr_extract is None:
        from aurum.sr_fib_features import extract as sr_extract
    anchors = np.arange(len(m5), dtype=np.int64)
    # the sr_fib extractor needs ~21 days of warmup
    sr_warmup = 24 * 21 * 12   # 21 days of M5 bars
    if len(m5) < sr_warmup + 100:
        raise ValueError(
            f"need >= {sr_warmup + 100} M5 bars for sr_fib features, "
            f"got {len(m5)}")
    sr_anchors = np.arange(sr_warmup, len(m5), dtype=np.int64)
    X_sr_partial = sr_extract(m5, sr_anchors)
    X_sr = np.zeros((len(m5), X_sr_partial.shape[1]), dtype=np.float32)
    X_sr[sr_warmup:] = X_sr_partial
    return np.concatenate([X_base, X_sr], axis=1).astype(np.float32)


def load_meta_gate(onnx_path: Path, spec_path: Path):
    import onnxruntime as ort
    spec = json.loads(spec_path.read_text())
    sess = ort.InferenceSession(str(onnx_path),
                                 providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    return sess, in_name, spec


# ---------------------------------------------------------------------------
# Data loader - flexible Kaggle/CSV/parquet ingest
# ---------------------------------------------------------------------------
def load_m5_data(path: Path) -> pd.DataFrame:
    """
    Load OHLC data from CSV/parquet, normalise to expected schema:
        columns = ['time', 'open', 'high', 'low', 'close']  (volume optional)
    Handles:
      - comma-separated CSV (standard)
      - tab-separated CSV (MetaTrader HST export)
      - column names wrapped in `<>` brackets like `<DATE>`, `<OPEN>`
      - separate `date` + `time` columns vs single `datetime` column
      - parquet files
    """
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        # Detect separator: peek at first non-empty line
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = ""
            for line in f:
                if line.strip():
                    first_line = line
                    break
        n_tabs   = first_line.count("\t")
        n_commas = first_line.count(",")
        sep = "\t" if n_tabs > n_commas else ","
        log.info("[backtest] detected separator: %s (tabs=%d, commas=%d)",
                 "TAB" if sep == "\t" else "COMMA", n_tabs, n_commas)
        df = pd.read_csv(path, sep=sep)

    # Normalise column names: strip <>, lowercase, strip whitespace + underscores
    df.columns = [str(c).strip().lower().lstrip("<").rstrip(">").strip("_ ")
                  for c in df.columns]

    # Common alias map
    time_aliases  = ["time", "datetime", "timestamp", "date_time", "datetime_utc"]
    date_aliases  = ["date"]
    open_aliases  = ["open", "o"]
    high_aliases  = ["high", "h"]
    low_aliases   = ["low", "l"]
    close_aliases = ["close", "c", "price"]

    def _find(aliases):
        for a in aliases:
            if a in df.columns: return a
        return None

    c_time  = _find(time_aliases)
    c_date  = _find(date_aliases)
    c_open  = _find(open_aliases)
    c_high  = _find(high_aliases)
    c_low   = _find(low_aliases)
    c_close = _find(close_aliases)
    if None in (c_open, c_high, c_low, c_close):
        raise ValueError(
            f"could not find OHLC columns in {path.name}. "
            f"Found: {list(df.columns)}")
    # MT5 HST exports include a <SPREAD> column in POINTS (typically 1 point
    # = 0.01 USD for GOLD). Carry it through if present so callers can build
    # a per-bar variable-spread cost model instead of a flat assumption.
    spread_aliases = ["spread", "spread_points"]
    c_spread = _find(spread_aliases)
    if c_spread:
        df = df.rename(columns={c_spread: "spread"})
    # Combine date + time if separate, otherwise use whichever is present
    if c_time and c_date and c_time != c_date:
        # MetaTrader HST format: <DATE>=YYYY.MM.DD, <TIME>=HH:MM
        df["time"] = pd.to_datetime(df[c_date].astype(str) + " "
                                     + df[c_time].astype(str),
                                     errors="coerce", utc=True)
    elif c_time:
        df["time"] = pd.to_datetime(df[c_time], errors="coerce", utc=True)
    elif c_date:
        df["time"] = pd.to_datetime(df[c_date], errors="coerce", utc=True)
    else:
        raise ValueError(
            f"no time/date column in {path.name}. Found: {list(df.columns)}")
    df = df.rename(columns={c_open: "open", c_high: "high",
                             c_low: "low", c_close: "close"})

    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.sort_values("time").reset_index(drop=True)
    log.info("[backtest] loaded %d bars  %s -> %s",
             len(df), df["time"].iloc[0], df["time"].iloc[-1])
    if "spread" in df.columns:
        log.info("[backtest] data includes <SPREAD> column: median=%.1f pts, mean=%.1f pts",
                 float(df["spread"].median()), float(df["spread"].mean()))

    # Detect granularity - resample to M5 if needed
    if len(df) > 1:
        dt = (df["time"].iloc[1] - df["time"].iloc[0]).total_seconds()
        if abs(dt - 300) > 60:
            log.info("[backtest] data is %.0fs cadence, resampling to M5", dt)
            agg = {"open": ("open", "first"), "high": ("high", "max"),
                   "low": ("low", "min"),     "close": ("close", "last")}
            if "spread" in df.columns:
                agg["spread"] = ("spread", "mean")
            g = df.set_index("time")
            df = g.resample("5min", label="left", closed="left").agg(**agg) \
                  .dropna(subset=["close"]).reset_index()
            log.info("[backtest] after resample: %d M5 bars", len(df))
    keep_cols = ["time", "open", "high", "low", "close"]
    if "spread" in df.columns: keep_cols.append("spread")
    return df[keep_cols]


# ---------------------------------------------------------------------------
# ATR(14) on M5
# ---------------------------------------------------------------------------
def atr14(m5: pd.DataFrame) -> np.ndarray:
    h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64)
    c = m5["close"].to_numpy(np.float64)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().to_numpy()


def primary_signal(close: np.ndarray) -> np.ndarray:
    ef = pd.Series(close).ewm(span=EMA_FAST, adjust=False).mean()
    es = pd.Series(close).ewm(span=EMA_SLOW, adjust=False).mean()
    return np.where(ef.to_numpy() > es.to_numpy(), 1, -1).astype(np.int64)


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------
def run_backtest(m5: pd.DataFrame, sess, in_name, spec: dict,
                 inputs: dict, deposit: float = 10000.0,
                 verbose: bool = False):
    """Returns dict with equity_curve, trades, summary stats."""
    n_features = spec["n_features"]
    act_thr = spec["act_threshold"]
    use_sr = (n_features == 24)

    log.info("[backtest] building features (n=%d, sr_fib=%s) ...",
             n_features, use_sr)
    X = build_features_at(m5, n_features, use_sr=use_sr)
    assert X.shape[1] == n_features, (X.shape, n_features)

    atr = atr14(m5)
    prim = primary_signal(m5["close"].to_numpy(np.float64))
    n = len(m5)
    closes = m5["close"].to_numpy(np.float64)
    highs  = m5["high"].to_numpy(np.float64)
    lows   = m5["low"].to_numpy(np.float64)

    # Run meta-gate per bar (batched would be faster but our ONNX is XGBoost
    # which doesn't batch cleanly via onnxruntime - per-row is fine, ~5ms each)
    log.info("[backtest] scoring meta-gate on %d bars ...", n)
    pact = np.zeros(n, dtype=np.float32)
    # XGBoost ONNX output: [labels, probs] - probs[0][1] is P(class=1)
    for i in range(n):
        x = X[i:i+1].astype(np.float32)
        out = sess.run(None, {in_name: x})
        pact[i] = float(out[1][0][1])

    # Backtest state
    equity = deposit
    peak_equity = deposit
    max_dd = 0.0
    max_dd_pct = 0.0
    open_pos: Optional[Position] = None
    trades: list[ClosedTrade] = []
    equity_curve = np.zeros(n, dtype=np.float64)

    inp = inputs
    # Cost model:
    #   - If `use_variable_spread=True` AND the data has a `spread` column,
    #     use per-bar broker spread in points. 1 point = 0.01 USD on GOLD.
    #     Cost per round-trip = spread_at_open * 0.01 * lot * contract_size
    #     (in points → USD-per-unit → USD-per-lot).
    #   - Otherwise fall back to the flat `spread_usd` assumption (the
    #     original behaviour). spread_usd is the round-trip cost in price
    #     units, multiplied by lot*contract_size to get USD.
    use_variable = bool(inp.get("use_variable_spread", False)) and ("spread" in m5.columns)
    if use_variable:
        spread_pts = m5["spread"].fillna(0).to_numpy(np.float64)
        log.info("[backtest] cost model: per-bar variable (mean spread %.1f pts)",
                 float(spread_pts.mean()))
    else:
        log.info("[backtest] cost model: flat spread_usd=$%.3f round-trip",
                 inp["spread_usd"])
    cost_per_lot = inp["spread_usd"] * inp["contract_size"]  # flat fallback

    # The first ~max(EMA200, sr_warmup) bars are warmup
    sr_warmup = (24 * 21 * 12) if use_sr else 0
    start_i = max(EMA_SLOW + 50, sr_warmup + 1, ATR_PERIOD)

    for i in range(start_i, n - 1):
        cur_close = closes[i]
        cur_atr = atr[i]
        if np.isnan(cur_atr) or cur_atr <= 0:
            equity_curve[i] = equity + _floating_pnl(open_pos, cur_close, inp)
            continue

        # === position management - mimic MT5 Model=1 tick generation ===
        # MT5's standard intra-bar tick order depends on the bar's direction:
        #   bullish bar (close >= open):  open -> low -> high -> close
        #   bearish bar (close <  open):  open -> high -> low -> close
        # We simulate that ordering: visit the adverse-or-favorable prices
        # in the right sequence, updating breakeven/trail and checking SL
        # hits in the order they would actually happen.
        if open_pos is not None:
            bullish_bar = closes[i] >= m5["open"].iloc[i]
            # for LONG position: favorable=high, adverse=low
            # for SHORT position: favorable=low,  adverse=high
            # MT5 order says LOW comes first in a bullish bar
            if open_pos.side > 0:
                fav, adv = highs[i], lows[i]
                adv_first = bullish_bar              # bullish -> low first
            else:
                fav, adv = lows[i], highs[i]
                adv_first = not bullish_bar          # bearish -> high first (adverse to a short)

            def update_trail_be(at_price):
                """Update breakeven + trail using `at_price` as the favorable touch."""
                prof_atr = ((at_price - open_pos.open_price) if open_pos.side > 0
                            else (open_pos.open_price - at_price)) / open_pos.atr_at_open
                # breakeven
                if inp["use_breakeven"] and not open_pos.breakeven_done \
                   and prof_atr >= inp["breakeven_atr"]:
                    buf = inp["breakeven_buffer"] * open_pos.atr_at_open
                    be = (open_pos.open_price + buf) if open_pos.side > 0 \
                         else (open_pos.open_price - buf)
                    if (open_pos.side > 0 and be > open_pos.sl) or \
                       (open_pos.side < 0 and be < open_pos.sl):
                        open_pos.sl = be
                        open_pos.breakeven_done = True
                # trailing
                if inp["use_trailing"] and prof_atr >= inp["trail_start_atr"]:
                    tr = (at_price - inp["trail_atr"] * open_pos.atr_at_open) \
                         if open_pos.side > 0 \
                         else (at_price + inp["trail_atr"] * open_pos.atr_at_open)
                    if (open_pos.side > 0 and tr > open_pos.sl) or \
                       (open_pos.side < 0 and tr < open_pos.sl):
                        open_pos.sl = tr

            def check_sl_hit():
                if open_pos.side > 0 and adv <= open_pos.sl: return True
                if open_pos.side < 0 and adv >= open_pos.sl: return True
                return False

            if adv_first:
                # adverse extreme reached first - test SL against ORIGINAL sl
                if check_sl_hit():
                    _close_position(open_pos, open_pos.sl, m5["time"].iloc[i],
                                    "sl", inp, cost_per_lot, trades,
                                    open_pos.spread_pts_at_open)
                    equity += trades[-1].pnl
                    open_pos = None
                else:
                    # SL not hit, then favorable price hits -> update trail
                    update_trail_be(fav)
            else:
                # favorable first - trail updates, THEN test SL with new (tighter) SL
                update_trail_be(fav)
                if check_sl_hit():
                    _close_position(open_pos, open_pos.sl, m5["time"].iloc[i],
                                    "sl", inp, cost_per_lot, trades,
                                    open_pos.spread_pts_at_open)
                    equity += trades[-1].pnl
                    open_pos = None

            # trend flip (close at next bar's open, or current close if same bar)
            if open_pos is not None and inp["exit_on_flip"] \
               and prim[i] != 0 and prim[i] != open_pos.side:
                _close_position(open_pos, cur_close, m5["time"].iloc[i],
                                "flip", inp, cost_per_lot, trades,
                                open_pos.spread_pts_at_open)
                equity += trades[-1].pnl
                open_pos = None

            # timeout
            if open_pos is not None and (i - open_pos.open_idx) >= inp["max_hold_bars"]:
                _close_position(open_pos, cur_close, m5["time"].iloc[i],
                                "timeout", inp, cost_per_lot, trades,
                                open_pos.spread_pts_at_open)
                equity += trades[-1].pnl
                open_pos = None

        # === entry decision (only on closed-bar basis) ===
        if open_pos is None and prim[i] != 0 and pact[i] >= act_thr:
            sl_dist = inp["sl_atr"] * cur_atr
            entry_price = cur_close
            sl = (entry_price - sl_dist) if prim[i] > 0 \
                 else (entry_price + sl_dist)
            open_pos = Position(
                open_idx    = i,
                spread_pts_at_open = (float(spread_pts[i]) if use_variable else -1.0),
                open_time   = m5["time"].iloc[i],
                open_price  = entry_price,
                side        = int(prim[i]),
                lot         = inp["base_lot"],
                sl          = sl,
                atr_at_open = cur_atr,
            )

        # equity curve (mark-to-market with floating)
        equity_curve[i] = equity + _floating_pnl(open_pos, cur_close, inp)

        # track DD on equity curve
        if equity_curve[i] > peak_equity: peak_equity = equity_curve[i]
        dd = peak_equity - equity_curve[i]
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / max(peak_equity, 1) * 100

        if verbose and i % 5000 == 0:
            log.info("[backtest] i=%d  bar=%s  equity=%.2f  trades=%d  open=%s",
                     i, m5["time"].iloc[i], equity_curve[i], len(trades),
                     "yes" if open_pos else "no")

    # close any open position at last bar
    if open_pos is not None:
        _close_position(open_pos, closes[-1], m5["time"].iloc[-1],
                        "end-of-data", inp, cost_per_lot, trades,
                        open_pos.spread_pts_at_open)
        equity += trades[-1].pnl
        equity_curve[-1] = equity

    # ---- summary stats ----
    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = float(wins.sum() / max(-losses.sum(), 1e-12)) if losses.any() else float("inf")
    summary = dict(
        n_trades       = len(trades),
        wins           = int((pnls > 0).sum()),
        losses         = int((pnls <= 0).sum()),
        win_rate_pct   = round(float((pnls > 0).mean() * 100), 2) if len(trades) else 0.0,
        profit_factor  = round(min(pf, 99.0), 3),
        avg_win        = round(float(wins.mean()), 2) if len(wins) else 0.0,
        avg_loss       = round(float(-losses.mean()), 2) if len(losses) else 0.0,
        total_pnl      = round(float(pnls.sum()), 2),
        final_equity   = round(float(equity), 2),
        return_pct     = round(float((equity - deposit) / deposit * 100), 2),
        peak_equity    = round(float(peak_equity), 2),
        max_dd         = round(float(max_dd), 2),
        max_dd_pct     = round(float(max_dd_pct), 2),
        n_bars         = int(n),
        n_active_bars  = int(n - start_i),
    )
    return dict(
        summary       = summary,
        trades        = [asdict(t) for t in trades],
        equity_curve  = equity_curve.tolist(),
    )


def _floating_pnl(pos: Optional[Position], cur_close: float, inp: dict) -> float:
    if pos is None: return 0.0
    move = (cur_close - pos.open_price) * pos.side
    return move * pos.lot * inp["contract_size"]


def _close_position(pos: Position, exit_price: float, exit_time: pd.Timestamp,
                    reason: str, inp: dict, cost_per_lot: float,
                    trades: list, spread_pts_at_open: float = -1.0):
    move = (exit_price - pos.open_price) * pos.side
    raw_pnl = move * pos.lot * inp["contract_size"]
    if spread_pts_at_open >= 0:
        # variable cost: spread in POINTS at open time. 1 point = 0.01 USD on GOLD
        # (configurable via inp["point_value_usd"], default 0.01)
        pt_val = inp.get("point_value_usd", 0.01)
        cost = spread_pts_at_open * pt_val * pos.lot * inp["contract_size"]
    else:
        cost = cost_per_lot * pos.lot
    net = raw_pnl - cost
    trades.append(ClosedTrade(
        open_time   = pos.open_time,
        close_time  = exit_time,
        side        = "long" if pos.side > 0 else "short",
        open_price  = pos.open_price,
        close_price = exit_price,
        lot         = pos.lot,
        raw_pnl     = round(raw_pnl, 4),
        cost        = round(cost, 4),
        pnl         = round(net, 4),
        exit_reason = reason,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",  required=True, help="CSV or parquet OHLC file")
    p.add_argument("--onnx",  required=True, help="Meta-gate ONNX path")
    p.add_argument("--spec",  required=True, help="Meta-gate spec.json path")
    p.add_argument("--deposit", type=float, default=10000.0)
    p.add_argument("--lot",     type=float, default=0.01)
    p.add_argument("--spread-usd", type=float, default=0.40,
                   help="Round-trip cost in USD price units (default 0.40)")
    p.add_argument("--sl-atr",   type=float, default=3.0)
    p.add_argument("--max-stack", type=int, default=1)
    p.add_argument("--output",  default=None,
                   help="Write JSON results here (defaults to stdout summary)")
    p.add_argument("--from-date", default=None,
                   help="Optional ISO date filter (e.g. 2025-11-20)")
    p.add_argument("--to-date",   default=None)
    args = p.parse_args(argv)

    m5 = load_m5_data(Path(args.data))
    if args.from_date:
        m5 = m5[m5["time"] >= pd.Timestamp(args.from_date, tz="UTC")]
    if args.to_date:
        m5 = m5[m5["time"] <= pd.Timestamp(args.to_date, tz="UTC")]
    m5 = m5.reset_index(drop=True)
    log.info("[backtest] window after filter: %d bars  %s -> %s",
             len(m5), m5["time"].iloc[0], m5["time"].iloc[-1])

    sess, in_name, spec = load_meta_gate(Path(args.onnx), Path(args.spec))
    log.info("[backtest] meta-gate ready: n_features=%d  act_thr=%.2f  "
             "version=%s",
             spec["n_features"], spec["act_threshold"], spec.get("version", "?"))

    inputs = dict(DEFAULT_INPUTS)
    inputs.update(dict(base_lot=args.lot, sl_atr=args.sl_atr,
                       max_stack=args.max_stack, spread_usd=args.spread_usd))

    results = run_backtest(m5, sess, in_name, spec, inputs,
                           deposit=args.deposit, verbose=True)
    s = results["summary"]
    print()
    print("=" * 70)
    print(f"  BACKTEST RESULT - {Path(args.data).name}")
    print("=" * 70)
    print(f"  Window:          {m5['time'].iloc[0]} -> {m5['time'].iloc[-1]}")
    print(f"  Deposit:         ${args.deposit:,.0f}    Lot: {args.lot}")
    print(f"  Spread (USD):    ${args.spread_usd}  ({args.spread_usd * 100} pips at $0.01)")
    print(f"  -------------------------------------------------")
    print(f"  Final equity:    ${s['final_equity']:,.2f}  ({s['return_pct']:+.2f}%)")
    print(f"  Trades:          {s['n_trades']}    Wins/Losses: {s['wins']}/{s['losses']}")
    print(f"  Win rate:        {s['win_rate_pct']:.1f}%")
    print(f"  Profit factor:   {s['profit_factor']}")
    print(f"  Avg win/loss:    +${s['avg_win']} / -${s['avg_loss']}")
    print(f"  Max drawdown:    ${s['max_dd']:,.2f}  ({s['max_dd_pct']:.2f}%)")
    print("=" * 70)

    if args.output:
        # don't dump the entire equity curve to the summary JSON
        compact = dict(results)
        compact["equity_curve_len"] = len(compact.pop("equity_curve"))
        Path(args.output).write_text(json.dumps(compact, indent=2, default=str))
        log.info("[backtest] results written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
