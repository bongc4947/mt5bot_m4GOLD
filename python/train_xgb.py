"""
train_xgb.py — gradient-boosted-trees alternative to the PRISM/APEX/GNN/CE
direction MLPs.

mk4.4 #2: drop-in alternative model class. Same 200-dim feature input
(parity-floored), same +1 / -1 / 0 label space, same MT5_COMMON_DIR
ONNX output path. Whether to use XGBoost vs the MLP is a runtime choice
made at training time; the EA reads whatever ONNX file is in place.

USAGE
-----
    python python/train_xgb.py prism --symbol EURUSD --skip-extract
    python python/train_xgb.py all   --skip-extract --estimators 800

Same CLI shape as train.py, so cloud/runner.sh / cloud/notebook_run.py
can target either by changing the train command.

WHAT IT DOES
------------
    1. Loads cached bars (or pulls fresh via MT5).
    2. Filters pre-MIN_BAR_DATE.
    3. Builds 200-dim features via feature_engine.
    4. Builds triple-barrier labels (mk4.4 #1) — the same labels the
       MLP path uses, so MLP-vs-XGB is an apples-to-apples comparison.
    5. Time-aware split (chronological + LABEL_FORWARD_BARS gap).
    6. Trains XGBoost direction head (LONG vs SHORT, ignoring FLAT).
    7. Reports val_acc + all-bars / conf-only PF/Sharpe (same metric
       harness as _train_agent.train_symbol).
    8. Exports ONNX + meta.json next to the existing PRISM ONNX so the
       EA loads whichever you deploy.
    9. Writes top-20 feature-importance to logs (free diagnostic).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FEATURE_DIM_DIR, VAL_SPLIT, LABEL_FORWARD_BARS, CONF_THRESHOLD,
    FOREX_SYMBOLS, METALS_SYMBOLS, INDICES_SYMBOLS, CE_SYMBOLS,
    MIN_BAR_DATE, ONNX_OUTPUT_DIR, meta_path,
)

log = logging.getLogger(__name__)

AGENTS_FOR_XGB = {
    "prism": ("PRISM", FOREX_SYMBOLS),
    "gnn":   ("GNN",   METALS_SYMBOLS),
    "apex":  ("APEX",  INDICES_SYMBOLS),
    "ce":    ("CE",    CE_SYMBOLS),
}


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


def _load_bars(symbol: str, skip_extract: bool):
    import pandas as pd
    from config import PARQUET_DIR
    if not skip_extract:
        from data_pipeline import run_full_pipeline, connect, disconnect
        from hardware_detector import get as get_hw
        if not connect():
            raise SystemExit("MT5 connect failed (use --skip-extract on Linux)")
        try:
            bars = run_full_pipeline(symbols=[symbol],
                                      max_bars=get_hw().max_bars).get(symbol)
        finally:
            disconnect()
    else:
        cached = sorted(PARQUET_DIR.glob(f"HYDRA4_FEAT_{symbol}_*.parquet"),
                        key=lambda p: p.stat().st_size, reverse=True)
        if not cached:
            raise SystemExit(f"No cached parquet for {symbol}; remove --skip-extract")
        bars = pd.read_parquet(cached[0])

    if bars is None or len(bars) == 0:
        raise SystemExit(f"{symbol}: zero bars")
    if MIN_BAR_DATE and "time" in bars.columns:
        cutoff = pd.Timestamp(MIN_BAR_DATE, tz="UTC")
        n_before = len(bars)
        bars = bars[bars["time"] >= cutoff].reset_index(drop=True)
        if len(bars) < n_before:
            log.info("[%s] Dropped %d bars older than %s",
                     symbol, n_before - len(bars), MIN_BAR_DATE)
    return bars


def _train_one(symbol: str, agent: str, *,
               estimators: int, max_depth: int, lr: float,
               skip_extract: bool, seed: int) -> dict:
    from feature_engine import build_feature_dataframe
    from labeler         import compute_direction_labels
    from eval_harness    import compute_pnl_metrics
    from models.xgb_head import XGBDirectionHead
    import pandas as pd

    log.info("[%s] loading bars (skip_extract=%s)", symbol, skip_extract)
    bars = _load_bars(symbol, skip_extract)
    log.info("[%s] %d bars after filter", symbol, len(bars))

    log.info("[%s] building 200-dim features", symbol)
    ps = pip_size(symbol)
    dir_feat, _ = build_feature_dataframe(bars, symbol, pip_size=ps)

    log.info("[%s] computing triple-barrier labels", symbol)
    labels, _ = compute_direction_labels(bars)
    labels = labels.astype(np.int8)

    # Use only +1 / -1 (long-profitable / short-profitable) for binary classifier;
    # 0 (FLAT / no clean signal) is excluded from training.
    mask = labels != 0
    X = dir_feat[mask].astype(np.float32)
    y = (labels[mask] > 0).astype(np.int8)   # +1 -> 1, -1 -> 0
    n_eff = len(y)
    log.info("[%s] non-FLAT rows: %d  LONG=%d  SHORT=%d",
             symbol, n_eff, int((y == 1).sum()), int((y == 0).sum()))

    # Chronological 80/20 split with the LABEL_FORWARD_BARS gap.
    n_val  = max(1, int(n_eff * VAL_SPLIT))
    n_tr   = max(1, n_eff - n_val - LABEL_FORWARD_BARS)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr + LABEL_FORWARD_BARS:], y[n_tr + LABEL_FORWARD_BARS:]
    log.info("[%s] split: train=%d  gap=%d  val=%d",
             symbol, n_tr, LABEL_FORWARD_BARS, len(y_va))

    # Train.
    t0 = time.time()
    head = XGBDirectionHead.train(
        X_tr, y_tr, X_va, y_va,
        n_estimators=estimators, max_depth=max_depth, learning_rate=lr,
        seed=seed,
    )
    elapsed = time.time() - t0
    val_acc = head.score(X_va, y_va)
    log.info("[%s] XGB val_acc=%.4f  trained in %.0fs", symbol, val_acc, elapsed)

    # PnL eval — use same forward-return-from-bars approach as the MLP path.
    closes  = bars["close"].to_numpy(dtype=np.float64)
    H       = LABEL_FORWARD_BARS
    fwd     = np.zeros_like(closes)
    fwd[:-H] = (closes[H:] - closes[:-H]) / closes[:-H]
    non_flat_idx = np.where(labels != 0)[0]
    val_bars_idx = non_flat_idx[n_tr + LABEL_FORWARD_BARS:]
    fwd_val = fwd[val_bars_idx]

    proba = head.predict_proba(X_va)
    pred_all  = (proba > 0.5).astype(np.int8)
    pred_conf = np.where(proba > CONF_THRESHOLD, 1,
                         np.where(proba < (1.0 - CONF_THRESHOLD), -1, 0)).astype(np.int8)

    cost = dict(bar="M5", commission=ps, slippage=ps)
    pnl_all  = compute_pnl_metrics(pred=pred_all,  forward_returns=fwd_val, **cost)
    pnl_conf = (compute_pnl_metrics(pred=pred_conf, forward_returns=fwd_val, **cost)
                if int(np.count_nonzero(pred_conf)) >= 50 else None)

    log.info("[%s] all-bars : n=%-6d WR=%.3f  PF=%s  Sharpe=%s  MDD=%.4f",
             symbol, pnl_all.n_trades, pnl_all.win_rate, pnl_all.profit_factor,
             pnl_all.sharpe, pnl_all.max_drawdown)
    if pnl_conf is not None:
        log.info("[%s] conf-only: n=%-6d WR=%.3f  PF=%s  Sharpe=%s   "
                 "(realistic — matches live EA's CONF_THRESHOLD)",
                 symbol, pnl_conf.n_trades, pnl_conf.win_rate,
                 pnl_conf.profit_factor, pnl_conf.sharpe)

    # Top-20 feature importance.
    log.info("[%s] top-20 feature importance (gain):", symbol)
    for fi, gain in head.feature_importance(top_k=20):
        log.info("    f%-3d  gain=%.4f", fi, gain)

    # Export ONNX next to the existing det path. Note: writes the same file
    # name the EA expects, so deploying the XGB model means the EA picks it
    # up on the next ModelWatcher poll.
    onnx_out = ONNX_OUTPUT_DIR / f"HYDRA4_{agent}_{symbol}_dir_det.onnx"
    head.export_onnx(onnx_out, n_features=FEATURE_DIM_DIR)

    # Meta.json for the EA quality gate.
    meta = {
        "agent": agent, "symbol": symbol, "model_class": "xgboost",
        "feat_dim": FEATURE_DIM_DIR, "val_acc": float(val_acc),
        "best_iter": int(head.model.best_iteration),
        "n_train": int(n_tr), "n_val": int(len(y_va)),
        "labeler": "triple_barrier",
        "min_bar_date": MIN_BAR_DATE,
    }
    for k, v in pnl_all.as_dict().items():     meta[f"pnl_{k}"] = v
    if pnl_conf is not None:
        for k, v in pnl_conf.as_dict().items(): meta[f"pnl_conf_{k}"] = v

    p_meta = meta_path(agent, symbol)
    p_meta.parent.mkdir(parents=True, exist_ok=True)
    p_meta.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("[%s] wrote meta: %s", symbol, p_meta.name)

    return {"symbol": symbol, "agent": agent, "val_acc": val_acc,
            "onnx_ok": onnx_out.exists()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    for key, (name, default_syms) in AGENTS_FOR_XGB.items():
        sp = sub.add_parser(key, help=f"Train {name} ({', '.join(default_syms)})")
        sp.add_argument("--symbol", action="append", default=None,
                        help=f"subset of {default_syms}; repeat for several")
        sp.add_argument("--skip-extract", action="store_true")
        sp.add_argument("--estimators", type=int, default=500)
        sp.add_argument("--max-depth", type=int, default=6)
        sp.add_argument("--lr", type=float, default=0.05)
        sp.add_argument("--seed", type=int, default=42)
        sp.set_defaults(_agent_key=key)

    sp_all = sub.add_parser("all", help="Train every default agent + symbol")
    sp_all.add_argument("--agents", nargs="+",
                        choices=list(AGENTS_FOR_XGB.keys()), default=None)
    sp_all.add_argument("--skip-extract", action="store_true")
    sp_all.add_argument("--estimators", type=int, default=500)
    sp_all.add_argument("--max-depth", type=int, default=6)
    sp_all.add_argument("--lr", type=float, default=0.05)
    sp_all.add_argument("--seed", type=int, default=42)

    args = p.parse_args(argv)

    # Build the work list.
    if args.command == "all":
        agent_keys = args.agents or list(AGENTS_FOR_XGB.keys())
        work: list[tuple[str, str]] = []
        for k in agent_keys:
            name, syms = AGENTS_FOR_XGB[k]
            for s in syms:
                work.append((s, name))
    else:
        key = args._agent_key
        name, default_syms = AGENTS_FOR_XGB[key]
        syms = args.symbol or default_syms
        work = [(s, name) for s in syms]

    failed: list[str] = []
    for symbol, agent_name in work:
        try:
            _train_one(symbol, agent_name,
                       estimators=args.estimators, max_depth=args.max_depth,
                       lr=args.lr, skip_extract=args.skip_extract,
                       seed=args.seed)
        except Exception as e:
            log.exception("[%s] failed: %s", symbol, e)
            failed.append(symbol)

    if failed:
        log.error("Failed: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
