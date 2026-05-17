"""
train_strategies.py — master driver for GOLD-only H1+H4+H5+H6 training.

Runs the four hypothesis trainers in sequence (or any subset) and prints a
combined deploy table. Designed to be the single entrypoint that cloud
runner.sh invokes when TRAIN_MODE=strategies.

  H1 — tick-level order-flow imbalance (XGBoost, ONNX-exported)
  H4 — long-horizon trend rule (MA-cross / momentum, deterministic)
  H5 — intraday scalp inside H4 trend (rule)
  H6 — GOLD-only intraday mean-reversion (z-score on H1 log close, rule)

Each hypothesis emits its own per-symbol meta JSON + (optionally) ONNX
under onnx_out/. This driver also writes a combined manifest
onnx_out/M4GOLD_STRATEGIES_summary.json so downstream tools (download
bundler, EA model watcher) know what's been deployed.

Usage:
    python python/train_strategies.py                       # all four on GOLD
    python python/train_strategies.py --strategies H4,H6    # subset
    python python/train_strategies.py GOLD --strategies H1  # explicit symbol
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from strategies_common import discover_tick_symbols

log = logging.getLogger(__name__)


# Filename pattern each per-strategy trainer writes a per-symbol meta JSON to.
# The subprocess runner reads these back after the child exits.
META_FILENAME_FMT = {
    "H1": "M4GOLD_H1OF_{sym}_meta.json",
    "H4": "M4GOLD_H4TREND_{sym}_spec.json",
    "H5": "M4GOLD_H5SCALP_{sym}_spec.json",
    "H6": "M4GOLD_H6MR_{sym}_spec.json",
}
SCRIPT_FOR = {
    "H1": "train_h1_orderflow.py",
    "H4": "train_h4_trend.py",
    "H5": "train_h5_scalp_gold.py",
    "H6": "train_h6_mr_gold.py",
}


STRATEGY_RUNNERS = {
    "H1": ("train_h1_orderflow", "H1_OF",
            "tick-level order-flow imbalance"),
    "H4": ("train_h4_trend",     "H4_TREND",
            "long-horizon trend-following"),
    "H5": ("train_h5_scalp_gold","H5_SCALP",
            "intraday scalp inside H4 trend"),
    "H6": ("train_h6_mr_gold",   "H6_MR",
            "GOLD-only mean-reversion (z-score)"),
}


def _data_depth_audit(symbols: list[str]) -> None:
    """
    Quick pre-flight: for each symbol, print tick-file size + span +
    estimated H1-bar count. Flags symbols below the recommended
    minimums:
       <2 years span      -> "thin: chronological val is < 6 months"
       <500 MB tick file  -> "low tick density"
    The 2-year cutoff matches what passes the H4 deploy gate in practice
    (USDJPY clears at 1.7 yr, but everything below ~1 yr fails because
    the val window is < 3 months and trend signals can't accumulate).
    """
    from config import TICKS_DIR
    import pandas as pd  # noqa: F401
    log.info("=" * 78)
    log.info(" DATA DEPTH AUDIT (pre-training)")
    log.info("=" * 78)
    log.info(" %-12s  %-9s  %-8s  %s",
             "Symbol", "Size", "Years", "Span / Notes")
    log.info(" %s", "-" * 74)
    rows = []
    for sym in symbols:
        p = TICKS_DIR / f"HYDRA4_TICKS_{sym}.parquet"
        if not p.exists():
            log.info(" %-12s  %-9s  %-8s  %s", sym, "MISSING", "-",
                     "no tick parquet on disk")
            continue
        size_mb = p.stat().st_size / 1e6
        # Read just the time column to compute span without loading the full
        # file. The parquet metadata layer makes this near-instant.
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(p, columns=["time_msc"]).to_pandas()
            t_min = pd.to_datetime(t["time_msc"].min(), unit="ms", utc=True)
            t_max = pd.to_datetime(t["time_msc"].max(), unit="ms", utc=True)
            years = (t_max - t_min).total_seconds() / (365.25 * 86400)
            span_str = f"{t_min.date()} -> {t_max.date()}"
            notes = []
            if years < 1.0:
                notes.append("VERY THIN — extend extraction to >2 yr")
            elif years < 2.0:
                notes.append("thin — borderline for H4 walk-forward")
            if size_mb < 100:
                notes.append("low tick density")
            note_str = "; ".join(notes)
            log.info(" %-12s  %6.0f MB  %5.1f yr  %s%s",
                     sym, size_mb, years, span_str,
                     f"  [{note_str}]" if note_str else "")
            rows.append((sym, size_mb, years))
        except Exception as e:  # noqa: BLE001
            log.info(" %-12s  %6.0f MB  ?        (couldn't read span: %s)",
                     sym, size_mb, e)
    log.info("=" * 78)
    thin = [r for r in rows if r[2] < 2.0]
    if thin:
        log.info(" Thin-data symbols (<2 years): %s",
                 ", ".join(f"{r[0]} ({r[2]:.1f}yr)" for r in thin))
        log.info(" These are expected to FAIL the H4 walk-forward gate. "
                 "Re-extract with extended history if you want them to compete.")
    log.info("")


def _kwargs_to_cli(strategy_key: str, kwargs: dict) -> list[str]:
    """Convert the strategy's Python kwargs into CLI flags for the
    standalone per-strategy script. Anything not in this map is dropped
    silently — defaults inside the per-strategy script will apply."""
    cli: list[str] = []
    if strategy_key == "H1":
        if "ticks_per_bar" in kwargs:
            cli += ["--ticks-per-bar", str(kwargs["ticks_per_bar"])]
        if "horizon" in kwargs:
            cli += ["--horizon", str(kwargs["horizon"])]
        if "n_estimators" in kwargs:
            cli += ["--estimators", str(kwargs["n_estimators"])]
        if "max_depth" in kwargs:
            cli += ["--max-depth", str(kwargs["max_depth"])]
        if "seed" in kwargs:
            cli += ["--seed", str(kwargs["seed"])]
        if kwargs.get("use_gpu"):
            cli += ["--use-gpu"]
        if "max_tick_file_gb" in kwargs:
            cli += ["--max-tick-file-gb", str(kwargs["max_tick_file_gb"])]
    elif strategy_key == "H5":
        if "pullback_k" in kwargs:
            cli += ["--pullback-k", str(kwargs["pullback_k"])]
        if "sl_atr" in kwargs:
            cli += ["--sl-atr", str(kwargs["sl_atr"])]
        if "tp_atr" in kwargs:
            cli += ["--tp-atr", str(kwargs["tp_atr"])]
        if "timeout_bars" in kwargs:
            cli += ["--timeout-bars", str(kwargs["timeout_bars"])]
    elif strategy_key == "H6":
        if "z_window" in kwargs:
            cli += ["--z-window", str(kwargs["z_window"])]
        if "z_in" in kwargs:
            cli += ["--z-in", str(kwargs["z_in"])]
        if "z_out" in kwargs:
            cli += ["--z-out", str(kwargs["z_out"])]
        if "z_stop" in kwargs:
            cli += ["--z-stop", str(kwargs["z_stop"])]
        if "timeout_bars" in kwargs:
            cli += ["--timeout-bars", str(kwargs["timeout_bars"])]
    elif strategy_key == "H4":
        if "timeframe" in kwargs:
            cli += ["--timeframe", str(kwargs["timeframe"])]
        if "fast" in kwargs:
            cli += ["--fast", str(kwargs["fast"])]
        if "slow" in kwargs:
            cli += ["--slow", str(kwargs["slow"])]
        if "mom_lookbacks" in kwargs:
            cli += ["--mom-lookbacks",
                    ",".join(str(x) for x in kwargs["mom_lookbacks"])]
        if kwargs.get("allow_short") is False:
            cli += ["--no-short"]
        if "vol_filter_ratio" in kwargs:
            cli += ["--vol-filter-ratio", str(kwargs["vol_filter_ratio"])]
        if "vol_filter_short" in kwargs:
            cli += ["--vol-filter-short", str(kwargs["vol_filter_short"])]
        if "vol_filter_long" in kwargs:
            cli += ["--vol-filter-long",  str(kwargs["vol_filter_long"])]
        if "seed" in kwargs:
            cli += ["--seed", str(kwargs["seed"])]
    return cli


def _subprocess_timeout_seconds(strategy_key: str) -> int:
    # Per-strategy timeout. H1 chews on every tick in the file (1.06 GB
    # GOLD = 173M ticks -> ~17 min just to aggregate before any training)
    # so it needs a much longer ceiling than the rule-based strategies.
    # Env override: M4GOLD_<KEY>_TIMEOUT_MIN (e.g. M4GOLD_H1_TIMEOUT_MIN=180).
    defaults_min = {"H1": 120, "H4": 30, "H5": 30, "H6": 30}
    env_key = f"M4GOLD_{strategy_key}_TIMEOUT_MIN"
    raw = os.environ.get(env_key)
    if raw:
        try:
            return int(float(raw) * 60)
        except ValueError:
            log.warning("ignoring %s=%r (not a number)", env_key, raw)
    return defaults_min.get(strategy_key, 30) * 60


def _run_subprocess(strategy_key: str, symbol: str, kwargs: dict) -> dict:
    """
    Spawn a fresh `python <script>.py SYMBOL <flags>` per (strategy, symbol).

    Full OS-level memory isolation: when the subprocess exits, ALL its
    pandas/numpy allocations are reclaimed regardless of glibc malloc's
    behaviour. The parent stays a lightweight orchestrator. After the
    subprocess writes its meta/spec JSON to onnx_out/, we load + return it.

    Per-strategy timeout — H1's tick-bar pipeline can take well over the
    legacy 30-min cap on a 1+ GB tick file; see _subprocess_timeout_seconds.
    """
    from config import ONNX_OUTPUT_DIR
    repo_root = Path(__file__).parent.parent
    script = repo_root / "python" / SCRIPT_FOR[strategy_key]
    cli = _kwargs_to_cli(strategy_key, kwargs)
    cmd = [sys.executable, "-u", str(script), symbol, *cli]
    timeout_s = _subprocess_timeout_seconds(strategy_key)
    log.info("[%s:%s] subprocess (timeout=%d min): %s",
             strategy_key, symbol, timeout_s // 60,
             " ".join(c for c in cmd if "/" not in c and "\\" not in c)
             or "python ...")
    # Stream the child's stdout to the parent live; don't capture so the
    # user sees per-symbol progress in real time.
    proc = subprocess.run(cmd, timeout=timeout_s, check=False)
    meta_name = META_FILENAME_FMT[strategy_key].format(sym=symbol)
    meta_path = ONNX_OUTPUT_DIR / meta_name
    if not meta_path.exists():
        return {"strategy": STRATEGY_RUNNERS[strategy_key][1],
                "symbol":  symbol, "ok": False,
                "reason":  f"subprocess rc={proc.returncode} but no "
                            f"{meta_name} written"}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"strategy": STRATEGY_RUNNERS[strategy_key][1],
                "symbol":  symbol, "ok": False,
                "reason":  f"meta JSON unreadable: {e}"}


def _worker_train(strategy_key: str, symbol: str, kwargs: dict) -> dict:
    """
    Top-level worker callable so ProcessPoolExecutor can pickle it.

    Re-initialises a stdout logger inside the child with a per-symbol prefix
    so logs interleave cleanly when N workers run side-by-side. Returns the
    meta dict the per-strategy trainer wrote (or an error dict if it raised).
    """
    import importlib
    import logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{strategy_key}:{symbol}] %(message)s",
        stream=sys.stdout, force=True,
    )
    module_name = STRATEGY_RUNNERS[strategy_key][0]
    mod = importlib.import_module(module_name)
    try:
        return mod.train_one_symbol(symbol, **{k: v for k, v in kwargs.items()
                                                if v is not None})
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).exception("worker crashed")
        return {"strategy": STRATEGY_RUNNERS[strategy_key][1],
                "symbol": symbol, "ok": False, "reason": str(e)}


def _run_one(strategy_key: str, symbols: list[str], extra_args: dict,
              n_workers: int = 1) -> list[dict]:
    """
    Run the trainer for one strategy across `symbols`.

    n_workers == 1 -> sequential (in-process, cheapest, no pickling overhead).
    n_workers >  1 -> ProcessPoolExecutor with `spawn` (Windows-safe), one
                       worker per symbol up to the cap.
    """
    module_name, _, _ = STRATEGY_RUNNERS[strategy_key]
    log.info("=" * 78)
    log.info(" %s — %s  (symbols=%d, workers=%d)",
             strategy_key, STRATEGY_RUNNERS[strategy_key][2],
             len(symbols), max(1, n_workers))
    log.info("=" * 78)
    rows: list[dict] = []
    t0 = time.time()

    worker_args = dict(extra_args)
    # Cap XGBoost intra-op threads so workers don't over-subscribe a 4-core box.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if n_workers <= 1:
        # mk4.8.3: subprocess.run per (strategy, symbol). The parent stays
        # a lightweight orchestrator (~50 MB resident); each subprocess
        # gets the full 30 GB RAM allowance and is OS-reclaimed on exit.
        # Beats ProcessPoolExecutor(max_tasks_per_child=1) because that
        # executor's PARENT accumulates state — the 15:30 UTC Kaggle run
        # died with BrokenProcessPool after symbol 4 even with recycled
        # workers. A fresh subprocess has zero coupling to the parent.
        for i, sym in enumerate(symbols, start=1):
            row = _run_subprocess(strategy_key, sym, worker_args)
            deploy = row.get("deploy", False)
            status = "DEPLOY" if deploy else ("FAIL" if not row.get("ok", True)
                                                else "blocked")
            log.info("[%s] (%d/%d) done %-12s  %s%s",
                     strategy_key, i, len(symbols), sym, status,
                     f"  reason={row.get('reason')}" if row.get("reason") else "")
            rows.append(row)
        log.info("[%s] all symbols done in %.0fs (%.1fs/sym avg)",
                 strategy_key, time.time() - t0,
                 (time.time() - t0) / max(1, len(symbols)))
        return rows

    # workers > 1: ProcessPoolExecutor path (RunPod/vast.ai/Lambda with
    # >40 GB RAM). Kept for users who explicitly opt in.
    ctx = multiprocessing.get_context("spawn")
    max_w = min(n_workers, len(symbols))
    try:
        ex = ProcessPoolExecutor(max_workers=max_w, mp_context=ctx,
                                  max_tasks_per_child=1)
    except TypeError:
        log.warning("ProcessPoolExecutor lacks max_tasks_per_child; "
                     "memory accumulation possible across symbols.")
        ex = ProcessPoolExecutor(max_workers=max_w, mp_context=ctx)
    WORKER_TIMEOUT = _subprocess_timeout_seconds(strategy_key)
    with ex:
        futures = {ex.submit(_worker_train, strategy_key, sym, worker_args): sym
                   for sym in symbols}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            done += 1
            try:
                row = fut.result(timeout=WORKER_TIMEOUT)
            except Exception as e:  # noqa: BLE001
                log.exception("[%s:%s] worker raised", strategy_key, sym)
                row = {"strategy": STRATEGY_RUNNERS[strategy_key][1],
                       "symbol":  sym, "ok": False,
                       "reason":  f"worker exception: {type(e).__name__}: {e}"}
            deploy = row.get("deploy", False)
            status = "DEPLOY" if deploy else ("FAIL" if not row.get("ok", True)
                                                else "blocked")
            log.info("[%s] (%d/%d) done %-12s  %s%s",
                     strategy_key, done, len(symbols), sym, status,
                     f"  reason={row.get('reason')}" if row.get("reason") else "")
            rows.append(row)

    log.info("[%s] all symbols done in %.0fs (%.1fs/sym avg)",
             strategy_key, time.time() - t0,
             (time.time() - t0) / max(1, len(symbols)))
    return rows


def _per_strategy_kwargs(strategy_key: str, args) -> dict:
    """Convert master-CLI args into strategy-specific keyword args."""
    if strategy_key == "H1":
        return {"ticks_per_bar":    args.h1_ticks_per_bar,
                "horizon":           args.h1_horizon,
                "n_estimators":      args.estimators,
                "max_depth":         args.max_depth,
                "seed":              args.seed,
                "use_gpu":           bool(args.use_gpu),
                "max_tick_file_gb":  args.h1_max_tick_file_gb}
    if strategy_key == "H5":
        return {"pullback_k":   args.h5_pullback_k,
                "sl_atr":       args.h5_sl_atr,
                "tp_atr":       args.h5_tp_atr,
                "timeout_bars": args.h5_timeout}
    if strategy_key == "H6":
        return {"z_window":     args.h6_z_window,
                "z_in":         args.h6_z_in,
                "z_out":        args.h6_z_out,
                "z_stop":       args.h6_z_stop,
                "timeout_bars": args.h6_timeout}
    if strategy_key == "H4":
        # H4 is rule-only (no XGBoost), so use_gpu is ignored.
        mom_lookbacks = [int(s) for s in args.h4_mom_lookbacks.split(",")
                          if s.strip()]
        return {"timeframe":         args.h4_timeframe,
                "fast":              args.h4_fast,
                "slow":              args.h4_slow,
                "mom_lookbacks":     mom_lookbacks,
                "allow_short":       not args.h4_no_short,
                "vol_filter_ratio":  args.h4_vol_filter_ratio,
                "vol_filter_short":  args.h4_vol_filter_short,
                "vol_filter_long":   args.h4_vol_filter_long,
                "seed":              args.seed}
    return {}


def _print_combined(by_strategy: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 92)
    print("  COMBINED STRATEGIES SUMMARY")
    print("=" * 92)
    print(f"  {'Strat':<5}  {'Symbol':<12}  {'metric_a':<18}  {'metric_b':<18}  "
          f"{'excess':>9}  Deploy")
    print(f"  {'-'*88}")
    for sk, rows in by_strategy.items():
        for r in rows:
            sym = r.get("symbol", "?")
            if sk == "H1":
                pf = r.get("best_pf"); n = r.get("best_n_trades", 0)
                direction = r.get("best_direction", "?")
                a_str = f"PF={pf:.3f}({direction[:3]})" if pf is not None else "PF=fail"
                b_str = f"N={int(n):>5d}"
                ex = r.get("excess_vs_passive", float("nan"))
            elif sk in ("H5", "H6"):
                pf = r.get("val_pf")
                wf = r.get("wf_consistency")
                a_str = f"PF={pf:.3f}" if pf is not None else "PF=fail"
                b_str = f"WF={wf:.2f}" if wf is not None else "WF=fail"
                ex = r.get("excess_vs_passive", float("nan"))
            elif sk == "H4":
                best_rule_name = r.get("best_rule")
                if best_rule_name and "summary" in r:
                    v = r["summary"][best_rule_name]["val"]
                    a_str = f"Sharpe={v['sharpe']:+.2f}"
                    b_str = f"MDD={100*v['mdd']:.1f}%"
                    ex = v["sharpe"] - 0.6   # cushion above 0.6 Sharpe gate
                else:
                    a_str = "Sharpe=fail"; b_str = "MDD=fail"
                    ex = float("nan")
            else:
                a_str = b_str = "?"; ex = float("nan")
            deploy = r.get("deploy", False)
            reason = ""
            if not deploy:
                if "reason" in r:
                    reason = f"  ({r['reason']})"
            print(f"  {sk:<5}  {sym:<12}  {a_str:<18}  {b_str:<18}  "
                  f"{ex:>+9.3f}  {'YES' if deploy else 'no':<3}{reason}")
    print("=" * 92)
    n_total = sum(len(v) for v in by_strategy.values())
    n_deploy = sum(1 for v in by_strategy.values() for r in v if r.get("deploy"))
    print(f"  {n_deploy}/{n_total} (strategy, symbol) cells deployable\n")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbols", nargs="*",
                   help="symbols (default: all with tick parquet)")
    p.add_argument("--all", action="store_true",
                   help="train on every symbol with a tick parquet")
    p.add_argument("--strategies", default="H1,H4,H5,H6",
                   help="comma-separated subset of H1,H4,H5,H6 (default: all)")
    # Shared XGBoost knobs
    p.add_argument("--estimators", type=int, default=300)
    p.add_argument("--max-depth",  type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    # H1 knobs
    p.add_argument("--h1-ticks-per-bar", type=int, default=100)
    p.add_argument("--h1-horizon",       type=int, default=10)
    p.add_argument("--h1-max-tick-file-gb", type=float, default=8.0,
                   help="skip H1 for tick files larger than this. Default 8.0 "
                        "fits any modern desktop / RunPod / Lambda; set to 1.0 "
                        "on Kaggle's free tier to pre-empt the 30 GB RAM OOM.")
    # H5 knobs
    p.add_argument("--h5-pullback-k", type=float, default=1.0)
    p.add_argument("--h5-sl-atr",     type=float, default=0.7)
    p.add_argument("--h5-tp-atr",     type=float, default=1.5)
    p.add_argument("--h5-timeout",    type=int,   default=12)
    # H6 knobs (GOLD-only mean-reversion)
    p.add_argument("--h6-z-window",   type=int,   default=200)
    p.add_argument("--h6-z-in",       type=float, default=2.0)
    p.add_argument("--h6-z-out",      type=float, default=0.5)
    p.add_argument("--h6-z-stop",     type=float, default=3.5)
    p.add_argument("--h6-timeout",    type=int,   default=48)
    # H4 knobs
    p.add_argument("--h4-timeframe", choices=("1h", "4h", "1d"), default="1h")
    p.add_argument("--h4-fast",      type=int, default=50)
    p.add_argument("--h4-slow",      type=int, default=200)
    p.add_argument("--h4-mom",       type=int, default=240,
                   help="legacy single-lookback; prefer --h4-mom-lookbacks")
    p.add_argument("--h4-mom-lookbacks", type=str, default="120,240",
                   help="comma-separated H4 momentum lookbacks (default 120,240)")
    p.add_argument("--h4-no-short",  action="store_true",
                   help="long-only for H4 (skip short side)")
    p.add_argument("--h4-vol-filter-ratio", type=float, default=0.0,
                   help="H4 vol-regime filter (0.0=off; 0.85-0.95 = chop guard)")
    p.add_argument("--h4-vol-filter-short", type=int, default=20)
    p.add_argument("--h4-vol-filter-long",  type=int, default=500)
    p.add_argument("--audit-first", action="store_true",
                   help="run audit_strategies.py before training (recommended)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel symbol workers per strategy. Default 1 "
                        "(sequential — safe on Kaggle's 30 GB RAM watchdog). "
                        "Set 2-4 on platforms with >40 GB RAM (RunPod / vast.ai / "
                        "Lambda); Kaggle nukes the kernel on >24 GB usage.")
    p.add_argument("--use-gpu", action="store_true",
                   help="GPU XGBoost (H1 + H2 meta). Modest RAM relief — the "
                        "training matrices + model move to VRAM (~50-200 MB). "
                        "Does NOT fix the tick-bar RAM bottleneck.")
    args = p.parse_args(argv)

    strategies = [s.strip().upper() for s in args.strategies.split(",") if s.strip()]
    bad = [s for s in strategies if s not in STRATEGY_RUNNERS]
    if bad:
        log.error("unknown strategies: %s — choose from %s",
                  bad, list(STRATEGY_RUNNERS))
        return 2

    # m4Gold is single-symbol by design. The user can still pass an explicit
    # symbol on the CLI (mostly for debugging), otherwise default to GOLD.
    symbols = args.symbols or ["GOLD"]
    if not all(s == "GOLD" for s in symbols):
        log.warning("m4Gold is single-symbol; non-GOLD inputs %s will be "
                    "trained but the EA dispatcher only honours GOLD.", symbols)
    log.info("strategies=%s  symbols=%s", strategies, symbols)

    # mk4.8.4: pre-training data-depth report — surfaces thin-data symbols
    # (notably BTCUSD with only ~4 months on disk) so the user understands
    # why those cells will fail before training even starts.
    _data_depth_audit(symbols)

    if args.audit_first:
        log.info("running audit_strategies before training ...")
        import audit_strategies
        rc = audit_strategies.main([])
        if rc != 0:
            log.error("audit failed (rc=%d). Aborting before training.", rc)
            return rc

    t0 = time.time()
    by_strategy: dict[str, list[dict]] = {}
    fatal_error: str | None = None
    try:
        for sk in strategies:
            by_strategy[sk] = _run_one(sk, symbols,
                                        _per_strategy_kwargs(sk, args),
                                        n_workers=args.workers)
    except Exception as e:  # noqa: BLE001
        log.exception("strategy loop crashed — flushing partial results")
        fatal_error = f"{type(e).__name__}: {e}"

    # Print whatever we have — even if the run was interrupted, partial
    # cells get the same summary treatment.
    if by_strategy:
        _print_combined(by_strategy)

    # Combined manifest — always emit, even on partial / crashed runs, so
    # downstream tools can pick up what completed.
    from config import ONNX_OUTPUT_DIR
    manifest_path = ONNX_OUTPUT_DIR / "M4GOLD_STRATEGIES_summary.json"
    manifest = {
        "elapsed_seconds": time.time() - t0,
        "strategies":      strategies,
        "symbols":         symbols,
        "n_workers":       int(args.workers),
        "args":            vars(args),
        "fatal_error":     fatal_error,
        "results":         by_strategy,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float))
    log.info("combined manifest -> %s", manifest_path.name)

    # Return 0 even if some symbols failed — "zero deploys" or "N failures"
    # is a valid answer the runner should bundle and surface, not a script
    # failure. Only return non-zero if we caught a top-level fatal crash.
    return 0 if fatal_error is None else 1


if __name__ == "__main__":
    sys.exit(main())
