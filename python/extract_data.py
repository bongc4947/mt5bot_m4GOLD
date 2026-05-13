"""
extract_data.py — Pull historical data from MetaTrader 5, cache to Parquet.

Usage:
    python extract_data.py                                  # all symbols, M5 bars
    python extract_data.py EURUSD GBPUSD                    # specific symbols, M5 bars
    python extract_data.py --list                           # list cached files
    python extract_data.py EURUSD --source ticks            # tick-mode extraction
    python extract_data.py EURUSD --source ticks --max-size-mb 1024 --ticks-per-bar 100

Tick-mode caches both raw-ish bars (aggregated to N-tick "tick-bars")
and (optionally with --save-raw-ticks) the underlying tick stream for
the random-window trainer. See PRODUCTION_GAPS.md → "tick-mode training".

Requires MetaTrader 5 to be running and logged in.
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from config import ALL_SYMBOLS, PARQUET_DIR
from hardware_detector import get as get_hw


def _list_cache():
    print(f"\nParquet cache: {PARQUET_DIR}\n")
    any_found = False
    for sym in ALL_SYMBOLS:
        # M5 bar caches
        for p in sorted(PARQUET_DIR.glob(f"HYDRA4_FEAT_{sym}_*.parquet")):
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.name:<55}  {size_mb:.1f} MB")
            any_found = True
        # Tick-bar caches
        for p in sorted(PARQUET_DIR.glob(f"HYDRA4_TBARS_{sym}_*.parquet")):
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.name:<55}  {size_mb:.1f} MB")
            any_found = True
        if not any(PARQUET_DIR.glob(f"HYDRA4_*_{sym}_*.parquet")):
            print(f"  {sym:<10}  [not cached]")
    if not any_found:
        print("  No parquet files found. Run extract_data.py to populate cache.")
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbols", nargs="*",
                   help=f"symbols to extract (default: all {len(ALL_SYMBOLS)})")
    p.add_argument("--list", action="store_true",
                   help="list cached parquets and exit")
    p.add_argument("--source", choices=["bars", "ticks"], default="bars",
                   help="bars (M5) or ticks (raw → N-tick bars). default=bars")
    p.add_argument("--max-size-mb", type=int, default=2048,
                   help="tick-mode: per-symbol size cap in MB. default=2048")
    p.add_argument("--ticks-per-bar", type=int, default=100,
                   help="tick-mode: ticks aggregated into one bar. default=100")
    p.add_argument("--chunk-days", type=int, default=30,
                   help="tick-mode: fetch chunk size in days. default=30")
    p.add_argument("--save-raw-ticks", action="store_true",
                   help="tick-mode: also save raw tick parquet "
                        "(needed by random-window trainer)")
    p.add_argument("--bundle", action="store_true",
                   help="after extraction, zip data/parquet/*.parquet into "
                        "data/HYDRA4_data_bundle.zip ready to upload to "
                        "Kaggle / Drive / S3 for cloud notebook training")
    return p


def main():
    args = _build_parser().parse_args()

    if args.list:
        _list_cache()
        return

    symbols = args.symbols or ALL_SYMBOLS
    invalid = [s for s in symbols if s not in ALL_SYMBOLS]
    if invalid:
        print(f"ERROR: Unknown symbols: {invalid}")
        print(f"Valid symbols: {ALL_SYMBOLS}")
        sys.exit(1)

    hw = get_hw()

    print(f"\n{'='*60}")
    print(f"  HYDRA mk4 — Data Extraction ({args.source} mode)")
    print(f"  Symbols  : {', '.join(symbols)}")
    if args.source == "bars":
        print(f"  Max bars : {hw.max_bars:,}")
    else:
        print(f"  Cap MB   : {args.max_size_mb}  per symbol")
        print(f"  Ticks/bar: {args.ticks_per_bar}")
        print(f"  Save raw : {args.save_raw_ticks}")
    print(f"{'='*60}\n")

    from data_pipeline import connect, disconnect, run_full_pipeline, run_tick_pipeline

    print("Connecting to MetaTrader 5...")
    terminal_path = os.environ.get("MT5_TERMINAL_PATH", "")
    if not connect(path=terminal_path):
        sys.exit(1)
    print("MT5 connected.\n")

    t0 = time.time()
    if args.source == "bars":
        results = run_full_pipeline(symbols=symbols, max_bars=hw.max_bars)
    else:
        results = run_tick_pipeline(symbols=symbols,
                                     max_size_mb=args.max_size_mb,
                                     ticks_per_bar=args.ticks_per_bar,
                                     chunk_days=args.chunk_days,
                                     save_raw_ticks=args.save_raw_ticks)
    disconnect()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    for sym in symbols:
        df = results.get(sym)
        if df is not None and len(df) > 0:
            from_dt = str(df["time"].iloc[0])[:10] if "time" in df.columns else "?"
            to_dt   = str(df["time"].iloc[-1])[:10] if "time" in df.columns else "?"
            unit    = "bars" if args.source == "bars" else "tick-bars"
            print(f"  {sym:<10}  {len(df):>10,} {unit}  {from_dt} -> {to_dt}  OK")
        else:
            print(f"  {sym:<10}  FAILED — no data returned")
    print(f"\n  Done in {elapsed:.0f}s. Parquet cache: {PARQUET_DIR}")
    print(f"{'='*60}\n")

    if args.bundle:
        # Single optional flag → zip everything in data/parquet/ ready for
        # upload. Reuses bundle_data.cmd_bundle so there's one bundling
        # implementation, not two.
        import argparse as _ap
        from bundle_data import cmd_bundle
        print(f"\n{'='*60}")
        print(f"  Bundling parquet cache → data/HYDRA4_data_bundle.zip")
        print(f"{'='*60}")
        cmd_bundle(_ap.Namespace(out=None))


if __name__ == "__main__":
    main()
