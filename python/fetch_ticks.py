"""
fetch_ticks.py — pull historical tick data from MetaTrader 5 into parquet.

The MT5 Python API exposes two tick-history calls:
  mt5.copy_ticks_from(symbol, datetime_from, count, flags)
  mt5.copy_ticks_range(symbol, datetime_from, datetime_to, flags)

flags (mql5 doc):
  COPY_TICKS_INFO  — only changes in bid/ask
  COPY_TICKS_TRADE — only ticks resulting from trades (most informative)
  COPY_TICKS_ALL   — every tick (largest volume)

REALITY CHECK (please read before pulling 5 years of ticks)
-----------------------------------------------------------
Tick history depth depends entirely on your broker. Many brokers store
~30 days; some (RoboForex, Pepperstone Pro, Tickmill PRO, certain ECNs)
keep multi-year tick history. Run this in --probe mode first:

    python python/fetch_ticks.py probe EURUSD

It will fetch a single tick from progressively older dates and report
the oldest one your broker still serves.

Tick volume is huge: a major FX pair averages 100k-1M ticks/day, so
5 years per symbol is roughly 200M-2B rows. Plan storage and bandwidth
accordingly. Compressed parquet (zstd) is ~3-4 bytes/tick = 0.6-8 GB
per symbol per 5 years. Don't try to upload that to Kaggle.

USAGE
-----
    # Probe how far back ticks are available:
    python python/fetch_ticks.py probe EURUSD

    # Fetch a fixed window:
    python python/fetch_ticks.py fetch EURUSD --from 2024-01-01 --to 2024-12-31

    # Fetch as much as the broker has, in 30-day chunks (resumable):
    python python/fetch_ticks.py fetch EURUSD --to 2025-12-31 --chunk-days 30

Output: data/ticks/<SYMBOL>_<YYYY-MM-DD>_<YYYY-MM-DD>.parquet
        Schema: time (us), bid, ask, last, volume, flags
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, ALL_SYMBOLS

log = logging.getLogger(__name__)
TICK_DIR = DATA_DIR / "ticks"


def _mt5():
    """Lazy MT5 import; fails clearly on Linux/Colab where the module isn't available."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise SystemExit(
            "MetaTrader5 module not available. Tick fetching requires:\n"
            "  - Windows host running MT5 terminal,\n"
            "  - `pip install MetaTrader5`,\n"
            "  - terminal logged in to a broker.\n"
            "On Linux/Colab use a Windows machine to fetch and copy the parquet over."
        )
    return mt5


def _connect(mt5) -> bool:
    if not mt5.initialize():
        log.error("mt5.initialize() failed: %s", mt5.last_error())
        return False
    info = mt5.terminal_info()
    log.info("MT5 connected: build=%s server=%s", info.build,
             getattr(mt5.account_info(), "server", "?"))
    return True


def _disconnect(mt5):
    mt5.shutdown()


def _ticks_to_df(arr) -> pd.DataFrame:
    """Convert mt5 tick struct array to a typed DataFrame."""
    if arr is None or len(arr) == 0:
        return pd.DataFrame(columns=["time", "bid", "ask", "last", "volume", "flags"])
    df = pd.DataFrame(arr)
    # mt5 returns 'time' as epoch seconds (int) and 'time_msc' as ms.
    # Use time_msc when present for sub-second resolution; fall back to time.
    if "time_msc" in df.columns:
        df["time"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        df = df.drop(columns=["time_msc"])
    else:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    keep = [c for c in ("time", "bid", "ask", "last", "volume", "flags") if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# probe — find the oldest tick the broker will serve
# ---------------------------------------------------------------------------

def cmd_probe(args: argparse.Namespace) -> int:
    mt5 = _mt5()
    if not _connect(mt5):
        return 1
    try:
        # Walk back in 6-month strides until we hit a 0-row response.
        symbol = args.symbol
        now = datetime.now(timezone.utc)
        oldest_found = None
        for years_back in (1, 2, 3, 5, 8, 12, 18, 25):
            probe_at = now - timedelta(days=365 * years_back)
            ticks = mt5.copy_ticks_from(symbol, probe_at, 1, mt5.COPY_TICKS_ALL)
            n = 0 if ticks is None else len(ticks)
            print(f"  {years_back:>2}y back ({probe_at.date()}): {n} tick(s) returned")
            if n > 0:
                oldest_found = probe_at
            else:
                break
        if oldest_found:
            print(f"\nBroker serves at least back to: {oldest_found.date()}")
        else:
            print("\nNo tick history available for this symbol. Check symbol name + login.")
        return 0
    finally:
        _disconnect(mt5)


# ---------------------------------------------------------------------------
# fetch — pull a date range, optionally in resumable chunks
# ---------------------------------------------------------------------------

def _out_path(symbol: str, start: datetime, end: datetime) -> Path:
    TICK_DIR.mkdir(parents=True, exist_ok=True)
    return TICK_DIR / f"{symbol}_{start.date()}_{end.date()}.parquet"


def _fetch_chunk(mt5, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Single mt5.copy_ticks_range call. Returns an empty df on failure."""
    arr = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
    if arr is None:
        log.warning("[%s %s..%s] mt5 returned None — last_error=%s",
                    symbol, start.date(), end.date(), mt5.last_error())
        return pd.DataFrame()
    df = _ticks_to_df(arr)
    return df


def cmd_fetch(args: argparse.Namespace) -> int:
    mt5 = _mt5()
    if not _connect(mt5):
        return 1
    try:
        symbol = args.symbol
        # Resolve date window (UTC).
        end = (datetime.fromisoformat(args.to_).replace(tzinfo=timezone.utc)
               if args.to_ else datetime.now(timezone.utc))
        start = (datetime.fromisoformat(args.from_).replace(tzinfo=timezone.utc)
                 if args.from_ else end - timedelta(days=365))

        if start >= end:
            print(f"--from must be < --to (got {start} >= {end})")
            return 2

        chunk_days = max(1, args.chunk_days)
        cur = start
        total_rows = 0
        files_written: list[Path] = []
        while cur < end:
            chunk_end = min(cur + timedelta(days=chunk_days), end)
            log.info("[%s] chunk %s -> %s", symbol, cur.date(), chunk_end.date())
            df = _fetch_chunk(mt5, symbol, cur, chunk_end)
            if not df.empty:
                p = _out_path(symbol, cur, chunk_end)
                df.to_parquet(p, index=False, compression="zstd",
                              compression_level=3)
                files_written.append(p)
                total_rows += len(df)
                log.info("  wrote %s  (%d rows, %.1f MB)",
                         p.name, len(df), p.stat().st_size / 1e6)
            else:
                log.info("  (empty chunk)")
            cur = chunk_end

        print()
        print(f"{symbol}: {total_rows:,} ticks across {len(files_written)} parquet files")
        for p in files_written[-5:]:
            print(f"  {p.name}")
        if len(files_written) > 5:
            print(f"  ... ({len(files_written) - 5} more)")
        return 0
    finally:
        _disconnect(mt5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_probe = sub.add_parser("probe",
        help="Report the oldest tick the broker will serve for a symbol.")
    sp_probe.add_argument("symbol", choices=ALL_SYMBOLS)
    sp_probe.set_defaults(func=cmd_probe)

    sp_fetch = sub.add_parser("fetch",
        help="Pull a tick window into data/ticks/<SYM>_<from>_<to>.parquet.")
    sp_fetch.add_argument("symbol", choices=ALL_SYMBOLS)
    sp_fetch.add_argument("--from", dest="from_", default=None,
                          help="ISO-8601 start (UTC). Default: 1 year before --to.")
    sp_fetch.add_argument("--to",   dest="to_",   default=None,
                          help="ISO-8601 end (UTC). Default: now.")
    sp_fetch.add_argument("--chunk-days", type=int, default=30,
                          help="Per-call chunk size in days (default 30). "
                               "Smaller = safer for brokers that throttle "
                               "large tick requests.")
    sp_fetch.set_defaults(func=cmd_fetch)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
