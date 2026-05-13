"""
fetch_deep_history.py — Phase 2: Deep historical bar data for HYDRA mk4.

MT5 typically holds 2-5 years of M5 data.  Training on 10+ years increases
regime diversity, improving generalisation across market cycles.

Source priority (per symbol):
  1. MetaTrader 5 (live, full quality — spread, volume, exact timestamps)
  2. yfinance     (free, goes back 10-30 years for most instruments)
  3. Dukascopy CSV (manual download fallback for Forex pairs)

Output:
  Merged, deduplicated, sorted, validated Parquet — same schema as
  run_full_pipeline() so _train_agent.py needs no changes.

Usage:
    python fetch_deep_history.py                    # all symbols
    python fetch_deep_history.py EURUSD GOLD        # specific symbols
    python fetch_deep_history.py --check            # validate existing cache
    python fetch_deep_history.py --source yf        # yfinance only
"""

import argparse
import hashlib
import logging
import sys
import time
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALL_SYMBOLS, PARQUET_DIR, FOREX_SYMBOLS, METALS_SYMBOLS,
    INDICES_SYMBOLS, CE_SYMBOLS, parquet_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical → yfinance ticker map
# ---------------------------------------------------------------------------

# Primary yfinance ticker per canonical symbol.
# Futures tickers (=F) are tried first; ETF fallbacks tried on empty result.
_YF_TICKER: Dict[str, str] = {
    # Forex
    "EURUSD":     "EURUSD=X",
    "GBPUSD":     "GBPUSD=X",
    "USDJPY":     "USDJPY=X",
    # Metals — continuous futures primary
    "GOLD":       "GC=F",
    "SILVER":     "SI=F",
    "PLATINUM":   "PL=F",
    "COPPER":     "HG=F",
    # Indices — cash index (longest history, no roll gap)
    "US_500":     "^GSPC",
    # mk4.8: NAS100 row dropped — broker doesn't quote NAS100; no roster
    # includes it.
    "UK_100":     "^FTSE",
    # Crypto
    "BTCUSD":     "BTC-USD",
    "ETHUSD":     "ETH-USD",
    "LTCUSD":     "LTC-USD",
    # Energy — continuous futures primary
    "CrudeOIL":   "CL=F",
    "BRENT_OIL":  "BZ=F",
    "NATURAL_GAS":"NG=F",
}

# ETF/proxy fallback tickers tried when the primary returns no data.
# ETFs have longer, more reliably served Yahoo history than futures roll contracts.
_YF_FALLBACK: Dict[str, str] = {
    "GOLD":        "GLD",    # SPDR Gold Shares
    "SILVER":      "SLV",    # iShares Silver Trust
    "PLATINUM":    "PPLT",   # abrdn Physical Platinum ETF
    "COPPER":      "CPER",   # US Copper Index ETF
    "CrudeOIL":    "USO",    # United States Oil Fund
    "BRENT_OIL":   "BNO",    # United States Brent Oil Fund
    "NATURAL_GAS": "UNG",    # United States Natural Gas Fund
    "US_500":      "SPY",    # SPDR S&P 500 ETF
    # mk4.8: NAS100 fallback dropped — no upstream roster includes it.
}

# ---------------------------------------------------------------------------
# Dukascopy CSV schema (for manual-download fallback)
# Dukascopy export format: Time (UTC),Open,High,Low,Close,Volume
# ---------------------------------------------------------------------------

_DUKA_COLS = ["time", "open", "high", "low", "close", "tick_volume"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase and ensure required columns exist."""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {
        "vol": "tick_volume", "volume": "tick_volume",
        "date": "time", "datetime": "time", "timestamp": "time",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 1.0
    if "spread" not in df.columns:
        df["spread"] = 0.0
    return df


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and localise the 'time' column to UTC-aware datetime."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    return df


def _validate_bars(df: pd.DataFrame, symbol: str,
                   max_gap_mult: float = 5.0) -> pd.DataFrame:
    """
    Bar continuity validation.

    Checks:
    1. No duplicate timestamps.
    2. Monotonically increasing time.
    3. No OHLC nulls or zero prices.
    4. No abnormal gaps (> max_gap_mult × median bar spacing).
    5. High >= Low, High >= Open/Close, Low <= Open/Close.

    Suspicious rows are DROPPED and logged.  Returns cleaned DataFrame.
    """
    original = len(df)
    df = df.copy()

    # Sort by time
    df = df.sort_values("time").reset_index(drop=True)

    # Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates(subset=["time"])
    if len(df) < before_dup:
        log.warning("%s: dropped %d duplicate timestamps", symbol, before_dup - len(df))

    # Drop null/zero prices
    price_cols = ["open", "high", "low", "close"]
    null_mask = df[price_cols].isnull().any(axis=1) | (df[price_cols] == 0).any(axis=1)
    if null_mask.sum():
        log.warning("%s: dropped %d rows with null/zero prices", symbol, null_mask.sum())
        df = df[~null_mask].reset_index(drop=True)

    # OHLC integrity: high must be the highest, low must be the lowest
    bad_hl = (df["high"] < df["low"]) | (df["high"] < df["open"]) | \
             (df["high"] < df["close"]) | (df["low"] > df["open"]) | \
             (df["low"] > df["close"])
    if bad_hl.sum():
        log.warning("%s: dropped %d rows with bad OHLC integrity", symbol, bad_hl.sum())
        df = df[~bad_hl].reset_index(drop=True)

    # Gap detection: flag gaps > max_gap_mult × median spacing
    if len(df) > 2:
        deltas = df["time"].diff().dt.total_seconds().dropna()
        positive_deltas = deltas[deltas > 0]
        if not positive_deltas.empty:
            median_gap = positive_deltas.median()
            threshold  = max_gap_mult * median_gap
            large_gaps = deltas[deltas > threshold]
            if not large_gaps.empty:
                log.info("%s: %d abnormal time gaps detected (> %.0fs = %.1fx median %.0fs)",
                         symbol, len(large_gaps), threshold,
                         max_gap_mult, median_gap)
                for idx in large_gaps.index[:5]:   # log first 5
                    t_prev = df["time"].iloc[idx - 1]
                    t_curr = df["time"].iloc[idx]
                    gap_h  = deltas.iloc[idx] / 3600
                    log.info("  Gap at %s → %s  (%.1f hours)", t_prev, t_curr, gap_h)

    removed = original - len(df)
    if removed:
        log.info("%s: validation removed %d/%d rows — %d clean",
                 symbol, removed, original, len(df))
    else:
        log.info("%s: validation OK — %d bars", symbol, len(df))

    return df.reset_index(drop=True)


def _merge_sources(existing: Optional[pd.DataFrame],
                   new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two bar DataFrames:
    - existing rows take priority on overlap (keep last after sort)
    - new_data fills gaps and extends history
    - result is sorted and deduplicated; caller is responsible for validation
    """
    if existing is None or existing.empty:
        return new_data.sort_values("time").drop_duplicates(
            subset=["time"]).reset_index(drop=True)

    combined = pd.concat([new_data, existing], ignore_index=True)
    combined = combined.sort_values("time")
    combined = combined.drop_duplicates(subset=["time"], keep="last")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source 1 — MetaTrader 5
# ---------------------------------------------------------------------------

def _fetch_mt5(symbol: str, max_bars: int = 5_000_000) -> Optional[pd.DataFrame]:
    try:
        from data_pipeline import connect, fetch_bars, disconnect
        import MetaTrader5 as mt5_mod
    except ImportError:
        log.warning("MetaTrader5 not available — skipping MT5 source")
        return None
    try:
        connect()
        df = fetch_bars(symbol, mt5_mod.TIMEFRAME_M5, n_bars=max_bars)
        return df
    except Exception as e:
        log.warning("MT5 fetch failed for %s: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# Source 2 — yfinance
# ---------------------------------------------------------------------------

def _yf_normalise(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalise a raw yfinance DataFrame to the standard OHLCV schema."""
    if raw is None or raw.empty:
        return None
    raw = raw.copy()
    # Flatten MultiIndex columns (yfinance ≥0.2 style)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw.reset_index()
    # Lowercase again: reset_index() promotes the DatetimeIndex as 'Date' or 'Datetime'
    # (capital first letter), which would not match the rename dict below.
    raw.columns = [str(c).lower() for c in raw.columns]
    raw.rename(columns={"index": "time", "datetime": "time",
                         "volume": "tick_volume", "date": "time"}, inplace=True)
    if "time" not in raw.columns:
        return None
    raw = _ensure_utc(raw)
    keep = [c for c in ["time", "open", "high", "low", "close", "tick_volume"]
            if c in raw.columns]
    raw = raw[keep].copy()
    if "tick_volume" not in raw.columns:
        raw["tick_volume"] = 0.0
    raw["spread"] = 0.0
    for col in ["open", "high", "low", "close", "tick_volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["open", "high", "low", "close"])
    return raw if not raw.empty else None


def _fetch_yfinance(symbol: str,
                    years_back: int = 10) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from yfinance.

    Yahoo Finance imposes a hard 730-day limit on 1h interval data.
    Strategy:
      - For history older than 729 days: use interval="1d" (daily bars, no limit).
      - For the last 729 days:           use interval="1h" (hourly bars).
    Both frames are merged; 1h bars take priority on the overlap period.

    Ticker priority:
      1. Primary ticker from _YF_TICKER (usually a futures contract, e.g. GC=F).
      2. ETF fallback from _YF_FALLBACK (e.g. GLD) if primary returns no data.
         ETFs have more reliably served Yahoo history than roll contracts.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed. Run: pip install yfinance")
        return None

    primary  = _YF_TICKER.get(symbol)
    fallback = _YF_FALLBACK.get(symbol)
    tickers  = [t for t in [primary, fallback] if t]

    if not tickers:
        log.warning("No yfinance ticker mapping for %s — skipping", symbol)
        return None

    now       = dt.datetime.now(dt.timezone.utc)
    date_to   = now.strftime("%Y-%m-%d")
    date_deep = (now - dt.timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    date_1h   = (now - dt.timedelta(days=729)).strftime("%Y-%m-%d")

    for ticker in tickers:
        frames = []

        # --- Daily bars: full years_back range (no Yahoo interval restriction) ---
        if years_back > 2:
            try:
                log.info("yfinance: %s (%s) 1d  %s → %s",
                         symbol, ticker, date_deep, date_to)
                raw_d = yf.download(ticker, start=date_deep, end=date_to,
                                    interval="1d", auto_adjust=True, progress=False)
                df_d = _yf_normalise(raw_d)
                if df_d is not None:
                    log.info("yfinance: %s 1d  %d bars (%s → %s)",
                             symbol, len(df_d),
                             str(df_d["time"].min())[:10], str(df_d["time"].max())[:10])
                    frames.append(df_d)
                else:
                    log.warning("yfinance returned no 1d data for %s (%s)",
                                symbol, ticker)
            except Exception as exc:
                log.warning("yfinance 1d fetch error for %s (%s): %s",
                            symbol, ticker, exc)

        # --- Hourly bars: last 729 days (always within Yahoo's allowed window) ---
        try:
            log.info("yfinance: %s (%s) 1h  %s → %s",
                     symbol, ticker, date_1h, date_to)
            raw_h = yf.download(ticker, start=date_1h, end=date_to,
                                interval="1h", auto_adjust=True, progress=False)
            df_h = _yf_normalise(raw_h)
            if df_h is not None:
                log.info("yfinance: %s 1h  %d bars (%s → %s)",
                         symbol, len(df_h),
                         str(df_h["time"].min())[:10], str(df_h["time"].max())[:10])
                frames.append(df_h)
            else:
                log.warning("yfinance returned no 1h data for %s (%s)",
                            symbol, ticker)
        except Exception as exc:
            log.warning("yfinance 1h fetch error for %s (%s): %s",
                        symbol, ticker, exc)

        if not frames:
            if ticker == primary and fallback:
                log.warning("yfinance: %s — primary ticker %s returned no data, "
                            "trying ETF fallback %s", symbol, ticker, fallback)
                continue   # try next ticker in the loop
            log.warning("yfinance: %s — no data from any ticker", symbol)
            return None

        # Merge: 1h bars override 1d on overlap (appended last → kept by keep="last")
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("time")
        combined = combined.drop_duplicates(subset=["time"], keep="last")
        combined = combined.reset_index(drop=True)
        log.info("yfinance: %s (%s) combined %d bars (%s → %s)",
                 symbol, ticker, len(combined),
                 str(combined["time"].min())[:10], str(combined["time"].max())[:10])
        return combined

    return None


# ---------------------------------------------------------------------------
# Source 3 — Dukascopy CSV (manual download fallback)
# ---------------------------------------------------------------------------

def _fetch_dukascopy(symbol: str,
                     csv_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load Dukascopy tick/bar CSV files from a directory.

    Expected file naming convention (Dukascopy export):
        <SYMBOL>_<TIMEFRAME>_*.csv   e.g. EURUSD_M5_2015.csv

    Dukascopy CSV format (default export):
        Gmt time,Open,High,Low,Close,Volume
    or:
        Time (UTC),Open,High,Low,Close,Volume

    Place downloaded CSVs in: MT5_bot_mk4/data/dukascopy/
    """
    from config import DATA_DIR
    csv_dir = csv_dir or (DATA_DIR / "dukascopy")
    if not csv_dir.exists():
        log.debug("Dukascopy dir not found: %s", csv_dir)
        return None

    # Find matching CSVs
    patterns = [f"{symbol}_*.csv", f"{symbol.lower()}_*.csv",
                f"{symbol}*.csv"]
    files = []
    for pat in patterns:
        files.extend(csv_dir.glob(pat))
    if not files:
        log.debug("No Dukascopy CSVs found for %s in %s", symbol, csv_dir)
        return None

    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, sep=",")
            df.columns = [c.strip().lower() for c in df.columns]
            # Normalise time column
            for tc in ["gmt time", "time (utc)", "time", "datetime", "date"]:
                if tc in df.columns:
                    df.rename(columns={tc: "time"}, inplace=True)
                    break
            # Normalise volume
            for vc in ["volume", "vol"]:
                if vc in df.columns:
                    df.rename(columns={vc: "tick_volume"}, inplace=True)
                    break
            df = _ensure_utc(df)
            dfs.append(df)
            log.info("Dukascopy: loaded %s — %d rows", f.name, len(df))
        except Exception as e:
            log.warning("Dukascopy: failed to load %s — %s", f.name, e)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined["spread"] = 0.0
    return combined


# ---------------------------------------------------------------------------
# Continuity report
# ---------------------------------------------------------------------------

def check_cache(symbols: Optional[List[str]] = None):
    """Print a continuity and freshness report for cached Parquet files."""
    symbols = symbols or ALL_SYMBOLS
    print(f"\n{'='*70}")
    print("  HYDRA mk4 — Bar Cache Continuity Report")
    print(f"{'='*70}")
    print(f"  {'Symbol':<14} {'Bars':>10} {'From':>12} {'To':>12} "
          f"{'Gaps':>6} {'Status'}")
    print("  " + "-" * 65)

    for sym in symbols:
        # Find any parquet for this symbol
        found = sorted(PARQUET_DIR.glob(f"HYDRA4_FEAT_{sym}_*.parquet"))
        if not found:
            print(f"  {sym:<14} {'—':>10} {'—':>12} {'—':>12} {'—':>6}  MISSING")
            continue
        p = found[-1]   # most recent
        try:
            df = pd.read_parquet(p)
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time")
            n = len(df)
            t_from = str(df["time"].iloc[0])[:10]
            t_to   = str(df["time"].iloc[-1])[:10]

            # Gap detection
            deltas = df["time"].diff().dt.total_seconds().dropna()
            pos = deltas[deltas > 0]
            if not pos.empty:
                median_s = pos.median()
                gaps = int((deltas > 5 * median_s).sum())
            else:
                gaps = 0

            # Freshness
            last_dt = df["time"].iloc[-1]
            if hasattr(last_dt, "to_pydatetime"):
                last_dt = last_dt.to_pydatetime()
            age_h = (dt.datetime.now(dt.timezone.utc) - last_dt).total_seconds() / 3600
            status = "OK" if (gaps == 0 and age_h < 48) else \
                     f"STALE({age_h:.0f}h)" if age_h >= 48 else f"GAPS({gaps})"

            print(f"  {sym:<14} {n:>10,} {t_from:>12} {t_to:>12} "
                  f"{gaps:>6}  {status}")
        except Exception as e:
            print(f"  {sym:<14} {'ERROR':>10}  {e}")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main fetch + merge function
# ---------------------------------------------------------------------------

def fetch_deep(symbol: str,
               years_back: int = 10,
               use_mt5: bool = True,
               use_yf: bool = True,
               use_duka: bool = True,
               duka_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Fetch deep history for one symbol from all available sources,
    merge, validate, and return a clean DataFrame.
    """
    frames = []

    if use_mt5:
        df_mt5 = _fetch_mt5(symbol)
        if df_mt5 is not None and not df_mt5.empty:
            log.info("%s MT5: %d bars", symbol, len(df_mt5))
            frames.append(("MT5", df_mt5))

    if use_yf:
        df_yf = _fetch_yfinance(symbol, years_back=years_back)
        if df_yf is not None and not df_yf.empty:
            log.info("%s yfinance: %d bars", symbol, len(df_yf))
            frames.append(("yfinance", df_yf))

    if use_duka:
        df_dk = _fetch_dukascopy(symbol, csv_dir=duka_dir)
        if df_dk is not None and not df_dk.empty:
            log.info("%s Dukascopy: %d bars", symbol, len(df_dk))
            frames.append(("Dukascopy", df_dk))

    if not frames:
        log.error("No data obtained for %s from any source", symbol)
        return None

    # Priority merge: MT5 > Dukascopy > yfinance
    # (MT5 has spread/volume quality; yf fills deep history gaps)
    priority = {"MT5": 3, "Dukascopy": 2, "yfinance": 1}
    frames.sort(key=lambda x: priority.get(x[0], 0), reverse=True)

    merged = None
    for src, df in frames:
        try:
            df_std = _std_cols(df)
            df_utc = _ensure_utc(df_std)
            merged = _merge_sources(merged, df_utc)
            log.info("%s after merging %s: %d bars", symbol, src, len(merged))
        except Exception as e:
            log.warning("Error merging %s data for %s: %s", src, symbol, e)

    if merged is None or merged.empty:
        return None

    # Single validation pass on the fully-merged result
    merged = _validate_bars(merged, symbol)
    log.info("%s FINAL: %d bars  (%s → %s)",
             symbol, len(merged),
             str(merged["time"].min())[:10],
             str(merged["time"].max())[:10])
    return merged


def save_deep(df: pd.DataFrame, symbol: str):
    """Save deep history Parquet to the standard cache location."""
    p = parquet_path(symbol, len(df))
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, compression="snappy")
    # Write metadata sidecar
    meta = {
        "symbol":    symbol,
        "n_bars":    len(df),
        "date_from": str(df["time"].min())[:10] if "time" in df.columns else "",
        "date_to":   str(df["time"].max())[:10] if "time" in df.columns else "",
        "sha256":    hashlib.sha256(
                         pd.util.hash_pandas_object(df, index=False)
                         .values.tobytes()
                     ).hexdigest()[:16],
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    import json
    meta_p = p.with_suffix(".json")
    meta_p.write_text(json.dumps(meta, indent=2))
    log.info("Saved: %s  (%d bars)", p.name, len(df))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HYDRA mk4 — Deep History Fetcher")
    parser.add_argument("symbols", nargs="*",
                        help="Canonical symbol names (default: all)")
    parser.add_argument("--check", action="store_true",
                        help="Print continuity report for existing cache and exit")
    parser.add_argument("--source", choices=["all", "mt5", "yf", "duka"],
                        default="all", help="Restrict to one data source")
    parser.add_argument("--years", type=int, default=10,
                        help="Years of history to request from yfinance (default: 10)")
    parser.add_argument("--duka-dir", type=str, default=None,
                        help="Path to directory containing Dukascopy CSV files")
    args = parser.parse_args()

    symbols = args.symbols or ALL_SYMBOLS
    invalid = [s for s in symbols if s not in ALL_SYMBOLS]
    if invalid:
        print(f"Unknown symbols: {invalid}.  Valid: {ALL_SYMBOLS}")
        sys.exit(1)

    if args.check:
        check_cache(symbols)
        return

    use_mt5  = args.source in ("all", "mt5")
    use_yf   = args.source in ("all", "yf")
    use_duka = args.source in ("all", "duka")
    duka_dir = Path(args.duka_dir) if args.duka_dir else None

    print(f"\n{'='*60}")
    print("  HYDRA mk4 — Deep History Fetch")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Sources : MT5={use_mt5}  yfinance={use_yf}  Dukascopy={use_duka}")
    print(f"  Years   : {args.years}")
    print(f"{'='*60}\n")

    t0 = time.time()
    ok, failed = [], []

    for sym in symbols:
        log.info("─── %s ───", sym)
        try:
            df = fetch_deep(sym, years_back=args.years,
                            use_mt5=use_mt5, use_yf=use_yf,
                            use_duka=use_duka, duka_dir=duka_dir)
            if df is not None and len(df) > 0:
                save_deep(df, sym)
                ok.append(sym)
            else:
                log.error("%s: no data after all sources", sym)
                failed.append(sym)
        except Exception as e:
            log.exception("Fetch failed for %s: %s", sym, e)
            failed.append(sym)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"  OK     : {', '.join(ok) if ok else 'none'}")
    print(f"  FAILED : {', '.join(failed) if failed else 'none'}")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
