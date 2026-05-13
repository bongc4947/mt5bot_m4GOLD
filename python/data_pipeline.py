"""
data_pipeline.py — Extract data from MT5 and write Parquet caches.
Handles bars, ticks, deal history, EA signal logs, and backtest reports.
"""

import re
import csv
import json
import logging
import hashlib
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from config import (
    ALL_SYMBOLS, MT5_COMMON_DIR, PARQUET_DIR,
    parquet_path, signal_log_path, closed_trades_log_path,
    mod_events_log_path,
)

log = logging.getLogger(__name__)


def _detect_csv_encoding(path: Path) -> str:
    """
    Detect file encoding from BOM bytes.
    MT5 on Windows writes CSV files as UTF-16 LE (BOM = FF FE).
    Falls back to utf-8 when no BOM is found.
    """
    with open(path, "rb") as f:
        bom = f.read(4)
    if bom[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    if bom[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    return "utf-8"


# ---------------------------------------------------------------------------
# MT5 connection helpers
# ---------------------------------------------------------------------------

def _mt5():
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        raise ImportError("MetaTrader5 Python package not installed. "
                          "Run: pip install MetaTrader5")


def _find_mt5_terminal() -> Optional[str]:
    """
    Search common installation paths for the MT5 terminal executable.
    Returns the path string if found, else None.
    """
    import glob as _glob

    candidates = [
        # AppData roaming (most common)
        r"C:\Users\{user}\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe",
        r"C:\Users\{user}\AppData\Roaming\MetaQuotes\Terminal\*\terminal.exe",
        # Program Files
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files\MetaTrader 5\terminal.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
    ]

    import os
    username = os.environ.get("USERNAME", os.environ.get("USER", ""))

    for pattern in candidates:
        expanded = pattern.replace("{user}", username)
        matches = _glob.glob(expanded)
        if matches:
            # Prefer terminal64.exe; pick newest modified if multiple
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

    return None


def connect(login: int = 0, server: str = "", password: str = "",
            path: str = "") -> bool:
    mt5 = _mt5()

    import os as _os

    # Attempt order:
    #   1. No path   — attaches to already-running MT5 (correct when MT5 is open)
    #   2. Env var   — user-specified path override
    #   3. Auto-find — searches Program Files / AppData for terminal64.exe
    attempts: list = [{}]

    explicit = path or _os.environ.get("MT5_TERMINAL_PATH", "")
    if explicit:
        attempts.append({"path": explicit})

    detected = _find_mt5_terminal()
    if detected and detected != explicit:
        attempts.append({"path": detected})

    last_error = None
    for kwargs in attempts:
        desc = kwargs.get("path", "<running instance>")
        log.info("MT5 initialize() → %s", desc)

        if mt5.initialize(**kwargs):
            if login:
                if not mt5.login(login, password=password, server=server):
                    log.error("MT5 login failed: %s", mt5.last_error())
                    mt5.shutdown()
                    return False
            info = mt5.terminal_info()
            log.info("Connected: %s  build=%s", info.name, info.build)
            return True

        last_error = mt5.last_error()
        log.warning("MT5 initialize() failed [%s]: %s", desc, last_error)

    log.error("All MT5 connection attempts failed. Last error: %s", last_error)
    # Error -6 = terminal found but refused connection (AutoTrading disabled)
    if last_error and last_error[0] == -6:
        _print_mt5_auth_help()
    else:
        _print_mt5_help(detected)
    return False


def _print_mt5_auth_help():
    print()
    print("  ── MT5 Error -6: Terminal refused connection ──────────────────")
    print()
    print("  MT5 is running but blocking the Python API. Fix:")
    print()
    print("  STEP 1 — Enable AutoTrading in the MT5 TOOLBAR")
    print("    Look for the 'AutoTrading' button in the top toolbar.")
    print("    It must be GREEN. Click it if it is grey or red.")
    print()
    print("  STEP 2 — Enable in Options (if not already done)")
    print("    Tools -> Options -> Expert Advisors")
    print("      ✓  Allow algorithmic trading")
    print("      ✓  Allow DLL imports")
    print("    Click OK.")
    print()
    print("  STEP 3 — Restart MT5 completely")
    print("    Close MT5, reopen it, log in, wait for charts to load.")
    print("    Then retry:  python extract_data.py")
    print()
    print("  If it still fails after restart:")
    print("    • Run both MT5 and this terminal as the same user (not Admin)")
    print("    • pip install --upgrade MetaTrader5")
    print("  ──────────────────────────────────────────────────────────────")
    print()


def _print_mt5_help(detected_path: Optional[str]):
    print()
    print("  ── MT5 connection failed ─────────────────────────────────────")
    print()
    print("  Most likely cause (MT5 is already running):")
    print("  -> Python API is not enabled in MT5.")
    print()
    print("  Fix:")
    print("    1. In MT5: Tools → Options → Expert Advisors")
    print("         ✓  Allow algorithmic trading")
    print("         ✓  Allow DLL imports (recommended)")
    print("    2. Click OK, then retry.")
    print()
    print("  Other causes:")
    print()
    print("  • MT5 is not running at all")
    print("    -> Open MetaTrader 5 and log in, then retry.")
    print()
    print("  • First-time API connection requires manual approval")
    print("    -> Check the MT5 taskbar icon for a permission prompt and accept it.")
    print()
    print("  • Running Python as Administrator but MT5 is not (or vice versa)")
    print("    -> Run both at the same privilege level.")
    print()
    print("  • MetaTrader5 Python package version does not match terminal build")
    print("    -> pip install --upgrade MetaTrader5")
    print()
    if detected_path:
        print(f"  Detected terminal : {detected_path}")
    else:
        print("  terminal64.exe    : not found in default locations")
    print("  ──────────────────────────────────────────────────────────────")
    print()


def disconnect():
    _mt5().shutdown()


# ---------------------------------------------------------------------------
# Bar extraction
# ---------------------------------------------------------------------------

def _resolve_broker_symbol(canonical: str) -> Optional[str]:
    """
    Return broker symbol name for a canonical name.
    Resolution order:
      1. Exact match
      2. Common broker suffixes (.pro, .ecn, #, etc.)
      3. Known canonical → broker alias map (GOLD→XAUUSD, etc.)
      4. Fuzzy scan: search all broker symbols for one that contains canonical
         (or whose description contains it), added to Market Watch if found
    """
    mt5 = _mt5()

    # Known canonical → common broker alias variants
    _ALIASES: Dict[str, List[str]] = {
        "GOLD":        ["XAUUSD", "XAUUSDm", "GOLD", "GOLDm", "XAUUSD.pro"],
        "SILVER":      ["XAGUSD", "XAGUSDm", "SILVER", "SILVERm", "XAGUSD.pro"],
        "PLATINUM":    ["XPTUSD", "XPTUSDm", "PLATINUM", "PLATINUMm"],
        "COPPER":      ["XCUUSD", "XCUUSDm", "COPPER", "COPPERm", "HG"],
        "US_500":      ["US500", "SPX500", "SP500", "US500m", "SP500m", "US.500", "USA500"],
        # mk4.8: NAS100 alias row removed — broker doesn't quote NAS100 on
        # the active terminal; no tick parquet is extracted for it.
        "UK_100":      ["UK100", "FTSE100", "UK100m", "GB100"],
        "CrudeOIL":    ["USOIL", "WTI", "CRUDE", "CL", "OILm", "USOILm"],
        "BRENT_OIL":   ["UKOIL", "BRENT", "BRENTOIL", "OIL.UK", "UKOILm"],
        "NATURAL_GAS": ["NGAS", "NATGAS", "NG", "NGASm", "GAS"],
    }

    # 1. Exact match
    info = mt5.symbol_info(canonical)
    if info is not None:
        mt5.symbol_select(canonical, True)
        return canonical

    # 2. Common suffixes
    for suffix in [".pro", ".ecn", "#", ".r", "m", "+", ".raw", ".s"]:
        name = canonical + suffix
        info = mt5.symbol_info(name)
        if info is not None:
            mt5.symbol_select(name, True)
            return name

    # 3. Known aliases
    for alias in _ALIASES.get(canonical, []):
        info = mt5.symbol_info(alias)
        if info is not None:
            mt5.symbol_select(alias, True)
            log.info("Resolved %s → %s (alias map)", canonical, alias)
            return alias

    # 4. Fuzzy scan: get all available symbols and look for substring match
    all_syms = mt5.symbols_get()
    if all_syms:
        canon_upper = canonical.upper()
        # First pass: symbol name contains canonical
        for s in all_syms:
            if canon_upper in s.name.upper():
                mt5.symbol_select(s.name, True)
                log.info("Resolved %s → %s (fuzzy name match)", canonical, s.name)
                return s.name
        # Second pass: description contains canonical
        for s in all_syms:
            if canon_upper in s.description.upper():
                mt5.symbol_select(s.name, True)
                log.info("Resolved %s → %s (fuzzy description match)", canonical, s.name)
                return s.name

    log.warning("Could not resolve broker symbol for canonical: %s", canonical)
    return None


def fetch_bars(canonical: str, timeframe_mt5, n_bars: int = 1_000_000,
               broker_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch up to n_bars M5 bars for a symbol.
    Returns DataFrame with columns: time, open, high, low, close, tick_volume, spread, real_volume
    """
    mt5 = _mt5()
    bname = broker_name or _resolve_broker_symbol(canonical)
    if bname is None:
        log.warning("Symbol %s not found on broker", canonical)
        return None

    rates = mt5.copy_rates_from_pos(bname, timeframe_mt5, 0, n_bars)
    if rates is None or len(rates) == 0:
        log.warning("No bars for %s: %s", canonical, mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["canonical"] = canonical
    log.info("Fetched %d bars for %s (%s → %s)",
             len(df), canonical, df["time"].iloc[0], df["time"].iloc[-1])
    return df


def fetch_h1_bars(canonical: str, n_bars: int = 50_000,
                  broker_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetch H1 bars for MTF context. n_bars ≈ 6 years at 1H."""
    mt5 = _mt5()
    return fetch_bars(canonical, mt5.TIMEFRAME_H1, n_bars=n_bars,
                      broker_name=broker_name)


def fetch_h4_bars(canonical: str, n_bars: int = 15_000,
                  broker_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetch H4 bars for MTF context. n_bars ≈ 7 years at H4."""
    mt5 = _mt5()
    return fetch_bars(canonical, mt5.TIMEFRAME_H4, n_bars=n_bars,
                      broker_name=broker_name)


def fetch_h8_bars(canonical: str, n_bars: int = 8_000,
                  broker_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetch H8 bars for extended MTF context. n_bars ≈ 7 years at H8."""
    mt5 = _mt5()
    return fetch_bars(canonical, mt5.TIMEFRAME_H8, n_bars=n_bars,
                      broker_name=broker_name)


def fetch_d1_bars(canonical: str, n_bars: int = 3_000,
                  broker_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetch Daily bars for macro MTF context. n_bars ≈ 12 years at D1."""
    mt5 = _mt5()
    return fetch_bars(canonical, mt5.TIMEFRAME_D1, n_bars=n_bars,
                      broker_name=broker_name)


def fetch_ticks(canonical: str, date_from: dt.datetime, date_to: dt.datetime,
                broker_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    mt5 = _mt5()
    bname = broker_name or _resolve_broker_symbol(canonical)
    if bname is None:
        return None

    ticks = mt5.copy_ticks_from(bname, date_from, (date_to - date_from).total_seconds() * 1000,
                                 mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return None

    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["canonical"] = canonical
    return df


# ---------------------------------------------------------------------------
# Tick-mode extraction (mk4.7) — fetch ticks newest-first up to a size cap.
# Used by the --source ticks path of extract_data.py to give the trainer the
# same data shape the live EA sees, instead of pre-aggregated M5 bars.
# ---------------------------------------------------------------------------

# zstd-level-9 compressed tick parquet runs ~14-18 bytes/row in practice.
# We use 16 as a planning estimate when checking the size cap.
_TICK_BYTES_PER_ROW_ESTIMATE = 16


def fetch_ticks_capped(canonical: str,
                        max_size_mb: int = 2048,
                        chunk_days: int = 30,
                        broker_name: Optional[str] = None,
                        max_empty_streak: int = 2,
                        ):
    """
    *Generator* yielding tick chunks newest-first, one chunk per yield.
    Each yielded chunk is a DataFrame in chronological order. Caller
    decides how to consume them (aggregate-on-the-fly, stream-write,
    accumulate, etc.) — keeps peak memory bounded to one chunk.

    Stops when:
      - accumulated rows >= cap derived from `max_size_mb` (output-size
        budget, useful for Kaggle dataset ceilings); or
      - `max_empty_streak` consecutive chunks return zero ticks
        (broker has no more history that far back). Each empty chunk
        is logged so the user can see why extraction stopped.
    """
    mt5 = _mt5()
    bname = broker_name or _resolve_broker_symbol(canonical)
    if bname is None:
        log.warning("fetch_ticks_capped: no broker mapping for %s", canonical)
        return

    cap_rows = int(max_size_mb * 1024 * 1024 / _TICK_BYTES_PER_ROW_ESTIMATE)
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None)
    total_rows = 0
    empty_streak = 0

    while total_rows < cap_rows:
        start = end - dt.timedelta(days=chunk_days)
        ticks = mt5.copy_ticks_range(bname, start, end, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            empty_streak += 1
            log.info("  %s: +0 ticks (chunk %s..%s)  [empty %d/%d]",
                     canonical,
                     start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                     empty_streak, max_empty_streak)
            if empty_streak >= max_empty_streak:
                log.info("  %s: %d consecutive empty chunks — broker has no "
                         "history before %s. Stopping.",
                         canonical, max_empty_streak,
                         end.strftime("%Y-%m-%d"))
                break
            end = start
            continue
        empty_streak = 0
        df = pd.DataFrame(ticks)
        if "time_msc" not in df.columns and "time" in df.columns:
            # MT5 returns 'time' in seconds; we want millisecond resolution
            df["time_msc"] = (df["time"].astype("int64") * 1000)
        total_rows += len(df)
        log.info("  %s: +%d ticks (chunk %s..%s)  total=%d / cap=%d",
                 canonical, len(df),
                 start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                 total_rows, cap_rows)
        yield df
        end = start


def clean_ticks(ticks_df: pd.DataFrame,
                  spread_outlier_mult: float = 5.0) -> pd.DataFrame:
    """
    Sub-phase 1c quality filters:
      - drop exact-duplicate (time_msc, bid, ask) rows (broker tick repeats)
      - drop ticks where spread > `spread_outlier_mult` x rolling-median spread
        (broker glitches, weekend gaps, news spikes that aren't tradable)

    Returns a fresh DataFrame; does not mutate input. Cheap on a single
    chunk; logs how many ticks were dropped.
    """
    if "bid" not in ticks_df.columns or "ask" not in ticks_df.columns:
        return ticks_df

    n0 = len(ticks_df)
    dedupe_keys = [c for c in ("time_msc", "bid", "ask") if c in ticks_df.columns]
    df = ticks_df.drop_duplicates(subset=dedupe_keys, keep="first")
    n_dedup = n0 - len(df)

    spread = (df["ask"] - df["bid"]).to_numpy(dtype=np.float64)
    rolling_med = pd.Series(spread).rolling(1000, min_periods=10).median().to_numpy()
    rolling_med = np.where(rolling_med > 1e-12, rolling_med, np.nanmedian(spread) or 1e-9)
    keep = spread <= rolling_med * spread_outlier_mult
    n_outliers = int((~keep).sum())
    df = df.loc[keep].reset_index(drop=True)

    if n_dedup or n_outliers:
        log.info("  tick clean: dropped %d duplicates, %d spread-outliers (kept %d/%d)",
                 n_dedup, n_outliers, len(df), n0)
    return df


def aggregate_ticks_to_bars(ticks_df: pd.DataFrame,
                             ticks_per_bar: int = 100) -> pd.DataFrame:
    """
    Aggregate a tick DataFrame into fixed-N-tick OHLCV bars.

    Each bar represents `ticks_per_bar` trades' worth of activity, so bars
    are *information-uniform* rather than time-uniform — closer to what
    the live EA sees (it reacts to ticks, not to wall-clock buckets).

    Output schema is OHLCV-compatible with the M5 bar pipeline plus two
    extra columns the trainer can ignore for now but that microstructure
    features can pick up later:
      - spread       : mean(ask - bid) over the bar
      - duration_sec : wall-clock duration of the bar (variable!)
    """
    needed = {"bid", "ask"}
    if not needed.issubset(ticks_df.columns):
        raise ValueError(f"ticks_df missing required columns {needed - set(ticks_df.columns)}")

    # Cheap NaN probe before paying for a dropna copy. MT5-returned ticks
    # very rarely have NaN bid/ask; on a 138 M-row chunk the unconditional
    # dropna copy alone tries to allocate ~9 GB and OOM'd on a 17 GB box.
    if ticks_df["bid"].isna().any() or ticks_df["ask"].isna().any():
        df = ticks_df.dropna(subset=["bid", "ask"]).reset_index(drop=True)
    else:
        df = ticks_df   # zero-copy; the index from pd.DataFrame(ticks) is RangeIndex
    n_bars = len(df) // ticks_per_bar
    if n_bars == 0:
        raise ValueError(f"need >= {ticks_per_bar} ticks; got {len(df)}")

    df = df.iloc[: n_bars * ticks_per_bar]
    mid    = ((df["bid"] + df["ask"]) / 2.0).to_numpy()
    spread = (df["ask"] - df["bid"]).to_numpy()
    vol    = df["volume"].to_numpy() if "volume" in df.columns else np.ones(len(df))
    # time_msc preferred (millisecond), fall back to time (seconds * 1000)
    if "time_msc" in df.columns:
        t_ms = df["time_msc"].to_numpy().astype("int64")
    else:
        t_ms = (df["time"].to_numpy().astype("int64") * 1000)

    mid_t    = mid.reshape(n_bars, ticks_per_bar)
    spread_t = spread.reshape(n_bars, ticks_per_bar)
    vol_t    = vol.reshape(n_bars, ticks_per_bar)
    time_t   = t_ms.reshape(n_bars, ticks_per_bar)

    bars = pd.DataFrame({
        "time":         pd.to_datetime(time_t[:, -1], unit="ms", utc=True),
        "open":         mid_t[:, 0],
        "high":         mid_t.max(axis=1),
        "low":          mid_t.min(axis=1),
        "close":        mid_t[:, -1],
        "tick_volume": vol_t.sum(axis=1).astype("float32"),
        "real_volume": vol_t.sum(axis=1).astype("float32"),
        "spread":       spread_t.mean(axis=1).astype("float32"),
        "duration_sec": ((time_t[:, -1] - time_t[:, 0]) / 1000.0).astype("float32"),
    })
    return bars


def fetch_deal_history(date_from: dt.datetime, date_to: dt.datetime) -> Optional[pd.DataFrame]:
    mt5 = _mt5()
    deals = mt5.history_deals_get(date_from, date_to)
    if deals is None or len(deals) == 0:
        return None
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


# ---------------------------------------------------------------------------
# EA signal log parser
# ---------------------------------------------------------------------------

def parse_signal_log(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse HYDRA4_signals.csv written by the EA.
    Returns DataFrame with per-bar signals and per-trade outcomes.
    """
    path = path or signal_log_path()
    if not path.exists():
        log.info("Signal log not found: %s", path)
        return pd.DataFrame()

    rows = []
    with open(path, newline="", encoding=_detect_csv_encoding(path)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Coerce numeric columns
    num_cols = ["confidence", "uncertainty", "trade_opened", "trade_result_pips",
                "drawdown_at_signal", "entry_price", "sl_pips", "tp_pips",
                "lot_size", "spread_pips", "equity_norm", "dd_pct",
                "exit_price", "pips", "pnl_usd", "hold_bars", "max_drawdown_pips"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("Parsed %d signal log rows from %s", len(df), path)
    return df


def parse_closed_trades_log(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse HYDRA4_closed_trades.csv written by RunLogger.LogClosedTrade().

    Schema (written by EA):
        timestamp, symbol, agent, direction, entry_price, exit_price,
        pips, pnl_usd, lot_size, hold_bars, sl_pips, tp_pips, close_reason

    Returns a clean DataFrame — only rows with a valid symbol and non-NaN pips.
    This is the source of truth for win-rate and PnL calculations in live_monitor.
    """
    path = path or closed_trades_log_path()
    if not path.exists():
        log.debug("Closed trades log not found: %s", path)
        return pd.DataFrame()

    rows = []
    with open(path, newline="", encoding=_detect_csv_encoding(path)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    num_cols = ["direction", "entry_price", "exit_price", "pips",
                "pnl_usd", "lot_size", "hold_bars", "sl_pips", "tp_pips"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without a real symbol or a valid pips value
    if "symbol" in df.columns:
        df = df[df["symbol"].str.strip() != ""]
    if "pips" in df.columns:
        df = df[df["pips"].notna()]

    log.info("Parsed %d closed trades from %s", len(df), path)
    return df


def parse_mod_events_log(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse HYDRA4_mod_events.csv written by RunLogger.LogModEvent().

    Schema:
        timestamp, symbol, ticket, action,
        confidence, floating_pnl_pips, sl_pips_before, sl_pips_after,
        close_now_conf
    """
    path = path or mod_events_log_path()
    if not path.exists():
        log.debug("Mod events log not found: %s", path)
        return pd.DataFrame()

    rows = []
    with open(path, newline="", encoding=_detect_csv_encoding(path)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    num_cols = ["confidence", "floating_pnl_pips",
                "sl_pips_before", "sl_pips_after", "close_now_conf"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    log.info("Parsed %d mod events from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Backtest report parser (MT5 HTML/XML Strategy Tester output)
# ---------------------------------------------------------------------------

def parse_backtest_report(path: Path) -> Dict:
    """
    Parse an MT5 Strategy Tester HTML report into a dict of summary stats
    and a DataFrame of individual trades.
    """
    if not path.exists():
        log.warning("Backtest report not found: %s", path)
        return {}

    text = path.read_text(encoding="utf-8", errors="replace")
    result: Dict = {"path": str(path), "trades": pd.DataFrame()}

    # Extract headline stats via regex
    stats_patterns = {
        "total_trades":  r"Total Trades\s*</td><td[^>]*>([\d]+)",
        "profit_factor": r"Profit Factor\s*</td><td[^>]*>([\d.]+)",
        "expected_pnl":  r"Expected Payoff\s*</td><td[^>]*>([+-]?[\d.]+)",
        "max_dd_pct":    r"Maximal Drawdown\s*</td><td[^>]*>[^<]+([\d.]+)\s*%",
        "win_rate":      r"Win Trades.*?(\d+)\s*\((\d+\.\d+)%\)",
        "sharpe":        r"Sharpe Ratio\s*</td><td[^>]*>([\d.]+)",
    }
    for key, pat in stats_patterns.items():
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                result[key] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

    # Extract trade table rows — simplified
    rows = re.findall(
        r"<tr[^>]*>(?:<td[^>]*>(.*?)</td>){7,}</tr>",
        text, re.DOTALL | re.IGNORECASE
    )
    parsed = []
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(cells) >= 7:
            parsed.append(cells)

    if parsed:
        cols = ["time", "type", "symbol", "volume", "price", "pnl", "balance"]
        try:
            result["trades"] = pd.DataFrame(parsed, columns=cols[:len(parsed[0])])
        except Exception:
            pass

    log.info("Parsed backtest report: %s  trades=%d",
             path.name, len(result.get("trades", pd.DataFrame())))
    return result


# ---------------------------------------------------------------------------
# Parquet cache save / load
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, symbol: str, n_bars: int):
    path = parquet_path(symbol, n_bars)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Downcast float64 → float32: halves RAM at load time, cutting peak training
    # memory from ~12 kB/bar to ~8 kB/bar → 50% more bars for the same RAM.
    float_cols = df.select_dtypes(include="float64").columns
    if len(float_cols):
        df = df.copy()
        df[float_cols] = df[float_cols].astype("float32")
    df.to_parquet(path, index=False, compression="snappy")
    log.info("Saved parquet: %s (%d rows, float32)", path.name, len(df))
    # Delete older parquets for this symbol — only keep the latest extract.
    from config import PARQUET_DIR
    for old in PARQUET_DIR.glob(f"HYDRA4_FEAT_{symbol}_*.parquet"):
        if old != path:
            try:
                old.unlink()
                log.debug("Removed stale parquet: %s", old.name)
            except Exception:
                pass


def load_parquet(symbol: str, n_bars: int) -> Optional[pd.DataFrame]:
    path = parquet_path(symbol, n_bars)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    log.info("Loaded parquet: %s (%d rows)", path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# Compute median spread per symbol (used by MarketHoursEncoder)
# ---------------------------------------------------------------------------

def compute_median_spreads(bars_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    medians = {}
    for sym, df in bars_dict.items():
        if "spread" in df.columns:
            medians[sym] = float(np.median(df["spread"].dropna()))
    return medians


# ---------------------------------------------------------------------------
# Full pipeline: fetch all symbols, save parquet
# ---------------------------------------------------------------------------

def run_full_pipeline(max_bars: int = 1_000_000,
                      symbols: Optional[List[str]] = None,
                      timeframe_mt5=None) -> Dict[str, pd.DataFrame]:
    mt5 = _mt5()
    if timeframe_mt5 is None:
        timeframe_mt5 = mt5.TIMEFRAME_M5

    symbols = symbols or ALL_SYMBOLS
    results = {}
    for sym in symbols:
        df = fetch_bars(sym, timeframe_mt5, n_bars=max_bars)
        if df is not None:
            save_parquet(df, sym, len(df))
            results[sym] = df
        else:
            log.warning("Skipping %s — no data", sym)

    return results


def run_tick_pipeline(symbols: Optional[List[str]] = None,
                       max_size_mb: int = 2048,
                       ticks_per_bar: int = 100,
                       chunk_days: int = 30,
                       save_raw_ticks: bool = False,
                       mtf_features: bool = True,
                       ) -> Dict[str, pd.DataFrame]:
    """
    Tick-mode extraction: fetch ticks per symbol up to `max_size_mb`,
    aggregate into N-tick bars, save as parquet alongside the M5 cache.

    The output bars have the same schema as M5 bars (so the existing
    feature engine and trainer work unchanged), with two extras:
    `spread` and `duration_sec`. The trainer treats these as bar bars;
    the only difference is that "1 bar" represents `ticks_per_bar`
    trades' worth of activity, not 5 minutes of wall-clock.

    Saves to PARQUET_DIR/HYDRA4_TBARS_<SYM>_<N>tpb.parquet; the audit
    and trainer's _find_parquet pick this up alongside HYDRA4_FEAT_*.

    Setting save_raw_ticks=True also writes the unaggregated tick
    parquet to TICKS_DIR/HYDRA4_TICKS_<SYM>.parquet — useful for
    the random-window trainer (sees raw ticks) and for re-aggregating
    at a different ticks_per_bar without refetching.
    """
    from config import tickbars_parquet_path, ticks_parquet_path
    symbols = symbols or ALL_SYMBOLS
    results: Dict[str, pd.DataFrame] = {}

    # Streaming raw-tick writer (only when save_raw_ticks=True). Avoids
    # holding 100M+ ticks in RAM by appending each chunk to the parquet
    # as soon as it's fetched.
    pa = pq = None
    if save_raw_ticks:
        try:
            import pyarrow as pa  # noqa: F811
            import pyarrow.parquet as pq  # noqa: F811
        except ImportError:
            log.warning("save_raw_ticks=True but pyarrow not importable; "
                        "raw-tick streaming disabled")
            save_raw_ticks = False

    for sym in symbols:
        log.info("[%s] tick-mode extraction (cap=%d MB, ticks_per_bar=%d)",
                 sym, max_size_mb, ticks_per_bar)

        bar_chunks: List[pd.DataFrame] = []
        total_ticks = 0
        chunks_seen = 0
        raw_writer = None
        raw_path = ticks_parquet_path(sym) if save_raw_ticks else None
        carry: Optional[pd.DataFrame] = None  # leftover ticks < ticks_per_bar

        try:
            for tick_chunk in fetch_ticks_capped(sym,
                                                  max_size_mb=max_size_mb,
                                                  chunk_days=chunk_days):
                chunks_seen += 1
                # 1c.4: clean (dedupe + spread outliers) BEFORE counting / aggregating.
                tick_chunk = clean_ticks(tick_chunk)
                total_ticks += len(tick_chunk)

                # 1. Stream raw ticks to disk (one parquet file, append-as-we-go).
                if save_raw_ticks:
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    table = pa.Table.from_pandas(tick_chunk, preserve_index=False)
                    if raw_writer is None:
                        raw_writer = pq.ParquetWriter(
                            raw_path, table.schema,
                            compression="zstd", compression_level=9)
                    raw_writer.write_table(table)
                    del table

                # 2. Aggregate this chunk to tick-bars; carry the tail (< 1 bar
                #    worth of ticks) into the next chunk so we don't waste them.
                merged = (pd.concat([carry, tick_chunk], ignore_index=True)
                          if carry is not None else tick_chunk)
                n_full = len(merged) // ticks_per_bar
                if n_full == 0:
                    carry = merged
                    continue
                head = merged.iloc[: n_full * ticks_per_bar]
                carry = (merged.iloc[n_full * ticks_per_bar:].reset_index(drop=True)
                          if len(merged) > n_full * ticks_per_bar else None)
                try:
                    bars_chunk = aggregate_ticks_to_bars(head, ticks_per_bar=ticks_per_bar)
                    # Sub-phase 1b.1-1b.4: order-flow microstructure features
                    # are aggregated from the *same* ticks at the *same* bar
                    # boundaries, then hstacked so each tick-bar has both
                    # OHLCV and orderflow columns in one row.
                    try:
                        from orderflow import aggregate_orderflow_to_bars
                        of_chunk = aggregate_orderflow_to_bars(head, ticks_per_bar=ticks_per_bar)
                        if len(of_chunk) == len(bars_chunk):
                            bars_chunk = pd.concat([bars_chunk.reset_index(drop=True),
                                                    of_chunk.reset_index(drop=True)],
                                                    axis=1)
                    except Exception as e:
                        log.warning("[%s] orderflow agg skipped: %s", sym, e)
                    bar_chunks.append(bars_chunk)
                except ValueError as e:
                    log.warning("[%s] chunk aggregation skipped: %s", sym, e)

                # Drop the raw chunk reference so the GC can reclaim memory
                # before the next fetch. With 10M-row chunks this matters.
                del tick_chunk, merged, head
        finally:
            if raw_writer is not None:
                raw_writer.close()

        if chunks_seen == 0:
            log.warning("[%s] no chunks returned — broker mapping or connection issue", sym)
            continue
        if not bar_chunks:
            log.warning("[%s] only %d ticks fetched (< %d ticks/bar) — skipped",
                        sym, total_ticks, ticks_per_bar)
            continue

        # Bars are 1/ticks_per_bar the size of raw ticks — concat is cheap.
        bars = pd.concat(bar_chunks, ignore_index=True)
        # Chunks arrive newest-first; within a chunk bars are oldest-first,
        # so the concatenated series is *not* time-sorted. Sort once at the
        # end (≈ 1.4M rows on GOLD = sub-second).
        bars = bars.sort_values("time").reset_index(drop=True)

        # mk4.7 sub-phase 1a step 1: bolt H1 + H4 context onto every
        # tick-bar so the model can condition on broader regime without
        # owning the full multi-timeframe stack itself. Strictly causal
        # (backward merge_asof). Failure here is non-fatal — we log and
        # write zero-filled MTF columns so the schema stays consistent.
        if mtf_features:
            try:
                h1_bars_df = fetch_h1_bars(sym, n_bars=20_000)
                h4_bars_df = fetch_h4_bars(sym, n_bars=10_000)
                from multi_timeframe import align_mtf_features
                bars = align_mtf_features(bars, h1_bars_df, h4_bars_df)
                log.info("[%s] MTF features aligned (h1 n=%s, h4 n=%s)",
                         sym,
                         len(h1_bars_df) if h1_bars_df is not None else 0,
                         len(h4_bars_df) if h4_bars_df is not None else 0)
            except Exception as e:
                log.warning("[%s] MTF feature alignment failed: %s — writing "
                            "tick-bars without H1/H4 context", sym, e)
                from multi_timeframe import align_mtf_features
                bars = align_mtf_features(bars, None, None)

        # Sub-phase 1c.1: time / session / calendar features. Always on,
        # always cheap, no external data — just date math from the bar's
        # `time` column.
        try:
            from session_features import compute_session_features, SESSION_FEATURE_COLUMNS
            sess = compute_session_features(bars)
            for c in SESSION_FEATURE_COLUMNS:
                bars[c] = sess[c].to_numpy()
        except Exception as e:
            log.warning("[%s] session feature computation failed: %s", sym, e)

        out_path = tickbars_parquet_path(sym, ticks_per_bar)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(out_path, compression="zstd", compression_level=9)
        mb = out_path.stat().st_size / 1e6
        log.info("[%s] wrote tick-bars: %s (%.1f MB, %d bars from %d ticks)",
                 sym, out_path.name, mb, len(bars), total_ticks)

        if save_raw_ticks and raw_path and raw_path.exists():
            mb_raw = raw_path.stat().st_size / 1e6
            log.info("[%s] wrote raw ticks: %s (%.1f MB, %d ticks streamed)",
                     sym, raw_path.name, mb_raw, total_ticks)

        results[sym] = bars

    return results


# ---------------------------------------------------------------------------
# Metadata JSON
# ---------------------------------------------------------------------------

def build_bar_metadata(df: pd.DataFrame, symbol: str) -> Dict:
    return {
        "symbol":    symbol,
        "n_bars":    len(df),
        "date_from": str(df["time"].iloc[0]) if "time" in df.columns else "",
        "date_to":   str(df["time"].iloc[-1]) if "time" in df.columns else "",
        "sha256":    hashlib.sha256(df.to_json().encode()).hexdigest()[:16],
    }
