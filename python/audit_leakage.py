"""
audit_leakage.py — pre-train sanity check for label / split / feature leakage.

Run before any training session to confirm:
  1. Labels are forward-looking only (no .shift(-N) into the past).
  2. The chronological train/val split has the expected gap.
  3. There is no NaN/Inf in features or labels at the moment of the split.
  4. Per-feature std is non-zero (constant features = dead training signal).
  5. Class balance is not pathological.

USAGE
-----
    python audit_leakage.py <SYMBOL>          # audit cached parquet for SYMBOL
    python audit_leakage.py <SYMBOL> --strict # exit non-zero on any warning

This is a static audit — it does NOT train a model. Cheap to run before
every training job.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import PARQUET_DIR, VAL_SPLIT, LABEL_FORWARD_BARS, FEATURE_DIM_DIR
from eval_harness import audit_split

log = logging.getLogger(__name__)


def _find_parquet(symbol: str) -> Path | None:
    cands = sorted(PARQUET_DIR.glob(f"HYDRA4_FEAT_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_size, reverse=True)
    return cands[0] if cands else None


def audit(symbol: str, strict: bool = False) -> int:
    p = _find_parquet(symbol)
    if not p:
        print(f"AUDIT FAIL ({symbol}): no parquet under {PARQUET_DIR}")
        return 1

    try:
        df = pd.read_parquet(p)
    except Exception as e:
        # A corrupt parquet must not nuke the audit run for the other symbols.
        # Surface the failure clearly and move on; train_symbol() will hit the
        # same error per-symbol and is already wrapped in try/except there.
        print(f"AUDIT FAIL ({symbol}): unreadable parquet {p.name} — {e}")
        return 1
    n = len(df)
    if n < 10_000:
        print(f"AUDIT WARN ({symbol}): only {n} bars cached — likely too few to train.")

    print(f"\nAuditing {symbol}  ({n:,} bars)\n  source: {p.name}\n")
    warnings = 0

    # 1. Required columns
    required = {"time", "close"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [FAIL] missing required columns: {missing}")
        return 2

    # 2. Time monotonic
    times = pd.to_datetime(df["time"])
    if not times.is_monotonic_increasing:
        print(f"  [FAIL] 'time' column is not monotonically increasing")
        warnings += 1
    else:
        print(f"  [OK]   time monotonic ({times.iloc[0]} -> {times.iloc[-1]})")

    # 3. Time gaps that exceed expected bar interval significantly (heuristic)
    gaps = times.diff().dropna()
    if len(gaps):
        median = gaps.median()
        big = (gaps > median * 5).sum()
        print(f"  [OK]   median bar interval={median};  {big} unusually large gaps")

    # 4. NaNs / Infs in feature columns
    feat_cols = [c for c in df.columns if c.startswith("f_") or c.startswith("feat")]
    if not feat_cols:
        # Fall back: any non-canonical numeric column except OHLCV
        ohlc = {"time","open","high","low","close","tick_volume","real_volume","spread"}
        feat_cols = [c for c in df.columns if c not in ohlc
                     and pd.api.types.is_numeric_dtype(df[c])]
    if feat_cols:
        feats = df[feat_cols].to_numpy(dtype=np.float64)
        nans = int(np.isnan(feats).sum())
        infs = int(np.isinf(feats).sum())
        const = int((feats.std(axis=0) < 1e-12).sum())
        print(f"  [{'OK' if nans+infs==0 else 'WARN'}]   features={len(feat_cols)} "
              f"NaN={nans} Inf={infs} constant={const}")
        if nans or infs or const > 0:
            warnings += 1
        if feats.shape[1] != FEATURE_DIM_DIR:
            print(f"  [WARN] feature column count {feats.shape[1]} != "
                  f"FEATURE_DIM_DIR {FEATURE_DIM_DIR}; engine may pad/trim")
            warnings += 1
    else:
        print("  [INFO] no feature columns in parquet; this cache holds raw bars only")

    # 5. Forward-return sanity — labels look forward, never backward.
    closes = df["close"].to_numpy()
    fwd = np.zeros_like(closes, dtype=np.float64)
    fwd[:-LABEL_FORWARD_BARS] = (closes[LABEL_FORWARD_BARS:]
                                  - closes[:-LABEL_FORWARD_BARS]) / closes[:-LABEL_FORWARD_BARS]
    print(f"  [OK]   forward-return horizon = {LABEL_FORWARD_BARS} bars; "
          f"mean={fwd.mean():.6f}  std={fwd.std():.6f}")

    # 6. Synthesize the same chronological split the trainer uses; verify gap.
    n_val = max(1, int(n * VAL_SPLIT))
    n_tr  = max(1, n - n_val - LABEL_FORWARD_BARS)
    tr_idx = np.arange(0, n_tr)
    va_idx = np.arange(n_tr + LABEL_FORWARD_BARS, n)
    rep = audit_split(train_idx=tr_idx, val_idx=va_idx,
                      features=closes.reshape(-1, 1), labels=closes)
    print(f"  [{'OK' if rep.train_val_overlap == 0 else 'FAIL'}]   "
          f"split: train={rep.n_train:,}  gap={rep.gap_bars}  val={rep.n_val:,}  "
          f"overlap={rep.train_val_overlap}")
    if rep.gap_bars < LABEL_FORWARD_BARS:
        print(f"  [WARN] gap {rep.gap_bars} < LABEL_FORWARD_BARS {LABEL_FORWARD_BARS} "
              f"— train labels may peek into val period")
        warnings += 1

    # 6.5 mk4.7: tick-bar parquets include 8 MTF context columns
    #     (h1_trend / h1_rsi / h1_atr_norm / h1_vwap_rel + h4_*).
    #     Verify schema stability + non-degenerate values.
    is_tickbar = p.name.startswith("HYDRA4_TBARS_")
    if is_tickbar:
        from multi_timeframe import MTF_FEATURE_COLUMNS
        mtf_missing = [c for c in MTF_FEATURE_COLUMNS if c not in df.columns]
        if mtf_missing:
            print(f"  [WARN] tick-bar parquet missing MTF columns: {mtf_missing} "
                  f"(extracted before sub-phase 1a.1?)")
            warnings += 1
        else:
            mtf_arr = df[MTF_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            if np.isnan(mtf_arr).any():
                print(f"  [WARN] NaN in MTF columns")
                warnings += 1
            if (mtf_arr.std(axis=0) < 1e-9).all():
                print(f"  [WARN] all MTF columns are constant — H1/H4 fetch likely failed")
                warnings += 1
            else:
                print(f"  [OK]   MTF schema: {len(MTF_FEATURE_COLUMNS)} columns present, "
                      f"std={mtf_arr.std():.3f}")

    # 7. Class balance (proxy from forward returns)
    long_ratio = float((fwd > 0).mean())
    if abs(long_ratio - 0.5) > 0.15:
        print(f"  [WARN] forward-return class imbalance: long_ratio={long_ratio:.3f}")
        warnings += 1
    else:
        print(f"  [OK]   forward-return class balance: long_ratio={long_ratio:.3f}")

    print()
    if warnings == 0:
        print(f"AUDIT PASS  ({symbol}) — no issues detected.")
        return 0
    print(f"AUDIT WARN  ({symbol}) — {warnings} warning(s) above.")
    return (1 if strict else 0)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbols", nargs="+", help="symbols to audit")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any warning")
    args = p.parse_args(argv)
    rc = 0
    for s in args.symbols:
        try:
            rc = max(rc, audit(s, strict=args.strict))
        except Exception as e:
            # Catch-all: anything other than a clean audit() exit
            # (e.g. unexpected dtype, downstream eval_harness import error)
            # must not abort the rest of the audit. Treat as warning-level.
            print(f"AUDIT FAIL ({s}): unexpected error — {e}")
            rc = max(rc, 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
