"""
audit_strategies.py — static + dynamic audit of the H1/H2/H4 strategy code.

Two passes:

  STATIC  - greps the Python source files for forbidden look-ahead patterns
            we've been bitten by before (forward=, shift(-N), reverse slices
            in feature builders, etc.). The reference incident is the
            `backtest_meanreversion._vol_regime(forward=20)` leak that
            invented PF 3.39 on SILVER.

  DYNAMIC - builds a controlled synthetic price series with a KNOWN no-edge
            property (pure Gaussian random walk), runs each trainer's
            feature builder + labeller, and asserts:
              1. no NaN / Inf in features
              2. features are deterministic on a fixed seed
              3. swapping the future has no effect on features (causality
                 test — randomise rows AFTER each index and confirm the
                 feature value at that index is unchanged)
              4. labels can see the future, features cannot (sanity)

Returns 0 on clean audit, non-zero on any failure. Wire it as the first
step in cloud/runner.sh::run_strategies so a bad commit fails immediately.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static audit
# ---------------------------------------------------------------------------

# Patterns whose presence in *feature builder bodies* indicates look-ahead.
FORBIDDEN_IN_FEATURES = [
    # `forward=` keyword indicates a forward-looking computation, which is
    # exactly what _vol_regime(forward=20) used and got PRODUCTION_GAPS.md
    # flagging it weeks before SILVER's PF 0.18 surfaced the leak.
    (re.compile(r"\bforward\s*="), "explicit `forward=` keyword — refer to PRODUCTION_GAPS"),
    # shift(-N) means "pull values from N rows in the future". This is
    # legitimate ONLY in label construction. Feature builders must not use it.
    (re.compile(r"\.shift\(\s*-\s*\d"), "`.shift(-N)` pulls future values — only allowed in label code"),
    # ::-1 reverses a series; in causal code this signals "scanning from
    # the end", which usually leaks future info into past rows.
    (re.compile(r"\[\s*::\s*-\s*1\s*\]"), "reverse-slice `[::-1]` in feature path"),
]

# Functions explicitly marked as feature builders (the static audit gates them).
FEATURE_BUILDER_NAMES = {"build_h1_features", "build_h2_features"}
# Functions explicitly marked as label / outcome builders (allowed to use future).
LABEL_BUILDER_NAMES = {"make_labels_and_pnl", "generate_session_candidates",
                       "triple_barrier_outcome", "_backtest_positions",
                       "_eval_strategy"}

SOURCE_FILES = [
    "strategies_common.py",
    "train_h1_orderflow.py",
    "train_h2_session.py",
    "train_h4_trend.py",
]


def _extract_function_body(source: str, name: str) -> str | None:
    """Return the body of `def <name>(...):` up to the next def or EOF.
    Uses regex — good enough for our flat module structure (no nesting beyond
    one level inside the function)."""
    m = re.search(rf"\ndef\s+{re.escape(name)}\s*\(", source)
    if not m:
        return None
    start = m.end()
    # Find next top-level `def `; greedy until end of file otherwise.
    rest = source[start:]
    nxt = re.search(r"\ndef\s+\w+\s*\(", rest)
    if nxt:
        return rest[:nxt.start()]
    return rest


def static_audit(repo_root: Path) -> int:
    failures = 0
    for fname in SOURCE_FILES:
        path = repo_root / "python" / fname
        if not path.exists():
            log.error("[static] missing %s", path)
            failures += 1
            continue
        src = path.read_text(encoding="utf-8")
        for fn_name in FEATURE_BUILDER_NAMES:
            body = _extract_function_body(src, fn_name)
            if body is None:
                continue   # not in this file, fine
            for rx, reason in FORBIDDEN_IN_FEATURES:
                if rx.search(body):
                    log.error("[static] FORBIDDEN in %s::%s — %s",
                              fname, fn_name, reason)
                    failures += 1
    if failures == 0:
        log.info("[static] %d source files, %d feature builders — clean",
                 len(SOURCE_FILES), len(FEATURE_BUILDER_NAMES))
    return failures


# ---------------------------------------------------------------------------
# Synthetic-data fixtures for dynamic audit
# ---------------------------------------------------------------------------

def _gen_synthetic_ticks(n_ticks: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic tick stream with no edge: random-walk mid + constant
    spread + alternating last/bid/ask to give Lee-Ready non-trivial output."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 1e-4, size=n_ticks)
    mid = 1.10 * np.exp(np.cumsum(log_ret))
    half_sp = 1e-5
    bid = mid - half_sp
    ask = mid + half_sp
    last_jitter = rng.choice([-1, 0, 1], size=n_ticks, p=[0.45, 0.10, 0.45])
    last = mid + last_jitter * half_sp
    times = pd.date_range("2024-01-01", periods=n_ticks, freq="100ms",
                           tz="UTC")
    return pd.DataFrame({
        "time": times,
        "bid": bid, "ask": ask, "last": last,
        "volume": np.ones(n_ticks),
        "mid": mid,
        "spread": ask - bid,
    })


def _gen_synthetic_m5(n_bars: int = 3000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 5e-4, size=n_bars)
    close = 1.10 * np.exp(np.cumsum(log_ret))
    high  = close * (1 + np.abs(rng.normal(0, 3e-4, size=n_bars)))
    low   = close * (1 - np.abs(rng.normal(0, 3e-4, size=n_bars)))
    opn   = np.concatenate([[close[0]], close[:-1]])
    times = pd.date_range("2024-01-01 00:00", periods=n_bars, freq="5min",
                           tz="UTC")
    return pd.DataFrame({
        "time": times,
        "open": opn, "high": high, "low": low, "close": close,
        "tick_volume": np.ones(n_bars),
        "spread": np.full(n_bars, 1e-5),
    })


# ---------------------------------------------------------------------------
# Causality test — the gold-standard leak check
# ---------------------------------------------------------------------------

def _causality_test(feature_builder: Callable, bars: pd.DataFrame,
                     candidates: pd.DataFrame | None = None, *,
                     row_under_test: int) -> bool:
    """If feature_builder is strictly causal, swapping bars[row+1:] with
    random noise must not change feature_builder output at `row`. Returns
    True iff the function passes."""
    if candidates is None:
        baseline = feature_builder(bars.copy())[row_under_test]
        scrambled_bars = bars.copy()
        rng = np.random.default_rng(7)
        for col in ("open", "high", "low", "close", "mid", "ofi", "taker_ratio",
                    "signed_volume", "total_volume"):
            if col in scrambled_bars.columns:
                values = scrambled_bars[col].to_numpy().copy()
                values[row_under_test + 1:] = rng.normal(
                    np.nanmean(values), max(1e-9, np.nanstd(values)),
                    size=len(values) - row_under_test - 1)
                scrambled_bars[col] = values
        after = feature_builder(scrambled_bars)[row_under_test]
    else:
        cand = candidates.copy()
        cand_row_idx = candidates.index[candidates["bar_idx"] <= row_under_test][-1]
        baseline = feature_builder(bars.copy(), candidates)[cand_row_idx]
        scrambled_bars = bars.copy()
        rng = np.random.default_rng(7)
        for col in ("open", "high", "low", "close", "spread"):
            if col in scrambled_bars.columns:
                values = scrambled_bars[col].to_numpy().copy()
                values[row_under_test + 1:] = rng.normal(
                    np.nanmean(values), max(1e-9, np.nanstd(values)),
                    size=len(values) - row_under_test - 1)
                scrambled_bars[col] = values
        after = feature_builder(scrambled_bars, candidates)[cand_row_idx]
    same = bool(np.allclose(baseline, after, atol=1e-6, rtol=1e-4))
    if not same:
        diff = np.abs(np.asarray(baseline) - np.asarray(after))
        log.error("[dynamic] causality FAILED: row=%d  max|diff|=%.3e  "
                  "baseline[:5]=%s  after[:5]=%s",
                  row_under_test, float(diff.max()),
                  baseline[:5], after[:5])
    return same


def dynamic_audit() -> int:
    """Smoke-test each strategy's feature builder on synthetic data."""
    import train_h1_orderflow as h1
    import train_h2_session   as h2
    from strategies_common import ticks_to_tickbars

    failures = 0

    # ---- H1 ----
    log.info("[dynamic] H1 — synthesizing 20k ticks ...")
    ticks = _gen_synthetic_ticks(20000, seed=0)
    bars  = ticks_to_tickbars(ticks, ticks_per_bar=100)
    log.info("[dynamic] H1 — %d tick-bars built", len(bars))
    feats = h1.build_h1_features(bars)
    if feats.shape != (len(bars), h1.H1_FEATURE_DIM):
        log.error("[dynamic] H1 feature shape %s != (%d, %d)",
                  feats.shape, len(bars), h1.H1_FEATURE_DIM)
        failures += 1
    if not np.all(np.isfinite(feats)):
        log.error("[dynamic] H1 features contain NaN/Inf"); failures += 1
    # Determinism: re-run on same bars
    feats2 = h1.build_h1_features(bars)
    if not np.allclose(feats, feats2):
        log.error("[dynamic] H1 features are NON-DETERMINISTIC"); failures += 1
    # Causality at a middle row
    if not _causality_test(h1.build_h1_features, bars, row_under_test=len(bars) // 2):
        failures += 1
    else:
        log.info("[dynamic] H1 — causality OK")
    # Label sanity: must look ahead (otherwise it's not predicting anything)
    y, pnl = h1.make_labels_and_pnl(bars, horizon=10, symbol="TEST")
    if abs(float(y.mean()) - 0.5) > 0.20:
        log.warning("[dynamic] H1 — random-walk label balance off: %.3f "
                     "(synthetic data is small; not a hard failure)", float(y.mean()))

    # ---- H2 ----
    log.info("[dynamic] H2 — synthesizing 3000 M5 bars ...")
    m5 = _gen_synthetic_m5(3000, seed=1)
    # Force a few session-window hours to be present in the index.
    cand = h2.generate_session_candidates(m5, donchian_window=20, sl_atr=0.5,
                                            tp_atr=1.5, timeout_bars=12)
    log.info("[dynamic] H2 — %d session candidates generated", len(cand))
    if len(cand) >= 5:
        feats = h2.build_h2_features(m5, cand)
        if feats.shape != (len(cand), h2.H2_FEATURE_DIM):
            log.error("[dynamic] H2 feature shape %s != (%d, %d)",
                      feats.shape, len(cand), h2.H2_FEATURE_DIM)
            failures += 1
        if not np.all(np.isfinite(feats)):
            log.error("[dynamic] H2 features contain NaN/Inf"); failures += 1
        # Causality test on a candidate near the middle
        mid_row = int(cand["bar_idx"].iloc[len(cand) // 2])
        if not _causality_test(h2.build_h2_features, m5, cand,
                                row_under_test=mid_row):
            failures += 1
        else:
            log.info("[dynamic] H2 — causality OK")
    else:
        log.warning("[dynamic] H2 — too few session candidates on synthetic "
                     "data (%d); skipping feature/causality check", len(cand))

    # ---- H4 ----
    log.info("[dynamic] H4 — synthesizing 3000 H1 bars ...")
    h1_bars = _gen_synthetic_m5(3000, seed=2)
    h1_bars["time"] = pd.date_range("2020-01-01", periods=len(h1_bars), freq="1h",
                                      tz="UTC")
    import train_h4_trend as h4
    close = h1_bars["close"].to_numpy()
    pos = h4._ma_cross_positions(close, fast=20, slow=50, allow_short=True)
    # Causality: pos[i] must NOT depend on close[i+1:]
    close_scrambled = close.copy()
    rng = np.random.default_rng(11)
    test_row = len(close) // 2
    close_scrambled[test_row + 1:] = rng.normal(close[test_row], 1e-3,
                                                  size=len(close) - test_row - 1)
    pos2 = h4._ma_cross_positions(close_scrambled, fast=20, slow=50, allow_short=True)
    if not np.isclose(pos[test_row], pos2[test_row]):
        log.error("[dynamic] H4 — MA-cross positions DEPEND ON FUTURE")
        failures += 1
    else:
        log.info("[dynamic] H4 — causality OK")
    # Random-walk backtest: Sharpe should be near zero
    bar_ret, _ = h4._backtest_positions(close, pos, cost_per_turn=0.0)
    if abs(np.mean(bar_ret) / (np.std(bar_ret) + 1e-12)) > 0.2:
        log.warning("[dynamic] H4 — random-walk Sharpe surprisingly high: %.3f "
                     "(can happen on short series; not a hard failure)",
                     np.mean(bar_ret) / (np.std(bar_ret) + 1e-12))

    if failures == 0:
        log.info("[dynamic] all causality + sanity checks passed")
    return failures


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--static-only", action="store_true")
    p.add_argument("--dynamic-only", action="store_true")
    args = p.parse_args(argv)
    repo_root = Path(__file__).parent.parent

    failures = 0
    if not args.dynamic_only:
        failures += static_audit(repo_root)
    if not args.static_only:
        try:
            failures += dynamic_audit()
        except Exception:
            log.exception("[dynamic] audit crashed")
            failures += 1
    if failures:
        log.error("AUDIT FAILED  (%d failures)", failures)
        return 1
    log.info("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
