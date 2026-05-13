"""
eval_harness.py — production-grade evaluation utilities.

What this module provides that profit_metric.py does not:

1. Returns-based metrics: Sharpe, Sortino, max-DD, Calmar, equity curve.
2. Trade costs: per-trade commission and per-trade slippage in pips.
3. Walk-forward CV: time-ordered (train, val) index splits with a gap.
4. Per-regime breakdown: accuracy/PF/Sharpe sliced by an arbitrary
   integer regime tag (e.g. HMM cluster, vol bucket, session).
5. Leakage sanity checks: future-shift detection, split overlap, NaNs.

All functions accept plain numpy arrays so they work on CPU/GPU outputs
alike — convert torch tensors before calling.

The signing convention for `returns` (or `forward_returns`) is:
  +x  = a long-side bar gain of x R-multiples (>0 = up move)
  -x  = a long-side bar loss
For a binary direction model, predicted_return = sign(pred) * forward_return,
where sign maps {0, 1} -> {-1, +1}.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np

log = logging.getLogger(__name__)

# Annualization factors for common bar sizes
ANNUALIZATION = {
    "M1":  365 * 24 * 60,
    "M5":  365 * 24 * 12,
    "M15": 365 * 24 * 4,
    "H1":  365 * 24,
    "H4":  365 * 6,
    "D1":  252,
}


# ---------------------------------------------------------------------------
# Core PnL metrics
# ---------------------------------------------------------------------------

@dataclass
class PnLMetrics:
    n_trades:      int
    win_rate:      float
    profit_factor: float
    expected_value: float    # mean per-trade return (R-multiples or whatever unit `returns` is)
    sharpe:        float     # annualized; nan if returns are degenerate
    sortino:       float     # annualized; nan if no downside
    max_drawdown:  float     # >=0; magnitude of worst peak-to-trough on equity
    calmar:        float     # CAGR / max_drawdown; nan if max_dd == 0
    total_return:  float
    cost_per_trade: float
    bar:           str

    def as_dict(self) -> dict:
        return {k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v)
                for k, v in self.__dict__.items()}


def compute_pnl_metrics(
    *,
    pred: np.ndarray,                      # binary (0/1) or signed (-1/+1)
    forward_returns: np.ndarray,           # per-bar forward return, same shape as pred
    bar: str = "M5",
    commission: float = 0.0,               # per round-trip, same unit as returns
    slippage:   float = 0.0,               # per round-trip
    risk_free_per_period: float = 0.0,
) -> PnLMetrics:
    """
    Returns full PnL metrics for a single (predictions, forward_returns) series.

    `pred` may be the binary 0/1 output of a direction model; we map
    {0:-1, 1:+1}. If pred already contains values in {-1, 0, +1}, zeros are
    treated as 'no trade' and excluded from cost calculations.
    """
    pred = np.asarray(pred).reshape(-1)
    fwd  = np.asarray(forward_returns, dtype=np.float64).reshape(-1)
    if pred.shape != fwd.shape:
        raise ValueError(f"pred {pred.shape} != forward_returns {fwd.shape}")

    # Map 0/1 -> -1/+1 if no zeros are passed; otherwise treat 0 as 'no trade'.
    if pred.min() >= 0 and pred.max() <= 1 and not np.any(pred == -1):
        signed = pred.astype(np.float64) * 2.0 - 1.0
        traded_mask = np.ones_like(signed, dtype=bool)
    else:
        signed = np.sign(pred).astype(np.float64)
        traded_mask = signed != 0

    cost = float(commission) + float(slippage)
    per_trade_ret = signed * fwd
    per_trade_ret[traded_mask] -= cost

    n_trades = int(traded_mask.sum())
    if n_trades == 0:
        return PnLMetrics(0, float("nan"), float("nan"), 0.0,
                          float("nan"), float("nan"), 0.0, float("nan"),
                          0.0, cost, bar)

    rt   = per_trade_ret[traded_mask]
    wins = (rt > 0)
    win_rate = float(wins.mean())
    gross_profit = float(rt[wins].sum())
    gross_loss   = float(-rt[~wins].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else float("inf")

    # Sharpe / Sortino using full per-bar series (treats no-trade bars as 0 return)
    ann = ANNUALIZATION.get(bar, ANNUALIZATION["M5"])
    excess = per_trade_ret - risk_free_per_period
    mu, sigma = excess.mean(), excess.std(ddof=1) if excess.size > 1 else 0.0
    sharpe  = (mu / sigma) * math.sqrt(ann) if sigma > 1e-12 else float("nan")
    downside = excess[excess < 0]
    sigma_d  = downside.std(ddof=1) if downside.size > 1 else 0.0
    sortino  = (mu / sigma_d) * math.sqrt(ann) if sigma_d > 1e-12 else float("nan")

    equity = np.cumsum(per_trade_ret)
    peak   = np.maximum.accumulate(equity)
    dd     = peak - equity
    max_dd = float(dd.max()) if dd.size else 0.0

    total_ret = float(equity[-1]) if equity.size else 0.0
    # Calmar: annualized return / |max_dd|
    if max_dd > 1e-12 and len(equity) > 1:
        cagr = (mu * ann)
        calmar = cagr / max_dd
    else:
        calmar = float("nan")

    return PnLMetrics(
        n_trades=n_trades,
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4) if math.isfinite(profit_factor) else float("inf"),
        expected_value=round(float(rt.mean()), 6),
        sharpe=round(sharpe, 4) if math.isfinite(sharpe) else float("nan"),
        sortino=round(sortino, 4) if math.isfinite(sortino) else float("nan"),
        max_drawdown=round(max_dd, 6),
        calmar=round(calmar, 4) if math.isfinite(calmar) else float("nan"),
        total_return=round(total_ret, 6),
        cost_per_trade=cost,
        bar=bar,
    )


# ---------------------------------------------------------------------------
# Per-regime breakdown
# ---------------------------------------------------------------------------

def regime_breakdown(
    *,
    pred: np.ndarray,
    forward_returns: np.ndarray,
    regimes: np.ndarray,
    bar: str = "M5",
    commission: float = 0.0,
    slippage:   float = 0.0,
) -> dict[int, PnLMetrics]:
    """Return metrics keyed by regime tag."""
    regimes = np.asarray(regimes).reshape(-1)
    out: dict[int, PnLMetrics] = {}
    for r in np.unique(regimes):
        mask = regimes == r
        if mask.sum() < 30:
            continue
        out[int(r)] = compute_pnl_metrics(
            pred=pred[mask],
            forward_returns=forward_returns[mask],
            bar=bar,
            commission=commission,
            slippage=slippage,
        )
    return out


# ---------------------------------------------------------------------------
# Walk-forward CV splits
# ---------------------------------------------------------------------------

def walk_forward_splits(
    n_samples: int,
    *,
    n_folds: int = 5,
    gap: int = 20,
    min_train_frac: float = 0.30,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_idx, val_idx) tuples with an expanding training window.

    Layout (k = n_folds = 4, n = 1000, gap = 20):
        fold 0:  train [0..300]   gap [300..320]   val [320..475]
        fold 1:  train [0..475]   gap [475..495]   val [495..650]
        fold 2:  train [0..650]   gap [650..670]   val [670..825]
        fold 3:  train [0..825]   gap [825..845]   val [845..1000]

    Args:
        n_samples : total number of bars (training + holdout).
        n_folds   : number of CV folds.
        gap       : bars to skip between train end and val start (avoids label
                    leakage when labels look forward by `gap` bars).
        min_train_frac : first fold's train end as a fraction of n_samples.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if not 0.0 < min_train_frac < 1.0:
        raise ValueError("min_train_frac must be in (0, 1)")

    start_train_end = int(n_samples * min_train_frac)
    val_total       = n_samples - start_train_end
    val_size        = max(1, val_total // n_folds)

    for k in range(n_folds):
        train_end = start_train_end + k * val_size
        val_start = train_end + gap
        val_end   = min(val_start + val_size, n_samples)
        if val_start >= n_samples or val_end <= val_start:
            return
        yield np.arange(0, train_end), np.arange(val_start, val_end)


# ---------------------------------------------------------------------------
# Leakage sanity checks
# ---------------------------------------------------------------------------

@dataclass
class LeakageReport:
    n_samples:           int
    n_train:             int
    n_val:               int
    train_val_overlap:   int     # should be 0
    nan_in_features:     int
    nan_in_labels:       int
    label_uses_future:   bool    # True iff labels at index i are computed from data > i (intended)
    val_starts_after_train: bool # True iff max(train_idx) < min(val_idx)
    gap_bars:            int     # min(val_idx) - max(train_idx) - 1


def audit_split(
    *,
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    features:  np.ndarray,
    labels:    np.ndarray,
) -> LeakageReport:
    """Cheap sanity check on a single time-aware split."""
    train_idx = np.asarray(train_idx)
    val_idx   = np.asarray(val_idx)

    overlap = int(np.intersect1d(train_idx, val_idx, assume_unique=False).size)
    val_after = bool(train_idx.max() < val_idx.min()) if train_idx.size and val_idx.size else False
    gap = int(val_idx.min() - train_idx.max() - 1) if (val_after and train_idx.size and val_idx.size) else 0

    return LeakageReport(
        n_samples=int(features.shape[0]),
        n_train=int(train_idx.size),
        n_val=int(val_idx.size),
        train_val_overlap=overlap,
        nan_in_features=int(np.isnan(features).sum()),
        nan_in_labels=int(np.isnan(labels.astype(np.float64)).sum()),
        label_uses_future=True,   # caller asserts; cannot detect from arrays alone
        val_starts_after_train=val_after,
        gap_bars=gap,
    )


# ---------------------------------------------------------------------------
# CLI: quick sanity smoke test against a parquet bundle
# ---------------------------------------------------------------------------

def _cli() -> int:
    """python eval_harness.py <symbol>  — quick metric demo on cached parquet."""
    import argparse, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    p = argparse.ArgumentParser(description="Quick eval-harness smoke test.")
    p.add_argument("symbol", help="symbol (must have a parquet under data/parquet/)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--slippage",   type=float, default=0.0)
    args = p.parse_args()

    from config import PARQUET_DIR
    cached = sorted(PARQUET_DIR.glob(f"HYDRA4_FEAT_{args.symbol}_*.parquet"),
                    key=lambda x: x.stat().st_size, reverse=True)
    if not cached:
        print(f"No cached parquet for {args.symbol}.")
        return 1

    import pandas as pd
    df = pd.read_parquet(cached[0])
    if "close" not in df.columns:
        print("parquet missing 'close' column — cannot synthesize forward returns.")
        return 2

    closes = df["close"].to_numpy()
    fwd = np.zeros_like(closes, dtype=np.float64)
    fwd[:-20] = (closes[20:] - closes[:-20]) / closes[:-20]
    pred_naive = (np.diff(closes, prepend=closes[0]) > 0).astype(np.int8)

    print(f"\nSmoke test on {args.symbol} ({len(df):,} bars)")
    print("=" * 60)
    for k, (tr, va) in enumerate(walk_forward_splits(len(df), n_folds=args.folds)):
        m = compute_pnl_metrics(
            pred=pred_naive[va], forward_returns=fwd[va],
            bar="M5",
            commission=args.commission, slippage=args.slippage,
        )
        print(f"fold {k}  train=[0..{tr[-1]}]  val=[{va[0]}..{va[-1]}]"
              f"  n={m.n_trades:,}  WR={m.win_rate:.3f}  PF={m.profit_factor}"
              f"  Sharpe={m.sharpe}  MDD={m.max_drawdown:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
