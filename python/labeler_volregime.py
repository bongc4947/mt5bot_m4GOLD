"""
labeler_volregime.py — volatility-regime labels (mk4.4 #4).

DIFFERENT PROBLEM, BETTER SNR
-----------------------------
Direction at M5 is mostly noise: WR around 50-55% is the ceiling for
retail TA features. But "is the *next* 20 bars high-vol or low-vol?"
has substantially better signal-to-noise — volatility is autocorrelated
(known fact since Engle 1982 ARCH).

The most profitable use of vol-regime classification is as a TRADE GATE
on top of the directional model:

    if vol_regime == "low":   skip   # spreads eat the move; don't trade
    if vol_regime == "high":  trade  # large moves can pay for costs
    if vol_regime == "med":   trade with reduced size

That single gate often turns an unprofitable directional model into a
profitable one, because the filter selects the bars where the cost-vs-
move ratio actually permits an edge.

LABEL DEFINITION
----------------
Forward realised volatility, normalised by recent rolling baseline:

    rv_fwd[i] = sqrt(  sum  (log_return[i+1..i+H])**2 )
    rv_base[i] = rolling_mean(rv_fwd, baseline_window) at bar i (backward)

    ratio[i]  = rv_fwd[i] / rv_base[i]

Bucket into 3 classes by ratio quantiles over the training set:
    0 = LOW   (ratio < q33)
    1 = MEDIUM (q33 <= ratio < q67)
    2 = HIGH  (ratio >= q67)

Labels are time-aware: quantiles computed on training set only, applied
to validation set.

NOT YET WIRED — proposed integration
------------------------------------
This file is a scaffold + reference. To deploy:

  (a) Train a separate vol-regime classifier on the same 200-dim
      features.  Output: 3-class probability.
  (b) At inference, EA combines direction model + vol-regime model:
        if vol_regime_pred != HIGH:
            skip trade
        else:
            use direction model's signal as-is
  (c) Export both ONNX files; EA loads both and gates on vol-regime
      output.

The label function below is ready to call. The MQL5 mirror + EA gating
logic are the unimplemented parts.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EPS = 1e-12


def compute_vol_regime_labels(
    df: pd.DataFrame,
    forward_bars: int = 20,
    baseline_window: int = 200,
    quantiles: Tuple[float, float] = (0.33, 0.67),
) -> Tuple[np.ndarray, dict]:
    """
    Returns (labels [N], stats_dict). Labels in {0=LOW, 1=MEDIUM, 2=HIGH}.

    Args:
        df             : OHLCV bars; needs a 'close' column.
        forward_bars   : how many bars ahead to measure realised vol.
        baseline_window: rolling window for the divisor (recent vol regime).
        quantiles      : 2-tuple (q_lo, q_hi) splitting LOW/MEDIUM/HIGH.

    The returned stats_dict carries the actual q_lo / q_hi values so the
    same thresholds can be applied at inference time without recomputing
    from a different sample.
    """
    close = df["close"].to_numpy(dtype=np.float64)
    N = len(close)
    log_ret = np.zeros(N)
    log_ret[1:] = np.log(np.maximum(close[1:], EPS) /
                         np.maximum(close[:-1], EPS))

    # Forward realised vol: sqrt(sum r^2 over the forward window). Sum-of-
    # squares not mean-of-squares because we want absolute vol scale.
    sq = log_ret * log_ret
    cumsum = np.concatenate([[0.0], np.cumsum(sq)])
    rv_fwd = np.zeros(N)
    H = forward_bars
    rv_fwd[:-H] = np.sqrt(cumsum[H+1 : N+1] - cumsum[1 : N-H+1])

    # Rolling baseline: mean of the last `baseline_window` rv_fwd values
    # at bar i (uses only past data — no leakage).
    bl = pd.Series(rv_fwd).rolling(baseline_window, min_periods=baseline_window // 2).mean().to_numpy()
    bl = np.where(np.isfinite(bl) & (bl > EPS), bl, EPS)
    ratio = rv_fwd / bl

    # Mask: discard the first baseline_window bars (no rolling stat) and
    # the last forward_bars bars (no forward window).
    valid = np.zeros(N, dtype=bool)
    valid[baseline_window:N - H] = True

    # Quantiles computed on the valid subset only.
    valid_ratios = ratio[valid]
    if valid_ratios.size < 100:
        log.warning("vol_regime: only %d valid samples — labels may be noisy",
                    valid_ratios.size)
    q_lo, q_hi = np.quantile(valid_ratios, quantiles[0]), np.quantile(valid_ratios, quantiles[1])

    labels = np.full(N, -1, dtype=np.int8)   # -1 = invalid (warmup or tail)
    labels[valid] = np.where(ratio[valid] < q_lo, 0,
                     np.where(ratio[valid] < q_hi, 1, 2)).astype(np.int8)

    stats = {
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
        "baseline_window": int(baseline_window),
        "forward_bars":    int(forward_bars),
        "n_low":   int((labels == 0).sum()),
        "n_med":   int((labels == 1).sum()),
        "n_high":  int((labels == 2).sum()),
        "n_invalid": int((labels == -1).sum()),
    }
    log.info("vol_regime labels: LOW=%d  MED=%d  HIGH=%d  INVALID=%d  "
             "(q_lo=%.4f q_hi=%.4f)",
             stats["n_low"], stats["n_med"], stats["n_high"],
             stats["n_invalid"], q_lo, q_hi)
    return labels, stats
