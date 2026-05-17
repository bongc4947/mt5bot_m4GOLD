"""
aurum/conformal.py — L5 conformal calibration.

Softmax probabilities are not calibrated confidences — a 0.55 long
probability is not "55% right". Split conformal prediction (Vovk;
Romano et al., NeurIPS 2019 for the quantile variant) turns the raw
scores into a distribution-free guarantee: with a calibration set, pick a
threshold q̂ such that the *prediction set* {classes with score >= 1-q̂}
contains the true class with probability >= 1 - alpha.

Trading rule: act only when the conformal prediction set is a SINGLETON.
A singleton means the model is confident enough, at the chosen error rate
alpha, that exactly one direction is plausible. Multi-class sets ->
abstain. This replaces the ad-hoc MC-dropout threshold.

The calibration produces a single scalar threshold, stored in the spec
JSON; the EA applies it with one comparison — no ONNX needed for L5.
"""

from __future__ import annotations

import logging

import numpy as np

from aurum.aurum_config import CONFORMAL_ALPHA

log = logging.getLogger(__name__)


def calibrate_threshold(cal_probs: np.ndarray, cal_labels: np.ndarray,
                        alpha: float = CONFORMAL_ALPHA) -> float:
    """
    Split-conformal calibration (APS-style score = 1 - p_true).

    cal_probs  : float[N, K] softmax probabilities on the calibration set.
    cal_labels : int[N]      true class indices.
    Returns q̂ — the score quantile that guarantees >= 1-alpha coverage.
    """
    n = len(cal_labels)
    if n < 20:
        log.warning("[conformal] tiny calibration set (n=%d) — threshold "
                    "guarantee is weak", n)
    true_p = cal_probs[np.arange(n), cal_labels]
    scores = 1.0 - true_p                      # nonconformity score
    # Finite-sample-corrected quantile level.
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(scores, level, method="higher"))
    log.info("[conformal] n=%d  alpha=%.2f  level=%.4f  q_hat=%.4f",
             n, alpha, level, q_hat)
    return q_hat


def prediction_set(probs: np.ndarray, q_hat: float) -> np.ndarray:
    """
    Boolean[N, K] — class k is in the set iff (1 - p_k) <= q_hat,
    i.e. p_k >= 1 - q_hat.
    """
    return (1.0 - probs) <= q_hat


def singleton_mask(probs: np.ndarray, q_hat: float) -> np.ndarray:
    """Boolean[N] — True where the conformal set has exactly one class."""
    return prediction_set(probs, q_hat).sum(axis=1) == 1


def evaluate_coverage(cal_probs: np.ndarray, cal_labels: np.ndarray,
                      q_hat: float) -> dict:
    """Empirical coverage + average set size — sanity check on a holdout."""
    pset = prediction_set(cal_probs, q_hat)
    n = len(cal_labels)
    covered = pset[np.arange(n), cal_labels].mean()
    avg_size = pset.sum(axis=1).mean()
    singleton_frac = (pset.sum(axis=1) == 1).mean()
    return {"coverage": float(covered),
            "avg_set_size": float(avg_size),
            "singleton_frac": float(singleton_frac)}
