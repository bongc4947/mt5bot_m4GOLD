"""
cointegration.py — Engle-Granger cointegration screen across symbol pairs.

For each candidate pair (A, B):
  1. Fit OLS:   B_t = α + β · A_t + ε_t   (rolling window)
  2. ADF test on residuals ε_t:  p-value < threshold → stationary residuals
  3. Repeat across multiple non-overlapping windows
  4. Pair "passes" only if ≥ N consecutive windows have p < threshold

Spurious cointegration (one window passes by chance) is the #1 hedge
trader's footgun. The multi-window requirement kills it.

Usage:
    from cointegration import screen_pairs
    pairs = screen_pairs(symbol_to_close_series, p_threshold=0.05,
                          window_bars=10000, min_passing_windows=3)
    # pairs: list of (sym_a, sym_b, beta, mean_pvalue) sorted by stability
"""
from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-window Engle-Granger test (no statsmodels dependency)
# ---------------------------------------------------------------------------

def _ols_beta(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Fit b = alpha + beta * a; return (alpha, beta, residuals)."""
    A = np.column_stack([np.ones(len(a)), a]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, b.astype(np.float64), rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    residuals = b - (alpha + beta * a)
    return alpha, beta, residuals


def _adf_pvalue(x: np.ndarray, max_lag: int = 5) -> float:
    """
    Augmented Dickey-Fuller p-value approximation (no statsmodels).

    Regression:  Δx_t = ρ * x_{t-1} + Σ γ_i * Δx_{t-i} + ε
    Test stat:   t = ρ_hat / SE(ρ_hat)
    P-value:     looked up in MacKinnon-style critical-value table.

    Returns p-value in (0, 1). Smaller = more confident the series is
    stationary (residuals are mean-reverting → cointegration).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 30:
        return 1.0
    dx = np.diff(x)
    x_lag = x[:-1]
    n_obs = len(dx)
    # Build design matrix with x_lag and lagged differences
    cols = [x_lag]
    for k in range(1, max_lag + 1):
        if k < n_obs:
            shift = np.zeros(n_obs)
            shift[k:] = dx[:-k]
            cols.append(shift)
    X = np.column_stack(cols).astype(np.float64)
    try:
        coef, residuals_lstsq, rank, sv = np.linalg.lstsq(X, dx, rcond=None)
    except np.linalg.LinAlgError:
        return 1.0
    rho = float(coef[0])
    pred = X @ coef
    resid = dx - pred
    ssr = float((resid ** 2).sum())
    if ssr <= 0 or rank < X.shape[1]:
        return 1.0
    sigma2 = ssr / max(1, n_obs - X.shape[1])
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_rho = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
    if se_rho <= 0:
        return 1.0
    t_stat = rho / se_rho

    # MacKinnon (1996) approximation for the no-constant ADF test.
    # Critical values at 1% / 5% / 10% are roughly -2.58 / -1.95 / -1.62.
    # Linear-interp for crude p-value mapping (good enough for a screen).
    table_t = np.array([-3.50, -3.00, -2.58, -2.23, -1.95, -1.62, -1.30, -1.00, -0.50, 0.0])
    table_p = np.array([0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.200, 0.300, 0.500, 1.0])
    # If t_stat is more extreme than -3.5, p ≈ 0.001
    if t_stat <= table_t[0]:
        return 0.001
    if t_stat >= table_t[-1]:
        return 1.0
    p = float(np.interp(t_stat, table_t, table_p))
    return p


# ---------------------------------------------------------------------------
# Pair screen
# ---------------------------------------------------------------------------

def screen_pair(close_a: np.ndarray, close_b: np.ndarray,
                  *,
                  window_bars: int = 10_000,
                  step_bars: int   = 5_000,
                  min_passing_windows: int = 3,
                  p_threshold: float = 0.05,
                  ) -> Dict[str, float]:
    """
    Run the multi-window Engle-Granger screen on one pair.

    Returns dict:
        {
          "passes":           bool,
          "n_windows_passed": int,
          "n_windows_total":  int,
          "mean_pvalue":      float,
          "mean_beta":        float,
        }
    """
    n = min(len(close_a), len(close_b))
    if n < window_bars * 2:
        return {"passes": False, "n_windows_passed": 0, "n_windows_total": 0,
                "mean_pvalue": 1.0, "mean_beta": 0.0}

    starts = list(range(0, n - window_bars + 1, step_bars))
    pvalues, betas, n_pass = [], [], 0
    for s in starts:
        a_win = close_a[s : s + window_bars]
        b_win = close_b[s : s + window_bars]
        _, beta, resid = _ols_beta(a_win, b_win)
        p = _adf_pvalue(resid)
        pvalues.append(p)
        betas.append(beta)
        if p < p_threshold:
            n_pass += 1

    return {
        "passes":           n_pass >= min_passing_windows,
        "n_windows_passed": int(n_pass),
        "n_windows_total":  int(len(starts)),
        "mean_pvalue":      float(np.mean(pvalues)),
        "mean_beta":        float(np.mean(betas)),
    }


def screen_all_pairs(symbol_close: Dict[str, np.ndarray],
                      *,
                      window_bars: int = 10_000,
                      step_bars: int   = 5_000,
                      min_passing_windows: int = 3,
                      p_threshold: float = 0.05,
                      ) -> List[Tuple[str, str, dict]]:
    """
    Run the screen on every (A, B) symbol pair (excluding A-A) and
    return only those that pass, sorted by mean_pvalue ascending
    (most-stable pair first).
    """
    syms = list(symbol_close.keys())
    results: List[Tuple[str, str, dict]] = []
    for a, b in itertools.combinations(syms, 2):
        n_min = min(len(symbol_close[a]), len(symbol_close[b]))
        if n_min < window_bars * 2:
            continue
        # Align lengths from the right (newest data)
        ca = symbol_close[a][-n_min:]
        cb = symbol_close[b][-n_min:]
        r = screen_pair(ca, cb,
                          window_bars=window_bars, step_bars=step_bars,
                          min_passing_windows=min_passing_windows,
                          p_threshold=p_threshold)
        if r["passes"]:
            results.append((a, b, r))
            log.info("PAIR %s/%s — beta=%.3f  passed %d/%d windows  mean_p=%.4f",
                     a, b, r["mean_beta"],
                     r["n_windows_passed"], r["n_windows_total"], r["mean_pvalue"])

    results.sort(key=lambda x: x[2]["mean_pvalue"])
    return results
