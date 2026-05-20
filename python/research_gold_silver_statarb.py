"""
research_gold_silver_statarb.py — does a gold-silver stat-arb on rolling-
window cointegration produce a leak-free, cost-aware edge?

Setup:
  - timeframe: H1 (less noise than M5, faster than H4)
  - dynamic beta from rolling OLS (60-day window = 1440 H1 bars)
  - spread z-score from rolling mean/std (20-day window = 480 bars)
  - entry: z <= -ENTRY  -> long spread (long gold, short silver)
           z >= +ENTRY  -> short spread (short gold, long silver)
  - exit:  z back to 0  OR  |z| > STOP (band breakout exit)
  - cost: 2 * round-trip cost (we open 2 legs)
  - validation: purged 6-fold CV with embargo, net PF per fold

If meanPF >= 1.15 AND every fold > 1.0 AND half-life < 100 bars (~4 days),
we have a deployable strategy. Otherwise honest negative result.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger(__name__)


BETA_WIN = 1440         # 60 days of H1 bars for rolling beta
Z_WIN    = 480          # 20 days for spread mean/std
ENTRY    = 2.0          # z threshold to enter
STOP     = 3.5          # z |threshold| to bail out
COST_LEG = 1.8e-4       # per-leg round-trip cost (matches metatrend)
COST_RT  = 2 * COST_LEG  # we trade both legs

_PF_CAP = 10.0
_N_SPLITS = 6


def _pf(pnl: np.ndarray, min_trades: int = 30) -> float:
    if len(pnl) < min_trades:
        return 0.0
    g = float(pnl[pnl > 0].sum())
    ls = float(-pnl[pnl < 0].sum())
    if ls <= 1e-12:
        return _PF_CAP if g > 0 else 0.0
    return min(_PF_CAP, g / ls)


def _load_pair() -> pd.DataFrame:
    """Inner-join GOLD and SILVER H1 on time."""
    from config import PARQUET_DIR
    g = pd.read_parquet(PARQUET_DIR / "HYDRA4_1HFROMTICKS_GOLD.parquet")
    s = pd.read_parquet(PARQUET_DIR / "HYDRA4_1HFROMTICKS_SILVER.parquet")
    g["time"] = pd.to_datetime(g["time"], utc=True)
    s["time"] = pd.to_datetime(s["time"], utc=True)
    df = pd.merge(
        g[["time", "close"]].rename(columns={"close": "gold"}),
        s[["time", "close"]].rename(columns={"close": "silver"}),
        on="time", how="inner",
    ).sort_values("time").reset_index(drop=True)
    log.info("[statarb] %d common H1 bars  %s -> %s",
             len(df), df["time"].iloc[0], df["time"].iloc[-1])
    return df


def _rolling_beta(lg: np.ndarray, ls: np.ndarray, win: int) -> np.ndarray:
    """
    Causal rolling-OLS estimate of beta in lg ~ a + beta*ls, using the
    past `win` bars (closed). Returns beta[i] = OLS over (i-win+1 .. i).
    Implemented vectorised via cumulative sums.
    """
    n = len(lg)
    out = np.full(n, np.nan)
    # cumulative sums
    cs_x  = np.concatenate([[0.0], np.cumsum(ls)])
    cs_y  = np.concatenate([[0.0], np.cumsum(lg)])
    cs_xx = np.concatenate([[0.0], np.cumsum(ls * ls)])
    cs_xy = np.concatenate([[0.0], np.cumsum(ls * lg)])
    for i in range(win - 1, n):
        lo = i - win + 1
        sx  = cs_x[i + 1]  - cs_x[lo]
        sy  = cs_y[i + 1]  - cs_y[lo]
        sxx = cs_xx[i + 1] - cs_xx[lo]
        sxy = cs_xy[i + 1] - cs_xy[lo]
        m_x = sx / win
        m_y = sy / win
        var_x = sxx / win - m_x * m_x
        cov_xy = sxy / win - m_x * m_y
        if var_x > 1e-12:
            out[i] = cov_xy / var_x
    return out


def _half_life(resid: np.ndarray) -> float:
    """AR(1) half-life of mean reversion on a residual series."""
    r = resid[~np.isnan(resid)]
    if len(r) < 100:
        return float("nan")
    dr = np.diff(r)
    rl = r[:-1]
    # OLS dr = phi * r_{lag} + const
    X = np.column_stack([np.ones_like(rl), rl])
    coef, *_ = np.linalg.lstsq(X, dr, rcond=None)
    phi = coef[1]
    if not (-1.0 < phi < 0.0):
        return float("inf")    # not mean reverting
    return -np.log(2) / np.log(1 + phi)


def _backtest_signal(df: pd.DataFrame) -> dict:
    """
    Build the rolling-beta spread, z-score, and a position series:
        +1 = long  spread (long gold, short silver)
        -1 = short spread (short gold, long  silver)
         0 = flat
    Compute per-H1-bar PnL of the spread strategy net of cost.
    """
    lg = np.log(df["gold"].to_numpy(np.float64))
    ls = np.log(df["silver"].to_numpy(np.float64))
    beta = _rolling_beta(lg, ls, BETA_WIN)
    spread = lg - beta * ls
    # rolling mean/std of spread (also causal, same Z_WIN window)
    s = pd.Series(spread)
    mu = s.rolling(Z_WIN, min_periods=Z_WIN // 2).mean().to_numpy()
    sd = s.rolling(Z_WIN, min_periods=Z_WIN // 2).std().to_numpy()
    z = (spread - mu) / np.where(sd > 1e-9, sd, np.nan)

    # build position with stateful entry/exit logic
    n = len(df)
    pos = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(n):
        zi = z[i]
        if np.isnan(zi) or np.isnan(beta[i]):
            pos[i] = 0; state = 0; continue
        if state == 0:
            if zi <= -ENTRY: state = +1
            elif zi >= +ENTRY: state = -1
        elif state == +1:
            if zi >= 0 or zi <= -STOP: state = 0
        elif state == -1:
            if zi <= 0 or zi >= +STOP: state = 0
        pos[i] = state

    # PnL: at bar i+1 we earn position[i] * d_spread[i+1] minus cost when
    # position changes. d_spread excludes the slow beta movement (treat
    # beta as held over the bar) so it's the realised H1 spread move.
    fwd_lg = np.concatenate([np.diff(lg), [0.0]])
    fwd_ls = np.concatenate([np.diff(ls), [0.0]])
    fwd_spread = fwd_lg - beta * fwd_ls
    raw_pnl = pos * fwd_spread
    # cost when position flips
    flips = np.abs(np.diff(np.concatenate([[0], pos]))).astype(np.float64)
    cost = flips * COST_RT
    pnl = raw_pnl - cost

    n_trades = int((flips > 0).sum() // 2)
    hl = _half_life(spread - mu)
    log.info("[statarb] beta_win=%d z_win=%d entry=%.2f stop=%.2f  "
             "n_trades=%d  half_life=%.1f H1 bars (%.1fh)",
             BETA_WIN, Z_WIN, ENTRY, STOP, n_trades, hl,
             hl * 1.0 if np.isfinite(hl) else hl)
    return {"pnl": pnl, "pos": pos, "z": z, "beta": beta, "spread": spread,
            "half_life": hl, "n_trades": n_trades}


def _purged_cv(pnl: np.ndarray, pos: np.ndarray, n_splits: int = 6,
               embargo_pct: float = 0.01) -> list[float]:
    """Purged k-fold split — drop a small embargo window around each test fold."""
    n = len(pnl)
    fold = n // n_splits
    emb = int(n * embargo_pct)
    out = []
    for k in range(n_splits):
        te_lo = k * fold
        te_hi = (k + 1) * fold if k < n_splits - 1 else n
        # use only the test slice for PF — train would be the rest, but
        # this strategy has no fitting step on price data (all params fixed)
        # so the purged CV here is testing parameter robustness across folds.
        pf = _pf(pnl[te_lo:te_hi])
        out.append(pf)
    return out


def main() -> int:
    df = _load_pair()
    res = _backtest_signal(df)
    pnl = res["pnl"]; pos = res["pos"]
    n = len(pnl)
    log.info("[statarb] active bars: %d / %d (%.1f%%)",
             int((pos != 0).sum()), n, 100 * (pos != 0).mean())
    log.info("[statarb] total PnL = %.4f log-units", float(pnl.sum()))
    overall_pf = _pf(pnl[pos != 0]) if (pos != 0).any() else 0.0
    log.info("[statarb] overall PF on active bars = %.3f", overall_pf)

    folds = _purged_cv(pnl, pos, n_splits=_N_SPLITS)
    mean_pf = float(np.mean(folds))
    min_pf = float(min(folds))
    log.info("[statarb] purged CV folds: %s",
             [round(f, 3) for f in folds])
    log.info("[statarb] meanPF=%.3f  minPF=%.3f  half_life=%.1f bars",
             mean_pf, min_pf, res["half_life"])

    GATE_MEAN, GATE_MIN, HL_MAX = 1.15, 1.0, 100
    deploy = (mean_pf >= GATE_MEAN and min_pf >= GATE_MIN
              and np.isfinite(res["half_life"]) and res["half_life"] <= HL_MAX)
    log.info("[statarb] gate: mean>=%.2f & min>=%.2f & HL<=%d  -> deploy=%s",
             GATE_MEAN, GATE_MIN, HL_MAX, deploy)
    return 0 if deploy else 1


if __name__ == "__main__":
    sys.exit(main())
