"""
feature_engine.py — 1160-dim direction features + 1200-dim exec features.

================================================================================
mk4.2.1: LEGACY / OPTIONAL.
The canonical feature implementation is now MQL5 — ea/includes/FeatureEncoder.mqh.

For new training runs prefer:
    1. Run ea/MT5_Bot_mk4_FeatureExport.mq5 once per symbol (drag onto chart).
    2. Train with `python python/train.py all --skip-extract --mt5-features`.

That path eliminates feature drift between training and live (one
implementation, not two). This module is retained as a fallback when
you can't run the MT5 exporter — and as a parity-check tool against
the MQL5 output.
================================================================================

mk4.2: macro / fundamentals / QTW / SIE blocks removed. All remaining
features are computed from MT5-supplied bars only.

Must match EA FeatureEncoder.mqh exactly for the direction feature block.

Direction feature layout (1160-dim):
  Block 0  M5       [0..399]      50 features × 8 statistical windows
  Block 1  H1       [400..609]    30 features × 7 statistical windows
  Block 2  H4       [610..729]    20 features × 6 statistical windows
  Block 3  H8       [730..849]    20 features × 6 statistical windows
  Block 4  D1       [850..969]    20 features × 6 statistical windows
  Block 5  Spectral [970..1029]   60 (FFT, autocorr, Hurst, entropy)
  Block 6  Pattern  [1030..1079]  50 (candlestick + price action)
  Block 7  StatReg  [1080..1139]  60 (vol/trend regime + rolling moments)
  Block 8  XAsset   [1140..1159]  20 (cross-asset correlations against the
                                     symbol your broker quotes; 0 if not)

Execution context (40-dim appended for exec model, absolute indices 1160..1199):
  Ctx 0-3   : microstructure (atr5/atr14, atr14/atr50, spread/atr14, vol spike)
  Ctx 4-39  : reserved for EA-injected position context (zeros at training)
"""

import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    FEATURE_DIM_DIR, FEATURE_DIM_EXEC, EXEC_CTX_DIM,
)
# mk4.3: dropped H1_DIM/H4_DIM/H8_DIM/D1_DIM/SPECTRAL_DIM/PATTERN_DIM/STAT_DIM/
# XASSET_DIM imports — those blocks no longer exist in the parity-floored
# 200-dim layout. The dead block-builder functions further below are kept
# only for archival reference and are no longer called.
_H1_DIM = _H4_DIM = _H8_DIM = _D1_DIM = 0          # legacy stubs
_SPECTRAL_DIM = _PATTERN_DIM = _STAT_DIM = _XASSET_DIM = 0
H1_DIM, H4_DIM, H8_DIM, D1_DIM = 0, 0, 0, 0         # used only by dead helpers
SPECTRAL_DIM = PATTERN_DIM = STAT_DIM = XASSET_DIM = 0

log = logging.getLogger(__name__)
EPS = 1e-10

# ═══════════════════════════════════════════════════════════════════════════
# Low-level indicator helpers (vectorised, operate on numpy arrays)
# ═══════════════════════════════════════════════════════════════════════════

def _log_return(close: np.ndarray, n: int) -> np.ndarray:
    """Vectorised log-return over n bars."""
    out = np.zeros(len(close))
    if len(close) > n:
        with np.errstate(divide='ignore', invalid='ignore'):
            lr = np.log(close[n:] / np.maximum(close[:-n], EPS))
        out[n:] = np.where(np.isfinite(lr), lr, 0.0)
    return out


def _realized_vol(close: np.ndarray, window: int) -> np.ndarray:
    """Vectorised realised volatility (rolling std of log-returns)."""
    lr = _log_return(close, 1)
    min_p = min(2, window)   # window=1 is valid; min_periods can't exceed window
    return pd.Series(lr).rolling(window, min_periods=min_p).std().fillna(0.0).to_numpy()


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Vectorised rolling mean using cumsum."""
    out = np.zeros_like(x, dtype=float)
    cs = np.cumsum(x)
    out[w:] = (cs[w:] - cs[:-w]) / w
    out[:w] = np.asarray(x, dtype=float)[:w].cumsum() / np.arange(1, w + 1)
    return out


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Vectorised rolling std — pandas C backend, replaces per-bar Python loop."""
    return pd.Series(x).rolling(w, min_periods=2).std().fillna(0.0).to_numpy()


def _rolling_zscore(x: np.ndarray, w: int) -> np.ndarray:
    mu = _rolling_mean(x, w)
    sd = _rolling_std(x, w)
    return (x - mu) / np.where(sd > EPS, sd, 1.0)  # denominator never 0


def _ema(x: np.ndarray, period: int) -> np.ndarray:
    """EMA — inherently sequential, kept as loop (fast at C speed via numpy scalars)."""
    out = np.zeros_like(x, dtype=float)
    alpha = 2.0 / (period + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.zeros_like(close)
    avg_l = np.zeros_like(close)
    if len(close) < period:
        return np.full_like(close, 0.5)
    avg_g[period - 1] = np.mean(gain[:period])
    avg_l[period - 1] = np.mean(loss[:period])
    for i in range(period, len(close)):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i]) / period
    rs = np.where(avg_l > EPS, avg_g / np.maximum(avg_l, EPS), 1e6)
    rsi = 1.0 / (1.0 + rs)
    return np.clip(rsi, 0.0, 1.0)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_c = np.roll(close, 1)
    prev_c[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr = np.zeros_like(close)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _macd_hist(close: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> np.ndarray:
    e_fast = _ema(close, fast)
    e_slow = _ema(close, slow)
    macd = e_fast - e_slow
    signal = _ema(macd, sig)
    hist = macd - signal
    std = float(np.std(hist)) + EPS
    return np.clip(hist / std, -3.0, 3.0) / 3.0


def _bb_pct(close: np.ndarray, period: int = 20, n_std: float = 2.0) -> np.ndarray:
    mu = _rolling_mean(close, period)
    sd = _rolling_std(close, period)
    upper = mu + n_std * sd
    lower = mu - n_std * sd
    width = upper - lower
    return np.where(width > EPS, (close - lower) / np.maximum(width, EPS), 0.5)


def _adx_proxy(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """Simplified directional persistence proxy (not the full ADX algorithm)."""
    atr = _atr(high, low, close, period)
    dm_plus  = np.maximum(np.diff(high,  prepend=high[0]),  0.0)
    dm_minus = np.maximum(np.diff(-low,  prepend=-low[0]),  0.0)
    mask = dm_plus < dm_minus
    dm_plus[mask] = 0.0
    dm_minus[~mask] = 0.0
    di_plus  = _ema(dm_plus,  period) / (atr + EPS)
    di_minus = _ema(dm_minus, period) / (atr + EPS)
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus + EPS)
    return np.clip(_ema(dx, period), 0.0, 1.0)


def _stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised stochastic oscillator using pandas rolling max/min."""
    hi = pd.Series(high).rolling(k_period, min_periods=1).max().to_numpy()
    lo = pd.Series(low).rolling(k_period, min_periods=1).min().to_numpy()
    k = np.clip((close - lo) / (hi - lo + EPS), 0.0, 1.0)
    d = np.clip(pd.Series(k).rolling(d_period, min_periods=1).mean().to_numpy(), 0.0, 1.0)
    return k, d


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorised CCI using pandas rolling mean and mean-absolute-deviation."""
    tp = (high + low + close) / 3.0
    s = pd.Series(tp)
    mu = s.rolling(period, min_periods=1).mean().to_numpy()
    mad = s.rolling(period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    ).fillna(0.0).to_numpy()
    raw = np.where(mad > EPS, (tp - mu) / (0.015 * np.maximum(mad, EPS)), 0.0)
    return np.clip(raw / 100.0, -1.0, 1.0)


def _williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Vectorised Williams %R using pandas rolling max/min."""
    h = pd.Series(high).rolling(period, min_periods=1).max().to_numpy()
    l = pd.Series(low).rolling(period, min_periods=1).min().to_numpy()
    return np.clip(1.0 - (h - close) / (h - l + EPS), 0.0, 1.0)


def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorised Money Flow Index using pandas rolling sum."""
    tp = (high + low + close) / 3.0
    mf = tp * np.where(volume > 0, volume, 1.0)
    delta_tp = np.diff(tp, prepend=tp[0])
    pos_mf = np.where(delta_tp > 0, mf, 0.0)
    neg_mf = np.where(delta_tp < 0, mf, 0.0)
    pos_sum = pd.Series(pos_mf).rolling(period, min_periods=1).sum().to_numpy()
    neg_sum = pd.Series(neg_mf).rolling(period, min_periods=1).sum().to_numpy()
    mfr = pos_sum / (neg_sum + EPS)
    return np.clip(1.0 - 1.0 / (1.0 + mfr), 0.0, 1.0)


def _obv_momentum(close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    obv = np.cumsum(np.where(delta > 0, volume, np.where(delta < 0, -volume, 0.0)))
    return _rolling_zscore(obv, period)


def _cmf(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorised Chaikin Money Flow using pandas rolling sum."""
    hl = high - low
    clv = ((close - low) - (high - close)) / np.where(hl > EPS, hl, 1.0)
    clv = np.where(hl > EPS, clv, 0.0)
    mfv = clv * volume
    mfv_sum = pd.Series(mfv).rolling(period, min_periods=1).sum().to_numpy()
    vol_sum = pd.Series(volume.astype(float)).rolling(period, min_periods=1).sum().to_numpy()
    return np.clip(np.where(vol_sum > EPS, mfv_sum / vol_sum, 0.0), -1.0, 1.0)


def _keltner_pos(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                 period: int = 20, atr_mult: float = 2.0) -> np.ndarray:
    basis = _ema(close, period)
    atr = _atr(high, low, close, period)
    upper = basis + atr_mult * atr
    lower = basis - atr_mult * atr
    return np.clip((close - lower) / (upper - lower + EPS), 0.0, 1.0)


def _linreg(x: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling linear regression — vectorised via stride tricks.
    Returns (slope_normalised, R²).
    """
    N = len(x)
    slope_out = np.zeros(N)
    r2_out    = np.zeros(N)
    if N < period:
        return slope_out, r2_out
    from numpy.lib.stride_tricks import sliding_window_view
    wins   = sliding_window_view(x.astype(float), period)   # (M, period)
    t      = np.arange(period, dtype=float)
    t_dev  = t - t.mean()
    t_var  = float(np.var(t)) + EPS
    y_mean = wins.mean(axis=1)                               # (M,)
    y_dev  = wins - y_mean[:, np.newaxis]                    # (M, period)
    s_vals = (y_dev * t_dev).mean(axis=1) / t_var           # (M,)
    ss_tot = np.sum(y_dev ** 2, axis=1) + EPS
    y_pred = y_mean[:, np.newaxis] + s_vals[:, np.newaxis] * t_dev
    ss_res = np.sum((wins - y_pred) ** 2, axis=1)
    slope_out[period - 1:] = np.clip(
        s_vals / (np.abs(y_mean) + EPS) * 10.0, -1.0, 1.0)
    r2_out[period - 1:]    = np.maximum(0.0, 1.0 - ss_res / ss_tot)
    return slope_out, r2_out


def _rolling_skew(x: np.ndarray, w: int) -> np.ndarray:
    """Vectorised rolling skewness — pandas C backend."""
    return (pd.Series(x).rolling(w, min_periods=w).skew()
              .fillna(0.0).clip(-3.0, 3.0).to_numpy() / 3.0)


def _rolling_kurt(x: np.ndarray, w: int) -> np.ndarray:
    """Vectorised rolling excess kurtosis — pandas C backend."""
    return (pd.Series(x).rolling(w, min_periods=w).kurt()
              .fillna(0.0).clip(-3.0, 3.0).to_numpy() / 3.0)


def _autocorr(x: np.ndarray, lag: int, window: int = 50) -> np.ndarray:
    """Vectorised rolling autocorrelation at given lag — pandas rolling corr."""
    s = pd.Series(x)
    return (s.rolling(window, min_periods=window)
             .corr(s.shift(lag))
             .fillna(0.0).clip(-1.0, 1.0).to_numpy())


def _pct_rank(x: np.ndarray, window: int = 20) -> np.ndarray:
    """Vectorised percentile rank via stride tricks."""
    N = len(x)
    out = np.zeros(N)
    from numpy.lib.stride_tricks import sliding_window_view
    if N >= window:
        wins = sliding_window_view(x, window)          # (N-window+1, window)
        out[window - 1:] = np.mean(wins <= wins[:, -1:], axis=1)
    for i in range(min(window - 1, N)):
        out[i] = float(np.mean(x[: i + 1] <= x[i]))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Block builders — return (N, block_dim) float32 array
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Shared: compute 50 raw M5-type features from aligned arrays
# ---------------------------------------------------------------------------

def _compute_50_raw_feats(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                           open_: np.ndarray, volume: np.ndarray,
                           timestamp_dt: Optional[List] = None,
                           pip_size: float = 0.0001,
                           spread: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute 50-dim raw feature vector per bar.
    Returns shape (N, 50).
    """
    N = len(close)
    out = np.zeros((N, 50), dtype=np.float32)

    if N < 4:
        return out

    spread_arr = spread if spread is not None else np.zeros(N)
    atr14   = _atr(high, low, close, 14)
    atr14_p = np.where(close > EPS, atr14 / close, 0.0)

    # --- F0-F23: existing 24 base features ---
    lr1  = _log_return(close, 1)
    lr2  = _log_return(close, 2)
    lr5  = _log_return(close, 5)
    vol10 = _realized_vol(close, 10)
    mom1 = np.diff(close, prepend=close[0]) / (close + EPS)
    tick_vol = np.log1p(np.maximum(volume, 0.0))
    hl_range = np.where(pip_size > EPS, (high - low) / pip_size, 0.0)
    hl_range_n = np.clip(hl_range / 100.0, 0.0, 1.0)
    hl_w = high - low + EPS
    body_pos  = np.clip((close - low) / hl_w, 0.0, 1.0)
    body2rng  = np.clip(np.abs(close - open_) / hl_w, 0.0, 1.0)
    upper_wick = np.clip((high - np.maximum(close, open_)) / hl_w, 0.0, 1.0)
    lower_wick = np.clip((np.minimum(close, open_) - low) / hl_w, 0.0, 1.0)
    rsi14 = _rsi(close, 14)
    adx   = _adx_proxy(high, low, close, 14)
    macd  = _macd_hist(close)
    bb    = _bb_pct(close, 20)
    vol_ratio = np.zeros(N)
    vol_ma = _rolling_mean(volume.astype(float), 20)
    vol_ratio = np.where(vol_ma > EPS, volume / vol_ma, 1.0)
    vol_ratio = np.clip(vol_ratio, 0.0, 3.0) / 3.0
    rsi14_sl = np.diff(rsi14, prepend=rsi14[0])
    rsi14_sl = np.where(_rolling_std(rsi14, 10) > EPS,
                        rsi14_sl / (_rolling_std(rsi14, 10) + EPS), 0.0)
    ema5  = _ema(close, 5)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema_5_20  = np.clip((ema5  / (ema20 + EPS) - 1.0) * 50.0, -1.0, 1.0)
    ema_20_50 = np.clip((ema20 / (ema50 + EPS) - 1.0) * 50.0, -1.0, 1.0)

    # Session time encoding — vectorised
    if timestamp_dt is not None and len(timestamp_dt) == N:
        hours = np.array([t.hour + t.minute / 60.0 for t in timestamp_dt])
        hour_int = np.array([t.hour for t in timestamp_dt], dtype=int)
    else:
        hours    = np.zeros(N)
        hour_int = np.zeros(N, dtype=int)
    sin_h = np.sin(2 * np.pi * hours / 24.0)
    cos_h = np.cos(2 * np.pi * hours / 24.0)

    spread_n     = np.clip(spread_arr / (atr14 + EPS), 0.0, 2.0) / 2.0
    session_flag = np.where((hour_int >= 7) & (hour_int < 17), 1.0, 0.0)

    out[:, 0]  = lr1.astype(np.float32)
    out[:, 1]  = lr2.astype(np.float32)
    out[:, 2]  = lr5.astype(np.float32)
    out[:, 3]  = vol10.astype(np.float32)
    out[:, 4]  = spread_n.astype(np.float32)
    out[:, 5]  = np.clip(mom1, -0.01, 0.01).astype(np.float32) / 0.01
    out[:, 6]  = (tick_vol / 15.0).clip(0, 1).astype(np.float32)
    out[:, 7]  = session_flag.astype(np.float32)
    out[:, 8]  = hl_range_n.astype(np.float32)
    out[:, 9]  = body_pos.astype(np.float32)
    out[:, 10] = body2rng.astype(np.float32)
    out[:, 11] = upper_wick.astype(np.float32)
    out[:, 12] = lower_wick.astype(np.float32)
    out[:, 13] = rsi14.astype(np.float32)
    out[:, 14] = np.clip(atr14_p * 100.0, 0.0, 2.0).astype(np.float32) / 2.0
    out[:, 15] = adx.astype(np.float32)
    out[:, 16] = macd.astype(np.float32)
    out[:, 17] = np.clip(bb, 0.0, 1.0).astype(np.float32)
    out[:, 18] = vol_ratio.astype(np.float32)
    out[:, 19] = np.clip(rsi14_sl, -1.0, 1.0).astype(np.float32)
    out[:, 20] = ema_5_20.astype(np.float32)
    out[:, 21] = ema_20_50.astype(np.float32)
    out[:, 22] = sin_h.astype(np.float32)
    out[:, 23] = cos_h.astype(np.float32)

    # --- F24-F49: new extended features ---
    lr10  = _log_return(close, 10)
    lr20  = _log_return(close, 20)
    vol20 = _realized_vol(close, 20)
    zsc20 = _rolling_zscore(close, 20)
    mom10 = np.where(close[10:].size > 0,
                     np.concatenate([np.zeros(10),
                                     (close[10:] - close[:-10]) / (close[:-10] + EPS)]),
                     np.zeros(N))
    rsi7 = _rsi(close, 7)
    stoch_k, stoch_d = _stochastic(high, low, close, 14, 3)
    cci14   = _cci(high, low, close, 14)
    wpr14   = _williams_r(high, low, close, 14)
    mfi14   = _mfi(high, low, close, volume, 14)
    obv_mom = np.tanh(_obv_momentum(close, volume, 20))
    cmf14   = _cmf(high, low, close, volume, 14)
    kelt    = _keltner_pos(close, high, low, 20)
    lr_slope, lr_r2 = _linreg(close, 14)
    sk10 = _rolling_skew(_log_return(close, 1), 10)
    kt10 = _rolling_kurt(_log_return(close, 1), 10)
    ac1  = _autocorr(_log_return(close, 1), 1, 30)
    ac2  = _autocorr(_log_return(close, 1), 2, 30)
    ac5  = _autocorr(_log_return(close, 1), 5, 30)
    atr5 = _atr(high, low, close, 5)
    hl_ratio = np.where(atr14 > EPS, atr5 / (atr14 + EPS), 1.0)
    hl_ratio = np.clip(hl_ratio, 0.0, 2.0) / 2.0

    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    gap_open   = np.clip((open_ - prev_close) / (atr14 + EPS), -2.0, 2.0) / 2.0

    # Body momentum: weighted sum of last 5 body directions
    body_dir = np.sign(close - open_)
    body_mom5 = np.zeros(N)
    for lag in range(1, 6):
        w = (6 - lag) / 15.0
        body_mom5 += w * np.roll(body_dir, lag)
    body_mom5 = np.clip(body_mom5, -1.0, 1.0)

    # Volume–price correlation over 10 bars — vectorised via pandas rolling
    lr1_arr = lr1  # already computed above
    vol_s   = pd.Series(volume.astype(float))
    pabs_s  = pd.Series(np.abs(lr1_arr))
    vol_price_corr = (vol_s.rolling(10, min_periods=10).corr(pabs_s)
                          .fillna(0.0).clip(-1.0, 1.0).to_numpy())

    pct_rk = _pct_rank(close, 20)

    out[:, 24] = np.clip(lr10 / 0.02, -1, 1).astype(np.float32)
    out[:, 25] = np.clip(lr20 / 0.03, -1, 1).astype(np.float32)
    out[:, 26] = np.clip(vol20 / (vol10 + EPS) - 1.0, -1, 1).astype(np.float32)
    out[:, 27] = np.clip(zsc20, -3, 3).astype(np.float32) / 3.0
    out[:, 28] = np.clip(mom10 / 0.015, -1, 1).astype(np.float32)
    out[:, 29] = rsi7.astype(np.float32)
    out[:, 30] = stoch_k.astype(np.float32)
    out[:, 31] = stoch_d.astype(np.float32)
    out[:, 32] = cci14.astype(np.float32)
    out[:, 33] = wpr14.astype(np.float32)
    out[:, 34] = mfi14.astype(np.float32)
    out[:, 35] = obv_mom.astype(np.float32)
    out[:, 36] = cmf14.astype(np.float32)
    out[:, 37] = kelt.astype(np.float32)
    out[:, 38] = lr_slope.astype(np.float32)
    out[:, 39] = lr_r2.astype(np.float32)
    out[:, 40] = sk10.astype(np.float32)
    out[:, 41] = kt10.astype(np.float32)
    out[:, 42] = ac1.astype(np.float32)
    out[:, 43] = ac2.astype(np.float32)
    out[:, 44] = ac5.astype(np.float32)
    out[:, 45] = hl_ratio.astype(np.float32)
    out[:, 46] = gap_open.astype(np.float32)
    out[:, 47] = body_mom5.astype(np.float32)
    out[:, 48] = vol_price_corr.astype(np.float32)
    out[:, 49] = pct_rk.astype(np.float32)

    return out


def _apply_m5_windows(raw50: np.ndarray) -> np.ndarray:
    """
    Apply 4 statistical transforms to a (N,50) raw feature array.
    Returns (N, 200).

    mk4.3: matches the EA-side Encode() exactly. The EA computes
        slot 0..49        raw value of feature f
        slot 50..99       rolling mean over the last ~20 bars
        slot 100..149     rolling std  over the last ~20 bars
        slot 150..199     delta = current - oldest-of-last-20

    Earlier mk4.x versions emitted 400 dims (8 transforms) which the EA
    never matched — so 50% of the model input was zero-padded at live
    inference. Trimming to 200 closes the parity gap.
    """
    N, F = raw50.shape   # F = RAW_FEATURES = 50
    out = np.zeros((N, F * 4), dtype=np.float32)
    for f in range(F):
        col = raw50[:, f].astype(float)
        out[:, f]         = raw50[:, f]                          # raw
        out[:, F + f]     = _rolling_mean(col, 20)               # mean20
        out[:, 2*F + f]   = _rolling_std(col, 20)                # std20
        # delta = current - value 20 bars ago (matches EA last - oldest)
        delta = np.zeros_like(col)
        delta[20:] = col[20:] - col[:-20]
        out[:, 3*F + f]   = delta
    return out


# ---------------------------------------------------------------------------
# Shared: compute 20 raw HTF features (H4/H8/D1 style)
# ---------------------------------------------------------------------------

def _compute_20_raw_htf(close: np.ndarray, high: np.ndarray,
                         low: np.ndarray, open_: np.ndarray,
                         volume: np.ndarray) -> np.ndarray:
    """Returns (N, 20) raw features for H4/H8/D1 timeframes."""
    N = len(close)
    out = np.zeros((N, 20), dtype=np.float32)
    if N < 4:
        return out

    atr14  = _atr(high, low, close, 14)
    ema20  = _ema(close, 20)
    ema50  = _ema(close, 50)

    out[:, 0] = np.clip(_log_return(close, 1), -0.05, 0.05).astype(np.float32) / 0.05
    out[:, 1] = np.clip(_log_return(close, 3), -0.08, 0.08).astype(np.float32) / 0.08
    out[:, 2] = np.clip(_realized_vol(close, 5), 0, 0.05).astype(np.float32) / 0.05
    out[:, 3] = np.clip(_atr(high, low, close, 14) / (close + EPS) * 100.0,
                        0.0, 2.0).astype(np.float32) / 2.0
    out[:, 4] = _rsi(close, 14).astype(np.float32)
    out[:, 5] = np.clip((ema20 / (ema50 + EPS) - 1.0) * 50.0, -1.0, 1.0).astype(np.float32)
    out[:, 6] = np.clip((_ema(close, 5) / (ema20 + EPS) - 1.0) * 50.0,
                        -1.0, 1.0).astype(np.float32)
    out[:, 7] = np.clip((high - low) / (close + EPS) * 100.0, 0.0, 2.0).astype(np.float32) / 2.0
    out[:, 8] = np.clip(_log_return(close, 5), -0.10, 0.10).astype(np.float32) / 0.10
    out[:, 9] = np.clip(_log_return(close, 20), -0.15, 0.15).astype(np.float32) / 0.15
    out[:, 10] = np.clip(_realized_vol(close, 10), 0, 0.05).astype(np.float32) / 0.05
    out[:, 11] = _rsi(close, 14).astype(np.float32)   # duplicate kept for window averaging
    out[:, 12] = np.clip(atr14 / (close + EPS) * 100.0, 0, 2).astype(np.float32) / 2.0
    out[:, 13] = _macd_hist(close).astype(np.float32)
    out[:, 14] = np.clip(_bb_pct(close, 20), 0, 1).astype(np.float32)
    hl_w = high - low + EPS
    out[:, 15] = np.clip((close - open_) / hl_w, -1, 1).astype(np.float32)
    lr_sl, lr_r2 = _linreg(close, 10)
    out[:, 16] = lr_sl.astype(np.float32)
    vol_ma = _rolling_mean(volume.astype(float), 20)
    out[:, 17] = np.clip(volume / (vol_ma + EPS), 0, 3).astype(np.float32) / 3.0
    mom5 = np.concatenate([np.zeros(5),
                            (close[5:] - close[:-5]) / (close[:-5] + EPS)])
    out[:, 18] = np.clip(mom5, -0.05, 0.05).astype(np.float32) / 0.05
    out[:, 19] = lr_r2.astype(np.float32)
    return out


def _apply_6_windows_htf(raw20: np.ndarray,
                          mean_w1: int = 4,
                          mean_w2: int = 20) -> np.ndarray:
    """Apply 6 statistical transforms to (N,20): raw, mean_w1, std_w1, mean_w2, std_w2, delta1 → (N,120)."""
    N, F = raw20.shape  # F=20
    out = np.zeros((N, F * 6), dtype=np.float32)
    for f in range(F):
        col = raw20[:, f].astype(float)
        out[:, f]       = raw20[:, f]                    # raw
        out[:, F + f]   = _rolling_mean(col, mean_w1)    # mean_w1
        out[:, 2*F + f] = _rolling_std(col, mean_w1)     # std_w1
        out[:, 3*F + f] = _rolling_mean(col, mean_w2)    # mean_w2
        out[:, 4*F + f] = _rolling_std(col, mean_w2)     # std_w2
        out[:, 5*F + f] = np.diff(col, prepend=col[0])    # delta1
    return out


def _compute_30_raw_h1(close: np.ndarray, high: np.ndarray,
                        low: np.ndarray, open_: np.ndarray,
                        volume: np.ndarray) -> np.ndarray:
    """Returns (N, 30) raw features for H1 timeframe."""
    base20 = _compute_20_raw_htf(close, high, low, open_, volume)
    N = len(close)
    ext = np.zeros((N, 10), dtype=np.float32)
    ext[:, 0] = _adx_proxy(high, low, close, 14).astype(np.float32)
    ext[:, 1], _ = [x.astype(np.float32) for x in _stochastic(high, low, close, 14, 3)]
    ext[:, 2] = _cci(high, low, close, 14).astype(np.float32)
    # session flag (0=off, 1=London, 0.7=NY only, 0.5=Asia)
    ext[:, 3] = np.zeros(N, dtype=np.float32)  # will be patched by caller if timestamps available
    _, lr_r2 = _linreg(close, 10)
    ext[:, 4] = lr_r2.astype(np.float32)
    ext[:, 5] = _autocorr(_log_return(close, 1), 1, 20).astype(np.float32)
    ext[:, 6] = _rolling_skew(_log_return(close, 1), 10).astype(np.float32)
    ext[:, 7] = _rolling_kurt(_log_return(close, 1), 10).astype(np.float32)
    # VWAP deviation (approximate using daily typical price mean)
    tp = (high + low + close) / 3.0
    vwap_approx = _rolling_mean(tp, 24)   # 24 H1 bars = 1 day
    ext[:, 8] = np.clip((close - vwap_approx) / (vwap_approx + EPS) * 100.0,
                        -2.0, 2.0).astype(np.float32) / 2.0
    ext[:, 9] = np.clip(_rolling_zscore(close, 20), -3.0, 3.0).astype(np.float32) / 3.0
    return np.concatenate([base20, ext], axis=1)


def _apply_7_windows_h1(raw30: np.ndarray) -> np.ndarray:
    """Apply 7 transforms to (N,30): raw, mean4, std4, mean20, std20, delta1, zscore20 → (N,210)."""
    N, F = raw30.shape  # F=30
    out = np.zeros((N, F * 7), dtype=np.float32)
    for f in range(F):
        col = raw30[:, f].astype(float)
        out[:, f]       = raw30[:, f]                    # raw
        out[:, F + f]   = _rolling_mean(col, 4)           # mean4
        out[:, 2*F + f] = _rolling_std(col, 4)            # std4
        out[:, 3*F + f] = _rolling_mean(col, 20)          # mean20
        out[:, 4*F + f] = _rolling_std(col, 20)           # std20
        out[:, 5*F + f] = np.diff(col, prepend=col[0])    # delta1
        out[:, 6*F + f] = np.clip(
                              _rolling_zscore(col, 20), -3.0, 3.0) / 3.0  # zscore20
    return out


# ---------------------------------------------------------------------------
# Block 0: M5
# ---------------------------------------------------------------------------

def _build_m5_block(bars: pd.DataFrame, pip_size: float) -> np.ndarray:
    close  = bars["close"].to_numpy(dtype=float)
    high   = bars["high"].to_numpy(dtype=float)
    low    = bars["low"].to_numpy(dtype=float)
    open_  = bars["open"].to_numpy(dtype=float)
    vol    = bars.get("tick_volume", bars.get("volume",
                pd.Series(np.ones(len(bars))))).to_numpy(dtype=float)
    spread = bars["spread"].to_numpy(dtype=float) * pip_size \
             if "spread" in bars.columns else np.zeros(len(bars))

    timestamps = None
    if "time" in bars.columns:
        t_col = bars["time"]
        if hasattr(t_col.iloc[0], "to_pydatetime"):
            timestamps = [t.to_pydatetime().replace(tzinfo=None) for t in t_col]
        else:
            timestamps = list(t_col)

    raw50 = _compute_50_raw_feats(close, high, low, open_, vol, timestamps, pip_size, spread)
    return _apply_m5_windows(raw50)    # (N, 400)


# ---------------------------------------------------------------------------
# Block 1: H1
# ---------------------------------------------------------------------------

def _build_h1_block(m5_bars: pd.DataFrame, h1_df: Optional[pd.DataFrame],
                    N: int) -> np.ndarray:
    """Returns (N, 150). Aligned by forward-fill to M5 bar timestamps."""
    if h1_df is None or len(h1_df) < 5:
        return np.zeros((N, H1_DIM), dtype=np.float32)

    close  = h1_df["close"].to_numpy(dtype=float)
    high   = h1_df["high"].to_numpy(dtype=float)
    low    = h1_df["low"].to_numpy(dtype=float)
    open_  = h1_df["open"].to_numpy(dtype=float)
    vol    = h1_df.get("tick_volume", h1_df.get("volume",
                pd.Series(np.ones(len(h1_df))))).to_numpy(dtype=float)

    raw30 = _compute_30_raw_h1(close, high, low, open_, vol)
    blk210 = _apply_7_windows_h1(raw30)  # (n_h1, 210) - increased from 150

    # Align to M5 timestamps by forward-fill
    if "time" not in m5_bars.columns or "time" not in h1_df.columns:
        return np.zeros((N, 210), dtype=np.float32)  # Updated dimension

    m5_times = pd.to_datetime(m5_bars["time"], utc=True)
    h1_times = pd.to_datetime(h1_df["time"],   utc=True)

    # Align by searchsorted (O(N log K) vs O(N) Python loop)
    m5_arr = m5_times.to_numpy(dtype="datetime64[ns]")
    h1_arr = h1_times.to_numpy(dtype="datetime64[ns]")
    idx    = np.searchsorted(h1_arr, m5_arr, side="right") - 1
    valid  = idx >= 0
    idx    = np.clip(idx, 0, len(blk210) - 1)  # Updated to blk210
    out    = np.where(valid[:, np.newaxis], blk210[idx], 0.0).astype(np.float32)  # Updated to blk210
    return out


# ---------------------------------------------------------------------------
# Block 2/3/4: H4 / H8 / D1  (all use same 20→80 pattern)
# ---------------------------------------------------------------------------

def _build_htf_block(m5_bars: pd.DataFrame, htf_df: Optional[pd.DataFrame],
                     N: int, block_dim: int,
                     mean_w1: int = 4, mean_w2: int = 20) -> np.ndarray:
    """Generic HTF block builder. Returns (N, block_dim) aligned to M5."""
    if htf_df is None or len(htf_df) < 5:
        return np.zeros((N, block_dim), dtype=np.float32)

    close  = htf_df["close"].to_numpy(dtype=float)
    high   = htf_df["high"].to_numpy(dtype=float)
    low    = htf_df["low"].to_numpy(dtype=float)
    open_  = htf_df["open"].to_numpy(dtype=float)
    vol    = htf_df.get("tick_volume", htf_df.get("volume",
                pd.Series(np.ones(len(htf_df))))).to_numpy(dtype=float)

    raw20  = _compute_20_raw_htf(close, high, low, open_, vol)
    blk    = _apply_6_windows_htf(raw20, mean_w1, mean_w2)  # (n_htf, 120) - increased from 80

    if "time" not in m5_bars.columns or "time" not in htf_df.columns:
        return np.zeros((N, 120), dtype=np.float32)  # Updated dimension to 120

    m5_times  = pd.to_datetime(m5_bars["time"],  utc=True)
    htf_times = pd.to_datetime(htf_df["time"],   utc=True)

    # Align by searchsorted
    m5_arr  = m5_times.to_numpy(dtype="datetime64[ns]")
    htf_arr = htf_times.to_numpy(dtype="datetime64[ns]")
    idx     = np.searchsorted(htf_arr, m5_arr, side="right") - 1
    valid   = idx >= 0
    idx     = np.clip(idx, 0, len(blk) - 1)
    out     = np.where(valid[:, np.newaxis], blk[idx], 0.0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Block 5: Spectral (60-dim)
# ---------------------------------------------------------------------------

def _hurst_rs(x: np.ndarray) -> float:
    """R/S Hurst exponent estimate over a series of log-returns."""
    if len(x) < 8:
        return 0.5
    n = len(x)
    mean = np.mean(x)
    z = np.cumsum(x - mean)
    R = float(np.max(z) - np.min(z))
    S = float(np.std(x, ddof=1)) + EPS
    if R < EPS:
        return 0.5
    return max(0.0, min(1.0, np.log(R / S) / np.log(n / 2)))


def _approx_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Approximate entropy (ApEn) — vectorised via stride tricks.
    Called on short segments (≤20 bars) so matrices stay tiny.
    """
    N = len(x)
    if N < m + 2:
        return 0.0
    r = r_factor * (np.std(x) + EPS)

    def _phi(m_: int) -> float:
        from numpy.lib.stride_tricks import sliding_window_view
        templates = sliding_window_view(x, m_)           # (K, m_)
        # Chebyshev distance between all template pairs
        diffs = np.abs(templates[:, np.newaxis, :] - templates[np.newaxis, :, :])
        matches = (diffs.max(axis=2) <= r).astype(float)  # (K, K)
        return float(matches.mean())

    phi_m  = _phi(m)
    phi_m1 = _phi(m + 1)
    return max(0.0, np.log(phi_m + EPS) - np.log(phi_m1 + EPS))


def _perm_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Permutation entropy."""
    N = len(x)
    if N < order * delay:
        return 0.0
    from itertools import permutations
    from math import factorial
    motifs: Dict[tuple, int] = {}
    for i in range(N - (order - 1) * delay):
        pattern = tuple(np.argsort(x[i:i + order * delay:delay]))
        motifs[pattern] = motifs.get(pattern, 0) + 1
    total = sum(motifs.values())
    probs = np.array(list(motifs.values())) / total
    return float(-np.sum(probs * np.log(probs + EPS)) / np.log(factorial(order)))


def _build_spectral_block(bars: pd.DataFrame) -> np.ndarray:
    """Returns (N, 60)."""
    N = len(bars)
    out = np.zeros((N, SPECTRAL_DIM), dtype=np.float32)
    close = bars["close"].to_numpy(dtype=float)

    FFT_LEN = 64
    FIB_LAGS = [1, 2, 3, 5, 8, 13, 21, 34]

    for i in range(FFT_LEN, N):
        seg = close[i - FFT_LEN:i]
        seg = seg - np.mean(seg)  # detrend
        seg = seg * np.hanning(FFT_LEN)  # window

        fft_c = np.fft.rfft(seg)
        mags  = np.abs(fft_c[1:])  # skip DC
        phases = np.angle(fft_c[1:]) / np.pi  # -1..1

        total_power = np.sum(mags ** 2) + EPS

        # Top-20 FFT magnitudes (normalised)
        top20_idx  = np.argsort(mags)[-20:][::-1]
        top20_mags = mags[top20_idx] / (np.max(mags) + EPS)
        out[i, :20] = top20_mags.astype(np.float32)

        # Top-10 FFT phases
        top10_idx = top20_idx[:10]
        out[i, 20:30] = phases[top10_idx].astype(np.float32)

        # Fibonacci autocorrelations
        lr = np.diff(np.log(seg[seg > 0] + EPS)) if np.min(seg) < 0 else np.diff(np.log(seg + EPS))
        for k, lag in enumerate(FIB_LAGS):
            if lag < len(lr):
                mu = np.mean(lr)
                var = np.var(lr)
                if var > EPS:
                    out[i, 30 + k] = float(np.mean(
                        (lr[:-lag] - mu) * (lr[lag:] - mu)) / var)

        # Autocorrelation lags 1-5 (replaces per-bar statsmodels PACF call)
        lr_full = np.diff(np.log(np.maximum(close[i - FFT_LEN:i], EPS)))
        lr_mu   = np.mean(lr_full)
        lr_var  = np.var(lr_full) + EPS
        for _lag_i, _lag in enumerate(range(1, 6)):
            if _lag < len(lr_full):
                _cov = np.mean(
                    (lr_full[:-_lag] - lr_mu) * (lr_full[_lag:] - lr_mu))
                out[i, 38 + _lag_i] = float(np.clip(_cov / lr_var, -1, 1))

        # Hurst exponent
        lr_full = np.diff(np.log(close[max(0, i - FFT_LEN):i] + EPS))
        out[i, 43] = _hurst_rs(lr_full)

        # Spectral entropy
        p_spec = mags[:20] ** 2 / total_power
        p_spec = p_spec / (np.sum(p_spec) + EPS)
        out[i, 44] = float(-np.sum(p_spec * np.log(p_spec + EPS)) / np.log(20))

        # Dominant period (normalised 0..1 over range 2..FFT_LEN//2)
        dom_idx = np.argmax(mags) + 1  # +1 for DC skip
        dom_period = FFT_LEN / (dom_idx + EPS)
        out[i, 45] = float(np.clip((dom_period - 2) / (FFT_LEN // 2 - 2), 0, 1))

        # Approximate entropy (fast, 20-bar window)
        seg20 = close[i - 20:i] if i >= 20 else close[:i]
        out[i, 46] = np.clip(_approx_entropy(seg20), 0, 2) / 2.0

        # Sample entropy (simplified: correlation dimension proxy)
        out[i, 47] = float(np.clip(np.std(seg20) / (np.mean(np.abs(seg20)) + EPS), 0, 2) / 2.0)

        # Permutation entropy
        out[i, 48] = _perm_entropy(seg20[:20], order=3)

        # Slots 49-59 reserved, left as 0

    return out


# ---------------------------------------------------------------------------
# Block 6: Pattern (50-dim)
# ---------------------------------------------------------------------------

def _build_pattern_block(bars: pd.DataFrame, pip_size: float) -> np.ndarray:
    """Returns (N, 50). 30 candlestick binary + 20 price action features."""
    N = len(bars)
    out = np.zeros((N, PATTERN_DIM), dtype=np.float32)

    close  = bars["close"].to_numpy(dtype=float)
    high   = bars["high"].to_numpy(dtype=float)
    low    = bars["low"].to_numpy(dtype=float)
    open_  = bars["open"].to_numpy(dtype=float)
    vol    = bars.get("tick_volume", bars.get("volume",
                pd.Series(np.ones(N)))).to_numpy(dtype=float)

    atr20  = _atr(high, low, close, 20)
    hl_w   = high - low + EPS
    body   = close - open_
    body_s = np.abs(body)

    for i in range(2, N):
        c, h, l, o = close[i], high[i], low[i], open_[i]
        c1, h1, l1, o1 = close[i-1], high[i-1], low[i-1], open_[i-1]
        c2, h2, l2, o2 = close[i-2], high[i-2], low[i-2], open_[i-2]
        bdy  = c - o;        bdy_abs = abs(bdy)
        bdy1 = c1 - o1;      bdy1_abs = abs(bdy1)
        hw   = h - l + EPS

        # Candlestick patterns (indices 0-29)
        # 0 Doji
        out[i, 0] = 1.0 if bdy_abs / hw < 0.1 else 0.0
        # 1 Hammer (small body at top, long lower wick, in downtrend)
        out[i, 1] = 1.0 if (bdy_abs / hw < 0.3 and
                              (min(c, o) - l) / hw > 0.6 and
                              (h - max(c, o)) / hw < 0.1) else 0.0
        # 2 Inverted Hammer
        out[i, 2] = 1.0 if (bdy_abs / hw < 0.3 and
                              (h - max(c, o)) / hw > 0.6 and
                              (min(c, o) - l) / hw < 0.1) else 0.0
        # 3 Hanging Man (hammer at top — caller must check trend)
        out[i, 3] = out[i, 1]
        # 4 Shooting Star (inv. hammer at top)
        out[i, 4] = out[i, 2]
        # 5 Bullish Engulfing
        out[i, 5] = 1.0 if (bdy > 0 and bdy1 < 0 and c > o1 and o < c1) else 0.0
        # 6 Bearish Engulfing
        out[i, 6] = 1.0 if (bdy < 0 and bdy1 > 0 and c < o1 and o > c1) else 0.0
        # 7 Bullish Harami
        out[i, 7] = 1.0 if (bdy > 0 and bdy1 < 0 and
                             o > c1 and c < o1 and bdy_abs < bdy1_abs) else 0.0
        # 8 Bearish Harami
        out[i, 8] = 1.0 if (bdy < 0 and bdy1 > 0 and
                             o < c1 and c > o1 and bdy_abs < bdy1_abs) else 0.0
        # 9 Morning Star (3-bar)
        if i >= 2:
            out[i, 9] = 1.0 if (bdy2 := (c2 - o2)) < 0 and \
                                 abs(bdy1) / (h1 - l1 + EPS) < 0.3 and \
                                 bdy > 0 and c > (o2 + c2) / 2 else 0.0
        # 10 Evening Star
        if i >= 2:
            out[i, 10] = 1.0 if (c2 - o2) > 0 and \
                                  abs(bdy1) / (h1 - l1 + EPS) < 0.3 and \
                                  bdy < 0 and c < (o2 + c2) / 2 else 0.0
        # 11 Three White Soldiers
        out[i, 11] = 1.0 if (bdy > 0 and bdy1 > 0 and (c2 - o2) > 0 and
                              c > c1 and c1 > c2) else 0.0
        # 12 Three Black Crows
        out[i, 12] = 1.0 if (bdy < 0 and bdy1 < 0 and (c2 - o2) < 0 and
                              c < c1 and c1 < c2) else 0.0
        # 13 Dragonfly Doji
        out[i, 13] = 1.0 if (bdy_abs / hw < 0.05 and
                              (min(c, o) - l) / hw > 0.7) else 0.0
        # 14 Gravestone Doji
        out[i, 14] = 1.0 if (bdy_abs / hw < 0.05 and
                              (h - max(c, o)) / hw > 0.7) else 0.0
        # 15 Spinning Top
        out[i, 15] = 1.0 if (bdy_abs / hw < 0.35 and
                              (min(c, o) - l) / hw > 0.2 and
                              (h - max(c, o)) / hw > 0.2) else 0.0
        # 16 Marubozu Bull
        out[i, 16] = 1.0 if (bdy > 0 and
                              (min(c, o) - l) / hw < 0.05 and
                              (h - max(c, o)) / hw < 0.05) else 0.0
        # 17 Marubozu Bear
        out[i, 17] = 1.0 if (bdy < 0 and
                              (min(c, o) - l) / hw < 0.05 and
                              (h - max(c, o)) / hw < 0.05) else 0.0
        # 18 Inside Bar
        out[i, 18] = 1.0 if (h < h1 and l > l1) else 0.0
        # 19 Outside Bar
        out[i, 19] = 1.0 if (h > h1 and l < l1) else 0.0
        # 20 Tweezer Top
        out[i, 20] = 1.0 if (abs(h - h1) < 0.1 * atr20[i] and
                              bdy < 0 and bdy1 > 0) else 0.0
        # 21 Tweezer Bottom
        out[i, 21] = 1.0 if (abs(l - l1) < 0.1 * atr20[i] and
                              bdy > 0 and bdy1 < 0) else 0.0
        # 22 Dark Cloud Cover
        out[i, 22] = 1.0 if (bdy1 > 0 and bdy < 0 and
                              o > h1 and c < (o1 + c1) / 2) else 0.0
        # 23 Piercing Line
        out[i, 23] = 1.0 if (bdy1 < 0 and bdy > 0 and
                              o < l1 and c > (o1 + c1) / 2) else 0.0
        # 24 Bullish Kicker
        out[i, 24] = 1.0 if (bdy1 < 0 and bdy > 0 and o > o1) else 0.0
        # 25 Bearish Kicker
        out[i, 25] = 1.0 if (bdy1 > 0 and bdy < 0 and o < o1) else 0.0
        # 26 High Wave
        out[i, 26] = 1.0 if (bdy_abs / hw < 0.25 and
                              (h - max(c, o)) / hw > 0.25 and
                              (min(c, o) - l) / hw > 0.25) else 0.0
        # 27 Gap Up
        out[i, 27] = 1.0 if (l > h1) else 0.0
        # 28 Gap Down
        out[i, 28] = 1.0 if (h < l1) else 0.0
        # 29 Pin Bar (hammer or shooting star body ≤ 0.25 range, wick ≥ 0.6)
        out[i, 29] = 1.0 if (bdy_abs / hw < 0.25 and
                              max((min(c, o) - l) / hw,
                                  (h - max(c, o)) / hw) >= 0.6) else 0.0

    # Price action features (indices 30-49) — rolling min/max vectorised
    _low_s  = pd.Series(low)
    _high_s = pd.Series(high)
    supp20 = _low_s.rolling(20,  min_periods=1).min().to_numpy()
    res20  = _high_s.rolling(20, min_periods=1).max().to_numpy()
    supp50 = _low_s.rolling(50,  min_periods=1).min().to_numpy()
    res50  = _high_s.rolling(50, min_periods=1).max().to_numpy()
    r20    = res20  - supp20  + EPS
    r50    = res50  - supp50  + EPS

    out[:, 30] = np.clip((close - supp20) / r20, 0, 1).astype(np.float32)
    out[:, 31] = np.clip((res20 - close) / r20, 0, 1).astype(np.float32)
    out[:, 32] = np.clip((close - supp50) / r50, 0, 1).astype(np.float32)
    out[:, 33] = np.clip((res50 - close) / r50, 0, 1).astype(np.float32)

    # Pivot point position
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    pp_range = r1 - s1 + EPS
    out[:, 34] = np.clip((close - s1) / pp_range, 0, 1).astype(np.float32)

    # Trend strength / direction (linreg)
    lr_sl, lr_r2 = _linreg(close, 20)
    out[:, 35] = lr_r2.astype(np.float32)
    out[:, 36] = (lr_sl * lr_r2).astype(np.float32)   # sign × strength

    # 60-bar breakout flag
    supp60 = _low_s.rolling(60,  min_periods=1).min().to_numpy()
    res60  = _high_s.rolling(60, min_periods=1).max().to_numpy()
    at_res  = (close >= res60 * 0.999).astype(np.float32)
    at_supp = (close <= supp60 * 1.001).astype(np.float32)
    out[:, 37] = (at_res - at_supp).astype(np.float32)

    # ATR compression (5-bar vs 20-bar)
    atr5 = _atr(high, low, close, 5)
    out[:, 38] = np.clip(atr5 / (atr20 + EPS), 0, 2).astype(np.float32) / 2.0

    # Momentum divergence (RSI vs price direction)
    rsi14 = _rsi(close, 14)
    price_dir = np.sign(np.diff(close, prepend=close[0]))
    rsi_dir   = np.sign(np.diff(rsi14,  prepend=rsi14[0]))
    out[:, 39] = (rsi_dir - price_dir).astype(np.float32) / 2.0  # -1..0..1

    # Swing high/low proximity
    for i in range(5, N):
        w = min(i, 20)
        swing_h = np.max(high[i-w:i])
        swing_l = np.min(low[i-w:i])
        rng = swing_h - swing_l + EPS
        out[i, 40] = float(np.clip(1.0 - (high[i] - swing_h) / rng, 0, 1))
        out[i, 41] = float(np.clip(1.0 - (swing_l - low[i])  / rng, 0, 1))

    # Consecutive same-direction bars
    consec = np.zeros(N)
    for i in range(1, N):
        d = np.sign(close[i] - close[i-1])
        consec[i] = consec[i-1] + d if np.sign(consec[i-1]) == d else d
    out[:, 42] = np.clip(consec / 10.0, -1, 1).astype(np.float32)

    # ATR expansion (current vs 20-bar mean ATR)
    atr_mean = _rolling_mean(atr20, 20)
    out[:, 43] = np.clip(atr20 / (atr_mean + EPS) - 1.0, -1, 1).astype(np.float32)

    # Body-to-wick asymmetry (upper wick minus lower wick)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    out[:, 44] = np.clip((upper_wick - lower_wick) / (atr20 + EPS), -2, 2).astype(np.float32) / 2

    # Session open distance
    if "time" in bars.columns:
        times = bars["time"]
        session_open = np.zeros(N)
        current_day_open = open_[0]
        current_day = 0
        for i in range(N):
            t = times.iloc[i]
            d = t.day if hasattr(t, "day") else 0
            if d != current_day:
                current_day = d
                current_day_open = open_[i]
            session_open[i] = current_day_open
        out[:, 45] = np.clip((close - session_open) / (atr20 + EPS), -3, 3).astype(np.float32) / 3

    # --- LONG-specific features (slots 46-49) ---
    # Support level proximity: how close to recent lows
    for i in range(20, N):
        recent_low = np.min(low[i-20:i])
        support_dist = (close[i] - recent_low) / (atr20[i] + EPS)
        out[i, 46] = np.clip(support_dist, 0, 3).astype(np.float32) / 3.0

    # Volume accumulation: buying pressure when price near support
    vol_ma = _rolling_mean(vol.astype(float), 20)
    for i in range(20, N):
        recent_low = np.min(low[i-20:i])
        near_support = (close[i] - recent_low) < 0.5 * atr20[i]
        if near_support:
            vol_accum = vol[i] / (vol_ma[i] + EPS)
            out[i, 47] = np.clip(vol_accum, 0, 3).astype(np.float32) / 3.0

    # Buying pressure: ratio of up bars to down bars in recent window
    for i in range(10, N):
        up_bars = np.sum(close[i-10:i] > open_[i-10:i])
        down_bars = np.sum(close[i-10:i] < open_[i-10:i])
        total = up_bars + down_bars + EPS
        out[i, 48] = (up_bars / total).astype(np.float32)

    # Bullish divergence: price makes lower low but RSI makes higher low
    rsi14 = _rsi(close, 14)
    for i in range(30, N):
        price_ll = np.min(low[i-20:i])
        price_ll_idx = np.argmin(low[i-20:i]) + i - 20
        rsi_at_ll = rsi14[price_ll_idx]
        rsi_prev_min = np.min(rsi14[i-30:price_ll_idx])
        if price_ll < np.min(low[i-30:i-20]) and rsi_at_ll > rsi_prev_min:
            out[i, 49] = 1.0

    return out


# ---------------------------------------------------------------------------
# Block 7: Statistical/Regime (60-dim)
# ---------------------------------------------------------------------------

def _build_statreg_block(bars: pd.DataFrame) -> np.ndarray:
    """Returns (N, 60)."""
    N = len(bars)
    out = np.zeros((N, STAT_DIM), dtype=np.float32)

    close  = bars["close"].to_numpy(dtype=float)
    high   = bars["high"].to_numpy(dtype=float)
    low    = bars["low"].to_numpy(dtype=float)
    open_  = bars["open"].to_numpy(dtype=float)
    vol    = bars.get("tick_volume", bars.get("volume",
                pd.Series(np.ones(N)))).to_numpy(dtype=float)
    lr1 = _log_return(close, 1)

    # [0-1] Skewness at 10 and 20 bars
    out[:, 0] = _rolling_skew(lr1, 10).astype(np.float32)
    out[:, 1] = _rolling_skew(lr1, 20).astype(np.float32)
    # [2-3] Kurtosis at 10 and 20 bars
    out[:, 2] = _rolling_kurt(lr1, 10).astype(np.float32)
    out[:, 3] = _rolling_kurt(lr1, 20).astype(np.float32)

    # [4-5] Max drawdown at 10 and 20 bars — vectorised rolling peak
    close_s      = pd.Series(close)
    peak10       = close_s.rolling(10, min_periods=1).max().to_numpy()
    peak20       = close_s.rolling(20, min_periods=1).max().to_numpy()
    out[:, 4]    = np.clip((peak10 - close) / (peak10 + EPS), 0, 1).astype(np.float32)
    out[:, 5]    = np.clip((peak20 - close) / (peak20 + EPS), 0, 1).astype(np.float32)

    # [6] Calmar ratio proxy (20-bar return / max_dd)
    ret20 = np.where(close[20:].size > 0,
                     np.concatenate([np.zeros(20),
                                     (close[20:] - close[:-20]) / (close[:-20] + EPS)]),
                     np.zeros(N))
    out[:, 6] = np.clip(ret20 / (out[:, 5] + 0.001), -2, 2).astype(np.float32) / 2

    # [7] Gain/loss ratio (20-bar) — vectorised rolling mean of positive/negative returns
    lr1_s     = pd.Series(lr1)
    mean_gain = lr1_s.clip(lower=0).rolling(20, min_periods=1).mean().to_numpy()
    mean_loss = (-lr1_s.clip(upper=0)).rolling(20, min_periods=1).mean().to_numpy()
    glr       = mean_gain / (mean_loss + EPS)
    out[:, 7] = np.clip(glr / 3.0, 0, 1).astype(np.float32)

    # [8-11] Volatility regime
    atr14 = _atr(high, low, close, 14)
    atr50 = _atr(high, low, close, 50)
    out[:, 8] = np.clip(atr14 / (atr50 + EPS), 0, 2).astype(np.float32) / 2  # vol expansion

    # EWMA vol (λ=0.94 RiskMetrics)
    ewma_var = np.zeros(N)
    ewma_var[0] = lr1[0] ** 2
    for i in range(1, N):
        ewma_var[i] = 0.94 * ewma_var[i-1] + 0.06 * lr1[i] ** 2
    out[:, 9] = np.clip(np.sqrt(ewma_var) / (atr14 + EPS), 0, 2).astype(np.float32) / 2

    # Vol-of-vol
    ewma_std = np.sqrt(ewma_var)
    out[:, 10] = np.clip(_rolling_std(ewma_std, 20) / (np.mean(ewma_std) + EPS),
                         0, 2).astype(np.float32) / 2

    # Vol percentile (20-bar)
    out[:, 11] = _pct_rank(atr14, 20).astype(np.float32)

    # [12-15] Trend/regime features
    adx14 = _adx_proxy(high, low, close, 14)
    out[:, 12] = adx14.astype(np.float32)

    dm_plus  = np.maximum(np.diff(high, prepend=high[0]), 0.0)
    dm_minus = np.maximum(np.diff(-low, prepend=-low[0]), 0.0)
    di_plus  = _ema(dm_plus, 14) / (atr14 + EPS)
    di_minus = _ema(dm_minus, 14) / (atr14 + EPS)
    out[:, 13] = np.clip((di_plus - di_minus) / (di_plus + di_minus + EPS),
                         -1, 1).astype(np.float32)

    # Efficiency ratio (Kaufman) — vectorised
    direction  = np.abs(close - pd.Series(close).shift(10).fillna(close[0]).to_numpy())
    path_sum   = (pd.Series(np.abs(np.diff(close, prepend=close[0])))
                    .rolling(10, min_periods=1).sum().to_numpy())
    out[:, 14] = np.clip(direction / (path_sum + EPS), 0, 1).astype(np.float32)

    # Hurst regime (rolling 64-bar) — keep loop; _hurst_rs is cheap per call
    for i in range(64, N):
        out[i, 15] = _hurst_rs(lr1[i - 64:i])

    # [16-19] Bollinger z-scores
    mu20 = _rolling_mean(close, 20)
    sd20 = _rolling_std(close, 20)
    mu100 = _rolling_mean(close, 100)
    sd100 = _rolling_std(close, 100)
    out[:, 16] = np.clip((close - mu20)  / np.where(sd20  > EPS, sd20,  1.0),
                         -3, 3).astype(np.float32) / 3
    out[:, 17] = np.clip((close - mu100) / np.where(sd100 > EPS, sd100, 1.0),
                         -3, 3).astype(np.float32) / 3
    out[:, 18] = np.clip(2 * sd20 / (mu20 + EPS), 0, 0.1).astype(np.float32) / 0.1  # BB width
    out[:, 19] = np.clip(_realized_vol(close, 1) / (_realized_vol(close, 20) + EPS),
                         0, 3).astype(np.float32) / 3

    # [20-22] Range/cycle position
    win60 = 60
    _high_sr = pd.Series(high)
    _low_sr  = pd.Series(low)
    h60 = _high_sr.rolling(win60, min_periods=1).max().to_numpy()
    l60 = _low_sr.rolling(win60,  min_periods=1).min().to_numpy()
    r60 = h60 - l60 + EPS
    out[:, 20] = np.clip((close - l60) / r60, 0, 1).astype(np.float32)  # range position

    # Days since 60-bar high/low (normalised) — keep loop (argmax of reversed slice)
    for i in range(win60, N):
        seg_h = high[i - win60:i + 1]
        seg_l = low[i  - win60:i + 1]
        out[i, 21] = float((win60 - np.argmax(seg_h[::-1])) / win60)
        out[i, 22] = float((win60 - np.argmin(seg_l[::-1])) / win60)

    # [23] Cycle phase via zero-crossing of detrended close
    mu = _rolling_mean(close, 20)
    det = close - mu
    phase = np.zeros(N)
    for i in range(1, N):
        if det[i-1] < 0 and det[i] >= 0:
            phase[i] = 0.0
        elif det[i-1] > 0 and det[i] <= 0:
            phase[i] = 0.5
        else:
            phase[i] = phase[i-1]
    out[:, 23] = phase.astype(np.float32)

    # [24] Range quartile (ATR percentile)
    out[:, 24] = _pct_rank(atr14, 100).astype(np.float32)

    # [25] Bar efficiency
    out[:, 25] = np.clip(np.abs(close - open_) / (high - low + EPS), 0, 1).astype(np.float32)

    # [26] Consecutive same-direction bars
    consec = np.zeros(N)
    for i in range(1, N):
        d = np.sign(close[i] - close[i-1])
        consec[i] = consec[i-1] + d if np.sign(consec[i-1]) == d else d
    out[:, 26] = np.clip(consec / 10.0, -1, 1).astype(np.float32)

    # [27-31] Wick stats
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    uw_ma = _rolling_mean(upper_wick, 10)
    lw_ma = _rolling_mean(lower_wick, 10)
    out[:, 27] = np.clip(uw_ma / (atr14 + EPS), 0, 1).astype(np.float32)
    out[:, 28] = np.clip(lw_ma / (atr14 + EPS), 0, 1).astype(np.float32)
    out[:, 29] = np.clip((uw_ma - lw_ma) / (atr14 + EPS), -1, 1).astype(np.float32)
    doji_flag = (np.abs(close - open_) / (high - low + EPS) < 0.1).astype(float)
    out[:, 30] = _rolling_mean(doji_flag, 10).astype(np.float32)

    # [31-34] Volume features
    vol_float = vol.astype(float)
    vol_ma20 = _rolling_mean(vol_float, 20)
    vol_std20 = _rolling_std(vol_float, 20)
    lr_for_corr = _log_return(close, 1)
    out[:, 31] = np.clip(
        _linreg(vol_float, 20)[0], -1, 1).astype(np.float32)
    out[:, 32] = np.clip(
        vol_float / (np.roll(vol_float, 5) + EPS) - 1, -1, 1).astype(np.float32)
    # vol×return correlation — vectorised pandas rolling corr
    _vol_s  = pd.Series(vol_float)
    _pabs_s = pd.Series(np.abs(lr_for_corr))
    out[:, 33] = (_vol_s.rolling(20, min_periods=20).corr(_pabs_s)
                        .fillna(0.0).clip(-1.0, 1.0).to_numpy().astype(np.float32))
    out[:, 34] = np.clip(
        (high - low) / (_rolling_mean(high - low, 20) + EPS),
        0, 3).astype(np.float32) / 3

    # [35-40] Session and calendar features — vectorised
    if "time" in bars.columns:
        import calendar as _cal
        times    = pd.to_datetime(bars["time"])
        hrs      = times.dt.hour.to_numpy(dtype=int)
        out[:, 35] = np.where((hrs >= 8)  & (hrs < 12), 1.0, 0.0).astype(np.float32)
        out[:, 36] = np.where((hrs >= 13) & (hrs < 17), 1.0, 0.0).astype(np.float32)
        out[:, 37] = np.where(hrs < 7, 1.0, 0.0).astype(np.float32)
        try:
            days_in_month = np.array(
                [_cal.monthrange(t.year, t.month)[1] for t in times], dtype=float)
            days_to_end = days_in_month - times.dt.day.to_numpy(dtype=float)
            out[:, 38] = np.clip(1.0 - days_to_end / 5.0, 0.0, 1.0).astype(np.float32)
            month_in_q  = ((times.dt.month.to_numpy(dtype=int) - 1) % 3) + 1
            out[:, 39]  = ((month_in_q == 3) & (days_to_end < 7)).astype(np.float32)
        except Exception:
            pass

    # [40] Volatility regime flag (ATR > 2× baseline)
    atr_baseline = _rolling_mean(atr14, 50)
    out[:, 40] = (atr14 > 2 * atr_baseline).astype(np.float32)

    # --- Regime-aware features (slots 41-59) ---
    # Bull/Bear regime based on moving average alignment
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    out[:, 41] = np.where(ema20 > ema50, 1.0, 0.0).astype(np.float32)  # Short-term bullish
    out[:, 42] = np.where(ema50 > ema200, 1.0, 0.0).astype(np.float32)  # Long-term bullish
    out[:, 43] = np.where(close > ema20, 1.0, 0.0).astype(np.float32)  # Price above short MA

    # Trend strength: how aligned are the MAs
    ma_alignment = 0.0
    for i in range(200, N):
        if ema20[i] > ema50[i] > ema200[i]:
            ma_alignment = 1.0  # Strong uptrend
        elif ema20[i] < ema50[i] < ema200[i]:
            ma_alignment = -1.0  # Strong downtrend
        else:
            ma_alignment = 0.0  # Sideways/transition
        out[i, 44] = ma_alignment

    # Regime transition detection: when MAs cross
    for i in range(50, N):
        prev_cross = (ema20[i-1] - ema50[i-1]) * (ema20[i] - ema50[i]) < 0
        out[i, 45] = 1.0 if prev_cross else 0.0

    # Market phase: expansion vs contraction
    for i in range(20, N):
        price_range = (high[i-20:i].max() - low[i-20:i].min()) / (close[i] + EPS)
        out[i, 46] = np.clip(price_range * 100, 0, 1).astype(np.float32)

    # Momentum regime: accelerating vs decelerating
    mom5 = np.diff(close, prepend=close[0])
    mom20 = np.zeros(N)
    for i in range(20, N):
        mom20[i] = close[i] - close[i-20]
    for i in range(25, N):
        mom_accel = (mom5[i] - mom5[i-5]) * (mom20[i] - mom20[i-5])
        out[i, 47] = 1.0 if mom_accel > 0 else 0.0

    # Volatility regime: low vs high vol periods
    vol_percentile = _pct_rank(atr14, 100)
    out[:, 48] = (vol_percentile > 0.7).astype(np.float32)  # High vol flag
    out[:, 49] = (vol_percentile < 0.3).astype(np.float32)  # Low vol flag

    # Trend persistence: how long has current trend lasted
    trend_duration = np.zeros(N)
    current_trend = 0
    for i in range(1, N):
        if close[i] > close[i-1]:
            if current_trend > 0:
                trend_duration[i] = trend_duration[i-1] + 1
            else:
                trend_duration[i] = 1
                current_trend = 1
        elif close[i] < close[i-1]:
            if current_trend < 0:
                trend_duration[i] = trend_duration[i-1] + 1
            else:
                trend_duration[i] = 1
                current_trend = -1
        else:
            trend_duration[i] = trend_duration[i-1]
    out[:, 50] = np.clip(trend_duration / 20.0, 0, 1).astype(np.float32)

    # Support/Resilience: how well price holds above support
    for i in range(20, N):
        support_level = np.min(low[i-20:i])
        resilience = (close[i] - support_level) / (atr14[i] + EPS)
        out[i, 51] = np.clip(resilience, 0, 3).astype(np.float32) / 3.0

    # --- MTF Alignment Features (slots 52-59) ---
    # These features will be populated by the main build_feature_dataframe function
    # which has access to all MTF data. For now, we initialize with zeros.
    # Slots 52-59: MTF alignment scores (will be filled in build_feature_dataframe)

    return out


# ---------------------------------------------------------------------------
# MTF Alignment Features
# ---------------------------------------------------------------------------

def _compute_mtf_trend(close: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Compute trend direction for a given timeframe.
    Returns: +1 for uptrend, -1 for downtrend, 0 for sideways
    """
    N = len(close)
    trend = np.zeros(N, dtype=np.float32)

    for i in range(window, N):
        recent = close[i-window:i]
        slope, _ = _linreg(recent, min(window, len(recent)))
        # Normalize slope by price level
        norm_slope = slope / (close[i] + EPS)
        if norm_slope > 0.0005:  # Uptrend threshold
            trend[i] = 1.0
        elif norm_slope < -0.0005:  # Downtrend threshold
            trend[i] = -1.0
        else:
            trend[i] = 0.0  # Sideways

    return trend


def _compute_mtf_alignment(m5_trend: np.ndarray, h1_trend: np.ndarray,
                          h4_trend: np.ndarray, d1_trend: np.ndarray) -> np.ndarray:
    """
    Compute MTF alignment score across timeframes.
    Returns: alignment score [0-1] where 1 = perfect alignment, 0 = complete conflict
    """
    N = len(m5_trend)
    alignment = np.zeros(N, dtype=np.float32)

    for i in range(N):
        trends = [m5_trend[i], h1_trend[i], h4_trend[i], d1_trend[i]]

        # Count agreements between all pairs
        agreements = 0
        total_pairs = 0

        for j in range(len(trends)):
            for k in range(j+1, len(trends)):
                total_pairs += 1
                if trends[j] == trends[k] and trends[j] != 0:
                    agreements += 1
                elif trends[j] == 0 or trends[k] == 0:
                    # Sideways doesn't count as agreement or disagreement
                    total_pairs -= 1

        if total_pairs > 0:
            alignment[i] = agreements / total_pairs
        else:
            alignment[i] = 0.5  # Neutral when all are sideways

    return alignment


def _compute_mtf_strength(m5_trend: np.ndarray, h1_trend: np.ndarray,
                         h4_trend: np.ndarray, d1_trend: np.ndarray) -> np.ndarray:
    """
    Compute MTF trend strength (how strong is the consensus).
    Returns: strength score [0-1] where 1 = strong consensus, 0 = weak/no consensus
    """
    N = len(m5_trend)
    strength = np.zeros(N, dtype=np.float32)

    for i in range(N):
        trends = [m5_trend[i], h1_trend[i], h4_trend[i], d1_trend[i]]

        # Count bullish and bearish votes
        bullish = sum(1 for t in trends if t > 0)
        bearish = sum(1 for t in trends if t < 0)
        total = bullish + bearish

        if total > 0:
            # Strength is the proportion of agreeing timeframes
            strength[i] = max(bullish, bearish) / 4.0
        else:
            strength[i] = 0.0

    return strength


def _compute_mtf_divergence(m5_trend: np.ndarray, h1_trend: np.ndarray,
                           h4_trend: np.ndarray, d1_trend: np.ndarray) -> np.ndarray:
    """
    Detect MTF divergence (lower timeframe opposite to higher timeframes).
    Returns: divergence flag [0-1] where 1 = strong divergence, 0 = no divergence
    """
    N = len(m5_trend)
    divergence = np.zeros(N, dtype=np.float32)

    for i in range(N):
        # Check if M5 is opposite to H4 and D1
        htf_bullish = (h4_trend[i] > 0 and d1_trend[i] > 0)
        htf_bearish = (h4_trend[i] < 0 and d1_trend[i] < 0)

        if htf_bullish and m5_trend[i] < 0:
            divergence[i] = 1.0  # Bearish divergence
        elif htf_bearish and m5_trend[i] > 0:
            divergence[i] = 1.0  # Bullish divergence
        else:
            divergence[i] = 0.0

    return divergence


def _build_mtf_alignment_block(m5_bars: pd.DataFrame,
                               h1_df: Optional[pd.DataFrame],
                               h4_df: Optional[pd.DataFrame],
                               d1_df: Optional[pd.DataFrame],
                               N: int) -> np.ndarray:
    """
    Build 8-dim MTF alignment block.
    Returns (N, 8) with:
      [0] MTF alignment score
      [1] MTF trend strength
      [2] MTF divergence flag
      [3] M5-H1 agreement
      [4] M5-H4 agreement
      [5] M5-D1 agreement
      [6] H1-H4 agreement
      [7] H4-D1 agreement
    """
    out = np.zeros((N, 8), dtype=np.float32)

    if h1_df is None or h4_df is None or d1_df is None:
        return out

    # Compute trends for each timeframe
    m5_close = m5_bars["close"].to_numpy(dtype=float)
    h1_close = h1_df["close"].to_numpy(dtype=float)
    h4_close = h4_df["close"].to_numpy(dtype=float)
    d1_close = d1_df["close"].to_numpy(dtype=float)

    # Align HTF data to M5 timestamps
    m5_times = pd.to_datetime(m5_bars["time"], utc=True)
    h1_times = pd.to_datetime(h1_df["time"], utc=True)
    h4_times = pd.to_datetime(h4_df["time"], utc=True)
    d1_times = pd.to_datetime(d1_df["time"], utc=True)

    m5_arr = m5_times.to_numpy(dtype="datetime64[ns]")
    h1_arr = h1_times.to_numpy(dtype="datetime64[ns]")
    h4_arr = h4_times.to_numpy(dtype="datetime64[ns]")
    d1_arr = d1_times.to_numpy(dtype="datetime64[ns]")

    # Get aligned indices
    h1_idx = np.clip(np.searchsorted(h1_arr, m5_arr, side="right") - 1, 0, len(h1_close) - 1)
    h4_idx = np.clip(np.searchsorted(h4_arr, m5_arr, side="right") - 1, 0, len(h4_close) - 1)
    d1_idx = np.clip(np.searchsorted(d1_arr, m5_arr, side="right") - 1, 0, len(d1_close) - 1)

    # Compute trends
    m5_trend = _compute_mtf_trend(m5_close, window=20)
    h1_trend_aligned = _compute_mtf_trend(h1_close[h1_idx], window=20)
    h4_trend_aligned = _compute_mtf_trend(h4_close[h4_idx], window=10)
    d1_trend_aligned = _compute_mtf_trend(d1_close[d1_idx], window=5)

    # Compute MTF alignment features
    out[:, 0] = _compute_mtf_alignment(m5_trend, h1_trend_aligned, h4_trend_aligned, d1_trend_aligned)
    out[:, 1] = _compute_mtf_strength(m5_trend, h1_trend_aligned, h4_trend_aligned, d1_trend_aligned)
    out[:, 2] = _compute_mtf_divergence(m5_trend, h1_trend_aligned, h4_trend_aligned, d1_trend_aligned)

    # Pairwise agreement scores
    out[:, 3] = (m5_trend == h1_trend_aligned).astype(np.float32)
    out[:, 4] = (m5_trend == h4_trend_aligned).astype(np.float32)
    out[:, 5] = (m5_trend == d1_trend_aligned).astype(np.float32)
    out[:, 6] = (h1_trend_aligned == h4_trend_aligned).astype(np.float32)
    out[:, 7] = (h4_trend_aligned == d1_trend_aligned).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Block 8: Cross-Asset (20-dim)
# ---------------------------------------------------------------------------

def _build_xasset_block(bars: pd.DataFrame, xasset_data: Optional[Dict],
                         symbol: str, N: int) -> np.ndarray:
    """
    Returns (N, 20). All zeros if cross-asset data not available.
    xasset_data: dict of canonical → DataFrame (with 'time', 'close' columns)
    """
    out = np.zeros((N, XASSET_DIM), dtype=np.float32)
    if xasset_data is None:
        return out

    close = bars["close"].to_numpy(dtype=float)
    if "time" not in bars.columns:
        return out
    m5_times = pd.to_datetime(bars["time"], utc=True)

    cross_syms = ["GOLD", "BTCUSD", "US_500"]
    windows    = [5, 10, 20, 50, 100]
    feat_idx   = 0

    lr1 = np.diff(np.log(close + EPS), prepend=0.0)

    for cross_sym in cross_syms:
        if cross_sym == symbol:
            feat_idx += len(windows)
            continue
        if cross_sym not in xasset_data:
            feat_idx += len(windows)
            continue

        df_x = xasset_data[cross_sym]
        if "close" not in df_x.columns or "time" not in df_x.columns:
            feat_idx += len(windows)
            continue

        x_times = pd.to_datetime(df_x["time"], utc=True)
        x_close = df_x["close"].to_numpy(dtype=float)
        x_lr1   = np.diff(np.log(x_close + EPS), prepend=0.0)

        # Align cross-asset to M5 — searchsorted
        m5_arr = m5_times.to_numpy(dtype="datetime64[ns]")
        xt_arr = x_times.to_numpy(dtype="datetime64[ns]")
        xi_idx = np.clip(
            np.searchsorted(xt_arr, m5_arr, side="right") - 1,
            0, len(x_lr1) - 1)
        x_aligned = x_lr1[xi_idx]

        # Rolling correlation — pandas vectorised
        _lr1_s = pd.Series(lr1)
        _xa_s  = pd.Series(x_aligned)
        for w in windows:
            corr_arr = (_lr1_s.rolling(w, min_periods=w).corr(_xa_s)
                               .fillna(0.0).clip(-1.0, 1.0).to_numpy())
            out[:, feat_idx] = corr_arr.astype(np.float32)
            feat_idx += 1

    # Remaining slots: relative performance (3) + regime flag (1)
    # (feat_idx should be at 15 after 3 symbols × 5 windows)
    rel_idx = 15
    for cross_sym in cross_syms[:3]:
        if rel_idx >= XASSET_DIM:
            break
        if cross_sym in xasset_data and cross_sym != symbol:
            df_x = xasset_data[cross_sym]
            if "close" in df_x.columns and "time" in df_x.columns:
                x_times = pd.to_datetime(df_x["time"], utc=True)
                x_close = df_x["close"].to_numpy(dtype=float)
                xt_arr2 = pd.to_datetime(df_x["time"], utc=True).to_numpy(
                    dtype="datetime64[ns]")
                xi2_idx = np.clip(
                    np.searchsorted(xt_arr2,
                                    m5_times.to_numpy(dtype="datetime64[ns]"),
                                    side="right") - 1,
                    0, len(x_close) - 1)
                x_aligned_c = x_close[xi2_idx]
                rel = np.where(x_aligned_c > EPS,
                               (close - x_aligned_c) / x_aligned_c, 0.0)
                out[:, rel_idx] = np.clip(rel * 5.0, -1, 1).astype(np.float32)
        rel_idx += 1

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Execution context block (40-dim, appended to direction features)
# ═══════════════════════════════════════════════════════════════════════════

def compute_exec_context(bars: pd.DataFrame, symbol: str,
                          pip_size: float) -> np.ndarray:
    """
    Compute 40-dim execution context block.
    Slots   0-3 : microstructure (atr ratios, spread/atr, volume spike)
    Slots   4-39: reserved for EA-injected position context (zeros at training)
    Returns (N, 40).
    """
    N = len(bars)
    out = np.zeros((N, EXEC_CTX_DIM), dtype=np.float32)

    close  = bars["close"].to_numpy(dtype=float)
    high   = bars["high"].to_numpy(dtype=float)
    low    = bars["low"].to_numpy(dtype=float)
    vol    = bars.get("tick_volume", bars.get("volume",
                pd.Series(np.ones(N)))).to_numpy(dtype=float)
    atr14  = _atr(high, low, close, 14)
    atr5   = _atr(high, low, close, 5)
    atr50  = _atr(high, low, close, 50)
    spread = bars["spread"].to_numpy(dtype=float) * pip_size \
             if "spread" in bars.columns else np.zeros(N)

    out[:, 0] = np.clip(atr5 / (atr14 + EPS), 0, 3).astype(np.float32) / 3
    out[:, 1] = np.clip(atr14 / (atr50 + EPS), 0, 3).astype(np.float32) / 3
    out[:, 2] = np.clip(spread / (atr14 + EPS), 0, 1).astype(np.float32)
    vol_ma = _rolling_mean(vol.astype(float), 20)
    out[:, 3] = np.clip(vol / (vol_ma + EPS), 0, 4).astype(np.float32) / 4
    # Slots 4-39: zeros at training; EA fills with live position context.

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main public API
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_dataframe(
        bars: pd.DataFrame,
        symbol: str,
        pip_size: float = 0.0001,
        h1_df: Optional[pd.DataFrame] = None,    # accepted for API compat; ignored
        h4_df: Optional[pd.DataFrame] = None,    # accepted for API compat; ignored
        h8_df: Optional[pd.DataFrame] = None,    # accepted for API compat; ignored
        d1_df: Optional[pd.DataFrame] = None,    # accepted for API compat; ignored
        xasset_data: Optional[Dict] = None,      # accepted for API compat; ignored
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 200-dim direction features and 240-dim execution features.

    mk4.3: parity-floored to what the EA-side FeatureEncoder.mqh actually
    populates (M5 block only — 50 raw features × 4 transforms). Higher-
    timeframe / spectral / pattern / statreg / xasset blocks are dropped
    because the EA never wrote them — training on them was producing a
    1160-dim model that received 83% zeros at inference.

    Returns:
        dir_feat  : (N, 200)  float32
        exec_feat : (N, 240)  float32  — dir_feat + 40-dim exec context
    """
    N = len(bars)
    log.info("[%s] Building %d-dim features for %d bars …", symbol, FEATURE_DIM_DIR, N)

    dir_feat = _build_m5_block(bars, pip_size).astype(np.float32)         # (N,200)

    assert dir_feat.shape == (N, FEATURE_DIM_DIR), \
        f"Direction feature shape mismatch: {dir_feat.shape} != ({N}, {FEATURE_DIM_DIR})"

    log.info("[%s] Direction features: %s", symbol, dir_feat.shape)

    # Execution context block — microstructure + EA-injected position slots
    exec_ctx  = compute_exec_context(bars, symbol, pip_size)               # (N, EXEC_CTX_DIM)
    exec_feat = np.concatenate([dir_feat, exec_ctx], axis=1).astype(np.float32)

    assert exec_feat.shape == (N, FEATURE_DIM_EXEC), \
        f"Exec feature shape mismatch: {exec_feat.shape} != ({N}, {FEATURE_DIM_EXEC})"

    log.info("[%s] Exec features:      %s", symbol, exec_feat.shape)

    return dir_feat, exec_feat
