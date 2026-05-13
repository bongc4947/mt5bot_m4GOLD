"""
session_features.py — time-of-day, session, and calendar features
derived from a bar's `time` column. All cyclic / one-hot / boolean.
Cheap to compute, no external data needed.

Columns added (all float32):
    tod_sin, tod_cos        — sin/cos of (seconds_into_UTC_day / 86400)
    dow_sin, dow_cos        — sin/cos of day-of-week / 7
    sess_asia, sess_london, sess_ny  — one-hot sessions (UTC)
    is_overlap_lon_ny       — bool 1.0 in 12:00-16:00 UTC overlap
    is_month_end            — bool 1.0 on last 3 days of calendar month
    is_quarter_end          — bool 1.0 on last 3 days of quarter
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SESSION_FEATURE_COLUMNS = [
    "tod_sin", "tod_cos",
    "dow_sin", "dow_cos",
    "sess_asia", "sess_london", "sess_ny",
    "is_overlap_lon_ny",
    "is_month_end", "is_quarter_end",
]
SESSION_FEATURE_DIM = len(SESSION_FEATURE_COLUMNS)


def compute_session_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute session/time/calendar features from a DataFrame with a
    'time' column (UTC). Returns a DataFrame of length len(bars).
    """
    if "time" not in bars.columns:
        raise ValueError("bars must have a 'time' column")
    t = pd.to_datetime(bars["time"], utc=True)
    sec_into_day = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy()
    tod_angle = 2 * np.pi * sec_into_day / 86400.0
    tod_sin = np.sin(tod_angle).astype(np.float32)
    tod_cos = np.cos(tod_angle).astype(np.float32)

    dow = t.dt.dayofweek.to_numpy()  # Mon=0..Sun=6
    dow_angle = 2 * np.pi * dow / 7.0
    dow_sin = np.sin(dow_angle).astype(np.float32)
    dow_cos = np.cos(dow_angle).astype(np.float32)

    hour = t.dt.hour.to_numpy()
    # Approximate UTC-tagged session windows. (Daylight savings drift is
    # small at M5+/tick-bar resolution.)
    sess_asia   = ((hour >= 0)  & (hour < 8)).astype(np.float32)
    sess_london = ((hour >= 7)  & (hour < 16)).astype(np.float32)
    sess_ny     = ((hour >= 12) & (hour < 21)).astype(np.float32)
    overlap     = ((hour >= 12) & (hour < 16)).astype(np.float32)

    day = t.dt.day.to_numpy()
    days_in_month = t.dt.days_in_month.to_numpy()
    is_month_end = (day > days_in_month - 3).astype(np.float32)

    month = t.dt.month.to_numpy()
    is_qend = (((month % 3) == 0) & (day > days_in_month - 3)).astype(np.float32)

    return pd.DataFrame({
        "tod_sin": tod_sin, "tod_cos": tod_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "sess_asia": sess_asia, "sess_london": sess_london, "sess_ny": sess_ny,
        "is_overlap_lon_ny": overlap,
        "is_month_end": is_month_end, "is_quarter_end": is_qend,
    })


def build_session_block(bars: pd.DataFrame) -> np.ndarray:
    """Return the (N, SESSION_FEATURE_DIM) session block."""
    n = len(bars)
    out = np.zeros((n, SESSION_FEATURE_DIM), dtype=np.float32)
    if "time" not in bars.columns:
        return out
    feats = compute_session_features(bars)
    for i, c in enumerate(SESSION_FEATURE_COLUMNS):
        out[:, i] = feats[c].to_numpy(dtype=np.float32)
    return out
