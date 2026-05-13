"""
market_hours.py — Encode broker-specific market hours into training features.
Handles DST, holidays, rollovers, and per-symbol session schedules.
"""

import datetime as dt
import math
import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known holiday windows (UTC dates) — market is thin / partially closed
# ---------------------------------------------------------------------------
_HOLIDAY_WINDOWS = [
    # Christmas / New Year thin window
    (12, 24), (12, 25), (12, 26), (12, 27), (12, 28), (12, 29),
    (12, 30), (12, 31), (1, 1), (1, 2),
    # US fixed holidays (approximate)
    (7, 4),   # Independence Day
    (11, 11),  # Veterans Day
]

# Easter offsets from Jan-1 are computed dynamically via algorithm.
# US Thanksgiving: 4th Thursday of November (computed below).

# ---------------------------------------------------------------------------
# DST helpers
# ---------------------------------------------------------------------------
def _us_dst_start(year: int) -> dt.date:
    """Second Sunday of March."""
    d = dt.date(year, 3, 1)
    sundays = 0
    while True:
        if d.weekday() == 6:
            sundays += 1
            if sundays == 2:
                return d
        d += dt.timedelta(days=1)


def _us_dst_end(year: int) -> dt.date:
    """First Sunday of November."""
    d = dt.date(year, 11, 1)
    while d.weekday() != 6:
        d += dt.timedelta(days=1)
    return d


def _eu_dst_start(year: int) -> dt.date:
    """Last Sunday of March."""
    d = dt.date(year, 3, 31)
    while d.weekday() != 6:
        d -= dt.timedelta(days=1)
    return d


def _eu_dst_end(year: int) -> dt.date:
    """Last Sunday of October."""
    d = dt.date(year, 10, 31)
    while d.weekday() != 6:
        d -= dt.timedelta(days=1)
    return d


def _easter(year: int) -> dt.date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def _us_thanksgiving(year: int) -> dt.date:
    """4th Thursday of November."""
    d = dt.date(year, 11, 1)
    thursdays = 0
    while True:
        if d.weekday() == 3:
            thursdays += 1
            if thursdays == 4:
                return d
        d += dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# MarketHoursEncoder
# ---------------------------------------------------------------------------
class MarketHoursEncoder:
    """
    For a given (timestamp_utc, symbol) returns a dict of session features
    that Python uses to build execution model labels and training vectors.
    """

    # Session boundaries in UTC hours (standard time; DST shifts NY by -1h)
    _SESSIONS = {
        "sydney":  (21, 6),    # prev-day 21:00 → 06:00 UTC
        "tokyo":   (0,  9),
        "london":  (8, 17),
        "ny":      (13, 22),
    }

    def __init__(self, median_spreads: Optional[Dict[str, float]] = None):
        """
        median_spreads: dict of {canonical_symbol: median_spread_pips}
        Used to compute spread_tier. Falls back to 1.0 if not provided.
        """
        self._median_spreads = median_spreads or {}
        self._dst_cache: Dict[int, Dict] = {}

    def _dst_info(self, year: int) -> Dict:
        if year not in self._dst_cache:
            self._dst_cache[year] = {
                "us_start": _us_dst_start(year),
                "us_end":   _us_dst_end(year),
                "eu_start": _eu_dst_start(year),
                "eu_end":   _eu_dst_end(year),
                "easter":   _easter(year),
                "thanksgiving": _us_thanksgiving(year),
            }
        return self._dst_cache[year]

    def _us_dst_active(self, d: dt.date) -> bool:
        info = self._dst_info(d.year)
        return info["us_start"] <= d < info["us_end"]

    def _eu_dst_active(self, d: dt.date) -> bool:
        info = self._dst_info(d.year)
        return info["eu_start"] <= d < info["eu_end"]

    def _is_dst_transition_week(self, d: dt.date) -> bool:
        for year in [d.year - 1, d.year, d.year + 1]:
            info = self._dst_info(year)
            for event_date in [info["us_start"], info["us_end"],
                                info["eu_start"], info["eu_end"]]:
                if abs((d - event_date).days) <= 3:
                    return True
        return False

    def _is_holiday_risk(self, d: dt.date) -> bool:
        # Fixed holidays
        for month, day in _HOLIDAY_WINDOWS:
            if d.month == month and d.day == day:
                return True
        # Easter window
        info = self._dst_info(d.year)
        easter = info["easter"]
        good_friday = easter - dt.timedelta(days=2)
        easter_monday = easter + dt.timedelta(days=1)
        if d in (good_friday, easter, easter_monday):
            return True
        # US Thanksgiving + Friday
        thanksgiving = info["thanksgiving"]
        if d in (thanksgiving, thanksgiving + dt.timedelta(days=1)):
            return True
        # Weekend gap risk
        if d.weekday() == 4 and d.hour >= 20:  # Friday after 20:00
            return True
        if d.weekday() == 6 and d.hour < 22:   # Sunday before 22:00
            return True
        return False

    def _session_flags(self, utc: dt.datetime) -> Dict[str, bool]:
        h = utc.hour + utc.minute / 60.0
        d = utc.date()

        # Adjust NY session for US DST
        ny_start = 13 if not self._us_dst_active(d) else 12
        ny_end   = 22 if not self._us_dst_active(d) else 21
        # Adjust London for EU DST
        lon_start = 8 if not self._eu_dst_active(d) else 7
        lon_end   = 17 if not self._eu_dst_active(d) else 16

        sydney = (h >= 21) or (h < 6)
        tokyo  = (0 <= h < 9)
        london = (lon_start <= h < lon_end)
        ny     = (ny_start <= h < ny_end)

        return {
            "london":  london,
            "ny":      ny,
            "tokyo":   tokyo,
            "sydney":  sydney,
            "overlap": london and ny,
        }

    def _minutes_to(self, utc: dt.datetime, target_hour: int, target_min: int = 0) -> float:
        """Minutes until next occurrence of target_hour:target_min UTC."""
        now_min = utc.hour * 60 + utc.minute
        tgt_min = target_hour * 60 + target_min
        diff = tgt_min - now_min
        if diff < 0:
            diff += 1440
        return float(diff)

    def encode(self, timestamp_utc: dt.datetime, symbol: str,
               live_spread_pips: float = 0.0) -> Dict:
        """
        Returns feature dict for one bar's timestamp.
        All time-to-X values in minutes, normalized by 1440.
        """
        d = timestamp_utc.date()
        flags = self._session_flags(timestamp_utc)

        # Time to next session open (London if not in session, else NY)
        if not flags["london"] and not flags["ny"]:
            minutes_to_open = self._minutes_to(timestamp_utc, 8)
        elif flags["london"] and not flags["ny"]:
            minutes_to_open = self._minutes_to(timestamp_utc, 13)
        else:
            minutes_to_open = 0.0

        # Time to session close
        if flags["ny"]:
            ny_end = 22 if not self._us_dst_active(d) else 21
            minutes_to_close = self._minutes_to(timestamp_utc, ny_end)
        elif flags["london"]:
            lon_end = 17 if not self._eu_dst_active(d) else 16
            minutes_to_close = self._minutes_to(timestamp_utc, lon_end)
        else:
            minutes_to_close = 0.0

        minutes_to_rollover     = self._minutes_to(timestamp_utc, 21)
        # Weekly close: Friday 21:00 UTC
        days_to_friday = (4 - timestamp_utc.weekday()) % 7
        minutes_to_weekly_close = days_to_friday * 1440 + self._minutes_to(timestamp_utc, 21)

        # Spread tier
        med = self._median_spreads.get(symbol, 0.0)
        spread_tier = (live_spread_pips / med) if med > 0 else 1.0

        return {
            "session_london":            flags["london"],
            "session_ny":                flags["ny"],
            "session_tokyo":             flags["tokyo"],
            "session_sydney":            flags["sydney"],
            "session_overlap":           flags["overlap"],
            "minutes_to_open":           min(minutes_to_open, 1440.0) / 1440.0,
            "minutes_to_close":          min(minutes_to_close, 1440.0) / 1440.0,
            "minutes_to_rollover":       min(minutes_to_rollover, 1440.0) / 1440.0,
            "minutes_to_weekly_close":   min(minutes_to_weekly_close, 10080.0) / 10080.0,
            "is_holiday_risk":           self._is_holiday_risk(timestamp_utc),
            "is_dst_week":               self._is_dst_transition_week(d),
            "spread_tier":               min(spread_tier, 5.0) / 5.0,
        }


def sin_cos_encode(value: float, period: float):
    """Return (sin, cos) for cyclical encoding."""
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)
