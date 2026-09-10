"""Timezone and 12-Hour Display Utilities for Dashboard.

Handles timezone conversions between UTC and Asia/Kolkata (IST / Chennai time).
Uses standard library zoneinfo for strict timezone-aware conversions.

CRITICAL INVARIANTS:
1. Internal timestamps remain UTC.
2. Conversion to Asia/Kolkata is done strictly via zoneinfo.ZoneInfo("Asia/Kolkata").
3. 12-hour formatting with AM/PM (e.g., '05:00 PM IST').
4. Calendar date mapping uses local Chennai date after timezone conversion.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional, Union
from zoneinfo import ZoneInfo

IST_ZONE = ZoneInfo("Asia/Kolkata")
UTC_ZONE = timezone.utc


def parse_to_utc_datetime(ts: Union[str, datetime, int, float]) -> datetime:
    """Parse various timestamp representations into a timezone-aware UTC datetime."""
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=UTC_ZONE)

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC_ZONE)
        return ts.astimezone(UTC_ZONE)

    # String format parsing
    ts_str = str(ts).strip()
    # Normalize ISO format with trailing 'Z'
    ts_clean = ts_str.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(ts_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC_ZONE)
        return dt.astimezone(UTC_ZONE)
    except ValueError:
        pass

    # Try common football data date/time formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(ts_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC_ZONE)
            return dt.astimezone(UTC_ZONE)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse timestamp into UTC datetime: {ts}")


def to_ist_datetime(ts: Union[str, datetime, int, float]) -> datetime:
    """Convert any UTC timestamp into Asia/Kolkata (IST) timezone-aware datetime."""
    utc_dt = parse_to_utc_datetime(ts)
    return utc_dt.astimezone(IST_ZONE)


def format_kickoff_ist(ts: Union[str, datetime, int, float], include_suffix: bool = True) -> str:
    """Format kickoff time in 12-hour format in IST (e.g. '05:00 PM IST' or '05:00 PM').

    Examples:
        11:30 UTC -> '05:00 PM IST'
        14:00 UTC -> '07:30 PM IST'
        16:30 UTC -> '10:00 PM IST'
        19:30 UTC -> '01:00 AM IST'
    """
    try:
        ist_dt = to_ist_datetime(ts)
        time_str = ist_dt.strftime("%I:%M %p")
        if include_suffix:
            return f"{time_str} IST"
        return time_str
    except Exception:
        return str(ts)


def format_kickoff_datetime_ist(ts: Union[str, datetime, int, float]) -> str:
    """Format full date and 12-hour kickoff in IST (e.g. '22 Aug 2026, 05:00 PM IST')."""
    try:
        ist_dt = to_ist_datetime(ts)
        return ist_dt.strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return str(ts)


def to_chennai_date(ts: Union[str, datetime, int, float]) -> str:
    """Return the local calendar date string (YYYY-MM-DD) in Asia/Kolkata timezone."""
    try:
        ist_dt = to_ist_datetime(ts)
        return ist_dt.strftime("%Y-%m-%d")
    except Exception:
        return str(ts)[:10]


def is_kickoff_on_chennai_date(ts: Union[str, datetime, int, float], target_date_str: str) -> bool:
    """Check if the fixture's kickoff falls on target_date_str in Chennai local time."""
    return to_chennai_date(ts) == target_date_str


def format_kickoff_utc(ts: Union[str, datetime, int, float], include_suffix: bool = True) -> str:
    """Format kickoff time in 24-hour format in UTC (e.g. '18:30 UTC' or '18:30').

    Examples:
        11:30 UTC -> '11:30 UTC' or '11:30'
        18:30 UTC -> '18:30 UTC' or '18:30'
        19:00 UTC -> '19:00 UTC' or '19:00'
    """
    try:
        utc_dt = parse_to_utc_datetime(ts)
        time_str = utc_dt.strftime("%H:%M")
        if include_suffix:
            return f"{time_str} UTC"
        return time_str
    except Exception:
        return str(ts)


def format_kickoff_datetime_utc(ts: Union[str, datetime, int, float]) -> str:
    """Format full date and 24-hour kickoff in UTC (e.g. '27 Aug 2026, 18:30 UTC')."""
    try:
        utc_dt = parse_to_utc_datetime(ts)
        return utc_dt.strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return str(ts)


def to_utc_date(ts: Union[str, datetime, int, float]) -> str:
    """Return the calendar date string (YYYY-MM-DD) in UTC timezone."""
    try:
        utc_dt = parse_to_utc_datetime(ts)
        return utc_dt.strftime("%Y-%m-%d")
    except Exception:
        return str(ts)[:10]


def is_kickoff_on_utc_date(ts: Union[str, datetime, int, float], target_date_str: str) -> bool:
    """Check if the fixture's kickoff falls on target_date_str in UTC time."""
    return to_utc_date(ts) == target_date_str

