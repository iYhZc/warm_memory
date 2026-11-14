"""Decide how long a stored memory stays usable.

The 0.3.x extractor happily stored one-day states ("明天有早八") as permanent
facts, and nothing ever expired them. The fix is structural rather than a
prompt tweak: every memory may carry ``valid_until``, and recall ignores rows
whose window has closed. This module resolves that timestamp from the text.

Rules, in order:

* an explicit absolute date from the model wins (``2026-11-02``);
* a periodic statement ("每周三有早八") is permanent;
* a relative day ("今天/明天/后天/今晚") expires at the end of that day;
* "下周" expires a week out, "这周/本周" at the end of the current week;
* anything else stays permanent.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time, timedelta
from datetime import timezone as _timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_RELATIVE_DAYS: tuple[tuple[str, int], ...] = (
    ("大后天", 3),
    ("后天", 2),
    ("明早", 1),
    ("明晚", 1),
    ("明天", 1),
    ("明日", 1),
    ("今晚", 0),
    ("今早", 0),
    ("今天", 0),
    ("今日", 0),
)

_PERIODIC = re.compile(r"每\s*(?:周|星期|礼拜|天|日|月|年)")
_THIS_WEEK = re.compile(r"(?:这|本)\s*(?:周|星期|礼拜)")
_NEXT_WEEK = re.compile(r"(?:下)\s*(?:周|星期|礼拜)")
_DATE = re.compile(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?")
_END_OF_DAY = time(23, 59, 59)


def resolve_source_time(value: object, fallback: float = 0.0) -> float:
    """Accept only usable capture timestamps, never time claimed in message text."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        return fallback
    try:
        datetime.fromtimestamp(value, _timezone.utc)
    except (OverflowError, OSError, ValueError):
        return fallback
    return float(value)


def resolve_timezone(name: str) -> ZoneInfo | _timezone:
    """Fall back to a fixed +08:00 when the host has no tz database."""
    if name in {"UTC", "Etc/UTC"}:
        return _timezone.utc
    try:
        return ZoneInfo(name or "Asia/Shanghai")
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return _timezone(timedelta(hours=8))


def parse_explicit_date(text: str) -> date | None:
    """Read ``2026-11-02`` / ``2026/11/2`` / ``2026年11月2日``."""
    match = _DATE.search(text or "")
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def has_periodic_marker(text: str) -> bool:
    return bool(_PERIODIC.search(text or ""))


def relative_day_offset(text: str) -> int | None:
    """Earliest relative-day word in the text, in days from today."""
    body = text or ""
    best: int | None = None
    best_index = len(body) + 1
    for word, offset in _RELATIVE_DAYS:
        index = body.find(word)
        if 0 <= index < best_index:
            best, best_index = offset, index
    return best


def end_of_day(day: date, tz: ZoneInfo | _timezone) -> float:
    return datetime.combine(day, _END_OF_DAY, tzinfo=tz).timestamp()


def local_today(now: float, tz: ZoneInfo | _timezone) -> date:
    return datetime.fromtimestamp(now, tz).date()


def compute_valid_until(
    content: str,
    explicit: str | None = None,
    *,
    now: float,
    timezone: str = "Asia/Shanghai",
) -> float | None:
    """Return the epoch at which this memory stops being injected.

    ``None`` means "permanent". The returned value may already be in the past;
    callers treat that as expired.
    """
    tz = resolve_timezone(timezone)
    today = local_today(now, tz)

    if explicit:
        day = parse_explicit_date(str(explicit))
        if day is not None:
            return end_of_day(day, tz)

    body = str(content or "")
    if has_periodic_marker(body):
        return None

    offset = relative_day_offset(body)
    if offset is not None:
        return end_of_day(today + timedelta(days=offset), tz)

    if _NEXT_WEEK.search(body):
        return end_of_day(today + timedelta(days=7), tz)
    if _THIS_WEEK.search(body):
        days_until_sunday = 6 - today.weekday()
        return end_of_day(today + timedelta(days=days_until_sunday), tz)
    return None


def is_expired(valid_until: float | None, now: float) -> bool:
    return valid_until is not None and valid_until <= now


def format_expiry(valid_until: float | None, *, timezone: str = "Asia/Shanghai") -> str:
    """``（有效期至 2026-11-03）`` for listings; empty when permanent."""
    if valid_until is None:
        return ""
    tz = resolve_timezone(timezone)
    day = datetime.fromtimestamp(valid_until, tz).date()
    return f"（有效期至 {day.isoformat()}）"
