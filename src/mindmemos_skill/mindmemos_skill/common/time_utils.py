"""Shared UTC defaults and millisecond Unix timestamp conversions."""

from datetime import UTC, datetime, timedelta, tzinfo
from time import time_ns

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
"""Default display format, with second precision and no timezone suffix."""
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
"""Timezone-aware Unix epoch used for exact integer conversions."""


def utcnow() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)


def current_timestamp_ms() -> int:
    """Return the current Unix timestamp in milliseconds (currently 13 digits)."""
    return time_ns() // 1_000_000


def datetime_to_timestamp_ms(value: datetime) -> int:
    """Convert an aware datetime to Unix milliseconds, flooring submilliseconds.

    Args:
        value: Datetime with an explicit timezone.

    Raises:
        ValueError: The datetime has no timezone.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Datetime must include a timezone")
    delta = value.astimezone(UTC) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def timestamp_ms_to_datetime(timestamp_ms: int, *, tz: tzinfo = UTC) -> datetime:
    """Convert Unix milliseconds to an aware datetime in the selected timezone.

    Args:
        timestamp_ms: Integer milliseconds, not seconds; historical values need not have 13 digits.
        tz: Output timezone; defaults to UTC, never the machine's local timezone.

    Raises:
        TypeError: The timestamp is not an integer or the timezone is invalid.
        OverflowError: The timestamp is outside datetime's supported range.
    """
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int):
        raise TypeError("Timestamp must be an integer number of milliseconds")
    if not isinstance(tz, tzinfo):
        raise TypeError("Timezone must be a tzinfo instance")
    return (_EPOCH + timedelta(milliseconds=timestamp_ms)).astimezone(tz)


def timestamp_ms_to_string(timestamp_ms: int, *, fmt: str = DEFAULT_TIME_FORMAT, tz: tzinfo = UTC) -> str:
    """Format Unix milliseconds as a string in the selected timezone.

    Args:
        timestamp_ms: Integer Unix timestamp in milliseconds.
        fmt: strftime format; include %f to retain subsecond precision and %z for an offset.
        tz: Output timezone; defaults to UTC.
    """
    return timestamp_ms_to_datetime(timestamp_ms, tz=tz).strftime(fmt)


def string_to_timestamp_ms(value: str, *, fmt: str = DEFAULT_TIME_FORMAT, tz: tzinfo = UTC) -> int:
    """Parse a formatted time string into Unix milliseconds.

    Args:
        value: Time string matching fmt.
        fmt: strptime format; include %f for fractional seconds or %z for an explicit offset.
        tz: Timezone for strings without an offset; an explicitly parsed offset takes precedence.

    Raises:
        ValueError: The string does not match the format.
        TypeError: The fallback timezone is not a tzinfo instance.
    """
    if not isinstance(tz, tzinfo):
        raise TypeError("Timezone must be a tzinfo instance")
    parsed = datetime.strptime(value, fmt)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return datetime_to_timestamp_ms(parsed)
