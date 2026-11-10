"""Time conversion units, timezone semantics, and integer precision."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from mindmemos_skill.common import time_utils


def test_current_timestamp_uses_integer_milliseconds(monkeypatch):
    monkeypatch.setattr(time_utils, "time_ns", lambda: 1_700_000_000_123_999_999)
    assert time_utils.current_timestamp_ms() == 1_700_000_000_123
    assert time_utils.utcnow().utcoffset() == timedelta(0)


def test_milliseconds_round_trip_with_explicit_offset():
    tz = timezone(timedelta(hours=8))
    value = datetime(2024, 1, 1, 8, 0, 0, 123000, tzinfo=tz)
    stamp = 1_704_067_200_123
    assert time_utils.datetime_to_timestamp_ms(value) == stamp
    assert time_utils.timestamp_ms_to_datetime(stamp, tz=tz) == value
    assert time_utils.timestamp_ms_to_datetime(stamp).hour == 0
    assert time_utils.timestamp_ms_to_datetime(stamp).tzinfo is UTC


def test_strings_default_to_utc_and_second_precision():
    stamp = 1_704_067_200_123
    assert time_utils.timestamp_ms_to_string(stamp) == "2024-01-01 00:00:00"
    assert time_utils.string_to_timestamp_ms("2024-01-01 00:00:00") == stamp - 123
    tz = timezone(timedelta(hours=8))
    assert time_utils.timestamp_ms_to_string(stamp, tz=tz) == "2024-01-01 08:00:00"
    assert time_utils.string_to_timestamp_ms("2024-01-01 08:00:00", tz=tz) == stamp - 123


def test_fractional_string_round_trip_and_parsed_offset_precedence():
    stamp = 1_704_067_200_123
    fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
    rendered = time_utils.timestamp_ms_to_string(stamp, fmt=fmt, tz=timezone(timedelta(hours=8)))
    assert time_utils.string_to_timestamp_ms(rendered, fmt=fmt, tz=UTC) == stamp


def test_flooring_and_dates_before_epoch():
    value = datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
    assert time_utils.datetime_to_timestamp_ms(value) == -1
    assert time_utils.timestamp_ms_to_datetime(-1).microsecond == 999000
    assert time_utils.timestamp_ms_to_datetime(0) == datetime(1970, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("value", [True, "1704067200123", 1704067200123.0])
def test_reject_non_integer_timestamps(value):
    with pytest.raises(TypeError):
        time_utils.timestamp_ms_to_datetime(value)


def test_reject_implicit_local_timezone_and_malformed_string():
    with pytest.raises(ValueError, match="timezone"):
        time_utils.datetime_to_timestamp_ms(datetime(2024, 1, 1))
    with pytest.raises(TypeError, match="Timezone"):
        time_utils.timestamp_ms_to_datetime(0, tz=None)
    with pytest.raises(TypeError, match="Timezone"):
        time_utils.string_to_timestamp_ms("2024-01-01 00:00:00", tz=None)
    with pytest.raises(ValueError):
        time_utils.string_to_timestamp_ms("not a time")
