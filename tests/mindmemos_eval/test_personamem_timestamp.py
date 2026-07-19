"""Tests for PersonaMem session-aware timestamp mapping (PR #20 tier-2)."""

from __future__ import annotations

from datetime import datetime, timezone

from mindmemos_eval.memory.envs.personamem.env import (
    _PERSONAMEM_EPOCH_MS,
    _build_session_timestamp_map_ms,
)


def _ms(y: int, mo: int, d: int) -> int:
    return int(datetime(y, mo, d, tzinfo=timezone.utc).timestamp() * 1000)


def test_epoch_constant_is_2026_01_01():
    assert _PERSONAMEM_EPOCH_MS == _ms(2026, 1, 1)


def test_single_session_turn_advances_one_day():
    # system, then two user+assistant turns.
    context = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    ts = _build_session_timestamp_map_ms(context)
    # system + first turn share day 0 (2026-01-01); same-turn messages share timestamp.
    assert ts[0] == _ms(2026, 1, 1)
    assert ts[1] == _ms(2026, 1, 1)
    assert ts[2] == _ms(2026, 1, 1)
    # second user turn advances one day.
    assert ts[3] == _ms(2026, 1, 2)
    assert ts[4] == _ms(2026, 1, 2)


def test_new_session_starts_next_month():
    # Session 0: system + 1 turn (ends 2026-01-01). Session 1 must start 2026-02-01.
    context = [
        {"role": "system", "content": "s0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "s1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    ts = _build_session_timestamp_map_ms(context)
    assert ts[0] == _ms(2026, 1, 1)  # session 0 start
    assert ts[3] == _ms(2026, 2, 1)  # session 1 start (next month's 1st)
    assert ts[4] == _ms(2026, 2, 1)


def test_missing_leading_system_inserts_session_zero():
    # Context without a leading system message still maps from index 0.
    context = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    ts = _build_session_timestamp_map_ms(context)
    assert ts[0] == _ms(2026, 1, 1)
    assert ts[1] == _ms(2026, 1, 1)


def test_year_rollover_from_december():
    # Session 0 spanning into a December-start would roll over to next January.
    # Build 12 sessions so the 12th starts in December, 13th rolls to next year.
    context: list[dict[str, str]] = []
    for _ in range(13):
        context.append({"role": "system", "content": "s"})
        context.append({"role": "user", "content": "u"})
        context.append({"role": "assistant", "content": "a"})
    ts = _build_session_timestamp_map_ms(context)
    # session index 11 (0-based) -> December 2026; session 12 -> January 2027.
    assert ts[11 * 3] == _ms(2026, 12, 1)
    assert ts[12 * 3] == _ms(2027, 1, 1)
