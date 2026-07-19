"""Tests for injecting retrieved-memory event dates into the CoT context (A)."""

from __future__ import annotations

from mindmemos_eval.memory.envs.personamem.env import _format_memory_with_date


def test_prefixes_date_from_event_time():
    out = _format_memory_with_date("User liked jazz.", "2026-05-03 00:00:00")
    assert out == "(2026-05-03) User liked jazz."


def test_date_only_event_time_without_time_part():
    out = _format_memory_with_date("User liked jazz.", "2026-05-03")
    assert out == "(2026-05-03) User liked jazz."


def test_none_event_time_returns_memory_unchanged():
    assert _format_memory_with_date("User liked jazz.", None) == "User liked jazz."


def test_empty_event_time_returns_memory_unchanged():
    assert _format_memory_with_date("User liked jazz.", "") == "User liked jazz."


def test_whitespace_only_event_time_returns_memory_unchanged():
    assert _format_memory_with_date("User liked jazz.", "   ") == "User liked jazz."
