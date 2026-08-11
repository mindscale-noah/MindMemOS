"""Deterministic line-addressed editing for a generated Skill patch."""

from __future__ import annotations

import json
import re
from typing import Any


class Trace2SkillEditError(ValueError):
    """Raised when model-produced line edits cannot be applied safely."""


def format_numbered(skill_md: str) -> str:
    """Render ``skill_md`` with the 1-based gutter referenced by edit operations."""

    lines = skill_md.splitlines()
    if not lines:
        return "(empty document)"
    width = len(str(len(lines)))
    return "\n".join(f"{index:>{width}}| {line}" for index, line in enumerate(lines, start=1))


def apply_patch_ops(skill_md: str, raw: str) -> str:
    """Parse and apply a model-produced edit payload."""

    return apply_edit_ops(skill_md, parse_edit_ops(raw))


def parse_edit_ops(raw: str) -> list[dict[str, Any]]:
    """Accept a JSON object with ``edits`` or a bare JSON edit list."""

    text = _strip_code_fence((raw or "").strip())
    if not text:
        raise Trace2SkillEditError("empty edit payload")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Trace2SkillEditError(f"edit payload is not valid JSON: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("edits", payload.get("ops"))
    if not isinstance(payload, list):
        raise Trace2SkillEditError("edit payload must be a list or an object with an 'edits' list")
    if any(not isinstance(item, dict) for item in payload):
        raise Trace2SkillEditError("every edit must be a JSON object")
    return payload


def apply_edit_ops(skill_md: str, edits: list[dict[str, Any]]) -> str:
    """Validate edits against original line numbers, then apply them bottom-up."""

    lines = skill_md.splitlines(keepends=True)
    line_count = len(lines)
    plans: list[tuple[int, int, list[str], str]] = []
    ranges: list[tuple[int, int, int]] = []

    for index, edit in enumerate(edits):
        operation = edit.get("op")
        if operation in {"replace", "delete"}:
            start = _require_line(edit, "start", index, lower=1, upper=line_count)
            end = _require_line(edit, "end", index, lower=start, upper=line_count)
            _check_line_guard(lines, start, edit, index)
            replacement = "" if operation == "delete" else _require_string(edit.get("new", ""))
            plans.append((start - 1, end, _to_lines(replacement), operation))
            ranges.append((start, end, index))
        elif operation == "insert":
            after = _require_line(edit, "after", index, lower=0, upper=line_count)
            replacement = _require_string(edit.get("new", ""))
            if not replacement:
                raise Trace2SkillEditError(f"edit #{index}: insert 'new' must be non-empty")
            plans.append((after, after, _to_lines(replacement), operation))
        else:
            raise Trace2SkillEditError(f"edit #{index} has unknown op {operation!r}")

    _check_overlap(ranges)
    for declared_index, plan in sorted(
        enumerate(plans),
        key=lambda item: (item[1][0], item[0]),
        reverse=True,
    ):
        del declared_index
        start, end, replacement, operation = plan
        if operation == "insert" and start > 0 and lines[start - 1] and not lines[start - 1].endswith("\n"):
            lines[start - 1] += "\n"
        lines[start:end] = replacement
    return "".join(lines)


_WHITESPACE = re.compile(r"\s+")


def _check_line_guard(lines: list[str], start: int, edit: dict[str, Any], index: int) -> None:
    expected = edit.get("old_string_prefix")
    if expected is None:
        return
    expected = _normalize(_require_string(expected))
    if not expected:
        raise Trace2SkillEditError(f"edit #{index}: old_string_prefix must be non-empty")
    actual = _normalize(lines[start - 1].rstrip("\n"))
    if not actual.startswith(expected):
        raise Trace2SkillEditError(
            f"edit #{index}: line {start} starts with {actual!r}, not old_string_prefix {expected!r}"
        )


def _check_overlap(ranges: list[tuple[int, int, int]]) -> None:
    ordered = sorted(ranges)
    for (_, first_end, first_index), (second_start, _, second_index) in zip(ordered, ordered[1:]):
        if second_start <= first_end:
            raise Trace2SkillEditError(f"edits #{first_index} and #{second_index} have overlapping ranges")


def _require_line(edit: dict[str, Any], key: str, index: int, *, lower: int, upper: int) -> int:
    value = edit.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Trace2SkillEditError(f"edit #{index}: {key!r} must be an integer")
    if value < lower or value > upper:
        raise Trace2SkillEditError(f"edit #{index}: {key}={value} is outside {lower}..{upper}")
    return value


def _to_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def _require_string(value: Any) -> str:
    if not isinstance(value, str):
        raise Trace2SkillEditError(f"expected string, got {type(value).__name__}")
    return value


def _normalize(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


__all__ = [
    "Trace2SkillEditError",
    "apply_edit_ops",
    "apply_patch_ops",
    "format_numbered",
    "parse_edit_ops",
]
