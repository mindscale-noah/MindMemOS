"""Deterministic SKILL.md find/replace validation and application."""

from __future__ import annotations

import json
import re
from typing import Any

from .contracts import SkillTextEdit


class EditError(ValueError):
    pass


_WS_RUN = re.compile(r"\s+")


def locate(edit: SkillTextEdit, text: str) -> tuple[int, int]:
    if edit.find == "":
        return len(text), len(text)
    first = text.find(edit.find)
    if first >= 0:
        if text.find(edit.find, first + 1) >= 0:
            raise EditError(f"find is not unique: {edit.find!r}")
        return first, first + len(edit.find)
    needle = _WS_RUN.sub(" ", edit.find).strip()
    if not needle:
        raise EditError(f"find not found: {edit.find!r}")
    pattern = re.compile(r"\s+".join(re.escape(token) for token in needle.split(" ")))
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise EditError(f"find not found: {edit.find!r}")
    return matches[0].start(), matches[0].end()


def validate_edits(edits: list[SkillTextEdit], text: str) -> tuple[list[SkillTextEdit], list[str]]:
    valid: list[SkillTextEdit] = []
    errors: list[str] = []
    for edit in edits:
        try:
            locate(edit, text)
            valid.append(edit)
        except EditError as exc:
            errors.append(str(exc))
    return valid, errors


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    if left[0] == left[1] and right[0] == right[1]:
        return False
    if left[0] == left[1]:
        return right[0] < left[0] < right[1]
    if right[0] == right[1]:
        return left[0] < right[0] < left[1]
    return left[0] < right[1] and right[0] < left[1]


def conflict_groups(edits: list[SkillTextEdit], text: str) -> list[list[int]]:
    spans = [locate(edit, text) for edit in edits]
    parent = list(range(len(edits)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(edits)):
        for right in range(left + 1, len(edits)):
            if _overlap(spans[left], spans[right]):
                parent[root(left)] = root(right)
    groups: dict[int, list[int]] = {}
    for index in range(len(edits)):
        groups.setdefault(root(index), []).append(index)
    return sorted(groups.values(), key=lambda group: min(spans[index][0] for index in group))


def apply_edits(edits: list[SkillTextEdit], text: str) -> str:
    conflicts = [group for group in conflict_groups(edits, text) if len(group) > 1]
    if conflicts:
        raise EditError(f"{len(conflicts)} conflict group(s): {conflicts}")
    planned = [(locate(edit, text), edit) for edit in edits]
    planned.sort(key=lambda item: item[0][0])
    output = text
    for (start, end), edit in reversed(planned):
        if edit.find == "":
            separator = "" if not output or output.endswith("\n\n") else "\n\n"
            output = output[:start] + separator + edit.replace + output[end:]
        else:
            output = output[:start] + edit.replace + output[end:]
    return output


def apply_best_effort(edits: list[SkillTextEdit], text: str) -> tuple[str, list[SkillTextEdit]]:
    valid, _ = validate_edits(edits, text)
    if not valid:
        return text, []
    groups = conflict_groups(valid, text)
    chosen = [valid[index] for index in sorted(group[0] for group in groups)]
    return apply_edits(chosen, text), chosen


def load_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_extract_first_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


def _extract_first_json_object(raw: str) -> str:
    start = raw.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return ""


__all__ = [
    "EditError",
    "apply_best_effort",
    "apply_edits",
    "conflict_groups",
    "load_json_object",
    "locate",
    "validate_edits",
]
