"""Resolve Memory Add Skill context from application-owned state."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..typing import SkillUsageType, compute_skill_content_hash
from .models import ManagedSkill

_TOOL_CALL_RE = re.compile(r"^\s*\[tool_call\]\s*([A-Za-z0-9_.-]+)\((.*)\)\s*$", re.DOTALL)
_SKILL_MD_RE = re.compile(r"(?:^|[/\\])SKILL\.md$")
_STRONG_USAGE = {SkillUsageType.MODIFIED: 2, SkillUsageType.INJECTED: 1}


class ResolvedSkillContext(BaseModel):
    """Exact Skill reference accepted by ``/v1/memory/add``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    base_version_id: str = ""
    version_label: str | None = None
    usage: SkillUsageType | None = None


@dataclass
class _Candidate:
    path: str
    content: str
    usage: SkillUsageType


def resolve_detected_contexts(
    messages: list[BaseModel | dict[str, Any]],
    *,
    skills: list[ManagedSkill],
) -> list[ResolvedSkillContext]:
    """Detect tool interactions and bind only unambiguous managed Skill families."""

    serialized = [_message_dict(message) for message in messages]
    candidates: dict[str, _Candidate] = {}
    for index, message in enumerate(serialized):
        if message.get("role") != "assistant":
            continue
        call = _parse_tool_call(str(message.get("content") or ""))
        if call is None:
            continue
        tool, args = call
        path = _arg_path(args)
        if not path or not _SKILL_MD_RE.search(path):
            continue
        if tool == "read":
            content = _next_tool_content(serialized, index)
            usage = SkillUsageType.INJECTED
        elif tool == "write":
            content = _arg_text(args, "content")
            usage = SkillUsageType.MODIFIED
        elif tool == "edit":
            content = _edit_content(args)
            usage = SkillUsageType.MODIFIED
        else:
            continue
        if not content:
            continue
        key = str(Path(path.replace("\\", "/")).parent)
        current = candidates.get(key)
        if current is None or _STRONG_USAGE[usage] >= _STRONG_USAGE[current.usage]:
            candidates[key] = _Candidate(path=path, content=content, usage=usage)

    resolved: list[ResolvedSkillContext] = []
    for candidate in candidates.values():
        detected_name = _find_simple_field(candidate.content, "name") or Path(candidate.path).parent.name
        matches = [skill for skill in skills if detected_name in {skill.name, skill.alias}]
        if len(matches) != 1:
            continue
        skill = matches[0]
        resolved.append(
            ResolvedSkillContext(
                name=skill.name,
                content_hash=compute_skill_content_hash({"SKILL.md": candidate.content}),
                base_version_id=skill.latest_version_id,
                version_label=_find_simple_field(candidate.content, "version"),
                usage=candidate.usage,
            )
        )
    return resolved


def _message_dict(message: BaseModel | dict[str, Any]) -> dict[str, Any]:
    return message.model_dump(mode="python") if isinstance(message, BaseModel) else dict(message)


def _parse_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    match = _TOOL_CALL_RE.match(content)
    if not match:
        return None
    try:
        args = json.loads(match.group(2).strip() or "{}")
    except json.JSONDecodeError:
        return None
    return (match.group(1).strip().lower(), args) if isinstance(args, dict) else None


def _arg_path(args: dict[str, Any]) -> str | None:
    for key in ("path", "file_path", "filepath"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _arg_text(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    return value if isinstance(value, str) else ""


def _edit_content(args: dict[str, Any]) -> str:
    for key in ("content", "new_content", "replacement", "replace"):
        value = _arg_text(args, key)
        if value:
            return value
    return ""


def _next_tool_content(messages: list[dict[str, Any]], index: int) -> str:
    if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
        return ""
    content = messages[index + 1].get("content")
    return content if isinstance(content, str) else ""


def _find_simple_field(content: str, field: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(field)}\s*:\s*[\"']?([^\"'\n#]+)", content, re.MULTILINE)
    return match.group(1).strip() if match and match.group(1).strip() else None


__all__ = ["ResolvedSkillContext", "resolve_detected_contexts"]
