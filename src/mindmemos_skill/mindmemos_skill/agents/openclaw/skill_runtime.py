"""OpenClaw-native filesystem Skill injection and trace binding."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...typing import (
    Skill,
    SkillBinding,
    SkillInjectionMode,
    SkillUsageType,
    Trajectory,
    compute_skill_content_hash,
    normalize_skill_text,
)
from ..skill_runtime import SkillInjection, SkillRuntime

_TOOL_CALL = re.compile(r"^\s*\[tool_call\]\s*([A-Za-z0-9_.-]+)\((.*)\)\s*$", re.DOTALL)
_SKILL_MD = re.compile(r"(?:^|[/\\])SKILL\.md$")
_FIELD = r"^\s*{field}\s*:\s*[\"']?([^\"'\n#]+)"
_USAGE_PRIORITY = {SkillUsageType.INJECTED: 1, SkillUsageType.MODIFIED: 2}


@dataclass(slots=True)
class _DetectedBinding:
    path: str
    content: str
    name: str
    version_label: str | None
    usage: SkillUsageType


class OpenClawSkillRuntime(SkillRuntime):
    """Materialize workspace Skills and bind OpenClaw read/write/edit evidence."""

    supported_modes = frozenset({SkillInjectionMode.FILESYSTEM})

    @contextmanager
    def inject(self, skills: list[Skill]) -> Iterator[SkillInjection]:
        workspace = Path(tempfile.mkdtemp(prefix="mindmemos_openclaw_"))
        try:
            skills_root = workspace / "skills"
            skills_root.mkdir()
            used_directories: set[str] = set()
            for skill in skills:
                directory_name = _safe_skill_name(skill.name)
                if directory_name in used_directories:
                    raise ValueError(f"duplicate OpenClaw Skill directory: {directory_name!r}")
                used_directories.add(directory_name)
                _materialize_skill(skills_root / directory_name, skill)
            yield SkillInjection(
                mode=self.mode,
                skill_names={skill.name for skill in skills},
                workspace=str(workspace),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def bind(self, trajectory: Trajectory) -> list[SkillBinding]:
        detected = _detect_bindings(trajectory.events, trajectory.injected_skills)
        bindings: list[SkillBinding] = []
        matched_version_ids: set[str] = set()
        for item in detected:
            content_hash = compute_skill_content_hash({"SKILL.md": item.content})
            base = _match_injected_skill(item, content_hash, trajectory.injected_skills)
            unchanged = base is not None and base.content_hash == content_hash
            if base is not None:
                matched_version_ids.add(base.version_id)
            bindings.append(
                SkillBinding(
                    name=item.name,
                    content_hash=content_hash,
                    skill_id=base.skill_id if base is not None else None,
                    base_version_id=base.version_id if base is not None else None,
                    version_id=base.version_id if unchanged else None,
                    version_label=item.version_label or (base.version_label if base is not None else None),
                    usage=item.usage,
                    injection_mode=self.mode,
                )
            )

        bindings.extend(
            SkillBinding(
                name=skill.name,
                content_hash=skill.content_hash,
                skill_id=skill.skill_id,
                version_id=skill.version_id,
                version_label=skill.version_label,
                usage=SkillUsageType.UNUSED,
                injection_mode=self.mode,
            )
            for skill in trajectory.injected_skills
            if skill.version_id not in matched_version_ids
        )
        return bindings


def _materialize_skill(skill_root: Path, skill: Skill) -> None:
    if "SKILL.md" not in skill.blob:
        raise ValueError(f"OpenClaw agents require {skill.name!r} to contain SKILL.md in blob")
    duplicate_paths = skill.blob.keys() & skill.resources.keys()
    if duplicate_paths:
        duplicates = ", ".join(sorted(duplicate_paths))
        raise ValueError(f"OpenClaw cannot materialize overlapping Skill paths: {duplicates}")
    root = skill_root.resolve()
    for relative, content in {**skill.blob, **skill.resources}.items():
        destination = (skill_root / relative).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Skill file path escapes its OpenClaw directory: {relative!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def _detect_bindings(messages: list[dict[str, Any]], injected: list[Skill]) -> list[_DetectedBinding]:
    candidates: dict[str, _DetectedBinding] = {}
    current_contents: dict[str, str] = {}
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        parsed = _parse_tool_call(message)
        if parsed is None:
            continue
        tool, arguments, call_id = parsed
        path = _argument_path(arguments)
        if path is None or _SKILL_MD.search(path) is None:
            continue
        key = path.replace("\\", "/").rsplit("/", 1)[0]
        base_content = current_contents.get(key) or _injected_content_for_path(path, injected)
        if tool == "read":
            content = _tool_result_content(messages, index, call_id)
            usage = SkillUsageType.INJECTED
        elif tool == "write":
            if _tool_call_failed(messages, index, call_id):
                continue
            content = _argument_text(arguments, "content")
            usage = SkillUsageType.MODIFIED
        elif tool == "edit":
            if _tool_call_failed(messages, index, call_id):
                continue
            content = _edited_content(arguments, base_content)
            usage = SkillUsageType.MODIFIED
        else:
            continue
        if not content:
            continue
        content = normalize_skill_text(content)
        current_contents[key] = content
        matched = _match_injected_path(path, injected)
        name = _frontmatter_value(content, "name") or (
            matched.name if matched is not None else Path(key).name or "skill"
        )
        candidate = _DetectedBinding(
            path=path,
            content=content,
            name=name,
            version_label=_frontmatter_value(content, "version"),
            usage=usage,
        )
        previous = candidates.get(key)
        if previous is None or _USAGE_PRIORITY[usage] >= _USAGE_PRIORITY[previous.usage]:
            candidates[key] = candidate
    return list(candidates.values())


def _parse_tool_call(message: dict[str, Any]) -> tuple[str, dict[str, Any], str | None] | None:
    match = _TOOL_CALL.match(str(message.get("content") or ""))
    if match is None:
        return None
    try:
        arguments = json.loads(match.group(2).strip() or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    call_id = message.get("tool_call_id")
    return match.group(1).strip().lower(), arguments, call_id if isinstance(call_id, str) else None


def _argument_path(arguments: dict[str, Any]) -> str | None:
    for key in ("path", "file_path", "filepath"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _argument_text(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    return value if isinstance(value, str) else ""


def _tool_result_content(messages: list[dict[str, Any]], index: int, call_id: str | None) -> str:
    for message in messages[index + 1 :]:
        if call_id is not None and message.get("tool_call_id") != call_id:
            continue
        if message.get("role") != "tool" or message.get("is_error") is True:
            if call_id is not None:
                continue
            return ""
        content = message.get("content")
        return content if isinstance(content, str) else ""
    return ""


def _tool_call_failed(messages: list[dict[str, Any]], index: int, call_id: str | None) -> bool:
    for message in messages[index + 1 :]:
        if message.get("role") != "tool":
            if call_id is None:
                return False
            continue
        if call_id is not None and message.get("tool_call_id") != call_id:
            continue
        return message.get("is_error") is True
    return False


def _edited_content(arguments: dict[str, Any], base_content: str) -> str:
    for key in ("content", "new_content", "replacement", "replace"):
        value = _argument_text(arguments, key)
        if value:
            return value
    if not base_content:
        return ""
    edits = arguments.get("edits")
    if isinstance(edits, list):
        content = base_content
        for edit in edits:
            if not isinstance(edit, dict):
                return ""
            old = _first_string(edit, "oldText", "old_string", "old")
            new = _first_string(edit, "newText", "new_string", "new")
            if old is None or new is None or old not in content:
                return ""
            content = content.replace(old, new, 1)
        return content
    old = _first_string(arguments, "oldText", "old_string", "old")
    new = _first_string(arguments, "newText", "new_string", "new")
    return base_content.replace(old, new, 1) if old is not None and new is not None and old in base_content else ""


def _first_string(values: dict[str, Any], *keys: str) -> str | None:
    return next((value for key in keys if isinstance((value := values.get(key)), str)), None)


def _frontmatter_value(content: str, field: str) -> str | None:
    match = re.search(_FIELD.format(field=re.escape(field)), content, re.MULTILINE)
    return match.group(1).strip() if match else None


def _match_injected_skill(item: _DetectedBinding, content_hash: str, skills: list[Skill]) -> Skill | None:
    matches = [skill for skill in skills if skill.name == item.name]
    if len(matches) == 1:
        return matches[0]
    path_match = _match_injected_path(item.path, skills)
    if path_match is not None:
        return path_match
    hash_matches = [skill for skill in skills if skill.content_hash == content_hash]
    return hash_matches[0] if len(hash_matches) == 1 else None


def _match_injected_path(path: str, skills: list[Skill]) -> Skill | None:
    directory = Path(path.replace("\\", "/")).parent.name
    matches = [skill for skill in skills if directory in {skill.name, _safe_skill_name(skill.name)}]
    return matches[0] if len(matches) == 1 else None


def _injected_content_for_path(path: str, skills: list[Skill]) -> str:
    matched = _match_injected_path(path, skills)
    return matched.content if matched is not None else ""


def _safe_skill_name(name: str) -> str:
    safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in name)
    return safe or "skill"


__all__ = ["OpenClawSkillRuntime"]
