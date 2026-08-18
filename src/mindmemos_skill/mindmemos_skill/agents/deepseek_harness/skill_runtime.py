"""DeepSeek Harness-native Skill injection and trace binding.

The Harness discovers Skills from local roots (``dsh-skill-filesystem``):
``<projectRoot>/.dsh/skills/<name>/SKILL.md`` is the highest-ranked project
root, where the project root is the nearest ``.git`` ancestor or the current
working directory.  Each Skill must carry kebab-case frontmatter ``name`` and a
``description`` or it is dropped from the model catalog.

Two injection modes are supported:

- ``filesystem``: materialize MindMemOS Skills into one temporary workspace's
  ``.dsh/skills/`` tree and register that tree through an isolated
  ``customSkillDirs`` provider.  The agent keeps the task working directory.
- ``system_prompt``: interactive environments (for example LiveMath) build
  the system message themselves from ``skills`` and drive the agent turn by
  turn through ``respond()``; this mode only records the injection and
  exposes the Skill text as a suffix instead of materializing files.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ...typing import (
    Skill,
    SkillBinding,
    SkillInjectionMode,
    Trajectory,
)
from ..skill_runtime import SkillInjection, SkillRuntime

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
_FIELD_RE = re.compile(r"^\s*{field}\s*:\s*(.+?)\s*$", re.MULTILINE)
_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SAFE_PLAIN_RE = re.compile(r"^[A-Za-z0-9 _./-]+$")


class DeepSeekHarnessSkillRuntime(SkillRuntime):
    """Materialize or record Skills for the Harness and bind usage evidence."""

    supported_modes = frozenset({SkillInjectionMode.FILESYSTEM, SkillInjectionMode.SYSTEM_PROMPT})

    @contextmanager
    def inject(self, skills: list[Skill]) -> Iterator[SkillInjection]:
        if not skills:
            yield SkillInjection(mode=self.mode)
            return

        if self.mode is SkillInjectionMode.SYSTEM_PROMPT:
            suffix = "\n\n".join(skill.content.strip() for skill in skills if skill.content.strip())
            yield SkillInjection(
                mode=self.mode,
                skill_names={skill.name for skill in skills},
                system_prompt_suffix=suffix or None,
            )
            return

        workspace = Path(tempfile.mkdtemp(prefix="mindmemos_dsh_"))
        try:
            skills_root = workspace / ".dsh" / "skills"
            skills_root.mkdir(parents=True)
            used_directories: set[str] = set()
            for skill in skills:
                directory_name = _safe_skill_name(skill.name)
                if directory_name in used_directories:
                    raise ValueError(f"duplicate DeepSeek Harness Skill directory: {directory_name!r}")
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
        loaded_names = _extract_used_skill_names(
            trajectory.events,
            trajectory.injected_skills,
            mode=self.mode,
        )
        return self._build_bindings(trajectory, loaded_names)


def _materialize_skill(skill_root: Path, skill: Skill) -> None:
    if "SKILL.md" not in skill.blob:
        raise ValueError(f"DeepSeek Harness agents require {skill.name!r} to contain SKILL.md in blob")
    duplicate_paths = skill.blob.keys() & skill.resources.keys()
    if duplicate_paths:
        duplicates = ", ".join(sorted(duplicate_paths))
        raise ValueError(f"DeepSeek Harness cannot materialize overlapping Skill paths: {duplicates}")
    root = skill_root.resolve()
    for relative, content in {**skill.blob, **skill.resources}.items():
        destination = (skill_root / relative).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Skill file path escapes its DeepSeek Harness directory: {relative!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    skill_md = skill_root / "SKILL.md"
    skill_md.write_text(
        _ensure_harness_frontmatter(
            skill_md.read_text(encoding="utf-8"),
            name=_safe_skill_name(skill.name),
            description=skill.description or f"MindMemOS Skill: {skill.name}",
        ),
        encoding="utf-8",
    )


def _ensure_harness_frontmatter(content: str, *, name: str, description: str) -> str:
    """Return ``content`` with valid Harness frontmatter (kebab-case name + description)."""

    frontmatter, body = _split_frontmatter(content)
    fields = _parse_frontmatter(frontmatter)
    fields["name"] = name
    fields["description"] = description
    rendered = "\n".join(f"{key}: {_render_value(value)}" for key, value in fields.items())
    return f"---\n{rendered}\n---\n\n{body}"


def _render_value(value: str) -> str:
    """Render one frontmatter value; quote it when plain YAML would break."""

    if _SAFE_PLAIN_RE.fullmatch(value):
        return value
    return json.dumps(value, ensure_ascii=False)


def _split_frontmatter(content: str) -> tuple[str, str]:
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        return "", content
    return match.group(1), content[match.end() :]


def _parse_frontmatter(frontmatter: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in (frontmatter or "").splitlines():
        match = _FIELD_RE.match(line)
        if match is None:
            continue
        key = match.group(1)
        if key in fields:
            continue
        fields[key] = match.group(2).strip().strip("\"'")
    return fields


def _extract_used_skill_names(
    messages: list[dict[str, Any]],
    skills: list[Skill],
    *,
    mode: SkillInjectionMode,
) -> set[str]:
    """Interpret only strong Harness evidence, never task-side name mentions."""

    used: set[str] = set()
    for skill in skills:
        slug = _safe_skill_name(skill.name)
        explicit_markers = (
            f".dsh/skills/{slug}/SKILL.md",
            f'<skill_content name="{slug}"',
            f"<skill_content name='{slug}'",
        )
        for message in messages:
            role = message.get("role")
            text = _message_text(message)
            if mode is SkillInjectionMode.SYSTEM_PROMPT:
                if role == "system" and skill.content.strip() and skill.content.strip() in text:
                    used.add(skill.name)
                    break
                continue
            if role != "assistant":
                continue
            if _has_skill_tool_call(message, skill_name=skill.name, slug=slug) or any(
                marker in text for marker in explicit_markers
            ):
                used.add(skill.name)
                break
    return used


def _has_skill_tool_call(message: dict[str, Any], *, skill_name: str, slug: str) -> bool:
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict) or function.get("name") != "skill":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        name = arguments.get("name")
        if name in {skill_name, slug}:
            return True
    return False


def _message_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        arguments = function.get("arguments", "") if isinstance(function, dict) else ""
        if isinstance(arguments, str):
            parts.append(arguments)
    return "\n".join(parts)


def _safe_skill_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug if _KEBAB_RE.match(slug) else "skill"


__all__ = ["DeepSeekHarnessSkillRuntime"]
