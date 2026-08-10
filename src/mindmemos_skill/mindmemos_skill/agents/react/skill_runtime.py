"""Skill runtimes supported by the OpenAI-compatible ReAct family."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from ...typing import Skill, SkillBinding, SkillInjectionMode, Trajectory
from ..skill_runtime import SkillInjection, SkillRuntime
from .tool import Tool

_SKILL_TOOL_NAME = "skill"
_LOADED_SKILL_PREFIX = "Loaded skill "


def _extract_loaded_skill_names(messages: list[dict[str, Any]]) -> set[str]:
    """Interpret successful legacy-style Skill loads from a ReAct trajectory."""

    requested_by_call_id: dict[str, str] = {}
    loaded: set[str] = set()
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for raw_call in message.get("tool_calls") or []:
                call = raw_call if isinstance(raw_call, Mapping) else {}
                function = call.get("function")
                function = function if isinstance(function, Mapping) else {}
                if function.get("name") != _SKILL_TOOL_NAME:
                    continue
                call_id = call.get("id")
                arguments = function.get("arguments", "{}")
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    continue
                name = parsed.get("name") if isinstance(parsed, Mapping) else None
                if isinstance(call_id, str) and isinstance(name, str) and name:
                    requested_by_call_id[call_id] = name
            continue

        if message.get("role") != "tool" or message.get("name") != _SKILL_TOOL_NAME:
            continue
        call_id = message.get("tool_call_id")
        name = requested_by_call_id.get(call_id) if isinstance(call_id, str) else None
        if name is None or index + 1 >= len(messages):
            continue
        result = messages[index + 1]
        content = result.get("content") if result.get("role") == "user" else None
        if isinstance(content, str) and content.startswith(f"{_LOADED_SKILL_PREFIX}'{name}'.\n"):
            loaded.add(name)
    return loaded


def _skill_catalog_suffix(skills: Mapping[str, Skill]) -> str:
    lines = ["<available_skills>"]
    for name, skill in skills.items():
        lines.extend(
            [
                "  <skill>",
                f"    <name>{escape(name)}</name>",
                f"    <description>{escape(skill.description or '')}</description>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def _skill_system_prompt_suffix(skills: Mapping[str, Skill]) -> str | None:
    snapshots = [(skill, skill.content.strip()) for skill in skills.values() if skill.content.strip()]
    if not snapshots:
        return None

    lines = ["<available_skills>"]
    for skill, content in snapshots:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{escape(skill.name)}</name>",
                f"    <description>{escape(skill.description or '')}</description>",
                f"    <content>{escape(content)}</content>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def _build_skill_tool(directories: Mapping[str, Path]) -> Tool:
    available = ", ".join(directories) or "(none)"

    def load_skill(name: str) -> str:
        directory = directories.get(name)
        if directory is None:
            return f"Error: unknown skill '{name}'. Available skills: {available}"
        instructions = (directory / "SKILL.md").read_text(encoding="utf-8")
        return (
            f"Loaded skill '{name}'.\n"
            f"Skill directory (absolute path): {directory}\n"
            "Reference files live under that directory; read or run them with the read/shell tools as needed.\n\n"
            f"----- {name}/SKILL.md -----\n{instructions}"
        )

    return Tool(
        name=_SKILL_TOOL_NAME,
        description=(
            "Load an expert skill to get detailed instructions and the absolute path to its reusable reference scripts. "
            f"Call this before starting the task. Available skills: {available}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": f"Skill to load. One of: {available}.",
                }
            },
            "required": ["name"],
        },
        func=load_skill,
        deliver_result_as_user=True,
    )


def _safe_skill_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "skill"


def _materialize_skill(directory: Path, skill: Skill) -> None:
    root = directory.resolve()
    for relative, content in {**skill.blob, **skill.resources}.items():
        destination = (directory / relative).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Skill file path escapes its ReAct directory: {relative!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


class ReactSkillRuntime(SkillRuntime):
    """Inject and bind Skills according to one ReAct-supported mode."""

    supported_modes = frozenset({SkillInjectionMode.TOOL, SkillInjectionMode.SYSTEM_PROMPT})

    @contextmanager
    def inject(self, skills: list[Skill]) -> Iterator[SkillInjection]:
        by_name: dict[str, Skill] = {}
        for skill in skills:
            if skill.name in by_name:
                raise ValueError(f"duplicate injected Skill name: {skill.name!r}")
            by_name[skill.name] = skill

        injection = SkillInjection(mode=self.mode, skill_names=set(by_name))
        if by_name and self.mode is SkillInjectionMode.TOOL:
            workspace = Path(tempfile.mkdtemp(prefix="mindmemos_react_"))
            try:
                skills_root = workspace / "skills"
                directories: dict[str, Path] = {}
                used_directories: set[Path] = set()
                for name, skill in by_name.items():
                    directory = (skills_root / _safe_skill_name(name)).resolve()
                    if directory in used_directories:
                        raise ValueError(f"duplicate injected Skill directory: {directory.name!r}")
                    used_directories.add(directory)
                    _materialize_skill(directory, skill)
                    directories[name] = directory
                injection.system_prompt_suffix = _skill_catalog_suffix(by_name)
                injection.tools.append(_build_skill_tool(directories))
                injection.workspace = str(workspace)
                yield injection
            finally:
                shutil.rmtree(workspace, ignore_errors=True)
            return
        elif by_name and self.mode is SkillInjectionMode.SYSTEM_PROMPT:
            injection.system_prompt_suffix = _skill_system_prompt_suffix(by_name)
        yield injection

    def bind(self, trajectory: Trajectory) -> list[SkillBinding]:
        loaded_names = (
            {skill.name for skill in trajectory.injected_skills}
            if self.mode is SkillInjectionMode.SYSTEM_PROMPT
            else _extract_loaded_skill_names(trajectory.events)
        )
        return self._build_bindings(trajectory, loaded_names)


__all__ = ["ReactSkillRuntime"]
