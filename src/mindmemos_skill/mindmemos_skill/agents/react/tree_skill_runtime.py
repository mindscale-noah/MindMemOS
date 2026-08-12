"""Query-aware TreeSkill routing for the OpenAI-compatible ReAct agent."""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from ...algos.trace2skill.treeskill.routing import TreeSkillRouter
from ...typing import AgentExecutionRequest, Skill, SkillBinding, SkillInjectionMode, Trajectory
from ..skill_runtime import RoutedSkillSnapshot, SkillInjection, SkillRoute, SkillRuntime


class ReactTreeSkillRuntime(SkillRuntime):
    """Route TreeSkill Markdown once, then inject the selected content."""

    supported_modes = frozenset({SkillInjectionMode.TREE_ROUTED_SYSTEM_PROMPT})

    def __init__(
        self,
        mode: SkillInjectionMode,
        *,
        llm: Any,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        super().__init__(mode)
        self._router = TreeSkillRouter(
            chat_model=llm,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def route(self, request: AgentExecutionRequest) -> SkillRoute:
        by_name = _unique_skills(request.skills)
        raw_context = request.metadata.get("treeskill_routing_context")
        routing_context = raw_context if isinstance(raw_context, Mapping) else None
        routed: list[RoutedSkillSnapshot] = []
        details: dict[str, Any] = {}
        full_chars = 0
        routed_chars = 0
        loaded_names: list[str] = []
        for name, skill in by_name.items():
            result = await self._router.route(
                skill=skill,
                task=request.task,
                env_ref=request.environment.env_ref,
                routing_context=routing_context,
            )
            detail = result.model_dump(mode="json", exclude={"skill_content"})
            details[name] = detail
            full_chars += result.full_char_count
            routed_chars += result.routed_char_count
            if result.skill_content.strip():
                loaded_names.append(name)
            routed.append(RoutedSkillSnapshot(skill_name=name, content=result.skill_content, metadata=detail))
        saving = 0.0 if full_chars == 0 else 1.0 - (routed_chars / full_chars)
        metadata = {
            "treeskill_routing": {
                "loaded_skill_names": loaded_names,
                "full_char_count": full_chars,
                "routed_char_count": routed_chars,
                "context_saving_ratio": saving,
                "skills": details,
            }
        }
        return SkillRoute(skills=tuple(routed), metadata=metadata)

    @contextmanager
    def inject(self, skills: list[Skill]) -> Iterator[SkillInjection]:
        """Inject full Skills when no query-aware request scope is available."""

        by_name = _unique_skills(skills)
        routed = {name: RoutedSkillSnapshot(skill_name=name, content=skill.content) for name, skill in by_name.items()}
        loaded = set(by_name)
        injection = SkillInjection(mode=self.mode, skill_names=loaded)
        if not loaded:
            yield injection
            return

        root = Path(tempfile.mkdtemp(prefix="mindmemos_treeskill_"))
        try:
            directories: dict[str, Path] = {}
            for name, skill in by_name.items():
                directory = root / "treeskill_routed_skills" / _safe_skill_name(name)
                _materialize_routed_skill(directory, skill, skill.content)
                directories[name] = directory
            injection.system_prompt_suffix = _routed_system_prompt_suffix(by_name, routed, directories)
            injection.workspace = str(root)
            yield injection
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @contextmanager
    def inject_routed(
        self,
        request: AgentExecutionRequest,
        route: SkillRoute,
    ) -> Iterator[SkillInjection]:
        by_name = _unique_skills(request.skills)
        routed_by_name = {item.skill_name: item for item in route.skills}
        if set(routed_by_name) != set(by_name):
            raise ValueError("TreeSkill route does not cover the request's exact Skill set")

        loaded = {name for name, item in routed_by_name.items() if item.content.strip()}
        injection = SkillInjection(mode=self.mode, skill_names=loaded, metadata=route.metadata)
        if not loaded:
            yield injection
            return

        running_dir = request.environment.running_dir
        owns_workspace = running_dir is None
        root = Path(tempfile.mkdtemp(prefix="mindmemos_treeskill_")) if owns_workspace else Path(running_dir)
        skills_root = root / "treeskill_routed_skills"
        try:
            directories: dict[str, Path] = {}
            used: set[Path] = set()
            for name in by_name:
                if name not in loaded:
                    continue
                directory = (skills_root / _safe_skill_name(name)).resolve()
                if directory in used:
                    raise ValueError(f"duplicate routed Skill directory: {directory.name!r}")
                used.add(directory)
                _materialize_routed_skill(directory, by_name[name], routed_by_name[name].content)
                directories[name] = directory
            injection.system_prompt_suffix = _routed_system_prompt_suffix(by_name, routed_by_name, directories)
            injection.workspace = str(root)
            yield injection
        finally:
            if owns_workspace:
                shutil.rmtree(root, ignore_errors=True)

    def bind(self, trajectory: Trajectory) -> list[SkillBinding]:
        metadata = trajectory.metadata.get("treeskill_routing")
        raw_names = metadata.get("loaded_skill_names", []) if isinstance(metadata, dict) else []
        loaded = {name for name in raw_names if isinstance(name, str)}
        return self._build_bindings(trajectory, loaded)


def _unique_skills(skills: list[Skill]) -> dict[str, Skill]:
    result: dict[str, Skill] = {}
    for skill in skills:
        if skill.name in result:
            raise ValueError(f"duplicate injected Skill name: {skill.name!r}")
        result[skill.name] = skill
    return result


def _safe_skill_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "skill"


def _materialize_routed_skill(directory: Path, skill: Skill, routed_content: str) -> None:
    root = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    files = {"SKILL.md": routed_content, **skill.resources}
    for relative, content in files.items():
        destination = (directory / relative).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Skill file path escapes its routed directory: {relative!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def _routed_system_prompt_suffix(
    skills: dict[str, Skill],
    routed: dict[str, RoutedSkillSnapshot],
    directories: dict[str, Path],
) -> str:
    lines = ["<available_skills>"]
    for name, skill in skills.items():
        item = routed[name]
        if not item.content.strip():
            continue
        lines.extend(
            [
                "  <skill>",
                f"    <name>{escape(name)}</name>",
                f"    <description>{escape(skill.description or '')}</description>",
                f"    <content>{escape(item.content.strip())}</content>",
            ]
        )
        if skill.resources:
            lines.append(f"    <resource_directory>{escape(str(directories[name]))}</resource_directory>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


__all__ = ["ReactTreeSkillRuntime"]
