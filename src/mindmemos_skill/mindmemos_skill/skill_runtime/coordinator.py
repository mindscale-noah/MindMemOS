"""Task-scope orchestration across heterogeneous Skill Runtime sessions."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from ..typing import Skill, Task
from .contracts import SkillResourcePayload, SkillRuntimeRequest, SkillRuntimeSession
from .registry import SkillRuntimeRegistry


class SkillRuntimeTask:
    """All Skill sessions mounted for one Agent task."""

    def __init__(self, sessions: Sequence[SkillRuntimeSession]) -> None:
        self.sessions = list(sessions)
        self._resource_owners: dict[str, SkillRuntimeSession] = {}
        for session in self.sessions:
            for descriptor in session.resources:
                if descriptor.resource_id in self._resource_owners:
                    raise ValueError(f"duplicate Skill resource_id: {descriptor.resource_id}")
                self._resource_owners[descriptor.resource_id] = session

    async def load(self, resource_id: str) -> SkillResourcePayload:
        owner = self._resource_owners.get(resource_id)
        if owner is None:
            raise KeyError(f"unknown Skill resource: {resource_id}")
        return await owner.load(resource_id)

    async def projected_skills(self, *, materialize_resources: bool = False) -> list[Skill]:
        """Project initial task content into the existing Agent adapter contract."""

        projected: list[Skill] = []
        for session in self.sessions:
            content = session.initial_content
            resources = dict(session.skill.resources)
            if materialize_resources and session.resources:
                lines = ["", "## Runtime resource files", ""]
                for index, descriptor in enumerate(session.resources, start=1):
                    payload = await session.materialize(descriptor.resource_id)
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", descriptor.name).strip("._") or "resource"
                    path = f"runtime_resources/{index:03d}-{safe_name}.md"
                    if path in resources:
                        raise ValueError(f"Runtime resource path conflicts with Skill resources: {path}")
                    resources[path] = payload.content
                    description = f" — {descriptor.description}" if descriptor.description else ""
                    lines.append(f"- `{path}`: {descriptor.name}{description}")
                lines.append("Read these files only when their guidance becomes relevant to the task.")
                content = content.rstrip() + "\n" + "\n".join(lines) + "\n"
            projected.append(
                session.skill.model_copy(
                    update={
                        "blob": {**session.skill.blob, "SKILL.md": content},
                        "resources": resources,
                    }
                )
            )
        return projected

    def trace(self) -> dict[str, Any]:
        return {
            "skills": [
                {
                    "skill_id": session.skill.skill_id,
                    "version_id": session.skill.version_id,
                    "runtime_type": session.skill.runtime_type,
                    "runtime_schema_version": session.skill.runtime_schema_version,
                    "available_resource_ids": [item.resource_id for item in session.resources],
                    "loaded_resource_ids": sorted(session.loaded_resource_ids),
                    "metadata": session.trace_metadata,
                }
                for session in self.sessions
            ]
        }


class SkillRuntimeCoordinator:
    def __init__(self, registry: SkillRuntimeRegistry) -> None:
        self.registry = registry

    @asynccontextmanager
    async def on_task(
        self,
        *,
        task: Task,
        skills: Sequence[Skill],
        context: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[SkillRuntimeTask]:
        sessions: list[SkillRuntimeSession] = []
        try:
            for skill in skills:
                runtime = self.registry.resolve(skill.runtime_type)
                runtime.validate_skill(skill)
                session = await runtime.on_task(SkillRuntimeRequest(task=task, skill=skill, context=context or {}))
                if session.skill.version_id != skill.version_id:
                    raise RuntimeError("Skill Runtime session changed immutable version identity")
                sessions.append(session)
            yield SkillRuntimeTask(sessions)
        finally:
            for session in reversed(sessions):
                await session.aclose()


__all__ = ["SkillRuntimeCoordinator", "SkillRuntimeTask"]
