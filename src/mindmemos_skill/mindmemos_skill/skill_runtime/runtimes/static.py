"""Compatibility Runtime for traditional full-text Skills."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..contracts import SkillResourcePayload, SkillRuntime, SkillRuntimeRequest, SkillRuntimeSession


class StaticRuntimeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StaticSkillRuntimeSession(SkillRuntimeSession):
    async def _load(self, resource_id: str) -> SkillResourcePayload:
        raise KeyError(f"static Skills expose no dynamic resource: {resource_id}")


class StaticSkillRuntime(SkillRuntime):
    runtime_type = "static"
    metadata_models = {1: StaticRuntimeMetadata}

    async def on_task(self, request: SkillRuntimeRequest) -> SkillRuntimeSession:
        self.parse_metadata(request.skill.runtime_schema_version, request.skill.runtime_metadata)
        return StaticSkillRuntimeSession(skill=request.skill, initial_content=request.skill.content)


__all__ = ["StaticRuntimeMetadata", "StaticSkillRuntime", "StaticSkillRuntimeSession"]
