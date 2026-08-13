"""Agent-neutral contracts for task-scoped dynamic Skill execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ..typing import Skill, Task


class SkillResourceDescriptor(BaseModel):
    """A resource advertised to an Agent without eagerly injecting its payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    media_type: str = "text/markdown"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SkillResourcePayload(BaseModel):
    """The result of one explicit task-scoped resource load."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_id: str = Field(min_length=1)
    content: str
    media_type: str = "text/markdown"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SkillRuntimeRequest(BaseModel):
    """Bounded facts passed to exactly one Skill Runtime for one task."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task: Task
    skill: Skill
    context: Mapping[str, Any] = Field(default_factory=dict)


class SkillRuntimeSession(ABC):
    """Mutable state isolated to one Skill and one physical rollout attempt."""

    def __init__(
        self,
        *,
        skill: Skill,
        initial_content: str,
        resources: list[SkillResourceDescriptor] | None = None,
    ) -> None:
        self.skill = skill
        self.initial_content = initial_content
        self.resources = list(resources or [])
        self.loaded_resource_ids: set[str] = set()
        self._closed = False

    async def load(self, resource_id: str) -> SkillResourcePayload:
        if self._closed:
            raise RuntimeError("Skill Runtime session is closed")
        known = {item.resource_id for item in self.resources}
        if resource_id not in known:
            raise KeyError(f"unknown Skill resource: {resource_id}")
        payload = await self._load(resource_id)
        if payload.resource_id != resource_id:
            raise RuntimeError("Skill Runtime returned a mismatched resource_id")
        self.loaded_resource_ids.add(resource_id)
        return payload

    async def materialize(self, resource_id: str) -> SkillResourcePayload:
        """Resolve a payload for a native filesystem adapter without marking it read."""

        if self._closed:
            raise RuntimeError("Skill Runtime session is closed")
        known = {item.resource_id for item in self.resources}
        if resource_id not in known:
            raise KeyError(f"unknown Skill resource: {resource_id}")
        payload = await self._load(resource_id)
        if payload.resource_id != resource_id:
            raise RuntimeError("Skill Runtime returned a mismatched resource_id")
        return payload

    @abstractmethod
    async def _load(self, resource_id: str) -> SkillResourcePayload:
        """Resolve one resource owned by this session."""

    async def aclose(self) -> None:
        self._closed = True


class SkillRuntime(ABC):
    """Parser and task callback implemented by one dynamic Skill scheme."""

    runtime_type: ClassVar[str]
    metadata_models: ClassVar[Mapping[int, type[BaseModel]]]

    def parse_metadata(self, schema_version: int, metadata: Mapping[str, JsonValue]) -> BaseModel:
        model = self.metadata_models.get(schema_version)
        if model is None:
            supported = ", ".join(str(item) for item in sorted(self.metadata_models)) or "<none>"
            raise ValueError(
                f"Runtime {self.runtime_type!r} does not support schema version {schema_version}; "
                f"supported: {supported}"
            )
        return model.model_validate(dict(metadata))

    @abstractmethod
    async def on_task(self, request: SkillRuntimeRequest) -> SkillRuntimeSession:
        """Assemble one isolated session for the concrete task."""


__all__ = [
    "SkillResourceDescriptor",
    "SkillResourcePayload",
    "SkillRuntime",
    "SkillRuntimeRequest",
    "SkillRuntimeSession",
]
