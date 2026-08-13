"""First dynamic Runtime scheme: task assembly plus progressive components."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts import (
    SkillResourceDescriptor,
    SkillResourcePayload,
    SkillRuntime,
    SkillRuntimeRequest,
    SkillRuntimeSession,
)


class VirtualComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    name: str = Field(min_length=1)
    description: str = ""
    content: str = Field(min_length=1)


class VirtualComponentsMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    components: list[VirtualComponent] = Field(min_length=1)
    max_initial_components: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def validate_components(self) -> VirtualComponentsMetadata:
        ids = [item.component_id for item in self.components]
        names = [item.name.casefold() for item in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("virtual component IDs must be unique")
        if len(names) != len(set(names)):
            raise ValueError("virtual component names must be unique")
        return self


ComponentSelector = Callable[
    [SkillRuntimeRequest, Sequence[VirtualComponent], int],
    Awaitable[Sequence[str]],
]


class VirtualComponentsSession(SkillRuntimeSession):
    def __init__(
        self,
        *,
        request: SkillRuntimeRequest,
        metadata: VirtualComponentsMetadata,
        selected_ids: set[str],
    ) -> None:
        self._components = {
            _resource_id(request.skill.version_id, item.component_id): item
            for item in metadata.components
            if item.component_id not in selected_ids
        }
        resources = [
            SkillResourceDescriptor(
                resource_id=resource_id,
                name=component.name,
                description=component.description,
                metadata={"component_id": component.component_id},
            )
            for resource_id, component in self._components.items()
        ]
        selected = [item for item in metadata.components if item.component_id in selected_ids]
        super().__init__(
            skill=request.skill,
            initial_content=_render_initial_content(request.skill.name, selected, resources),
            resources=resources,
        )

    async def _load(self, resource_id: str) -> SkillResourcePayload:
        component = self._components[resource_id]
        return SkillResourcePayload(
            resource_id=resource_id,
            content=component.content,
            metadata={"component_id": component.component_id, "name": component.name},
        )


class VirtualComponentsRuntime(SkillRuntime):
    runtime_type = "virtual_components"
    metadata_models = {1: VirtualComponentsMetadata}

    def __init__(self, selector: ComponentSelector | None = None) -> None:
        self._selector = selector

    async def on_task(self, request: SkillRuntimeRequest) -> SkillRuntimeSession:
        metadata = VirtualComponentsMetadata.model_validate(
            self.parse_metadata(request.skill.runtime_schema_version, request.skill.runtime_metadata)
        )
        if self._selector is None:
            selected_ids = _select_lexically(
                request.task.instruction,
                metadata.components,
                metadata.max_initial_components,
            )
        else:
            selected_ids = set(await self._selector(request, metadata.components, metadata.max_initial_components))
        valid_ids = {item.component_id for item in metadata.components}
        unknown = selected_ids - valid_ids
        if unknown:
            raise ValueError(f"component selector returned unknown IDs: {', '.join(sorted(unknown))}")
        if len(selected_ids) > metadata.max_initial_components:
            raise ValueError("component selector exceeded max_initial_components")
        return VirtualComponentsSession(request=request, metadata=metadata, selected_ids=selected_ids)


def _resource_id(version_id: str, component_id: str) -> str:
    return f"skill-resource:{version_id}:{component_id}"


def _tokens(value: str) -> set[str]:
    lowered = value.casefold()
    words = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    words.update(char for char in lowered if "\u4e00" <= char <= "\u9fff")
    return words


def _select_lexically(
    instruction: str,
    components: Sequence[VirtualComponent],
    limit: int,
) -> set[str]:
    if limit == 0:
        return set()
    task_tokens = _tokens(instruction)
    ranked = []
    for index, component in enumerate(components):
        title_tokens = _tokens(f"{component.name} {component.description}")
        score = len(task_tokens & title_tokens)
        if score:
            ranked.append((-score, index, component.component_id))
    ranked.sort()
    return {item[2] for item in ranked[:limit]}


def _render_initial_content(
    skill_name: str,
    selected: Sequence[VirtualComponent],
    resources: Sequence[SkillResourceDescriptor],
) -> str:
    lines = [f"# Task-assembled Skill: {skill_name}"]
    if selected:
        for component in selected:
            lines.extend(["", f"## {component.name}", "", component.content.strip()])
    else:
        lines.extend(["", "No component was selected eagerly for this task."])
    if resources:
        lines.extend(["", "## Additional resources", ""])
        lines.append("Load a resource when it becomes relevant during execution:")
        for item in resources:
            description = f" — {item.description}" if item.description else ""
            lines.append(f"- `{item.resource_id}`: {item.name}{description}")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ComponentSelector",
    "VirtualComponent",
    "VirtualComponentsMetadata",
    "VirtualComponentsRuntime",
    "VirtualComponentsSession",
]
