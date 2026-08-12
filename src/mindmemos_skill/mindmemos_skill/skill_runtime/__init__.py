"""Public dynamic Skill Runtime extension surface."""

from .contracts import (
    SkillResourceDescriptor,
    SkillResourcePayload,
    SkillRuntime,
    SkillRuntimeRequest,
    SkillRuntimeSession,
)
from .coordinator import SkillRuntimeCoordinator, SkillRuntimeTask
from .registry import SkillRuntimeRegistry
from .runtimes import (
    ComponentSelector,
    StaticSkillRuntime,
    VirtualComponent,
    VirtualComponentsMetadata,
    VirtualComponentsRuntime,
)


def build_default_skill_runtime_registry(
    *,
    virtual_component_selector: ComponentSelector | None = None,
) -> SkillRuntimeRegistry:
    return SkillRuntimeRegistry(
        (
            StaticSkillRuntime(),
            VirtualComponentsRuntime(selector=virtual_component_selector),
        )
    )


def build_default_skill_runtime_coordinator(
    *,
    virtual_component_selector: ComponentSelector | None = None,
) -> SkillRuntimeCoordinator:
    return SkillRuntimeCoordinator(
        build_default_skill_runtime_registry(
            virtual_component_selector=virtual_component_selector,
        )
    )


__all__ = [
    "ComponentSelector",
    "SkillResourceDescriptor",
    "SkillResourcePayload",
    "SkillRuntime",
    "SkillRuntimeCoordinator",
    "SkillRuntimeRegistry",
    "SkillRuntimeRequest",
    "SkillRuntimeSession",
    "SkillRuntimeTask",
    "StaticSkillRuntime",
    "VirtualComponent",
    "VirtualComponentsMetadata",
    "VirtualComponentsRuntime",
    "build_default_skill_runtime_coordinator",
    "build_default_skill_runtime_registry",
]
