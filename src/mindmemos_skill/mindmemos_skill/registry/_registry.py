"""对外提供统一的组件注册能力, env, datasets等"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..agents.base import Agent
    from ..agents.config import AgentConfig
    from ..envs.base import BaseEnv
    from ..typing import AgentType, EnvConfig


class ComponentType(StrEnum):
    ENV = "env"
    DATASET = "dataset"
    ALGO = "algo"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class ComponentRequirements:
    """Cross-component references and runtime support required by a component."""

    requires_model_ref: bool = False
    required_model_roles: frozenset[str] = frozenset()
    supported_skill_injection_modes: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Construction and validation metadata for one registered component."""

    component_type: ComponentType
    name: str
    factory: type[Any]
    config_model: type[BaseModel] | None = None
    capabilities: frozenset[str] = frozenset()
    requirements: ComponentRequirements = ComponentRequirements()


_COMPONENT_REGISTRY: dict[ComponentType, dict[str, ComponentSpec]] = {}
_BUILTINS_LOADED = False
_BUILTIN_MODULES = (
    "..agents.claude",
    "..agents.openclaw",
    "..agents.react",
    "..envs.registered_envs",
    "..datasets.registered_datasets",
    "..algos.evolve.skill_grpo_with_experience_validation",
    "..algos.evolve.skill_grpo_without_replay_buffer",
    "..algos.evolve.skill_grpo_with_replay_buffer",
    "..algos.evolve.task_virtual_skill",
    "..algos.trace2skill.trajectory_evidence_patch",
)


def register(
    *,
    type: ComponentType,
    name: str,
    config_model: type[BaseModel] | None = None,
    capabilities: frozenset[str] | set[str] = frozenset(),
    requirements: ComponentRequirements = ComponentRequirements(),
):
    """Register a component class under a type/name pair."""

    _require_component_type(type)
    if not name:
        raise ValueError("component name must not be empty")

    def decorator(cls: type[Any]) -> type[Any]:
        components = _COMPONENT_REGISTRY.setdefault(type, {})
        if name in components:
            raise ValueError(f"{type} component {name!r} is already registered")
        resolved_config_model = config_model or getattr(cls, "config_type", None)
        resolved_requirements = requirements
        if type is ComponentType.AGENT and not requirements.supported_skill_injection_modes:
            runtime_types = getattr(cls, "skill_runtime_types", {})
            resolved_requirements = replace(
                requirements,
                supported_skill_injection_modes=frozenset(
                    mode.value if hasattr(mode, "value") else str(mode) for mode in runtime_types
                ),
            )
        components[name] = ComponentSpec(
            component_type=type,
            name=name,
            factory=cls,
            config_model=resolved_config_model,
            capabilities=frozenset(capabilities),
            requirements=resolved_requirements,
        )
        return cls

    return decorator


def create(*, type: ComponentType, name: str, **kwargs: Any) -> Any:
    """Create a registered component by type/name."""

    _require_component_type(type)
    load_builtin_components()
    component = _COMPONENT_REGISTRY.get(type, {}).get(name)
    if component is None:
        available = ", ".join(sorted(_COMPONENT_REGISTRY.get(type, {}))) or "<none>"
        raise ValueError(f"Unknown {type} component {name!r}. Available {type} components: {available}")
    return component.factory(**kwargs)


def get_component(*, type: ComponentType, name: str) -> ComponentSpec:
    """Return the immutable catalog entry for one registered component."""

    _require_component_type(type)
    load_builtin_components()
    component = _COMPONENT_REGISTRY.get(type, {}).get(name)
    if component is None:
        available = ", ".join(sorted(_COMPONENT_REGISTRY.get(type, {}))) or "<none>"
        raise ValueError(f"Unknown {type} component {name!r}. Available {type} components: {available}")
    return component


def list_components(*, type: ComponentType | None = None) -> dict[str, list[str]]:
    """List all registered components, optionally filtered by type."""

    load_builtin_components()
    if type is not None:
        _require_component_type(type)
        return {type.value: sorted(_COMPONENT_REGISTRY.get(type, {}))}
    return {component_type.value: sorted(names) for component_type, names in _COMPONENT_REGISTRY.items()}


def _require_component_type(value: ComponentType) -> None:
    if not isinstance(value, ComponentType):
        valid = ", ".join(item.value for item in ComponentType)
        raise TypeError(f"component type must be a ComponentType enum member; valid types: {valid}")


def get_agent(
    *,
    agent_type: AgentType | str,
    config: AgentConfig | Mapping[str, Any],
    **kwargs: Any,
) -> Agent[Any]:
    """Create a configured Agent through the unified component registry."""

    from ..agents.base import Agent
    from ..typing import AgentType

    try:
        normalized_type = AgentType(agent_type)
    except ValueError as exc:
        raise ValueError(f"Unknown agent type: {agent_type!r}") from exc
    return cast(Agent[Any], create(type=ComponentType.AGENT, name=normalized_type.value, config=config, **kwargs))


def list_agents() -> list[str]:
    """List Agent names registered in the unified component registry."""

    return list_components(type=ComponentType.AGENT).get(ComponentType.AGENT.value, [])


def get_env(
    *,
    name: str,
    config: EnvConfig | Mapping[str, Any],
    **kwargs: Any,
) -> BaseEnv[Any]:
    """Create an environment selected by a future trainer configuration."""

    from ..envs.base import BaseEnv

    return cast(BaseEnv[Any], create(type=ComponentType.ENV, name=name, config=config, **kwargs))


def list_envs() -> list[str]:
    """List built-in and package-external registered environment names."""

    return list_components(type=ComponentType.ENV).get(ComponentType.ENV.value, [])


def load_builtin_components() -> None:
    """Import built-in component modules so their decorators run."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    # Import built-in components so their @register decorators fire.
    # Mark the registry loaded only after every import succeeds so a partial
    # import cannot permanently hide missing components.
    for module_name in _BUILTIN_MODULES:
        import_module(module_name, package=__package__)
    _BUILTINS_LOADED = True


__all__ = [
    "ComponentRequirements",
    "ComponentSpec",
    "ComponentType",
    "create",
    "get_agent",
    "get_component",
    "get_env",
    "list_agents",
    "list_components",
    "list_envs",
    "load_builtin_components",
    "register",
]
