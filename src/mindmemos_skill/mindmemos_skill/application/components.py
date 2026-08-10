"""Internal component composition used by ``SkillApplication.from_config``."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..agents import Agent
from ..config import CompiledSkillApplicationConfig, CompiledSkillModelConfig, SkillExecutionConfig
from ..errors import SkillConfigurationError
from ..llm import LLMClient, get_router
from ..service import SkillAlgorithms
from ..service.protocols import SkillAnalyzer, SkillOptimizer
from .enums import SkillApplicationCapability


@dataclass(frozen=True, slots=True)
class AlgorithmBuildContext:
    """Resolved dependencies passed to a registered algorithm component."""

    models: Mapping[str, LLMClient]
    agents: Mapping[str, Agent[Any]]
    execution: SkillExecutionConfig
    config_hash: str


@dataclass(slots=True)
class RuntimeComponents:
    """Internally composed runtime resources with ordered lifecycle handling."""

    agents: dict[str, Agent[Any]] = field(default_factory=dict)
    skill_algorithms: SkillAlgorithms | None = None
    algorithm_owners: dict[str, str] = field(default_factory=dict)
    resources: list[Any] = field(default_factory=list)
    _started_resources: list[Any] = field(default_factory=list)
    _started: bool = False
    _closed: bool = False

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Skill runtime components are closed")
        if self._started:
            return
        try:
            for resource in self.resources:
                if getattr(resource, "start", None) is not None or getattr(resource, "close", None) is not None:
                    self._started_resources.append(resource)
                await _run_optional_hook(resource, "start")
        except BaseException:
            try:
                await self._close_started_resources()
            except BaseException:
                pass
            raise
        self._started = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        try:
            await self._close_started_resources()
        except BaseException as exc:
            if error is None:
                error = exc
        self._started = False
        if error is not None:
            raise error

    async def _close_started_resources(self) -> None:
        error: BaseException | None = None
        while self._started_resources:
            resource = self._started_resources.pop()
            try:
                await _run_optional_hook(resource, "close")
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error


def compose_runtime(config: CompiledSkillApplicationConfig) -> RuntimeComponents:
    """Create model clients, Agents, and algorithms from compiled configuration."""

    required_model_refs = {
        agent.model_ref
        for agent in config.runtime.agents.values()
        if agent.model_ref is not None and agent.component.requirements.requires_model_ref
    }
    required_model_refs.update(
        model_ref
        for algorithm in config.runtime.algorithms.values()
        for model_ref in algorithm.model_roles.values()
    )
    model_clients = {
        name: _build_model_client(name, config.runtime.models[name])
        for name in sorted(required_model_refs)
    }
    agents = _build_agents(config, model_clients)
    algorithm_instances = _build_algorithms(config, model_clients, agents)
    skill_algorithms, algorithm_owners = _build_skill_algorithms(config, algorithm_instances, agents)
    resources = _unique_resources([*model_clients.values(), *agents.values(), *algorithm_instances.values()])
    return RuntimeComponents(
        agents=agents,
        skill_algorithms=skill_algorithms,
        algorithm_owners=algorithm_owners,
        resources=resources,
    )


def _build_model_client(name: str, model: CompiledSkillModelConfig) -> LLMClient:
    endpoint = dict(model.options)
    reserved = {"model", "api_key", "api_base", "temperature"} & endpoint.keys()
    if reserved:
        raise SkillConfigurationError(
            f"model {name!r} options duplicate explicit fields: {', '.join(sorted(reserved))}"
        )
    endpoint.update(
        {
            "model": model.model,
            "api_base": model.api_base,
            "api_key": model.api_key.get_secret_value() if model.api_key is not None else None,
            "temperature": model.temperature,
        }
    )
    router, max_retries = get_router({"endpoints": [endpoint]}, model.model)
    return LLMClient(router, default_model=model.model, max_attempts=max_retries + 1)


def _build_agents(
    config: CompiledSkillApplicationConfig,
    model_clients: Mapping[str, LLMClient],
) -> dict[str, Agent[Any]]:
    agents: dict[str, Agent[Any]] = {}
    for name, compiled in config.runtime.agents.items():
        kwargs: dict[str, Any] = {"config": compiled.config}
        if compiled.component.requirements.requires_model_ref:
            assert compiled.model_ref is not None
            kwargs["llm"] = model_clients[compiled.model_ref]
        try:
            instance = compiled.component.factory(**kwargs)
        except Exception as exc:
            raise SkillConfigurationError(f"failed to construct agent {name!r}: {exc}") from exc
        if not isinstance(instance, Agent):
            raise SkillConfigurationError(f"agent component {name!r} did not construct an Agent")
        if compiled.component.requirements.requires_model_ref:
            assert compiled.model_ref is not None
            instance.attach_model_profile(_build_agent_model_profile(config.runtime.models[compiled.model_ref]))
        agents[name] = instance
    return agents


def _build_agent_model_profile(model: CompiledSkillModelConfig) -> dict[str, Any]:
    """Flatten the effective endpoint into the common AgentProfile vocabulary."""

    profile: dict[str, Any] = {
        "provider": model.provider,
        "model": model.model,
        "api_base": model.api_base,
        "api_key": model.api_key.get_secret_value() if model.api_key is not None else None,
        "temperature": model.temperature,
    }
    profile.update(model.options)
    return {name: value for name, value in profile.items() if value is not None}


def _build_algorithms(
    config: CompiledSkillApplicationConfig,
    model_clients: Mapping[str, LLMClient],
    agents: Mapping[str, Agent[Any]],
) -> dict[str, Any]:
    algorithms: dict[str, Any] = {}
    for name, compiled in config.runtime.algorithms.items():
        context = AlgorithmBuildContext(
            models=MappingProxyType(
                {role: model_clients[model_ref] for role, model_ref in compiled.model_roles.items()}
            ),
            agents=MappingProxyType(dict(agents)),
            execution=config.runtime.execution,
            config_hash=config.config_hash,
        )
        try:
            algorithms[name] = compiled.component.factory(config=compiled.config, context=context)
        except Exception as exc:
            raise SkillConfigurationError(f"failed to construct algorithm {name!r}: {exc}") from exc
    return algorithms


def _build_skill_algorithms(
    config: CompiledSkillApplicationConfig,
    instances: Mapping[str, Any],
    agents: Mapping[str, Agent[Any]],
) -> tuple[SkillAlgorithms | None, dict[str, str]]:
    analyzer: SkillAnalyzer | None = None
    optimizer: SkillOptimizer | None = None
    owners: dict[str, str] = {}
    for name, instance in instances.items():
        capabilities = config.runtime.algorithms[name].component.capabilities
        if SkillApplicationCapability.ANALYZE.value in capabilities:
            if analyzer is not None:
                raise SkillConfigurationError("multiple configured algorithms provide analyze")
            if not isinstance(instance, SkillAnalyzer):
                raise SkillConfigurationError(f"algorithm {name!r} declares analyze but does not implement it")
            analyzer = instance
            owners[SkillApplicationCapability.ANALYZE.value] = name
        if SkillApplicationCapability.OPTIMIZE.value in capabilities:
            if optimizer is not None:
                raise SkillConfigurationError("multiple configured algorithms provide optimize")
            if not isinstance(instance, SkillOptimizer):
                raise SkillConfigurationError(f"algorithm {name!r} declares optimize but does not implement it")
            optimizer = instance
            owners[SkillApplicationCapability.OPTIMIZE.value] = name
    if analyzer is None and optimizer is None:
        return None, owners
    return SkillAlgorithms(analyzer=analyzer, optimizer=optimizer, agents=agents), owners


def _unique_resources(resources: list[Any]) -> list[Any]:
    unique: list[Any] = []
    identities: set[int] = set()
    for resource in resources:
        if id(resource) not in identities:
            identities.add(id(resource))
            unique.append(resource)
    return unique


async def _run_optional_hook(resource: Any, name: str) -> None:
    hook = getattr(resource, name, None)
    if hook is None:
        return
    result = hook()
    if inspect.isawaitable(result):
        await result


__all__ = ["AlgorithmBuildContext", "RuntimeComponents", "compose_runtime"]
