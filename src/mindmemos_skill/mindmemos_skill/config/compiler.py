"""Compile user-authored Skill configuration into a safe composition input."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SecretStr, ValidationError

from ..errors import SkillConfigurationError
from ..infra.database import DatabaseRegistry, DatabaseRequirements, register_builtin_databases
from ..llm.router import resolve_model_provider
from ..registry import ComponentSpec, ComponentType, get_component
from .models import (
    CompiledSkillAgentConfig,
    CompiledSkillAlgorithmConfig,
    CompiledSkillApplicationConfig,
    CompiledSkillDatabaseConfig,
    CompiledSkillLocalConfig,
    CompiledSkillModelConfig,
    CompiledSkillRuntimeConfig,
    SkillApplicationConfig,
    SkillConfigSource,
    SkillModelConfig,
)

ComponentResolver = Callable[..., ComponentSpec]

_REPOSITORY_REQUIREMENTS = DatabaseRequirements(
    metadata_filtering=True,
    batch_record_io=True,
    atomic_batch_write=True,
    transactions=True,
    compare_and_swap=True,
)
_SECRET_NAMES = {
    "api_key",
    "authorization",
    "client_secret",
    "password",
    "secret",
    "token",
}


class SkillConfigCompiler:
    """Resolve references and component contracts before opening resources."""

    def __init__(
        self,
        *,
        database_registry: DatabaseRegistry | None = None,
        component_resolver: ComponentResolver = get_component,
    ) -> None:
        if database_registry is None:
            database_registry = DatabaseRegistry()
            register_builtin_databases(database_registry)
        self._database_registry = database_registry
        self._component_resolver = component_resolver

    def compile(self, source: SkillConfigSource) -> CompiledSkillApplicationConfig:
        """Validate and normalize one mapping or typed application config."""

        config = self._parse_source(source)
        root_dir = _resolve_path(config.local.root_dir)
        artifacts_dir = _resolve_path(config.local.artifacts_dir or root_dir / "artifacts", base=root_dir)
        database = self._compile_database(config, root_dir=root_dir)
        models = self._compile_models(config.runtime.models)
        agents = self._compile_agents(config.runtime.agents, models=models)
        algorithms = self._compile_algorithms(config.runtime.algorithms, models=models)
        runtime = CompiledSkillRuntimeConfig(
            models=models,
            agents=agents,
            algorithms=algorithms,
            execution=config.runtime.execution,
        )
        local = CompiledSkillLocalConfig(
            root_dir=root_dir,
            database=database,
            artifacts_dir=artifacts_dir,
        )
        snapshot = _build_safe_snapshot(local=local, runtime=runtime)
        config_hash = hashlib.sha256(
            json.dumps(snapshot, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return CompiledSkillApplicationConfig(
            local=local,
            runtime=runtime,
            config_snapshot=snapshot,
            config_hash=config_hash,
        )

    @staticmethod
    def _parse_source(source: SkillConfigSource) -> SkillApplicationConfig:
        if isinstance(source, (str, Path)):
            raise SkillConfigurationError(
                "SkillConfigCompiler accepts a mapping or SkillApplicationConfig; "
                "the deployment config loader must read YAML or files first"
            )
        raw = source.model_dump(mode="python") if isinstance(source, BaseModel) else source
        try:
            return SkillApplicationConfig.model_validate(raw)
        except ValidationError as exc:
            raise SkillConfigurationError(f"invalid Skill application config: {exc}") from exc

    def _compile_database(
        self,
        config: SkillApplicationConfig,
        *,
        root_dir: Path,
    ) -> CompiledSkillDatabaseConfig:
        database = config.local.database
        provider = database.provider.strip().lower()
        if provider not in self._database_registry.providers:
            available = ", ".join(self._database_registry.providers) or "<none>"
            raise SkillConfigurationError(
                f"unknown database provider {database.provider!r}; available providers: {available}"
            )
        options = dict(database.options)
        configured_option_path = options.get("path")
        if configured_option_path is not None and not isinstance(configured_option_path, str):
            raise SkillConfigurationError("database.options.path must be a string")
        if database.path is not None and configured_option_path is not None:
            explicit = _resolve_path(database.path, base=root_dir)
            option = _resolve_path(configured_option_path, base=root_dir)
            if explicit != option:
                raise SkillConfigurationError("database.path conflicts with database.options.path")
        if database.path is not None or configured_option_path is not None or provider == "sqlite":
            selected_path = database.path or configured_option_path or Path("state.db")
            options["path"] = str(_resolve_path(selected_path, base=root_dir))
        return CompiledSkillDatabaseConfig(
            provider=provider,
            options=options,
            required=_REPOSITORY_REQUIREMENTS,
        )

    @staticmethod
    def _compile_models(models: Mapping[str, SkillModelConfig]) -> dict[str, CompiledSkillModelConfig]:
        compiled: dict[str, CompiledSkillModelConfig] = {}
        for name, model in models.items():
            normalized_name = _validate_reference_name(name, kind="model")
            reserved = {"provider", "model", "api_key", "api_base", "temperature"} & model.options.keys()
            if reserved:
                raise SkillConfigurationError(
                    f"model {normalized_name!r} options duplicate explicit fields: {', '.join(sorted(reserved))}"
                )
            model_id = model.model.strip()
            try:
                provider = resolve_model_provider(model_id, api_base=model.api_base)
            except ValueError as exc:
                raise SkillConfigurationError(f"invalid model {normalized_name!r}: {exc}") from exc
            compiled[normalized_name] = CompiledSkillModelConfig(
                provider=provider,
                model=model_id,
                api_base=model.api_base,
                api_key=model.api_key,
                temperature=model.temperature,
                options=dict(model.options),
            )
        return compiled

    def _compile_agents(
        self,
        agents: Mapping[str, Any],
        *,
        models: Mapping[str, CompiledSkillModelConfig],
    ) -> dict[str, CompiledSkillAgentConfig]:
        compiled: dict[str, CompiledSkillAgentConfig] = {}
        for name, agent in agents.items():
            normalized_name = _validate_reference_name(name, kind="agent")
            component = self._resolve_component(ComponentType.AGENT, agent.type)
            if component.config_model is None:
                raise SkillConfigurationError(f"agent component {agent.type!r} does not declare config_model")
            if component.requirements.requires_model_ref and agent.model_ref is None:
                raise SkillConfigurationError(f"agent {normalized_name!r} requires model_ref")
            model = None
            if agent.model_ref is not None:
                model = models.get(agent.model_ref)
                if model is None:
                    raise SkillConfigurationError(
                        f"agent {normalized_name!r} references unknown model {agent.model_ref!r}"
                    )
            payload = dict(agent.config)
            if "model" in payload:
                raise SkillConfigurationError(f"agent {normalized_name!r} must use model_ref instead of config.model")
            if model is not None:
                payload["model"] = model.model
            if agent.skill_injection_mode is not None:
                configured_mode = payload.get("skill_injection_mode")
                if configured_mode is not None and configured_mode != agent.skill_injection_mode:
                    raise SkillConfigurationError(
                        f"agent {normalized_name!r} declares conflicting skill_injection_mode values"
                    )
                payload["skill_injection_mode"] = agent.skill_injection_mode
            validated = _validate_component_config(
                component,
                payload,
                label=f"agent {normalized_name!r}",
            )
            mode = getattr(validated, "skill_injection_mode", None)
            supported = component.requirements.supported_skill_injection_modes
            if mode is not None and supported and mode.value not in supported:
                available = ", ".join(sorted(supported))
                raise SkillConfigurationError(
                    f"agent {normalized_name!r} does not support {mode.value!r} Skill injection; "
                    f"supported modes: {available}"
                )
            compiled[normalized_name] = CompiledSkillAgentConfig(
                name=normalized_name,
                type=component.name,
                model_ref=agent.model_ref,
                component=component,
                config=validated,
            )
        return compiled

    def _compile_algorithms(
        self,
        algorithms: Mapping[str, Any],
        *,
        models: Mapping[str, CompiledSkillModelConfig],
    ) -> dict[str, CompiledSkillAlgorithmConfig]:
        compiled: dict[str, CompiledSkillAlgorithmConfig] = {}
        for name, algorithm in algorithms.items():
            normalized_name = _validate_reference_name(name, kind="algorithm")
            component = self._resolve_component(ComponentType.ALGO, algorithm.type)
            if component.config_model is None:
                raise SkillConfigurationError(f"algorithm component {algorithm.type!r} does not declare config_model")
            if not component.capabilities.intersection({"analyze", "optimize", "evolve"}):
                raise SkillConfigurationError(
                    f"algorithm component {algorithm.type!r} must declare analyze, optimize, or evolve capability"
                )
            missing_roles = component.requirements.required_model_roles - algorithm.model_roles.keys()
            if missing_roles:
                raise SkillConfigurationError(
                    f"algorithm {normalized_name!r} is missing model roles: {', '.join(sorted(missing_roles))}"
                )
            for role, model_ref in algorithm.model_roles.items():
                _validate_reference_name(role, kind=f"model role for algorithm {normalized_name!r}")
                if model_ref not in models:
                    raise SkillConfigurationError(
                        f"algorithm {normalized_name!r} role {role!r} references unknown model {model_ref!r}"
                    )
            validated = _validate_component_config(
                component,
                algorithm.config,
                label=f"algorithm {normalized_name!r}",
            )
            compiled[normalized_name] = CompiledSkillAlgorithmConfig(
                name=normalized_name,
                type=component.name,
                model_roles=dict(algorithm.model_roles),
                component=component,
                config=validated,
            )
        return compiled

    def _resolve_component(self, component_type: ComponentType, name: str) -> ComponentSpec:
        try:
            return self._component_resolver(type=component_type, name=name.strip().lower())
        except ValueError as exc:
            raise SkillConfigurationError(str(exc)) from exc


def _validate_component_config(component: ComponentSpec, payload: Mapping[str, Any], *, label: str) -> BaseModel:
    assert component.config_model is not None
    try:
        return component.config_model.model_validate(payload)
    except ValidationError as exc:
        raise SkillConfigurationError(f"invalid config for {label}: {exc}") from exc


def _validate_reference_name(name: str, *, kind: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise SkillConfigurationError(f"{kind} name must not be empty")
    if normalized != name:
        raise SkillConfigurationError(f"{kind} name must not contain leading or trailing whitespace: {name!r}")
    return normalized


def _resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve(strict=False)


def _build_safe_snapshot(
    *,
    local: CompiledSkillLocalConfig,
    runtime: CompiledSkillRuntimeConfig,
) -> dict[str, Any]:
    models = {
        name: {
            "provider": model.provider,
            "model": model.model,
            "api_base": model.api_base,
            "api_key_configured": model.api_key is not None,
            "temperature": model.temperature,
            "options": _redact_secrets(model.options),
        }
        for name, model in sorted(runtime.models.items())
    }
    agents = {
        name: {
            "type": agent.type,
            "model_ref": agent.model_ref,
            "config": _redact_secrets(agent.config.model_dump(mode="json", exclude_none=True)),
        }
        for name, agent in sorted(runtime.agents.items())
    }
    algorithms = {
        name: {
            "type": algorithm.type,
            "model_roles": dict(sorted(algorithm.model_roles.items())),
            "config": _redact_secrets(algorithm.config.model_dump(mode="json", exclude_none=True)),
        }
        for name, algorithm in sorted(runtime.algorithms.items())
    }
    return {
        "local": {
            "root_dir": str(local.root_dir),
            "database": {
                "provider": local.database.provider,
                "options": _redact_secrets(local.database.options),
                "required": {
                    "metadata_filtering": local.database.required.metadata_filtering,
                    "batch_record_io": local.database.required.batch_record_io,
                    "atomic_batch_write": local.database.required.atomic_batch_write,
                    "transactions": local.database.required.transactions,
                    "compare_and_swap": local.database.required.compare_and_swap,
                },
            },
            "artifacts_dir": str(local.artifacts_dir),
        },
        "runtime": {
            "models": models,
            "agents": agents,
            "algorithms": algorithms,
            "execution": runtime.execution.model_dump(mode="json", exclude_none=True),
        },
    }


def _redact_secrets(value: Any, *, field_name: str | None = None) -> Any:
    if isinstance(value, SecretStr):
        return "<redacted>"
    if field_name is not None and field_name.lower() in _SECRET_NAMES:
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(key): _redact_secrets(item, field_name=str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    return value


__all__ = ["SkillConfigCompiler"]
