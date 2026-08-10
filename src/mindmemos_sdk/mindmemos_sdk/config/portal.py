"""SDK portal v2 configuration and compilation.

The portal owns connection routing and composition.  Skill business settings
remain typed and validated by ``mindmemos_skill``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias

from mindmemos_skill.config import (
    CompiledSkillApplicationConfig,
    SkillApplicationConfig,
    SkillConfigCompiler,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..errors import ConfigValidationError
from .models import ConnectionConfig, DefaultsConfig, HttpConnectionConfig, MemoryDefaultsConfig

PortalConfigSourceV2: TypeAlias = Mapping[str, Any] | BaseModel


class _StrictPortalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SDKMemoryPortalConfigV2(_StrictPortalModel):
    """Memory route and request defaults for one profile."""

    connection: str | None = Field(default=None, min_length=1)
    defaults: MemoryDefaultsConfig = Field(default_factory=MemoryDefaultsConfig)


class SDKSkillRemoteConfigV2(_StrictPortalModel):
    """SDK-owned binding from Skill remote operations to a named connection."""

    connection: str | None = Field(default=None, min_length=1)


class SDKSkillPortalConfigV2(_StrictPortalModel):
    """Skill portal route plus the application config owned by the Skill package."""

    remote: SDKSkillRemoteConfigV2 = Field(default_factory=SDKSkillRemoteConfigV2)
    application: SkillApplicationConfig = Field(default_factory=SkillApplicationConfig)


class SDKProfileConfigV2(_StrictPortalModel):
    """One complete SDK composition profile."""

    default_connection: str = Field(default="default", min_length=1)
    connections: dict[str, ConnectionConfig] = Field(default_factory=dict)
    identity: DefaultsConfig = Field(default_factory=DefaultsConfig)
    memory: SDKMemoryPortalConfigV2 = Field(default_factory=SDKMemoryPortalConfigV2)
    skill: SDKSkillPortalConfigV2 = Field(default_factory=SDKSkillPortalConfigV2)


class SDKPortalConfigV2(_StrictPortalModel):
    """Top-level SDK portal v2 user intent."""

    version: Literal[2] = 2
    active_profile: str = Field(default="default", min_length=1)
    profiles: dict[str, SDKProfileConfigV2] = Field(default_factory=dict)


class CompiledSDKProfileV2(BaseModel):
    """Validated profile consumed by ``SDKPortalRuntime``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    connections: dict[str, ConnectionConfig]
    default_connection: str
    memory_connection: str
    skill_connection: str | None
    identity: DefaultsConfig
    memory_defaults: MemoryDefaultsConfig
    skill_application: CompiledSkillApplicationConfig


class CompiledSDKPortalConfigV2(BaseModel):
    """Resolved portal v2 config with one selected profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[2] = 2
    active_profile: str
    profile: CompiledSDKProfileV2


class SDKConfigMigrationPlanV1ToV2(BaseModel):
    """Validated, non-mutating description of one v1-to-v2 config migration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: Path
    target_path: Path
    backup_path: Path
    portal: SDKPortalConfigV2
    ignored_local_skill_entries: int = Field(ge=0)
    target_exists: bool
    changes_required: bool


class SDKConfigCompilerV2:
    """Resolve portal routes and delegate Skill validation to its owner package."""

    def __init__(self, *, skill_compiler: SkillConfigCompiler | None = None) -> None:
        self._skill_compiler = skill_compiler or SkillConfigCompiler()

    def compile(self, source: PortalConfigSourceV2) -> CompiledSDKPortalConfigV2:
        raw = source.model_dump(mode="python") if isinstance(source, BaseModel) else source
        try:
            config = SDKPortalConfigV2.model_validate(raw)
        except ValidationError as exc:
            raise ConfigValidationError(f"Invalid SDK portal config: {exc}") from exc
        try:
            profile = config.profiles[config.active_profile]
        except KeyError as exc:
            raise ConfigValidationError(f"active SDK profile does not exist: {config.active_profile!r}") from exc
        connections = dict(profile.connections)
        if not connections:
            raise ConfigValidationError(f"SDK profile {config.active_profile!r} must define at least one connection")
        _validate_connection_names(connections)
        memory_connection = profile.memory.connection or profile.default_connection
        skill_connection = profile.skill.remote.connection
        routes = [
            ("default", profile.default_connection),
            ("memory", memory_connection),
        ]
        if skill_connection is not None:
            routes.append(("skill.remote", skill_connection))
        for route, name in routes:
            if name not in connections:
                raise ConfigValidationError(
                    f"SDK profile {config.active_profile!r} {route} connection does not exist: {name!r}"
                )
        skill_application = self._skill_compiler.compile(profile.skill.application)
        return CompiledSDKPortalConfigV2(
            active_profile=config.active_profile,
            profile=CompiledSDKProfileV2(
                name=config.active_profile,
                connections=connections,
                default_connection=profile.default_connection,
                memory_connection=memory_connection,
                skill_connection=skill_connection,
                identity=profile.identity,
                memory_defaults=profile.memory.defaults,
                skill_application=skill_application,
            ),
        )


def _validate_connection_names(connections: Mapping[str, HttpConnectionConfig]) -> None:
    invalid = sorted(name for name in connections if not name or name.strip() != name)
    if invalid:
        raise ConfigValidationError(f"invalid SDK connection names: {', '.join(repr(name) for name in invalid)}")


__all__ = [
    "CompiledSDKPortalConfigV2",
    "CompiledSDKProfileV2",
    "PortalConfigSourceV2",
    "SDKConfigCompilerV2",
    "SDKConfigMigrationPlanV1ToV2",
    "SDKMemoryPortalConfigV2",
    "SDKPortalConfigV2",
    "SDKProfileConfigV2",
    "SDKSkillPortalConfigV2",
    "SDKSkillRemoteConfigV2",
]
