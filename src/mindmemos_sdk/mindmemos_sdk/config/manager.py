"""Read, migrate, and atomically update SDK portal configuration.

``config.yaml`` is the only runtime configuration. ``settings.json`` is retained
only as a v1 migration input and is never updated by current SDK operations.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from mindmemos_skill.config import SkillApplicationConfig, SkillDatabaseConfig, SkillLocalConfig
from pydantic import BaseModel, SecretStr, ValidationError

from ..errors import ConfigNotFoundError, ConfigValidationError
from .models import HttpConnectionConfig, SDKConfigV1
from .portal import (
    CompiledSDKPortalConfigV2,
    SDKConfigCompilerV2,
    SDKConfigMigrationPlanV1ToV2,
    SDKMemoryPortalConfigV2,
    SDKPortalConfigV2,
    SDKProfileConfigV2,
    SDKSkillPortalConfigV2,
    SDKSkillRemoteConfigV2,
)

DEFAULT_CONFIG_DIR = Path.home() / ".mindmemos"
CONFIG_FILE_NAME = "settings.json"
PORTAL_CONFIG_FILE_NAME = "config.yaml"
PORTAL_V1_BACKUP_FILE_NAME = "settings.json.v1.bak"
CONFIG_DIR_ENV = "MINDMEMOS_CONFIG_DIR"


class ConfigManager:
    """Own the portal v2 lifecycle and the read-only v1 migration boundary."""

    def __init__(self, config_dir: str | os.PathLike[str] | None = None) -> None:
        """Handle init."""
        if config_dir is None:
            env_dir = os.environ.get(CONFIG_DIR_ENV)
            config_dir = Path(env_dir) if env_dir else DEFAULT_CONFIG_DIR
        self.config_dir = Path(config_dir).expanduser()
        self.config_path = self.config_dir / CONFIG_FILE_NAME
        self.portal_path = self.config_dir / PORTAL_CONFIG_FILE_NAME
        self.portal_v1_backup_path = self.config_dir / PORTAL_V1_BACKUP_FILE_NAME

    def legacy_exists(self) -> bool:
        """Return whether a v1 migration source exists."""

        return self.config_path.is_file()

    def portal_exists(self) -> bool:
        """Return whether the SDK portal v2 YAML exists."""

        return self.portal_path.is_file()

    def ensure_portal(self) -> bool:
        """Create portal v2 from legacy config when it is the only config present.

        Returns ``True`` only when this call applied the migration. Existing v2
        configuration is authoritative and is never overwritten.
        """

        if self.portal_exists() or not self.legacy_exists():
            return False
        self.migrate_portal(dry_run=False)
        return True

    def load_portal(self, *, auto_migrate: bool = True) -> SDKPortalConfigV2:
        """Load portal v2, transparently migrating a lone legacy config by default."""

        if auto_migrate:
            self.ensure_portal()

        if not self.portal_exists():
            raise ConfigNotFoundError(f"No SDK portal config at {self.portal_path}.")
        try:
            raw = yaml.safe_load(self.portal_path.read_text(encoding="utf-8"))
            return SDKPortalConfigV2.model_validate(raw)
        except (yaml.YAMLError, ValidationError) as exc:
            raise ConfigValidationError(f"Invalid SDK portal config at {self.portal_path}: {exc}") from exc

    def compile_portal(self, *, compiler: SDKConfigCompilerV2 | None = None) -> CompiledSDKPortalConfigV2:
        """Load and compile the active portal profile and Skill subtree."""

        return (compiler or SDKConfigCompilerV2()).compile(self.load_portal())

    def load_or_default_portal(self) -> SDKPortalConfigV2:
        """Load the active portal or return an unwritten default portal."""

        if self.portal_exists() or self.legacy_exists():
            return self.load_portal()
        return self.default_portal()

    def default_portal(self) -> SDKPortalConfigV2:
        """Build the default v2 portal without reading or writing disk."""

        return self.convert_legacy_config(SDKConfigV1())

    def convert_legacy_config(self, config: SDKConfigV1) -> SDKPortalConfigV2:
        """Convert an in-memory v1 config to portal v2 without filesystem writes."""

        return self._portal_from_legacy(SDKConfigV1.model_validate(config))

    def save_portal(self, config: SDKPortalConfigV2) -> None:
        """Atomically persist one validated portal v2 YAML."""

        validated = SDKPortalConfigV2.model_validate(config)
        SDKConfigCompilerV2().compile(validated)
        payload = yaml.safe_dump(
            _serialize_portal_value(validated),
            allow_unicode=True,
            sort_keys=False,
        )
        self.config_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.config_dir, prefix=".config-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.portal_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def plan_portal_migration(self) -> SDKConfigMigrationPlanV1ToV2:
        """Compile a deterministic v1-to-v2 plan without modifying any file."""

        legacy = self.load_legacy()
        portal = self.convert_legacy_config(legacy)
        SDKConfigCompilerV2().compile(portal)
        target_exists = self.portal_exists()
        changes_required = not target_exists or self.load_portal(auto_migrate=False) != portal
        return SDKConfigMigrationPlanV1ToV2(
            source_path=self.config_path,
            target_path=self.portal_path,
            backup_path=self.portal_v1_backup_path,
            portal=portal,
            ignored_local_skill_entries=len(legacy.skills),
            target_exists=target_exists,
            changes_required=changes_required,
        )

    def migrate_portal(self, *, dry_run: bool = True) -> SDKConfigMigrationPlanV1ToV2:
        """Write portal v2 atomically after validation, preserving v1 unchanged.

        This migration intentionally does not inspect, import, modify, or delete
        any local Skill state.
        """

        plan = self.plan_portal_migration()
        if dry_run or not plan.changes_required:
            return plan
        if plan.target_exists:
            raise ConfigValidationError(f"SDK portal config already exists with different content: {self.portal_path}")
        source_bytes = self.config_path.read_bytes()
        if self.portal_v1_backup_path.exists():
            if self.portal_v1_backup_path.read_bytes() != source_bytes:
                raise ConfigValidationError(
                    f"SDK v1 backup already exists with different content: {self.portal_v1_backup_path}"
                )
        else:
            _atomic_write_bytes(self.portal_v1_backup_path, source_bytes)
        self.save_portal(plan.portal)
        self.compile_portal()
        return plan

    def _portal_from_legacy(self, legacy: SDKConfigV1) -> SDKPortalConfigV2:
        """Project only configuration intent; legacy Skill state is ignored."""

        if legacy.connections:
            connections = dict(legacy.connections)
            memory_connection = legacy.clients.memory.connection
            skill_connection = legacy.clients.skills.connection
            default_connection = memory_connection
        else:
            default_connection = "mindmemos_main"
            memory_connection = default_connection
            skill_connection = default_connection
            connections = {
                default_connection: HttpConnectionConfig(
                    base_url=legacy.base_url,
                    api_key=legacy.auth.api_key,
                    timeout_seconds=legacy.network.timeout_seconds,
                    max_retries=legacy.network.max_retries,
                )
            }
        skill_root = self.config_dir / "skill"
        return SDKPortalConfigV2(
            active_profile="default",
            profiles={
                "default": SDKProfileConfigV2(
                    default_connection=default_connection,
                    connections=connections,
                    identity=legacy.defaults,
                    memory=SDKMemoryPortalConfigV2(
                        connection=memory_connection,
                        defaults=legacy.memory,
                    ),
                    skill=SDKSkillPortalConfigV2(
                        remote=SDKSkillRemoteConfigV2(connection=skill_connection),
                        application=SkillApplicationConfig(
                            local=SkillLocalConfig(
                                root_dir=skill_root,
                                database=SkillDatabaseConfig(path=skill_root / "state.db"),
                                artifacts_dir=skill_root / "artifacts",
                            )
                        ),
                    ),
                )
            },
        )

    def load_legacy(self) -> SDKConfigV1:
        """Load and validate the v1 migration source without changing it."""

        if not self.legacy_exists():
            raise ConfigNotFoundError(f"No SDK config at {self.config_path}. Run `mindmemos auth` to create it.")
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
            return SDKConfigV1.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigValidationError(f"Invalid SDK config at {self.config_path}: {exc}") from exc

    def reset(self) -> bool:
        """Delete active portal config and a legacy source that could remigrate."""

        removed = False
        for path in (self.portal_path, self.config_path):
            if path.is_file():
                path.unlink()
                removed = True
        return removed

    def update_auth(
        self,
        *,
        base_url: str,
        api_key: str,
        user_id: str | None = None,
    ) -> SDKPortalConfigV2:
        """Update the active profile's default connection and identity in portal v2."""

        config = self.load_or_default_portal()
        profile = config.profiles[config.active_profile]
        connection_name = profile.default_connection
        connection = profile.connections[connection_name]
        profile.connections[connection_name] = connection.model_copy(
            update={"base_url": base_url, "api_key": api_key}
        )
        profile.identity = profile.identity.model_copy(
            update={
                "user_id": user_id,
                "app_id": None,
                "agent_id": None,
                "session_id": None,
            }
        )
        SDKConfigCompilerV2().compile(config)
        self.save_portal(config)
        return config


def mask_secret(secret: str | None, *, visible: int = 4) -> str:
    """Mask a secret while preserving a short suffix."""
    if not secret:
        return "(not set)"
    if len(secret) <= visible:
        return "*" * len(secret)
    return "*" * (len(secret) - visible) + secret[-visible:]


def _serialize_portal_value(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, BaseModel):
        return {name: _serialize_portal_value(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, dict):
        return {key: _serialize_portal_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_portal_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
