"""Read, write, and update local SDK configuration.

``ConfigManager`` is the single entry point for SDK identity and credentials. CLI
commands and SDK clients both go through it instead of touching the JSON file
directly. Writes are atomic (temp file + ``os.replace``) so an interrupted process
never leaves a corrupted ``settings.v2.json``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from ..errors import ConfigNotFoundError, ConfigValidationError
from .models import CONFIG_SCHEMA_VERSION, DEFAULT_BASE_URL, MemoryServiceConfig, SDKConfig
from .paths import get_mindmemos_home

CONFIG_FILE_NAME = "settings.v2.json"
LEGACY_CONFIG_FILE_NAME = "settings.json"
RESET_CONFIG = "null"
V1_DEFAULT_BASE_URL = "https://api.mindmemos.example.com"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_atomic(path: Path, payload: str, *, prefix: str) -> None:
    """Replace one JSON file only after its complete payload reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


class ConfigManager:
    """Owns SDK connection settings under the selected home."""

    def __init__(self, config_dir: str | os.PathLike[str] | None = None) -> None:
        """Select a stable home for this manager.

        Args:
            config_dir: Explicit home overriding environment settings. If omitted,
                use MINDMEMOS_HOME, MINDMEMOS_CONFIG_DIR, the saved local
                default, then ~/.mindmemos.
        """
        if config_dir is None:
            config_dir = get_mindmemos_home()
        self.config_dir = Path(config_dir).expanduser()
        self.config_path = self.config_dir / CONFIG_FILE_NAME
        self.legacy_config_path = self.config_dir / LEGACY_CONFIG_FILE_NAME

    @property
    def source_path(self) -> Path:
        """Return the effective settings file, including any read-only legacy fallback."""
        return self.config_path if self.config_path.is_file() else self.legacy_config_path

    def exists(self) -> bool:
        """Return whether the configuration file exists."""
        path = self.source_path
        try:
            return path.read_bytes().strip() != RESET_CONFIG.encode() or path != self.config_path
        except FileNotFoundError:
            return False

    def load(self) -> SDKConfig:
        """Load settings, interpreting legacy files without replacing them.

        Returns:
            Validated v2 settings bound to this manager's home.

        Raises:
            ConfigNotFoundError: The settings file does not exist.
            ConfigValidationError: The stored configuration cannot be upgraded or validated.
        """
        path = self.source_path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigNotFoundError(f"No SDK config at {path}. Run `mindmemos auth` to create it.") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigValidationError(f"Invalid SDK config at {path}") from exc
        if raw is None and path == self.config_path:
            raise ConfigNotFoundError(f"SDK config at {path} was reset. Run `mindmemos auth` to configure it.")
        try:
            if not isinstance(raw, dict):
                raise ValueError("settings must be a JSON object")
            version = raw.get("version", 1)
            if version == 1:
                config = self._migrate_v1(raw)
            elif version == CONFIG_SCHEMA_VERSION:
                raw.pop("storage", None)
                config = SDKConfig.model_validate(raw)
            else:
                raise ValueError(f"unsupported SDK config version: {version}")
            return config
        except ValueError as exc:
            raise ConfigValidationError(f"Invalid SDK config at {path}: {exc}") from exc

    def _migrate_v1(self, raw: dict) -> SDKConfig:
        """Interpret v1 connection settings as a v2 document in memory."""
        memory_url = raw.get("base_url", V1_DEFAULT_BASE_URL)
        skill_settings = raw.get("skill_service", {})
        if not isinstance(skill_settings, dict):
            raise ValueError("v1 skill_service must be an object")
        skill_settings = skill_settings.copy()
        skill_settings.pop("mode", None)
        skill_url = skill_settings.get("base_url")
        official_memory = isinstance(memory_url, str) and memory_url.strip().rstrip("/") in {
            DEFAULT_BASE_URL,
            V1_DEFAULT_BASE_URL,
        }
        separate_skill = isinstance(skill_url, str) and bool(skill_url.strip())
        service_type = "official" if official_memory and not separate_skill else "custom"
        migrated_memory_url = (
            None if service_type == "official" else (DEFAULT_BASE_URL if official_memory else memory_url)
        )
        return SDKConfig.model_validate(
            {
                "version": CONFIG_SCHEMA_VERSION,
                "service_type": service_type,
                "memory_service": {"base_url": migrated_memory_url, "auth": raw.get("auth", {})},
                "skill_service": skill_settings,
                "defaults": raw.get("defaults", {}),
                "network": raw.get("network", {}),
                "metadata": raw.get("metadata", {}),
            }
        )

    def load_or_default(self) -> SDKConfig:
        """Load existing configuration or return defaults without writing."""
        try:
            return self.load()
        except ConfigNotFoundError:
            return SDKConfig()

    def save(self, config: SDKConfig) -> None:
        """Atomically write validated v2 settings and refresh timestamps.

        Args:
            config: Settings to persist in this manager's home.

        Raises:
            ConfigValidationError: The settings violate the v2 schema.
        """
        now = _utc_now_iso()
        if config.metadata.created_at is None:
            config.metadata.created_at = now
        config.metadata.updated_at = now
        try:
            validated = SDKConfig.model_validate(config.model_dump())
        except ValidationError as exc:
            raise ConfigValidationError(f"Invalid SDK config: {exc}") from exc
        payload = json.dumps(validated.model_dump(exclude_none=True), indent=2, ensure_ascii=False)
        _write_atomic(self.config_path, payload, prefix=".settings-")

    def reset(self) -> bool:
        """Clear v2 settings without reviving or deleting a legacy credential."""
        if not self.exists():
            return False
        if self.legacy_config_path.is_file():
            _write_atomic(self.config_path, RESET_CONFIG, prefix=".settings-")
        else:
            self.config_path.unlink()
        return True

    def update_auth(
        self,
        *,
        base_url: str,
        api_key: str,
        user_id: str | None = None,
    ) -> SDKConfig:
        """Update the Memory connection while preserving independent Skill settings.

        Args:
            base_url: Official gateway URL or custom Memory API root.
            api_key: Bearer key for Memory requests.
            user_id: Optional default business user identity.

        Returns:
            Updated v2 settings.

        Raises:
            ConfigValidationError: The Memory URL or resulting settings are invalid.
        """
        config = self.load_or_default()
        try:
            normalized_url = MemoryServiceConfig.model_validate({"base_url": base_url}).base_url
        except ValidationError as exc:
            raise ConfigValidationError(f"Invalid Memory base_url: {exc}") from exc
        if normalized_url is None:
            raise ConfigValidationError("Memory base_url cannot be empty")
        official_url = normalized_url == DEFAULT_BASE_URL
        config.service_type = "official" if official_url and config.skill_service.base_url is None else "custom"
        config.memory_service.base_url = None if config.service_type == "official" else normalized_url
        config.memory_service.auth.api_key = api_key
        config.defaults.user_id = user_id
        self.save(config)
        return config


def mask_secret(secret: str | None, *, visible: int = 4) -> str:
    """Mask a secret while preserving a short suffix."""
    if not secret:
        return "(not set)"
    if len(secret) <= visible:
        return "*" * len(secret)
    return "*" * (len(secret) - visible) + secret[-visible:]
