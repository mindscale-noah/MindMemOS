"""Read, write, and update local SDK configuration.

``ConfigManager`` is the single entry point for SDK identity and credentials. CLI
commands and SDK clients both go through it instead of touching the JSON file
directly. Writes are atomic (temp file + ``os.replace``) so an interrupted process
never leaves a corrupted ``settings.json``.
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
from .models import SDKConfig, StorageConfig
from .paths import get_mindmemos_home

CONFIG_FILE_NAME = "settings.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ConfigManager:
    """Owns settings and default storage under the selected SDK home."""

    def __init__(self, config_dir: str | os.PathLike[str] | None = None) -> None:
        """Select a stable home for this manager.

        Args:
            config_dir: Explicit home overriding environment settings. If omitted,
                use MINDMEMOS_HOME, MINDMEMOS_CONFIG_DIR, then ~/.mindmemos.
        """
        if config_dir is None:
            config_dir = get_mindmemos_home()
        self.config_dir = Path(config_dir).expanduser()
        self.config_path = self.config_dir / CONFIG_FILE_NAME

    def exists(self) -> bool:
        """Return whether the configuration file exists."""
        return self.config_path.is_file()

    def load(self) -> SDKConfig:
        """Load and validate configuration from disk."""
        if not self.exists():
            raise ConfigNotFoundError(f"No SDK config at {self.config_path}. Run `mindmemos auth` to create it.")
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
            config = SDKConfig.model_validate(raw)
            if "skill_cache_dir" not in config.storage.model_fields_set:
                config.storage.skill_cache_dir = str(self.config_dir / "skills/cache")
            if "skill_backup_dir" not in config.storage.model_fields_set:
                config.storage.skill_backup_dir = str(self.config_dir / "skills/backups")
            return config
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigValidationError(f"Invalid SDK config at {self.config_path}: {exc}") from exc

    def load_or_default(self) -> SDKConfig:
        """Load existing configuration or return defaults without writing."""
        if self.exists():
            return self.load()
        return SDKConfig(
            storage=StorageConfig(
                skill_cache_dir=str(self.config_dir / "skills/cache"),
                skill_backup_dir=str(self.config_dir / "skills/backups"),
            )
        )

    def save(self, config: SDKConfig) -> None:
        """Atomically write configuration and refresh metadata timestamps."""
        now = _utc_now_iso()
        if config.metadata.created_at is None:
            config.metadata.created_at = now
        config.metadata.updated_at = now

        self.config_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config.model_dump(), indent=2, ensure_ascii=False)

        fd, tmp_path = tempfile.mkstemp(dir=self.config_dir, prefix=".settings-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.config_path)
        except BaseException:
            # Clean up the temp file on any failure so partial writes never linger.
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def reset(self) -> bool:
        """Clear in-process Kafka registry state for tests."""
        if self.exists():
            self.config_path.unlink()
            return True
        return False

    def update_auth(
        self,
        *,
        base_url: str,
        api_key: str,
        user_id: str | None = None,
    ) -> SDKConfig:
        """Merge authentication settings into existing configuration and save it."""
        config = self.load_or_default()
        config.base_url = base_url
        config.auth.api_key = api_key
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
