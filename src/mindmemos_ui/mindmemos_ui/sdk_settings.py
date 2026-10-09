"""Edit the complete v2 SDK settings document in one JSON file."""

import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from filelock import FileLock
from mindmemos_sdk.config import ConfigManager, SDKConfig
from mindmemos_sdk.errors import ConfigValidationError
from pydantic import ValidationError

from .typing import SDKSettingsUpdate


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON fields instead of silently replacing them."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _parse(content: str) -> SDKConfig:
    """Validate one complete v2 settings document."""
    try:
        raw = json.loads(content, object_pairs_hook=_unique_pairs)
        if not isinstance(raw, dict):
            raise ValueError("Expected a JSON object")
        return SDKConfig.model_validate(raw)
    except ValidationError as exc:
        fields = sorted({".".join(map(str, error["loc"])) or "services" for error in exc.errors()})
        raise HTTPException(422, f"Invalid v2 settings fields: {', '.join(fields)}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(422, "Invalid v2 settings JSON") from exc


def _document(config: SDKConfig, path: Path) -> dict:
    """Return the current settings and a revision of the stored bytes."""
    raw = path.read_bytes() if path.exists() else None
    values = config.model_dump(mode="json", exclude_none=True)
    return {
        "path": str(path),
        "exists": raw is not None,
        "revision": hashlib.sha256(raw).hexdigest() if raw is not None else None,
        "values": values,
        "content": json.dumps(values, ensure_ascii=False, indent=2) + "\n",
    }


def sdk_settings_router(runtime_home: Path) -> APIRouter:
    """Expose one v2 settings.json for the selected SDK home.

    Args:
        runtime_home: Default SDK configuration directory for this UI process.

    Returns:
        Router for reading, validating, and saving SDK settings.
    """
    router = APIRouter(prefix="/api/client-configurations/sdk")

    def manager_for(home: str | None) -> ConfigManager:
        """Resolve one local SDK home and reject linked settings files."""
        directory = Path(home).expanduser() if home else runtime_home
        manager = ConfigManager(directory.resolve())
        if manager.config_path.is_symlink():
            raise HTTPException(400, "Symbolic link configuration files are not supported")
        return manager

    def read(manager: ConfigManager) -> dict:
        """Read settings after any automatic v1-to-v2 upgrade."""
        try:
            return _document(manager.load_or_default(), manager.config_path)
        except ConfigValidationError as exc:
            raise HTTPException(422, "Invalid SDK settings file") from exc

    @router.get("")
    def get(home: str | None = None):
        return read(manager_for(home))

    @router.post("/normalize")
    def normalize(body: SDKSettingsUpdate):
        values = _parse(body.content).model_dump(mode="json", exclude_none=True)
        return {"values": values, "content": json.dumps(values, ensure_ascii=False, indent=2) + "\n"}

    @router.put("")
    def save(body: SDKSettingsUpdate, home: str | None = None):
        config = _parse(body.content)
        manager = manager_for(home)
        manager.config_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(manager.config_dir / ".config.lock")):
            current = read(manager)
            if current["revision"] != body.revision:
                raise HTTPException(409, "Configuration changed on disk; reload before saving")
            manager.save(config)
            return read(manager)

    return router
