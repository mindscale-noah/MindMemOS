"""Typed runtime settings from ``configs/*.yaml`` and ``.env``.

``configs/default.yaml`` holds every non-secret default; an overlay file
selected via :func:`use_config_file` is deep-merged on top. ``.env`` holds
credentials only, read by upper-case name (``LLM_API_KEY``). An exported
environment variable outranks both; for a config key its name is the dotted
path upper-cased with dots as underscores (``PARALLELISM_EXTRACT``).
Required readers raise on an absent key; ``setting_optional_*`` return None.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

_ENV_LOADED = False
_ENV_LOCK = threading.Lock()

_CONFIG_LOCK = threading.RLock()
_CONFIG_PATH: Path | None = None
_CONFIG_CACHE: dict[str, Any] | None = None

DEFAULT_CONFIG_RELATIVE = Path("configs") / "default.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_path() -> Path:
    return _repo_root() / ".env"


def default_config_path() -> Path:
    return _repo_root() / DEFAULT_CONFIG_RELATIVE


def use_config_file(path: str | Path | None) -> Path:
    """Bind the overlay file (``None`` = default alone); rebinding clears the cache."""
    global _CONFIG_PATH, _CONFIG_CACHE
    resolved = Path(path).expanduser().resolve() if path else default_config_path()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing config file: {resolved}")
    with _CONFIG_LOCK:
        _CONFIG_PATH = resolved
        _CONFIG_CACHE = None
    return resolved


def active_config_path() -> Path:
    """Return the bound config file, defaulting on first use."""
    with _CONFIG_LOCK:
        if _CONFIG_PATH is not None:
            return _CONFIG_PATH
    return use_config_file(None)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML file as a mapping; an empty file reads as ``{}``."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a top-level mapping")
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``overlay`` merged over ``base``, recursively."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config() -> dict[str, Any]:
    global _CONFIG_CACHE
    with _CONFIG_LOCK:
        if _CONFIG_CACHE is not None:
            return _CONFIG_CACHE
    path = active_config_path()
    default_path = default_config_path()
    merged = _load_yaml_mapping(default_path)
    if path != default_path:
        merged = _deep_merge(merged, _load_yaml_mapping(path))
    with _CONFIG_LOCK:
        _CONFIG_CACHE = merged
        return merged


_SECRET_HINTS = ("key", "token", "secret", "password", "credential", "credentials", "authorization")


def _is_secret_key(name: str) -> bool:
    """A key whose value may be a credential: all-upper-case marker or a
    credential-hinting name. The config YAML holds no secrets by convention,
    but a snapshot is meant to be safe to hash and persist."""
    lowered = name.lower()
    return name == name.upper() or any(lowered == hint or lowered.endswith(f"_{hint}") for hint in _SECRET_HINTS)


def _redact_secrets(node: Any) -> Any:
    """Deep-copy ``node``, dropping any key whose value is credential-shaped."""
    if isinstance(node, dict):
        return {key: _redact_secrets(value) for key, value in node.items() if not _is_secret_key(str(key))}
    if isinstance(node, list):
        return [_redact_secrets(item) for item in node]
    return node


def _effective_snapshot(node: Any, prefix: str = "") -> Any:
    """Snapshot settings through the same resolver used by runtime readers."""
    if isinstance(node, dict):
        return {
            key: _effective_snapshot(value, f"{prefix}.{key}" if prefix else str(key))
            for key, value in node.items()
            if not _is_secret_key(str(key))
        }
    if isinstance(node, list):
        # List settings deliberately read YAML only (setting_optional_list).
        return _redact_secrets(node)
    return _resolve(prefix)


def resolved_config_snapshot() -> dict[str, Any]:
    """Return the deep-merged, secret-free vendor config in effect.

    ``_config()`` resolves the deep-merged YAML (defaults plus the active
    overlay). Scalars use the runtime resolver's raw string representation,
    including environment overrides and empty overrides that mask defaults.
    List settings retain YAML values, matching ``setting_optional_list``.
    Credential keys are omitted before looking up environment values.
    """
    return _effective_snapshot(_config())


def _env_name(dotted: str) -> str:
    return dotted.replace(".", "_").upper()


def _config_lookup(dotted: str) -> Any:
    """Walk a dotted path, returning ``None`` when any segment is absent."""
    node: Any = _config()
    for segment in dotted.split("."):
        if not isinstance(node, dict) or segment not in node:
            return None
        node = node[segment]
    return node


def load_project_env_once() -> None:
    """Load repository-level ``.env`` without overriding exported variables.

    When the .env file is absent this function returns gracefully rather than
    raising; the wrapper injects credentials via the process environment so
    a missing marker file is not an error.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    with _ENV_LOCK:
        if _ENV_LOADED:
            return

        path = _env_path()
        if not path.exists():
            # .env missing is no longer an error: the host wrapper maps
            # credentials from the process environment, so vendored code can
            # still resolve LLM_API_KEY/LLM_BASE_URL/etc from os.getenv().
            _ENV_LOADED = True
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
        _ENV_LOADED = True


def _is_credential(name: str) -> bool:
    """Upper case marks a credential; anything else is a config key."""
    return name == name.upper()


def _resolve(name: str) -> str | None:
    """Resolve one setting to its raw string, or ``None`` when unset."""
    load_project_env_once()
    if _is_credential(name):
        raw = os.getenv(name)
        return None if raw is None or raw == "" else raw
    override = os.getenv(_env_name(name))
    # An exported empty string means "unset this key" and masks the yaml value.
    if override is not None:
        return override if override != "" else None
    value = _config_lookup(name)
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _source_hint(name: str) -> str:
    if _is_credential(name):
        return f"{_env_path()}"
    return f"{active_config_path()} (or ${_env_name(name)})"


def setting_raw(name: str) -> str:
    """Return a required non-empty setting value."""
    raw = _resolve(name)
    if raw is None:
        raise KeyError(f"required setting {name!r} is missing from {_source_hint(name)}")
    return raw


def setting_int(name: str) -> int:
    raw = setting_raw(name)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid int")


def setting_float(name: str) -> float:
    raw = setting_raw(name)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid float")


def setting_optional_str(name: str) -> str | None:
    """Return the setting's value if set and non-empty, else None."""
    raw = _resolve(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def setting_flag(name: str) -> bool:
    """Return True when the setting is ``1``/``true``/``yes``/``on`` (case-insensitive)."""
    raw = setting_optional_str(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def setting_optional_int(name: str) -> int | None:
    """Return the setting as int if set and non-empty, else None."""
    raw = setting_optional_str(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid int")


def setting_optional_float(name: str) -> float | None:
    """Return the setting as float if set and non-empty, else None."""
    raw = setting_optional_str(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid float")


def setting_optional_list(name: str) -> tuple[str, ...] | None:
    """Return the setting as a tuple of stripped strings, or ``None`` when unset.

    ``_resolve`` cannot carry lists, so this reads the YAML directly (no env
    override). Every element must be a non-empty string; otherwise ValueError.
    """
    value = _config_lookup(name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings, got {type(value).__name__}")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string, got {item!r}")
    return tuple(item.strip() for item in value)


__all__ = [
    "DEFAULT_CONFIG_RELATIVE",
    "active_config_path",
    "default_config_path",
    "load_project_env_once",
    "resolved_config_snapshot",
    "use_config_file",
    "setting_raw",
    "setting_int",
    "setting_float",
    "setting_optional_str",
    "setting_optional_int",
    "setting_optional_float",
    "setting_flag",
]
