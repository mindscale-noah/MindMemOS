"""Run identity and API-key file helpers."""

from __future__ import annotations

import os
import secrets
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCOPES = ("memory:read", "memory:write")


@dataclass(frozen=True)
class RunIdentity:
    """One benchmark run identity bound to a fixed memory algorithm."""

    benchmark: str
    run_id: str
    key_id: str
    api_key: str
    project_id: str
    memory_algorithm: str
    profile: str | None = None
    project_override_config: dict[str, Any] | None = None

    def to_api_key_entry(self) -> dict[str, Any]:
        """Render this identity as one ``api_keys`` YAML entry."""
        entry = {
            "key_id": self.key_id,
            "api_key": self.api_key,
            "project_id": self.project_id,
            "memory_algorithm": self.memory_algorithm,
            "enabled": True,
            "scopes": list(DEFAULT_SCOPES),
        }
        if self.project_override_config:
            entry["project_override_config"] = self.project_override_config
        return entry




def new_identity(
    benchmark: str,
    memory_algorithm: str,
    *,
    now: datetime | None = None,
    profile: str | None = None,
    project_override_config: dict[str, Any] | None = None,
) -> RunIdentity:
    """Create a unique run identity."""
    timestamp = now or datetime.now(UTC)
    suffix = f"{timestamp:%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"
    return RunIdentity(
        benchmark=benchmark,
        run_id=suffix,
        key_id=f"key_{benchmark}_{memory_algorithm}_{suffix}",
        api_key=f"dev-api-key-{benchmark}-{memory_algorithm}-{suffix}".replace("_", "-"),
        project_id=f"proj_{benchmark}_{memory_algorithm}_{suffix}",
        memory_algorithm=memory_algorithm,
        profile=profile,
        project_override_config=project_override_config,
    )


def load_reused_identity(
    path: str | Path,
    benchmark: str,
    memory_algorithm: str,
    *,
    profile: str | None = None,
    project_override_config: dict[str, Any] | None = None,
) -> RunIdentity:
    """Rebuild a prior run identity from an existing ``api_keys`` YAML file.

    Used by ``--reuse-identity`` to skip the add stage and search against the
    memories a previous run already ingested. Matches the entry whose ``key_id``
    was minted for this ``benchmark`` and ``memory_algorithm``.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"cannot reuse identity: api-key file not found at {target}; run an add pass first"
        )
    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    entries = payload.get("api_keys") or []
    key_prefix = f"key_{benchmark}_{memory_algorithm}_"
    project_prefix = f"proj_{benchmark}_{memory_algorithm}_"
    matches = [entry for entry in entries if str(entry.get("key_id", "")).startswith(key_prefix)]
    if not matches:
        raise ValueError(
            f"cannot reuse identity: no api_keys entry for benchmark '{benchmark}' "
            f"and algorithm '{memory_algorithm}' in {target}; run an add pass first"
        )
    if len(matches) > 1:
        raise ValueError(
            f"cannot reuse identity: multiple api_keys entries for benchmark '{benchmark}' "
            f"and algorithm '{memory_algorithm}' in {target}; expected exactly one"
        )
    entry = matches[0]
    required = ("key_id", "api_key", "project_id")
    missing = [name for name in required if not entry.get(name)]
    if missing:
        raise ValueError(f"cannot reuse identity: api_keys entry is missing {', '.join(missing)}")
    project_id = str(entry["project_id"])
    if not project_id.startswith(project_prefix):
        raise ValueError(
            f"cannot reuse identity: project_id does not match expected prefix {project_prefix!r}"
        )
    run_id = project_id[len(project_prefix):]
    return RunIdentity(
        benchmark=benchmark,
        run_id=run_id,
        key_id=str(entry["key_id"]),
        api_key=str(entry["api_key"]),
        project_id=project_id,
        memory_algorithm=str(entry.get("memory_algorithm") or memory_algorithm),
        profile=profile,
        project_override_config=entry.get("project_override_config") or project_override_config,
    )


def write_api_keys(path: str | Path, identities: list[RunIdentity]) -> None:
    """Write generated identities to an ``api_keys`` YAML file atomically."""
    target = Path(path)
    entries = [identity.to_api_key_entry() for identity in identities]
    _atomic_write_yaml(target, {"api_keys": entries})


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
