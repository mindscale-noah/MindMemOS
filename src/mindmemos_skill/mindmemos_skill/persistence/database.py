"""Composition helpers for the canonical local Skill state database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..infra.database import DatabaseConfig, ScopedDatabase, bootstrap_database
from .migrations import CURRENT_SCHEMA_VERSION, SKILL_SCHEMA_MIGRATIONS, SKILL_SCHEMA_NAMESPACE
from .tables import build_persistence_tables

DEFAULT_SKILL_DATABASE_PATH = Path("~/.mindmemos/skill/state.db").expanduser()


@dataclass(frozen=True, slots=True)
class SkillDatabaseStatus:
    path: Path
    current_version: int
    target_version: int
    pending_versions: tuple[int, ...]
    database_is_newer: bool


def default_skill_database_config(path: str | Path | None = None) -> DatabaseConfig:
    """Build the SQLite config without opening or mutating the filesystem."""

    resolved = DEFAULT_SKILL_DATABASE_PATH if path is None else Path(path).expanduser()
    return DatabaseConfig(
        provider="sqlite",
        options={"path": str(resolved)},
    )


async def bootstrap_skill_database(path: str | Path | None = None) -> ScopedDatabase:
    """Open the canonical Skill database and apply its explicit schema ledger."""

    resolved = DEFAULT_SKILL_DATABASE_PATH if path is None else Path(path).expanduser()
    status = get_skill_database_status(resolved)
    if status.database_is_newer:
        raise RuntimeError(
            f"Skill database schema version {status.current_version} is newer than supported version "
            f"{status.target_version}"
        )
    if status.current_version > 0 and status.pending_versions:
        backup_skill_database(resolved)
    return await bootstrap_database(default_skill_database_config(resolved), build_persistence_tables())


def get_skill_database_status(path: str | Path | None = None) -> SkillDatabaseStatus:
    """Inspect the local schema version without mutating or opening the runtime backend."""

    resolved = DEFAULT_SKILL_DATABASE_PATH if path is None else Path(path).expanduser()
    applied_versions: set[int] = set()
    if resolved.exists() and resolved.stat().st_size:
        connection = sqlite3.connect(resolved)
        try:
            migration_table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='__mindmemos_migrations'"
            ).fetchone()
            if migration_table_exists:
                applied_versions = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT version FROM __mindmemos_migrations WHERE namespace = ?",
                        (SKILL_SCHEMA_NAMESPACE,),
                    ).fetchall()
                }
        finally:
            connection.close()
    current_version = max(applied_versions, default=0)
    registered_versions = tuple(migration.version for migration in SKILL_SCHEMA_MIGRATIONS)
    return SkillDatabaseStatus(
        path=resolved,
        current_version=current_version,
        target_version=CURRENT_SCHEMA_VERSION,
        pending_versions=tuple(version for version in registered_versions if version not in applied_versions),
        database_is_newer=current_version > CURRENT_SCHEMA_VERSION,
    )


def backup_skill_database(path: str | Path | None = None) -> Path:
    """Create a transactionally consistent backup of an existing SQLite database."""

    resolved = DEFAULT_SKILL_DATABASE_PATH if path is None else Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    status = get_skill_database_status(resolved)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = resolved.with_name(
        f"{resolved.name}.backup-v{status.current_version}-to-v{status.target_version}-{timestamp}"
    )
    source = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    destination = sqlite3.connect(backup_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return backup_path


__all__ = [
    "DEFAULT_SKILL_DATABASE_PATH",
    "SkillDatabaseStatus",
    "backup_skill_database",
    "bootstrap_skill_database",
    "default_skill_database_config",
    "get_skill_database_status",
]
