"""Composition helpers for the canonical local Skill state database."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from ..contracts import SkillBundle, canonical_request_hash
from ..infra.database import DatabaseConfig, ScopedDatabase, bootstrap_database
from .models import SkillRecord, SkillRemoteOperationRecord, SkillSyncStateRecord, TrajectoryRecord
from .records import to_database_record
from .tables import build_persistence_tables

DEFAULT_SKILL_DATABASE_PATH = Path("~/.mindmemos/skill/state.db").expanduser()


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
    _prepare_legacy_v1_migration(resolved)
    database = await bootstrap_database(default_skill_database_config(resolved), build_persistence_tables())
    await _backfill_legacy_v1(database, resolved)
    return database


def _prepare_legacy_v1_migration(path: Path) -> None:
    """Atomically preserve legacy physical tables before the v2 schema is created."""

    if not path.exists():
        return
    connection = sqlite3.connect(path)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info('skill_versions')").fetchall()
        }
        if "blob" not in columns or "bundle" in columns:
            return
        connection.execute("BEGIN IMMEDIATE")
        for source, target in (
            ("skill_versions", "__legacy_skill_versions_v1"),
            ("skill_family_state", "__legacy_skill_family_state_v1"),
            ("trajectories", "__legacy_trajectories_v1"),
        ):
            if _sqlite_table_exists(connection, source) and not _sqlite_table_exists(connection, target):
                connection.execute(f'ALTER TABLE "{source}" RENAME TO "{target}"')
        for index_name in (
            "skill_versions_label_uq",
            "skill_versions_hash_idx",
            "skill_family_state_effective_idx",
            "skill_family_state_published_idx",
            "trajectories_rollout_attempt_uq",
            "trajectories_task_idx",
        ):
            connection.execute(f'DROP INDEX IF EXISTS "{index_name}"')
        connection.execute(
            "CREATE TABLE IF NOT EXISTS __mindmemos_skill_v2_backfill "
            "(migration TEXT PRIMARY KEY, status TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO __mindmemos_skill_v2_backfill (migration, status, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(migration) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
            ("legacy-v1", "prepared", datetime.now().astimezone().isoformat()),
        )
        if _sqlite_table_exists(connection, "__mindmemos_schema"):
            connection.execute(
                "DELETE FROM __mindmemos_schema WHERE table_name IN (?, ?, ?)",
                ("skill_versions", "skill_family_state", "trajectories"),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


async def _backfill_legacy_v1(database: ScopedDatabase, path: Path) -> None:
    if not path.exists():
        return
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        if not _sqlite_table_exists(connection, "__mindmemos_skill_v2_backfill"):
            return
        marker = connection.execute(
            "SELECT status FROM __mindmemos_skill_v2_backfill WHERE migration = 'legacy-v1'"
        ).fetchone()
        if marker is None or marker["status"] == "completed":
            return
        versions = _read_legacy_rows(connection, "__legacy_skill_versions_v1")
        families = _read_legacy_rows(connection, "__legacy_skill_family_state_v1")
        trajectories = _read_legacy_rows(connection, "__legacy_trajectories_v1")
    finally:
        connection.close()

    version_records: list[SkillRecord] = []
    for row in versions:
        blob = _json_object(row["blob"])
        bundle = SkillBundle.from_files(blob)
        metadata = _json_object(row["metadata"])
        snapshot = metadata.pop("snapshot", None)
        local_snapshot_hash = (
            snapshot.get("local_snapshot_hash") if isinstance(snapshot, dict) else None
        ) or bundle.content_hash
        version_records.append(
            SkillRecord(
                skill_id=row["skill_id"],
                version_id=row["version_id"],
                cloud_skill_id=row["cloud_skill_id"],
                parent_version_ids=_json_list(row["parent_version_ids"]),
                name=row["name"],
                description=row["description"],
                alias=row["alias"],
                bundle=bundle.canonical_json(),
                resources=row["resources"] or "{}",
                content_hash=bundle.content_hash,
                local_snapshot_hash=str(local_snapshot_hash),
                status=row["status"],
                version_label=row["version_label"],
                commit_message=row["commit_message"],
                metadata=metadata,
                local_metadata={"snapshot": snapshot} if isinstance(snapshot, dict) else {},
                origin=row["origin"],
                created_at=row["created_at"],
                updated_at=row["created_at"],
            )
        )
    if version_records:
        await database.upsert_records("skill_versions", [to_database_record(item) for item in version_records])

    sync_states: list[SkillSyncStateRecord] = []
    operations: list[SkillRemoteOperationRecord] = []
    for row in families:
        sync_states.append(
            SkillSyncStateRecord(
                skill_id=row["skill_id"],
                last_version_sync_at=row["last_sync_at"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )
        for payload in _json_list(row["pending_operations"]):
            if not isinstance(payload, dict) or payload.get("operation_type") != "push_version":
                continue
            facts = {
                key: payload.get(key)
                for key in ("skill_id", "cloud_skill_id", "version_id", "trajectory_id")
            }
            operations.append(
                SkillRemoteOperationRecord(
                    operation_id=str(payload["operation_id"]),
                    operation_type="push_version",
                    skill_id=payload.get("skill_id") or row["skill_id"],
                    cloud_skill_id=payload.get("cloud_skill_id"),
                    version_id=payload.get("version_id"),
                    request_hash=str(payload.get("request_hash") or canonical_request_hash(facts)),
                    status=str(payload.get("status") or "pending"),
                    attempt_count=int(payload.get("attempt_count") or 0),
                    lease_expires_at=payload.get("lease_expires_at"),
                    next_retry_at=payload.get("next_retry_at"),
                    last_error_code=payload.get("last_error_code"),
                    remote_result=payload.get("remote_result"),
                    created_at=payload.get("created_at") or row["created_at"],
                    updated_at=payload.get("updated_at") or row["updated_at"],
                )
            )
    if sync_states:
        await database.upsert_records("skill_sync_state", [to_database_record(item) for item in sync_states])
    if operations:
        await database.upsert_records(
            "skill_remote_operations",
            [to_database_record(item) for item in operations],
        )

    trajectory_records: list[TrajectoryRecord] = []
    for row in trajectories:
        payload = {
            key: _restore_legacy_value(row[key])
            for key in row.keys()
            if key not in {"_scope_key", "_scope", "_record_id"}
        }
        created_at = payload.get("finished_at") or payload["started_at"]
        payload.update(
            {
                "trajectory_hash": hashlib.sha256(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
                ).hexdigest(),
                "metadata_revision": 0,
                "metadata_updated_at": None,
                "source": "skill_runtime",
                "source_add_record_id": None,
                "created_at": created_at,
                "received_at": None,
            }
        )
        trajectory_records.append(TrajectoryRecord.model_validate(payload))
    if trajectory_records:
        await database.upsert_records("trajectories", [to_database_record(item) for item in trajectory_records])

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE __mindmemos_skill_v2_backfill SET status=?, updated_at=? WHERE migration=?",
            ("completed", datetime.now().astimezone().isoformat(), "legacy-v1"),
        )
        connection.commit()
    finally:
        connection.close()


def _sqlite_table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def _read_legacy_rows(connection: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    if not _sqlite_table_exists(connection, table):
        return []
    return list(connection.execute(f'SELECT * FROM "{table}"').fetchall())


def _json_object(value) -> dict:
    parsed = json.loads(value) if isinstance(value, str) else value
    return dict(parsed or {})


def _json_list(value) -> list:
    parsed = json.loads(value) if isinstance(value, str) else value
    return list(parsed or [])


def _restore_legacy_value(value):
    if not isinstance(value, str):
        return value
    if value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


__all__ = [
    "DEFAULT_SKILL_DATABASE_PATH",
    "bootstrap_skill_database",
    "default_skill_database_config",
]
