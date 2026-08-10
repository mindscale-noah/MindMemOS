from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest
from mindmemos_skill.contracts import SkillBundle
from mindmemos_skill.infra.database import DatabaseConfig, DatabaseScope, bootstrap_database
from mindmemos_skill.management import LocalSkillManager
from mindmemos_skill.persistence import (
    SKILL_TABLE,
    TRAJECTORY_TABLE,
    SkillRecord,
    TrajectoryRecord,
    build_persistence_tables,
    from_database_record,
    to_database_record,
)


@pytest.mark.asyncio
async def test_persistence_catalog_round_trips_business_models_through_sqlite(tmp_path) -> None:
    database = await bootstrap_database(
        DatabaseConfig(provider="sqlite", options={"path": str(tmp_path / "state.db")}),
        build_persistence_tables(),
    )
    bundle = SkillBundle.from_files({"SKILL.md": "# Research brief"})
    skill = SkillRecord(
        skill_id="skill-1",
        version_id="version-1",
        name="research-brief",
        bundle=bundle.canonical_json(),
        content_hash=bundle.content_hash,
        local_snapshot_hash=bundle.content_hash,
        version_label="1.0.0",
    )

    await database.upsert_records(SKILL_TABLE, (to_database_record(skill),))
    stored = await database.get_records(SKILL_TABLE, DatabaseScope(), ("version-1",))

    assert from_database_record(stored[0], SkillRecord) == skill
    await database.close()


@pytest.mark.asyncio
async def test_persistence_catalog_enforces_one_attempt_number_per_rollout(tmp_path) -> None:
    database = await bootstrap_database(
        DatabaseConfig(provider="sqlite", options={"path": str(tmp_path / "state.db")}),
        build_persistence_tables(),
    )
    first = TrajectoryRecord(
        trajectory_id="trajectory-1",
        trajectory_hash="hash-1",
        task_id="task-1",
        rollout_id="rollout-1",
        attempt_no=0,
        task_instruction="Solve the task",
    )
    duplicate_attempt = first.model_copy(update={"trajectory_id": "trajectory-2"})

    await database.upsert_records(TRAJECTORY_TABLE, (to_database_record(first),))
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        await database.upsert_records(TRAJECTORY_TABLE, (to_database_record(duplicate_attempt),))

    await database.close()


@pytest.mark.asyncio
async def test_legacy_pointer_schema_is_restartably_backfilled_without_heads(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    now = datetime(2026, 8, 7, tzinfo=UTC).isoformat()
    pending = [
        {
            "operation_id": "push-v1",
            "operation_type": "push_version",
            "skill_id": "skill-1",
            "version_id": "v1",
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        },
        {
            "operation_id": "promote-v1",
            "operation_type": "promote",
            "skill_id": "skill-1",
            "version_id": "v1",
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        },
    ]
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE skill_versions (
          _scope_key TEXT, _scope TEXT, _record_id TEXT PRIMARY KEY,
          skill_id TEXT, version_id TEXT, cloud_skill_id TEXT, parent_version_ids TEXT,
          name TEXT, description TEXT, alias TEXT, blob TEXT, resources TEXT,
          content_hash TEXT, status TEXT, version_label TEXT, commit_message TEXT,
          metadata TEXT, created_at TEXT, origin TEXT
        );
        CREATE TABLE skill_family_state (
          _scope_key TEXT, _scope TEXT, _record_id TEXT PRIMARY KEY,
          skill_id TEXT, effective_version_id TEXT, published_head_id TEXT,
          cloud_revision INTEGER, last_sync_at TEXT, pending_operations TEXT,
          created_at TEXT, updated_at TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO skill_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "",
            "{}",
            "v1",
            "skill-1",
            "v1",
            None,
            "[]",
            "demo",
            None,
            "demo",
            json.dumps({"SKILL.md": "demo\n"}),
            "{}",
            "legacy-hash",
            "draft",
            "1.0.0",
            None,
            json.dumps({"snapshot": {"local_snapshot_hash": "local-hash", "files": []}}),
            now,
            "local",
        ),
    )
    connection.execute(
        "INSERT INTO skill_family_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("", "{}", "skill-1", "skill-1", "v1", "v1", 4, now, json.dumps(pending), now, now),
    )
    connection.commit()
    connection.close()

    manager = await LocalSkillManager.open(path)
    detail = await manager.get_skill("demo")
    operations = await manager.repository.list_operations(skill_id="skill-1")
    assert detail.latest_version.version_id == "v1"
    assert detail.latest_version.content_hash == SkillBundle.from_files({"SKILL.md": "demo\n"}).content_hash
    assert not {"effective_version_id", "published_head_id", "pending_operations"}.intersection(
        detail.sync_state.model_dump()
    )
    assert [item.operation_id for item in operations] == ["push-v1"]
    await manager.close()

    reopened = await LocalSkillManager.open(path)
    assert (await reopened.get_skill("demo")).latest_version.version_id == "v1"
    await reopened.close()

    connection = sqlite3.connect(path)
    assert connection.execute(
        "SELECT status FROM __mindmemos_skill_v2_backfill WHERE migration='legacy-v1'"
    ).fetchone() == ("completed",)
    assert connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='__legacy_skill_versions_v1'"
    ).fetchone() == (1,)
    connection.close()
