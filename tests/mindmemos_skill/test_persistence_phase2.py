from __future__ import annotations

import asyncio
import multiprocessing
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from typing import Any

import pytest
from mindmemos_skill.contracts import SkillBundle
from mindmemos_skill.infra.database import (
    DatabaseConfig,
    DatabaseScope,
    FieldSpec,
    FieldType,
    SchemaMigration,
    TableRegistry,
    TableSpec,
    bootstrap_database,
)
from mindmemos_skill.persistence import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_SKILL_DATABASE_PATH,
    SKILL_REMOTE_OPERATION_TABLE,
    SKILL_SYNC_STATE_TABLE,
    SKILL_TABLE,
    SkillRecord,
    SkillRemoteOperationRecord,
    SkillSyncStateRecord,
    backup_skill_database,
    bootstrap_skill_database,
    default_skill_database_config,
    from_database_record,
    get_skill_database_status,
    to_database_record,
)


def _skill(version_id: str = "version-1", version_label: str = "1.0.0") -> SkillRecord:
    bundle = SkillBundle.from_files({"SKILL.md": "# Research brief"})
    return SkillRecord(
        skill_id="skill-1",
        version_id=version_id,
        name="research-brief",
        bundle=bundle.canonical_json(),
        content_hash=bundle.content_hash,
        local_snapshot_hash=bundle.content_hash,
        version_label=version_label,
    )


def _state(cursor: str = "cursor-1") -> SkillSyncStateRecord:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    return SkillSyncStateRecord(
        skill_id="skill-1",
        trajectory_pull_cursor=cursor,
        created_at=now,
        updated_at=now,
    )


def _operation() -> SkillRemoteOperationRecord:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    return SkillRemoteOperationRecord(
        operation_id="push-skill-1-version-1",
        operation_type="push_version",
        skill_id="skill-1",
        version_id="version-1",
        request_hash="sha256:request-1",
        status="pending",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_persistence_bootstrap_registers_the_public_v1_catalog(tmp_path) -> None:
    path = tmp_path / "state.db"
    migrated = await bootstrap_skill_database(path)
    await migrated.close()

    with sqlite3.connect(path) as connection:
        migrations = connection.execute(
            "SELECT namespace, version, name FROM __mindmemos_migrations ORDER BY version"
        ).fetchall()
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert migrations == [
        ("mindmemos-skill", 1, "initial_schema"),
    ]
    assert {
        "algorithm_logs",
        "llm_calls",
        "skill_remote_operations",
        "skill_sync_state",
        "skill_versions",
        "trajectories",
    } <= tables
    status = get_skill_database_status(path)
    assert status.current_version == CURRENT_SCHEMA_VERSION == 1
    assert status.target_version == 1
    assert status.pending_versions == ()
    assert status.database_is_newer is False


@pytest.mark.asyncio
async def test_skill_database_backup_is_a_consistent_sqlite_copy(tmp_path) -> None:
    path = tmp_path / "state.db"
    database = await bootstrap_skill_database(path)
    await database.upsert_records(SKILL_TABLE, (to_database_record(_skill()),))
    await database.close()

    backup_path = backup_skill_database(path)

    assert backup_path.name.startswith("state.db.backup-v1-to-v1-")
    with sqlite3.connect(backup_path) as connection:
        assert connection.execute("SELECT version_id FROM skill_versions").fetchall() == [("version-1",)]


@pytest.mark.asyncio
async def test_failed_schema_migration_rolls_back_its_catalog_changes(tmp_path) -> None:
    path = tmp_path / "state.db"
    first = TableSpec(
        name="first_table",
        primary_key="record_id",
        fields=(FieldSpec(name="record_id", field_type=FieldType.TEXT, nullable=False),),
        scope_scoped=False,
    )
    omitted = TableSpec(
        name="omitted_table",
        primary_key="record_id",
        fields=(FieldSpec(name="record_id", field_type=FieldType.TEXT, nullable=False),),
        scope_scoped=False,
    )
    invalid_catalog = TableRegistry(
        (first, omitted),
        migrations=(
            SchemaMigration(
                namespace="rollback-test",
                version=1,
                name="incomplete_catalog",
                tables=("first_table",),
            ),
        ),
    )
    invalid_catalog.freeze()

    with pytest.raises(RuntimeError, match="missing an explicit schema migration"):
        await bootstrap_database(DatabaseConfig(options={"path": str(path)}), invalid_catalog)

    with sqlite3.connect(path) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert tables == []


@pytest.mark.asyncio
async def test_transaction_commits_version_sync_state_and_outbox_together(tmp_path) -> None:
    database = await bootstrap_skill_database(tmp_path / "state.db")
    skill = _skill()
    state = _state()
    operation = _operation()

    async with database.transaction() as unit_of_work:
        await unit_of_work.upsert_records(SKILL_TABLE, (to_database_record(skill),))
        await unit_of_work.upsert_records(SKILL_SYNC_STATE_TABLE, (to_database_record(state),))
        await unit_of_work.upsert_records(SKILL_REMOTE_OPERATION_TABLE, (to_database_record(operation),))

    stored_skill = await database.get_records(SKILL_TABLE, DatabaseScope(), (skill.version_id,))
    stored_state = await database.get_records(SKILL_SYNC_STATE_TABLE, DatabaseScope(), (state.skill_id,))
    stored_operation = await database.get_records(
        SKILL_REMOTE_OPERATION_TABLE, DatabaseScope(), (operation.operation_id,)
    )
    assert from_database_record(stored_skill[0], SkillRecord) == skill
    assert from_database_record(stored_state[0], SkillSyncStateRecord) == state
    assert from_database_record(stored_operation[0], SkillRemoteOperationRecord) == operation
    await database.close()


@pytest.mark.asyncio
async def test_transaction_rolls_back_version_sync_state_and_outbox_together(tmp_path) -> None:
    database = await bootstrap_skill_database(tmp_path / "state.db")
    skill = _skill()
    state = _state()
    operation = _operation()

    with pytest.raises(RuntimeError, match="abort transaction"):
        async with database.transaction() as unit_of_work:
            await unit_of_work.upsert_records(SKILL_TABLE, (to_database_record(skill),))
            await unit_of_work.upsert_records(SKILL_SYNC_STATE_TABLE, (to_database_record(state),))
            await unit_of_work.upsert_records(SKILL_REMOTE_OPERATION_TABLE, (to_database_record(operation),))
            raise RuntimeError("abort transaction")

    assert await database.get_records(SKILL_TABLE, DatabaseScope(), (skill.version_id,)) == []
    assert await database.get_records(SKILL_SYNC_STATE_TABLE, DatabaseScope(), (state.skill_id,)) == []
    assert await database.get_records(SKILL_REMOTE_OPERATION_TABLE, DatabaseScope(), (operation.operation_id,)) == []
    await database.close()


@pytest.mark.asyncio
async def test_compare_and_swap_updates_sync_cursor_only_for_expected_value(tmp_path) -> None:
    database = await bootstrap_skill_database(tmp_path / "state.db")
    await database.upsert_records(SKILL_SYNC_STATE_TABLE, (to_database_record(_state()),))

    won = await database.compare_and_swap_record(
        SKILL_SYNC_STATE_TABLE,
        DatabaseScope(),
        "skill-1",
        expected={"trajectory_pull_cursor": "cursor-1"},
        changes={"trajectory_pull_cursor": "cursor-2"},
    )
    stale = await database.compare_and_swap_record(
        SKILL_SYNC_STATE_TABLE,
        DatabaseScope(),
        "skill-1",
        expected={"trajectory_pull_cursor": "cursor-1"},
        changes={"trajectory_pull_cursor": "cursor-3"},
    )

    stored = await database.get_records(SKILL_SYNC_STATE_TABLE, DatabaseScope(), ("skill-1",))
    assert won is True
    assert stale is False
    assert stored[0].payload["trajectory_pull_cursor"] == "cursor-2"
    await database.close()


def _cas_process(
    path: str,
    replacement: str,
    ready: Any,
    start: Any,
    results: Any,
) -> None:
    async def run() -> None:
        database = await bootstrap_skill_database(path)
        ready.put(replacement)
        start.wait(timeout=10)
        try:
            changed = await database.compare_and_swap_record(
                SKILL_SYNC_STATE_TABLE,
                DatabaseScope(),
                "skill-1",
                expected={"trajectory_pull_cursor": "cursor-1"},
                changes={"trajectory_pull_cursor": replacement},
            )
            results.put((replacement, changed, None))
        except BaseException as exc:
            results.put((replacement, False, repr(exc)))
            raise
        finally:
            await database.close()

    asyncio.run(run())


@pytest.mark.asyncio
async def test_compare_and_swap_is_safe_across_processes(tmp_path) -> None:
    path = tmp_path / "state.db"
    database = await bootstrap_skill_database(path)
    await database.upsert_records(SKILL_SYNC_STATE_TABLE, (to_database_record(_state()),))
    await database.close()

    context = multiprocessing.get_context("spawn")
    ready: Queue[str] = context.Queue()
    start = context.Event()
    results: Queue[tuple[str, bool, str | None]] = context.Queue()
    processes = [
        context.Process(target=_cas_process, args=(str(path), replacement, ready, start, results))
        for replacement in ("cursor-2", "cursor-3")
    ]
    for process in processes:
        process.start()
    assert {ready.get(timeout=15), ready.get(timeout=15)} == {"cursor-2", "cursor-3"}
    start.set()
    outcomes = [results.get(timeout=15), results.get(timeout=15)]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    assert sorted(changed for _, changed, error in outcomes if error is None) == [False, True]
    reopened = await bootstrap_skill_database(path)
    stored = await reopened.get_records(SKILL_SYNC_STATE_TABLE, DatabaseScope(), ("skill-1",))
    assert stored[0].payload["trajectory_pull_cursor"] in {"cursor-2", "cursor-3"}
    await reopened.close()


def test_default_skill_database_path_is_canonical_and_side_effect_free() -> None:
    assert DEFAULT_SKILL_DATABASE_PATH == Path.home() / ".mindmemos" / "skill" / "state.db"
    assert default_skill_database_config().options["path"] == str(DEFAULT_SKILL_DATABASE_PATH)
