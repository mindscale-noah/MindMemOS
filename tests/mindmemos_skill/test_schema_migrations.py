from __future__ import annotations

import sqlite3

import pytest
from mindmemos_skill.infra.database import (
    DatabaseConfig,
    DatabaseScope,
    FieldSpec,
    FieldType,
    Record,
    SchemaMigration,
    TableRegistry,
    TableSpec,
    bootstrap_database,
)


def _registry(*, version: int, invalid_statement: bool = False, omit_statement: bool = False) -> TableRegistry:
    fields = [FieldSpec(name="record_id", field_type=FieldType.TEXT, nullable=False)]
    migrations = [
        SchemaMigration(
            namespace="migration-test",
            version=1,
            name="initial_schema",
            tables=("events",),
        )
    ]
    if version >= 2:
        fields.append(FieldSpec(name="source", field_type=FieldType.TEXT))
        statements = () if omit_statement else ('ALTER TABLE "events" ADD COLUMN "source" TEXT',)
        if invalid_statement:
            statements = (*statements, "INVALID MIGRATION SQL")
        migrations.append(
            SchemaMigration(
                namespace="migration-test",
                version=2,
                name="add_source",
                tables=("events",),
                sqlite_statements=statements,
                postgres_statements=('ALTER TABLE "events" ADD COLUMN "source" TEXT',),
            )
        )
    registry = TableRegistry(
        (
            TableSpec(
                name="events",
                primary_key="record_id",
                fields=tuple(fields),
                scope_scoped=False,
            ),
        ),
        migrations=migrations,
    )
    registry.freeze()
    return registry


def _config(path) -> DatabaseConfig:
    return DatabaseConfig(provider="sqlite", options={"path": str(path)})


@pytest.mark.asyncio
async def test_sqlite_applies_pending_forward_migration_and_preserves_rows(tmp_path) -> None:
    path = tmp_path / "state.db"
    v1 = await bootstrap_database(_config(path), _registry(version=1))
    await v1.upsert_records(
        "events",
        (
            Record(
                table="events",
                record_id="event-1",
                payload={"record_id": "event-1"},
            ),
        ),
    )
    await v1.close()
    with sqlite3.connect(path) as connection:
        v1_checksum = connection.execute(
            "SELECT checksum FROM __mindmemos_migrations WHERE namespace='migration-test' AND version=1"
        ).fetchone()[0]

    v2 = await bootstrap_database(_config(path), _registry(version=2))
    stored = await v2.get_records("events", DatabaseScope(), ("event-1",))
    assert stored[0].payload == {"record_id": "event-1", "source": None}
    await v2.close()

    with sqlite3.connect(path) as connection:
        assert [row[1] for row in connection.execute('PRAGMA table_info("events")')] == [
            "_scope_key",
            "_scope",
            "_record_id",
            "record_id",
            "source",
        ]
        assert (
            connection.execute(
                "SELECT checksum FROM __mindmemos_migrations WHERE namespace='migration-test' AND version=1"
            ).fetchone()[0]
            == v1_checksum
        )
        assert connection.execute(
            "SELECT version FROM __mindmemos_migrations WHERE namespace='migration-test' ORDER BY version"
        ).fetchall() == [(1,), (2,)]

    reopened = await bootstrap_database(_config(path), _registry(version=2))
    await reopened.close()


@pytest.mark.asyncio
async def test_fresh_database_uses_latest_catalog_without_replaying_alter_statements(tmp_path) -> None:
    path = tmp_path / "state.db"
    database = await bootstrap_database(_config(path), _registry(version=2))
    await database.close()

    with sqlite3.connect(path) as connection:
        columns = [row[1] for row in connection.execute('PRAGMA table_info("events")')]
        versions = connection.execute(
            "SELECT version FROM __mindmemos_migrations WHERE namespace='migration-test' ORDER BY version"
        ).fetchall()
    assert columns.count("source") == 1
    assert versions == [(1,), (2,)]


@pytest.mark.asyncio
async def test_failed_sqlite_migration_rolls_back_ddl_and_ledger(tmp_path) -> None:
    path = tmp_path / "state.db"
    v1 = await bootstrap_database(_config(path), _registry(version=1))
    await v1.close()

    with pytest.raises(sqlite3.OperationalError):
        await bootstrap_database(_config(path), _registry(version=2, invalid_statement=True))

    with sqlite3.connect(path) as connection:
        columns = [row[1] for row in connection.execute('PRAGMA table_info("events")')]
        versions = connection.execute(
            "SELECT version FROM __mindmemos_migrations WHERE namespace='migration-test' ORDER BY version"
        ).fetchall()
    assert "source" not in columns
    assert versions == [(1,)]


@pytest.mark.asyncio
async def test_migration_cannot_stamp_a_table_that_does_not_match_latest_spec(tmp_path) -> None:
    path = tmp_path / "state.db"
    v1 = await bootstrap_database(_config(path), _registry(version=1))
    await v1.close()

    with pytest.raises(RuntimeError, match="physical schema"):
        await bootstrap_database(_config(path), _registry(version=2, omit_statement=True))

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT version FROM __mindmemos_migrations WHERE namespace='migration-test' ORDER BY version"
        ).fetchall() == [(1,)]


@pytest.mark.asyncio
async def test_database_newer_than_code_is_rejected(tmp_path) -> None:
    path = tmp_path / "state.db"
    v2 = await bootstrap_database(_config(path), _registry(version=2))
    await v2.close()

    with pytest.raises(RuntimeError, match="unknown migrations"):
        await bootstrap_database(_config(path), _registry(version=1))


def test_migration_versions_must_start_at_one_and_be_contiguous() -> None:
    table = TableSpec(
        name="events",
        primary_key="record_id",
        fields=(FieldSpec(name="record_id", field_type=FieldType.TEXT, nullable=False),),
    )
    with pytest.raises(ValueError, match="expected version 1, got 2"):
        TableRegistry(
            (table,),
            migrations=(
                SchemaMigration(
                    namespace="migration-test",
                    version=2,
                    name="skipped_initial",
                    tables=("events",),
                ),
            ),
        )
