"""AC-P3-001 (unit portion) + insert_record_if_absent contract on SQLite."""

from __future__ import annotations

import sqlite3

import pytest
from mindmemos_skill.infra.database import (
    DatabaseConfig,
    DatabaseScope,
    FieldSpec,
    FieldType,
    Record,
    TableRegistry,
    TableSpec,
    bootstrap_database,
)
from mindmemos_skill.persistence.tables import build_persistence_tables


def _sqlite_config(path) -> DatabaseConfig:
    return DatabaseConfig(provider="sqlite", options={"path": str(path)})


@pytest.mark.asyncio
async def test_fresh_bootstrap_includes_skill_manage_tables_and_v0002_ledger(tmp_path) -> None:
    path = tmp_path / "state.db"
    database = await bootstrap_database(_sqlite_config(path), build_persistence_tables())
    await database.close()

    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        versions = connection.execute(
            "SELECT version FROM __mindmemos_migrations WHERE namespace='mindmemos-skill' ORDER BY version"
        ).fetchall()

    assert "skill_manage_graphs" in tables
    assert "skill_manage_operations" in tables
    assert "skill_manage_nodes" in tables
    assert versions == [(1,), (2,)]


@pytest.mark.asyncio
async def test_second_bootstrap_is_noop(tmp_path) -> None:
    path = tmp_path / "state.db"
    first = await bootstrap_database(_sqlite_config(path), build_persistence_tables())
    await first.close()
    second = await bootstrap_database(_sqlite_config(path), build_persistence_tables())
    await second.close()


def _events_registry() -> TableRegistry:
    registry = TableRegistry(
        (
            TableSpec(
                name="events",
                primary_key="record_id",
                fields=(FieldSpec(name="record_id", field_type=FieldType.TEXT, nullable=False),),
                scope_scoped=True,
            ),
        ),
    )
    registry.freeze()
    return registry


@pytest.mark.asyncio
async def test_insert_record_if_absent_semantics_on_sqlite(tmp_path) -> None:
    database = await bootstrap_database(_sqlite_config(tmp_path / "state.db"), _events_registry())
    scope = DatabaseScope(project_id="p1", graph_id="g1")
    record = Record(table="events", record_id="event-1", scope=scope, payload={"record_id": "event-1"})

    first = await database.insert_record_if_absent("events", record)
    second = await database.insert_record_if_absent("events", record)

    assert first is True
    assert second is False
    stored = await database.get_records("events", scope, ("event-1",))
    assert stored[0].payload == {"record_id": "event-1"}
    await database.close()


@pytest.mark.asyncio
async def test_insert_record_if_absent_inside_a_transaction_on_sqlite(tmp_path) -> None:
    """The unit-of-work variant must behave like the autocommit one (it is what a first publish uses)."""

    database = await bootstrap_database(_sqlite_config(tmp_path / "state.db"), _events_registry())
    scope = DatabaseScope(project_id="p1", graph_id="g1")
    record = Record(table="events", record_id="event-1", scope=scope, payload={"record_id": "event-1"})

    async with database.transaction() as transaction:
        first = await transaction.insert_record_if_absent("events", record)
        second = await transaction.insert_record_if_absent("events", record)
        assert (first, second) == (True, False)
    assert await database.insert_record_if_absent("events", record) is False
    stored = await database.get_records("events", scope, ("event-1",))
    assert stored[0].payload == {"record_id": "event-1"}
    await database.close()


@pytest.mark.asyncio
async def test_insert_record_if_absent_isolates_by_scope(tmp_path) -> None:
    database = await bootstrap_database(_sqlite_config(tmp_path / "state.db"), _events_registry())
    scope_a = DatabaseScope(project_id="p1", graph_id="g1")
    scope_b = DatabaseScope(project_id="p1", graph_id="g2")

    first = await database.insert_record_if_absent(
        "events", Record(table="events", record_id="e", scope=scope_a, payload={"record_id": "e"})
    )
    second = await database.insert_record_if_absent(
        "events", Record(table="events", record_id="e", scope=scope_b, payload={"record_id": "e"})
    )

    assert first is True
    assert second is True  # same record_id, different scope -> independent identities
    await database.close()
