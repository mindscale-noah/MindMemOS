from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from mindmemos_skill.infra.database import (
    DatabaseConfig,
    DatabaseRegistry,
    DatabaseRequirements,
    DatabaseScope,
    FieldSpec,
    FieldType,
    IndexSpec,
    Page,
    Predicate,
    Record,
    RecordQuery,
    Sort,
    TableRegistry,
    TableSpec,
    bootstrap_database,
    create_database,
)
from mindmemos_skill.infra.database.database_impl import SqliteBackend, create_sqlite_backend
from mindmemos_skill.infra.vector_store import (
    BackendConfig,
    VectorFieldSpec,
    create_vector_store,
)
from mindmemos_skill.infra.vector_store import (
    TableRegistry as VectorTableRegistry,
)
from mindmemos_skill.infra.vector_store import (
    TableSpec as VectorTableSpec,
)
from mindmemos_skill.infra.vector_store.vector_store_impl import PgVectorBackend


def _storage_tables() -> TableRegistry:
    tables = TableRegistry(
        (
            TableSpec(
                name="runtime_logs",
                primary_key="log_id",
                fields=(
                    FieldSpec(name="log_id", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="level", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="message", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="context", field_type=FieldType.JSON, nullable=False, default={}),
                    FieldSpec(name="created_at", field_type=FieldType.DATETIME, nullable=False),
                ),
                indexes=(IndexSpec(name="runtime_logs_level_idx", fields=("level",)),),
            ),
            TableSpec(
                name="raw_trajectories",
                primary_key="trajectory_id",
                fields=(
                    FieldSpec(name="trajectory_id", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="task_id", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="steps", field_type=FieldType.JSON, nullable=False),
                ),
                indexes=(IndexSpec(name="raw_trajectories_task_idx", fields=("task_id",)),),
            ),
            TableSpec(
                name="skill_info",
                primary_key="skill_id",
                fields=(
                    FieldSpec(name="skill_id", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="name", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="version", field_type=FieldType.INTEGER, nullable=False),
                    FieldSpec(name="metadata", field_type=FieldType.JSON, nullable=False, default={}),
                ),
                indexes=(IndexSpec(name="skill_info_name_version_uq", fields=("name", "version"), unique=True),),
            ),
        )
    )
    tables.freeze()
    return tables


def test_database_and_vector_store_have_independent_builtin_registries(tmp_path) -> None:
    tables = _storage_tables()

    sqlite = create_database(
        DatabaseConfig(provider="sqlite", options={"path": str(tmp_path / "skill.db")}),
        tables,
    )
    vectors = VectorTableRegistry(
        (
            VectorTableSpec(
                name="skill_vectors",
                primary_key="skill_id",
                vectors=(VectorFieldSpec(name="embedding", dimensions=3),),
            ),
        )
    )
    vectors.freeze()
    postgres = create_vector_store(BackendConfig(provider="pgvector", options={"dsn": "postgresql://unused"}), vectors)

    assert isinstance(sqlite, SqliteBackend)
    assert sqlite.capabilities.metadata_filtering is True
    assert isinstance(postgres, PgVectorBackend)
    assert postgres.capabilities.dense_vector is True
    assert postgres.capabilities.hybrid_search is True

    with pytest.raises(ValueError, match="unsupported database backend 'pgvector'"):
        create_database(DatabaseConfig(provider="pgvector"), tables)
    with pytest.raises(ValueError, match="unsupported vector backend 'sqlite'"):
        create_vector_store(BackendConfig(provider="sqlite"), vectors)


def test_database_registry_supports_custom_backend_selection(tmp_path) -> None:
    tables = _storage_tables()
    registry = DatabaseRegistry()
    registry.register("custom-sqlite", create_sqlite_backend)

    database = create_database(
        DatabaseConfig(
            provider="custom-sqlite",
            options={"path": str(tmp_path / "custom.db")},
            required=DatabaseRequirements(atomic_batch_write=True),
        ),
        tables,
        registry=registry,
    )

    assert isinstance(database, SqliteBackend)
    assert registry.providers == ("custom-sqlite",)


def test_database_infra_does_not_depend_on_business_or_vector_store() -> None:
    package = (
        Path(__file__).parents[2]
        / "src"
        / "mindmemos_skill"
        / "mindmemos_skill"
        / "infra"
        / "database"
    )
    forbidden = {"mindmemos_skill.persistence", "mindmemos_skill.infra.vector_store"}

    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert not any(name == prefix or name.startswith(f"{prefix}.") for name in imported for prefix in forbidden)


@pytest.mark.asyncio
async def test_sqlite_stores_logs_trajectories_and_skill_info_with_scope_and_restart(tmp_path) -> None:
    path = tmp_path / "nested" / "skill.db"
    tables = _storage_tables()
    config = DatabaseConfig(provider="sqlite", options={"path": str(path)})
    backend = await bootstrap_database(config, tables)
    project_a = DatabaseScope(project_id="project-a", run_id="run-1")
    project_b = DatabaseScope(project_id="project-b", run_id="run-1")
    created_at = datetime(2026, 8, 3, 8, 30, tzinfo=UTC)

    await backend.upsert_records(
        "runtime_logs",
        (
            Record(
                table="runtime_logs",
                record_id="log-1",
                scope=project_a,
                payload={
                    "log_id": "log-1",
                    "level": "INFO",
                    "message": "started",
                    "context": {"component": "evolver", "attempt": 1},
                    "created_at": created_at,
                },
            ),
            Record(
                table="runtime_logs",
                record_id="log-2",
                scope=project_a,
                payload={
                    "log_id": "log-2",
                    "level": "ERROR",
                    "message": "model failed",
                    "context": {"component": "optimizer", "attempt": 2},
                    "created_at": created_at,
                },
            ),
            Record(
                table="runtime_logs",
                record_id="log-1",
                scope=project_b,
                payload={
                    "log_id": "log-1",
                    "level": "INFO",
                    "message": "other project",
                    "context": {"component": "runner"},
                    "created_at": created_at,
                },
            ),
        ),
    )
    await backend.upsert_records(
        "raw_trajectories",
        (
            Record(
                table="raw_trajectories",
                record_id="trajectory-1",
                scope=project_a,
                payload={
                    "trajectory_id": "trajectory-1",
                    "task_id": "task-1",
                    "steps": [{"tool": "search", "ok": True}, {"tool": "write", "ok": True}],
                },
            ),
        ),
    )
    await backend.upsert_records(
        "skill_info",
        (
            Record(
                table="skill_info",
                record_id="skill-1-v1",
                scope=DatabaseScope(project_id="project-a"),
                payload={
                    "skill_id": "skill-1-v1",
                    "name": "research-brief",
                    "version": 1,
                    "metadata": {"status": "published", "tags": ["research", "writing"]},
                },
            ),
        ),
    )

    error_logs, cursor = await backend.query_records(
        "runtime_logs",
        RecordQuery(
            scope=DatabaseScope(project_id="project-a"),
            filters=Predicate(field="context.component", op="eq", value="optimizer"),
            sort=(Sort(field="created_at", direction="desc"),),
            page=Page(limit=1),
        ),
    )
    assert [record.record_id for record in error_logs] == ["log-2"]
    assert cursor is None
    assert error_logs[0].payload["created_at"] == created_at

    first_page, cursor = await backend.query_records(
        "runtime_logs",
        RecordQuery(scope=DatabaseScope(project_id="project-a"), page=Page(limit=1)),
    )
    second_page, next_cursor = await backend.query_records(
        "runtime_logs",
        RecordQuery(scope=DatabaseScope(project_id="project-a"), page=Page(limit=1, cursor=cursor)),
    )
    assert [record.record_id for record in first_page + second_page] == ["log-1", "log-2"]
    assert cursor is not None
    assert next_cursor is None

    await backend.patch_record("runtime_logs", project_a, "log-1", {"level": "WARNING"})
    scoped = await backend.get_records("runtime_logs", project_a, ("log-1",))
    other_scope = await backend.get_records("runtime_logs", project_b, ("log-1",))
    assert scoped[0].payload["level"] == "WARNING"
    assert other_scope[0].payload["level"] == "INFO"
    await backend.close()

    reopened = await bootstrap_database(config, tables)
    trajectories = await reopened.get_records("raw_trajectories", project_a, ("trajectory-1",))
    skills = await reopened.get_records("skill_info", DatabaseScope(project_id="project-a"), ("skill-1-v1",))
    assert trajectories[0].payload["steps"][1] == {"tool": "write", "ok": True}
    assert skills[0].payload["metadata"]["tags"] == ["research", "writing"]
    await reopened.delete_records("runtime_logs", project_a, ("log-1",))
    assert await reopened.get_records("runtime_logs", project_a, ("log-1",)) == []
    assert (await reopened.get_records("runtime_logs", project_b, ("log-1",)))[0].payload["message"] == "other project"
    await reopened.close()


@pytest.mark.asyncio
async def test_sqlite_detects_registered_schema_drift(tmp_path) -> None:
    path = tmp_path / "skill.db"
    original = _storage_tables()
    backend = await bootstrap_database(DatabaseConfig(provider="sqlite", options={"path": str(path)}), original)
    await backend.close()

    changed_specs = list(original.specs)
    changed_specs[0] = TableSpec(
        name="runtime_logs",
        primary_key="log_id",
        fields=(*changed_specs[0].fields, FieldSpec(name="source", field_type=FieldType.TEXT)),
        indexes=changed_specs[0].indexes,
    )
    changed = TableRegistry(changed_specs)
    changed.freeze()

    with pytest.raises(RuntimeError, match="schema drift"):
        await bootstrap_database(DatabaseConfig(provider="sqlite", options={"path": str(path)}), changed)


def _typed_tables() -> TableRegistry:
    tables = TableRegistry(
        (
            TableSpec(
                name="typed_records",
                primary_key="record_id",
                fields=(
                    FieldSpec(name="record_id", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="enabled", field_type=FieldType.BOOLEAN, nullable=False),
                    FieldSpec(name="external_id", field_type=FieldType.UUID, nullable=False),
                    FieldSpec(name="happened_at", field_type=FieldType.DATETIME, nullable=False),
                    FieldSpec(name="tags", field_type=FieldType.TEXT_ARRAY, nullable=False),
                    FieldSpec(name="related_ids", field_type=FieldType.UUID_ARRAY, nullable=False),
                    FieldSpec(name="metadata", field_type=FieldType.JSON, nullable=False),
                ),
                scope_scoped=False,
            ),
        )
    )
    tables.freeze()
    return tables


@pytest.mark.asyncio
async def test_sqlite_strictly_validates_declared_field_types() -> None:
    database = await bootstrap_database(DatabaseConfig(), _typed_tables())
    external_id = uuid4()
    related_id = uuid4()
    valid = Record(
        table="typed_records",
        record_id="valid",
        payload={
            "record_id": "valid",
            "enabled": False,
            "external_id": str(external_id),
            "happened_at": "2026-08-04T10:30:00+08:00",
            "tags": ["reviewed"],
            "related_ids": [str(related_id)],
            "metadata": {"attempt": 1},
        },
    )

    await database.upsert_records("typed_records", (valid,))
    stored = (await database.get_records("typed_records", DatabaseScope(), ("valid",)))[0]

    assert stored.payload["enabled"] is False
    assert stored.payload["external_id"] == external_id
    assert stored.payload["happened_at"] == datetime(2026, 8, 4, 2, 30, tzinfo=UTC)
    assert stored.payload["related_ids"] == [related_id]

    with pytest.raises(TypeError, match="requires a boolean value"):
        await database.upsert_records(
            "typed_records",
            (Record(table="typed_records", record_id="bad-bool", payload={**valid.payload, "record_id": "bad-bool", "enabled": "false"}),),
        )
    with pytest.raises(ValueError, match="requires a valid UUID value"):
        await database.upsert_records(
            "typed_records",
            (
                Record(
                    table="typed_records",
                    record_id="bad-uuid",
                    payload={**valid.payload, "record_id": "bad-uuid", "external_id": "not-a-uuid"},
                ),
            ),
        )
    with pytest.raises(ValueError, match="requires a timezone-aware datetime"):
        await database.upsert_records(
            "typed_records",
            (
                Record(
                    table="typed_records",
                    record_id="bad-datetime",
                    payload={**valid.payload, "record_id": "bad-datetime", "happened_at": "2026-08-04T10:30:00"},
                ),
            ),
        )
    await database.close()


@pytest.mark.asyncio
async def test_sqlite_normalizes_datetimes_to_utc_before_sorting_and_filtering() -> None:
    database = await bootstrap_database(DatabaseConfig(), _typed_tables())
    common = {
        "enabled": True,
        "external_id": uuid4(),
        "tags": [],
        "related_ids": [],
        "metadata": {},
    }
    earlier = datetime(2025, 12, 31, 23, 30, tzinfo=UTC)
    earlier_with_offset = earlier.astimezone(timezone(timedelta(hours=1)))
    later = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    await database.upsert_records(
        "typed_records",
        (
            Record(
                table="typed_records",
                record_id="earlier",
                payload={**common, "record_id": "earlier", "happened_at": earlier_with_offset},
            ),
            Record(
                table="typed_records",
                record_id="later",
                payload={**common, "record_id": "later", "happened_at": later},
            ),
        ),
    )

    ordered, _ = await database.query_records(
        "typed_records",
        RecordQuery(sort=(Sort(field="happened_at"),)),
    )
    filtered, _ = await database.query_records(
        "typed_records",
        RecordQuery(filters=Predicate(field="happened_at", op="gte", value=later)),
    )

    assert [record.record_id for record in ordered] == ["earlier", "later"]
    assert [record.payload["happened_at"] for record in ordered] == [earlier, later]
    assert [record.record_id for record in filtered] == ["later"]
    await database.close()


@pytest.mark.asyncio
async def test_sqlite_json_contains_supports_nested_objects_and_arrays() -> None:
    database = await bootstrap_database(DatabaseConfig(), _typed_tables())
    external_id = UUID("df3c2a85-efce-489e-b4db-f3b88a49da4c")
    await database.upsert_records(
        "typed_records",
        (
            Record(
                table="typed_records",
                record_id="matching",
                payload={
                    "record_id": "matching",
                    "enabled": True,
                    "external_id": external_id,
                    "happened_at": datetime(2026, 8, 4, tzinfo=UTC),
                    "tags": ["research"],
                    "related_ids": [],
                    "metadata": {"business": {"priority": 2}, "tags": ["research", "writing"]},
                },
            ),
            Record(
                table="typed_records",
                record_id="other",
                payload={
                    "record_id": "other",
                    "enabled": True,
                    "external_id": external_id,
                    "happened_at": datetime(2026, 8, 4, tzinfo=UTC),
                    "tags": ["operations"],
                    "related_ids": [],
                    "metadata": {"business": {"priority": 1}, "tags": ["operations"]},
                },
            ),
        ),
    )

    nested, _ = await database.query_records(
        "typed_records",
        RecordQuery(filters=Predicate(field="metadata", op="contains", value={"business": {"priority": 2}})),
    )
    array, _ = await database.query_records(
        "typed_records",
        RecordQuery(filters=Predicate(field="metadata", op="contains", value={"tags": ["research"]})),
    )

    assert [record.record_id for record in nested] == ["matching"]
    assert [record.record_id for record in array] == ["matching"]
    await database.close()
