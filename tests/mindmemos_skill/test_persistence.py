"""Business schema, cloud compatibility, and real SQLite persistence contracts."""

from datetime import UTC, datetime

import pytest
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence import (
    SkillRecord,
    SkillVersionRecord,
    TrajectoryRecord,
    build_persistence_tables,
    create_persistence,
    ensure_collection,
)
from mindmemos_skill.persistence.constants import (
    REMOTE_OPERATIONS_TABLE,
    REMOTE_VERSION_BINDINGS_TABLE,
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_REVISIONS_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
    TRAJECTORIES_TABLE,
)
from mindmemos_skill.persistence.db_services.remote import (
    RemoteOperationRecord,
    RemoteSkillBindingRecord,
    RemoteTrajectoryBindingRecord,
    RemoteVersionBindingRecord,
)
from mindmemos_skill.persistence.migrations import migrate_table_names_to_v1
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable


@pytest.fixture
def store(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'business.db'}"))
    persistence = create_persistence(database)
    ensure_collection(persistence)
    try:
        yield persistence
    finally:
        database.close()


def version(version_id="local-v1", skill_id="local-skill"):
    return SkillVersionRecord(
        project_id="local-project",
        version_id=version_id,
        skill_id=skill_id,
        name="Skill",
        bundle="local representation",
        content_hash="local-content-hash",
        version_label=f"display-{version_id}",
    )


def test_skill_source_roundtrip_preserves_metadata(store):
    from mindmemos_skill.skill import Skill, SkillBundle, SkillResources

    snapshot = (
        Skill(
            bundle=SkillBundle(files={"SKILL.md": "Instructions: "}),
            resources=SkillResources(files={"references/guide.md": b"unchanged"}),
        )
        .snapshot()
        .model_copy(
            update={
                "implementation": "test_persisted_dynamic",
                "config_schema_version": 7,
                "config": {"legacy_option": "preserved"},
            }
        )
    )
    skill = Skill.from_snapshot(snapshot)
    record = version().model_copy(update={"local_metadata": {"note": "keep"}})
    stored = store.skills.add_skill(skill, record)
    assert stored.implementation == "test_persisted_dynamic"
    assert stored.bundle == skill.bundle.serialize()
    assert stored.content_hash == skill.bundle.content_hash
    assert stored.resources == skill.resources.serialize() and stored.local_metadata == record.local_metadata
    assert record.bundle == "local representation"
    restored = store.skills.get_skill(stored.version_id)
    assert type(restored) is Skill
    assert restored.resources == skill.resources
    assert restored.snapshot() == snapshot
    assert store.skills.get(stored.version_id).bundle == skill.bundle.serialize()
    assert store.skills.get_skill("missing") is None


def test_previous_skill_column_migration_preserves_rows(tmp_path):
    import sqlite3
    from pathlib import Path

    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'previous.db'}"))
    persistence = create_persistence(database)
    ensure_collection(persistence)
    original = version()
    persistence.skills.add(original)
    database.close()
    with sqlite3.connect(tmp_path / "previous.db") as connection:
        connection.executescript(
            f"ALTER TABLE {SKILL_VERSIONS_TABLE} RENAME COLUMN implementation TO runtime_type;"
            f"ALTER TABLE {SKILL_VERSIONS_TABLE} RENAME COLUMN config_schema_version TO runtime_schema_version;"
            f"ALTER TABLE {SKILL_VERSIONS_TABLE} RENAME COLUMN config TO runtime_metadata;"
        )
        migration = Path(__file__).resolve().parents[2] / "scripts/skill/migrate_skill_columns.sql"
        # Historical SQL targets the original, unversioned schema.
        connection.execute(f"ALTER TABLE {SKILL_VERSIONS_TABLE} RENAME TO skill_versions")
        connection.executescript(migration.read_text())
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'previous.db'}"))
    try:
        persistence = create_persistence(database)
        migrate_table_names_to_v1(database)
        assert persistence.skills.get(original.version_id) == original
    finally:
        database.close()


def trajectory(trajectory_id="t1", rollout_id="r1"):
    return TrajectoryRecord(
        project_id="local-project",
        trajectory_id=trajectory_id,
        trajectory_hash="trace-hash",
        task_id="task-1",
        rollout_id=rollout_id,
        task_instruction="Solve the task",
        env_ref="livemath",
        status="succeeded",
        trajectory=[{"role": "assistant", "content": "42"}],
        reward_score=1.0,
    )


def binding(project_id="project-a", cloud_skill_id="cloud-skill"):
    return RemoteSkillBindingRecord(
        remote_id="production",
        project_id="local-project",
        cloud_project_id=project_id,
        cloud_skill_id=cloud_skill_id,
        skill_id="local-skill",
    )


def remote_version(project_id="project-a", **changes):
    values = dict(
        remote_id="production",
        project_id="local-project",
        cloud_project_id=project_id,
        version_id="cloud-v1",
        cloud_skill_id="cloud-skill",
        local_version_id="local-v1",
        skill_name="Cloud Skill",
        content_hash="cloud-hash",
        content="exact cloud text",
        parent_version_id="cloud-parent",
        version_label="display-only",
        status="published",
        origin="cloud",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return RemoteVersionBindingRecord(**(values | changes))


def operation(operation_id="op-1", **changes):
    values = dict(
        remote_id="production",
        project_id="local-project",
        cloud_project_id="project-a",
        operation_id=operation_id,
        idempotency_key="request-1",
        operation_type="register_skill",
        local_version_id="local-v1",
        request_hash="request-hash",
        request_payload={"name": "Skill", "content": "text", "parent_version_id": "cloud-parent"},
    )
    return RemoteOperationRecord(**(values | changes))


def test_registration_is_lazy_and_all_columns_are_documented(tmp_path):
    path = tmp_path / "lazy.db"
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    try:
        persistence = create_persistence(database)
        assert not path.exists()
        assert len(database.tables.values()) == 12
        for spec in database.tables.values():
            assert all(column.comment for column in spec.table.columns)
            assert set(spec.insert_model.model_fields) == set(spec.table.c.keys())
            assert str(CreateTable(spec.table).compile(dialect=postgresql.dialect()))
        ensure_collection(persistence)
        ensure_collection(persistence)
        assert path.exists()
    finally:
        database.close()


def test_core_records_round_trip_with_distinct_labels(store):
    first = version()
    second = version("local-v2").model_copy(update={"parent_version_ids": ["local-v1"]})
    store.skills.add(first)
    store.skills.add(second)
    assert store.skills.get("local-v1") == first
    assert len(store.skills.list_versions("local-skill")) == 2
    record = trajectory()
    store.trajectories.add(record)
    assert store.trajectories.get("t1") == record
    with pytest.raises(IntegrityError):
        store.trajectories.add(trajectory("t2"))
    assert store.trajectories.get("t2") is None


def test_multiple_remotes_and_projects_do_not_alias_local_identity(store):
    store.skills.add(version())
    for project in ("project-a", "project-b"):
        store.remote.bind_skill(binding(project))
        store.remote.bind_version(remote_version(project))
    assert store.database.count(REMOTE_VERSION_BINDINGS_TABLE) == 2
    row = store.database.get(
        REMOTE_VERSION_BINDINGS_TABLE,
        {
            "remote_id": "production",
            "project_id": "local-project",
            "cloud_project_id": "project-a",
            "version_id": "cloud-v1",
        },
    )
    assert row["content_hash"] == "cloud-hash"
    assert row["parent_version_id"] == "cloud-parent"
    assert store.skills.get("local-v1").content_hash == "local-content-hash"
    store.database.update(
        REMOTE_VERSION_BINDINGS_TABLE, {"status": "superseded"}, filters={"cloud_project_id": "project-a"}
    )
    assert (
        store.database.select(REMOTE_VERSION_BINDINGS_TABLE, filters={"cloud_project_id": "project-b"})[0]["status"]
        == "published"
    )


def test_remote_labels_and_missing_ancestors_are_compatible(store):
    store.skills.add(version())
    store.skills.add(version("local-v2"))
    store.remote.bind_skill(binding())
    store.remote.bind_version(remote_version())
    store.remote.bind_version(remote_version(version_id="cloud-v2", local_version_id="local-v2"))
    assert store.database.count(REMOTE_VERSION_BINDINGS_TABLE) == 2
    # The cloud parent need not have been downloaded yet.
    assert store.database.count(REMOTE_VERSION_BINDINGS_TABLE, filters={"version_id": "cloud-parent"}) == 0


def test_binding_rejects_wrong_local_family_and_missing_trajectory(store):
    store.skills.add(version())
    store.skills.add(version("unrelated-v", "another-skill"))
    store.remote.bind_skill(binding())
    with pytest.raises(ValueError, match="family"):
        store.remote.bind_version(remote_version(local_version_id="unrelated-v"))
    with pytest.raises(IntegrityError):
        store.remote.bind_trajectory(
            RemoteTrajectoryBindingRecord(
                remote_id="production",
                project_id="local-project",
                cloud_project_id="project-a",
                trajectory_id="missing",
                add_record_id="add-1",
            )
        )
    assert store.database.count(REMOTE_VERSION_BINDINGS_TABLE) == 0


def test_add_record_mapping_and_foreign_key_protection(store):
    store.trajectories.add(trajectory())
    store.remote.bind_trajectory(
        RemoteTrajectoryBindingRecord(
            remote_id="production",
            project_id="local-project",
            cloud_project_id="project-a",
            trajectory_id="t1",
            add_record_id="add-1",
        )
    )
    with pytest.raises(IntegrityError):
        store.database.delete(TRAJECTORIES_TABLE, filters={"trajectory_id": "t1"})
    assert store.trajectories.get("t1") is not None


def test_outbox_deduplication_and_uncertain_result(store):
    store.skills.add(version())
    first = store.remote.enqueue(operation())
    assert store.remote.enqueue(operation("op-2")).operation_id == first.operation_id
    with pytest.raises(ValueError, match="different request"):
        store.remote.enqueue(operation("op-3", request_payload={"content": "changed"}))
    assert store.database.count(REMOTE_OPERATIONS_TABLE) == 1
    changed = store.database.update(
        REMOTE_OPERATIONS_TABLE,
        {"status": "uncertain", "attempt_count": 1},
        filters={
            "remote_id": "production",
            "project_id": "local-project",
            "cloud_project_id": "project-a",
            "operation_id": "op-1",
            "status": "pending",
        },
    )
    assert changed == 1
    assert store.remote.enqueue(operation("op-4")).status == "uncertain"
    assert store.database.update(REMOTE_OPERATIONS_TABLE, {"request_payload": {"changed": True}}, filters={}) == 1
    assert store.database.select(REMOTE_OPERATIONS_TABLE)[0]["request_payload"] == {"changed": True}


def test_transaction_rolls_back_across_services(store):
    with pytest.raises(IntegrityError):
        with store.database.transaction() as database:
            database.insert(SKILL_VERSIONS_TABLE, [version().model_dump()])
            database.insert(TRAJECTORIES_TABLE, [trajectory().model_dump(), trajectory("t2").model_dump()])
    assert store.database.count(SKILL_VERSIONS_TABLE) == 0
    assert store.database.count(TRAJECTORIES_TABLE) == 0


def test_core_only_and_no_global_registry_state(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'core.db'}"))
    try:
        persistence = create_persistence(database, include_remote=False)
        assert persistence.remote is None
        assert {s.name for s in database.tables.values()} == {
            SKILLS_TABLE,
            SKILL_VERSIONS_TABLE,
            TRAJECTORIES_TABLE,
            SKILL_GRAPHS_TABLE,
            SKILL_GRAPH_VERSIONS_TABLE,
            SKILL_GRAPH_NODES_TABLE,
            SKILL_GRAPH_EDGES_TABLE,
            SKILL_GRAPH_REVISIONS_TABLE,
        }
        assert len(build_persistence_tables().values()) == 12
        ensure_collection(persistence)
        persistence.skills.add(version())
    finally:
        database.close()


def test_invalid_core_values_are_rejected():
    with pytest.raises(ValueError):
        SkillVersionRecord(**(version().model_dump() | {"parent_version_ids": ["local-v1"]}))
    with pytest.raises(ValueError):
        TrajectoryRecord(**(trajectory().model_dump() | {"reward_score": float("nan")}))
    with pytest.raises(ValueError):
        RemoteVersionBindingRecord(**(remote_version().model_dump() | {"origin": "local"}))


@pytest.mark.asyncio
async def test_service_factory_assembles_borrowed_data_services(tmp_path):
    from mindmemos_skill import ServiceConfig, create_service

    path = tmp_path / "factory.db"
    service = await create_service(
        ServiceConfig.model_validate(
            {
                "persistence": {
                    "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{path}"}},
                    "config": {"include_remote": False},
                },
            }
        )
    )
    try:
        persistence = service.backends["persistence"]
        with pytest.raises(RuntimeError, match="session is required"):
            persistence.skills.add(version())
        with pytest.raises(ValueError, match="project_id"):
            with persistence.session(Scope(fields={"agent_id": "agent"})):
                pass
        with persistence.session(Scope(fields={"project_id": "local-project", "agent_id": "agent"})) as scoped:
            with pytest.raises(ValueError, match="agent_id"):
                scoped.skills.list_skills("local-project", limit=10, offset=0)
        with persistence.session(Scope(fields={"project_id": "local-project", "user_id": "user"})) as scoped:
            with pytest.raises(ValueError, match="user_id"):
                scoped.skills.list_skills("local-project", limit=10, offset=0)
        with pytest.raises(ValueError, match="unexpected"):
            with persistence.session(Scope(fields={"project_id": "local-project", "dataset_id": "dataset"})):
                pass
        assert path.exists()
        with persistence.session(Scope(fields={"project_id": "local-project"})) as scoped:
            scoped.skills.add(version())
            assert scoped.skills.get("local-v1") is not None
    finally:
        await service.close()


def test_json_arrays_and_nested_metadata_use_infra_validation(store):
    record = trajectory().model_copy(
        update={
            "injected_skills": [{"version_id": "v1", "options": {"nested": [1, True, None]}}],
            "metadata": {"attempts": [1, 2]},
        }
    )
    store.trajectories.add(record)
    assert store.trajectories.get("t1") == record
    with pytest.raises(ValueError):
        store.database.insert(
            TRAJECTORIES_TABLE, [trajectory("bad", "bad-run").model_dump() | {"metadata": {"bad": object()}}]
        )
    assert store.trajectories.get("bad") is None


def central_skill(version_record, *, alias=None):
    return SkillRecord(
        project_id=version_record.project_id,
        skill_id=version_record.skill_id,
        name=version_record.name,
        alias=alias,
        registration_source="/original/import/path",
        registration_message="Initial import",
        current_version_id=version_record.version_id,
    )


def test_central_skill_switch_preserves_history_and_checks_owner(store):
    first, second = version(), version("local-v2")
    record = central_skill(first, alias="example")
    key = {"project_id": record.project_id, "skill_id": record.skill_id}
    with store.database.transaction() as database:
        database.insert(SKILL_VERSIONS_TABLE, [first.model_dump(), second.model_dump()])
        database.insert(SKILLS_TABLE, [record.model_dump()])
    assert SkillRecord.model_validate(store.database.get(SKILLS_TABLE, key)) == record
    assert (
        store.database.update(
            SKILLS_TABLE,
            {"current_version_id": second.version_id},
            filters=key | {"current_version_id": first.version_id},
        )
        == 1
    )
    assert (
        store.database.update(
            SKILLS_TABLE,
            {"current_version_id": first.version_id},
            filters=key | {"current_version_id": first.version_id},
        )
        == 0
    )
    assert store.skills.get(first.version_id) == first
    for invalid in (
        version("other-family", "another-skill"),
        version("other-project").model_copy(update={"project_id": "another-project"}),
    ):
        store.skills.add(invalid)
        with pytest.raises(IntegrityError):
            store.database.update(SKILLS_TABLE, {"current_version_id": invalid.version_id}, filters=key)
    with pytest.raises(ValueError):
        store.database.update(SKILLS_TABLE, {"current_version_id": None}, filters=key)
    with pytest.raises(IntegrityError):
        store.database.delete(SKILL_VERSIONS_TABLE, filters={"version_id": second.version_id})
    assert store.database.update(SKILLS_TABLE, {"registration_source": "changed"}, filters=key) == 1
    assert store.database.get(SKILLS_TABLE, key)["registration_source"] == "changed"
    assert store.database.get(SKILLS_TABLE, key)["current_version_id"] == second.version_id


def test_central_alias_scope_and_registration_rollback(store):
    first = version()
    store.skills.add(first)
    record = central_skill(first, alias="example")
    store.database.insert(SKILLS_TABLE, [record.model_dump()])
    duplicate = version("second-v", "second-skill")
    with pytest.raises(IntegrityError):
        with store.database.transaction() as database:
            database.insert(SKILL_VERSIONS_TABLE, [duplicate.model_dump()])
            database.insert(SKILLS_TABLE, [central_skill(duplicate, alias="example").model_dump()])
    assert store.skills.get(duplicate.version_id) is None
    other = version("third-v").model_copy(update={"project_id": "another-project"})
    store.skills.add(other)
    store.database.insert(SKILLS_TABLE, [central_skill(other, alias="example").model_dump()])
    for index in range(2):
        unaliased = version(f"unaliased-v{index}", f"unaliased-s{index}")
        store.skills.add(unaliased)
        store.database.insert(SKILLS_TABLE, [central_skill(unaliased).model_dump()])
    key = {"project_id": record.project_id, "skill_id": record.skill_id}
    store.database.update(SKILLS_TABLE, {"alias": None}, filters=key)
    assert store.database.get(SKILLS_TABLE, key)["alias"] is None


def test_existing_version_database_accepts_central_table_upgrade(tmp_path):
    from pathlib import Path

    from sqlalchemy import text

    path = tmp_path / "old-local.db"
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    catalog = build_persistence_tables(include_remote=False)
    original = version()
    with database.engine.begin() as connection:
        catalog.get(SKILL_VERSIONS_TABLE).table.create(connection)
        connection.execute(catalog.get(SKILL_VERSIONS_TABLE).table.insert(), original.model_dump())
        connection.execute(text("DROP INDEX skill_versions_owner_version_uq"))
    migration = Path(__file__).resolve().parents[2] / "scripts/skill/migrate_central_skills.sql"
    import sqlite3

    with sqlite3.connect(path) as connection:
        # Preserve the physical name expected by this historical migration.
        connection.execute(f"ALTER TABLE {SKILL_VERSIONS_TABLE} RENAME TO skill_versions")
        connection.executescript(migration.read_text())
        connection.executescript(migration.read_text())
    try:
        persistence = create_persistence(database)
        migrate_table_names_to_v1(database)
        ensure_collection(persistence)
        ensure_collection(persistence)
        assert persistence.skills.get(original.version_id) == original
        database.insert(SKILLS_TABLE, [central_skill(original).model_dump()])
        assert database.count(SKILLS_TABLE) == 1
    finally:
        database.close()


@pytest.mark.parametrize("partial", [False, True])
def test_table_name_migration_preserves_data_and_resumes(store, partial):
    from sqlalchemy import inspect

    original = version()
    store.skills.add(original)
    store.database.insert(SKILLS_TABLE, [central_skill(original).model_dump()])
    specs = list(store.database.tables.values())
    # A partially renamed schema models an interrupted upgrade.
    legacy_specs = specs[::2] if partial else specs
    with store.database.engine.begin() as connection:
        for spec in legacy_specs:
            connection.exec_driver_sql(f"ALTER TABLE {spec.name} RENAME TO {spec.name.removesuffix('_v1')}")
    migrate_table_names_to_v1(store.database)
    migrate_table_names_to_v1(store.database)
    ensure_collection(store)
    assert set(inspect(store.database.engine).get_table_names()) == {spec.name for spec in specs}
    assert store.skills.get(original.version_id) == original
    assert store.database.count(SKILLS_TABLE) == 1
    assert store.database.update(SKILLS_TABLE, {"alias": "restored"}, filters={"skill_id": original.skill_id}) == 1


def test_table_name_migration_rejects_conflicting_tables(store):
    legacy_name = SKILLS_TABLE.removesuffix("_v1")
    with store.database.engine.begin() as connection:
        connection.exec_driver_sql(f"CREATE TABLE {legacy_name} (marker TEXT)")
    with pytest.raises(RuntimeError, match="both .* exist"):
        migrate_table_names_to_v1(store.database)
