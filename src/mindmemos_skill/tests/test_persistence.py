"""Business schema and real SQLite persistence contracts."""

from uuid import NAMESPACE_URL, uuid5

import pytest
from mindmemos_skill.errors import SkillAliasConflictError, SkillConflictError, SkillNotFoundError
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
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    SKILL_TASK_LOGS_TABLE,
    SKILL_TASKS_TABLE,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
    TRAJECTORIES_TABLE,
    TRAJECTORY_CONSUMPTIONS_TABLE,
)
from mindmemos_skill.skill import SkillContent, SkillSnapshot
from sqlalchemy import event
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable


@pytest.fixture
async def store(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'business.db'}"))
    persistence = create_persistence(database)
    (await ensure_collection(persistence))
    try:
        yield persistence
    finally:
        (await database.close())


def version(version_id="local-v1", skill_id="local-skill"):
    return SkillVersionRecord(
        project_id="local-project",
        version_id=version_id,
        skill_id=skill_id,
        **SkillSnapshot.from_content(
            SkillContent(content="local representation", name="Skill", description="Description", standard_format=False)
        ).model_dump(),
        version_label=f"display-{version_id}",
    )


async def test_skill_source_roundtrip_preserves_metadata(store):
    from mindmemos_skill.skill import Skill, SkillContent

    snapshot = (
        Skill(
            SkillContent(
                content="Instructions: ",
                name="Skill",
                description="Description",
                standard_format=False,
                resources={"references/guide.md": b"unchanged"},
            ),
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
    stored = await store.skills.add_skill(skill, record)
    (await store.database.insert(SKILLS_TABLE, [central_skill(stored).model_dump()]))
    assert stored.implementation == "test_persisted_dynamic"
    assert stored.content == skill.content
    assert stored.content_hash == skill.snapshot().content_hash
    assert stored.resources == skill.snapshot().resources and stored.local_metadata == record.local_metadata
    assert record.content == "local representation"
    restored = await store.skills.get_skill(stored.version_id)
    assert type(restored) is Skill
    assert restored.resources == skill.resources
    assert restored.snapshot() == snapshot
    assert (await store.skills.get(stored.version_id)).content == skill.content
    assert (await store.skills.get_skill("missing")) is None


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


async def test_registration_is_lazy_and_all_columns_are_documented(tmp_path):
    path = tmp_path / "lazy.db"
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    try:
        persistence = create_persistence(database)
        assert not path.exists()
        assert len(database.tables.values()) == 10
        for spec in database.tables.values():
            assert all(column.comment for column in spec.table.columns)
            assert set(spec.insert_model.model_fields) == set(spec.table.c.keys())
            assert str(CreateTable(spec.table).compile(dialect=postgresql.dialect()))
        (await ensure_collection(persistence))
        (await ensure_collection(persistence))
        assert path.exists()
    finally:
        (await database.close())


async def test_core_records_round_trip_with_distinct_labels(store):
    first = version()
    second = version("local-v2").model_copy(update={"parent_version_ids": ["local-v1"]})
    (await store.skills.add(first))
    (await store.skills.add(second))
    (await store.database.insert(SKILLS_TABLE, [central_skill(first).model_dump()]))
    assert (await store.skills.get("local-v1")) == first
    assert len((await store.skills.list_versions("local-skill"))) == 2
    record = trajectory()
    (await store.trajectories.add(record))
    assert (await store.trajectories.get("t1")) == record
    with pytest.raises(IntegrityError):
        (await store.trajectories.add(trajectory("t2")))
    assert (await store.trajectories.get("t2")) is None


async def test_trajectory_consumption_counts_exact_versions_and_projects(store):
    """Count both origins once, preserve first use, and reject foreign evidence."""
    for identity, project, version_id, source in [
        ("one", "local-project", "v1", "local"),
        ("two", "local-project", "v1", "reported"),
        ("three", "local-project", "v2", "local"),
        ("foreign", "other-project", "v1", "local"),
    ]:
        (
            await store.trajectories.add(
                trajectory(identity, identity).model_copy(
                    update={
                        "project_id": project,
                        "skill_version_id": version_id,
                        "source": source,
                        "skill_id": "family",
                        "user_id": "user",
                        "agent_id": "agent",
                        "received_at": trajectory().created_at,
                    }
                )
            )
        )
    with store.session(Scope(fields={"project_id": "local-project"})) as scoped:
        assert (await scoped.trajectories.counts("v1")).model_dump() == {"total": 2, "consumed": 0, "unconsumed": 2}
        (await scoped.trajectories.mark_consumed("v1", ["one", "one"], "first"))
        (await scoped.trajectories.mark_consumed("v1", ["one"], "repeat"))
        assert (await scoped.trajectories.counts("v1")).model_dump() == {"total": 2, "consumed": 1, "unconsumed": 1}
        assert (await scoped.database.select(TRAJECTORY_CONSUMPTIONS_TABLE))[0]["run_id"] == "first"
        for identities in [["two", "foreign"], ["three"], ["missing"]]:
            with pytest.raises(ValueError, match="exact Skill version and project"):
                (await scoped.trajectories.mark_consumed("v1", identities, "invalid"))
        assert (await scoped.trajectories.counts("v1")).consumed == 1
        assert (await scoped.trajectories.counts("v2")).model_dump() == {"total": 1, "consumed": 0, "unconsumed": 1}
    with store.session(Scope(fields={"project_id": "other-project"})) as scoped:
        assert (await scoped.trajectories.counts("v1")).model_dump() == {"total": 1, "consumed": 0, "unconsumed": 1}
    with store.session(Scope(fields={"project_id": "local-project"})) as scoped:
        assert (await scoped.trajectories.counts("v1")).consumed == 1


async def test_transaction_rolls_back_across_services(store):
    with pytest.raises(IntegrityError):
        async with store.database.transaction() as database:
            (await database.insert(SKILL_VERSIONS_TABLE, [version().model_dump()]))
            (await database.insert(TRAJECTORIES_TABLE, [trajectory().model_dump(), trajectory("t2").model_dump()]))
    assert (await store.database.count(SKILL_VERSIONS_TABLE)) == 0
    assert (await store.database.count(TRAJECTORIES_TABLE)) == 0


async def test_core_only_and_no_global_registry_state(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'core.db'}"))
    try:
        persistence = create_persistence(database)
        assert {s.name for s in database.tables.values()} == {
            TRAJECTORY_CONSUMPTIONS_TABLE,
            SKILLS_TABLE,
            SKILL_VERSIONS_TABLE,
            TRAJECTORIES_TABLE,
            SKILL_GRAPHS_TABLE,
            SKILL_GRAPH_VERSIONS_TABLE,
            SKILL_GRAPH_NODES_TABLE,
            SKILL_GRAPH_EDGES_TABLE,
            SKILL_TASKS_TABLE,
            SKILL_TASK_LOGS_TABLE,
        }
        assert len(build_persistence_tables().values()) == 10
        (await ensure_collection(persistence))
        (await persistence.skills.add(version()))
    finally:
        (await database.close())


def test_invalid_core_values_are_rejected():
    with pytest.raises(ValueError):
        SkillVersionRecord(**(version().model_dump() | {"parent_version_ids": ["local-v1"]}))
    with pytest.raises(ValueError):
        TrajectoryRecord(**(trajectory().model_dump() | {"reward_score": float("nan")}))


@pytest.mark.asyncio
async def test_service_factory_assembles_borrowed_data_services(tmp_path):
    from mindmemos_skill import ServiceConfig, create_service

    path = tmp_path / "factory.db"
    service = await create_service(
        ServiceConfig.model_validate(
            {
                "home": tmp_path / "home",
                "persistence": {
                    "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{path}"}},
                },
            }
        )
    )
    try:
        persistence = service.backends["persistence"]
        with pytest.raises(RuntimeError, match="session is required"):
            (await persistence.skills.add(version()))
        with pytest.raises(ValueError, match="project_id"):
            with persistence.session(Scope(fields={"agent_id": "agent"})):
                pass
        with persistence.session(Scope(fields={"project_id": "local-project", "agent_id": "agent"})) as scoped:
            with pytest.raises(ValueError, match="agent_id"):
                (await scoped.skills.list_skills("local-project", limit=10, offset=0))
        with persistence.session(Scope(fields={"project_id": "local-project", "user_id": "user"})) as scoped:
            with pytest.raises(ValueError, match="user_id"):
                (await scoped.skills.list_skills("local-project", limit=10, offset=0))
        with pytest.raises(ValueError, match="unexpected"):
            with persistence.session(Scope(fields={"project_id": "local-project", "dataset_id": "dataset"})):
                pass
        assert path.exists()
        with persistence.session(Scope(fields={"project_id": "local-project"})) as scoped:
            (await scoped.skills.add(version()))
            (await scoped.database.insert(SKILLS_TABLE, [central_skill(version()).model_dump()]))
            assert (await scoped.skills.get("local-v1")) is not None
    finally:
        await service.close()


async def test_json_arrays_and_nested_metadata_use_infra_validation(store):
    record = trajectory().model_copy(
        update={
            "injected_skills": [{"version_id": "v1", "options": {"nested": [1, True, None]}}],
            "metadata": {"attempts": [1, 2]},
        }
    )
    (await store.trajectories.add(record))
    assert (await store.trajectories.get("t1")) == record
    with pytest.raises(ValueError):
        (
            await store.database.insert(
                TRAJECTORIES_TABLE, [trajectory("bad", "bad-run").model_dump() | {"metadata": {"bad": object()}}]
            )
        )
    assert (await store.trajectories.get("bad")) is None


def central_skill(version_record, *, alias=None):
    return SkillRecord(
        project_id=version_record.project_id,
        skill_id=version_record.skill_id,
        name=version_record.name,
        alias=alias,
        registration_source="/original/import/path",
        registration_message="Initial import",
        active_version_id=version_record.version_id,
    )


async def test_resolve_keeps_all_prefix_candidates_and_literal_matching(store):
    skill_ids = ["sk_one", "sk_two", "sk_%literal", "skXfalse", "SK_upper"]
    versions = [version(f"local-v{i}", skill_id) for i, skill_id in enumerate(skill_ids)]
    (await store.database.insert(SKILL_VERSIONS_TABLE, [record.model_dump() for record in versions]))
    (
        await store.database.insert(
            SKILLS_TABLE,
            [central_skill(record, alias=f"alias-{index}").model_dump() for index, record in enumerate(versions)],
        )
    )

    statements = []

    def count_select(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(store.database.engine.sync_engine, "before_cursor_execute", count_select)
    try:
        assert (await store.skills.resolve_family("local-project", "sk_one")).skill_id == "sk_one"
    finally:
        event.remove(store.database.engine.sync_engine, "before_cursor_execute", count_select)
    assert len(statements) == 1
    assert (await store.skills.resolve_family("local-project", "alias-1")).skill_id == "sk_two"
    assert (await store.skills.resolve_family("local-project", "sk_%")).skill_id == "sk_%literal"
    with pytest.raises(SkillConflictError, match="Ambiguous") as conflict:
        (await store.skills.resolve_family("local-project", "sk_"))
    for skill_id in skill_ids[:3]:
        assert skill_id in str(conflict.value)
    assert "skXfalse" not in str(conflict.value)
    assert "SK_upper" not in str(conflict.value)
    with pytest.raises(SkillNotFoundError):
        (await store.skills.resolve_family("other-project", "sk_one"))

    (await store.database.update(SKILLS_TABLE, {"alias": "sk_one"}, filters={"skill_id": "sk_two"}))
    with pytest.raises(SkillAliasConflictError):
        (await store.skills.resolve_family("local-project", "sk_one"))


async def test_resolve_selects_current_family_version_or_unique_historical_prefix(store):
    """Family references follow the head while version references stay pinned."""
    first = version("skv-aaa", "sk-family")
    second = version("skv-aab", "sk-family")
    (await store.database.insert(SKILL_VERSIONS_TABLE, [first.model_dump(), second.model_dump()]))
    (await store.database.insert(SKILLS_TABLE, [central_skill(first, alias="shortcut").model_dump()]))
    family, current = await store.skills.resolve("local-project", "shortcut")
    assert family.skill_id == "sk-family"
    assert current.version_id == "skv-aaa"
    assert (await store.skills.resolve("local-project", "sk-f"))[1].version_id == "skv-aaa"
    assert (await store.skills.resolve("local-project", "skv-aab"))[1].version_id == "skv-aab"
    with pytest.raises(SkillConflictError, match="Ambiguous"):
        (await store.skills.resolve("local-project", "skv-aa"))
    with pytest.raises(SkillNotFoundError):
        (await store.skills.resolve("other-project", "skv-aaa"))


async def test_algorithm_publication_finds_legacy_deterministic_id_after_rollout(store):
    """A retried old training run does not publish a second prefixed version."""
    run_id = "run-1"
    input_hash = "input-1"
    legacy_id = "v_" + uuid5(NAMESPACE_URL, f"training:local-project:{run_id}").hex
    record = version(legacy_id).model_copy(
        update={"local_metadata": {"training": {"run_id": run_id, "input_hash": input_hash}}}
    )
    (await store.database.insert(SKILL_VERSIONS_TABLE, [record.model_dump()]))
    (await store.database.insert(SKILLS_TABLE, [central_skill(record).model_dump()]))
    assert (await store.skills.algorithm_publication("local-project", run_id, input_hash)).version_id == legacy_id


async def test_central_skill_switch_preserves_history_and_checks_owner(store):
    first, second = version(), version("local-v2")
    record = central_skill(first, alias="example")
    key = {"project_id": record.project_id, "skill_id": record.skill_id}
    async with store.database.transaction() as database:
        (await database.insert(SKILL_VERSIONS_TABLE, [first.model_dump(), second.model_dump()]))
        (await database.insert(SKILLS_TABLE, [record.model_dump()]))
    assert SkillRecord.model_validate((await store.database.get(SKILLS_TABLE, key))) == record
    assert (
        await store.database.update(
            SKILLS_TABLE,
            {"active_version_id": second.version_id},
            filters=key | {"active_version_id": first.version_id},
        )
    ) == 1
    assert (
        await store.database.update(
            SKILLS_TABLE,
            {"active_version_id": first.version_id},
            filters=key | {"active_version_id": first.version_id},
        )
    ) == 0
    assert (await store.skills.get(first.version_id)) == first
    for invalid in (
        version("other-family", "another-skill"),
        version("other-project").model_copy(update={"project_id": "another-project"}),
    ):
        (await store.skills.add(invalid))
        with pytest.raises(IntegrityError):
            (await store.database.update(SKILLS_TABLE, {"active_version_id": invalid.version_id}, filters=key))
    with pytest.raises(ValueError):
        (await store.database.update(SKILLS_TABLE, {"active_version_id": None}, filters=key))
    with pytest.raises(IntegrityError):
        (await store.database.delete(SKILL_VERSIONS_TABLE, filters={"version_id": second.version_id}))
    assert (await store.database.update(SKILLS_TABLE, {"registration_source": "changed"}, filters=key)) == 1
    assert (await store.database.get(SKILLS_TABLE, key))["registration_source"] == "changed"
    assert (await store.database.get(SKILLS_TABLE, key))["active_version_id"] == second.version_id


async def test_central_alias_scope_and_registration_rollback(store):
    first = version()
    (await store.skills.add(first))
    record = central_skill(first, alias="example")
    (await store.database.insert(SKILLS_TABLE, [record.model_dump()]))
    duplicate = version("second-v", "second-skill")
    with pytest.raises(IntegrityError):
        async with store.database.transaction() as database:
            (await database.insert(SKILL_VERSIONS_TABLE, [duplicate.model_dump()]))
            (await database.insert(SKILLS_TABLE, [central_skill(duplicate, alias="example").model_dump()]))
    assert (await store.skills.get(duplicate.version_id)) is None
    other = version("third-v").model_copy(update={"project_id": "another-project"})
    (await store.skills.add(other))
    (await store.database.insert(SKILLS_TABLE, [central_skill(other, alias="example").model_dump()]))
    for index in range(2):
        unaliased = version(f"unaliased-v{index}", f"unaliased-s{index}")
        (await store.skills.add(unaliased))
        (await store.database.insert(SKILLS_TABLE, [central_skill(unaliased).model_dump()]))
    key = {"project_id": record.project_id, "skill_id": record.skill_id}
    (await store.database.update(SKILLS_TABLE, {"alias": None}, filters=key))
    assert (await store.database.get(SKILLS_TABLE, key))["alias"] is None


async def test_trajectory_statistics_calendar_scope_and_bounded_projection(store):
    """Aggregate Shanghai days in SQL, including zero days and exact project ownership."""
    from datetime import UTC, date, datetime

    for identity, created, project, version_id in [
        ("before", "2026-09-09T15:59:59+00:00", "local-project", "v1"),
        ("first", "2026-09-09T16:00:00+00:00", "local-project", "v1"),
        ("today", "2026-10-08T16:00:00+00:00", "local-project", "v1"),
        ("last", "2026-10-09T15:59:59+00:00", "local-project", "v1"),
        ("future", "2026-10-09T16:00:00+00:00", "local-project", "v1"),
        ("foreign", "2026-10-09T00:00:00+00:00", "foreign", "v1"),
        ("other-version", "2026-10-09T00:00:00+00:00", "local-project", "v2"),
    ]:
        (
            await store.trajectories.add(
                trajectory(identity, identity).model_copy(
                    update={
                        "project_id": project,
                        "skill_version_id": version_id,
                        "created_at": datetime.fromisoformat(created),
                    }
                )
            )
        )
    engine = store.database.engine.sync_engine
    statements = []

    def capture(_conn, _cursor, statement, parameters, _context, _many):
        statements.append((statement, parameters))

    with store.session(Scope(fields={"project_id": "local-project"})) as scoped:
        (await scoped.trajectories.mark_consumed("v1", ["before", "first", "today"], "run"))
        event.listen(engine, "before_cursor_execute", capture)
        try:
            result = await scoped.trajectories.statistics("v1", now=datetime(2026, 10, 9, 3, tzinfo=UTC))
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert result.totals.model_dump() == {"total": 5, "consumed": 3, "unconsumed": 2}
        assert len(result.days) == 30
        assert result.days[0].day == date(2026, 9, 10)
        assert result.days[0].model_dump(exclude={"day"}) == {"total": 1, "consumed": 1, "unconsumed": 0}
        assert result.days[-1].day == date(2026, 10, 9)
        assert result.days[-1].model_dump(exclude={"day"}) == {"total": 2, "consumed": 1, "unconsumed": 1}
        assert all(day.total == 0 for day in result.days[1:-1])
        assert result.last_evolved_at is None
        empty = await scoped.trajectories.statistics("missing", now=datetime(2026, 10, 9, 3, tzinfo=UTC))
        assert len(empty.days) == 30 and not any(day.total for day in empty.days)
    selects = [item for item in statements if item[0].startswith("SELECT")]
    assert len(selects) == 3
    daily_sql, parameters = selects[-1]
    assert "GROUP BY" in daily_sql
    assert "task_instruction" not in daily_sql and "reward_score" not in daily_sql
    assert "trajectory_hash" not in daily_sql and "SELECT *" not in daily_sql
    async with store.database.engine.connect() as connection:
        plan = (await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + daily_sql, parameters)).fetchall()
    assert any("trajectories_version_created_idx" in row[-1] for row in plan)
