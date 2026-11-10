"""Repository composition, ownership, extension, and shared transaction behavior."""

from uuid import uuid4

import pytest
from mindmemos_skill.common.time_utils import utcnow
from mindmemos_skill.errors import GraphNotFoundError, SkillAliasConflictError, SkillNotFoundError
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase, TableRegistry, TableSpec
from mindmemos_skill.infra.task_manager.typing import TaskSpec
from mindmemos_skill.persistence import SkillPersistence, SkillPersistenceSession
from mindmemos_skill.persistence.config import SkillPersistenceConfig
from mindmemos_skill.persistence.constants import SKILL_GRAPHS_TABLE, SKILLS_TABLE
from mindmemos_skill.persistence.db_services.graphs.typing import GraphWriteRequest, SkillGraphRecord
from mindmemos_skill.persistence.observability.constants import OBSERVABILITY_RUNS_TABLE
from mindmemos_skill.persistence.observability.typing import AuditRun
from mindmemos_skill.skill import SkillContent, SkillSnapshot
from pydantic import BaseModel
from sqlalchemy import Column, MetaData, Table, Text

NOTES_TABLE = "repository_test_notes"
"""Table owned by the custom repository fixture."""


@pytest.fixture
async def persistence(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'entrypoint.db'}"))
    store = SkillPersistence(SkillPersistenceConfig(), database=database)
    await store.initialize_storage()
    try:
        yield store, database
    finally:
        await database.close()


async def register(repository, alias=None):
    """Register a complete source through the repository's atomic business operation."""
    return await repository.register(
        name="Example",
        alias=alias,
        source=None,
        message=None,
        snapshot=SkillSnapshot.from_content(
            SkillContent(content="Example", name="Example", description="Example source", standard_format=False)
        ),
    )


async def test_entrypoint_requires_scope_and_retained_handles_expire(persistence):
    store, _ = persistence
    with pytest.raises(RuntimeError, match="session is required"):
        store.skills
    with store.session(Scope(fields={"project_id": "one"})) as session:
        record = await register(session.skills)
        assert store.skills is session.skills
        assert store.graphs is session.graphs
        assert store.tasks is session.tasks
        assert session.audit is None
        with pytest.raises(RuntimeError, match="change scope"):
            with store.session(Scope(fields={"project_id": "two"})):
                pass
    with pytest.raises(RuntimeError, match="closed"):
        await session.skills.get(record.active_version_id)
    with pytest.raises(RuntimeError, match="closed"):
        session.repository(frozenset({SKILLS_TABLE}))


async def test_project_id_properties_follow_sessions_and_transactions(persistence):
    """Read-only project access follows the active context without a cached identity."""
    store, _ = persistence
    with pytest.raises(RuntimeError, match="session is required"):
        store.project_id
    for project in ("one", "two"):
        with store.session(Scope(fields={"project_id": project})) as session:
            repository = session.skills.database
            for target in (store, session, repository):
                assert target.project_id == project
                with pytest.raises(AttributeError):
                    target.project_id = "other"
                assert target.project_id == project
            async with store.transaction() as transaction:
                assert store.project_id == transaction.project_id == transaction.graphs.database.project_id == project
            assert store.project_id == project
            with pytest.raises(RuntimeError, match="closed"):
                transaction.project_id
        for target in (session, repository):
            with pytest.raises(RuntimeError, match="closed"):
                target.project_id


async def test_project_id_properties_reject_maintenance_and_expired_parent_scope(persistence):
    """Unscoped maintenance and an expired parent cannot expose a project identity."""
    store, database = persistence
    with store.maintenance() as session:
        for target in (store, session, session.skills.database):
            with pytest.raises(RuntimeError, match="project scope is required"):
                target.project_id
    with database.scoped(Scope(fields={"project_id": "one"})) as scoped:
        session = SkillPersistenceSession(scoped, prefixed_ids=True)
        repository = session.skills.database
        assert session.project_id == repository.project_id == "one"
    try:
        for target in (session, repository):
            with pytest.raises(RuntimeError, match="closed"):
                target.project_id
    finally:
        session.close()


async def test_repositories_preserve_project_and_table_boundaries(persistence):
    store, database = persistence
    for project in ("one", "two"):
        with store.session(Scope(fields={"project_id": project})) as session:
            await register(session.skills)
        await database.insert(
            SKILL_GRAPHS_TABLE,
            [SkillGraphRecord(project_id=project, graph_id="skg-shared", alias="same", name=project).model_dump()],
        )
    with store.session(Scope(fields={"project_id": "one"})) as session:
        assert (await session.graphs.get_graph("same")).name == "one"
        assert len(await session.graphs.list_graphs()) == 1
        assert len(await session.skills.list_skills(limit=None, offset=0)) == 1
        with pytest.raises(PermissionError, match="cannot access"):
            await session.skills.database.count(SKILL_GRAPHS_TABLE)
        reader = session.repository(frozenset(), read_tables=frozenset({SKILLS_TABLE}))
        assert await reader.count(SKILLS_TABLE) == 1
        with pytest.raises(PermissionError, match="cannot write"):
            await reader.update(SKILLS_TABLE, {"name": "bad"}, filters={})
        with pytest.raises(ValueError, match="scope"):
            await reader.count(SKILLS_TABLE, filters={"project_id": "two"})
        async with reader.transaction() as transaction:
            with pytest.raises(PermissionError, match="cannot write"):
                await transaction.delete(SKILLS_TABLE, filters={})


async def test_transaction_commits_across_repositories_and_rebinds_entrypoint(persistence):
    store, _ = persistence
    with store.session(Scope(fields={"project_id": "one"})) as session:
        async with store.transaction() as transaction:
            assert store.skills is transaction.skills
            record = await register(store.skills)
            await transaction.tasks.create(TaskSpec(task_id="task", project_id="one", kind="train", payload={}))
            with pytest.raises(RuntimeError, match="Nested"):
                async with transaction.transaction():
                    pass
        assert store.skills is session.skills
        assert await session.skills.get(record.active_version_id)
        assert (await session.tasks.get("task")).project_id == "one"
        with pytest.raises(RuntimeError, match="closed"):
            await transaction.tasks.get("task")


async def test_caught_atomic_failure_rolls_back_the_shared_transaction(persistence):
    store, _ = persistence
    with store.session(Scope(fields={"project_id": "one"})) as session:
        with pytest.raises(RuntimeError, match="failed operation"):
            async with store.transaction() as transaction:
                await register(transaction.skills, alias="duplicate")
                with pytest.raises(SkillAliasConflictError):
                    await register(transaction.skills, alias="duplicate")
        assert await session.skills.list_skills(limit=None, offset=0) == []


async def test_exception_rolls_back_multiple_repositories(persistence):
    store, _ = persistence
    with store.session(Scope(fields={"project_id": "one"})) as session:
        with pytest.raises(ValueError, match="abort"):
            async with store.transaction() as transaction:
                await register(transaction.skills)
                await transaction.tasks.create(TaskSpec(task_id="task", project_id="one", kind="train", payload={}))
                raise ValueError("abort")
        assert await session.skills.list_skills(limit=None, offset=0) == []
        assert await session.tasks.get("task") is None


async def test_worker_capability_is_separate_from_project_tasks(persistence):
    store, _ = persistence
    worker = store.task_worker()
    for project in ("one", "two"):
        with store.session(Scope(fields={"project_id": project})) as session:
            await session.tasks.create(TaskSpec(task_id=project, project_id=project, kind="train", payload={}))
            with pytest.raises(RuntimeError, match="Worker capabilities"):
                store.task_worker()
            with pytest.raises(AttributeError):
                session.tasks.claim
    assert {record.project_id for record in await worker.due()} == {"one", "two"}
    assert await worker.scoped(Scope(fields={"project_id": "one"})).get("two") is None
    with store.session(Scope(fields={"project_id": "one"})) as session:
        assert await session.tasks.get("two") is None


async def test_graph_publication_adapter_uses_session_owned_skill_reader(persistence):
    store, _ = persistence
    with store.session(Scope(fields={"project_id": "one"})) as session:
        record = await register(session.skills)
        source = await session.skills.get_skill(record.active_version_id)
        writer = session.graphs.writer(
            GraphWriteRequest(
                graph_id="skg-" + uuid4().hex,
                actor_id="actor",
                version_ids={record.skill_id: record.active_version_id},
                expected_snapshot=None,
            )
        )
        assert await writer.validate_skills({record.skill_id: source}) == {record.skill_id: record.active_version_id}
        assert writer.project_id == "one" and writer.check_head is True
        with pytest.raises(PermissionError, match="cannot write"):
            await session.graphs.skills.delete(record)
        await session.skills.delete(record)
        with pytest.raises(SkillNotFoundError):
            await writer.validate_skills({record.skill_id: source})


async def test_graph_reference_queries_do_not_scan_the_project(persistence, monkeypatch):
    store, database = persistence
    await database.insert(
        SKILL_GRAPHS_TABLE,
        [SkillGraphRecord(project_id="one", graph_id="skg-abc", alias="example", name="Example").model_dump()],
    )
    with store.session(Scope(fields={"project_id": "one"})) as session:

        async def forbid_scan():
            raise AssertionError("Single-reference resolution must not list all graphs")

        monkeypatch.setattr(session.graphs, "list_graphs", forbid_scan)
        assert (await session.graphs.get_graph("example")).graph_id == "skg-abc"
        assert (await session.graphs.resolve("example"))["graph_id"] == "skg-abc"
        await session.graphs.delete("example")
        with pytest.raises(GraphNotFoundError):
            await session.graphs.get_graph("example")


async def test_optional_scope_dimensions_are_never_dropped(persistence):
    store, _ = persistence
    with store.session(Scope(fields={"project_id": "one", "agent_id": "agent"})) as session:
        # Graph tables do not declare agent ownership; widening to project-only is forbidden.
        with pytest.raises(ValueError, match="agent_id"):
            await session.graphs.list_graphs()


async def test_graph_deletion_reports_a_competing_delete(persistence, monkeypatch):
    store, database = persistence
    await database.insert(
        SKILL_GRAPHS_TABLE,
        [SkillGraphRecord(project_id="one", graph_id="skg-race", name="Example").model_dump()],
    )
    with store.session(Scope(fields={"project_id": "one"})) as session:
        resolve = session.graphs.get_graph

        async def competing_delete(reference):
            graph = await resolve(reference)
            await database.update(
                SKILL_GRAPHS_TABLE, {"is_deleted": True}, filters={"project_id": "one", "graph_id": graph.graph_id}
            )
            return graph

        monkeypatch.setattr(session.graphs, "get_graph", competing_delete)
        with pytest.raises(GraphNotFoundError, match="during deletion"):
            await session.graphs.delete("skg-race")


async def test_independent_audit_queries_keep_scope_and_do_not_join_business_transactions(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'business.db'}"))
    audit_database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'audit.db'}"))
    store = SkillPersistence(SkillPersistenceConfig(), database=database, audit_database=audit_database)
    try:
        await store.initialize_storage()
        for project in ("one", "two"):
            record = AuditRun(
                project_id=project,
                run_id="same",
                algorithm="evolve",
                base_version_id="source",
                status="running",
                audit_status="complete",
                capture_level="info",
                trace_id="a" * 32,
                started_at=utcnow(),
            )
            await audit_database.insert(OBSERVABILITY_RUNS_TABLE, [record.model_dump()])
        with store.session(Scope(fields={"project_id": "one"})) as session:
            assert (await session.audit.get_run("same")).project_id == "one"
            assert await session.audit.latest_evolution("source", ["evolve"])
            with pytest.raises(PermissionError, match="cannot write"):
                await session.audit.database.delete(OBSERVABILITY_RUNS_TABLE, filters={})
            async with store.transaction() as transaction:
                assert transaction.audit is None
        with pytest.raises(RuntimeError, match="closed"):
            await session.audit.get_run("same")
    finally:
        await audit_database.close()
        await database.close()


class Note(BaseModel):
    """Small record used by a custom backend repository."""

    project_id: str
    """Owning project."""

    note_id: str
    """Project-local note identity."""

    text: str
    """Stored note body."""


class NoteService:
    """Implement a custom query using only the granted repository handle."""

    def __init__(self, database):
        self.database = database

    async def matching(self, prefix):
        from mindmemos_skill.infra.database import Filter

        rows = await self.database.select(NOTES_TABLE, filters=[Filter(field="text", op="starts_with", value=prefix)])
        return [Note.model_validate(row) for row in rows]


class CustomSession(SkillPersistenceSession):
    """Add a named typed repository without a second registration framework."""

    def __init__(self, database, *, prefixed_ids, audit_database=None, state=None):
        super().__init__(database, prefixed_ids=prefixed_ids, audit_database=audit_database, state=state)
        self.notes = NoteService(self.repository(frozenset({NOTES_TABLE})))


class CustomPersistence(SkillPersistence):
    """Compose custom table definitions using the existing persistence family."""

    session_type = CustomSession

    def build_tables(self):
        table = Table(
            NOTES_TABLE,
            MetaData(),
            Column("project_id", Text, primary_key=True),
            Column("note_id", Text, primary_key=True),
            Column("text", Text, nullable=False),
        )
        return TableRegistry(
            (*super().build_tables().values(), TableSpec(name=NOTES_TABLE, table=table, insert_model=Note))
        )


async def test_custom_repository_queries_transactions_and_scope(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'custom.db'}"))
    store = CustomPersistence(SkillPersistenceConfig(), database=database)
    try:
        await store.initialize_storage()
        for project in ("one", "two"):
            with store.session(Scope(fields={"project_id": project})):
                async with store.transaction() as transaction:
                    await transaction.notes.database.insert(NOTES_TABLE, [{"note_id": "same", "text": "hello"}])
        with store.session(Scope(fields={"project_id": "one"})) as session:
            assert [note.project_id for note in await session.notes.matching("hel")] == ["one"]
            with pytest.raises(PermissionError, match="cannot access"):
                await session.notes.database.count(SKILLS_TABLE)
        with pytest.raises(RuntimeError, match="closed"):
            await session.notes.matching("hel")
    finally:
        await database.close()


async def test_scope_supplies_registration_and_trajectory_ownership(persistence):
    """Writers derive ownership and readers retain the authorized project boundary."""
    store, _ = persistence
    for project in ("one", "two"):
        with store.session(Scope(fields={"project_id": project})) as session:
            family = await register(session.skills, alias="shared")
            assert family.project_id == project
            assert (await session.skills.resolve_family("shared")).skill_id == family.skill_id
            trajectory = await session.trajectories.report(
                skill_version_ids=[family.active_version_id],
                user_id=None,
                agent_id="agent",
                messages=[],
                session_id=None,
                status="success",
                started_at=None,
                finished_at=None,
            )
            assert trajectory.project_id == project
            assert (await session.skills.resolve("shared"))[1].project_id == project
    with store.session(Scope(fields={"project_id": "one"})) as session:
        assert await session.trajectories.get(trajectory.trajectory_id) is None
        with pytest.raises(SkillNotFoundError):
            await session.skills.resolve(family.skill_id)
