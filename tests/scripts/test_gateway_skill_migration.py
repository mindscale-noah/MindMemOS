"""Exercise the gateway job against legacy Qdrant payloads and live Skill tables."""

import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from fastapi.encoders import jsonable_encoder
from mindmemos.components.skill.bundle import compute_content_hash, serialize_bundle
from mindmemos.config import QdrantConfig
from mindmemos.infra.db.engine import QdrantEngine
from mindmemos.mappers.skill import skill_blob_id, skill_version_id
from mindmemos.pipelines.memory_db.operation_records import MemoryOperationRecorder
from mindmemos.typing.memory import DialogueMessage, MemoryRequestContext, TextMessage
from mindmemos.typing.service import AddPipelineInput
from mindmemos.typing.skill import SkillBinding
from mindmemos.workers import legacy_skill_trajectory as delivery_worker
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.service import SkillService
from mindmemos_skill.config.config import FastAPIConfig, HttpAuthConfig
from mindmemos_skill.errors import SkillNotFoundError
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence.constants import (
    LEGACY_SKILL_SYNC_ITEMS,
    LEGACY_SKILL_SYNC_PROJECTS,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
    TRAJECTORIES_TABLE,
    TRAJECTORY_VERSIONS_TABLE,
)
from mindmemos_skill.persistence.db_services.skills.typing import SkillRecord, SkillVersionRecord
from mindmemos_skill.persistence.db_services.trajectories.service import TrajectoryDataService
from mindmemos_skill.persistence.factory import create_persistence
from mindmemos_skill.persistence.observability import AlgorithmAuditPersistence
from mindmemos_skill.skill.typing import SkillContent, SkillSnapshot
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from scripts.gateway.skill_migration import __main__ as cli
from scripts.gateway.skill_migration.migrate import discover_projects, migrate_project
from scripts.gateway.skill_migration.sync import sync_project
from scripts.gateway.skill_migration.typing import MigrationCounts, MigrationFailure, ProjectMigration

PROJECT = "gateway-project"
OTHER = "other-gateway-project"
FAMILY = str(UUID(int=101))
OTHER_FAMILY = str(UUID(int=102))
TRACE = str(UUID(int=201))
SECOND_TRACE = str(UUID(int=202))
WHEN = "2025-12-01T12:00:00Z"
ROOT_CONTENT = "---\nname: from-header\ndescription: Valid description\n---\nOriginal.\n"
CHILD_CONTENT = "---\ndescription: Kept description\n---\nEdited.\n"


async def _version(client, config, content, *, parent=None, project=PROJECT, family=FAMILY, **fields):
    """Seed a version and its canonical legacy bundle."""
    files = {"SKILL.md": content}
    digest = compute_content_hash(files)
    version_id = skill_version_id(project, digest, parent)
    await client.upsert(
        config.skill_blob_collection,
        [
            PointStruct(
                id=skill_blob_id(project, digest),
                vector={},
                payload={
                    "project_id": project,
                    "content_hash": digest,
                    "content": serialize_bundle(files),
                    "created_at": WHEN,
                },
            )
        ],
    )
    await client.upsert(
        config.skill_version_collection,
        [
            PointStruct(
                id=version_id,
                vector={},
                payload={
                    "project_id": project,
                    "cloud_skill_id": family,
                    "version_id": version_id,
                    "parent_version_id": parent,
                    "content_hash": digest,
                    "status": "observed",
                    "origin": "edge",
                    "created_at": WHEN,
                    **fields,
                },
            )
        ],
    )
    return version_id


async def _add(client, config, identifier, bindings, **fields):
    """Seed a Memory add record with unchanged dialogue and actor fields."""
    await client.upsert(
        config.add_record_collection,
        [
            PointStruct(
                id=identifier,
                vector={},
                payload={
                    "project_id": PROJECT,
                    "add_record_id": identifier,
                    "request_submitted_at": WHEN,
                    "task_completed_at": "2025-12-01T12:00:02Z",
                    "status": "ok",
                    "messages": [{"role": "user", "content": "Do the task"}, {"role": "assistant", "content": "Done"}],
                    "skill_bindings": bindings,
                    "user_id": "business-user",
                    "agent_id": "agent",
                    "session_id": "session",
                    "task_id": "shared-task",
                    "score": 0.8,
                    **fields,
                },
            )
        ],
    )


@pytest.fixture
async def stores(tmp_path):
    """Supply seeded-compatible storage with a reusable on-disk target."""
    cfg = QdrantConfig(auto_create=False, batch_upsert_enabled=False)
    client = AsyncQdrantClient(":memory:")
    for collection in (
        cfg.skill_version_collection,
        cfg.skill_blob_collection,
        cfg.add_record_collection,
        cfg.skill_trace_summary_collection,
    ):
        await client.create_collection(collection, vectors_config={})
    source = QdrantEngine(cfg, client=client)
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'target.db'}"))
    target = create_persistence(database)
    await target.initialize_storage()
    try:
        yield client, cfg, source, target
    finally:
        await database.close()
        await source.close()


async def test_preview_apply_and_replay(stores):
    client, cfg, source, target = stores
    root = await _version(client, cfg, ROOT_CONTENT, version_label="v2.0.0", status="published", skill_name="old-name")
    child = await _version(
        client, cfg, CHILD_CONTENT, parent=root, version_label="old tag", skill_name="from-registration"
    )
    foreign = await _version(client, cfg, "No metadata.", project=OTHER, family=FAMILY)
    await _add(
        client,
        cfg,
        TRACE,
        [
            {"version_id": child, "usage": "modified"},
            {"version_id": root},
            {"version_id": foreign},
            {"version_id": None},
        ],
    )
    await _add(client, cfg, SECOND_TRACE, [{"version_id": None}])
    await _add(client, cfg, str(UUID(int=203)), [{"version_id": root}], project_id=OTHER)
    await _add(client, cfg, str(UUID(int=204)), [], project_id="no-registered-skills")
    await _add(client, cfg, str(UUID(int=205)), [{"version_id": root}], task_id=None, score=None)

    projects = await discover_projects(source, cfg, page_size=1, unscoped_skills=[])
    assert projects == [PROJECT, OTHER]
    preview = await migrate_project(source, cfg, target, PROJECT, apply=False, page_size=1)
    assert preview.counts.model_dump() == {
        "families": 1,
        "versions": 2,
        "new_families": 1,
        "verified_families": 1,
        "new_versions": 2,
        "trajectories": 2,
        "new_trajectories": 2,
        "verified_trajectories": 2,
        "skipped_records": 1,
        "skipped_bindings": 3,
        "fallback_versions": 1,
    }
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert await session.database.count(SKILLS_TABLE) == 0
        assert await session.database.count(TRAJECTORIES_TABLE) == 0

    applied = await migrate_project(source, cfg, target, PROJECT, apply=True, page_size=1)
    assert applied == preview
    await migrate_project(source, cfg, target, OTHER, apply=True, page_size=1)
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        (family,) = await session.skills.list_skills(limit=None, offset=0)
        root_id = family.active_version_id
        assert re.fullmatch(r"sk-[0-9a-f]{32}", family.skill_id)
        assert re.fullmatch(r"skv-[0-9a-f]{32}", root_id)
        assert family.name == "from-header"
        assert (await session.skills.get(root_id)).version_label == "v2.0.0"
        trace = await session.trajectories.get(TRACE)
        child_id, bound_root = list(trace.bound_versions)
        assert bound_root == root_id
        assert re.fullmatch(r"skv-[0-9a-f]{32}", child_id)
        child_row = await session.skills.get(child_id)
        assert child_row.content == CHILD_CONTENT
        assert (child_row.name, child_row.description) == ("from-registration", "Kept description")
        assert not child_row.standard_format
        assert child_row.parent_version_ids == [root_id]
        assert child_row.local_metadata["legacy-memory"]["version_id"] == child
        assert child_row.local_metadata["legacy-memory"]["cloud_skill_id"] == FAMILY
        assert child_row.local_metadata["legacy-memory"]["version_label"] == "old tag"
        assert child_row.version_label == "v1.0.1-legacy-memory"
        assert trace.bound_versions == {child_id: family.skill_id, root_id: family.skill_id}
        assert trace.skill_bindings[0]["usage"] == "modified"
        assert trace.trajectory[1] == {"role": "assistant", "content": "Done"}
        assert (trace.user_id, trace.agent_id, trace.session_id) == ("business-user", "agent", "session")
        assert (trace.task_id, trace.reward_score) == ("shared-task", 0.8)
        assert trace.received_at == datetime(2025, 12, 1, 12, tzinfo=UTC)
        assert trace.started_at is None and trace.finished_at is None and trace.status == "unknown"
        assert trace.metadata["legacy-memory"]["add_status"] == "ok"
        assert (await session.trajectories.counts(root_id)).total == 2
        assert (await session.trajectories.counts(child_id)).total == 1
        (reported,) = await session.trajectories.list_reported(
            family.skill_id,
            skill_version_id=child_id,
            status=None,
            received_at_from=None,
            received_at_to=None,
            snapshot_at=datetime.now(UTC),
            after_received_at=None,
            after_trajectory_id=None,
            limit=10,
        )
        assert reported.trajectory_id == TRACE
        assert await session.trajectories.get(SECOND_TRACE) is None
        missing_fields = await session.trajectories.get(str(UUID(int=205)))
        assert (missing_fields.task_id, missing_fields.reward_score) == (str(UUID(int=205)), None)
        await session.skills.activate(family, child_id)

    replay = (await migrate_project(source, cfg, target, PROJECT, apply=True, page_size=2)).counts
    assert (replay.new_families, replay.new_versions, replay.new_trajectories) == (0, 0, 0)
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert (await session.skills.resolve_family(family.skill_id)).active_version_id == child_id
        assert await session.database.count(TRAJECTORIES_TABLE) == 2
    with target.session(Scope(fields={"project_id": OTHER})) as session:
        (foreign_family,) = await session.skills.list_skills(limit=None, offset=0)
        assert foreign_family.skill_id != family.skill_id
        assert (await session.skills.get(foreign_family.active_version_id)).local_metadata["legacy-memory"][
            "version_id"
        ] == foreign
        assert await session.database.count(TRAJECTORIES_TABLE) == 0


async def test_transaction_failure_then_restart(stores, monkeypatch):
    client, cfg, source, target = stores
    version = await _version(client, cfg, "Bare body.")
    later_trace = str(UUID(int=203))
    for identifier in (TRACE, SECOND_TRACE, later_trace):
        await _add(client, cfg, identifier, [{"version_id": version}])
    index_versions = TrajectoryDataService._index_versions

    async def interrupted_index(transaction, record):
        await index_versions(transaction, record)
        if record.trajectory_id == SECOND_TRACE:
            raise RuntimeError("simulated interruption during write")

    with monkeypatch.context() as patch:
        patch.setattr(TrajectoryDataService, "_index_versions", staticmethod(interrupted_index))
        result = await migrate_project(source, cfg, target, PROJECT, apply=True, page_size=1)
    assert result.counts.new_trajectories == 2
    (failure,) = result.failed_trajectories
    assert (failure.trajectory_id, failure.reason) == (SECOND_TRACE, "operation_failed")

    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        (family,) = await session.skills.list_skills(limit=None, offset=0)
        version_id = family.active_version_id
        assert (await session.skills.get(version_id)).name == f"legacy-skill-{FAMILY}"
        assert await session.trajectories.get(TRACE) is not None
        assert await session.trajectories.get(SECOND_TRACE) is None
        assert await session.trajectories.get(later_trace) is not None
        assert (await session.trajectories.counts(version_id)).total == 2

    retry = await migrate_project(source, cfg, target, PROJECT, apply=True, page_size=1)
    assert not retry.failed_trajectories
    assert (retry.counts.new_versions, retry.counts.new_trajectories) == (0, 1)
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert (await session.trajectories.counts(version_id)).total == 3

    await _add(client, cfg, TRACE, [{"version_id": version}], score=0.5)
    conflict = await migrate_project(source, cfg, target, PROJECT, apply=True, page_size=1)
    (failure,) = conflict.failed_trajectories
    assert (failure.trajectory_id, failure.reason) == (TRACE, "target_trajectory_conflict")
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert (await session.trajectories.get(TRACE)).reward_score == 0.8


@pytest.mark.parametrize("reason", ["missing_parent", "missing_bundle", "bundle_owner_mismatch"])
async def test_invalid_skill_does_not_block_other_records(stores, reason):
    client, cfg, source, target = stores
    parent = str(UUID(int=999)) if reason == "missing_parent" else None
    bad_version = await _version(client, cfg, ROOT_CONTENT, parent=parent)
    blob = skill_blob_id(PROJECT, compute_content_hash({"SKILL.md": ROOT_CONTENT}))
    if reason == "missing_bundle":
        await client.delete(cfg.skill_blob_collection, [blob])
    if reason == "bundle_owner_mismatch":
        await client.set_payload(cfg.skill_blob_collection, {"project_id": OTHER}, [blob])
    good_version = await _version(client, cfg, "Good", family=OTHER_FAMILY)
    await _add(client, cfg, TRACE, [{"version_id": good_version}, {"version_id": bad_version}])
    await _add(client, cfg, SECOND_TRACE, [{"version_id": good_version}])
    result = await migrate_project(source, cfg, target, PROJECT, apply=True, page_size=1)
    (failure,) = result.failed_skills
    assert (failure.skill_id, failure.version_ids, failure.reason) == (FAMILY, [bad_version], reason)
    (failure,) = result.failed_trajectories
    assert (failure.trajectory_id, failure.reason) == (TRACE, "skill_migration_failed")
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert await session.database.count(SKILLS_TABLE) == 1
        assert await session.database.count(SKILL_VERSIONS_TABLE) == 1
        assert await session.trajectories.get(TRACE) is None
        assert await session.trajectories.get(SECOND_TRACE) is not None


async def test_previous_import_is_not_duplicated(stores):
    client, cfg, source, target = stores
    old_version = await _version(client, cfg, ROOT_CONTENT)
    snapshot = SkillSnapshot.from_content(SkillContent(content=ROOT_CONTENT))
    version = SkillVersionRecord(
        **snapshot.model_dump(), project_id=PROJECT, skill_id=FAMILY, version_id=old_version, origin="legacy-memory"
    )
    family = SkillRecord(
        project_id=PROJECT,
        skill_id=FAMILY,
        name=snapshot.name,
        active_version_id=old_version,
        registration_source="legacy-memory",
    )
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        await session.database.insert(SKILL_VERSIONS_TABLE, [version.model_dump()])
        await session.database.insert(SKILLS_TABLE, [family.model_dump()])
    result = await migrate_project(source, cfg, target, PROJECT, apply=True, page_size=1)
    (failure,) = result.failed_skills
    assert (failure.skill_id, failure.reason) == (FAMILY, "old_id_import")
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert await session.database.count(SKILLS_TABLE) == 1
        assert await session.database.count(SKILL_VERSIONS_TABLE) == 1


@pytest.mark.parametrize("apply", [True, False])
def test_cli_writes_summary_and_failure_ids(tmp_path, monkeypatch, capsys, apply):
    output = tmp_path / "report.json"
    args = ["migration", "--qdrant-url", "http://localhost", "--report", str(output)]
    if apply:
        args.append("--apply")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setenv(cli.TARGET_DATABASE_ENV, "test-connection")

    async def collect(args, connection_url, run):
        assert args.apply is apply and connection_url == "test-connection"
        run.projects_discovered = 1
        run.projects.append(
            ProjectMigration(
                project_id=PROJECT,
                counts=MigrationCounts(
                    families=3,
                    new_families=1,
                    verified_families=2,
                    trajectories=2,
                    new_trajectories=1,
                    verified_trajectories=1,
                    skipped_records=1,
                ),
                failed_skills=[
                    MigrationFailure(
                        project_id=PROJECT, skill_id=FAMILY, reason="missing_bundle", detail="Missing bundle"
                    )
                ],
                failed_trajectories=[
                    MigrationFailure(
                        project_id=PROJECT,
                        trajectory_id=TRACE,
                        reason="skill_migration_failed",
                        detail="Missing version",
                    )
                ],
            )
        )

    monkeypatch.setattr(cli, "_run", collect)
    assert cli.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed.pop("report_path") == str(output)
    report = json.loads(output.read_text())
    assert report["summary"] == printed
    assert printed["status"] == "completed_with_failures"
    assert printed["projects"] == {"total": 1, "completed": 0, "with_failures": 1, "incomplete": 0, "unprocessed": 0}
    assert printed["skills"] == {
        "total": 3,
        "migrated": int(apply),
        "would_migrate": int(not apply),
        "already_migrated": 1,
        "failed": 1,
        "unprocessed": 0,
    }
    assert printed["trajectories"]["scanned"] == 3
    assert printed["trajectories"]["skipped"] == 1
    assert printed["failure_reasons"]["failed_skills"] == {"missing_bundle": 1}
    assert printed["failure_reasons"]["failed_trajectories"] == {"skill_migration_failed": 1}
    assert report["failed_skills"][0]["skill_id"] == FAMILY
    assert report["failed_trajectories"][0]["trajectory_id"] == TRACE
    assert cli.main() == 1
    assert json.loads(output.read_text()) == report


def test_cli_keeps_report_when_setup_is_incomplete(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["migration", "--qdrant-url", "http://localhost", "--report", str(output)])
    monkeypatch.delenv(cli.TARGET_DATABASE_ENV, raising=False)
    assert cli.main() == 1
    report = json.loads(output.read_text())
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "incomplete"
    assert printed["skills"]["migrated"] == printed["trajectories"]["migrated"] == 0
    assert report["summary"]["failure_reasons"]["run_failures"] == {"missing_target": 1}


async def test_continuous_sync_preserves_native_activation_and_rebinds_legacy_trajectories(stores):
    client, cfg, source, target = stores
    root = await _version(client, cfg, ROOT_CONTENT)
    await _add(client, cfg, TRACE, [{"version_id": root}, {"version_id": None}])
    result = await sync_project(source, cfg, target, PROJECT, page_size=1)
    assert not result.failed_skills and not result.failed_trajectories, result
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        family = await session.skills.resolve_family(FAMILY)
        imported_root = await session.skills.version(family, root)
        assert family.active_version_id is None
        assert (await session.database.get(LEGACY_SKILL_SYNC_PROJECTS, PROJECT))["ready"]
        mapping = {"project_id": PROJECT, "kind": "trajectory", "legacy_id": TRACE}
        assert (await session.database.get(LEGACY_SKILL_SYNC_ITEMS, mapping))["pending"]

    await client.set_payload(cfg.skill_version_collection, {"status": "published"}, [root])
    result = await sync_project(source, cfg, target, PROJECT, page_size=1)
    assert not result.failed_skills
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        family = await session.skills.resolve_family(FAMILY)
        assert family.active_version_id == imported_root.version_id
        native = await session.skills.commit(
            family, imported_root.snapshot(), message="native edit", base_version_id=root
        )
        assert native.native_owned_at is not None
        assert native.active_version_id != imported_root.version_id

    child = await _version(
        client, cfg, CHILD_CONTENT, parent=root, status="published", created_at="2025-12-02T12:00:00Z"
    )
    await _add(client, cfg, TRACE, [{"version_id": root}, {"version_id": child}])
    for _ in range(2):
        result = await sync_project(source, cfg, target, PROJECT, page_size=1)
        assert not result.failed_skills and not result.failed_trajectories, result
        assert result.counts.new_trajectories == 0
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        family = await session.skills.resolve_family(FAMILY)
        assert family.active_version_id == native.active_version_id
        mapped_child = await session.skills.get(child)
        assert mapped_child.parent_version_ids == [imported_root.version_id]
        assert (await session.trajectories.get(TRACE)).bound_versions == {
            imported_root.version_id: family.skill_id,
            mapped_child.version_id: family.skill_id,
        }
        assert await session.database.count(TRAJECTORY_VERSIONS_TABLE, filters={"trajectory_id": TRACE}) == 2
        assert not (await session.database.get(LEGACY_SKILL_SYNC_ITEMS, mapping))["pending"]
    with target.session(Scope(fields={"project_id": OTHER})) as session:
        with pytest.raises(SkillNotFoundError, match="Version not found"):
            await session.skills.get(root)
    await client.delete(cfg.skill_version_collection, [root, child])
    result = await sync_project(source, cfg, target, PROJECT, page_size=1)
    assert not result.failed_skills
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert (await session.skills.resolve_family(FAMILY)).active_version_id == native.active_version_id


async def test_continuous_sync_resumes_after_interrupted_page(stores, monkeypatch):
    client, cfg, source, target = stores
    root = await _version(client, cfg, ROOT_CONTENT, status="published")
    await _add(client, cfg, TRACE, [{"version_id": root}])
    await _add(client, cfg, SECOND_TRACE, [{"version_id": root}])
    scroll = source.scroll

    async def interrupt(collection, **kwargs):
        if collection == cfg.add_record_collection and kwargs.get("offset") is not None:
            raise ConnectionError("interrupted")
        return await scroll(collection, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(source, "scroll", interrupt)
        with pytest.raises(ConnectionError, match="interrupted"):
            await sync_project(source, cfg, target, PROJECT, page_size=1)
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        checkpoint = await session.database.get(LEGACY_SKILL_SYNC_PROJECTS, PROJECT)
        assert not checkpoint["ready"]
        assert checkpoint["trajectory_cursor"] is not None
        assert await session.database.count(TRAJECTORIES_TABLE, filters={}) == 1
    resumed = await sync_project(source, cfg, target, PROJECT, page_size=1)
    assert resumed.counts.new_trajectories == 1
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert await session.database.count(TRAJECTORIES_TABLE, filters={}) == 2
        assert (await session.database.get(LEGACY_SKILL_SYNC_PROJECTS, PROJECT))["ready"]


async def test_watch_discovers_only_active_platform_projects_and_keeps_deleted_mirrors(stores):
    client, cfg, source, target = stores
    root = await _version(client, cfg, ROOT_CONTENT, status="published")
    await _version(client, cfg, ROOT_CONTENT, project=OTHER)
    await _add(client, cfg, TRACE, [{"version_id": root}])
    await client.upsert(
        cfg.skill_trace_summary_collection,
        [
            PointStruct(
                id=TRACE,
                vector={},
                payload={
                    "project_id": PROJECT,
                    "summary_id": TRACE,
                    "add_record_id": TRACE,
                    "cloud_skill_id": FAMILY,
                    "skill_name": "example",
                    "summary": "Legacy evidence",
                    "created_at": WHEN,
                    "consumed_version_id": root,
                },
            )
        ],
    )
    registry = create_async_engine("sqlite+aiosqlite://")
    try:
        async with registry.begin() as connection:
            await connection.execute(text("CREATE TABLE memory_projects (namespace TEXT, active BOOLEAN)"))
            await connection.execute(
                text("INSERT INTO memory_projects VALUES (:project, true), (:other, false)"),
                {"project": PROJECT, "other": OTHER},
            )
        assert await cli._watch_projects(registry, source, cfg, target, page_size=1) == [PROJECT]
        with target.session(Scope(fields={"project_id": PROJECT})) as session:
            assert not (await session.database.get(LEGACY_SKILL_SYNC_PROJECTS, PROJECT))["ready"]
        result = await sync_project(source, cfg, target, PROJECT, page_size=1)
        assert not result.failed_skills and not result.failed_trajectories, result
        with target.session(Scope(fields={"project_id": PROJECT})) as session:
            trajectory = await session.trajectories.get(TRACE)
            assert trajectory.metadata["legacy-memory"]["summary"]["consumed_version_id"] == root
        await client.delete(cfg.skill_version_collection, [root])
        assert await cli._watch_projects(registry, source, cfg, target, page_size=1) == [PROJECT]
        await sync_project(source, cfg, target, PROJECT, page_size=1)
        with target.session(Scope(fields={"project_id": PROJECT})) as session:
            with pytest.raises(SkillNotFoundError):
                await session.skills.resolve_family(FAMILY)
            assert await session.trajectories.get(TRACE) == trajectory
        with target.session(Scope(fields={"project_id": OTHER})) as session:
            assert await session.database.get(LEGACY_SKILL_SYNC_PROJECTS, OTHER) is None
    finally:
        await registry.dispose()


async def test_saved_add_delivers_without_waiting_and_converges_with_reconciliation(stores, monkeypatch):
    client, cfg, source, target = stores
    root = await _version(client, cfg, ROOT_CONTENT, status="published")
    await sync_project(source, cfg, target, PROJECT, page_size=1)

    async def append(point):
        await client.upsert(
            cfg.add_record_collection,
            [PointStruct(id=point.add_record_id, vector={}, payload=jsonable_encoder(point.payload))],
        )

    async def read_records(project_id, record_ids):
        records = await source.retrieve(cfg.add_record_collection, record_ids)
        return [record for record in records if record.payload["project_id"] == project_id]

    monkeypatch.setenv("MINDMEMOS_LEGACY_SKILL_INGEST_SECRET", "private-ingest-token")
    monkeypatch.setattr(
        delivery_worker,
        "get_database_clients",
        lambda: SimpleNamespace(qdrant=SimpleNamespace(get_add_records_by_ids=read_records)),
    )
    service = SkillService(
        backends={}, observability=AlgorithmAuditPersistence(None), persistence=target, default_project_id=PROJECT
    )
    policy = FastAPIConfig(auth=HttpAuthConfig(mode="none", default_project_id=PROJECT))
    app = create_app(service, api_config=policy)
    queue = asyncio.Queue()
    monkeypatch.setattr(delivery_worker, "_queue", queue)
    outbound_started = asyncio.Event()
    release = asyncio.Event()
    transport = httpx.ASGITransport(app=app)

    async def hold_delivery(request):
        outbound_started.set()
        await release.wait()
        return await transport.handle_async_request(request)

    recorder = MemoryOperationRecorder(add_record_store=SimpleNamespace(append=append))
    context = MemoryRequestContext(request_id="request", project_id=PROJECT, account_id="account", api_key_uuid="key")
    add_input = AddPipelineInput(
        messages=[
            DialogueMessage(role="user", content="Remember this", timestamp=1770000000000),
            TextMessage(text="Standalone note"),
        ],
        mode="sync",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(hold_delivery),
        base_url="http://127.0.0.1",
        headers={"Authorization": "Bearer private-ingest-token"},
    ) as http:
        consumer = asyncio.create_task(delivery_worker._consume(queue, http))
        try:
            await recorder.record_add_input(
                add_input,
                ctx=context,
                request_submitted_at=datetime.fromisoformat(WHEN),
                add_record_id=SECOND_TRACE,
                status="processing",
            )
            assert queue.empty()
            stored_id = await asyncio.wait_for(
                recorder.record_add_input(
                    add_input,
                    ctx=context,
                    request_submitted_at=datetime.fromisoformat(WHEN),
                    add_record_id=TRACE,
                    status="processing",
                    skill_bindings=[
                        SkillBinding(
                            name="from-header",
                            version_id=root,
                            content_hash=compute_content_hash({"SKILL.md": ROOT_CONTENT}),
                        )
                    ],
                ),
                timeout=2,
            )
            assert stored_id == TRACE
            await asyncio.wait_for(outbound_started.wait(), timeout=2)
            with target.session(Scope(fields={"project_id": PROJECT})) as session:
                assert await session.trajectories.get(TRACE) is None
            release.set()
            await asyncio.wait_for(queue.join(), timeout=2)
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
    replay = await sync_project(source, cfg, target, PROJECT, page_size=1)
    assert not replay.failed_trajectories, replay
    assert replay.counts.new_trajectories == 0
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        trajectory = await session.trajectories.get(TRACE)
        assert trajectory is not None
        assert trajectory.trajectory[-1] == {"text": "Standalone note"}
        assert await session.trajectories.get(SECOND_TRACE) is None
        assert await session.database.count(TRAJECTORIES_TABLE, filters={}) == 1
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as http:
        denied = await http.post(
            "/internal/v1/legacy-skill/trajectories",
            headers={"Authorization": "Bearer original-project-key"},
            json={
                "project_id": PROJECT,
                "add_record_id": TRACE,
                "messages": [{"role": "user", "content": "Another message"}],
                "skill_bindings": [{"version_id": root}],
            },
        )
        assert denied.status_code == 401
