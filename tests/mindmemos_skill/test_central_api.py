"""Central repository workflows, isolation, conflicts, and complete exports."""

from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.errors import SkillConflictError, SkillInUseError, SkillNotFoundError
from mindmemos_skill.persistence import TrajectoryRecord
from mindmemos_skill.persistence.constants import (
    REMOTE_SKILL_BINDINGS_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    TRAJECTORIES_TABLE,
)
from mindmemos_skill.persistence.db_services.remote.typing import RemoteSkillBindingRecord, RemoteVersionBindingRecord
from mindmemos_skill.persistence.db_services.skills import SkillDataService
from mindmemos_skill.pipelines.typing import CommitRequest, RegisterRequest, SkillRequest
from mindmemos_skill.skill import Skill, SkillBundle, SkillResources

from mindmemos_skill import ServiceConfig, create_service


@pytest_asyncio.fixture
async def service(tmp_path):
    config = ServiceConfig.model_validate(
        {
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'central.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
        }
    )
    result = await create_service(config)
    await result.initialize_storage()
    try:
        yield result
    finally:
        await result.close()


def snapshot(text="first", resources=None):
    return Skill(
        bundle=SkillBundle(files={"SKILL.md": text}), resources=SkillResources(files=resources or {})
    ).snapshot()


@pytest.mark.asyncio
async def test_cloud_status_tracks_exact_local_version_and_scope(service):
    item = await service.register_skill(RegisterRequest(project_id="p", source=snapshot(), name="Cloud status"))
    request = SkillRequest(project_id="p", skill_ref=item.skill.skill_id)
    assert await service.skill_cloud_status(request) == "not_synced"
    remote = service.repository.persistence.remote
    remote.bind_skill(
        RemoteSkillBindingRecord(
            remote_id="test",
            project_id="cloud-project",
            cloud_skill_id="cloud-skill",
            skill_id=item.skill.skill_id,
        )
    )
    assert await service.skill_cloud_status(request) == "not_synced"
    remote.bind_version(
        RemoteVersionBindingRecord(
            remote_id="test",
            project_id="cloud-project",
            cloud_skill_id="cloud-skill",
            version_id="cloud-version",
            local_version_id=item.version.version_id,
            skill_name=item.skill.name,
            content_hash="cloud-hash",
            content="cloud snapshot",
            status="published",
            origin="edge",
        )
    )
    assert await service.skill_cloud_status(request) == "synced"
    await service.commit_skill(CommitRequest(project_id="p", skill_ref=item.skill.skill_id, source=snapshot("second")))
    assert await service.skill_cloud_status(request) == "history_only"
    with pytest.raises(SkillNotFoundError):
        await service.skill_cloud_status(SkillRequest(project_id="other", skill_ref=item.skill.skill_id))


async def register(service, *, project="p", alias="example", source=None):
    return await service.register_skill(
        RegisterRequest(
            project_id=project, name="Example", alias=alias, source=source or snapshot(), message="Initial registration"
        )
    )


def ref(detail, **extra):
    return {"project_id": detail.skill.project_id, "skill_ref": detail.skill.skill_id, **extra}


@pytest.mark.asyncio
async def test_full_central_workflow_leaves_external_directories_untracked(service, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("original")
    (source / "asset.bin").write_bytes(b"\0old")
    first = await register(service, source=str(source))
    (source / "SKILL.md").write_text("edited externally")
    assert "original" in (await service.show_skill(ref(first))).version.bundle
    assert (await service.show_skill({"project_id": "p", "skill_ref": "example"})).skill == first.skill
    assert (await service.show_skill({"project_id": "p", "skill_ref": first.skill.skill_id[:12]})).skill == first.skill
    second = await service.commit_skill(CommitRequest(**ref(first), source=str(source), message="New version"))
    assert second.version.parent_version_ids == [first.version.version_id]
    assert second.skill.registration_message == "Initial registration"
    assert second.version.commit_message == "New version"
    exported = tmp_path / "application"
    await service.export_skill(ref(first, directory=str(exported)))
    assert (exported / "SKILL.md").read_text() == "edited externally"
    await service.switch_skill(ref(first, version_id=first.version.version_id))
    assert (exported / "SKILL.md").read_text() == "edited externally"
    with pytest.raises(SkillConflictError):
        await service.export_skill(ref(first, directory=str(exported)))
    (exported / "stale.txt").write_text("stale")
    await service.export_skill(ref(first, directory=str(exported), overwrite=True))
    assert not (exported / "stale.txt").exists()
    assert (exported / "SKILL.md").read_text() == "original"
    assert (exported / "asset.bin").read_bytes() == b"\0old"
    await service.export_skill(
        ref(first, directory=str(exported), version_id=second.version.version_id, overwrite=True)
    )
    assert (await service.show_skill(ref(first))).skill.current_version_id == first.version.version_id
    assert len(await service.skill_history(ref(first))) == 2
    changed = await service.set_skill_meta(ref(first, name="Renamed", alias="renamed"))
    assert changed.name == "Renamed" and changed.alias == "renamed"
    assert (await service.set_skill_meta(ref(first, clear_alias=True))).alias is None
    await service.delete_skill(ref(first))
    assert exported.exists() and source.exists()
    with pytest.raises(SkillNotFoundError):
        await service.show_skill(ref(first))


@pytest.mark.asyncio
async def test_isolation_aliases_prefixes_and_owned_versions(service):
    first = await register(service)
    second = await register(service, alias="other")
    foreign = await register(service, project="other-project")
    with pytest.raises(SkillConflictError, match="Ambiguous"):
        await service.show_skill({"project_id": "p", "skill_ref": "sk_"})
    for wrong in (second, foreign):
        with pytest.raises(SkillNotFoundError):
            await service.show_skill(ref(first, version_id=wrong.version.version_id))
        with pytest.raises(SkillNotFoundError):
            await service.switch_skill(ref(first, version_id=wrong.version.version_id))
    with pytest.raises(SkillNotFoundError):
        await service.show_skill({"project_id": "other-project", "skill_ref": first.skill.skill_id})
    with pytest.raises(SkillConflictError):
        await register(service)
    with pytest.raises(SkillConflictError):
        await service.set_skill_meta(ref(first, alias=second.skill.skill_id))
    with pytest.raises(ValueError):
        await service.set_skill_meta(ref(first, alias="x", clear_alias=True))
    assert len(await service.list_skills({"project_id": "p"})) == 2
    assert len(await service.list_skills({"project_id": "other-project"})) == 1


@pytest.mark.asyncio
async def test_stale_commit_rolls_back_new_version(service, monkeypatch):
    first = await register(service)
    second = await service.commit_skill(ref(first, source=snapshot("second")))
    with pytest.raises(SkillConflictError):
        await service.commit_skill(ref(first, source=snapshot("third"), base_version_id=first.version.version_id))
    store = SkillDataService(service.backends["persistence"].database)
    with pytest.raises(SkillConflictError):
        store.commit(first.skill, snapshot("stale"), message="must roll back")
    assert len(await service.skill_history(ref(first))) == 2
    assert (await service.show_skill(ref(first))).skill.current_version_id == second.version.version_id
    original = service.repository._source

    def racing_source(source, **kwargs):
        store.switch(second.skill, first.version.version_id)
        return original(source, **kwargs)

    monkeypatch.setattr(service.repository, "_source", racing_source)
    with pytest.raises(SkillConflictError):
        await service.commit_skill(ref(first, source=snapshot("racing")))
    assert len(await service.skill_history(ref(first))) == 2


@pytest.mark.asyncio
async def test_text_binary_diff_and_delete_reference_protection(service):
    first = await register(service, source=snapshot("before\n", {"asset": b"\0a"}))
    second = await service.commit_skill(ref(first, source=snapshot("after\n", {"asset": b"\0b", "added.txt": b"new"})))
    diff = await service.diff_skill(
        ref(first, from_version_id=first.version.version_id, to_version_id=second.version.version_id)
    )
    assert {change.path for change in diff.files} == {"SKILL.md", "asset", "added.txt"}
    assert next(change for change in diff.files if change.path == "asset").binary
    assert "-before" in next(change for change in diff.files if change.path == "SKILL.md").diff
    database = service.backends["persistence"].database
    trace = TrajectoryRecord(
        trajectory_id="t",
        trajectory_hash="hash",
        task_id="task",
        rollout_id="r",
        task_instruction="test",
        env_ref="test",
        skill_bindings=[{"version_id": first.version.version_id}],
    )
    database.insert(TRAJECTORIES_TABLE, [trace.model_dump()])
    with pytest.raises(SkillInUseError):
        await service.delete_skill(ref(first))
    assert len(await service.skill_history(ref(first))) == 2


def test_http_operations_validate_and_delegate(service, tmp_path):
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/skills/register",
            json={"project_id": "p", "name": "HTTP", "alias": "http", "source": snapshot().model_dump(mode="json")},
        )
        assert response.status_code == 200, response.text
        skill = response.json()["skill"]
        reference = {"project_id": "p", "skill_ref": skill["skill_id"]}
        assert client.post("/skills/show", json=reference).status_code == 200
        assert client.post("/skills/list", json={"project_id": "p"}).json()[0]["alias"] == "http"
        assert (
            client.post(
                "/skills/commit", json=reference | {"source": snapshot("next").model_dump(mode="json")}
            ).status_code
            == 200
        )
        assert len(client.post("/skills/history", json=reference).json()) == 2
        assert (
            client.post("/skills/diff", json=reference | {"to_version_id": skill["current_version_id"]}).status_code
            == 200
        )
        assert (
            client.post("/skills/switch", json=reference | {"version_id": skill["current_version_id"]}).status_code
            == 200
        )
        assert client.post("/skills/set-meta", json=reference | {"clear_alias": True}).status_code == 200
        assert (
            client.post("/skills/export", json=reference | {"directory": str(tmp_path / "http-export")}).status_code
            == 200
        )
        assert client.post("/skills/show", json={"project_id": "p", "skill_ref": "missing"}).status_code == 404
        assert (
            client.post(
                "/skills/commit",
                json=reference | {"source": snapshot().model_dump(mode="json"), "base_version_id": "stale"},
            ).status_code
            == 409
        )
        assert client.post("/skills/set-meta", json=reference).status_code == 422
        assert client.post("/skills/register", json={}).status_code == 422
        assert client.post("/skills/delete", json=reference).status_code == 200


@pytest.mark.asyncio
async def test_export_failure_restores_previous_directory(service, tmp_path, monkeypatch):
    import mindmemos_skill.skill.io as skill_io

    first = await register(service)
    target = tmp_path / "export"
    target.mkdir()
    (target / "old").write_text("keep")
    replace = skill_io.os.replace

    def fail_stage(source, destination):
        if Path(source).is_dir() and "-export-" in Path(source).name and Path(destination) == target:
            raise OSError("simulated final rename failure")
        return replace(source, destination)

    monkeypatch.setattr(skill_io.os, "replace", fail_stage)
    with pytest.raises(OSError):
        await service.export_skill(ref(first, directory=str(target), overwrite=True))
    assert (target / "old").read_text() == "keep"
    assert not list(tmp_path.glob(".export-*"))
    with pytest.raises(ValueError):
        await service.export_skill(ref(first, directory=str(tmp_path), overwrite=True))


@pytest.mark.asyncio
async def test_directory_commit_keeps_implementation_configuration(service, tmp_path):
    initial = snapshot()
    initial.implementation = "custom-runtime"
    initial.config = {"routing": "fixed"}
    initial.config_schema_version = 2
    first = await register(service, source=initial)
    directory = tmp_path / "edited"
    directory.mkdir()
    (directory / "SKILL.md").write_text("new content")
    committed = await service.commit_skill(ref(first, source=str(directory)))
    assert committed.version.implementation == "custom-runtime"
    assert committed.version.config == {"routing": "fixed"}
    assert committed.version.config_schema_version == 2
    assert committed.version.content_hash != first.version.content_hash


@pytest.mark.asyncio
async def test_delete_checks_remote_and_parent_references_atomically(service):
    from mindmemos_skill.persistence.db_services.remote import RemoteSkillBindingRecord

    first = await register(service)
    store = service.backends["persistence"]
    store.remote.bind_skill(
        RemoteSkillBindingRecord(
            remote_id="remote", project_id="cloud-project", cloud_skill_id="cloud-skill", skill_id=first.skill.skill_id
        )
    )
    with pytest.raises(SkillInUseError):
        await service.delete_skill(ref(first))
    store.database.delete(REMOTE_SKILL_BINDINGS_TABLE, filters={"skill_id": first.skill.skill_id})
    unrelated = await register(service, alias="dependent")
    dependent = unrelated.version.model_copy(
        update={
            "version_id": "dependent-version",
            "version_label": "v1.0.1",
            "parent_version_ids": [first.version.version_id],
        }
    )
    store.skills.add(dependent)
    with pytest.raises(SkillInUseError):
        await service.delete_skill(ref(first))
    assert (await service.show_skill(ref(first))).version == first.version


@pytest.mark.asyncio
async def test_delete_graph_foreign_key_rolls_back_identity(service):
    from uuid import uuid4

    from mindmemos_skill.persistence import SkillGraphNodeRecord, SkillGraphRecord, SkillGraphVersionRecord

    first = await register(service)
    database = service.backends["persistence"].database
    graph = SkillGraphRecord(project_id="p", graph_id="g", name="Graph")
    graph_version = SkillGraphVersionRecord(project_id="p", graph_id="g", graph_version_id=uuid4())
    node = SkillGraphNodeRecord(
        project_id="p",
        graph_id="g",
        graph_version_id=graph_version.graph_version_id,
        node_id="n",
        name="Node",
        skill_version_id=first.version.version_id,
    )
    database.insert(SKILL_GRAPHS_TABLE, [graph.model_dump()])
    database.insert(SKILL_GRAPH_VERSIONS_TABLE, [graph_version.model_dump()])
    database.insert(SKILL_GRAPH_NODES_TABLE, [node.model_dump()])
    with pytest.raises(SkillInUseError):
        await service.delete_skill(ref(first))
    assert (await service.show_skill(ref(first))).skill == first.skill
