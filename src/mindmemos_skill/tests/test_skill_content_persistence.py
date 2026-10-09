"""End-to-end Python writes and reads with complete source content."""

import pytest
import pytest_asyncio
from mindmemos_skill.api.typing import (
    CommitFromLocalRequest,
    CommitRequest,
    ExportRequest,
    FileSkillInfo,
    RegisterFromLocalRequest,
    RegisterRequest,
    ShowRequest,
    StructuredSkillInfo,
)
from mindmemos_skill.common.skill_versions import snapshot_from_version
from mindmemos_skill.errors import SkillNotFoundError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.constants import SKILL_VERSIONS_TABLE, SKILLS_TABLE
from pydantic import ValidationError

from mindmemos_skill import ServiceConfig, create_service


@pytest_asyncio.fixture
async def service(tmp_path):
    config = ServiceConfig.model_validate(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'store.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
        }
    )
    result = await create_service(
        config, project_authorizer=lambda principal, project_id: Scope(fields={"project_id": project_id})
    )
    try:
        with result.request_principal("test"):
            yield result
    finally:
        await result.close()


def plain(content="plain\r\n", **kwargs):
    return StructuredSkillInfo(
        content=content, name="Example", description="Description", standard_format=False, **kwargs
    )


def standard(content="body"):
    return f"---\r\nname: Example\r\ndescription: Description\r\nlicense: MIT\r\n---\r\n{content}\r\n"


@pytest.mark.asyncio
async def test_structured_registration_commit_read_and_export(service, tmp_path):
    registered = await service.register_skill(
        RegisterRequest(project_id="p", skill=plain(resources={"asset.bin": "AP8="}))
    )
    first = await service.show_skill(ShowRequest(project_id="p", version_id=registered.skill_version_id))
    assert first.version.content == "plain\r\n"
    assert first.version.resources == {"asset.bin": "AP8="}
    assert (first.version.content_hash, first.version.skill_hash) == (registered.content_hash, registered.skill_hash)
    second = await service.commit_skill(
        CommitRequest(
            project_id="p",
            skill_id=registered.skill_id,
            skill=plain("new", commit_message="update"),
            inherit_resources=True,
        )
    )
    assert second.version.parent_version_ids == [registered.skill_version_id]
    assert second.version.resources == first.version.resources
    assert second.version.commit_message == "update"
    assert second.skill.active_version_id == second.version.version_id
    path = tmp_path / "export"
    await service.export_skill(ExportRequest(project_id="p", skill_id=registered.skill_id, directory=str(path)))
    assert (path / "SKILL.md").read_bytes() == b"new"
    assert (path / "asset.bin").read_bytes() == b"\x00\xff"
    assert snapshot_from_version(second.version).skill_hash == second.version.skill_hash


@pytest.mark.asyncio
async def test_local_and_structured_inputs_have_identical_hashes(service, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_bytes(standard().encode())
    (source / "binary").write_bytes(b"\x00\xff")
    local = await service.register_from_local(
        RegisterFromLocalRequest(project_id="p", skill=FileSkillInfo(path=str(source)))
    )
    structured = await service.register_skill(
        RegisterRequest(project_id="p", skill=StructuredSkillInfo(content=standard(), resources={"binary": "AP8="}))
    )
    assert (local.content_hash, local.skill_hash) == (structured.content_hash, structured.skill_hash)
    (source / "SKILL.md").write_bytes(standard("updated").encode())
    committed = await service.commit_from_local(
        CommitFromLocalRequest(
            project_id="p", skill_id=local.skill_id, skill=FileSkillInfo(path=str(source)), active=False
        )
    )
    assert committed.version.content == standard("updated")
    assert committed.skill.active_version_id == local.skill_version_id
    assert committed.version.version_label == "v1.0.1"


@pytest.mark.asyncio
async def test_invalid_file_inputs_leave_no_orphan_versions(service, tmp_path):
    path = tmp_path / "missing" / "SKILL.md"
    with pytest.raises(OSError):
        await service.register_from_local(RegisterFromLocalRequest(project_id="p", skill=FileSkillInfo(path=str(path))))
    path.parent.mkdir()
    path.write_text("invalid")
    with pytest.raises(ValidationError):
        await service.register_from_local(RegisterFromLocalRequest(project_id="p", skill=FileSkillInfo(path=str(path))))
    with service.repository.persistence.session(Scope(fields={"project_id": "p"})) as session:
        assert session.database.count(SKILLS_TABLE) == 0
        assert session.database.count(SKILL_VERSIONS_TABLE) == 0


@pytest.mark.asyncio
async def test_cross_project_parent_is_rejected_without_inserting(service):
    first = await service.register_skill(RegisterRequest(project_id="p", skill=plain()))
    other = await service.register_skill(RegisterRequest(project_id="other", skill=plain()))
    with pytest.raises(SkillNotFoundError):
        await service.commit_skill(
            CommitRequest(
                project_id="p", skill_id=first.skill_id, base_version_id=other.skill_version_id, skill=plain("new")
            )
        )
    with service.repository.persistence.session(Scope(fields={"project_id": "p"})) as session:
        assert session.database.count(SKILL_VERSIONS_TABLE) == 1


@pytest.mark.asyncio
async def test_commit_failure_rolls_back_version_pointer_and_metadata(service, monkeypatch):
    from mindmemos_skill.persistence.db_services.skills.service import SkillDataService

    registered = await service.register_skill(RegisterRequest(project_id="p", skill=plain(metadata={"keep": True})))
    insert_child = SkillDataService._insert_child

    def fail_after_insert(self, *args, **kwargs):
        insert_child(self, *args, **kwargs)
        raise RuntimeError("write interrupted")

    monkeypatch.setattr(SkillDataService, "_insert_child", fail_after_insert)
    with pytest.raises(RuntimeError, match="write interrupted"):
        await service.commit_skill(
            CommitRequest(
                project_id="p", skill_id=registered.skill_id, skill=plain("changed", metadata={"keep": False})
            )
        )
    detail = await service.show_skill(ShowRequest(project_id="p", skill_id=registered.skill_id))
    assert detail.version.version_id == registered.skill_version_id
    assert detail.skill.metadata == {"keep": True}
    with service.repository.persistence.session(Scope(fields={"project_id": "p"})) as session:
        assert session.database.count(SKILL_VERSIONS_TABLE) == 1


@pytest.mark.asyncio
async def test_stale_store_commit_leaves_no_orphan_version(service):
    from mindmemos_skill.errors import SkillConflictError

    registered = await service.register_skill(RegisterRequest(project_id="p", skill=plain()))
    first = await service.show_skill(ShowRequest(project_id="p", skill_id=registered.skill_id))
    await service.commit_skill(CommitRequest(project_id="p", skill_id=registered.skill_id, skill=plain("second")))
    with service.repository.persistence.session(Scope(fields={"project_id": "p"})) as session:
        with pytest.raises(SkillConflictError):
            session.skills.commit(first.skill, snapshot_from_version(first.version), message="stale")
        assert session.database.count(SKILL_VERSIONS_TABLE) == 2


@pytest.mark.asyncio
async def test_committed_name_metadata_and_source_are_explicit(service):
    registered = await service.register_skill(RegisterRequest(project_id="p", skill=plain()))
    result = await service.commit_skill(
        CommitRequest(
            project_id="p",
            skill_id=registered.skill_id,
            skill=StructuredSkillInfo(
                content="plain\r\n",
                name="Version name",
                description="Version description",
                standard_format=False,
                metadata={"changed": True},
                commit_source="algo",
            ),
        )
    )
    assert result.skill.name == "Example"
    assert result.version.name == "Version name"
    assert result.version.description == "Version description"
    assert result.skill.metadata == result.version.metadata == {"changed": True}
    assert result.version.origin == "algo"
    assert (result.version.content_hash, result.version.skill_hash) == (registered.content_hash, registered.skill_hash)


@pytest.mark.asyncio
async def test_corrupted_persisted_content_is_rejected_on_read(service, tmp_path):
    import sqlite3

    registered = await service.register_skill(RegisterRequest(project_id="p", skill=plain()))
    with sqlite3.connect(tmp_path / "store.db") as connection:
        connection.execute(
            f"UPDATE {SKILL_VERSIONS_TABLE} SET content = ? WHERE version_id = ?",
            ("tampered", registered.skill_version_id),
        )
    with pytest.raises(ValidationError, match="hash mismatch"):
        await service.show_skill(ShowRequest(project_id="p", version_id=registered.skill_version_id))


@pytest.mark.asyncio
async def test_local_files_are_read_only_after_authorization(service, monkeypatch):
    from mindmemos_skill.pipelines.builtin_pipelines.repository import repository

    def deny(principal, project_id):
        raise PermissionError("denied")

    def unexpected_read(path):
        raise AssertionError("Local file was read before authorization")

    service._project_authorizer = deny
    monkeypatch.setattr(repository, "read_skill", unexpected_read)
    with pytest.raises(PermissionError, match="denied"):
        await service.register_from_local(
            RegisterFromLocalRequest(project_id="p", skill=FileSkillInfo(path="SKILL.md"))
        )


@pytest.mark.asyncio
async def test_invalid_local_commit_preserves_version_count_and_default(service, tmp_path):
    registered = await service.register_skill(RegisterRequest(project_id="p", skill=plain()))
    path = tmp_path / "SKILL.md"
    path.write_text("invalid standard Skill")
    with pytest.raises(ValidationError):
        await service.commit_from_local(
            CommitFromLocalRequest(project_id="p", skill_id=registered.skill_id, skill=FileSkillInfo(path=str(path)))
        )
    with service.repository.persistence.session(Scope(fields={"project_id": "p"})) as session:
        assert session.database.count(SKILL_VERSIONS_TABLE) == 1
        family = session.skills.resolve_family("p", registered.skill_id)
        assert family.active_version_id == registered.skill_version_id
        history = session.skills.history(family, limit=None, offset=0)
        assert history[0].skill_hash == registered.skill_hash
