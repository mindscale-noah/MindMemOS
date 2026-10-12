"""Repository integration across local files, snapshots, and project boundaries."""

import pytest
import pytest_asyncio
from mindmemos_skill.api.typing import (
    ActiveRequest,
    CommitFromLocalRequest,
    CommitRequest,
    FileSkillInfo,
    ListTrajectoriesRequest,
    RegisterFromLocalRequest,
    RegisterRequest,
    ReportTrajectoryRequest,
    ShowRequest,
    ShowTrajectoryRequest,
    StructuredSkillInfo,
)
from mindmemos_skill.errors import SkillNotFoundError, SkillVersionNotFoundError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.skill import Skill, SkillContent, decode_resources

from mindmemos_skill import ServiceConfig, create_service

from .skill_sources import register_detail


@pytest_asyncio.fixture
async def service(tmp_path):
    config = ServiceConfig.model_validate(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'central.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
        }
    )
    result = await create_service(
        config, project_authorizer=lambda principal, project_id: Scope(fields={"project_id": project_id})
    )
    try:
        with result.request_principal("test-user"):
            yield result
    finally:
        await result.close()


def info(content="first", **options):
    return StructuredSkillInfo(
        content=content, name="Example", description="Instructions", standard_format=False, **options
    )


async def register(service, *, project="p", alias="example", skill=None):
    return await register_detail(service, RegisterRequest(project_id=project, alias=alias, skill=skill or info()))


def ref(detail, **extra):
    return {"project_id": detail.skill.project_id, "skill_id": detail.skill.skill_id, **extra}


@pytest.mark.asyncio
async def test_external_directories_are_untracked_and_export_is_explicit(service, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    header = "---\r\nname: Example\r\ndescription: Instructions\r\n---\r\n"
    (source / "SKILL.md").write_bytes((header + "original").encode())
    (source / "asset.bin").write_bytes(b"\0old")
    registered = await service.register_from_local(
        RegisterFromLocalRequest(
            project_id="p",
            alias="example",
            skill=FileSkillInfo(path=str(source), commit_message="Initial registration"),
        )
    )
    first = await service.show_skill(ShowRequest(project_id="p", version_id=registered.skill_version_id))
    (source / "SKILL.md").write_bytes((header + "edited externally").encode())
    assert (await service.show_skill(ref(first))).version.content == header + "original"
    second = await service.commit_from_local(
        CommitFromLocalRequest(
            **ref(first),
            skill=FileSkillInfo(path=str(source), commit_message="New version"),
        )
    )
    assert second.version.parent_version_ids == [first.version.version_id]
    assert second.version.commit_message == "New version"
    exported = tmp_path / "application"
    await service.export_skill(ref(first, directory=str(exported)))
    await service.active_skill(ActiveRequest(project_id="p", version_id=first.version.version_id))
    assert (exported / "SKILL.md").read_bytes() == (header + "edited externally").encode()
    with pytest.raises(ValueError, match="not empty"):
        await service.export_skill(ref(first, directory=str(exported)))
    (exported / "stale.txt").write_text("stale")
    await service.export_skill(ref(first, directory=str(exported), overwrite=True))
    assert not (exported / "stale.txt").exists()
    assert (exported / "SKILL.md").read_bytes() == (header + "original").encode()
    assert (exported / "asset.bin").read_bytes() == b"\0old"
    assert len(await service.skill_history(ref(first))) == 2


@pytest.mark.asyncio
async def test_historical_branch_inherits_only_selected_parent_resources(service):
    first = await register(service, skill=info(resources={"asset.bin": "Zmlyc3Q="}))
    current = await service.commit_skill(
        CommitRequest(**ref(first), skill=info("second", resources={"asset.bin": "c2Vjb25k"}))
    )
    branch = await service.commit_skill(
        CommitRequest(
            **ref(first),
            skill=info("historical child", commit_version="v1.1.0"),
            base_version_id=first.version.version_id,
            inherit_resources=True,
            active=False,
        )
    )
    assert branch.version.parent_version_ids == [first.version.version_id]
    assert branch.skill.active_version_id == current.version.version_id
    assert decode_resources(branch.version.resources) == {"asset.bin": b"first"}
    other = await register(service, alias="other")
    with pytest.raises(SkillNotFoundError):
        await service.commit_skill(
            CommitRequest(**ref(first), skill=info("foreign"), base_version_id=other.version.version_id)
        )


@pytest.mark.asyncio
async def test_inactive_commit_keeps_captured_parent_after_active_moves(service):
    first = await register(service)
    current = await service.commit_skill(CommitRequest(**ref(first), skill=info("new active")))
    with service.repository.persistence.session(Scope(fields={"project_id": "p"})) as store:
        version_id = store.skills.new_version_id()
        family = await store.skills.commit(
            first.skill,
            Skill(
                SkillContent(
                    content="historical child",
                    name="Example",
                    description="Instructions",
                    standard_format=False,
                )
            ).snapshot(),
            message="Historical branch",
            active=False,
            version_id=version_id,
        )
        child = await store.skills.version_by_id("p", version_id)
    assert family.active_version_id == current.version.version_id
    assert child.parent_version_ids == [first.version.version_id]


@pytest.mark.asyncio
async def test_export_failure_restores_previous_directory(service, tmp_path, monkeypatch):
    import mindmemos_skill.skill.io as skill_io

    first = await register(service)
    target = tmp_path / "export"
    target.mkdir()
    (target / "old").write_text("keep")
    replace = skill_io.os.replace

    def fail_stage(source, destination):
        if "-export-" in str(source) and destination == target:
            raise OSError("simulated final rename failure")
        return replace(source, destination)

    monkeypatch.setattr(skill_io.os, "replace", fail_stage)
    with pytest.raises(OSError):
        await service.export_skill(ref(first, directory=str(target), overwrite=True))
    assert (target / "old").read_text() == "keep"
    with pytest.raises(ValueError):
        await service.export_skill(ref(first, directory=str(tmp_path), overwrite=True))


@pytest.mark.asyncio
async def test_directory_commit_keeps_implementation_configuration(service, tmp_path):
    first = await register(service)
    source = Skill(
        SkillContent(content="first", name="Example", description="Instructions", standard_format=False)
    ).snapshot()
    source.implementation = "custom-runtime"
    source.config = {"routing": "fixed"}
    source.config_schema_version = 2
    with service.repository.persistence.session(Scope(fields={"project_id": "p"})) as store:
        (await store.skills.commit(first.skill, source, message="Runtime settings"))
    directory = tmp_path / "edited"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: Example\ndescription: Instructions\n---\nnew content")
    committed = await service.commit_from_local(
        CommitFromLocalRequest(**ref(first), skill=FileSkillInfo(path=str(directory)))
    )
    assert committed.version.implementation == "custom-runtime"
    assert committed.version.config == {"routing": "fixed"}
    assert committed.version.config_schema_version == 2
    assert committed.version.content_hash != first.version.content_hash


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id,agent_id", [(None, None), ("user", None), (None, "agent"), ("user", "agent")])
async def test_reported_trajectory_roundtrip_allows_optional_actors(service, user_id, agent_id):
    """Persist, read and list reports without requiring either actor identity."""
    first = await register(service)
    report = await service.report_trajectory(
        ReportTrajectoryRequest(
            project_id="p",
            skill_version_id=first.version.version_id,
            user_id=user_id,
            agent_id=agent_id,
            messages=[{"role": "user", "content": "Question"}],
        )
    )
    restored = await service.show_trajectory(ShowTrajectoryRequest(project_id="p", trajectory_id=report.trajectory_id))
    summaries = await service.list_trajectories(ListTrajectoriesRequest(project_id="p", skill_id=first.skill.skill_id))
    assert restored == report
    assert len(summaries) == 1
    assert report.user_id == summaries[0].user_id == user_id
    assert report.agent_id == summaries[0].agent_id == agent_id
    assert summaries[0].skill_version_id == first.version.version_id
    assert summaries[0].received_at == report.received_at


@pytest.mark.asyncio
async def test_report_trajectory_rejects_cross_project_version_and_read(service):
    first = await register(service)
    request = ReportTrajectoryRequest(
        project_id="p",
        skill_version_id=first.version.version_id,
        user_id="user",
        agent_id="agent",
        messages=[{"role": "user", "content": "Question"}],
    )
    report = await service.report_trajectory(request)
    assert report.skill_version_id == first.version.version_id
    assert (
        await service.show_trajectory(ShowTrajectoryRequest(project_id="p", trajectory_id=report.trajectory_id))
        == report
    )
    with pytest.raises(SkillVersionNotFoundError):
        await service.report_trajectory(request.model_copy(update={"project_id": "other"}))
    with pytest.raises(SkillNotFoundError):
        await service.show_trajectory(ShowTrajectoryRequest(project_id="other", trajectory_id=report.trajectory_id))
