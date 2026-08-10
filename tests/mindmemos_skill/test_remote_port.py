"""Transport-neutral edge-cloud Skill v2 contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mindmemos_skill import (
    RemoteEvolveRequest,
    RemoteEvolveResult,
    RemotePushRequest,
    RemotePushResult,
    RemoteSyncRequest,
    RemoteSyncResult,
    RemoteTrajectoryListRequest,
    RemoteTrajectoryPage,
    RemoteTrajectoryReportRequest,
    RemoteTrajectoryReportResult,
    RemoteVersionContent,
    RemoteVersionsPage,
    SkillBundle,
    SkillRemotePort,
    SkillVersionCore,
    compute_remote_skill_content_hash,
    deserialize_remote_skill_content,
    serialize_remote_skill_content,
)


def _version() -> SkillVersionCore:
    now = datetime(2026, 8, 7, tzinfo=UTC)
    bundle = SkillBundle.from_files({"SKILL.md": "Demo\n"})
    return SkillVersionCore(
        version_id="version-1",
        cloud_skill_id="cloud-1",
        parent_version_ids=[],
        name="demo",
        content_hash=bundle.content_hash,
        version_label="1.0.0",
        status="draft",
        origin="local",
        created_at=now,
        updated_at=now,
    )


class _Remote:
    async def push_version(self, request: RemotePushRequest) -> RemotePushResult:
        return RemotePushResult(version=request.version)

    async def pull_versions(self, _cloud_skill_id: str, cursor: str | None = None) -> RemoteVersionsPage:
        return RemoteVersionsPage(versions=[_version()], next_cursor=cursor)

    async def pull_content(self, _cloud_skill_id: str, _version_id: str) -> RemoteVersionContent:
        return RemoteVersionContent(version=_version(), bundle=SkillBundle.from_files({"SKILL.md": "Demo\n"}))

    async def sync(self, _request: RemoteSyncRequest) -> RemoteSyncResult:
        return RemoteSyncResult()

    async def report_trajectories(
        self, request: RemoteTrajectoryReportRequest
    ) -> RemoteTrajectoryReportResult:
        return RemoteTrajectoryReportResult(items=[])

    async def list_trajectories(self, _request: RemoteTrajectoryListRequest) -> RemoteTrajectoryPage:
        return RemoteTrajectoryPage()

    async def evolve(self, request: RemoteEvolveRequest) -> RemoteEvolveResult:
        return RemoteEvolveResult(
            operation_id=request.operation_id,
            evolution_run_id="run-1",
            cloud_skill_id=request.cloud_skill_id,
            base_version_id=request.base_version_id,
            status="no_change",
        )

def test_remote_port_is_structural_and_has_no_promote_or_head_contract() -> None:
    assert isinstance(_Remote(), SkillRemotePort)
    assert "promote" not in SkillRemotePort.__dict__
    assert "merge" not in SkillRemotePort.__dict__
    assert "effective_version_id" not in RemoteSyncRequest.model_fields
    assert "published_head_id" not in RemoteSyncRequest.model_fields
    with pytest.raises(ValidationError, match="base_url"):
        RemoteSyncRequest.model_validate({"items": [], "base_url": "https://example.test"})


def test_remote_bundle_is_exactly_canonical_skill_md() -> None:
    content = serialize_remote_skill_content({"SKILL.md": "Demo\r\n"})

    assert deserialize_remote_skill_content(content) == {"SKILL.md": "Demo\n"}
    assert compute_remote_skill_content_hash(content) == SkillBundle.from_files({"SKILL.md": "Demo\n"}).content_hash
    for path in ("scripts/x.py", "references/a.md", "../secret", "/tmp/secret"):
        with pytest.raises(ValueError):
            serialize_remote_skill_content({"SKILL.md": "Demo\n", path: "private"})


@pytest.mark.asyncio
async def test_remote_port_uses_v2_push_sync_and_evolve_dtos() -> None:
    remote: SkillRemotePort = _Remote()
    version = _version()
    bundle = SkillBundle.from_files({"SKILL.md": "Demo\n"})

    pushed = await remote.push_version(
        RemotePushRequest(operation_id="push-1", version=version, bundle=bundle)
    )
    evolved = await remote.evolve(
        RemoteEvolveRequest(
            operation_id="evolve-1",
            cloud_skill_id="cloud-1",
            base_version_id="version-1",
            algorithm="test",
        )
    )
    assert pushed.version == version
    assert evolved.status == "no_change"
