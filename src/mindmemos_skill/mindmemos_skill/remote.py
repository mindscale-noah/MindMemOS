"""Transport-neutral edge-cloud Skill v2 protocol."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from .contracts import (
    ContractModel,
    SkillBundle,
    SkillTrajectory,
    SkillTrajectoryReportRequest,
    SkillTrajectoryReportResult,
    SkillVersionCore,
    canonical_request_hash,
    parse_skill_bundle,
)

CLOUD_SKILL_ROOT_FILE = "SKILL.md"


class RemotePushRequest(ContractModel):
    operation_id: str = Field(min_length=1)
    cloud_skill_id: str | None = None
    version: SkillVersionCore
    bundle: SkillBundle


class RemotePushResult(ContractModel):
    version: SkillVersionCore

    @property
    def cloud_skill_id(self) -> str:
        assert self.version.cloud_skill_id is not None
        return self.version.cloud_skill_id

    @property
    def version_id(self) -> str:
        return self.version.version_id

    @property
    def content_hash(self) -> str:
        return self.version.content_hash


RemoteVersionSummary = SkillVersionCore


class RemoteVersionContent(ContractModel):
    version: SkillVersionCore
    bundle: SkillBundle


class RemoteVersionsPage(ContractModel):
    versions: list[SkillVersionCore] = Field(default_factory=list)
    next_cursor: str | None = None


class RemoteSyncItem(ContractModel):
    cloud_skill_id: str = Field(min_length=1)
    known_version_revisions: dict[str, int] = Field(default_factory=dict)


class RemoteSyncRequest(ContractModel):
    items: list[RemoteSyncItem] = Field(default_factory=list)


class RemoteSyncResultItem(ContractModel):
    cloud_skill_id: str
    versions: list[SkillVersionCore] = Field(default_factory=list)


class RemoteSyncResult(ContractModel):
    items: list[RemoteSyncResultItem] = Field(default_factory=list)


RemoteTrajectoryReportRequest = SkillTrajectoryReportRequest
RemoteTrajectoryReportResult = SkillTrajectoryReportResult


class RemoteTrajectoryPullRequest(ContractModel):
    cloud_skill_id: str = Field(min_length=1)
    version_id: str | None = None
    since: datetime | None = None
    cursor: str | None = None
    limit: int = Field(default=100, ge=1, le=500)
    status: str | None = None
    min_score: float | None = None
    include_events: bool = True


class RemoteTrajectoryPage(ContractModel):
    items: list[SkillTrajectory] = Field(default_factory=list)
    returned_count: int = Field(default=0, ge=0)
    next_cursor: str | None = None
    has_more: bool = False


class RemoteEvolveRequest(ContractModel):
    operation_id: str = Field(min_length=1)
    cloud_skill_id: str = Field(min_length=1)
    base_version_id: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    mode: Literal["sync", "async"] = "sync"
    reuse_evidence: bool = False
    trajectory_ids: list[str] | None = None


class RemoteEvolveResult(ContractModel):
    operation_id: str
    evolution_run_id: str
    cloud_skill_id: str
    base_version_id: str
    status: Literal["queued", "succeeded", "no_change", "failed"]
    candidate_version_ids: list[str] = Field(default_factory=list)
    selected_version_id: str | None = None


@runtime_checkable
class SkillRemotePort(Protocol):
    async def push_version(self, request: RemotePushRequest) -> RemotePushResult: ...

    async def pull_versions(self, cloud_skill_id: str, cursor: str | None = None) -> RemoteVersionsPage: ...

    async def pull_content(self, cloud_skill_id: str, version_id: str) -> RemoteVersionContent: ...

    async def sync(self, request: RemoteSyncRequest) -> RemoteSyncResult: ...

    async def report_trajectories(
        self,
        request: RemoteTrajectoryReportRequest,
    ) -> RemoteTrajectoryReportResult: ...

    async def pull_trajectories(self, request: RemoteTrajectoryPullRequest) -> RemoteTrajectoryPage: ...

    async def evolve(self, request: RemoteEvolveRequest) -> RemoteEvolveResult: ...

def normalize_remote_skill_bundle(blob: Mapping[str, str]) -> dict[str, str]:
    bundle = SkillBundle.from_files(dict(blob))
    return {item.path: item.content for item in bundle.files}


def serialize_remote_skill_content(blob: Mapping[str, str]) -> str:
    return SkillBundle.from_files(dict(blob)).canonical_json()


def deserialize_remote_skill_content(content: str) -> dict[str, str]:
    bundle = parse_skill_bundle(content)
    return {item.path: item.content for item in bundle.files}


def compute_remote_skill_content_hash(content: str) -> str:
    return parse_skill_bundle(content).content_hash


def is_remote_skill_bundle_path(path: str) -> bool:
    try:
        SkillBundle.from_files({path: ""})
    except ValueError:
        return False
    return path == CLOUD_SKILL_ROOT_FILE


__all__ = [
    "CLOUD_SKILL_ROOT_FILE",
    "RemoteEvolveRequest",
    "RemoteEvolveResult",
    "RemotePushRequest",
    "RemotePushResult",
    "RemoteSyncItem",
    "RemoteSyncRequest",
    "RemoteSyncResult",
    "RemoteSyncResultItem",
    "RemoteTrajectoryPage",
    "RemoteTrajectoryPullRequest",
    "RemoteTrajectoryReportRequest",
    "RemoteTrajectoryReportResult",
    "RemoteVersionContent",
    "RemoteVersionSummary",
    "RemoteVersionsPage",
    "SkillRemotePort",
    "canonical_request_hash",
    "compute_remote_skill_content_hash",
    "deserialize_remote_skill_content",
    "is_remote_skill_bundle_path",
    "normalize_remote_skill_bundle",
    "serialize_remote_skill_content",
]
