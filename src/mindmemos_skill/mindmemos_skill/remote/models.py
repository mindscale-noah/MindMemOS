"""Remote Skill request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..contracts import (
    ContractModel,
    SkillBundle,
    SkillTrajectory,
    SkillTrajectoryListRequest,
    SkillTrajectoryReportRequest,
    SkillTrajectoryReportResult,
    SkillVersionCore,
)


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
RemoteTrajectoryListRequest = SkillTrajectoryListRequest


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
