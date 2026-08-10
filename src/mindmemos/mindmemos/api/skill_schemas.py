"""Unified HTTP projections for cloud Skill versions and trajectories."""

from __future__ import annotations

from typing import Literal

from mindmemos_skill.contracts import (
    ContractModel,
    SkillBundle,
    SkillTrajectory,
    SkillTrajectoryListRequest,
    SkillTrajectoryReportRequest,
    SkillTrajectoryReportResult,
    SkillVersionCore,
    SkillVersionStatus,
)
from pydantic import Field


class SkillRegisterRequest(ContractModel):
    operation_id: str = Field(min_length=1)
    version: SkillVersionCore
    bundle: SkillBundle


class SkillRegisterData(ContractModel):
    version: SkillVersionCore


class SkillSummaryData(ContractModel):
    cloud_skill_id: str
    name: str
    latest_version: SkillVersionCore


class SkillListData(ContractModel):
    skills: list[SkillSummaryData]


class SkillVersionsData(ContractModel):
    versions: list[SkillVersionCore]
    next_cursor: str | None = None


class SkillContentData(ContractModel):
    version: SkillVersionCore
    bundle: SkillBundle


class SkillRemoteSyncItem(ContractModel):
    cloud_skill_id: str = Field(min_length=1)
    known_version_revisions: dict[str, int] = Field(default_factory=dict)


class SkillRemoteSyncRequest(ContractModel):
    items: list[SkillRemoteSyncItem] = Field(min_length=1)


class SkillRemoteSyncResultItem(ContractModel):
    cloud_skill_id: str
    versions: list[SkillVersionCore]


class SkillRemoteSyncData(ContractModel):
    items: list[SkillRemoteSyncResultItem]


class SkillVersionStatusRequest(ContractModel):
    status: SkillVersionStatus
    expected_revision: int = Field(ge=0)


class SkillTrajectoryPageData(ContractModel):
    items: list[SkillTrajectory]
    returned_count: int = Field(ge=0)
    next_cursor: str | None = None
    has_more: bool = False


class SkillEvolveRequest(ContractModel):
    operation_id: str = Field(min_length=1)
    cloud_skill_id: str = Field(min_length=1)
    base_version_id: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    mode: Literal["sync", "async"] = "sync"
    reuse_evidence: bool = False
    trajectory_ids: list[str] | None = None


class SkillEvolveData(ContractModel):
    operation_id: str
    evolution_run_id: str
    cloud_skill_id: str
    base_version_id: str
    status: Literal["queued", "succeeded", "no_change", "failed"]
    candidate_version_ids: list[str] = Field(default_factory=list)
    selected_version_id: str | None = None


class MemorySkillTrajectoryRef(ContractModel):
    trajectory_id: str
    trajectory_hash: str
    delivery: Literal["required", "async"]


SkillTrajectoryReportData = SkillTrajectoryReportResult


__all__ = [
    "MemorySkillTrajectoryRef",
    "SkillContentData",
    "SkillEvolveData",
    "SkillEvolveRequest",
    "SkillListData",
    "SkillRegisterData",
    "SkillRegisterRequest",
    "SkillRemoteSyncData",
    "SkillRemoteSyncItem",
    "SkillRemoteSyncRequest",
    "SkillRemoteSyncResultItem",
    "SkillSummaryData",
    "SkillTrajectoryListRequest",
    "SkillTrajectoryPageData",
    "SkillTrajectoryReportData",
    "SkillTrajectoryReportRequest",
    "SkillVersionStatusRequest",
    "SkillVersionsData",
]
