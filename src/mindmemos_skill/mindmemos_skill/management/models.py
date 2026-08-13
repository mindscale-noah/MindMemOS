"""Public local Skill management contracts without persisted head pointers."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ..persistence import SkillRecord, SkillRemoteOperationRecord, SkillSyncStateRecord, SkillVersionStatus


class SnapshotFileRole(StrEnum):
    ALGORITHM = "algorithm"
    SCRIPT = "script"
    REFERENCE = "reference"
    RESOURCE = "resource"


class SnapshotFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    content_hash: str
    byte_size: int = Field(ge=0)
    mode: int | None = None
    media_type: str | None = None
    role: SnapshotFileRole


class SkillSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blob: dict[str, str]
    resources: dict[str, str] = Field(default_factory=dict)
    files: list[SnapshotFile]
    content_hash: str
    local_snapshot_hash: str
    runtime_type: str = "static"
    runtime_schema_version: int = 1
    runtime_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def file_contents(self) -> dict[str, str]:
        return {**self.blob, **self.resources}


class DuplicateAction(StrEnum):
    REUSE = "reuse"
    CREATE_NEW = "create_new"


class PendingSkillOperationType(StrEnum):
    PUSH_VERSION = "push_version"
    REPORT_TRAJECTORY = "report_trajectory"
    EVOLVE = "evolve"
    MERGE = "merge"


class PendingSkillOperationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"


class PendingSkillOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str
    operation_type: PendingSkillOperationType
    skill_id: str | None = None
    cloud_skill_id: str | None = None
    version_id: str | None = None
    trajectory_id: str | None = None
    request_hash: str
    status: PendingSkillOperationStatus = PendingSkillOperationStatus.PENDING
    attempt_count: int = 0
    next_retry_at: datetime | None = None
    lease_expires_at: datetime | None = None
    last_error_code: str | None = None
    remote_result: dict | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: SkillRemoteOperationRecord) -> PendingSkillOperation:
        return cls.model_validate(record.model_dump(mode="json"))


def push_operation_id(skill_id: str, version_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mindmemos:push:{skill_id}:{version_id}"))


def trajectory_report_operation_id(trajectory_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mindmemos:trajectory-report:{trajectory_id}"))


class RegisterSkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    source_path: str | Path
    name: str | None = None
    alias: str | None = None
    version_label: str | None = None
    commit_message: str | None = None
    duplicate_action: DuplicateAction | None = None
    runtime_type: str = "static"
    runtime_schema_version: int = Field(default=1, ge=1)
    runtime_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RegisterSkillResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["created", "reused"]
    skill_id: str
    version_id: str


class PublishSkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    skill_ref: str
    base_version_id: str | None = None
    source_path: str | Path | None = None
    content: str | None = None
    files: dict[str, str] | None = None
    version_label: str | None = None
    commit_message: str | None = None
    runtime_type: str | None = None
    runtime_schema_version: int | None = Field(default=None, ge=1)
    runtime_metadata: dict[str, JsonValue] | None = None


class PublishSkillResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    version_id: str
    local_snapshot_hash: str


class PushResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str
    skill_id: str
    version_id: str
    cloud_skill_id: str
    remote_content_hash: str
    status: SkillVersionStatus


class PullResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    cloud_skill_id: str
    imported_version_ids: list[str] = Field(default_factory=list)
    matched_version_ids: list[str] = Field(default_factory=list)


class ManagedSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    name: str
    description: str | None = None
    alias: str | None = None
    cloud_skill_id: str | None = None
    latest_version_id: str
    latest_version_label: str
    last_version_sync_at: datetime | None = None
    last_trajectory_pull_at: datetime | None = None
    version_count: int
    pending_count: int
    created_at: datetime
    updated_at: datetime


class SkillDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill: ManagedSkill
    latest_version: SkillRecord
    sync_state: SkillSyncStateRecord


class SkillManagementSyncState(StrEnum):
    LOCAL_ONLY = "local_only"
    PENDING = "pending"
    SYNCED = "synced"
    CONFLICT = "conflict"
    FAILED = "failed"


class SkillManagementSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill: ManagedSkill
    sync_state: SkillManagementSyncState


class SkillManagementOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skills: list[SkillManagementSummary] = Field(default_factory=list)
    pending_operations: list[PendingSkillOperation] = Field(default_factory=list)


class SkillManagementDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill: ManagedSkill
    versions: list[SkillRecord] = Field(default_factory=list)
    latest_version: SkillRecord
    pending_operations: list[PendingSkillOperation] = Field(default_factory=list)
    sync_state: SkillManagementSyncState


class ExportSkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    skill_ref: str
    target_path: str | Path
    version_id: str | None = None
    replace: bool = True


class ExportSkillResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    version_id: str
    target_path: str
    exported_files: list[str]
    local_snapshot_hash: str


class SkillDiffResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    from_version_id: str
    to_version_id: str
    diff: str
    changed_files: list[str]


__all__ = [
    "DuplicateAction",
    "ExportSkillRequest",
    "ExportSkillResult",
    "ManagedSkill",
    "PendingSkillOperation",
    "PendingSkillOperationStatus",
    "PendingSkillOperationType",
    "PublishSkillRequest",
    "PublishSkillResult",
    "PullResult",
    "PushResult",
    "RegisterSkillRequest",
    "RegisterSkillResult",
    "SkillDetail",
    "SkillDiffResult",
    "SkillManagementDetail",
    "SkillManagementOverview",
    "SkillManagementSummary",
    "SkillManagementSyncState",
    "SkillSnapshot",
    "SnapshotFile",
    "SnapshotFileRole",
    "push_operation_id",
    "trajectory_report_operation_id",
]
