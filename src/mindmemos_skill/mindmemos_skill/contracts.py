"""Shared edge-cloud Skill facts and canonical wire helpers.

This module is deliberately transport and persistence neutral.  Local SQLite,
SDK HTTP adapters and the cloud relational repository all project these same
facts without importing one another's storage types.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

_SHA256_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_SECRET_KEY_PATTERN = re.compile(
    r"(?:authorization|cookie|api[_-]?key|access[_-]?token|refresh[_-]?token|password|credential|secret)",
    re.IGNORECASE,
)
_ABSOLUTE_PATH_PATTERN = re.compile(r"(?:^|[\s\"'])(?:/[A-Za-z0-9_.-]+/|[A-Za-z]:[\\/])")


class ContractModel(BaseModel):
    """Strict immutable base for shared facts and wire payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SkillVersionStatus(StrEnum):
    DRAFT = "draft"
    REJECTED = "rejected"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class SkillVersionOrigin(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"
    EVOLUTION = "evolution"
    MERGE = "merge"


class TrajectorySource(StrEnum):
    SKILL_RUNTIME = "skill_runtime"
    MEMORY_ADD = "memory_add"


class TrajectoryStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RolloutType(StrEnum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    TEST = "test"
    INFERENCE = "inference"


class AgentType(StrEnum):
    CLAUDE = "claude"
    CLAUDE_SDK = "claude_sdk"
    REACT = "react"
    CODEX = "codex"
    OPENCLAW = "openclaw"
    OPENCODE = "opencode"
    GEMINI_CLI = "gemini_cli"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class SkillUsageType(StrEnum):
    INJECTED = "injected"
    MODIFIED = "modified"
    UNUSED = "unused"


class SkillInjectionMode(StrEnum):
    TOOL = "tool"
    SYSTEM_PROMPT = "system_prompt"
    FILESYSTEM = "filesystem"


class SkillRemoteOperationType(StrEnum):
    PUSH_VERSION = "push_version"
    REPORT_TRAJECTORY = "report_trajectory"
    EVOLVE = "evolve"
    MERGE = "merge"


class SkillRemoteOperationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    QUEUED = "queued"
    SUCCEEDED = "succeeded"
    NO_CHANGE = "no_change"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SkillBundleFile(ContractModel):
    path: str = Field(min_length=1)
    content: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("Skill bundle paths must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"invalid Skill bundle path: {value}")
        if path.as_posix() != value:
            raise ValueError(f"Skill bundle path is not canonical: {value}")
        return value

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        return normalized.rstrip("\n") + "\n"


class SkillBundle(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    files: list[SkillBundleFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_files(self) -> SkillBundle:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Skill bundle paths must be unique")
        if self.schema_version == 1 and paths != ["SKILL.md"]:
            raise ValueError("Skill bundle schema v1 requires exactly one SKILL.md")
        return self

    @classmethod
    def from_files(cls, files: dict[str, str]) -> SkillBundle:
        return cls(files=[SkillBundleFile(path=path, content=content) for path, content in sorted(files.items())])

    def canonical_dict(self) -> dict[str, JsonValue]:
        return {
            "files": [item.model_dump(mode="json") for item in sorted(self.files, key=lambda item: item.path)],
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def parse_skill_bundle(value: SkillBundle | str | dict[str, Any]) -> SkillBundle:
    if isinstance(value, SkillBundle):
        return value
    payload = json.loads(value) if isinstance(value, str) else value
    bundle = SkillBundle.model_validate(payload)
    if isinstance(value, str) and value != bundle.canonical_json():
        raise ValueError("Skill bundle is not canonically serialized")
    return bundle


class SkillVersionCore(ContractModel):
    version_id: str = Field(min_length=1)
    cloud_skill_id: str | None = None
    parent_version_ids: list[str] = Field(default_factory=list)
    name: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    version_label: str = Field(min_length=1)
    commit_message: str | None = None
    status: SkillVersionStatus = SkillVersionStatus.DRAFT
    version_revision: int = Field(default=0, ge=0)
    origin: SkillVersionOrigin
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    received_at: datetime | None = None

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("content_hash must be a SHA-256 digest")
        return value.removeprefix("sha256:")

    @model_validator(mode="after")
    def validate_version(self) -> SkillVersionCore:
        if self.version_id in self.parent_version_ids:
            raise ValueError("a Skill version cannot be its own parent")
        if len(self.parent_version_ids) != len(set(self.parent_version_ids)):
            raise ValueError("parent_version_ids must be unique and ordered")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class SkillTrajectoryBinding(ContractModel):
    name: str = Field(min_length=1)
    cloud_skill_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    base_version_id: str | None = None
    content_hash: str = Field(min_length=1)
    version_label: str | None = None
    usage: SkillUsageType
    injection_mode: SkillInjectionMode | None = None

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("binding content_hash must be a SHA-256 digest")
        return value.removeprefix("sha256:")


class SkillTrajectory(ContractModel):
    trajectory_id: str = Field(min_length=1)
    trajectory_hash: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    attempt_no: int = Field(default=0, ge=0)
    rollout_type: RolloutType = RolloutType.INFERENCE
    task_instruction: str = Field(min_length=1)
    task_system_prompt: str | None = None
    task_tags: list[str] = Field(default_factory=list)
    task_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    env_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    agent_type: AgentType = AgentType.UNKNOWN
    agent_profile: dict[str, JsonValue] = Field(default_factory=dict)
    status: TrajectoryStatus
    trajectory: list[dict[str, JsonValue]] = Field(default_factory=list)
    skill_bindings: list[SkillTrajectoryBinding] = Field(default_factory=list)
    reward_score: float | None = None
    reward_detail: str | None = None
    reward_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime | None = None
    n_turn: int = Field(default=0, ge=0)
    error_info: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    metadata_revision: int = Field(default=0, ge=0)
    metadata_updated_at: datetime | None = None
    source: TrajectorySource
    source_add_record_id: str | None = None
    created_at: datetime
    received_at: datetime | None = None

    @field_validator("trajectory_hash")
    @classmethod
    def validate_trajectory_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("trajectory_hash must be a SHA-256 digest")
        return value.removeprefix("sha256:")

    @model_validator(mode="after")
    def validate_trajectory(self) -> SkillTrajectory:
        if self.status is not TrajectoryStatus.RUNNING and self.finished_at is None:
            raise ValueError("terminal trajectories require finished_at")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        identities = [(item.cloud_skill_id, item.version_id, item.usage) for item in self.skill_bindings]
        if len(identities) != len(set(identities)):
            raise ValueError("trajectory bindings must be unique by family, version and usage")
        if self.trajectory_hash != compute_trajectory_hash(self):
            raise ValueError("trajectory_hash does not match canonical source facts")
        return self


_TRAJECTORY_DERIVED_FIELDS = frozenset(
    {"trajectory_hash", "metadata_revision", "metadata_updated_at", "received_at"}
)


def trajectory_source_payload(value: SkillTrajectory | dict[str, Any]) -> dict[str, JsonValue]:
    payload = value.model_dump(mode="json") if isinstance(value, SkillTrajectory) else dict(value)
    for field in _TRAJECTORY_DERIVED_FIELDS:
        payload.pop(field, None)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata.pop("summaries", None)
        payload["metadata"] = metadata
    return payload


def compute_trajectory_hash(value: SkillTrajectory | dict[str, Any]) -> str:
    payload = trajectory_source_payload(value)
    normalized = _SkillTrajectorySourceFacts.model_validate(payload).model_dump(mode="json")
    canonical = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _SkillTrajectorySourceFacts(ContractModel):
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    attempt_no: int = Field(default=0, ge=0)
    rollout_type: RolloutType = RolloutType.INFERENCE
    task_instruction: str = Field(min_length=1)
    task_system_prompt: str | None = None
    task_tags: list[str] = Field(default_factory=list)
    task_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    env_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    agent_type: AgentType = AgentType.UNKNOWN
    agent_profile: dict[str, JsonValue] = Field(default_factory=dict)
    status: TrajectoryStatus
    trajectory: list[dict[str, JsonValue]] = Field(default_factory=list)
    skill_bindings: list[SkillTrajectoryBinding] = Field(default_factory=list, max_length=64)
    reward_score: float | None = None
    reward_detail: str | None = None
    reward_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime | None = None
    n_turn: int = Field(default=0, ge=0)
    error_info: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    source: TrajectorySource
    source_add_record_id: str | None = None
    created_at: datetime


class SkillTrajectoryUpload(_SkillTrajectorySourceFacts):
    """Client-controlled source facts; server-controlled fields are absent."""

    trajectory_hash: str = Field(min_length=1)
    skill_bindings: list[SkillTrajectoryBinding] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_upload(self) -> SkillTrajectoryUpload:
        if self.status is TrajectoryStatus.RUNNING:
            raise ValueError("cloud trajectory upload requires a terminal status")
        if self.finished_at is None:
            raise ValueError("terminal trajectory uploads require finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if "summaries" in self.metadata:
            raise ValueError("clients may not upload metadata.summaries")
        if self.trajectory_hash.removeprefix("sha256:") != compute_trajectory_hash(self.model_dump(mode="json")):
            raise ValueError("trajectory_hash does not match canonical source facts")
        return self


class SkillTrajectoryUploadItem(ContractModel):
    trajectory: SkillTrajectoryUpload


class SkillTrajectoryReportRequest(ContractModel):
    operation_id: str = Field(min_length=1)
    mode: Literal["sync", "async"] = "sync"
    items: list[SkillTrajectoryUploadItem] = Field(min_length=1, max_length=100)


class SkillTrajectoryListRequest(ContractModel):
    cloud_skill_id: str = Field(min_length=1)
    version_id: str | None = None
    since: datetime | None = None
    cursor: str | None = None
    limit: int = Field(default=100, ge=1, le=500)
    status: str | None = None
    min_score: float | None = None
    include_events: bool = True


class SkillTrajectoryReportResultItem(ContractModel):
    trajectory_id: str
    status: Literal["stored", "duplicate", "queued", "rejected"]
    error_code: str | None = None


class SkillTrajectoryReportResult(ContractModel):
    items: list[SkillTrajectoryReportResultItem]


class SkillRemoteOperation(ContractModel):
    operation_id: str = Field(min_length=1)
    operation_type: SkillRemoteOperationType
    cloud_skill_id: str | None = None
    version_id: str | None = None
    trajectory_id: str | None = None
    request_hash: str = Field(min_length=1)
    status: SkillRemoteOperationStatus
    result: dict[str, JsonValue] | None = None
    error_code: str | None = None
    created_at: datetime
    updated_at: datetime


def canonical_request_hash(value: BaseModel | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class TrajectorySanitizer:
    """Reject secret-bearing/path-bearing trajectory uploads before hashing."""

    def __init__(
        self,
        *,
        max_events: int = 2_000,
        max_event_chars: int = 100_000,
        max_trajectory_bytes: int = 5 * 1024 * 1024,
        max_metadata_bytes: int = 256 * 1024,
    ) -> None:
        self.max_events = max_events
        self.max_event_chars = max_event_chars
        self.max_trajectory_bytes = max_trajectory_bytes
        self.max_metadata_bytes = max_metadata_bytes

    def validate(self, upload: SkillTrajectoryUpload) -> SkillTrajectoryUpload:
        if len(upload.trajectory) > self.max_events:
            raise ValueError("trajectory contains too many events")
        for index, event in enumerate(upload.trajectory):
            encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            if len(encoded) > self.max_event_chars:
                raise ValueError(f"trajectory event {index} is too large")
        payload = upload.model_dump(mode="json")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if len(encoded.encode("utf-8")) > self.max_trajectory_bytes:
            raise ValueError("trajectory payload is too large")
        metadata = json.dumps(upload.metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if len(metadata.encode("utf-8")) > self.max_metadata_bytes:
            raise ValueError("trajectory metadata is too large")
        self._scan(payload, path="trajectory")
        return upload

    def _scan(self, value: Any, *, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if _SECRET_KEY_PATTERN.search(str(key)):
                    if isinstance(item, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", item):
                        continue
                    raise ValueError(f"secret-bearing field is not allowed: {path}.{key}")
                if key in {"running_dir", "injected_skills"}:
                    raise ValueError(f"local-only field is not allowed: {path}.{key}")
                self._scan(item, path=f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._scan(item, path=f"{path}[{index}]")
        elif isinstance(value, str) and _ABSOLUTE_PATH_PATTERN.search(value):
            raise ValueError(f"absolute local path is not allowed: {path}")


__all__ = [
    "AgentType",
    "ContractModel",
    "RolloutType",
    "SkillBundle",
    "SkillBundleFile",
    "SkillInjectionMode",
    "SkillRemoteOperation",
    "SkillRemoteOperationStatus",
    "SkillRemoteOperationType",
    "SkillTrajectory",
    "SkillTrajectoryBinding",
    "SkillTrajectoryListRequest",
    "SkillTrajectoryReportRequest",
    "SkillTrajectoryReportResult",
    "SkillTrajectoryReportResultItem",
    "SkillTrajectoryUpload",
    "SkillTrajectoryUploadItem",
    "SkillUsageType",
    "SkillVersionCore",
    "SkillVersionOrigin",
    "SkillVersionStatus",
    "TrajectorySanitizer",
    "TrajectorySource",
    "TrajectoryStatus",
    "canonical_request_hash",
    "compute_trajectory_hash",
    "parse_skill_bundle",
    "trajectory_source_payload",
]
