"""Flat local projections for unified Skill facts and control state."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from ..contracts import SkillRuntimeSpec, parse_skill_bundle
from .enums import AgentType, RolloutType, SkillVersionOrigin, SkillVersionStatus, TrajectoryStatus


def utcnow() -> datetime:
    return datetime.now(UTC)


class PersistenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SkillRecord(PersistenceModel):
    skill_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    cloud_skill_id: str | None = None
    parent_version_ids: list[str] = Field(default_factory=list)
    name: str = Field(min_length=1)
    description: str | None = None
    alias: str | None = None
    bundle: str = Field(min_length=1)
    resources: str = "{}"
    content_hash: str = Field(min_length=1)
    local_snapshot_hash: str = Field(min_length=1)
    runtime_type: str = "static"
    runtime_schema_version: int = Field(default=1, ge=1)
    runtime_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    status: SkillVersionStatus = SkillVersionStatus.DRAFT
    version_revision: int = Field(default=0, ge=0)
    version_label: str = Field(min_length=1)
    commit_message: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    local_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    origin: SkillVersionOrigin = SkillVersionOrigin.LOCAL
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    received_at: datetime | None = None

    @field_validator("bundle")
    @classmethod
    def validate_bundle(cls, value: str) -> str:
        return parse_skill_bundle(value).canonical_json()

    @field_validator("resources")
    @classmethod
    def validate_resources(cls, value: str) -> str:
        _parse_serialized_files(value)
        return value

    @model_validator(mode="after")
    def validate_record(self) -> SkillRecord:
        SkillRuntimeSpec(
            runtime_type=self.runtime_type,
            runtime_schema_version=self.runtime_schema_version,
            runtime_metadata=self.runtime_metadata,
        )
        if self.version_id in self.parent_version_ids:
            raise ValueError("a Skill version cannot be its own parent")
        if len(self.parent_version_ids) != len(set(self.parent_version_ids)):
            raise ValueError("parent_version_ids must be unique and ordered")
        bundle = parse_skill_bundle(self.bundle)
        if bundle.content_hash != self.content_hash.removeprefix("sha256:"):
            raise ValueError("content_hash does not match canonical bundle")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self

    @property
    def blob(self) -> str:
        """Compatibility view for internal snapshot code; not a physical column."""

        bundle = parse_skill_bundle(self.bundle)
        return json.dumps(
            {item.path: item.content for item in bundle.files},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class SkillSyncStateRecord(PersistenceModel):
    skill_id: str = Field(min_length=1)
    last_version_sync_at: datetime | None = None
    trajectory_pull_cursor: str | None = None
    last_trajectory_pull_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_timestamps(self) -> SkillSyncStateRecord:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


# Source compatibility while callers migrate; the physical table is skill_sync_state.
SkillFamilyStateRecord = SkillSyncStateRecord


class SkillRemoteOperationRecord(PersistenceModel):
    operation_id: str = Field(min_length=1)
    operation_type: str = Field(min_length=1)
    skill_id: str | None = None
    cloud_skill_id: str | None = None
    version_id: str | None = None
    trajectory_id: str | None = None
    request_hash: str = Field(min_length=1)
    status: str = Field(min_length=1)
    attempt_count: int = Field(default=0, ge=0)
    lease_expires_at: datetime | None = None
    next_retry_at: datetime | None = None
    last_error_code: str | None = None
    remote_result: dict[str, JsonValue] | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class TrajectoryRecord(PersistenceModel):
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
    env_ref: str = Field(default="unknown", min_length=1)
    running_dir: str | None = None
    env_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    injected_skills: list[dict[str, JsonValue]] = Field(default_factory=list)
    agent_type: AgentType = AgentType.UNKNOWN
    agent_profile: dict[str, Any] = Field(default_factory=dict)
    status: TrajectoryStatus = TrajectoryStatus.RUNNING
    trajectory: list[dict[str, JsonValue]] = Field(default_factory=list)
    skill_bindings: list[dict[str, JsonValue]] = Field(default_factory=list)
    reward_score: float | None = None
    reward_detail: str | None = None
    reward_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    n_turn: int = Field(default=0, ge=0)
    error_info: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    metadata_revision: int = Field(default=0, ge=0)
    metadata_updated_at: datetime | None = None
    source: str = "skill_runtime"
    source_add_record_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    received_at: datetime | None = None

    @model_validator(mode="after")
    def validate_values(self) -> TrajectoryRecord:
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.reward_score is not None and not math.isfinite(self.reward_score):
            raise ValueError("reward_score must be finite")
        return self


class LLMCallRecord(PersistenceModel):
    call_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    call_type: str = Field(pattern=r"^(chat|embedding)$")
    request: dict[str, JsonValue]
    response: dict[str, JsonValue] | None = None
    model: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    status: str = Field(pattern=r"^(succeeded|failed)$")
    error: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime = Field(default_factory=utcnow)
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_timestamps(self) -> LLMCallRecord:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self


class AlgorithmLogRecord(PersistenceModel):
    log_id: str = Field(min_length=1)
    algorithm_name: str = Field(min_length=1)
    algorithm_version: str | None = None
    component_name: str = Field(min_length=1)
    step_name: str = Field(min_length=1)
    status: str | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


def _parse_serialized_files(value: str) -> dict[str, str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("serialized files must be valid JSON") from exc
    if not isinstance(parsed, dict) or any(
        not isinstance(path, str) or not path or not isinstance(content, str) for path, content in parsed.items()
    ):
        raise ValueError("serialized files must map non-empty paths to text")
    return parsed


__all__ = [
    "AgentType",
    "AlgorithmLogRecord",
    "LLMCallRecord",
    "PersistenceModel",
    "RolloutType",
    "SkillFamilyStateRecord",
    "SkillRecord",
    "SkillRemoteOperationRecord",
    "SkillSyncStateRecord",
    "SkillVersionOrigin",
    "SkillVersionStatus",
    "TrajectoryRecord",
    "TrajectoryStatus",
    "utcnow",
]
