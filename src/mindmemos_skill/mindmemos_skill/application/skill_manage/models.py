"""Application-facing domain models for ``skill_manage`` operations.

These carry the durable operation/generation facts that P4/P5 orchestration
and repair reconcile against. They are frozen, storage-neutral DTOs: no
psycopg/SQL/pgvector types appear here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...contracts import canonical_request_hash
from ...typing.skill_manage import (
    EmbeddingProfile,
    RetrievalMode,
    SkillManageRemoveSelector,
    SkillVersionRef,
)
from .state_machine import (
    GenerationStatus,
    OperationStatus,
    OperationType,
    PendingRepairReason,
)


def compute_skill_manage_request_hash(request: BaseModel) -> str:
    """Canonical, secret-free request hash used for operation idempotency."""

    return canonical_request_hash(request)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _PersistWriteRequest(_FrozenModel):
    """Common fields for persistable skill_manage write requests."""

    project_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    expected_revision: int | None = None
    idempotency_key: str | None = Field(default=None, min_length=1)
    commit_policy: Literal["dry_run", "persist"] = "persist"
    algorithm_name: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _dry_run_requires_no_idempotency_key(self) -> _PersistWriteRequest:
        if self.commit_policy == "dry_run" and self.idempotency_key is not None:
            raise ValueError("dry_run requests must not provide an idempotency_key")
        return self


class SkillManageBuildRequest(_PersistWriteRequest):
    """Public request for ``SkillApplication.build_skill_index``.

    A graph holds one node per Skill family (``graph_node_id`` derives from
    ``skill_id``), so two versions of the same family cannot be built together.
    """

    version_refs: tuple[SkillVersionRef, ...] = Field(default_factory=tuple)
    build_mode: Literal["full"] = "full"
    embedding_profile: EmbeddingProfile | None = None
    build_scene: bool = True
    llm_linking: bool = False

    @model_validator(mode="after")
    def _one_version_per_skill(self) -> SkillManageBuildRequest:
        seen: set[str] = set()
        for ref in self.version_refs:
            if ref.skill_id in seen:
                raise ValueError(f"version_refs contains skill_id {ref.skill_id!r} more than once")
            seen.add(ref.skill_id)
        return self


class SkillManageUpsertRequest(_PersistWriteRequest):
    """Public request for ``SkillApplication.upsert_skill_index``."""

    version_ref: SkillVersionRef
    embedding_profile: EmbeddingProfile | None = None


class SkillManageRemoveRequest(_PersistWriteRequest):
    """Public request for ``SkillApplication.remove_skill_from_index``."""

    selector: SkillManageRemoveSelector


class SkillManageRetrieveRequest(_FrozenModel):
    """Public request for ``SkillApplication.retrieve_skills``.

    ``seed_top_k`` bounds the KNN seed list fed to PPR (the GOS ``SEED_TOP_K``
    analogue); ``score_threshold`` applies to the seeds' vector similarity and
    is pushed down to the vector store.
    """

    project_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, gt=0)
    seed_top_k: int = Field(default=5, gt=0)
    mode: RetrievalMode = RetrievalMode.SKILL_ONLY
    score_threshold: float | None = None
    revision: int | None = None
    generation_id: str | None = None
    algorithm_name: str | None = Field(default=None, min_length=1)


class SkillManageRetrieveResult(_FrozenModel):
    """Public result for ``SkillApplication.retrieve_skills``."""

    query: str
    generation_id: str | None = None
    revision: int | None = None
    hits: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    warnings: tuple[dict[str, Any], ...] = Field(default_factory=tuple)


class SkillManageOperation(_FrozenModel):
    """Durable operation record with the full attempt/timing envelope."""

    operation_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    request_hash: str = Field(min_length=1)
    operation_type: OperationType
    status: OperationStatus

    expected_revision: int | None = None
    base_revision: int | None = None
    result_revision: int | None = None
    current_generation_id: str | None = None
    attempt_no: int = Field(default=1, ge=1)

    algorithm_name: str | None = None
    config_hash: str | None = None
    request_summary: dict[str, Any] = Field(default_factory=dict)
    progress_payload: dict[str, Any] = Field(default_factory=dict)
    result_schema_version: int | None = None
    result_payload: dict[str, Any] | None = None

    error_code: str | None = None
    error_message: str | None = None
    error_retryable: bool | None = None
    error_details: dict[str, Any] | None = None
    pending_repair_reason: PendingRepairReason | None = None
    writer_fencing_token: int | None = None

    created_at: datetime
    updated_at: datetime
    first_started_at: datetime | None = None
    attempt_started_at: datetime | None = None
    validating_at: datetime | None = None
    completed_at: datetime | None = None
    repaired_at: datetime | None = None
    repair_count: int = Field(default=0, ge=0)
    last_repair_action: str | None = None


class SkillGraphGeneration(_FrozenModel):
    """Durable generation record with immutable count/hash evidence."""

    generation_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    attempt_no: int = Field(default=1, ge=1)
    base_revision: int | None = None
    target_revision: int | None = None
    status: GenerationStatus

    content_hash: str | None = None
    config_hash: str | None = None
    node_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    vector_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    counts_hash: str | None = None

    failure_code: str | None = None
    failure_message: str | None = None
    failure_retryable: bool | None = None

    created_at: datetime
    ready_at: datetime | None = None
    activated_at: datetime | None = None
    retired_at: datetime | None = None
    failed_at: datetime | None = None
    deleted_at: datetime | None = None


class SkillManageOperationResult(_FrozenModel):
    """Public, secret-free result for a completed write operation.

    ``operation_id`` is ``None`` only for a dry-run result; persist results
    always carry the operation ID.
    """

    operation_id: str | None = None
    status: OperationStatus
    base_revision: int | None = None
    result_revision: int | None = None
    generation_id: str | None = None
    inserted: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    reused: int = Field(default=0, ge=0)
    removed: int = Field(default=0, ge=0)
    node_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    scene_count: int = Field(default=0, ge=0)
    vector_count: int = Field(default=0, ge=0)
    warnings: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    error_code: str | None = None
    error_retryable: bool | None = None


class SkillManageRepairAction(_FrozenModel):
    """One audit entry produced by an explicit repair pass.

    ``operation_unresolved`` means the row was examined but could not be
    converged (head/generation state is ambiguous): it stays ``pending_repair``
    and the report carries a ``SKILL_MANAGE_CONSISTENCY_VIOLATION`` warning
    for manual handling, as spec §9 requires.
    """

    kind: Literal["operation_reconciled", "operation_unresolved", "candidate_discarded", "operation_skipped"]
    operation_id: str | None = None
    generation_id: str | None = None
    from_status: str | None = None
    to_status: str | None = None
    reason: str | None = None


class SkillManageRepairReport(_FrozenModel):
    """Audit-only outcome of ``SkillManageService.repair_skill_index``.

    Repair never advances the canonical version or the head; it only
    converges operation rows and clears stranded candidate generations.
    """

    project_id: str
    graph_id: str
    actions: tuple[SkillManageRepairAction, ...] = Field(default_factory=tuple)
    warnings: tuple[dict[str, Any], ...] = Field(default_factory=tuple)


class SkillManageMaintenanceReport(_FrozenModel):
    """Outcome of an explicit ``SkillManageService.maintain_skill_index`` GC pass."""

    project_id: str
    graph_id: str
    removed_generation_ids: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[dict[str, Any], ...] = Field(default_factory=tuple)


class SkillManageServiceSettings(_FrozenModel):
    """Write-path timing and retention policy owned by :class:`SkillManageService`.

    Mirrors the timing/retention fields of the algorithm's ``SkillManageConfig``
    so production composition can copy them across without the application
    importing the algorithm package.
    """

    operation_wait_timeout_seconds: float = Field(default=30.0, gt=0)
    operation_poll_interval_seconds: float = Field(default=0.1, gt=0)
    writer_lease_ttl_seconds: float = Field(default=60.0, gt=0)
    writer_lease_renew_interval_seconds: float = Field(default=20.0, gt=0)
    retired_generations_to_keep: int = Field(default=2, ge=0)
    retired_generation_min_age_seconds: float = Field(default=86400.0, ge=0)

    @model_validator(mode="after")
    def _validate_timing_relationships(self) -> SkillManageServiceSettings:
        if self.operation_poll_interval_seconds >= self.operation_wait_timeout_seconds:
            raise ValueError("operation_poll_interval_seconds must be less than operation_wait_timeout_seconds")
        if self.writer_lease_renew_interval_seconds >= self.writer_lease_ttl_seconds / 2:
            raise ValueError("writer_lease_renew_interval_seconds must be less than writer_lease_ttl_seconds / 2")
        return self

    @classmethod
    def from_algorithm_config(cls, config: BaseModel) -> SkillManageServiceSettings:
        """Copy the matching fields from a validated algorithm config; missing ones keep defaults."""

        values = {name: getattr(config, name) for name in cls.model_fields if hasattr(config, name)}
        return cls(**values)


__all__ = [
    "SkillGraphGeneration",
    "SkillManageBuildRequest",
    "SkillManageMaintenanceReport",
    "SkillManageOperation",
    "SkillManageOperationResult",
    "SkillManageRemoveRequest",
    "SkillManageRepairAction",
    "SkillManageRepairReport",
    "SkillManageRetrieveRequest",
    "SkillManageRetrieveResult",
    "SkillManageServiceSettings",
    "SkillManageUpsertRequest",
    "compute_skill_manage_request_hash",
]
