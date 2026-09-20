"""Validated business records; remote and local identities remain independent."""

from typing import Literal

from pydantic import AwareDatetime, Field, JsonValue

from ....common.time_utils import utcnow
from ...typing import RecordModel


class RemoteSkillBindingRecord(RecordModel):
    """Complete insertable record for remote_skill_bindings."""

    remote_id: str = Field(min_length=1)
    """Stable configured remote deployment alias; contains no credentials."""

    project_id: str = Field(min_length=1)
    """Cloud project resolved under the authenticated remote identity."""

    cloud_skill_id: str = Field(min_length=1)
    """Cloud lineage identity returned by the existing service."""

    skill_id: str = Field(min_length=1)
    """Local Skill family associated with this cloud lineage."""

    last_checked_version_id: str | None = None
    """Cloud version ID most recently reported as local_version_id to sync."""

    published_version_id: str | None = None
    """Last observed published head; not a local version ID."""

    gating_status: str | None = None
    """Last gating result returned by the cloud sync endpoint."""

    last_synced_at: AwareDatetime | None = None
    """Last successfully completed sync; not a trajectory-feed cursor."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time this association was established."""

    updated_at: AwareDatetime = Field(default_factory=utcnow)
    """Last change to this association."""


class RemoteVersionBindingRecord(RecordModel):
    """Complete insertable record for remote_version_bindings."""

    remote_id: str = Field(min_length=1)
    """Stable configured remote deployment alias; contains no credentials."""

    project_id: str = Field(min_length=1)
    """Cloud project resolved under the authenticated remote identity."""

    version_id: str = Field(min_length=1)
    """Authoritative cloud version identity; preserved verbatim."""

    cloud_skill_id: str = Field(min_length=1)
    """Cloud lineage owning this remote version."""

    local_version_id: str = Field(min_length=1)
    """Corresponding local skill_versions.version_id."""

    skill_name: str = Field(min_length=1)
    """Cloud name as returned in SkillVersion metadata."""

    content_hash: str = Field(min_length=1)
    """Cloud canonical bundle hash; distinct from the local bundle hash."""

    parent_version_id: str | None = None
    """Single cloud parent; null for a cloud root."""

    version_label: str | None = None
    """Cloud display label; may be null or repeat."""

    status: Literal["observed", "draft", "evaluating", "published", "superseded", "rolled_back"] = ...
    """Lifecycle state returned by the stable cloud contract."""

    origin: Literal["edge", "cloud"] = ...
    """Origin returned by the cloud, independent of local provenance."""

    content: str = Field(min_length=1)
    """Exact cloud bundle text from registration or the content endpoint."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Original cloud creation time."""

    received_at: AwareDatetime = Field(default_factory=utcnow)
    """Time this cloud snapshot was stored locally."""


class RemoteTrajectoryBindingRecord(RecordModel):
    """Complete insertable record for remote_trajectory_bindings."""

    remote_id: str = Field(min_length=1)
    """Stable configured remote deployment alias; contains no credentials."""

    project_id: str = Field(min_length=1)
    """Cloud project resolved under the authenticated remote identity."""

    trajectory_id: str = Field(min_length=1)
    """Local rollout attempt corresponding to one cloud add record."""

    add_record_id: str = Field(min_length=1)
    """Cloud memory/add record identity returned by the service."""

    request_id: str | None = None
    """Cloud request identity, when returned; not an assumed idempotency guarantee."""

    synced_at: AwareDatetime = Field(default_factory=utcnow)
    """Time the cloud add record identity was confirmed."""


class RemoteOperationRecord(RecordModel):
    """Complete insertable record for remote_operations."""

    remote_id: str = Field(min_length=1)
    """Stable configured remote deployment alias; contains no credentials."""

    project_id: str = Field(min_length=1)
    """Cloud project resolved under the authenticated remote identity."""

    operation_id: str = Field(min_length=1)
    """Local durable operation identity."""

    idempotency_key: str = Field(min_length=1)
    """Local deduplication key; the cloud need not accept this key."""

    operation_type: Literal["register_skill", "sync_skill", "add_trace", "evolve_skill"] = ...
    """Existing cloud capability requested by this local operation."""

    local_version_id: str | None = None
    """Optional local input version retained until the operation is resolved."""

    trajectory_id: str | None = None
    """Optional local input trajectory for a memory/add request."""

    cloud_skill_id: str | None = None
    """Target cloud lineage, if already known."""

    request_hash: str = Field(min_length=1)
    """Digest of the immutable credential-free request payload."""

    request_payload: dict[str, JsonValue] = Field(default_factory=dict)
    """Exact business request needed for replay; no authorization headers or secrets."""

    status: Literal["pending", "running", "succeeded", "failed", "uncertain"] = "pending"
    """Uncertain means the remote side effect must be reconciled before replay."""

    attempt_count: int = Field(default=0, ge=0)
    """Number of remote attempts already started."""

    lease_owner: str | None = None
    """Identity of the worker holding this operation."""

    lease_expires_at: AwareDatetime | None = None
    """Expiry of the current processing lease."""

    next_retry_at: AwareDatetime | None = None
    """Next permitted retry after a known retryable failure."""

    last_error: str | None = None
    """Last failure or reconciliation explanation."""

    response_payload: dict[str, JsonValue] = Field(default_factory=dict)
    """Last confirmed cloud result, including returned remote identities."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time the operation was enqueued."""

    updated_at: AwareDatetime = Field(default_factory=utcnow)
    """Last local operation-state change."""
