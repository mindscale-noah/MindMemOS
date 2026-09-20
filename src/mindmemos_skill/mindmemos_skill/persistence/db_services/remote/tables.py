"""Explicit SQLAlchemy business tables consumed by the infra registry."""

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)

from ....infra.database import TableSpec
from ...constants import (
    REMOTE_OPERATIONS_TABLE,
    REMOTE_SKILL_BINDINGS_TABLE,
    REMOTE_TRAJECTORY_BINDINGS_TABLE,
    REMOTE_VERSION_BINDINGS_TABLE,
    SKILL_VERSIONS_TABLE,
    TRAJECTORIES_TABLE,
)
from ...typing import UTCDateTime
from .typing import (
    RemoteOperationRecord,
    RemoteSkillBindingRecord,
    RemoteTrajectoryBindingRecord,
    RemoteVersionBindingRecord,
)


def build_tables(metadata: MetaData) -> tuple[TableSpec, ...]:
    """Declare tables without opening a connection.

    Args:
        metadata: Shared metadata containing local and remote tables.

    Returns:
        Validated table registrations.
    """
    specs = []
    table = Table(
        REMOTE_SKILL_BINDINGS_TABLE,
        metadata,
        Column(
            "remote_id",
            Text,
            primary_key=True,
            comment="Stable configured remote deployment alias; contains no credentials.",
        ),
        Column(
            "project_id",
            Text,
            primary_key=True,
            comment="Cloud project resolved under the authenticated remote identity.",
        ),
        Column(
            "cloud_skill_id", Text, primary_key=True, comment="Cloud lineage identity returned by the existing service."
        ),
        Column("skill_id", Text, nullable=False, comment="Local Skill family associated with this cloud lineage."),
        Column(
            "last_checked_version_id",
            Text,
            nullable=True,
            comment="Cloud version ID most recently reported as local_version_id to sync.",
        ),
        Column(
            "published_version_id", Text, nullable=True, comment="Last observed published head; not a local version ID."
        ),
        Column("gating_status", Text, nullable=True, comment="Last gating result returned by the cloud sync endpoint."),
        Column(
            "last_synced_at",
            UTCDateTime(),
            nullable=True,
            comment="Last successfully completed sync; not a trajectory-feed cursor.",
        ),
        Column("created_at", UTCDateTime(), nullable=False, comment="Time this association was established."),
        Column("updated_at", UTCDateTime(), nullable=False, comment="Last change to this association."),
        UniqueConstraint("remote_id", "project_id", "skill_id"),
    )
    specs.append(
        TableSpec(
            name=REMOTE_SKILL_BINDINGS_TABLE,
            table=table,
            insert_model=RemoteSkillBindingRecord,
        )
    )
    table = Table(
        REMOTE_VERSION_BINDINGS_TABLE,
        metadata,
        Column(
            "remote_id",
            Text,
            primary_key=True,
            comment="Stable configured remote deployment alias; contains no credentials.",
        ),
        Column(
            "project_id",
            Text,
            primary_key=True,
            comment="Cloud project resolved under the authenticated remote identity.",
        ),
        Column(
            "version_id", Text, primary_key=True, comment="Authoritative cloud version identity; preserved verbatim."
        ),
        Column(
            "cloud_skill_id", Text, nullable=False, comment="Cloud lineage identity returned by the existing service."
        ),
        Column("local_version_id", Text, nullable=False, comment="Corresponding local skill_versions.version_id."),
        Column("skill_name", Text, nullable=False, comment="Cloud name as returned in SkillVersion metadata."),
        Column(
            "content_hash",
            Text,
            nullable=False,
            comment="Cloud canonical bundle hash; distinct from the local bundle hash.",
        ),
        Column("parent_version_id", Text, nullable=True, comment="Single cloud parent; null for a cloud root."),
        Column("version_label", Text, nullable=True, comment="Cloud display label; may be null or repeat."),
        Column("status", Text, nullable=False, comment="Lifecycle state returned by the stable cloud contract."),
        Column(
            "origin", Text, nullable=False, comment="Origin returned by the cloud, independent of local provenance."
        ),
        Column(
            "content",
            Text,
            nullable=False,
            comment="Exact cloud bundle text from registration or the content endpoint.",
        ),
        Column("created_at", UTCDateTime(), nullable=False, comment="Time this association was established."),
        Column("received_at", UTCDateTime(), nullable=False, comment="Time this cloud snapshot was stored locally."),
        ForeignKeyConstraint(
            ["remote_id", "project_id", "cloud_skill_id"],
            [
                f"{REMOTE_SKILL_BINDINGS_TABLE}.remote_id",
                f"{REMOTE_SKILL_BINDINGS_TABLE}.project_id",
                f"{REMOTE_SKILL_BINDINGS_TABLE}.cloud_skill_id",
            ],
        ),
        ForeignKeyConstraint(["local_version_id"], [f"{SKILL_VERSIONS_TABLE}.version_id"]),
        UniqueConstraint("remote_id", "project_id", "local_version_id"),
        Index("remote_versions_family_idx", "remote_id", "project_id", "cloud_skill_id", "created_at"),
    )
    specs.append(
        TableSpec(
            name=REMOTE_VERSION_BINDINGS_TABLE,
            table=table,
            insert_model=RemoteVersionBindingRecord,
        )
    )
    table = Table(
        REMOTE_TRAJECTORY_BINDINGS_TABLE,
        metadata,
        Column(
            "remote_id",
            Text,
            primary_key=True,
            comment="Stable configured remote deployment alias; contains no credentials.",
        ),
        Column(
            "project_id",
            Text,
            primary_key=True,
            comment="Cloud project resolved under the authenticated remote identity.",
        ),
        Column(
            "trajectory_id",
            Text,
            primary_key=True,
            comment="Local rollout attempt corresponding to one cloud add record.",
        ),
        Column(
            "add_record_id", Text, nullable=False, comment="Cloud memory/add record identity returned by the service."
        ),
        Column(
            "request_id",
            Text,
            nullable=True,
            comment="Cloud request identity, when returned; not an assumed idempotency guarantee.",
        ),
        Column("synced_at", UTCDateTime(), nullable=False, comment="Time the cloud add record identity was confirmed."),
        ForeignKeyConstraint(["trajectory_id"], [f"{TRAJECTORIES_TABLE}.trajectory_id"]),
        UniqueConstraint("remote_id", "project_id", "add_record_id"),
    )
    specs.append(
        TableSpec(name=REMOTE_TRAJECTORY_BINDINGS_TABLE, table=table, insert_model=RemoteTrajectoryBindingRecord)
    )
    table = Table(
        REMOTE_OPERATIONS_TABLE,
        metadata,
        Column(
            "remote_id",
            Text,
            primary_key=True,
            comment="Stable configured remote deployment alias; contains no credentials.",
        ),
        Column(
            "project_id",
            Text,
            primary_key=True,
            comment="Cloud project resolved under the authenticated remote identity.",
        ),
        Column("operation_id", Text, primary_key=True, comment="Local durable operation identity."),
        Column(
            "idempotency_key",
            Text,
            nullable=False,
            comment="Local deduplication key; the cloud need not accept this key.",
        ),
        Column(
            "operation_type",
            Text,
            nullable=False,
            comment="Existing cloud capability requested by this local operation.",
        ),
        Column("local_version_id", Text, nullable=True, comment="Corresponding local skill_versions.version_id."),
        Column(
            "trajectory_id", Text, nullable=True, comment="Local rollout attempt corresponding to one cloud add record."
        ),
        Column(
            "cloud_skill_id", Text, nullable=True, comment="Cloud lineage identity returned by the existing service."
        ),
        Column(
            "request_hash", Text, nullable=False, comment="Digest of the immutable credential-free request payload."
        ),
        Column(
            "request_payload",
            JSON,
            nullable=False,
            comment="Exact business request needed for replay; no authorization headers or secrets.",
        ),
        Column("status", Text, nullable=False, comment="Lifecycle state returned by the stable cloud contract."),
        Column("attempt_count", Integer, nullable=False, comment="Number of remote attempts already started."),
        Column("lease_owner", Text, nullable=True, comment="Identity of the worker holding this operation."),
        Column("lease_expires_at", UTCDateTime(), nullable=True, comment="Expiry of the current processing lease."),
        Column(
            "next_retry_at",
            UTCDateTime(),
            nullable=True,
            comment="Next permitted retry after a known retryable failure.",
        ),
        Column("last_error", Text, nullable=True, comment="Last failure or reconciliation explanation."),
        Column(
            "response_payload",
            JSON,
            nullable=False,
            comment="Last confirmed cloud result, including returned remote identities.",
        ),
        Column("created_at", UTCDateTime(), nullable=False, comment="Time this association was established."),
        Column("updated_at", UTCDateTime(), nullable=False, comment="Last change to this association."),
        ForeignKeyConstraint(["local_version_id"], [f"{SKILL_VERSIONS_TABLE}.version_id"]),
        ForeignKeyConstraint(["trajectory_id"], [f"{TRAJECTORIES_TABLE}.trajectory_id"]),
        UniqueConstraint("remote_id", "project_id", "idempotency_key"),
        CheckConstraint("attempt_count >= 0"),
        Index("remote_operations_retry_idx", "remote_id", "project_id", "status", "next_retry_at"),
    )
    specs.append(
        TableSpec(
            name=REMOTE_OPERATIONS_TABLE,
            table=table,
            insert_model=RemoteOperationRecord,
        )
    )
    return tuple(specs)
