"""Task and progress tables in the existing Skill database."""

from sqlalchemy import JSON, Column, ForeignKeyConstraint, Index, Integer, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB

from ....infra.database import TableSpec
from ....infra.database.typing import UTCDateTime
from ...constants import SKILL_TASK_PROGRESS_TABLE, SKILL_TASKS_TABLE
from .typing import TaskProgressRecord, TaskRecord


def build_tables(metadata: MetaData) -> tuple[TableSpec, ...]:
    """Register additive task tables without changing existing business rows.

    Args:
        metadata: Shared business table metadata.

    Returns:
        Task and progress registrations.
    """
    json_type = JSON().with_variant(JSONB(), "postgresql")
    tasks = Table(
        SKILL_TASKS_TABLE,
        metadata,
        Column("task_id", Text, primary_key=True, comment="Globally unique task identity."),
        Column("project_id", Text, nullable=False, comment="Project owning the task."),
        Column("kind", Text, nullable=False, comment="Trusted handler identity."),
        Column("payload", json_type, nullable=False, comment="Immutable request with explicit configuration."),
        Column("status", Text, nullable=False, comment="Current task lifecycle state."),
        Column("progress", json_type, comment="Most recent validated progress event."),
        Column("progress_seq", Integer, nullable=False, comment="Last allocated event sequence."),
        Column("result", json_type, comment="Successful business result."),
        Column("error_type", Text, comment="Exception type without sensitive text."),
        Column("attempt", Integer, nullable=False, comment="Number of physical attempts."),
        Column("lease_token", Text, comment="Worker fencing token."),
        Column("lease_expires_at", UTCDateTime(), comment="Lease expiry for recovery."),
        Column("next_notify_at", UTCDateTime(), nullable=False, comment="Next Kafka notification time."),
        Column("created_at", UTCDateTime(), nullable=False, comment="Initial commit time."),
        Column("updated_at", UTCDateTime(), nullable=False, comment="Latest state change time."),
        Index("skill_tasks_project_idx", "project_id", "created_at"),
        Index("skill_tasks_dispatch_idx", "status", "next_notify_at"),
        Index("skill_tasks_lease_idx", "status", "lease_expires_at"),
    )
    events = Table(
        SKILL_TASK_PROGRESS_TABLE,
        metadata,
        Column("project_id", Text, primary_key=True, comment="Project owning the event."),
        Column("task_id", Text, primary_key=True, comment="Task owning the event."),
        Column("sequence", Integer, primary_key=True, comment="One-based event order."),
        Column("attempt", Integer, nullable=False, comment="Physical execution attempt."),
        Column("event", json_type, nullable=False, comment="Validated progress payload."),
        Column("occurred_at", UTCDateTime(), nullable=False, comment="Progress report time."),
        ForeignKeyConstraint(["task_id"], [f"{SKILL_TASKS_TABLE}.task_id"]),
        Index("skill_task_progress_recent_idx", "project_id", "task_id", "sequence"),
    )
    return (
        TableSpec(name=SKILL_TASKS_TABLE, table=tasks, insert_model=TaskRecord),
        TableSpec(name=SKILL_TASK_PROGRESS_TABLE, table=events, insert_model=TaskProgressRecord),
    )
