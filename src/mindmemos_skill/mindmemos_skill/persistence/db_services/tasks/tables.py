"""Task state and log tables in the existing Skill database."""

from sqlalchemy import JSON, Column, ForeignKeyConstraint, Index, Integer, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB

from ....infra.database import TableSpec
from ....infra.database.typing import UTCDateTime
from ....logging.typing import TaskLogRecord
from ...constants import SKILL_TASK_LOGS_TABLE, SKILL_TASKS_TABLE
from .typing import TaskRecord


def build_tables(metadata: MetaData) -> tuple[TableSpec, ...]:
    """Register additive task tables without changing existing business rows.

    Args:
        metadata: Shared business table metadata.

    Returns:
        Task and log registrations.
    """
    json_type = JSON().with_variant(JSONB(), "postgresql")
    tasks = Table(
        SKILL_TASKS_TABLE,
        metadata,
        Column("task_id", Text, primary_key=True, comment="Globally unique task identity."),
        Column("project_id", Text, nullable=False, comment="Project owning the task."),
        Column("kind", Text, nullable=False, comment="Trusted handler identity."),
        Column("payload", json_type, nullable=False, comment="Immutable request with explicit configuration."),
        Column("trace_context", json_type, nullable=False, comment="Submission W3C propagation headers."),
        Column("request_id", Text, comment="Original submission request identity."),
        Column("started_at", UTCDateTime(), comment="Current attempt start time."),
        Column("finished_at", UTCDateTime(), comment="Terminal result commit time."),
        Column("status", Text, nullable=False, comment="Current task lifecycle state."),
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
    logs = Table(
        SKILL_TASK_LOGS_TABLE,
        metadata,
        Column("sequence", Integer, primary_key=True, autoincrement=True, comment="Increasing pagination cursor."),
        Column("log_id", Text, nullable=False, unique=True, comment="Unique immutable log identity."),
        Column("project_id", Text, nullable=False, comment="Authorized owning project."),
        Column("task_id", Text, nullable=False, comment="Durable task identity."),
        Column("attempt", Integer, nullable=False, comment="Physical execution attempt."),
        Column("occurred_at", UTCDateTime(), nullable=False, comment="Original Python log timestamp."),
        Column("payload", json_type, nullable=False, comment="Sanitized message, body, traceback and correlation."),
        ForeignKeyConstraint(["task_id"], [f"{SKILL_TASKS_TABLE}.task_id"]),
        Index("skill_task_logs_recent_idx", "project_id", "task_id", "sequence"),
    )
    return (
        TableSpec(name=SKILL_TASK_LOGS_TABLE, table=logs, insert_model=TaskLogRecord),
        TableSpec(name=SKILL_TASKS_TABLE, table=tasks, insert_model=TaskRecord),
    )
