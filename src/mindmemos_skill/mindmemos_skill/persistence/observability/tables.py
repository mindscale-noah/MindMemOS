"""Portable observability tables for supported SQL backends."""

from sqlalchemy import JSON, Column, Index, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB

from ...infra.database import TableRegistry, TableSpec
from ...infra.database.typing import UTCDateTime
from .constants import OBSERVABILITY_LOGS_TABLE, OBSERVABILITY_RUNS_TABLE, OBSERVABILITY_SPANS_TABLE
from .typing import AuditLog, AuditRun, AuditSpan


def build_tables() -> TableRegistry:
    """Build an instance-scoped observability table catalog.

    Returns:
        Tables for run state, completed spans, and trace-correlated logs.
    """
    metadata = MetaData()
    json_value = JSON().with_variant(JSONB(), "postgresql")
    runs = Table(
        OBSERVABILITY_RUNS_TABLE,
        metadata,
        Column("project_id", Text, primary_key=True),
        Column("run_id", Text, primary_key=True),
        Column("algorithm", Text, nullable=False),
        Column("base_version_id", Text, nullable=False),
        Column("published_version_id", Text),
        Column("status", Text, nullable=False),
        Column("audit_status", Text, nullable=False),
        Column("capture_level", Text, nullable=False),
        Column("trace_id", Text, nullable=False),
        Column("started_at", UTCDateTime(), nullable=False),
        Column("finished_at", UTCDateTime()),
        Column("error_type", Text),
        Index("observability_runs_recent_idx", "project_id", "started_at"),
        Index("observability_runs_version_idx", "project_id", "base_version_id", "started_at", "algorithm"),
    )
    spans = Table(
        OBSERVABILITY_SPANS_TABLE,
        metadata,
        Column("trace_id", Text, primary_key=True),
        Column("span_id", Text, primary_key=True),
        Column("parent_span_id", Text),
        Column("project_id", Text, nullable=False),
        Column("run_id", Text, nullable=False),
        Column("name", Text, nullable=False),
        Column("kind", Text, nullable=False),
        Column("status", Text, nullable=False),
        Column("started_at", UTCDateTime(), nullable=False),
        Column("finished_at", UTCDateTime(), nullable=False),
        Column("attributes", json_value, nullable=False),
        Column("events", json_value, nullable=False),
        Column("links", json_value, nullable=False),
        Column("resource", json_value, nullable=False),
        Index("observability_spans_run_idx", "project_id", "run_id", "started_at"),
        Index("observability_spans_parent_idx", "trace_id", "parent_span_id"),
    )
    logs = Table(
        OBSERVABILITY_LOGS_TABLE,
        metadata,
        Column("event_id", Text, primary_key=True),
        Column("project_id", Text, nullable=False),
        Column("run_id", Text, nullable=False),
        Column("trace_id", Text, nullable=False),
        Column("span_id", Text, nullable=False),
        Column("occurred_at", UTCDateTime(), nullable=False),
        Column("level", Text, nullable=False),
        Column("name", Text, nullable=False),
        Column("body", json_value, nullable=False),
        Index("observability_logs_run_idx", "project_id", "run_id", "occurred_at"),
        Index("observability_logs_span_idx", "trace_id", "span_id"),
    )
    return TableRegistry(
        (
            TableSpec(name=OBSERVABILITY_RUNS_TABLE, table=runs, insert_model=AuditRun),
            TableSpec(name=OBSERVABILITY_SPANS_TABLE, table=spans, insert_model=AuditSpan),
            TableSpec(name=OBSERVABILITY_LOGS_TABLE, table=logs, insert_model=AuditLog),
        )
    )
