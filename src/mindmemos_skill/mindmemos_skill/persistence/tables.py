"""Compose the table catalogs owned by individual data services."""

from sqlalchemy import JSON, Boolean, Column, Index, Integer, MetaData, Table, Text

from ..infra.database import TableRegistry, TableSpec
from ..infra.database.typing import UTCDateTime
from .constants import LEGACY_SKILL_SYNC_ITEMS, LEGACY_SKILL_SYNC_PROJECTS
from .db_services.evaluations.tables import build_tables as build_evaluation_tables
from .db_services.graphs.tables import build_tables as build_graph_tables
from .db_services.skills.tables import build_tables as build_skill_tables
from .db_services.tasks.tables import build_tables as build_task_tables
from .db_services.trajectories.tables import build_tables as build_trajectory_tables
from .typing import LegacySkillSyncItem, LegacySkillSyncProject


def _build_legacy_sync_tables(metadata: MetaData) -> tuple[TableSpec, ...]:
    """Declare replication tables alongside the business data they checkpoint."""
    projects = Table(
        LEGACY_SKILL_SYNC_PROJECTS,
        metadata,
        Column("project_id", Text, primary_key=True, comment="Original Platform project namespace."),
        Column("ready", Boolean, nullable=False, comment="Initial backfill completed successfully."),
        Column("generation", Integer, nullable=False, comment="Monotonic worker fence for this namespace."),
        Column("lease_until", UTCDateTime(), comment="Time when a replacement worker can take over."),
        Column("trajectory_cursor", JSON, comment="Next source page to read after interruption."),
        Column("last_checked_at", UTCDateTime(), comment="Completion time of the last full reconciliation."),
        Column("failures", Integer, nullable=False, comment="Failed records in the current or most recent scan."),
    )
    items = Table(
        LEGACY_SKILL_SYNC_ITEMS,
        metadata,
        Column("project_id", Text, primary_key=True, comment="Namespace shared by source and destination."),
        Column("kind", Text, primary_key=True, comment="Family, version, or trajectory."),
        Column("legacy_id", Text, primary_key=True, comment="Exact ID in the source namespace."),
        Column("target_id", Text, comment="Stable destination ID, absent until the object is imported."),
        Column("generation", Integer, nullable=False, comment="Most recent scan that verified the source record."),
        Column("pending", Boolean, nullable=False, comment="At least one source trajectory binding is unresolved."),
        Column("source_deleted", Boolean, nullable=False, comment="The original service no longer manages the family."),
        Column("published_version_id", Text, comment="Exact old published version, when present."),
        Index("legacy_skill_sync_items_scan_idx", "project_id", "kind", "generation"),
    )
    return (
        TableSpec(name=projects.name, table=projects, insert_model=LegacySkillSyncProject),
        TableSpec(name=items.name, table=items, insert_model=LegacySkillSyncItem),
    )


def build_persistence_tables() -> TableRegistry:
    """Build an isolated registry without connecting to a database.

    Returns:
        Combined definitions with a shared foreign-key metadata graph.
    """
    metadata = MetaData()
    return TableRegistry(
        build_skill_tables(metadata)
        + build_trajectory_tables(metadata)
        + build_graph_tables(metadata)
        + build_task_tables(metadata)
        + build_evaluation_tables(metadata)
        + _build_legacy_sync_tables(metadata)
    )
