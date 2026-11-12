"""Compose the table catalogs owned by individual data services."""

from sqlalchemy import Column, MetaData, Table, Text

from ..infra.database import TableRegistry, TableSpec
from .constants import LEGACY_SKILL_ID_MAP
from .db_services.evaluations.tables import build_tables as build_evaluation_tables
from .db_services.graphs.tables import build_tables as build_graph_tables
from .db_services.skills.tables import build_tables as build_skill_tables
from .db_services.tasks.tables import build_tables as build_task_tables
from .db_services.trajectories.tables import build_tables as build_trajectory_tables
from .typing import LegacySkillIdMapping


def build_persistence_tables() -> TableRegistry:
    """Build an isolated registry without connecting to a database.

    Returns:
        Combined definitions with a shared foreign-key metadata graph.
    """
    metadata = MetaData()
    legacy_ids = Table(
        LEGACY_SKILL_ID_MAP,
        metadata,
        Column("project_id", Text, primary_key=True, comment="Namespace shared by source and destination."),
        Column("kind", Text, primary_key=True, comment="Skill family or version."),
        Column("legacy_id", Text, primary_key=True, comment="Exact ID in the source namespace."),
        Column("target_id", Text, nullable=False, comment="Stable destination ID."),
    )
    return TableRegistry(
        build_skill_tables(metadata)
        + build_trajectory_tables(metadata)
        + build_graph_tables(metadata)
        + build_task_tables(metadata)
        + build_evaluation_tables(metadata)
        + (TableSpec(name=legacy_ids.name, table=legacy_ids, insert_model=LegacySkillIdMapping),)
    )
