"""Compose the table catalogs owned by individual data services."""

from sqlalchemy import MetaData

from ..infra.database import TableRegistry
from .db_services.remote.tables import build_tables as build_remote_tables
from .db_services.skill_management.tables import build_tables as build_graph_tables
from .db_services.skills.tables import build_tables as build_skill_tables
from .db_services.trajectories.tables import build_tables as build_trajectory_tables


def build_persistence_tables(*, include_remote: bool = True) -> TableRegistry:
    """Build an isolated registry without connecting to a database.

    Args:
        include_remote: Include cloud mappings and the durable operation queue.

    Returns:
        Combined definitions with a shared foreign-key metadata graph.
    """
    metadata = MetaData()
    specs = build_skill_tables(metadata) + build_trajectory_tables(metadata) + build_graph_tables(metadata)
    if include_remote:
        specs += build_remote_tables(metadata)
    return TableRegistry(specs)
