"""Compose business data services and explicitly ensure their collections."""

from ..infra.database import Database
from .base import Persistence
from .store import SkillPersistence, remove_remote_tables


def create_persistence(database: Database) -> SkillPersistence:
    """Register and assemble services without opening a database connection.

    Args:
        database: Borrowed infra database with no conflicting table registrations.

    Returns:
        Services sharing the same database and transaction boundary.
    """
    return Persistence.build("skill", {}, database=database, require_session=False)


def ensure_collection(persistence: SkillPersistence) -> None:
    """Create registered tables and retire obsolete synchronization tables.

    Args:
        persistence: Assembled data services whose collections should exist.
    """
    persistence.database.create_tables()
    remove_remote_tables(persistence.database)
