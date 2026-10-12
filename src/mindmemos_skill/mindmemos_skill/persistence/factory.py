"""Compose business data services and explicitly ensure their collections."""

from ..infra.database import Database
from .base import Persistence
from .store import SkillPersistence


def create_persistence(database: Database) -> SkillPersistence:
    """Register and assemble services without opening a database connection.

    Args:
        database: Borrowed infra database with no conflicting table registrations.

    Returns:
        Services sharing the same database and transaction boundary.
    """
    return Persistence.build("skill", {}, database=database, require_session=False)


async def ensure_collection(persistence: SkillPersistence) -> None:
    """Create the current registered tables.

    Args:
        persistence: Assembled data services whose collections should exist.
    """
    await persistence.database.create_tables()
