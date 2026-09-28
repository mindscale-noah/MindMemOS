"""Compose business data services and explicitly ensure their collections."""

from ..infra.database import Database
from .base import Persistence
from .store import SkillPersistence


def create_persistence(database: Database, *, include_remote: bool = True) -> SkillPersistence:
    """Register and assemble services without opening a database connection.

    Args:
        database: Borrowed infra database with no conflicting table registrations.
        include_remote: Include local cloud bindings and operation queue.

    Returns:
        Services sharing the same database and transaction boundary.
    """
    return Persistence.build("skill", {"include_remote": include_remote}, database=database, require_session=False)


def ensure_collection(persistence: SkillPersistence) -> None:
    """Create missing registered tables without migrating existing cloud or local schemas.

    Args:
        persistence: Assembled data services whose collections should exist.
    """
    persistence.database.create_tables()
