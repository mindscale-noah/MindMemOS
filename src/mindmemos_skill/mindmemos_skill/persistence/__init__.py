"""Business-owned records, table definitions, and persistence registration."""

from .base import Persistence
from .config import SkillPersistenceConfig
from .db_services.skill_management import (
    SkillGraphEdgeRecord,
    SkillGraphNodeRecord,
    SkillGraphRecord,
    SkillGraphVersionRecord,
)
from .db_services.skills import SkillRecord, SkillVersionRecord, SkillVersionSummary
from .db_services.trajectories import TrajectoryRecord, TrajectorySummary
from .factory import create_persistence, ensure_collection
from .registry import ensure_registered
from .tables import build_persistence_tables

ensure_registered()

__all__ = [
    "Persistence",
    "create_persistence",
    "ensure_collection",
    "SkillPersistenceConfig",
    "SkillVersionRecord",
    "SkillVersionSummary",
    "SkillRecord",
    "SkillGraphRecord",
    "SkillGraphVersionRecord",
    "SkillGraphNodeRecord",
    "SkillGraphEdgeRecord",
    "TrajectoryRecord",
    "TrajectorySummary",
    "build_persistence_tables",
]
