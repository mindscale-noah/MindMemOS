"""Register business-owned schemas on a borrowed infra database."""

from ..infra.database import Database, TableRegistry
from .base import Persistence
from .config import SkillPersistenceConfig
from .db_services.remote.service import RemoteDataService
from .db_services.skills import SkillDataService
from .db_services.trajectories import TrajectoryDataService
from .tables import build_persistence_tables


@Persistence.register("skill")
class SkillPersistence(Persistence[SkillPersistenceConfig]):
    """Attach business tables without connecting or changing existing schemas.

    Args:
        config: Whether remote tables are needed.
        database: Borrowed database whose lifecycle remains with its owner.
    """

    config_type = SkillPersistenceConfig
    """Settings used by the shared backend factory."""

    def __init__(self, config: SkillPersistenceConfig, *, database: Database):
        super().__init__(config)
        if not isinstance(database, Database):
            raise TypeError("Skill persistence requires an infra Database")
        additions = build_persistence_tables(include_remote=config.include_remote)
        # Validate all collisions before mutating the borrowed registry.
        TableRegistry((*database.tables.values(), *additions.values()))
        for spec in additions.values():
            database.tables.register(spec)
        self.database = database
        self.skills = SkillDataService(database)
        self.trajectories = TrajectoryDataService(database)
        self.remote = RemoteDataService(database) if config.include_remote else None
