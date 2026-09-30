"""Register business-owned schemas on a borrowed infra database."""

from contextlib import contextmanager
from contextvars import ContextVar

from ..infra.database import Database, Scope, TableRegistry
from .base import Persistence
from .config import SkillPersistenceConfig
from .constants import OPTIONAL_SCOPE_FIELDS, REQUIRED_SCOPE_FIELDS
from .db_services.remote.service import RemoteDataService
from .db_services.skills import SkillDataService
from .db_services.skills.migrations import migrate_version_labels
from .db_services.trajectories import TrajectoryDataService
from .tables import build_persistence_tables


def validate_skill_scope(scope: Scope) -> None:
    """Require the business-declared isolation fields before accessing storage.

    Args:
        scope: Authorized field values for one request.

    Raises:
        ValueError: The scope has missing or unexpected business fields.
    """
    actual = {name for name, _ in scope.fields}
    missing = REQUIRED_SCOPE_FIELDS - actual
    unexpected = actual - REQUIRED_SCOPE_FIELDS - OPTIONAL_SCOPE_FIELDS
    if missing or unexpected:
        raise ValueError(f"Invalid Skill scope fields: missing={sorted(missing)}, unexpected={sorted(unexpected)}")


class SkillPersistenceSession:
    """Expose business data services bound to one active project scope."""

    def __init__(self, database, *, include_remote: bool):
        self.database = database
        self.skills = SkillDataService(database)
        self.trajectories = TrajectoryDataService(database)
        self.remote = RemoteDataService(database) if include_remote else None


@Persistence.register("skill")
class SkillPersistence(Persistence[SkillPersistenceConfig]):
    """Attach business tables without connecting or changing existing schemas.

    Args:
        config: Whether remote tables are needed.
        database: Borrowed database whose lifecycle remains with its owner.
    """

    config_type = SkillPersistenceConfig
    """Settings used by the shared backend factory."""

    def __init__(self, config: SkillPersistenceConfig, *, database: Database, require_session: bool = True):
        super().__init__(config)
        if not isinstance(database, Database):
            raise TypeError("Skill persistence requires an infra Database")
        additions = build_persistence_tables(include_remote=config.include_remote)
        # Validate all collisions before mutating the borrowed registry.
        TableRegistry((*database.tables.values(), *additions.values()))
        for spec in additions.values():
            database.tables.register(spec)
        self._database = database
        self._require_session = require_session
        self._current_session: ContextVar[SkillPersistenceSession | None] = ContextVar(
            "skill_persistence_session", default=None
        )
        self._admin_services = SkillPersistenceSession(database, include_remote=config.include_remote)

    @property
    def database(self):
        """Return the scoped database; low-level admin assembly retains its raw handle."""
        return self._session().database

    @property
    def skills(self) -> SkillDataService:
        """Return Skill operations for the active project scope."""
        return self._session().skills

    @property
    def trajectories(self) -> TrajectoryDataService:
        """Return trajectory operations for the active project scope."""
        return self._session().trajectories

    @property
    def remote(self) -> RemoteDataService | None:
        """Return remote mappings for the active project scope when enabled."""
        return self._session().remote

    def _session(self) -> SkillPersistenceSession:
        """Reject normal service access outside an active session."""
        current = self._current_session.get()
        if current is not None:
            if not current.database._active:
                raise RuntimeError("Persistence session is closed")
            return current
        if self._require_session:
            raise RuntimeError("Persistence session is required")
        return self._admin_services

    def local_database_path(self) -> str | None:
        """Expose only the SQLite file path needed for export destination checks."""
        self._session()
        engine = self._database.engine
        if engine.dialect.name != "sqlite":
            return None
        path = engine.url.database
        return path if path and path != ":memory:" else None

    @contextmanager
    def session(self, scope: Scope):
        """Bind all persistence services to one immutable project scope.

        Args:
            scope: Authorized project and optional additional dimensions.

        Yields:
            Services that reject access after the context exits.
        """
        validate_skill_scope(scope)
        active = self._current_session.get()
        if active is not None and active.database.scope != scope:
            raise RuntimeError("A nested persistence session cannot change scope")
        with self._database.scoped(scope) as database:
            current = SkillPersistenceSession(database, include_remote=self.config.include_remote)
            token = self._current_session.set(current)
            try:
                yield current
            finally:
                self._current_session.reset(token)

    def initialize_storage(self) -> None:
        """Create missing tables and idempotently migrate existing version labels."""
        self._database.create_tables()
        migrate_version_labels(self._database)
