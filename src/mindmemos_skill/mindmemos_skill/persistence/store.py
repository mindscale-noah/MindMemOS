"""Register business-owned schemas on a borrowed infra database."""

from contextlib import ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar

from ..infra.database import Database, Scope, TableRegistry
from ..infra.database.scoped import ScopedDatabase
from .base import Persistence
from .config import SkillPersistenceConfig
from .constants import OPTIONAL_SCOPE_FIELDS, REQUIRED_SCOPE_FIELDS
from .db_services.audit.service import AuditDataService
from .db_services.graphs.service import GraphDataService
from .db_services.skills.service import SkillDataService
from .db_services.tasks.service import TaskDataService
from .db_services.tasks.worker import TaskWorkerService
from .db_services.trajectories.service import TrajectoryDataService
from .session import SkillPersistenceSession
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


@Persistence.register("skill")
class SkillPersistence(Persistence[SkillPersistenceConfig]):
    """Attach business tables without connecting or changing existing schemas.

    Args:
        config: Skill repository persistence settings.
        database: Borrowed database whose lifecycle remains with its owner.
    """

    config_type = SkillPersistenceConfig
    """Settings used by the shared backend factory."""

    session_type = SkillPersistenceSession
    """Session implementation; subclasses may assemble additional typed repositories."""

    def __init__(self, config: SkillPersistenceConfig, *, database: Database, audit_database: Database | None = None):
        """Register owned tables and retain borrowed storage resources.

        Args:
            config: Validated repository settings.
            database: Business database owned by the service assembly.
            audit_database: Optional independent database for project audit queries.
        """
        super().__init__(config)
        if not isinstance(database, Database):
            raise TypeError("Skill persistence requires an infra Database")
        additions = self.build_tables()
        # Validate all collisions before mutating the borrowed registry.
        TableRegistry((*database.tables.values(), *additions.values()))
        for spec in additions.values():
            database.tables.register(spec)
        self._database = database
        self._audit_database = None
        self._current_session: ContextVar[SkillPersistenceSession | None] = ContextVar(
            "skill_persistence_session", default=None
        )
        self._task_worker = TaskWorkerService(database)
        if audit_database is not None:
            self.attach_audit_database(audit_database)

    def build_tables(self) -> TableRegistry:
        """Declare owned tables; subclasses may append explicit custom repository tables.

        Returns:
            Instance-local table definitions with validated foreign keys.
        """
        return build_persistence_tables()

    def task_worker(self) -> TaskWorkerService:
        """Grant scheduling access to a trusted startup or maintenance owner.

        Returns:
            Cross-project worker operations for injection into the task manager.

        Raises:
            RuntimeError: A request or maintenance session is already active.
        """
        if self._current_session.get() is not None:
            raise RuntimeError("Worker capabilities cannot be acquired inside a persistence session")
        return self._task_worker

    @property
    def project_id(self) -> str:
        """Read the project identity of the current persistence session.

        Returns:
            Project identity derived from the active session's scope.

        Raises:
            RuntimeError: No active project session exists or its handle is closed.
            ValueError: The project scope is missing or has a non-string identity.
        """
        return self._session().project_id

    @property
    def database(self) -> Database | ScopedDatabase:
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
    def graphs(self) -> GraphDataService:
        """Return graph operations for the active project scope."""
        return self._session().graphs

    @property
    def tasks(self) -> TaskDataService:
        """Return project task operations without worker scheduling capabilities."""
        return self._session().tasks

    @property
    def audit(self) -> AuditDataService | None:
        """Return configured project audit queries, or None when storage is disabled."""
        return self._session().audit

    def _session(self) -> SkillPersistenceSession:
        """Reject normal service access outside an active session."""
        current = self._current_session.get()
        if current is not None:
            if not current._active:
                raise RuntimeError("Persistence session is closed")
            return current
        raise RuntimeError("Persistence session is required")

    def attach_audit_database(self, database: Database) -> None:
        """Attach a borrowed audit database before serving project sessions.

        Args:
            database: Independent audit database whose lifecycle belongs to its owner.

        Raises:
            RuntimeError: An audit database is already attached or a session is active.
        """
        from .observability.tables import build_tables

        if self._audit_database is not None or self._current_session.get() is not None:
            raise RuntimeError("Audit storage must be attached once before opening sessions")
        additions = build_tables()
        existing = {spec.name: spec for spec in database.tables.values()}
        for spec in additions.values():
            if spec.name in existing:
                if existing[spec.name].insert_model is not spec.insert_model:
                    raise ValueError(f"Audit table has a conflicting record model: {spec.name!r}")
            else:
                database.tables.register(spec)
        self._audit_database = database

    @contextmanager
    def maintenance(self):
        """Grant explicit unscoped repository access for trusted setup and maintenance.

        Yields:
            Repositories without a project boundary; never use this for request handling.

        Raises:
            RuntimeError: Called from an active business or maintenance session.
        """
        if self._current_session.get() is not None:
            raise RuntimeError("Maintenance cannot run inside a persistence session")
        current = self.session_type(self._database, prefixed_ids=self.config.prefixed_ids)
        token = self._current_session.set(current)
        try:
            yield current
        finally:
            current.close()
            self._current_session.reset(token)

    @asynccontextmanager
    async def transaction(self):
        """Bind all entrypoint repositories to the active session's SQL transaction.

        Yields:
            Transaction-bound repositories; independent audit storage is unavailable here.
        """
        async with self._session().transaction() as current:
            token = self._current_session.set(current)
            try:
                yield current
            finally:
                self._current_session.reset(token)

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
        if active is not None and active._state is not None:
            raise RuntimeError("Cannot open a persistence session inside a transaction")
        if active is not None and (active.scope is not None and active.scope != scope):
            raise RuntimeError("A nested persistence session cannot change scope")
        with ExitStack() as stack:
            database = stack.enter_context(self._database.scoped(scope))
            audit_database = (
                stack.enter_context(self._audit_database.scoped(scope)) if self._audit_database is not None else None
            )
            current = self.session_type(database, prefixed_ids=self.config.prefixed_ids, audit_database=audit_database)
            token = self._current_session.set(current)
            try:
                yield current
            finally:
                current.close()
                self._current_session.reset(token)

    async def initialize_storage(self) -> None:
        """Create tables for the current Skill repository schema."""
        await self._database.create_tables()
        if self._audit_database is not None:
            await self._audit_database.create_tables()
