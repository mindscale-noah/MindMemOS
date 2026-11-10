"""Assemble typed repositories on one scoped or transaction-bound database."""

from contextlib import asynccontextmanager

from ..infra.database.scoped import ScopedDatabase
from .access import RepositoryDatabase, _project_id_from_database
from .constants import (
    LEGACY_SKILL_SYNC_ITEMS,
    SKILL_EVALUATIONS_TABLE,
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    SKILL_TASK_LOGS_TABLE,
    SKILL_TASKS_TABLE,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
    TRAJECTORIES_TABLE,
    TRAJECTORY_CONSUMPTIONS_TABLE,
    TRAJECTORY_VERSIONS_TABLE,
)
from .db_services.audit.service import AuditDataService
from .db_services.evaluations.service import EvaluationDataService
from .db_services.graphs.service import GraphDataService
from .db_services.skills.service import SkillDataService
from .db_services.tasks.service import TaskDataService
from .db_services.trajectories.service import TrajectoryDataService
from .observability.constants import OBSERVABILITY_LOGS_TABLE, OBSERVABILITY_RUNS_TABLE, OBSERVABILITY_SPANS_TABLE
from .typing import TransactionState


class SkillPersistenceSession:
    """Provide repositories sharing one immutable scope and optional transaction."""

    def __init__(self, database, *, prefixed_ids: bool, audit_database=None, state: TransactionState | None = None):
        """Assemble repositories without opening connections.

        Args:
            database: Active project or explicit maintenance database handle.
            prefixed_ids: Format used to allocate new Skill identities.
            audit_database: Optional independently scoped audit database.
            state: Shared failure state for a transaction session.
        """
        self.database = database
        self.scope = database.scope if isinstance(database, ScopedDatabase) else None
        self._prefixed_ids = prefixed_ids
        self._audit_database = audit_database
        self._state = state
        self._active = True
        self._handles: list[RepositoryDatabase] = []
        skill_tables = frozenset({SKILLS_TABLE, SKILL_VERSIONS_TABLE})
        self.skills = SkillDataService(
            self.repository(skill_tables, read_tables=frozenset({LEGACY_SKILL_SYNC_ITEMS})), prefixed_ids=prefixed_ids
        )
        skill_reader = SkillDataService(
            self.repository(frozenset(), read_tables=skill_tables | {LEGACY_SKILL_SYNC_ITEMS}),
            prefixed_ids=prefixed_ids,
        )
        self.trajectories = TrajectoryDataService(
            self.repository(frozenset({TRAJECTORIES_TABLE, TRAJECTORY_CONSUMPTIONS_TABLE, TRAJECTORY_VERSIONS_TABLE})),
            skills=skill_reader,
        )
        self.graphs = GraphDataService(
            self.repository(
                frozenset(
                    {SKILL_GRAPHS_TABLE, SKILL_GRAPH_VERSIONS_TABLE, SKILL_GRAPH_NODES_TABLE, SKILL_GRAPH_EDGES_TABLE}
                )
            ),
            skills=skill_reader,
        )
        self.tasks = TaskDataService(self.repository(frozenset({SKILL_TASKS_TABLE, SKILL_TASK_LOGS_TABLE})))
        self.evaluations = EvaluationDataService(self.repository(frozenset({SKILL_EVALUATIONS_TABLE})))
        self.audit: AuditDataService | None = None
        if audit_database is not None:
            audit = RepositoryDatabase(
                audit_database,
                tables=frozenset(),
                read_tables=frozenset({OBSERVABILITY_RUNS_TABLE, OBSERVABILITY_SPANS_TABLE, OBSERVABILITY_LOGS_TABLE}),
            )
            self._handles.append(audit)
            self.audit = AuditDataService(audit)

    @property
    def project_id(self) -> str:
        """Read project ownership from this session's active database scope.

        Returns:
            Current project identity without maintaining a separate stored value.

        Raises:
            RuntimeError: The session is closed or has no project scope.
            ValueError: The project scope is missing or has a non-string identity.
        """
        if not self._active:
            raise RuntimeError("Persistence session is closed")
        return _project_id_from_database(self.database)

    def repository(self, tables: frozenset[str], *, read_tables: frozenset[str] = frozenset()) -> RepositoryDatabase:
        """Bind a custom service to its registered tables and the current scope.

        Args:
            tables: Tables owned by the service, allowing reads and writes.
            read_tables: Explicit read-only dependencies owned by other services.

        Returns:
            Access handle that cannot widen the session's row scope.
        """
        if not self._active:
            raise RuntimeError("Persistence session is closed")
        handle = RepositoryDatabase(self.database, tables=tables, read_tables=read_tables, state=self._state)
        self._handles.append(handle)
        return handle

    def close(self) -> None:
        """Invalidate repository handles without closing borrowed databases."""
        self._active = False
        for handle in self._handles:
            handle._active = False

    @asynccontextmanager
    async def transaction(self):
        """Share one business transaction across repositories without nesting.

        Yields:
            Repositories using one connection, table restrictions, and scope.

        Raises:
            RuntimeError: The session is closed, nested, or contains a failed operation.
        """
        if not self._active:
            raise RuntimeError("Persistence session is closed")
        if self._state is not None:
            raise RuntimeError("Nested persistence transactions are not supported")
        async with self.database.transaction() as database:
            state = TransactionState()
            # An independent audit database cannot participate in this SQL transaction.
            session = type(self)(database, prefixed_ids=self._prefixed_ids, state=state)
            try:
                yield session
                if state.failed:
                    raise RuntimeError("Persistence transaction contains a failed operation")
            finally:
                session.close()
