"""Read algorithm audit records through an authorized repository handle."""

from datetime import datetime

from ....infra.database import Filter
from ...observability.constants import OBSERVABILITY_LOGS_TABLE, OBSERVABILITY_RUNS_TABLE, OBSERVABILITY_SPANS_TABLE
from ...observability.typing import AuditLog, AuditRun, AuditSpan


class AuditDataService:
    """Query runs, spans, and logs without owning telemetry or database resources."""

    def __init__(self, database):
        """Borrow the audit repository for an active scope.

        Args:
            database: Scope-bound handle restricted to audit tables.
        """
        self.database = database

    async def get_run(self, run_id: str) -> AuditRun | None:
        """Read one run in the authorized project.

        Args:
            run_id: Exact run identity.

        Returns:
            Validated run or None when absent.
        """
        rows = await self.database.select(OBSERVABILITY_RUNS_TABLE, filters={"run_id": run_id}, limit=1)
        return AuditRun.model_validate(rows[0]) if rows else None

    async def latest_evolution(self, version_id: str, algorithms: list[str]) -> datetime | None:
        """Read the latest evolution start for a version and algorithm set.

        Args:
            version_id: Exact input Skill version.
            algorithms: Accepted configured aliases and implementation names.

        Returns:
            Latest start time or None without a matching run.
        """
        if not algorithms:
            return None
        rows = await self.database.select(
            OBSERVABILITY_RUNS_TABLE,
            filters=[
                Filter(field="base_version_id", value=version_id),
                Filter(field="algorithm", op="in", value=algorithms),
            ],
            fields=["started_at"],
            order_by=["-started_at"],
            limit=1,
        )
        return rows[0]["started_at"] if rows else None

    async def spans(self, run_id: str, *, limit: int = 500, offset: int = 0) -> list[AuditSpan]:
        """Read one stable page of a run's completed spans.

        Args:
            run_id: Exact run identity.
            limit: Maximum records, from 1 to 1000.
            offset: Number of ordered records to skip.

        Returns:
            Spans ordered by start time and span ID.
        """
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Invalid observability page")
        rows = await self.database.select(
            OBSERVABILITY_SPANS_TABLE,
            filters={"run_id": run_id},
            order_by=["started_at", "span_id"],
            limit=limit,
            offset=offset,
        )
        return [AuditSpan.model_validate(row) for row in rows]

    async def logs(self, run_id: str, *, limit: int = 200, offset: int = 0) -> list[AuditLog]:
        """Read one stable page of a run's correlated logs.

        Args:
            run_id: Exact run identity.
            limit: Maximum records, from 1 to 1000.
            offset: Number of ordered records to skip.

        Returns:
            Logs ordered by occurrence time and event ID.
        """
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Invalid observability page")
        rows = await self.database.select(
            OBSERVABILITY_LOGS_TABLE,
            filters={"run_id": run_id},
            order_by=["occurred_at", "event_id"],
            limit=limit,
            offset=offset,
        )
        return [AuditLog.model_validate(row) for row in rows]
