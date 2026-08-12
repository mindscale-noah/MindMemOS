"""Feedback event store for the ``feedback_evo`` mode (``feedback_event_v1``)."""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ...typing import FeedbackEvoEvent, MemoryRequestContext
from .collections import FeedbackEventRepository
from .filters import match_value
from .registry import resolve_database_clients


class FeedbackEventStore:
    """Write and read task-end feedback events that feed the evolution loop."""

    def __init__(self, *, repo: FeedbackEventRepository | None = None) -> None:
        self._repo = repo

    def _repo_impl(self) -> FeedbackEventRepository:
        if self._repo is None:
            self._repo = resolve_database_clients().qdrant.feedback_event
        return self._repo

    async def append(self, context: MemoryRequestContext, event: FeedbackEvoEvent) -> None:
        """Persist one feedback event for the request's project."""

        payload = event.model_dump(mode="json")
        payload["account_id"] = context.account_id
        payload["project_id"] = context.project_id
        payload["api_key_uuid"] = context.api_key_uuid
        await self._repo_impl().upsert(event.event_id, payload)

    async def list_events(
        self,
        project_id: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[FeedbackEvoEvent]:
        """List feedback events for one project, newest first (optionally filtered)."""

        conditions: list[Any] = []
        if user_id:
            conditions.append(match_value("user_id", user_id))
        if session_id:
            conditions.append(match_value("session_id", session_id))
        filter_ = qmodels.Filter(must=conditions) if conditions else None
        records, _ = await self._repo_impl().scroll(
            project_id,
            filter_=filter_,
            limit=limit,
            order_by=qmodels.OrderBy(
                key="submitted_at",
                direction=qmodels.Direction.DESC,
            ),
        )
        events: list[FeedbackEvoEvent] = []
        for record in records:
            try:
                events.append(FeedbackEvoEvent.model_validate(record.payload))
            except Exception:
                # Skip malformed/foreign payloads; the log is best-effort input.
                continue
        return events
