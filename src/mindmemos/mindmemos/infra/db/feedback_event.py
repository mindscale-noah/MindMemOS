"""Feedback event store for the ``feedback_evo`` mode (``feedback_event_v1``)."""

from __future__ import annotations

from datetime import UTC, datetime
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
        unconsumed_only: bool = False,
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
                event = FeedbackEvoEvent.model_validate(record.payload)
                # Payloads written before consumption support are treated as
                # pending.  Include processing events as recoverable work so a
                # process restart cannot strand a round forever.
                if unconsumed_only and event.consumption_status == "consumed":
                    continue
                events.append(event)
            except Exception:
                # Skip malformed/foreign payloads; the log is best-effort input.
                continue
        return events

    async def claim(
        self,
        project_id: str,
        events: list[FeedbackEvoEvent],
        consumption_id: str,
    ) -> list[FeedbackEvoEvent]:
        """Mark selected events as processing and return their new payloads."""

        claimed: list[FeedbackEvoEvent] = []
        repo = self._repo_impl()
        for event in events:
            record = await repo.get(project_id, event.event_id)
            if record is None:
                continue
            current = FeedbackEvoEvent.model_validate(record.payload)
            if current.consumption_status == "consumed":
                continue
            payload = dict(record.payload)
            payload.update(
                {
                    "consumption_status": "processing",
                    "consumption_id": consumption_id,
                    "consumed_at": None,
                    "consumed_version": None,
                    "consumption_outcome": None,
                }
            )
            await repo.update_payload(event.event_id, payload)
            claimed.append(FeedbackEvoEvent.model_validate(payload))
        return claimed

    async def finish(
        self,
        project_id: str,
        events: list[FeedbackEvoEvent],
        *,
        consumption_id: str,
        version: int,
        outcome: str,
    ) -> int:
        """Mark processing events consumed after a successful evolution run."""

        finished = 0
        repo = self._repo_impl()
        now = datetime.now(UTC).isoformat()
        for event in events:
            record = await repo.get(project_id, event.event_id)
            if record is None:
                continue
            current = FeedbackEvoEvent.model_validate(record.payload)
            if current.consumption_status == "consumed":
                continue
            if current.consumption_id != consumption_id:
                continue
            payload = dict(record.payload)
            payload.update(
                {
                    "consumption_status": "consumed",
                    "consumption_id": consumption_id,
                    "consumed_at": now,
                    "consumed_version": version,
                    "consumption_outcome": outcome,
                }
            )
            await repo.update_payload(event.event_id, payload)
            finished += 1
        return finished

    async def release(
        self,
        project_id: str,
        events: list[FeedbackEvoEvent],
        *,
        consumption_id: str,
    ) -> int:
        """Return processing events to pending after a failed run."""

        released = 0
        repo = self._repo_impl()
        for event in events:
            record = await repo.get(project_id, event.event_id)
            if record is None:
                continue
            current = FeedbackEvoEvent.model_validate(record.payload)
            if current.consumption_id != consumption_id or current.consumption_status != "processing":
                continue
            payload = dict(record.payload)
            payload.update(
                {
                    "consumption_status": "pending",
                    "consumption_id": None,
                    "consumed_at": None,
                    "consumed_version": None,
                    "consumption_outcome": None,
                }
            )
            await repo.update_payload(event.event_id, payload)
            released += 1
        return released
