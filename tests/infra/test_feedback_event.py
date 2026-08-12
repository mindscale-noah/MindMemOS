"""Tests for the feedback event store (``feedback_event_v1``)."""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client import models as qmodels

from mindmemos.infra.db.collections.feedback_event import FeedbackEventRepository
from mindmemos.infra.db.feedback_event import FeedbackEventStore
from mindmemos.infra.db.models import QdrantRecord


class _FakeFeedbackEventRepository(FeedbackEventRepository):
    """In-memory repo recording the scroll ordering used by the store."""

    def __init__(self) -> None:
        self._records: list[QdrantRecord] = []
        self.last_order_by: Any = None

    @property
    def collection(self) -> str:
        return "feedback_event_v1"

    async def scroll(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        del project_id, filter_, cursor
        self.last_order_by = order_by
        return self._records[:limit], None


def _record(event_id: str, submitted_at: str) -> QdrantRecord:
    return QdrantRecord(
        point_id=event_id,
        payload={
            "event_id": event_id,
            "account_id": "acc",
            "project_id": "p",
            "api_key_uuid": "key",
            "submitted_at": submitted_at,
            "signals": [],
        },
    )


@pytest.mark.asyncio
async def test_list_events_orders_newest_first():
    repo = _FakeFeedbackEventRepository()
    repo._records = [
        _record("evt-old", "2026-08-01T00:00:00Z"),
        _record("evt-new", "2026-08-05T00:00:00Z"),
    ]
    store = FeedbackEventStore(repo=repo)  # type: ignore[arg-type]

    events = await store.list_events("p", limit=10)

    assert repo.last_order_by is not None
    assert repo.last_order_by.key == "submitted_at"
    assert repo.last_order_by.direction == qmodels.Direction.DESC
    assert [event.event_id for event in events] == ["evt-old", "evt-new"]
