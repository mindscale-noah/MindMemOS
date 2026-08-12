"""Repository for the ``evolution_state_v1`` collection (payload-only version log)."""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ..models import QdrantRecord
from .base import CollectionRepository


class EvolutionStateRepository(CollectionRepository):
    """Typed adapter for ``evolution_state_v1`` (feedback_evo versioned state)."""

    @property
    def collection(self) -> str:
        return self._cfg.evolution_state_collection

    async def upsert(self, point_id: str, payload: dict[str, Any]) -> None:
        """Upsert one evolution-state version point."""

        await self._engine.upsert(
            self.collection,
            [self._payload_point(point_id, payload)],
        )

    async def get(self, project_id: str, point_id: str) -> QdrantRecord | None:
        """Retrieve one version point, scoped to ``project_id``."""

        return await self._get_one_scoped(project_id, point_id)

    async def scroll(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll evolution-state versions inside one project."""

        return await self._scroll_scoped(
            project_id,
            filter_=filter_,
            limit=limit,
            cursor=cursor,
            order_by=order_by,
        )
