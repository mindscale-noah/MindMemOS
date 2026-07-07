"""Repository for the ``provider_binding_v1`` collection."""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ..models import ProviderBindingPoint, QdrantRecord
from .base import CollectionRepository


class ProviderBindingRepository(CollectionRepository):
    """Typed adapter for dynamic provider bindings."""

    @property
    def collection(self) -> str:
        return self._cfg.provider_binding_collection

    async def upsert(self, points: list[ProviderBindingPoint]) -> None:
        """Upsert provider binding points."""

        await self._engine.upsert(self.collection, [self._payload_point(point.binding_id, point.payload) for point in points])

    async def get(self, project_id: str, binding_id: str) -> QdrantRecord | None:
        """Retrieve one provider binding by project and id."""

        return await self._get_one_scoped(project_id, binding_id, with_vectors=False)

    async def scroll(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 100,
        cursor: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll provider bindings in one project."""

        return await self._scroll_scoped(project_id, filter_=filter_, limit=limit, cursor=cursor)
