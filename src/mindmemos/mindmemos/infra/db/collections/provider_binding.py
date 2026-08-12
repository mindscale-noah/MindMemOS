"""Repository for the ``provider_binding_v1`` collection."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import models as qmodels

from ..filters import PROVIDER_BINDING_PAYLOAD_INDEX_SCHEMA
from ..models import ProviderBindingPoint, QdrantRecord
from .base import CollectionRepository


class ProviderBindingRepository(CollectionRepository):
    """Typed adapter for dynamic provider bindings."""

    @property
    def collection(self) -> str:
        return self._cfg.provider_binding_collection

    async def upsert(self, points: list[ProviderBindingPoint]) -> None:
        """Upsert provider binding points."""

        await self._upsert_payload_points_by_project(
            [(_provider_binding_point_id(point.binding_id), point.payload) for point in points],
            payload_indexes=list(PROVIDER_BINDING_PAYLOAD_INDEX_SCHEMA),
        )

    async def get(self, project_id: str, binding_id: str) -> QdrantRecord | None:
        """Retrieve one provider binding by project and id."""

        return await self._get_one_scoped(project_id, _provider_binding_point_id(binding_id), with_vectors=False)

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


def _provider_binding_point_id(binding_id: str) -> str:
    """Return a Qdrant-compatible deterministic UUID for one binding id."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mindmemos:provider-binding:{binding_id}"))
