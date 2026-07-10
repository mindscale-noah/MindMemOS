"""Shared base for per-collection Qdrant repositories.

Each table under ``collections/`` binds exactly one Qdrant collection and adds
its own typed upsert/read methods. The cross-cutting mechanics — project-scoped
retrieve/scroll and point-struct building — live here so the concrete
repositories stay small and free of duplication. All low-level work is delegated
to the shared :class:`QdrantEngine`.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from qdrant_client import models as qmodels

from ....config import QdrantConfig
from ..engine import QdrantEngine
from ..models import PayloadIndexSpec, QdrantCollectionSpec, QdrantRecord


class CollectionRepository:
    """Typed adapter bound to a single Qdrant collection."""

    def __init__(self, engine: QdrantEngine, cfg: QdrantConfig) -> None:
        self._engine = engine
        self._cfg = cfg

    @property
    def collection(self) -> str:
        """Configured collection name (bound by the subclass)."""

        raise NotImplementedError

    def collection_for_project(self, project_id: str | None) -> str:
        """Return the physical collection used for one project."""

        if not self._cfg.project_collection_namespace_enabled or not project_id:
            return self.collection
        digest = hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:16]
        return f"{self.collection}__p_{digest}"

    async def _project_collection_exists(self, project_id: str) -> bool:
        if not self._cfg.project_collection_namespace_enabled:
            return True
        return await self._engine.collection_exists(self.collection_for_project(project_id))

    @property
    def semantic_vector_name(self) -> str:
        """Configured dense vector name."""

        return self._cfg.semantic_vector_name

    @property
    def bm25_vector_name(self) -> str:
        """Configured sparse vector name."""

        return self._cfg.bm25_vector_name

    async def _get_one_scoped(
        self, project_id: str, point_id: str, *, with_vectors: bool = False
    ) -> QdrantRecord | None:
        """Retrieve one point and return it only if it belongs to ``project_id``."""

        if not await self._project_collection_exists(project_id):
            return None
        records = await self._engine.retrieve(
            self.collection_for_project(project_id),
            [point_id],
            with_vectors=with_vectors,
        )
        return self._engine.first_project_match(records, project_id)

    async def _retrieve_scoped(
        self, project_id: str, point_ids: list[str], *, with_vectors: bool = False
    ) -> list[QdrantRecord]:
        """Retrieve points by id, keeping only those owned by ``project_id``."""

        if not await self._project_collection_exists(project_id):
            return []
        records = await self._engine.retrieve(
            self.collection_for_project(project_id),
            point_ids,
            with_vectors=with_vectors,
        )
        return [record for record in records if record.payload.get("project_id") == project_id]

    async def _scroll_scoped(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int,
        cursor: Any | None = None,
        order_by: Any | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll the collection inside one project."""

        if not await self._project_collection_exists(project_id):
            return [], None
        return await self._engine.scroll(
            self.collection_for_project(project_id),
            scroll_filter=self._engine.project_filter(project_id, filter_=filter_),
            limit=limit,
            offset=cursor,
            order_by=order_by,
            with_vectors=with_vectors,
        )

    async def _count_scoped(self, project_id: str, *, filter_: qmodels.Filter | None = None) -> int:
        """Count points inside one project."""

        if not await self._project_collection_exists(project_id):
            return 0
        return await self._engine.count(
            self.collection_for_project(project_id),
            count_filter=self._engine.project_filter(project_id, filter_=filter_),
            exact=True,
        )

    async def _ensure_project_vector_collection(
        self,
        project_id: str | None,
        *,
        vector_size: int,
        enable_sparse: bool,
        payload_indexes: list[PayloadIndexSpec],
        on_disk_payload: bool | None = None,
    ) -> str:
        """Ensure the vector collection for one project exists and return its name."""

        collection = self.collection_for_project(project_id)
        if not self._cfg.project_collection_namespace_enabled:
            return collection
        await self._engine.ensure_collection(
            QdrantCollectionSpec(
                name=collection,
                vector_size=vector_size,
                dense_vector_name=self.semantic_vector_name,
                sparse_vector_name=self.bm25_vector_name,
                distance=self._cfg.distance,  # type: ignore[arg-type]
                enable_dense=True,
                enable_sparse=enable_sparse,
                on_disk_payload=on_disk_payload,
                payload_indexes=payload_indexes,
            )
        )
        return collection

    async def _ensure_project_payload_collection(
        self,
        project_id: str | None,
        *,
        payload_indexes: list[PayloadIndexSpec],
        on_disk_payload: bool | None = None,
    ) -> str:
        """Ensure the payload-only collection for one project exists and return its name."""

        collection = self.collection_for_project(project_id)
        if not self._cfg.project_collection_namespace_enabled or not project_id:
            return collection
        await self._engine.ensure_collection(
            QdrantCollectionSpec(
                name=collection,
                vector_size=self._cfg.vector_size,
                dense_vector_name=self.semantic_vector_name,
                sparse_vector_name=self.bm25_vector_name,
                distance=self._cfg.distance,  # type: ignore[arg-type]
                enable_dense=False,
                enable_sparse=False,
                on_disk_payload=on_disk_payload,
                payload_indexes=payload_indexes,
            )
        )
        return collection

    async def _upsert_payload_points_by_project(
        self,
        points: list[tuple[str, dict[str, Any]]],
        *,
        payload_indexes: list[PayloadIndexSpec],
    ) -> None:
        """Upsert payload-only points into the physical collection for each project."""

        if not points:
            return
        grouped: dict[str | None, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for point_id, payload in points:
            project_id = payload.get("project_id")
            grouped[project_id if isinstance(project_id, str) and project_id else None].append((point_id, payload))
        for project_id, project_points in grouped.items():
            collection = await self._ensure_project_payload_collection(project_id, payload_indexes=payload_indexes)
            await self._engine.upsert(
                collection,
                [self._payload_point(point_id, payload) for point_id, payload in project_points],
            )

    async def _global_payload_collection_names(self) -> list[str]:
        """Return physical collections that may hold payload-only points for this repository."""

        if not self._cfg.project_collection_namespace_enabled:
            return [self.collection]
        names = set(await self._engine.collection_names())
        prefix = f"{self.collection}__p_"
        collections = [name for name in sorted(names) if name.startswith(prefix)]
        if self.collection in names:
            collections.insert(0, self.collection)
        return collections

    async def _scroll_payload_global(
        self,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll payload-only records across base and project-scoped collections."""

        if not self._cfg.project_collection_namespace_enabled:
            return await self._engine.scroll(
                self.collection,
                scroll_filter=filter_,
                limit=limit,
                offset=cursor,
                order_by=order_by,
            )

        collections = await self._global_payload_collection_names()
        if not collections:
            return [], None
        if isinstance(cursor, dict):
            collection_index = int(cursor.get("collection_index") or 0)
            offset = cursor.get("offset")
        else:
            collection_index = 0
            offset = cursor

        records: list[QdrantRecord] = []
        for index in range(collection_index, len(collections)):
            page, next_offset = await self._engine.scroll(
                collections[index],
                scroll_filter=filter_,
                limit=limit - len(records),
                offset=offset if index == collection_index else None,
                order_by=order_by,
            )
            records.extend(page)
            if next_offset is not None:
                return records, {"collection_index": index, "offset": next_offset}
            if len(records) >= limit:
                next_index = index + 1
                if next_index < len(collections):
                    return records, {"collection_index": next_index, "offset": None}
                return records, None
        return records, None

    async def _delete_payload_points_global(self, point_ids: list[str]) -> None:
        """Delete payload-only points by id from every collection that may contain them."""

        if not point_ids:
            return
        for collection in await self._global_payload_collection_names():
            await self._engine.delete(collection, point_ids)

    def _dense_point(self, point_id: str, vector: list[float] | None, payload: dict[str, Any]) -> qmodels.PointStruct:
        """Build a point carrying a single dense vector (zero-filled when absent)."""

        return qmodels.PointStruct(
            id=point_id,
            vector={self.semantic_vector_name: vector or self._engine.zero_vector()},
            payload=self._engine.safe_payload(payload),
        )

    def _payload_point(self, point_id: str, payload: dict[str, Any]) -> qmodels.PointStruct:
        """Build a vector-less point (payload + filter only)."""

        return qmodels.PointStruct(id=point_id, vector={}, payload=self._engine.safe_payload(payload))
