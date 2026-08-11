"""Shared base for per-collection Qdrant repositories.

Each table under ``collections/`` binds one logical Qdrant collection and adds
its own typed upsert/read methods. A logical vector collection may map to a
small set of dimension-compatible physical collections; payload-only data stays
in its base collection. The cross-cutting mechanics — project-scoped
retrieve/scroll and point-struct building — live here so concrete repositories
stay small and free of duplication. All low-level work is delegated to the
shared :class:`QdrantEngine`.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ....config import QdrantConfig
from ..engine import QdrantEngine
from ..models import PayloadIndexSpec, QdrantCollectionSpec, QdrantRecord


class CollectionRepository:
    """Typed adapter bound to one logical Qdrant collection."""

    def __init__(self, engine: QdrantEngine, cfg: QdrantConfig) -> None:
        self._engine = engine
        self._cfg = cfg

    @property
    def collection(self) -> str:
        """Configured collection name (bound by the subclass)."""

        raise NotImplementedError

    def collection_for_project(self, project_id: str | None) -> str:
        """Return the default physical collection for a project.

        Payload-only repositories always use their base collection. Vector
        repositories use the configured default dimension; operations that
        already have an actual vector call :meth:`collection_for_vector_size`.
        """

        del project_id
        if not self._cfg.project_collection_namespace_enabled or not self._is_vector_repository:
            return self.collection
        return self.collection_for_vector_size(self._cfg.vector_size)

    def collection_for_vector_size(self, vector_size: int) -> str:
        """Return the shared physical collection for a dense-vector dimension."""

        if not self._cfg.project_collection_namespace_enabled:
            return self.collection
        return f"{self.collection}__d_{vector_size}"

    @property
    def _is_vector_repository(self) -> bool:
        return self.collection in {
            getattr(self._cfg, "memory_collection", None),
            getattr(self._cfg, "entity_collection", None),
            getattr(self._cfg, "source_collection", None),
        }

    async def _dimension_collection_names(self) -> list[str]:
        """Return this repository's shared vector collections, ordered by dimension."""

        if not self._cfg.project_collection_namespace_enabled:
            return [self.collection] if await self._engine.collection_exists(self.collection) else []
        prefix = f"{self.collection}__d_"
        names = [name for name in await self._engine.collection_names() if name.startswith(prefix)]

        def dimension(name: str) -> int:
            suffix = name.removeprefix(prefix)
            return int(suffix) if suffix.isdigit() else 2**31 - 1

        return sorted(names, key=lambda name: (dimension(name), name))

    async def _collection_holding_project(
        self,
        project_id: str,
        *,
        vector_size: int | None = None,
    ) -> str | None:
        """Locate the shared vector collection containing one project's data."""

        if not self._cfg.project_collection_namespace_enabled or not self._is_vector_repository:
            return self.collection
        preferred = self.collection_for_vector_size(vector_size) if vector_size else None
        candidates = await self._dimension_collection_names()
        if preferred in candidates:
            candidates.remove(preferred)
            candidates.insert(0, preferred)
        for collection in candidates:
            count = await self._engine.count(
                collection,
                count_filter=self._engine.project_filter(project_id),
                exact=True,
            )
            if count:
                return collection
        return None

    async def _project_collection_exists(self, project_id: str) -> bool:
        if not self._cfg.project_collection_namespace_enabled:
            return True
        if not self._is_vector_repository:
            return True
        return await self._collection_holding_project(project_id) is not None

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

        collection = (
            await self._collection_holding_project(project_id)
            if self._is_vector_repository
            else self.collection
        )
        if collection is None:
            return None
        records = await self._engine.retrieve(
            collection,
            [point_id],
            with_vectors=with_vectors,
        )
        return self._engine.first_project_match(records, project_id)

    async def _retrieve_scoped(
        self, project_id: str, point_ids: list[str], *, with_vectors: bool = False
    ) -> list[QdrantRecord]:
        """Retrieve points by id, keeping only those owned by ``project_id``."""

        collection = (
            await self._collection_holding_project(project_id)
            if self._is_vector_repository
            else self.collection
        )
        if collection is None:
            return []
        records = await self._engine.retrieve(
            collection,
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

        collection = (
            await self._collection_holding_project(project_id)
            if self._is_vector_repository
            else self.collection
        )
        if collection is None:
            return [], None
        return await self._engine.scroll(
            collection,
            scroll_filter=self._engine.project_filter(project_id, filter_=filter_),
            limit=limit,
            offset=cursor,
            order_by=order_by,
            with_vectors=with_vectors,
        )

    async def _count_scoped(self, project_id: str, *, filter_: qmodels.Filter | None = None) -> int:
        """Count points inside one project."""

        collection = (
            await self._collection_holding_project(project_id)
            if self._is_vector_repository
            else self.collection
        )
        if collection is None:
            return 0
        return await self._engine.count(
            collection,
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
        """Ensure the dimension-compatible shared vector collection exists."""

        del project_id
        collection = self.collection_for_vector_size(vector_size)
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
        """Return the already bootstrapped shared payload-only collection."""

        del project_id, payload_indexes, on_disk_payload
        return self.collection

    async def _upsert_payload_points_by_project(
        self,
        points: list[tuple[str, dict[str, Any]]],
        *,
        payload_indexes: list[PayloadIndexSpec],
    ) -> None:
        """Upsert payload-only points into the shared base collection."""

        if not points:
            return
        collection = await self._ensure_project_payload_collection(None, payload_indexes=payload_indexes)
        await self._engine.upsert(
            collection,
            [self._payload_point(point_id, payload) for point_id, payload in points],
        )

    async def _global_payload_collection_names(self) -> list[str]:
        """Return physical collections that may hold payload-only points for this repository."""

        return [self.collection] if await self._engine.collection_exists(self.collection) else []

    async def _scroll_payload_global(
        self,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll payload-only records in the shared base collection."""

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
