"""Repository for the ``memory_item_v1`` collection.

The only table carrying both a dense semantic vector and a sparse BM25 vector,
so dense/sparse/RRF search and single-request payload+vector patches live here
rather than on the generic base.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ..filters import MEMORY_PAYLOAD_INDEX_SCHEMA
from ..models import MemoryPoint, QdrantRecord, QdrantSearchRecord, SparseVectorData
from .base import CollectionRepository


class MemoryRepository(CollectionRepository):
    """Typed adapter for ``memory_item_v1``."""

    @property
    def collection(self) -> str:
        return self._cfg.memory_collection

    async def upsert(self, points: list[MemoryPoint]) -> None:
        """Upsert many memory points."""

        by_collection: dict[str, list[MemoryPoint]] = {}
        for point in points:
            project_id = str(point.payload.get("project_id") or "")
            vector_size = len(point.semantic_vector or []) or point.semantic_dimension or self._cfg.vector_size
            collection = await self._ensure_project_vector_collection(
                project_id,
                vector_size=vector_size,
                enable_sparse=True,
                payload_indexes=list(MEMORY_PAYLOAD_INDEX_SCHEMA),
                on_disk_payload=self._cfg.memory_on_disk_payload,
            )
            by_collection.setdefault(collection, []).append(point)
        for collection, collection_points in by_collection.items():
            vector_size = await self._engine.dense_vector_size(collection, self.semantic_vector_name)
            await self._engine.upsert(
                collection,
                [self._point(point, vector_size=vector_size) for point in collection_points],
            )

    async def get(self, project_id: str, memory_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one memory by project and id."""

        records = await self.get_many(project_id, [memory_id], with_vectors=with_vectors)
        return records[0] if records else None

    async def get_many(
        self, project_id: str, memory_ids: list[str], *, with_vectors: bool = False
    ) -> list[QdrantRecord]:
        """Retrieve memories by project and ids."""

        return await self._retrieve_scoped(project_id, memory_ids, with_vectors=with_vectors)

    async def search_dense(
        self,
        project_id: str,
        vector: list[float],
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search via dense semantic vector."""

        collection = self.collection_for_vector_size(len(vector))
        if self._cfg.project_collection_namespace_enabled and not await self._engine.collection_exists(collection):
            return []
        return await self._engine.query(
            collection,
            source="semantic",
            query=vector,
            using=self.semantic_vector_name,
            query_filter=self._engine.project_filter(project_id, filter_=filter_),
            limit=limit,
            with_payload=with_payload,
            score_threshold=score_threshold,
        )

    async def search_sparse(
        self,
        project_id: str,
        vector: SparseVectorData,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search via sparse BM25 vector."""

        collections = await self._collections_holding_project(project_id)
        if not collections:
            return []
        hits: dict[str, QdrantSearchRecord] = {}
        for collection in collections:
            records = await self._engine.query(
                collection,
                source="bm25",
                query=self._engine.to_qdrant_sparse(vector),
                using=self.bm25_vector_name,
                query_filter=self._engine.project_filter(project_id, filter_=filter_),
                limit=limit,
                with_payload=with_payload,
            )
            for record in records:
                current = hits.get(record.point_id)
                if current is None or record.score > current.score:
                    hits[record.point_id] = record
        return sorted(hits.values(), key=lambda record: record.score, reverse=True)[:limit]

    async def search_rrf(
        self,
        project_id: str,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Run Qdrant-side RRF over dense and sparse prefetches."""

        collection = self.collection_for_vector_size(len(dense_vector))
        if self._cfg.project_collection_namespace_enabled and not await self._engine.collection_exists(collection):
            return []
        scoped_filter = self._engine.project_filter(project_id, filter_=filter_)
        return await self._engine.query(
            collection,
            source="rrf",
            prefetch=[
                qmodels.Prefetch(
                    query=self._engine.to_qdrant_sparse(sparse_vector),
                    using=self.bm25_vector_name,
                    filter=scoped_filter,
                    limit=sparse_limit or max(limit * 3, 30),
                ),
                qmodels.Prefetch(
                    query=dense_vector,
                    using=self.semantic_vector_name,
                    filter=scoped_filter,
                    limit=dense_limit or max(limit * 3, 30),
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit,
            with_payload=with_payload,
        )

    async def update_payload(self, project_id: str, memory_id: str, payload: dict[str, Any]) -> None:
        """Set payload fields after project ownership is checked."""

        record = await self.get(project_id, memory_id)
        if record is None:
            return
        for collection in await self._collections_holding_record(project_id, memory_id):
            await self._engine.set_payload(collection, memory_id, payload)

    async def patch(
        self,
        project_id: str,
        memory_id: str,
        payload: dict[str, Any],
        *,
        dense_vector: list[float] | None = None,
        sparse_vector: SparseVectorData | None = None,
        record: QdrantRecord | None = None,
    ) -> None:
        """Apply a payload patch and optional vectors in one ``batch_update_points`` call.

        ``record`` may be passed by callers that already fetched the point (with
        project scoping) to skip a redundant ownership read; otherwise the point
        is fetched here.
        """

        if record is None:
            record = await self.get(project_id, memory_id)
        if record is None:
            return
        record_collections = await self._collections_holding_record(project_id, memory_id)
        target_collection: str | None = None
        if dense_vector is not None and self._cfg.project_collection_namespace_enabled:
            target_collection = await self._ensure_project_vector_collection(
                project_id,
                vector_size=len(dense_vector),
                enable_sparse=True,
                payload_indexes=list(MEMORY_PAYLOAD_INDEX_SCHEMA),
                on_disk_payload=self._cfg.memory_on_disk_payload,
            )
            if target_collection not in record_collections:
                await self.upsert(
                    [
                        MemoryPoint(
                            memory_id=memory_id,
                            payload={**record.payload, **payload},
                            semantic_vector=dense_vector,
                            semantic_dimension=len(dense_vector),
                            bm25_vector=sparse_vector,
                        )
                    ]
                )
                for collection in record_collections:
                    await self._engine.delete(collection, [memory_id])
                return
        operations: list[qmodels.UpdateOperation] = [
            qmodels.SetPayloadOperation(
                set_payload=qmodels.SetPayload(payload=self._engine.safe_payload(payload), points=[memory_id])
            )
        ]
        vectors: dict[str, list[float] | qmodels.SparseVector] = {}
        if dense_vector is not None:
            vectors[self.semantic_vector_name] = dense_vector
        if sparse_vector is not None:
            vectors[self.bm25_vector_name] = self._engine.to_qdrant_sparse(sparse_vector)
        if vectors:
            operations.append(
                qmodels.UpdateVectorsOperation(
                    update_vectors=qmodels.UpdateVectors(points=[qmodels.PointVectors(id=memory_id, vector=vectors)])
                )
            )
        for collection in record_collections:
            if target_collection is not None and collection != target_collection:
                await self._engine.delete(collection, [memory_id])
                continue
            await self._engine.batch_update(collection, operations)

    async def delete(self, project_id: str, memory_id: str) -> None:
        """Delete one memory after project ownership is checked."""

        record = await self.get(project_id, memory_id)
        if record is None:
            return
        for collection in await self._collections_holding_record(project_id, memory_id):
            await self._engine.delete(collection, [memory_id])

    async def scroll(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll memories in one project."""

        return await self._scroll_scoped(
            project_id, filter_=filter_, limit=limit, cursor=cursor, with_vectors=with_vectors
        )

    async def count(self, project_id: str, *, filter_: qmodels.Filter | None = None) -> int:
        """Count memories in one project."""

        return await self._count_scoped(project_id, filter_=filter_)

    def _point(self, point: MemoryPoint, *, vector_size: int) -> qmodels.PointStruct:
        return qmodels.PointStruct(
            id=point.memory_id,
            vector=self._vectors(point, vector_size=vector_size),
            payload=self._engine.safe_payload(point.payload),
        )

    def _vectors(self, point: MemoryPoint, *, vector_size: int) -> dict[str, list[float] | qmodels.SparseVector]:
        vectors: dict[str, list[float] | qmodels.SparseVector] = {
            self.semantic_vector_name: point.semantic_vector or [0.0] * vector_size
        }
        if point.bm25_vector is not None:
            vectors[self.bm25_vector_name] = self._engine.to_qdrant_sparse(point.bm25_vector)
        return vectors
