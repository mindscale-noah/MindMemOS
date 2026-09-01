"""Repository for the ``entity_item_v1`` collection."""

from __future__ import annotations

from qdrant_client import models as qmodels

from ..filters import ENTITY_PAYLOAD_INDEX_SCHEMA
from ..models import EntityPoint, QdrantRecord, QdrantSearchRecord, SparseVectorData
from .base import CollectionRepository


class EntityRepository(CollectionRepository):
    """Typed adapter for ``entity_item_v1``."""

    @property
    def collection(self) -> str:
        return self._cfg.entity_collection

    async def upsert(self, points: list[EntityPoint]) -> None:
        """Upsert many entity points."""

        by_collection: dict[str, list[EntityPoint]] = {}
        for point in points:
            project_id = str(point.payload.get("project_id") or "")
            vector_size = len(point.vector or []) or point.semantic_dimension or self._cfg.vector_size
            collection = await self._ensure_project_vector_collection(
                project_id,
                vector_size=vector_size,
                enable_sparse=True,
                payload_indexes=list(ENTITY_PAYLOAD_INDEX_SCHEMA),
            )
            by_collection.setdefault(collection, []).append(point)
        for collection, collection_points in by_collection.items():
            vector_size = await self._engine.dense_vector_size(collection, self.semantic_vector_name)
            await self._engine.upsert(
                collection,
                [self._point(point, vector_size=vector_size) for point in collection_points],
            )

    async def get(self, project_id: str, entity_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one entity by project and id."""

        return await self._get_one_scoped(project_id, entity_id, with_vectors=with_vectors)

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
        """Search entities via dense semantic vector."""

        collection = self.collection_for_vector_size(len(vector))
        if self._cfg.project_collection_namespace_enabled and not await self._engine.collection_exists(collection):
            return []
        return await self._engine.query(
            collection,
            source="entity_semantic",
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
        """Search entities via sparse BM25 vector."""

        collections = await self._collections_holding_project(project_id)
        if not collections:
            return []
        hits: dict[str, QdrantSearchRecord] = {}
        for collection in collections:
            records = await self._engine.query(
                collection,
                source="entity_bm25",
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
        """Run Qdrant-side RRF over dense and sparse entity prefetches."""

        collection = self.collection_for_vector_size(len(dense_vector))
        if self._cfg.project_collection_namespace_enabled and not await self._engine.collection_exists(collection):
            return []
        scoped_filter = self._engine.project_filter(project_id, filter_=filter_)
        return await self._engine.query(
            collection,
            source="entity_rrf",
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

    def _point(self, point: EntityPoint, *, vector_size: int) -> qmodels.PointStruct:
        return qmodels.PointStruct(
            id=point.entity_id,
            vector=self._vectors(point, vector_size=vector_size),
            payload=self._engine.safe_payload(point.payload),
        )

    def _vectors(self, point: EntityPoint, *, vector_size: int) -> dict[str, list[float] | qmodels.SparseVector]:
        vectors: dict[str, list[float] | qmodels.SparseVector] = {
            self.semantic_vector_name: point.vector or [0.0] * vector_size
        }
        if point.bm25_vector is not None:
            vectors[self.bm25_vector_name] = self._engine.to_qdrant_sparse(point.bm25_vector)
        return vectors
