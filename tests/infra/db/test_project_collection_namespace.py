import pytest
from mindmemos.infra.db.filters import build_filter, match_value
from qdrant_client import AsyncQdrantClient

from mindmemos.config import QdrantConfig
from mindmemos.infra.db import (
    AddRecordPoint,
    MemoryPoint,
    QdrantStore,
    SchemaAddBufferPoint,
    SearchRecordPoint,
    SparseVectorData,
)


def _memory_point(project_id: str, memory_id: str, vector: list[float]) -> MemoryPoint:
    return MemoryPoint(
        memory_id=memory_id,
        semantic_vector=vector,
        bm25_vector=SparseVectorData(indices=[1], values=[1.0]),
        payload={
            "memory_id": memory_id,
            "project_id": project_id,
            "content": f"{project_id} memory",
            "status": "active",
            "mem_type": "fact",
        },
    )


@pytest.mark.asyncio
async def test_project_collection_namespace_is_disabled_by_default() -> None:
    client = AsyncQdrantClient(":memory:")
    cfg = QdrantConfig(
        url="http://unused",
        memory_collection="default_memos",
        entity_collection="default_entities",
        source_collection="default_sources",
        vector_size=2,
    )
    store = QdrantStore(cfg, client=client)

    try:
        await store.ensure_schema()
        collections = {item.name for item in (await client.get_collections()).collections}
        assert "default_memos" in collections
        assert store.memory.collection_for_project("proj-a") == "default_memos"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_project_collection_namespace_allows_per_project_vector_dimensions() -> None:
    client = AsyncQdrantClient(":memory:")
    cfg = QdrantConfig(
        url="http://unused",
        memory_collection="tenant_memos",
        entity_collection="tenant_entities",
        source_collection="tenant_sources",
        vector_size=2,
        project_collection_namespace_enabled=True,
    )
    store = QdrantStore(cfg, client=client)

    try:
        await store.ensure_schema()
        await store.upsert_memory(_memory_point("proj-a", "00000000-0000-0000-0000-000000000001", [1.0, 0.0]))
        await store.upsert_memory(
            _memory_point("proj-b", "00000000-0000-0000-0000-000000000002", [1.0, 0.0, 0.0])
        )

        collections = {item.name for item in (await client.get_collections()).collections}
        assert "tenant_memos" not in collections
        assert store.memory.collection_for_project("proj-a") in collections
        assert store.memory.collection_for_project("proj-b") in collections
        assert store.memory.collection_for_project("proj-a") != store.memory.collection_for_project("proj-b")

        a_hits = await store.search_memory_dense(
            "proj-a",
            [1.0, 0.0],
            filter_=build_filter(must=[match_value("project_id", "proj-a")]),
            limit=1,
        )
        b_hits = await store.search_memory_dense(
            "proj-b",
            [1.0, 0.0, 0.0],
            filter_=build_filter(must=[match_value("project_id", "proj-b")]),
            limit=1,
        )
        assert [hit.point_id for hit in a_hits] == ["00000000-0000-0000-0000-000000000001"]
        assert [hit.point_id for hit in b_hits] == ["00000000-0000-0000-0000-000000000002"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_project_collection_namespace_scopes_payload_only_records() -> None:
    client = AsyncQdrantClient(":memory:")
    cfg = QdrantConfig(
        url="http://unused",
        memory_collection="tenant_memos",
        entity_collection="tenant_entities",
        source_collection="tenant_sources",
        add_record_collection="tenant_add_records",
        schema_add_buffer_collection="tenant_schema_add_buffer",
        search_record_collection="tenant_search_records",
        vector_size=2,
        project_collection_namespace_enabled=True,
    )
    store = QdrantStore(cfg, client=client)

    try:
        await store.ensure_schema()
        await store.upsert_add_record(
            AddRecordPoint(
                add_record_id="00000000-0000-0000-0000-000000000101",
                payload={
                    "add_record_id": "00000000-0000-0000-0000-000000000101",
                    "project_id": "proj-a",
                    "request_id": "00000000-0000-0000-0000-000000000101",
                    "status": "ok",
                },
            )
        )
        await store.upsert_schema_add_buffer_records(
            [
                SchemaAddBufferPoint(
                    schema_buffer_record_id="00000000-0000-0000-0000-000000000201",
                    payload={
                        "schema_buffer_record_id": "00000000-0000-0000-0000-000000000201",
                        "source_add_record_id": "00000000-0000-0000-0000-000000000101",
                        "project_id": "proj-a",
                        "buffer_key": "proj-a:session-a",
                        "buffer_status": "buffered",
                        "status": "queued",
                    },
                )
            ]
        )
        await store.upsert_search_record(
            SearchRecordPoint(
                search_record_id="00000000-0000-0000-0000-000000000301",
                payload={
                    "project_id": "proj-a",
                    "request_id": "00000000-0000-0000-0000-000000000301",
                    "status": "ok",
                },
            )
        )

        collections = {item.name for item in (await client.get_collections()).collections}
        assert store.add_record.collection_for_project("proj-a") in collections
        assert store.schema_add_buffer.collection_for_project("proj-a") in collections
        assert store.search_record.collection_for_project("proj-a") in collections

        add_records, _ = await store.scroll_add_records("proj-a")
        add_records_other, _ = await store.scroll_add_records("proj-b")
        schema_records, _ = await store.scroll_schema_add_buffer_records("proj-a")
        schema_records_other, _ = await store.scroll_schema_add_buffer_records("proj-b")
        search_records, _ = await store.scroll_search_records("proj-a")
        search_records_other, _ = await store.scroll_search_records("proj-b")

        assert [record.payload["request_id"] for record in add_records] == [
            "00000000-0000-0000-0000-000000000101"
        ]
        assert add_records_other == []
        assert [record.payload["schema_buffer_record_id"] for record in schema_records] == [
            "00000000-0000-0000-0000-000000000201"
        ]
        assert schema_records_other == []
        assert [record.payload["request_id"] for record in search_records] == [
            "00000000-0000-0000-0000-000000000301"
        ]
        assert search_records_other == []

        base_add_records, _ = await store.engine.scroll(
            store.add_record_collection,
            scroll_filter=None,
            limit=10,
        )
        base_schema_records, _ = await store.engine.scroll(
            store.schema_add_buffer_collection,
            scroll_filter=None,
            limit=10,
        )
        base_search_records, _ = await store.engine.scroll(
            store.search_record_collection,
            scroll_filter=None,
            limit=10,
        )
        assert base_add_records == []
        assert base_schema_records == []
        assert base_search_records == []

        await store.patch_schema_add_buffer_record(
            "proj-a",
            "00000000-0000-0000-0000-000000000201",
            {"status": "processing"},
        )
        await store.patch_schema_add_buffer_record(
            "proj-b",
            "00000000-0000-0000-0000-000000000201",
            {"status": "wrong-project"},
        )
        patched = await store.get_schema_add_buffer_records_by_ids(
            "proj-a",
            ["00000000-0000-0000-0000-000000000201"],
        )
        assert patched[0].payload["status"] == "processing"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_project_collection_namespace_disabled_keeps_payload_records_in_base_collection() -> None:
    client = AsyncQdrantClient(":memory:")
    cfg = QdrantConfig(
        url="http://unused",
        memory_collection="default_memos",
        entity_collection="default_entities",
        source_collection="default_sources",
        add_record_collection="default_add_records",
        schema_add_buffer_collection="default_schema_add_buffer",
        search_record_collection="default_search_records",
        vector_size=2,
        project_collection_namespace_enabled=False,
    )
    store = QdrantStore(cfg, client=client)

    try:
        await store.ensure_schema()
        await store.upsert_schema_add_buffer_records(
            [
                SchemaAddBufferPoint(
                    schema_buffer_record_id="00000000-0000-0000-0000-000000000401",
                    payload={
                        "schema_buffer_record_id": "00000000-0000-0000-0000-000000000401",
                        "project_id": "proj-a",
                        "buffer_key": "proj-a:session-a",
                        "buffer_status": "buffered",
                        "status": "queued",
                    },
                )
            ]
        )

        assert store.schema_add_buffer.collection_for_project("proj-a") == store.schema_add_buffer_collection

        base_records, _ = await store.engine.scroll(
            store.schema_add_buffer_collection,
            scroll_filter=None,
            limit=10,
        )
        scoped_records, _ = await store.scroll_schema_add_buffer_records("proj-a")
        collections = {item.name for item in (await client.get_collections()).collections}

        assert [record.payload["schema_buffer_record_id"] for record in base_records] == [
            "00000000-0000-0000-0000-000000000401"
        ]
        assert [record.payload["schema_buffer_record_id"] for record in scoped_records] == [
            "00000000-0000-0000-0000-000000000401"
        ]
        assert not any(name.startswith("default_schema_add_buffer__p_") for name in collections)
    finally:
        await client.close()
