from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from mindmemos.infra.db import QdrantRecord
from mindmemos.pipelines.memory_db.vector_repair import (
    VectorRepairPolicy,
    VectorRepairRequest,
    VectorRepairService,
    _configured_embedding_identity,
)
from mindmemos.typing import MemoryRequestContext


def _ctx() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="repair-request",
        account_id="platform",
        project_id="proj-1",
        api_key_uuid="internal",
        scopes=["memory:read", "memory:write"],
    )


def _record(memory_id: str = "mem-1", *, metadata: dict | None = None) -> QdrantRecord:
    return QdrantRecord(
        point_id=memory_id,
        payload={
            "memory_id": memory_id,
            "project_id": "proj-1",
            "account_id": "acct-1",
            "api_key_uuid": "key-1",
            "user_id": "business-user",
            "session_id": "session-1",
            "content": "unchanged memory content",
            "metadata": {
                "vector_pending": True,
                "vector_expected_dimension": 1024,
                "vector_retry_count": 0,
                "vector_next_retry_at_ms": 0,
                **(metadata or {}),
            },
        },
    )


class FakeReader:
    def __init__(self, records: list[QdrantRecord]) -> None:
        self.records = {record.point_id: record for record in records}
        self.last_filter = None

    async def list_memory_records(self, ctx, *, filters=None, limit=50, cursor=None):
        self.last_filter = filters
        return list(self.records.values())[:limit], None

    async def get_memory_record(self, ctx, memory_id):
        return self.records.get(memory_id)

    async def count_memory_records(self, ctx, *, filters=None):
        return len(self.records)


class FakeWriter:
    def __init__(self, record: QdrantRecord, *, fail_content: bool = False) -> None:
        self.record = record
        self.fail_content = fail_content
        self.commands = []

    async def update_memory(self, ctx, command):
        self.commands.append((ctx, command))
        if command.refresh_vectors and self.fail_content:
            raise RuntimeError("provider secret should never be persisted: sk-private")
        metadata = dict(self.record.payload.get("metadata") or {})
        metadata.update(command.metadata_patch)
        self.record.payload["metadata"] = metadata
        return SimpleNamespace(changed=True)


@pytest.mark.asyncio
async def test_repairs_existing_content_with_stored_actor_and_clears_pending() -> None:
    record = _record()
    reader = FakeReader([record])
    writer = FakeWriter(record)
    seen_contexts = []

    async def provider_context(ctx):
        seen_contexts.append(ctx)
        return nullcontext()

    service = VectorRepairService(
        reader=reader,
        writer=writer,
        provider_context_factory=provider_context,
        now_ms=lambda: 1_000,
    )

    result = await service.repair(_ctx(), VectorRepairRequest(limit=10))

    assert result.repaired == 1
    assert result.failed == 0
    assert seen_contexts[0].account_id == "acct-1"
    assert seen_contexts[0].user_id == "business-user"
    content_commands = [command for _, command in writer.commands if command.refresh_vectors]
    assert len(content_commands) == 1
    assert content_commands[0].memory_id == "mem-1"
    assert content_commands[0].content is None
    assert content_commands[0].metadata_patch["vector_pending"] is False
    assert content_commands[0].metadata_patch["vector_retry_count"] == 0
    assert content_commands[0].metadata_patch["vector_repaired_at"] == 1_000


@pytest.mark.asyncio
async def test_failure_persists_only_stable_error_and_exponential_backoff() -> None:
    record = _record(metadata={"vector_retry_count": 2})
    writer = FakeWriter(record, fail_content=True)
    service = VectorRepairService(
        reader=FakeReader([record]),
        writer=writer,
        provider_context_factory=lambda ctx: _async_nullcontext(),
        policy=VectorRepairPolicy(base_backoff_seconds=10, max_backoff_seconds=60, max_attempts=5),
        now_ms=lambda: 2_000,
    )

    result = await service.repair(_ctx(), VectorRepairRequest(limit=10))

    assert result.failed == 1
    metadata_commands = [command for _, command in writer.commands if not command.refresh_vectors]
    assert len(metadata_commands) == 1
    patch = metadata_commands[0].metadata_patch
    assert patch["vector_pending"] is True
    assert patch["vector_retry_count"] == 3
    assert patch["vector_next_retry_at_ms"] == 42_000
    assert patch["vector_last_error_code"] == "embedding.provider_unavailable"
    assert "secret" not in repr(patch)
    assert "sk-private" not in repr(result)


@pytest.mark.asyncio
async def test_automatic_repair_skips_future_and_exhausted_records() -> None:
    future = _record("future", metadata={"vector_next_retry_at_ms": 5_000})
    exhausted = _record("exhausted", metadata={"vector_retry_count": 5})
    reader = FakeReader([future, exhausted])
    writer = FakeWriter(future)
    service = VectorRepairService(
        reader=reader,
        writer=writer,
        policy=VectorRepairPolicy(max_attempts=5),
        provider_context_factory=lambda ctx: _async_nullcontext(),
        now_ms=lambda: 1_000,
    )

    result = await service.repair(_ctx(), VectorRepairRequest(limit=10))

    assert result.repaired == 0
    assert result.failed == 0
    assert result.skipped == 2
    assert writer.commands == []


@pytest.mark.asyncio
async def test_explicit_force_repairs_non_pending_historical_memory() -> None:
    record = _record(metadata={"vector_pending": False, "vector_retry_count": 9})
    writer = FakeWriter(record)
    service = VectorRepairService(
        reader=FakeReader([record]),
        writer=writer,
        provider_context_factory=lambda ctx: _async_nullcontext(),
        now_ms=lambda: 7_000,
    )

    result = await service.repair(
        _ctx(),
        VectorRepairRequest(memory_ids=["mem-1"], force=True, limit=10),
    )

    assert result.repaired == 1
    assert writer.commands[0][1].refresh_vectors is True
    assert writer.commands[0][1].touch_update_at is False


def test_static_yaml_uses_qdrant_vector_size_when_endpoint_omits_dimensions(monkeypatch) -> None:
    from mindmemos.pipelines.memory_db import vector_repair as repair_module

    config = SimpleNamespace(
        provider_binding=SimpleNamespace(enabled=False),
        database=SimpleNamespace(qdrant=SimpleNamespace(vector_size=1536)),
        embed_model_router=SimpleNamespace(
            endpoints=[SimpleNamespace(model="text-embedding", transport="litellm", dimensions=None)]
        ),
    )
    monkeypatch.setattr(repair_module, "get_config", lambda: config)

    dimension, fingerprint = _configured_embedding_identity()

    assert dimension == 1536
    assert fingerprint


async def _async_nullcontext():
    return nullcontext()
