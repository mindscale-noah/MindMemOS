from datetime import UTC, datetime
from typing import Any

import pytest
from mindmemos.pipelines.memory_db import MemoryCatalog
from mindmemos.typing.memory import MemoryRequestContext, MemoryView, SearchFilter
from mindmemos.typing.service import MemoryListPipelineInput, MemoryScrollPipelineInput


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeReader:
    def __init__(self, memories: list[MemoryView]) -> None:
        self.memories = memories
        self.calls: list[tuple[str, SearchFilter | None, int, Any | None]] = []
        self.count_calls: list[tuple[str, SearchFilter | None]] = []

    async def list_memories(
        self,
        ctx: MemoryRequestContext,
        *,
        filters: SearchFilter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
    ) -> tuple[list[MemoryView], Any | None]:
        self.calls.append((ctx.project_id, filters, limit, cursor))
        start = int(cursor) if cursor is not None else 0
        end = start + limit
        next_cursor = str(end) if end < len(self.memories) else None
        return self.memories[start:end], next_cursor

    async def count_memories(
        self,
        ctx: MemoryRequestContext,
        *,
        filters: SearchFilter | None = None,
    ) -> int:
        self.count_calls.append((ctx.project_id, filters))
        return len(self.memories)


def make_memory(index: int) -> MemoryView:
    return MemoryView(
        memory_id=f"mem-{index}",
        project_id="proj-1",
        content=f"Memory {index}",
        mem_type="fact",
        status="active",
        created_at=datetime(2026, 1, index, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_catalog_lists_active_memories_with_page_metadata() -> None:
    reader = FakeReader([make_memory(index) for index in range(1, 6)])
    catalog = MemoryCatalog(reader=reader)

    result = await catalog.list(MemoryListPipelineInput(page=2, page_size=2), make_context())

    assert [item.id for item in result.memories] == ["mem-3", "mem-4"]
    assert result.total == 5
    assert result.has_more is True
    assert reader.calls[0][0] == "proj-1"
    assert reader.calls[0][2] == 5
    assert any(condition.field == "status" and condition.value == "active" for condition in reader.calls[0][1].must)
    assert len(reader.count_calls) == 1


@pytest.mark.asyncio
async def test_catalog_scrolls_active_memories_with_reader_cursor() -> None:
    reader = FakeReader([make_memory(index) for index in range(1, 5)])
    catalog = MemoryCatalog(reader=reader)

    first = await catalog.scroll(MemoryScrollPipelineInput(limit=2), make_context())
    second = await catalog.scroll(MemoryScrollPipelineInput(limit=2, cursor=first.next_cursor), make_context())

    assert [item.id for item in first.memories] == ["mem-1", "mem-2"]
    assert first.next_cursor == "2"
    assert [item.id for item in second.memories] == ["mem-3", "mem-4"]
    assert second.next_cursor is None
    assert reader.calls[1][3] == "2"
