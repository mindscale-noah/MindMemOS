from datetime import UTC, datetime
from typing import Any

import pytest
from mindmemos.pipelines.get import DefaultGetPipeline
from mindmemos.typing.memory import MemoryRequestContext, MemoryView, SearchFilter
from mindmemos.typing.service import GetPipelineInput, MemoryListPipelineInput, MemoryScrollPipelineInput


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


class FakeWriter:
    pass


@pytest.mark.asyncio
async def test_get_returns_hydrated_memory_items() -> None:
    memory = MemoryView(
        memory_id="mem-1",
        project_id="proj-1",
        content="Project uses Qdrant.",
        mem_type="fact",
        status="active",
        metadata={"source_timestamp_ms": 1700000000000},
        validate_from=datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, 8, 30, tzinfo=UTC),
        update_at=datetime(2026, 1, 2, 9, 45, tzinfo=UTC),
    )
    reader = FakeReader([memory])
    pipeline = DefaultGetPipeline(db_reader=reader, db_writer=FakeWriter())

    result = await pipeline.get(GetPipelineInput(top_k=5), make_context())

    assert len(reader.calls) == 1
    project_id, sent_filter, limit, cursor = reader.calls[0]
    assert project_id == "proj-1"
    assert limit == 5
    assert cursor is None
    # An "active" status condition is always appended to the parsed filter.
    assert any(cond.field == "status" and cond.value == "active" for cond in sent_filter.must)
    assert result.status == "ok"
    assert result.message is None
    assert len(result.memories) == 1
    assert result.memories[0].id == "mem-1"
    assert result.memories[0].memory == "Project uses Qdrant."
    assert result.memories[0].memory_type == "fact"
    assert result.memories[0].last_update_at == "2026-01-02 09:45:00"
    assert result.memories[0].event_time == "2023-11-14 22:13:20"
    assert result.memories[0].source_timestamp == "2023-11-14 22:13:20"


@pytest.mark.asyncio
async def test_get_prefers_resolved_event_date_over_source_time() -> None:
    memory = MemoryView(
        memory_id="mem-1",
        project_id="proj-1",
        content="On 2023-01-19 (yesterday), Caroline visited an LGBTQ support group.",
        mem_type="episodic",
        status="active",
        metadata={
            "resolved_event_date": "2023-01-19",
            "event_time_text": "yesterday",
            "source_timestamp_ms": 1674172800000,
        },
        validate_from=datetime(2023, 1, 20, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, 8, 30, tzinfo=UTC),
    )
    pipeline = DefaultGetPipeline(db_reader=FakeReader([memory]), db_writer=FakeWriter())

    result = await pipeline.get(GetPipelineInput(filters={"memory_id": "mem-1"}), make_context())

    assert result.memories[0].event_time == "2023-01-19 00:00:00"
    assert result.memories[0].source_timestamp == "2023-01-20 00:00:00"


@pytest.mark.asyncio
async def test_get_returns_ok_with_empty_list_when_nothing_matches() -> None:
    reader = FakeReader([])
    pipeline = DefaultGetPipeline(db_reader=reader, db_writer=FakeWriter())

    result = await pipeline.get(GetPipelineInput(), make_context())

    # top_k defaults to None -> no explicit limit passed, reader uses its default (50).
    assert reader.calls[0][2] == 50
    assert result.status == "ok"
    assert result.memories == []
    assert result.message is None


@pytest.mark.asyncio
async def test_list_returns_page_metadata_and_total() -> None:
    memories = [
        MemoryView(
            memory_id=f"mem-{index}",
            project_id="proj-1",
            content=f"Memory {index}",
            mem_type="fact",
            status="active",
            created_at=datetime(2026, 1, index, tzinfo=UTC),
        )
        for index in range(1, 6)
    ]
    reader = FakeReader(memories)
    pipeline = DefaultGetPipeline(db_reader=reader, db_writer=FakeWriter())

    result = await pipeline.list(MemoryListPipelineInput(page=2, page_size=2, include_total=True), make_context())

    assert [item.id for item in result.memories] == ["mem-3", "mem-4"]
    assert result.page == 2
    assert result.page_size == 2
    assert result.total == 5
    assert result.has_more is True
    assert reader.calls[0][2] == 5
    assert len(reader.count_calls) == 1


@pytest.mark.asyncio
async def test_list_can_skip_total_count() -> None:
    reader = FakeReader(
        [
            MemoryView(
                memory_id=f"mem-{index}",
                project_id="proj-1",
                content=f"Memory {index}",
                mem_type="fact",
                status="active",
                created_at=datetime(2026, 1, index, tzinfo=UTC),
            )
            for index in range(1, 4)
        ]
    )
    pipeline = DefaultGetPipeline(db_reader=reader, db_writer=FakeWriter())

    result = await pipeline.list(MemoryListPipelineInput(page=1, page_size=2, include_total=False), make_context())

    assert [item.id for item in result.memories] == ["mem-1", "mem-2"]
    assert result.total is None
    assert result.has_more is True
    assert reader.count_calls == []


@pytest.mark.asyncio
async def test_scroll_returns_next_cursor() -> None:
    reader = FakeReader(
        [
            MemoryView(
                memory_id=f"mem-{index}",
                project_id="proj-1",
                content=f"Memory {index}",
                mem_type="fact",
                status="active",
                created_at=datetime(2026, 1, index, tzinfo=UTC),
            )
            for index in range(1, 5)
        ]
    )
    pipeline = DefaultGetPipeline(db_reader=reader, db_writer=FakeWriter())

    first = await pipeline.scroll(MemoryScrollPipelineInput(limit=2), make_context())
    second = await pipeline.scroll(MemoryScrollPipelineInput(limit=2, cursor=first.next_cursor), make_context())

    assert [item.id for item in first.memories] == ["mem-1", "mem-2"]
    assert first.next_cursor == "2"
    assert [item.id for item in second.memories] == ["mem-3", "mem-4"]
    assert second.next_cursor is None
