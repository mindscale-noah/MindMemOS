"""Project-scoped management reads backed by the memory database."""

from __future__ import annotations

from ...mappers import parse_search_dsl
from ...typing import (
    FieldCondition,
    GetPipelineInput,
    GetPipelineResult,
    MemoryListPipelineInput,
    MemoryListPipelineResult,
    MemoryRequestContext,
    MemoryScrollPipelineInput,
    MemoryScrollPipelineResult,
    MemorySearchItem,
    MemoryView,
    SearchFilter,
)
from ..utils import format_datetime, format_memory_event_time, format_source_timestamp
from .reader import MemoryDbReader


class MemoryCatalog:
    """Format project-scoped memory reads for management endpoints."""

    def __init__(self, *, reader: MemoryDbReader | None = None) -> None:
        self.reader = reader or MemoryDbReader()

    async def get(self, inp: GetPipelineInput, context: MemoryRequestContext) -> GetPipelineResult:
        """Return active memories matching an unscored filter."""

        limit_kwargs = {"limit": inp.top_k} if inp.top_k is not None else {}
        memories, _ = await self.reader.list_memories(
            context,
            filters=_active_filter(inp),
            **limit_kwargs,
        )
        return GetPipelineResult(
            status="ok",
            memories=[_to_memory_search_item(memory) for memory in memories],
            message=None,
        )

    async def list(
        self,
        inp: MemoryListPipelineInput,
        context: MemoryRequestContext,
    ) -> MemoryListPipelineResult:
        """Return a management page from the project-scoped catalog."""

        filters = _active_filter(inp)
        offset = (inp.page - 1) * inp.page_size
        read_limit = offset + inp.page_size + 1
        memories, _ = await self.reader.list_memories(
            context,
            filters=filters,
            limit=read_limit,
        )
        page_memories = memories[offset : offset + inp.page_size]
        total = await self.reader.count_memories(context, filters=filters) if inp.include_total else None
        has_more = (inp.page * inp.page_size < total) if total is not None else len(memories) > offset + inp.page_size
        return MemoryListPipelineResult(
            status="ok",
            memories=[_to_memory_search_item(memory) for memory in page_memories],
            page=inp.page,
            page_size=inp.page_size,
            total=total,
            has_more=has_more,
            message=None,
        )

    async def scroll(
        self,
        inp: MemoryScrollPipelineInput,
        context: MemoryRequestContext,
    ) -> MemoryScrollPipelineResult:
        """Return a cursor page from the project-scoped catalog."""

        memories, next_cursor = await self.reader.list_memories(
            context,
            filters=_active_filter(inp),
            limit=inp.limit,
            cursor=inp.cursor,
        )
        return MemoryScrollPipelineResult(
            status="ok",
            memories=[_to_memory_search_item(memory) for memory in memories],
            next_cursor=str(next_cursor) if next_cursor is not None else None,
            message=None,
        )


def _active_filter(inp: GetPipelineInput) -> SearchFilter:
    base = parse_search_dsl(inp.filters)
    if getattr(inp, "include_inactive", False):
        return base
    return SearchFilter(
        must=[
            *base.must,
            FieldCondition(field="status", op="match", value="active"),
        ],
        should=base.should,
        must_not=base.must_not,
    )


def _to_memory_search_item(memory: MemoryView) -> MemorySearchItem:
    updated_at = memory.update_at or memory.created_at
    return MemorySearchItem(
        id=memory.memory_id,
        memory=memory.content,
        memory_type=memory.mem_type,
        last_update_at=format_datetime(updated_at),
        event_time=format_memory_event_time(memory),
        source_timestamp=format_source_timestamp(memory),
        metadata=dict(memory.metadata or {}),
        status=memory.status,
        entity_id=memory.entity_id,
        entity_type=memory.entity_type,
        property_name=memory.property_name,
    )
