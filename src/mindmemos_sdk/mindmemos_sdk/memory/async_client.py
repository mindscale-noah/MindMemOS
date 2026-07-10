# -*- coding: utf-8 -*-
"""Async counterpart of :class:`MemoryClient` for the ``/v1/memory/*`` API."""

from __future__ import annotations

from typing import Any

from ..transport import AsyncHttpTransport
from .core import MemoryCore, MemoryDefaults
from .models import (
    AddMode,
    AddResult,
    FeedbackMode,
    GetResult,
    MemoryPageResult,
    MemoryScrollResult,
    MemorySearchHit,
    Message,
    SearchResult,
    SearchStrategy,
    StatusResult,
)


class AsyncMemoryClient:
    """Async memory API resource client."""

    def __init__(
        self,
        transport: AsyncHttpTransport,
        *,
        default_user_id: str | None = None,
        default_app_id: str | None = None,
        default_agent_id: str | None = None,
        default_session_id: str | None = None,
    ) -> None:
        self._transport = transport
        self._core = MemoryCore(
            MemoryDefaults(
                user_id=default_user_id,
                app_id=default_app_id,
                agent_id=default_agent_id,
                session_id=default_session_id,
            )
        )

    async def add(
        self,
        messages: list[Message | dict[str, Any]],
        *,
        user_id: str | None = None,
        mode: AddMode = "sync",
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        skill_context: list[Any] | None = None,
        score: float | None = None,
        task_id: str | None = None,
    ) -> AddResult:
        """Add content to the memory store."""
        request = self._core.add(
            messages=messages,
            user_id=user_id,
            mode=mode,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
            skill_context=skill_context,
            score=score,
            task_id=task_id,
        )
        envelope = await self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = 10,
        user_id: str | None = None,
        search_strategy: SearchStrategy = "fast",
        rerank: bool = False,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> SearchResult:
        """Search memories."""
        request = self._core.search(
            query,
            top_k=top_k,
            user_id=user_id,
            search_strategy=search_strategy,
            rerank=rerank,
            score_threshold=score_threshold,
            filters=filters,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        envelope = await self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    async def get(
        self,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> GetResult:
        """List or filter memories in the current project."""
        request = self._core.get(filters=filters, top_k=top_k)
        envelope = await self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    async def list(
        self,
        *,
        filters: dict[str, Any] | None = None,
        page: int = 1,
        page_size: int = 20,
        include_total: bool = True,
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryPageResult:
        """List memories using page/page_size pagination."""
        request = self._core.list(
            filters=filters,
            page=page,
            page_size=page_size,
            include_total=include_total,
            user_id=user_id,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        envelope = await self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    async def scroll(
        self,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        cursor: str | None = None,
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryScrollResult:
        """Scroll memories using an opaque cursor."""
        request = self._core.scroll(
            filters=filters,
            limit=limit,
            cursor=cursor,
            user_id=user_id,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        envelope = await self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    async def update(
        self,
        memory_id: str,
        content: str,
    ) -> StatusResult:
        """Update one memory by id."""
        request = self._core.update(memory_id, content)
        envelope = await self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    async def delete(
        self,
        memory_id: str,
        *,
        hard: bool | None = None,
    ) -> StatusResult:
        """Delete one memory by id."""
        request = self._core.delete(memory_id, hard=hard)
        envelope = await self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    async def dreaming(
        self,
        *,
        mode: AddMode = "async",
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> StatusResult:
        """Trigger the dreaming pipeline."""
        request = self._core.dreaming(
            mode=mode,
            user_id=user_id,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        envelope = await self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    async def feedback(
        self,
        *,
        feedback: str | None = None,
        mode: FeedbackMode | None = None,
        messages: list[Message | dict[str, Any]] | None = None,
        recalled_memories: list[MemorySearchHit | dict[str, Any]] | None = None,
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> StatusResult:
        """Trigger the feedback workflow."""
        request = self._core.feedback(
            feedback=feedback,
            mode=mode,
            messages=messages,
            recalled_memories=recalled_memories,
            user_id=user_id,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        envelope = await self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)
