# -*- coding: utf-8 -*-
"""Async counterpart of :class:`MemoryClient` for the ``/v1/memory/*`` API."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..config import HttpConnectionConfig
from ..connections import HttpConnection
from ..transport import AsyncHttpTransport
from .backends import AsyncMemoryBackend, HttpMemoryBackend
from .core import MemoryCore, MemoryDefaults
from .models import (
    AddMode,
    AddResult,
    FeedbackMode,
    GetResult,
    MemorySearchHit,
    Message,
    SearchResult,
    SearchStrategy,
    StatusResult,
)

_UNSET = object()


class AsyncMemoryClient:
    """Async memory API resource client."""

    def __init__(
        self,
        backend: AsyncHttpTransport | AsyncMemoryBackend,
        *,
        default_user_id: str | None = None,
        default_app_id: str | None = None,
        default_agent_id: str | None = None,
        default_session_id: str | None = None,
        memory_defaults: MemoryDefaults | None = None,
    ) -> None:
        if isinstance(backend, AsyncHttpTransport):
            connection = HttpConnection(
                HttpConnectionConfig(base_url="https://external.invalid"),
                transport=backend,
                owns_transport=False,
            )
            backend = HttpMemoryBackend(connection)
        if not isinstance(backend, AsyncMemoryBackend):
            raise TypeError("backend must be an AsyncHttpTransport or AsyncMemoryBackend")
        self._backend = backend
        base_defaults = memory_defaults or MemoryDefaults()
        self._defaults = replace(
            base_defaults,
            user_id=default_user_id if default_user_id is not None else base_defaults.user_id,
            app_id=default_app_id if default_app_id is not None else base_defaults.app_id,
            agent_id=default_agent_id if default_agent_id is not None else base_defaults.agent_id,
            session_id=default_session_id if default_session_id is not None else base_defaults.session_id,
        )
        self._core = MemoryCore(self._defaults)

    async def add(
        self,
        messages: list[Message | dict[str, Any]],
        *,
        user_id: str | None = None,
        mode: AddMode | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        skill_context: list[Any] | None = None,
        score: float | None = None,
        task_id: str | None = None,
        task: str | None = None,
    ) -> AddResult:
        """Add content to the memory store."""
        request = self._core.add(
            messages=messages,
            user_id=user_id,
            mode=mode if mode is not None else self._defaults.add_mode,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
            skill_context=skill_context,
            score=score,
            task_id=task_id,
            task=task,
        )
        return await self._backend.execute(request)

    async def search(
        self,
        query: str,
        *,
        top_k: int | None | object = _UNSET,
        user_id: str | None = None,
        search_strategy: SearchStrategy | None = None,
        rerank: bool | None = None,
        score_threshold: float | None | object = _UNSET,
        filters: dict[str, Any] | None | object = _UNSET,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        task_top_k: int | None | object = _UNSET,
    ) -> SearchResult:
        """Search memories."""
        request = self._core.search(
            query,
            top_k=self._defaults.search_top_k if top_k is _UNSET else top_k,
            user_id=user_id,
            search_strategy=search_strategy or self._defaults.search_strategy,
            rerank=self._defaults.search_rerank if rerank is None else rerank,
            score_threshold=(
                self._defaults.search_score_threshold if score_threshold is _UNSET else score_threshold
            ),
            filters=self._defaults.search_filters if filters is _UNSET else filters,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
            task_top_k=None if task_top_k is _UNSET else task_top_k,
        )
        return await self._backend.execute(request)

    async def get(
        self,
        *,
        filters: dict[str, Any] | None | object = _UNSET,
        top_k: int | None | object = _UNSET,
    ) -> GetResult:
        """List or filter memories in the current project."""
        request = self._core.get(
            filters=self._defaults.get_filters if filters is _UNSET else filters,
            top_k=self._defaults.get_top_k if top_k is _UNSET else top_k,
        )
        return await self._backend.execute(request)

    async def update(
        self,
        memory_id: str,
        content: str,
    ) -> StatusResult:
        """Update one memory by id."""
        request = self._core.update(memory_id, content)
        return await self._backend.execute(request)

    async def delete(
        self,
        memory_id: str,
    ) -> StatusResult:
        """Delete one memory by id."""
        request = self._core.delete(memory_id)
        return await self._backend.execute(request)

    async def dreaming(
        self,
        *,
        mode: AddMode | None = None,
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> StatusResult:
        """Trigger the dreaming pipeline."""
        request = self._core.dreaming(
            mode=mode if mode is not None else self._defaults.dreaming_mode,
            user_id=user_id,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        return await self._backend.execute(request)

    async def feedback(
        self,
        *,
        feedback: str | None = None,
        mode: FeedbackMode | None | object = _UNSET,
        messages: list[Message | dict[str, Any]] | None = None,
        recalled_memories: list[MemorySearchHit | dict[str, Any]] | None = None,
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> StatusResult:
        """Trigger the feedback workflow.

        Explicit feedback text requires messages context. Omit ``feedback`` to
        run implicit feedback from recent add records.
        """
        request = self._core.feedback(
            feedback=feedback,
            mode=self._defaults.feedback_mode if mode is _UNSET else mode,
            messages=messages,
            recalled_memories=recalled_memories,
            user_id=user_id,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        return await self._backend.execute(request)
