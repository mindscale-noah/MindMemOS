"""Typed client for the ``/v1/memory/*`` API.

Holds default actor identity (``user_id`` / ``app_id`` / ``agent_id`` /
``session_id``) so callers don't repeat it on every call; any field can be
overridden per request. All HTTP I/O goes through :class:`HttpTransport`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..skills import SkillManager, detect_skill_context
from ..transport import HttpTransport
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


class MemoryClient:
    """Memory API resource client for add, search, and status operations."""

    def __init__(
        self,
        transport: HttpTransport,
        *,
        default_user_id: str | None = None,
        default_app_id: str | None = None,
        default_agent_id: str | None = None,
        default_session_id: str | None = None,
        skill_manager: SkillManager | None = None,
        memory_defaults: MemoryDefaults | None = None,
    ) -> None:
        self._transport = transport
        self._skill_manager = skill_manager
        base_defaults = memory_defaults or MemoryDefaults()
        self._defaults = replace(
            base_defaults,
            user_id=default_user_id if default_user_id is not None else base_defaults.user_id,
            app_id=default_app_id if default_app_id is not None else base_defaults.app_id,
            agent_id=default_agent_id if default_agent_id is not None else base_defaults.agent_id,
            session_id=default_session_id if default_session_id is not None else base_defaults.session_id,
        )
        self._core = MemoryCore(self._defaults)

    def add(
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
    ) -> AddResult:
        """Add content to the memory store."""
        messages = self._apply_default_role(messages)
        if self._skill_manager is not None:
            if self._defaults.add_auto_skill_context and skill_context is None:
                skill_context = self._detect_and_ensure_skill_context(messages)
            else:
                skill_context = self._ensure_provided_skill_context(skill_context)

        request = self._core.add(
            messages,
            user_id=user_id,
            mode=mode if mode is not None else self._defaults.add_mode,
            app_id=app_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
            skill_context=skill_context,
            score=score,
            task_id=task_id,
        )
        envelope = self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    def _apply_default_role(self, messages: list[Message | dict[str, Any]]) -> list[Message | dict[str, Any]]:
        """Fill the configured role only for raw dialogue-shaped dictionaries."""
        if not self._defaults.add_default_role:
            return messages
        result: list[Message | dict[str, Any]] = []
        for message in messages:
            if isinstance(message, dict) and "content" in message and "role" not in message:
                result.append({**message, "role": self._defaults.add_default_role})
            else:
                result.append(message)
        return result

    def _detect_and_ensure_skill_context(self, messages: list[Message | dict[str, Any]]) -> list[Any]:
        contexts = detect_skill_context(messages, registry=self._skill_manager.registry)
        return self._ensure_provided_skill_context(contexts)

    def _ensure_provided_skill_context(self, skill_context: list[Any]) -> list[Any]:
        ensured: list[Any] = []
        for raw_context in skill_context:
            context = raw_context if hasattr(raw_context, "name") else self._skill_context_from_dict(raw_context)
            skill_id = self._skill_manager.skill_id_for_context(context)
            if skill_id:
                ensured.append(self._skill_manager.ensure_skill_context(skill_id, usage=context.usage))
            else:
                ensured.append(context)
        if ensured:
            self._skill_manager.flush_pending_uploads()
        return ensured

    @staticmethod
    def _skill_context_from_dict(value: Any) -> Any:
        from ..skills import SkillContext

        return SkillContext.model_validate(value) if isinstance(value, dict) else value

    def search(
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
        )
        envelope = self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    def get(
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
        envelope = self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    def update(
        self,
        memory_id: str,
        content: str,
    ) -> StatusResult:
        """Update one memory by id."""
        request = self._core.update(memory_id, content)
        envelope = self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    def delete(
        self,
        memory_id: str,
    ) -> StatusResult:
        """Delete one memory by id."""
        request = self._core.delete(memory_id)
        envelope = self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    def feedback(
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
        envelope = self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)

    def dreaming(
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
        envelope = self._transport.post_envelope(request.path, json=request.body)
        return request.parse(envelope)
