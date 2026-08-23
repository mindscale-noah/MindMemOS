"""Task experience search: match a task entity and return its one-hop experiences."""

from __future__ import annotations

import asyncio

from ...components.text import SparseVectorEncoder, TextPreprocessor, get_text_preprocessor
from ...llm import get_embed_client
from ...logging import get_logger, traced
from ...typing import (
    MemoryRequestContext,
    MemorySearchItem,
    MemoryView,
    SearchPipelineInput,
    SearchPipelineResult,
    TaskSearchEntity,
    TaskSearchGroup,
)
from ..base import MemoryPersistencePipelineMixin
from ..registry import register
from ..utils import format_datetime, format_memory_event_time, format_source_timestamp

logger = get_logger(__name__)
_CLIENT_UNSET = object()


def _try_get_embed():
    """Try to resolve the global embedding client; return None if unavailable."""
    try:
        return get_embed_client()
    except Exception:  # noqa: BLE001
        logger.debug("embed_client_not_available", exc_info=True)
        return None


def _to_item(memory: MemoryView) -> MemorySearchItem:
    return MemorySearchItem(
        id=memory.memory_id,
        memory=memory.content,
        memory_type=memory.mem_type,
        last_update_at=format_datetime(memory.update_at or memory.created_at),
        event_time=format_memory_event_time(memory, fallback_to_source_timestamp=True),
        source_timestamp=format_source_timestamp(memory),
    )


@register(type="search", name="task_experience_search")
class TaskExperienceSearchPipeline(MemoryPersistencePipelineMixin):
    """Find the task entity matching a task text and return the experiences it
    connects to via TASK_EXPERIENCE edges (one graph hop).

    The query is the task text; the answer is "this task's reusable lessons".
    """

    def __init__(
        self,
        *,
        top_k: int = 20,
        task_top_k: int = 3,
        score_threshold: float | None = 0.45,
        sparse_encoder: SparseVectorEncoder | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        embed_client=_CLIENT_UNSET,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._top_k = max(1, top_k)
        self._task_top_k = max(1, task_top_k)
        self._score_threshold = score_threshold
        self._sparse_encoder = sparse_encoder
        self._text_preprocessor = text_preprocessor
        self._embed_client = _try_get_embed() if embed_client is _CLIENT_UNSET else embed_client

    @classmethod
    def from_config(cls, config, **kwargs):
        threshold = getattr(
            getattr(getattr(config, "algo_config", None), "trajectory", None),
            "task_search_score_threshold",
            None,
        )
        if "score_threshold" not in kwargs:
            kwargs["score_threshold"] = threshold if threshold is not None else 0.45
        if "sparse_encoder" not in kwargs or "text_preprocessor" not in kwargs:
            text_config = getattr(getattr(config, "algo_config", None), "text_processing", None)
            kwargs.setdefault("sparse_encoder", SparseVectorEncoder(text_config))
            kwargs.setdefault("text_preprocessor", get_text_preprocessor(text_config))
        return cls(**kwargs)

    @traced("search.task_experience_search")
    async def search(
        self,
        request: SearchPipelineInput,
        context: MemoryRequestContext,
    ) -> SearchPipelineResult:
        task_text = (request.query or "").strip()
        if not task_text:
            return SearchPipelineResult(status="ok", memories=[])

        # Per-request score_threshold overrides the configured task-search floor.
        threshold = (
            request.score_threshold
            if request.score_threshold is not None
            else self._score_threshold
        )
        records = await self._recall_tasks(context, task_text, score_threshold=threshold)

        task_limit = max(1, request.task_top_k or self._task_top_k)
        exp_limit = request.top_k or self._top_k
        tasks: list[TaskSearchGroup] = []
        for record in records[:task_limit]:
            task_entity = TaskSearchEntity(
                entity_id=record.record_id,
                entity_name=record.payload.get("entity_name") or task_text,
                entity_type=record.payload.get("entity_type") or "task",
            )
            views = await self.persistence.get_task_experiences(context, task_entity.entity_id, limit=exp_limit)
            tasks.append(TaskSearchGroup(task_entity=task_entity, memories=[_to_item(view) for view in views]))

        top = tasks[0] if tasks else None
        return SearchPipelineResult(
            status="ok",
            memories=top.memories if top else [],
            task_entity=top.task_entity if top else None,
            tasks=tasks,
        )

    async def _recall_tasks(
        self, context: MemoryRequestContext, task_text: str, *, score_threshold: float | None
    ):
        """Hybrid task recall (dense + BM25 over the task entity table).

        Mirrors vanilla search: the query is embedded and BM25-encoded, and the
        two channels are fused over task entities. ``score_threshold`` drops
        candidates whose dense similarity is too low. When semantic retrieval is
        unavailable the persistence layer degrades to exact/substring lookup.
        """
        dense_vector: list[float] | None = None
        if self._embed_client is not None:
            try:
                response = await self._embed_client.embed(
                    task="memory.trajectory.task.recall",
                    text=task_text,
                )
                if response and response.embeddings and response.embeddings[0]:
                    dense_vector = list(response.embeddings[0])
            except Exception:  # noqa: BLE001
                logger.warning("task_experience_embed_failed", exc_info=True)

        sparse_indices: list[int] | None = None
        sparse_values: list[float] | None = None
        if self._sparse_encoder is not None and self._text_preprocessor is not None:
            try:
                preprocessed = await asyncio.to_thread(
                    self._text_preprocessor.preprocess_text,
                    task_text,
                    include_entities=False,
                )
                encoded = self._sparse_encoder.encode_query(list(preprocessed.tokens))
                sparse_indices = list(encoded.indices)
                sparse_values = list(encoded.values)
            except Exception:  # noqa: BLE001
                logger.warning("task_experience_sparse_failed", exc_info=True)

        # search_task_entities falls back to name lookup when no dense vector is
        # available; the score threshold (over dense similarity) is authoritative.
        return await self.persistence.search_task_entities(
            context,
            task_text,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=10,
            score_threshold=score_threshold,
        )


__all__ = ["TaskExperienceSearchPipeline"]