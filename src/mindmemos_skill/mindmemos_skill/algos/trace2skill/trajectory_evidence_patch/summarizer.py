"""Concurrent analytical summarization of normalized trajectory evidence."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from ..contracts import TraceEvidence
from .models import TrajectorySummary
from .prompts import SUMMARY_SYSTEM, summarize_trajectory_user


class ChatModel(Protocol):
    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...


class TrajectoryEvidenceSummarizer:
    """Summarize independent trajectories with bounded concurrency."""

    def __init__(self, *, chat_model: ChatModel, task: str, concurrency: int) -> None:
        self._chat_model = chat_model
        self._task = task
        self._concurrency = concurrency

    async def summarize(
        self,
        skill_name: str,
        evidence: list[TraceEvidence],
    ) -> tuple[list[TrajectorySummary], list[str]]:
        semaphore = asyncio.Semaphore(self._concurrency)

        async def run(item: TraceEvidence) -> tuple[TrajectorySummary | None, str | None]:
            messages = [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": summarize_trajectory_user(skill_name, item.transcript)},
            ]
            try:
                async with semaphore:
                    response = await self._chat_model.chat(task=self._task, messages=messages)
            except Exception:
                return None, item.trajectory_id
            summary = (response.content or "").strip()
            if not summary:
                return None, item.trajectory_id
            return (
                TrajectorySummary(
                    trajectory_id=item.trajectory_id,
                    task_id=item.task_id,
                    summary=summary,
                    score=item.score,
                    annotation_detail=item.annotation_detail,
                    annotation_metadata=item.annotation_metadata,
                ),
                None,
            )

        results = await asyncio.gather(*(run(item) for item in evidence))
        summaries = [summary for summary, _ in results if summary is not None]
        failures = [trajectory_id for _, trajectory_id in results if trajectory_id is not None]
        return summaries, failures


__all__ = ["ChatModel", "TrajectoryEvidenceSummarizer"]
