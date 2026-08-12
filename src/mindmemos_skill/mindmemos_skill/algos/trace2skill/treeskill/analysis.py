"""Outcome-aware trajectory analysis for TreeSkill."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol

from ....typing import Trajectory
from ..contracts import TraceEvidence
from .json_utils import parse_model
from .models import TrajectoryAnalysisRecord
from .prompts import ANALYSIS_SYSTEM_PROMPT, analysis_user_prompt


class ChatModel(Protocol):
    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...


class TrajectoryAnalyzer(Protocol):
    async def analyze(
        self,
        evidence: list[TraceEvidence],
        *,
        trajectories_by_id: Mapping[str, Trajectory],
    ) -> tuple[list[TrajectoryAnalysisRecord], list[str]]: ...


class TreeSkillTrajectoryAnalyzer:
    """Analyze independent physical trajectories with bounded concurrency."""

    def __init__(
        self,
        *,
        chat_model: ChatModel,
        task: str,
        concurrency: int,
        success_score_threshold: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._chat_model = chat_model
        self._task = task
        self._concurrency = concurrency
        self._success_score_threshold = success_score_threshold
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def analyze(
        self,
        evidence: list[TraceEvidence],
        *,
        trajectories_by_id: Mapping[str, Trajectory],
    ) -> tuple[list[TrajectoryAnalysisRecord], list[str]]:
        del trajectories_by_id
        semaphore = asyncio.Semaphore(self._concurrency)

        async def run(item: TraceEvidence) -> tuple[TrajectoryAnalysisRecord | None, str | None]:
            source = self._record_source(item)
            messages = [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": analysis_user_prompt(item, source=source)},
            ]
            try:
                async with semaphore:
                    response = await self._chat_model.chat(
                        task=self._task,
                        messages=messages,
                        format_parser=lambda text: parse_model(text, TrajectoryAnalysisRecord),
                        feedback_on_parse_error=True,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
                record = getattr(response, "parsed", None) or parse_model(
                    response.content or "", TrajectoryAnalysisRecord
                )
                if record.instance_id != item.trajectory_id:
                    raise ValueError("analysis instance_id does not match the source trajectory")
                if record.task_id != item.task_id or record.record_source != source:
                    raise ValueError("analysis task or outcome source does not match the source trajectory")
                return record, None
            except Exception:
                return None, item.trajectory_id

        results = await asyncio.gather(*(run(item) for item in evidence))
        records = [record for record, _ in results if record is not None]
        failures = [trajectory_id for _, trajectory_id in results if trajectory_id is not None]
        return records, failures

    def _record_source(self, evidence: TraceEvidence) -> str:
        if evidence.score is None:
            return "unlabeled"
        return "success" if evidence.score >= self._success_score_threshold else "error"


__all__ = ["ChatModel", "TrajectoryAnalyzer", "TreeSkillTrajectoryAnalyzer"]
