"""Concurrent per-trajectory key-point summarization."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ...trace2skill.evidence import render_trajectory
from ..skill_grpo_with_replay_buffer.models import ChatModel
from .models import TrajectoryKeyPoints
from .prompts import SUMMARY_SYSTEM, summarize_trajectory_user


class TrajectoryKeyPointSummarizer:
    def __init__(self, *, chat_model: ChatModel, concurrency: int, transcript_max_chars: int) -> None:
        self._chat_model = chat_model
        self._concurrency = concurrency
        self._transcript_max_chars = transcript_max_chars

    async def summarize(self, outcomes: list[Any], *, skill_name: str) -> tuple[list[TrajectoryKeyPoints], list[str]]:
        semaphore = asyncio.Semaphore(self._concurrency)

        async def run(outcome: Any) -> tuple[TrajectoryKeyPoints | None, str | None]:
            trajectory = outcome.trajectory
            if trajectory is None:
                return None, outcome.spec.rollout_id
            messages = [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {
                    "role": "user",
                    "content": summarize_trajectory_user(
                        skill_name=skill_name,
                        trajectory_id=trajectory.trajectory_id,
                        task_instruction=trajectory.task.instruction,
                        transcript=render_trajectory(trajectory.events, self._transcript_max_chars),
                    ),
                },
            ]
            try:
                async with semaphore:
                    response = await self._chat_model.chat(
                        task=f"task_virtual_skill.summary.{trajectory.trajectory_id}",
                        messages=messages,
                        format_parser=parse_summary,
                        feedback_on_parse_error=True,
                    )
                parsed = _parsed(response, parse_summary)
            except Exception:
                return None, trajectory.trajectory_id
            return (
                TrajectoryKeyPoints(
                    trajectory_id=trajectory.trajectory_id,
                    task_id=trajectory.task.task_id,
                    score=trajectory.reward.score,
                    **parsed,
                ),
                None,
            )

        results = await asyncio.gather(*(run(outcome) for outcome in outcomes))
        return (
            [summary for summary, _ in results if summary is not None],
            [trajectory_id for _, trajectory_id in results if trajectory_id is not None],
        )


def parse_summary(value: str) -> dict[str, Any]:
    text = value.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced is not None:
        text = fenced.group(1)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("trajectory summary must be a JSON object")
    required = {"task_goal", "task_family", "key_actions", "turning_points", "skill_usage", "outcome"}
    if set(payload) != required:
        raise ValueError(f"trajectory summary keys must be exactly: {', '.join(sorted(required))}")
    return payload


def _parsed(response: Any, parser: Any) -> dict[str, Any]:
    if isinstance(response, str):
        return parser(response)
    if isinstance(response, dict):
        return response.get("parsed") or parser(str(response.get("content") or ""))
    return getattr(response, "parsed", None) or parser(str(getattr(response, "content", "") or ""))


__all__ = ["TrajectoryKeyPointSummarizer", "parse_summary"]
