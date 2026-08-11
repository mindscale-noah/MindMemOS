"""Reflexion-style feedback between consecutive rollouts of one task."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ....typing import Task, Trajectory
from ..skill_grpo_with_replay_buffer.models import ChatModel, chat_content

# Adapted from the official Reflexion reasoning and ALFWorld prompts:
# https://github.com/noahshinn/reflexion/blob/main/hotpotqa_runs/prompts.py
# https://github.com/noahshinn/reflexion/blob/main/alfworld_runs/generate_reflections.py
REFLECTION_PROMPT_VERSION = "reflexion-shinn-v1"
REFLECTION_SYSTEM_PROMPT = "You are an advanced reasoning agent that can improve based on self-reflection."
REFLECTION_USER_PROMPT = """You will be given a previous reasoning trial in which you attempted a task and were unsuccessful.
Do not merely summarize the trial. Diagnose the strategy and path taken, identify a likely reason for failure, and devise a concise new high-level plan that accounts for the mistake. Refer to specific actions when the trajectory supports them. Use a few complete sentences.

## Task
{task}

## Previous trial
{trajectory}

## Outcome
The attempt was unsuccessful with scalar reward {reward}.

## Reflection and new plan
"""


class ReflectionGenerator:
    """Turn one unsuccessful trajectory into bounded verbal feedback."""

    def __init__(
        self,
        chat_model: ChatModel,
        *,
        max_trajectory_chars: int,
        max_reflection_chars: int,
        max_concurrency: int | None,
    ) -> None:
        self._chat_model = chat_model
        self._max_trajectory_chars = max_trajectory_chars
        self._max_reflection_chars = max_reflection_chars
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

    async def reflect(self, trajectory: Trajectory, *, sample_index: int) -> str:
        messages = reflection_messages(trajectory, max_trajectory_chars=self._max_trajectory_chars)

        async def call() -> str:
            return await chat_content(
                self._chat_model,
                task=f"skill_grpo.reflection.{trajectory.task.task_id}.{sample_index}",
                messages=messages,
            )

        if self._semaphore is None:
            reflection = await call()
        else:
            async with self._semaphore:
                reflection = await call()
        return reflection[: self._max_reflection_chars].strip()


def reflection_messages(trajectory: Trajectory, *, max_trajectory_chars: int) -> list[dict[str, str]]:
    events = [event for event in trajectory.events if event.get("role") != "system"]
    rendered = json.dumps(events, ensure_ascii=False, indent=2, default=str)
    if len(rendered) > max_trajectory_chars:
        rendered = "[earlier trajectory content truncated]\n" + rendered[-max_trajectory_chars:]
    reward = trajectory.reward.score
    return [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REFLECTION_USER_PROMPT.format(
                task=trajectory.task.instruction,
                trajectory=rendered,
                reward="unavailable" if reward is None else reward,
            ),
        },
    ]


def previous_answer(trajectory: Trajectory, *, max_chars: int) -> str:
    """Return the final assistant message without exposing evaluator-only feedback."""

    for event in reversed(trajectory.events):
        model_response = event.get("model_response")
        if isinstance(model_response, str) and model_response:
            return model_response[-max_chars:]
        if event.get("role") != "assistant":
            continue
        content: Any = event.get("content")
        if isinstance(content, str) and content:
            return content[-max_chars:]
        return json.dumps(event, ensure_ascii=False, default=str)[-max_chars:]
    return "(no assistant answer was recorded)"


def task_with_reflection(
    task: Task,
    *,
    answer: str,
    reflection: str,
) -> Task:
    """Attach the immediately preceding answer and reflection to a retry task."""

    context = (
        "## Previous unsuccessful rollout\n"
        f"Previous answer:\n{answer}\n\n"
        f"Reflection:\n{reflection}\n\n"
        "Use this evidence to avoid repeating the same failure and solve the original task again."
    )
    metadata = dict(task.metadata)
    question = metadata.get("question")
    if isinstance(question, str):
        # LiveMath renders metadata.question rather than Task.instruction.
        metadata["question"] = f"{question.rstrip()}\n\n{context}"
    return task.model_copy(
        update={
            "instruction": f"{task.instruction.rstrip()}\n\n{context}",
            "metadata": metadata,
        },
        deep=True,
    )


__all__ = [
    "REFLECTION_PROMPT_VERSION",
    "REFLECTION_SYSTEM_PROMPT",
    "ReflectionGenerator",
    "previous_answer",
    "reflection_messages",
    "task_with_reflection",
]
