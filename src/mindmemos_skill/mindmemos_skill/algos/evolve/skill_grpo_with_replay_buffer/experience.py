"""Task-grouped experience extraction."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from ....typing import Skill, Trajectory
from .contracts import ExtractedExperience
from .models import ChatModel, chat_content
from .prompts import experience_extraction_messages


class ExperienceExtractor:
    def __init__(self, chat_model: ChatModel, *, max_experiences: int, max_concurrency: int | None) -> None:
        self._chat_model = chat_model
        self._max_experiences = max_experiences
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

    async def extract(self, trajectories: list[Trajectory], skill: Skill) -> list[ExtractedExperience]:
        grouped: dict[str, list[Trajectory]] = defaultdict(list)
        for trajectory in trajectories:
            grouped[trajectory.task.task_id].append(trajectory)

        async def one(task_id: str, items: list[Trajectory]) -> ExtractedExperience:
            async def call() -> str:
                return await chat_content(
                    self._chat_model,
                    task=f"skill_grpo.experience.{task_id}",
                    messages=experience_extraction_messages(
                        task=items[0].task,
                        skill=skill,
                        trajectories=items,
                        max_experiences=self._max_experiences,
                    ),
                )

            if self._semaphore is None:
                content = await call()
            else:
                async with self._semaphore:
                    content = await call()
            return ExtractedExperience(task_id=task_id, content=content, rollout_count=len(items))

        return await asyncio.gather(*(one(task_id, items) for task_id, items in grouped.items()))


__all__ = ["ExperienceExtractor"]
