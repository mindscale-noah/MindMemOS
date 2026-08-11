"""Extract contrast, failure, and success experience sets."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable

from ....typing import Skill
from ..skill_grpo_with_replay_buffer.contracts import RolloutOutcome
from ..skill_grpo_with_replay_buffer.models import ChatModel, chat_content
from .contracts import ExperienceSource, ExtractedExperienceSet
from .prompts import cross_task_messages, failure_to_success_messages


class ExperienceExtractor:
    def __init__(self, chat_model: ChatModel, *, max_experiences: int, max_concurrency: int | None) -> None:
        self._chat_model = chat_model
        self._max_experiences = max_experiences
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

    async def extract(
        self,
        outcomes: list[RolloutOutcome],
        skill: Skill,
        *,
        mini_batch_size: int,
        success_reward: float,
    ) -> list[ExtractedExperienceSet]:
        grouped: dict[str, list[RolloutOutcome]] = defaultdict(list)
        task_order: list[str] = []
        for outcome in outcomes:
            task_id = outcome.spec.task.task_id
            if task_id not in grouped:
                task_order.append(task_id)
            if outcome.trajectory is not None:
                grouped[task_id].append(outcome)
        for items in grouped.values():
            items.sort(key=lambda item: item.spec.sample_index)

        jobs: list[Callable[[], Awaitable[ExtractedExperienceSet]]] = []
        for task_id in task_order:
            items = grouped[task_id]
            success_index = self._first_success_index(items, success_reward=success_reward)
            if success_index is not None and success_index > 0:
                jobs.append(self._contrast_job(task_id, items[: success_index + 1], skill))

        first_failures = [
            (task_id, failure)
            for task_id in task_order
            if (failure := self._first_failure(grouped[task_id], success_reward=success_reward)) is not None
        ]
        for group_index, group in enumerate(self._chunks(first_failures, mini_batch_size)):
            jobs.append(self._cross_task_job(ExperienceSource.FAILURE, group_index, group, skill))

        last_successes = [
            (task_id, success)
            for task_id in task_order
            if (success := self._last_success(grouped[task_id], success_reward=success_reward)) is not None
        ]
        for group_index, group in enumerate(self._chunks(last_successes, mini_batch_size)):
            jobs.append(self._cross_task_job(ExperienceSource.SUCCESS, group_index, group, skill))

        experiences = await asyncio.gather(*(job() for job in jobs))
        source_order = {ExperienceSource.CONTRAST: 0, ExperienceSource.FAILURE: 1, ExperienceSource.SUCCESS: 2}
        experiences.sort(key=lambda item: source_order[item.source])
        return experiences

    def _contrast_job(
        self, task_id: str, items: list[RolloutOutcome], skill: Skill
    ) -> Callable[[], Awaitable[ExtractedExperienceSet]]:
        trajectories = [item.trajectory for item in items if item.trajectory is not None]

        async def job() -> ExtractedExperienceSet:
            content = await self._call(
                task=f"skill_grpo_experience_validation.experience.contrast.{task_id}",
                messages=failure_to_success_messages(
                    skill=skill,
                    task_id=task_id,
                    trajectories=trajectories,
                    max_experiences=self._max_experiences,
                ),
            )
            return ExtractedExperienceSet(
                task_id=task_id,
                task_ids=[task_id],
                source=ExperienceSource.CONTRAST,
                content=content,
                rollout_count=len(trajectories),
            )

        return job

    def _cross_task_job(
        self,
        source: ExperienceSource,
        group_index: int,
        items: list[tuple[str, RolloutOutcome]],
        skill: Skill,
    ) -> Callable[[], Awaitable[ExtractedExperienceSet]]:
        trajectories = [(task_id, item.trajectory) for task_id, item in items if item.trajectory is not None]
        task_ids = [task_id for task_id, _ in trajectories]

        async def job() -> ExtractedExperienceSet:
            content = await self._call(
                task=f"skill_grpo_experience_validation.experience.{source.value}.{group_index}",
                messages=cross_task_messages(
                    skill=skill,
                    items=trajectories,
                    source=source.value,
                    max_experiences=self._max_experiences,
                ),
            )
            return ExtractedExperienceSet(
                task_id=f"{source.value}-mini-batch-{group_index + 1}",
                task_ids=task_ids,
                source=source,
                content=content,
                rollout_count=len(trajectories),
            )

        return job

    async def _call(self, *, task: str, messages: list[dict[str, str]]) -> str:
        async def call() -> str:
            return await chat_content(self._chat_model, task=task, messages=messages)

        if self._semaphore is None:
            return await call()
        async with self._semaphore:
            return await call()

    @staticmethod
    def _first_success_index(items: list[RolloutOutcome], *, success_reward: float) -> int | None:
        return next(
            (index for index, item in enumerate(items) if ExperienceExtractor.is_success(item, success_reward)),
            None,
        )

    @staticmethod
    def _first_failure(items: list[RolloutOutcome], *, success_reward: float) -> RolloutOutcome | None:
        return next((item for item in items if not ExperienceExtractor.is_success(item, success_reward)), None)

    @staticmethod
    def _last_success(items: list[RolloutOutcome], *, success_reward: float) -> RolloutOutcome | None:
        return next((item for item in reversed(items) if ExperienceExtractor.is_success(item, success_reward)), None)

    @staticmethod
    def is_success(item: RolloutOutcome, success_reward: float) -> bool:
        score = item.trajectory.reward.score if item.trajectory is not None else None
        return score is not None and score >= success_reward

    @staticmethod
    def _chunks(items: list[tuple[str, RolloutOutcome]], size: int) -> list[list[tuple[str, RolloutOutcome]]]:
        return [items[index : index + size] for index in range(0, len(items), size)]


__all__ = ["ExperienceExtractor"]
