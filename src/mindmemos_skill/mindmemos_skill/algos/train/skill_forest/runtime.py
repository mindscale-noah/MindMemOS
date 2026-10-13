"""Bounded model and rollout calls with node-local trajectory records."""

import asyncio
import json
import os
import shutil
from collections.abc import Coroutine, Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import JsonValue

from ....agents import Agent
from ....envs.base import Environment
from ....envs.data.typing import DataSample
from ....envs.typing import RolloutResult
from ....logging import get_logger
from ....skill import Skill, SkillSet
from .config import SkillForestTrainerConfig
from .env_adapter import SkillForestEnvAdapter
from .scheduler import RolloutScheduler

logger = get_logger(__name__)

ResultT = TypeVar("ResultT")


class ForestRuntime:
    """Share physical-call limits and workspace cleanup across the forest."""

    def __init__(
        self,
        config: SkillForestTrainerConfig,
        *,
        work_dir: Path | None = None,
        env_adapter: SkillForestEnvAdapter | None = None,
        resume: bool = False,
    ) -> None:
        """Create bounded execution resources without starting provider calls.

        Args:
            config: Forest-wide execution limits and workspace policy.
            work_dir: Run directory retained for callers of the public runtime.
            env_adapter: Run-local adapter selected from the environment; defaults to generic for standalone use.
            resume: Accepted for compatibility; individual calls are never cached.
        """
        self.config = config
        self.work_dir = work_dir
        self.resume = resume
        if env_adapter is not None:
            self.env_adapter = env_adapter
        else:
            self.env_adapter = SkillForestEnvAdapter.build("generic", {})
        self.metrics = {
            "model_calls": 0,
            "model_errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usage_reported_calls": 0,
            "proposed_patches": 0,
            "proposed_edits": 0,
            "invalid_patches": 0,
        }
        self.model_slots = asyncio.Semaphore(config.model_concurrency)
        self.scheduler = RolloutScheduler(self._rollout, concurrency=config.rollout_concurrency)

    @staticmethod
    def _trajectory_path(directory: Path) -> Path | None:
        """Map a task workspace to its owning node's trajectory stream."""
        parts = directory.parts
        if "workspaces" not in parts:
            return None
        index = len(parts) - 1 - parts[::-1].index("workspaces")
        if index + 1 >= len(parts):
            return None
        if any(part.startswith("selection-") for part in parts[index + 2 :]):
            return None
        name = {"train": "rollouts.jsonl", "eval": "evaluation.jsonl"}.get(parts[index + 1])
        return Path(*parts[:index]) / name if name is not None else None

    @staticmethod
    def _append_trajectory(path: Path, payload: dict[str, JsonValue]) -> None:
        """Append one complete JSON line without creating a companion lock file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(line)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("Could not append Skill-Forest trajectory")
                view = view[written:]
        finally:
            os.close(descriptor)

    async def _rollout(self, *, env: Environment, sample: DataSample, skill: Skill, directory: Path) -> RolloutResult:
        """Execute one task and store its complete trajectory on the owning node."""
        directory.mkdir(parents=True, exist_ok=True)
        try:
            kwargs = {
                "task": sample.task,
                "reward_data": sample.reward_data,
                "skill_set": SkillSet({"candidate": skill}, max_concurrency=1),
                "work_dir": directory,
                "resume": False,
            }
            result = await env.rollout(**kwargs)
            path = self._trajectory_path(directory)
            if path is not None:
                self._append_trajectory(
                    path,
                    {
                        "task_id": sample.task.id,
                        "skill_hash": skill.source.skill_hash,
                        "physical": True,
                        "result": result.model_dump(mode="json"),
                    },
                )
            return result
        finally:
            if self.config.cleanup_workspaces:
                shutil.rmtree(directory, ignore_errors=True)

    async def chat(self, *, messages: list[dict[str, str]], directory: Path) -> str:
        """Make one bounded optimizer call without a per-call checkpoint.

        Args:
            messages: Complete two-message optimizer request.
            directory: Logical generation stage retained by existing callers.

        Returns:
            Nonempty optimizer response text.
        """
        logger.info("[skill_forest.optimizer] started")
        async with self.model_slots:
            agent = Agent.from_config(self.config.optimizer)
            self.metrics["model_calls"] += 1
            try:
                result = await agent.run(messages=messages)
                if not isinstance(result.output, str) or not result.output.strip():
                    raise ValueError("Optimizer must return nonempty text")
            except BaseException:
                self.metrics["model_errors"] += 1
                raise
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                self.metrics[key] += int(result.usage.get(key, 0))
            self.metrics["usage_reported_calls"] += bool(result.usage)
            logger.info(f"[skill_forest.optimizer] progress | model_calls=1 | usage_reported={bool(result.usage)}")
            return result.output

    async def close(self) -> None:
        """Drain owned rollout workers without closing the caller's environment."""
        await self.scheduler.close()


async def parallel(coroutines: Iterable[Coroutine[Any, Any, ResultT]]) -> list[ResultT]:
    """Await sibling work in order and drain all calls when one fails.

    Args:
        coroutines: Independent operations in the same training phase.

    Returns:
        Results in submission order.
    """
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(coroutine) for coroutine in coroutines]
    return [task.result() for task in tasks]
