"""Run-scoped bounded I/O and physical-call auditing, without tree policy."""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ....agents import OpenAIChatAgent
from ....common.checkpoint import boundary
from ....envs.base import Environment
from ....envs.typing import RolloutResult
from ....skill import SkillSet
from .artifacts import record
from .checkpoint import recovery_event
from .config import SkillForestTrainerConfig
from .env_adapter import SkillForestEnvAdapter
from .scheduler import RolloutScheduler


class ForestRuntime:
    """Share one set of concurrency limits across the complete forest run."""

    def __init__(
        self,
        config: SkillForestTrainerConfig,
        *,
        agent_factory=OpenAIChatAgent,
        work_dir: Path | None = None,
        env_adapter: SkillForestEnvAdapter | None = None,
        resume: bool = False,
    ):
        """Create bounded execution resources without starting model calls.

        Args:
            config: Forest-wide physical call limits and optimizer configuration.
            resume: Reuse completed requests after validating their identities.
            agent_factory: Injectable transport constructor for offline verification.
            work_dir: Optional run directory for scheduler diagnostics.
            env_adapter: Adapter bound by the trainer, or constructed from config.
        """
        self.resume = resume
        self.work_dir = work_dir
        self.config = config
        self.env_adapter = (
            env_adapter
            if env_adapter is not None
            else SkillForestEnvAdapter.build(config.env_adapter.class_name, config.env_adapter.config)
        )
        self.agent_factory = agent_factory
        self.model_slots = asyncio.Semaphore(config.model_concurrency)
        self.pool = ThreadPoolExecutor(max_workers=config.rollout_concurrency + config.model_concurrency)
        self.scheduler = RolloutScheduler(
            self._rollout,
            load_result=self._load_rollout if resume else None,
            concurrency=config.rollout_concurrency,
            log_path=work_dir / "scheduler.jsonl" if work_dir is not None else None,
        )

    def _rollout_identity(self, env, sample, skill):
        """Bind cached results to complete task, reward, Skill and environment inputs."""
        return {
            "sample": sample.model_dump(mode="json"),
            "skill": skill.snapshot().model_dump(mode="json"),
            "environment": f"{type(env).__module__}.{type(env).__qualname__}",
            "config": env.config.model_dump(mode="json", exclude={"agent": {"api_key"}})
            if isinstance(env, Environment)
            else None,
        }

    def _load_rollout(self, *, env, sample, skill, directory):
        """Load completed tasks before worker allocation and early rejection."""
        saved = boundary(directory / "completed.json", self._rollout_identity(env, sample, skill))
        if saved is None:
            return None
        result_type = env.result_type if isinstance(env, Environment) else RolloutResult
        result = result_type.model_validate(saved["result"])
        if self.work_dir is not None:
            recovery_event(self.work_dir, "rollout_reused", task_id=sample.task.id, directory=str(directory))
        return result

    def _archive_call(self, directory):
        """Preserve previous physical attempts before retrying unfinished work."""
        path = directory / "call.json"
        if path.exists():
            destination = directory / "attempts" / uuid4().hex
            destination.mkdir(parents=True)
            path.replace(destination / "call.json")

    async def _execute(self, function, **kwargs):
        """Drain an already started synchronous operation before releasing its slot."""
        call = functools.partial(function, **kwargs)
        future = asyncio.get_running_loop().run_in_executor(self.pool, copy_context().run, call)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await future
            raise

    async def _rollout(self, *, env, sample, skill, directory: Path):
        """Run and audit one physical environment call under the shared cap.

        Args:
            env: Caller-owned synchronous environment.
            sample: Public task and private scoring data.
            skill: Complete candidate passed through SkillSet.
            directory: Unique directory for this task and candidate.

        Returns:
            The environment's original rollout result.
        """
        self._archive_call(directory)
        start = perf_counter()
        payload = {"task_id": sample.task.id, "content_hash": skill.bundle.content_hash}
        record(directory / "call.json", {**payload, "status": "running"})
        try:
            result = await self._execute(
                env.rollout,
                task=sample.task,
                reward_data=sample.reward_data,
                skill_set=SkillSet({"candidate": skill}, max_concurrency=1),
                work_dir=directory,
                resume=False,
            )
        except BaseException as exc:
            record(
                directory / "call.json",
                {
                    **payload,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_s": perf_counter() - start,
                },
            )
            raise
        record(
            directory / "call.json",
            {
                **payload,
                "status": "completed",
                "duration_s": perf_counter() - start,
                "result": result.model_dump(mode="json"),
            },
        )
        boundary(
            directory / "completed.json",
            self._rollout_identity(env, sample, skill),
            payload={"result": result.model_dump(mode="json")},
        )
        return result

    async def chat(self, *, messages: list[dict[str, str]], directory: Path) -> str:
        """Reuse a matching response or make one bounded, audited optimizer call.

        Args:
            messages: Complete stage input constructed by the environment adapter.
            directory: Unique call audit directory.

        Returns:
            Nonempty text returned by the optimizer.
        """
        identity = {
            "messages": messages,
            "optimizer": self.config.optimizer.model_dump(mode="json", exclude={"api_key"}),
        }
        if self.resume:
            saved = boundary(directory / "completed.json", identity)
            if saved is not None:
                if self.work_dir is not None:
                    recovery_event(self.work_dir, "optimizer_reused", directory=str(directory))
                return saved["output"]
        self._archive_call(directory)
        async with self.model_slots:
            payload = {"messages": messages}
            record(directory / "call.json", {**payload, "status": "running"})
            start = perf_counter()
            try:
                agent = self.agent_factory(self.config.optimizer)
                result = await self._execute(agent.run, messages=messages)
                if not isinstance(result.output, str) or not result.output.strip():
                    raise ValueError("Optimizer must return nonempty text")
            except BaseException as exc:
                record(
                    directory / "call.json",
                    {
                        **payload,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "duration_s": perf_counter() - start,
                    },
                )
                raise
            record(
                directory / "call.json",
                {
                    **payload,
                    "status": "completed",
                    "duration_s": perf_counter() - start,
                    "response": result.model_dump(mode="json"),
                },
            )
            boundary(directory / "completed.json", identity, payload={"output": result.output})
            return result.output

    async def close(self) -> None:
        """Drain owned workers; leave the caller-owned environment open."""
        try:
            await self.scheduler.close()
        finally:
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)


async def parallel(coroutines):
    """Await sibling work in order and cancel/drain it if any operation fails.

    Args:
        coroutines: Independent operations belonging to the same training phase.

    Returns:
        Results in the original scheduling order.
    """
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(coroutine) for coroutine in coroutines]
    return [task.result() for task in tasks]
