"""Round-robin rollout groups with completion-driven rejection and draining."""

import asyncio
import logging
from collections import deque
from collections.abc import Callable, Sequence
from contextvars import copy_context
from pathlib import Path
from time import perf_counter

from pydantic import JsonValue

from mindmemos_skill.common.ids import generate_id

from ....common.json import append_jsonl
from ....envs.base import Environment
from ....envs.data.typing import DataSample
from ....envs.typing import RolloutResult
from ....logging import get_logger
from ....logging.tracing import log_failure
from ....skill import Skill
from .typing import RolloutExecutor, RolloutLoader, StopPredicate


class RolloutGroup:
    """Execute one task group and stop queued work without cancelling physical calls."""

    def __init__(
        self,
        *,
        samples: Sequence[DataSample],
        skill: Skill,
        env: Environment,
        directory: Path,
        stop_when: StopPredicate | None,
        stage_name: str,
    ) -> None:
        """Initialize one caller-owned group and its completion signal.

        Args:
            samples: Ordered environment samples to execute once.
            skill: Candidate shared by this group's tasks.
            env: Caller-owned rollout environment.
            directory: Unique artifact directory identifying the group.
            stop_when: Optional predicate evaluated against completed results.
            stage_name: Explicit algorithm phase for this rollout group.
        """
        self.pending = deque(enumerate(samples))
        self.skill = skill
        self.env = env
        self.directory = directory
        self.stop_when = stop_when
        self.results: dict[str, RolloutResult] = {}
        self.active = 0
        self.stopped = False
        self.error: Exception | None = None
        self.done = asyncio.Event()
        self.submitted_at = perf_counter()
        self.context = copy_context()
        self.stage_name = stage_name

    def finish_if_drained(self) -> None:
        """Signal completion only after all started calls have returned."""
        if not self.pending and not self.active:
            self.done.set()


class RolloutScheduler:
    """Share fixed workers across groups, taking one task per group per turn."""

    def __init__(
        self,
        execute: RolloutExecutor,
        *,
        concurrency: int,
        log_path: Path | None = None,
        load_result: RolloutLoader | None = None,
    ) -> None:
        """Configure lazy workers and an optional append-only diagnostic log.

        Args:
            execute: Audited asynchronous single-rollout executor.
            load_result: Optional synchronous loader for completed matching requests.
            concurrency: Maximum number of simultaneously executing rollouts.
            log_path: Run-local scheduler JSONL path, or None for library callers.
        """
        self.load_result = load_result
        self.execute = execute
        self.concurrency = concurrency
        self.log_path = log_path
        self.queue: deque[RolloutGroup] = deque()
        self.groups: set[RolloutGroup] = set()
        self.wake = asyncio.Event()
        self.workers: list[asyncio.Task[None]] = []
        self.active = 0
        self.closed = False
        self.started_at = perf_counter()
        self.sequence = 0
        self.session_id = generate_id()

    @staticmethod
    def _log_failure(group: RolloutGroup) -> None:
        """Keep worker diagnostics in the submitting task's logging context."""
        group.context.run(
            log_failure,
            get_logger(__name__),
            "algorithm.item.failed",
            body={"operation": "scheduler", "stage": group.stage_name, "group_id": str(group.directory)},
        )

    async def _log(self, event: str, group: RolloutGroup, **details: JsonValue) -> None:
        """Record queue occupancy and relative timestamps without task content."""
        if event != "rollout_released":
            detail_text = " ".join(f"{key}={value}" for key, value in details.items())
            group.context.run(
                get_logger(__name__).log,
                logging.ERROR if event == "rollout_failed" else logging.INFO,
                f"[{group.stage_name}] {event} | group_id={group.directory}"
                f" | done={len(group.results)} running={group.active} pending={len(group.pending)}"
                f" | elapsed={perf_counter() - group.submitted_at:.1f}s | {detail_text}",
            )
        if self.log_path is None:
            return
        self.sequence += 1
        await append_jsonl(
            self.log_path,
            {
                "session_id": self.session_id,
                "sequence": self.sequence,
                "event": event,
                "elapsed_s": perf_counter() - self.started_at,
                "group": str(group.directory),
                "active": self.active,
                "queued": sum(len(item.pending) for item in self.queue),
                "groups": len(self.groups),
                "group_active": group.active,
                "group_queued": len(group.pending),
                **details,
            },
        )

    async def _stop(self, group: RolloutGroup, reason: str) -> None:
        """Remove only unstarted work and record the remaining drain size."""
        if group.stopped:
            return
        group.stopped = True
        removed = len(group.pending)
        group.pending.clear()
        self.queue = deque(item for item in self.queue if item is not group)
        group.finish_if_drained()
        await self._log("group_stopped", group, reason=reason, removed=removed)

    async def collect(
        self,
        *,
        env: Environment,
        samples: Sequence[DataSample],
        skill: Skill,
        directory: Path,
        stop_when: Callable[[dict[str, RolloutResult]], bool] | None = None,
        stage_name: str = "skill_forest.rollout",
    ) -> dict[str, RolloutResult]:
        """Submit a group and return all completed results after physical draining.

        Args:
            env: Caller-owned concurrent rollout environment.
            samples: Ordered unique task samples, including private scoring data.
            skill: Complete candidate to evaluate.
            directory: Unique group directory for numbered physical calls.
            stage_name: Training, root, candidate or selection evaluation phase.
            stop_when: Synchronous rejection predicate checked before submission
                and after each result, before that worker starts another task.

        Returns:
            Completed results keyed by task ID, including calls drained after rejection.

        Raises:
            RuntimeError: The scheduler has already closed.
            Exception: Physical execution, rejection predicate or logging failed.
        """
        if self.closed:
            raise RuntimeError("Rollout scheduler is closed")
        group = RolloutGroup(
            samples=samples, skill=skill, env=env, directory=directory, stop_when=stop_when, stage_name=stage_name
        )
        if self.load_result is not None:
            pending = deque()
            for index, sample in group.pending:
                result = self.load_result(env=env, sample=sample, skill=skill, directory=directory / f"{index:06d}")
                if result is None:
                    pending.append((index, sample))
                else:
                    group.results[sample.task.id] = result
            group.pending = pending
        self.groups.add(group)
        try:
            self.queue.append(group)
            await self._log("group_submitted", group, tasks=len(samples), reused=len(group.results))
            if stop_when is not None and stop_when(group.results):
                await self._stop(group, "early_rejection")
            elif not group.pending:
                self.queue.remove(group)
            group.finish_if_drained()
            if not self.workers:
                self.workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]
            self.wake.set()
            await group.done.wait()
            if group.error is not None:
                raise group.error
            await self._log("group_completed", group, completed=len(group.results), stopped=group.stopped)
            return group.results
        except BaseException:
            try:
                await self._stop(group, "caller_cancelled_or_failed")
            finally:
                await group.done.wait()
            raise
        finally:
            self.groups.discard(group)

    async def _worker(self) -> None:
        """Take one item per group, process its result, then request another item."""
        while True:
            if not self.queue:
                self.wake.clear()
                await self.wake.wait()
                continue
            group = self.queue.popleft()
            index, sample = group.pending.popleft()
            if group.pending:
                self.queue.append(group)
            group.active += 1
            self.active += 1
            started = perf_counter()
            try:
                await self._log(
                    "rollout_started",
                    group,
                    task_id=sample.task.id,
                    index=index,
                    queue_wait_s=started - group.submitted_at,
                )
                result = await asyncio.create_task(
                    self.execute(
                        env=group.env,
                        sample=sample,
                        skill=group.skill,
                        directory=group.directory / f"{index:06d}",
                    ),
                    context=group.context.copy(),
                )
                group.results[sample.task.id] = result
                await self._log(
                    "rollout_completed",
                    group,
                    task_id=sample.task.id,
                    index=index,
                    execution_s=perf_counter() - started,
                    after_stop=group.stopped,
                )
                if not group.stopped and group.stop_when is not None and group.stop_when(group.results):
                    await self._stop(group, "early_rejection")
            except Exception as exc:
                self._log_failure(group)
                group.error = group.error or exc
                try:
                    await self._stop(group, "execution_or_logging_failed")
                    await self._log(
                        "rollout_failed",
                        group,
                        task_id=sample.task.id,
                        index=index,
                        execution_s=perf_counter() - started,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception as log_error:
                    self._log_failure(group)
                    group.error = group.error or log_error
            finally:
                group.active -= 1
                self.active -= 1
                try:
                    await self._log("rollout_released", group, task_id=sample.task.id, index=index)
                except Exception as exc:
                    self._log_failure(group)
                    group.error = group.error or exc
                    try:
                        await self._stop(group, "logging_failed")
                    except Exception:
                        self._log_failure(group)
                        pass
                group.finish_if_drained()

    async def close(self) -> None:
        """Stop queued work, drain executing calls and release idle workers."""
        self.closed = True
        error = None
        try:
            for group in list(self.groups):
                try:
                    await self._stop(group, "scheduler_closed")
                except Exception as exc:
                    self._log_failure(group)
                    error = error or exc
        finally:
            await asyncio.gather(*(group.done.wait() for group in list(self.groups)))
            for worker in self.workers:
                worker.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
        if error is not None:
            raise error
