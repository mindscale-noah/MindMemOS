"""Single-layer bounded worker pool that calls Env directly."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from .....agents.base import Agent
from .....envs.base import BaseEnv, EnvRolloutContext
from .....registry import get_env
from .....typing import Rollout, RolloutType
from ..config import RolloutConfig
from ..contracts import RolloutAttempt, RolloutOutcome, RolloutPhase, RolloutSpec


class AgentResolver(Protocol):
    def resolve(self, ref: str) -> Agent[Any]: ...


class EnvFactory(Protocol):
    def create(self, ref: str, config: Mapping[str, Any]) -> BaseEnv[Any]: ...


class MappingAgentResolver:
    """Minimal resolver useful to runners and tests without a controller."""

    def __init__(self, agents: Mapping[str, Agent[Any]]) -> None:
        self._agents = dict(agents)

    def resolve(self, ref: str) -> Agent[Any]:
        try:
            return self._agents[ref]
        except KeyError as exc:
            raise ValueError(f"unknown agent ref {ref!r}") from exc


class RegistryEnvFactory:
    def create(self, ref: str, config: Mapping[str, Any]) -> BaseEnv[Any]:
        return get_env(name=ref, config=config)


OutcomeCallback = Callable[[RolloutOutcome], Awaitable[None]]


class RolloutScheduler:
    """Own the one and only rollout concurrency budget."""

    def __init__(
        self,
        *,
        agent_resolver: AgentResolver,
        env_factory: EnvFactory,
        config: RolloutConfig,
        on_outcome: OutcomeCallback | None = None,
    ) -> None:
        self._agent_resolver = agent_resolver
        self._env_factory = env_factory
        self._config = config
        self._on_outcome = on_outcome
        self._rollout_slots = asyncio.Semaphore(config.max_concurrent_rollouts)

    async def run(self, specs: list[RolloutSpec]) -> list[RolloutOutcome]:
        if not specs:
            return []
        worker_count = min(self._config.max_concurrent_rollouts, len(specs))
        queue: asyncio.Queue[RolloutSpec | None] = asyncio.Queue(maxsize=self._config.queue_capacity)
        outcomes: list[RolloutOutcome] = []
        failed = asyncio.Event()

        async def producer() -> None:
            for spec in specs:
                if failed.is_set() and self._config.fail_fast:
                    break
                await queue.put(spec)
            for _ in range(worker_count):
                await queue.put(None)

        async def worker() -> None:
            while True:
                spec = await queue.get()
                try:
                    if spec is None:
                        return
                    if failed.is_set() and self._config.fail_fast:
                        continue
                    # Multiple phases may share this scheduler concurrently. Keep
                    # one run-wide rollout budget instead of multiplying the
                    # configured limit by the number of active scheduler runs.
                    async with self._rollout_slots:
                        outcome = await self._run_spec(spec)
                    outcomes.append(outcome)
                    if not outcome.succeeded:
                        failed.set()
                    if self._on_outcome is not None:
                        await self._on_outcome(outcome)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        producer_task = asyncio.create_task(producer())
        await asyncio.gather(producer_task, *workers)
        outcomes.sort(key=lambda item: item.spec.sequence_no)

        if self._config.fail_fast:
            failure = next((item for item in outcomes if not item.succeeded), None)
            if failure is not None:
                last = failure.attempts[-1]
                raise RuntimeError(
                    f"rollout {failure.spec.rollout_id} failed after {len(failure.attempts)} attempt(s): "
                    f"{last.error_type}: {last.error}"
                )
        return outcomes

    async def _run_spec(self, spec: RolloutSpec) -> RolloutOutcome:
        attempts: list[RolloutAttempt] = []
        winning = None
        for attempt_no in range(self._config.retry.max_attempts):
            started_at = datetime.now(UTC)
            try:
                trajectory = await self._execute_attempt(spec, attempt_no)
                finished_at = datetime.now(UTC)
                attempts.append(
                    RolloutAttempt(
                        attempt_no=attempt_no,
                        trajectory=trajectory,
                        started_at=started_at,
                        finished_at=finished_at,
                    )
                )
                # A returned trajectory is valid task evidence even when its
                # Agent status is FAILED (for example max-turn exhaustion).
                # Retry only physical exceptions that produced no trajectory.
                winning = trajectory
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempts.append(
                    RolloutAttempt(
                        attempt_no=attempt_no,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                    )
                )
            if attempt_no + 1 < self._config.retry.max_attempts and self._config.retry.backoff_seconds:
                await asyncio.sleep(self._config.retry.backoff_seconds)
        return RolloutOutcome(
            spec=spec,
            attempts=attempts,
            trajectory=winning,
            succeeded=winning is not None,
        )

    async def _execute_attempt(self, spec: RolloutSpec, attempt_no: int):
        rollout_type = {
            RolloutPhase.TRAIN: RolloutType.TRAIN,
            RolloutPhase.VALIDATION: RolloutType.EVALUATE,
            RolloutPhase.TEST: RolloutType.TEST,
            RolloutPhase.ABLATION_BEFORE: RolloutType.EVALUATE,
            RolloutPhase.ABLATION_AFTER: RolloutType.EVALUATE,
        }[spec.phase]
        context = EnvRolloutContext(
            rollout=Rollout(
                rollout_id=spec.rollout_id,
                attempt_no=attempt_no,
                rollout_type=rollout_type,
            ),
            workspace_root=self._config.workspace_root,
            workspace_scope=spec.phase.value,
            agent_options={
                **spec.agent_options,
                **({"temperature": spec.temperature} if spec.temperature is not None else {}),
            },
            metadata={
                **spec.metadata,
                "sample_index": spec.sample_index,
                "sequence_no": spec.sequence_no,
                "seed": spec.seed,
                "candidate_id": spec.candidate_id,
                "pair_id": spec.pair_id,
            },
        )
        agent = self._agent_resolver.resolve(spec.agent_ref)
        env = self._env_factory.create(spec.env_ref, spec.env_options)
        async with env:
            call = env.rollout(agent, spec.task, spec.skills, context=context)
            if self._config.timeout_seconds is None:
                return await call
            return await asyncio.wait_for(call, timeout=self._config.timeout_seconds)


__all__ = [
    "AgentResolver",
    "EnvFactory",
    "MappingAgentResolver",
    "RegistryEnvFactory",
    "RolloutScheduler",
]
