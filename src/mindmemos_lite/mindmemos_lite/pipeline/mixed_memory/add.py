"""Concurrent add fan-out across configured memory modes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from ...config import MemoryConfig
from ...logging import get_logger, traced
from ...typing import AddPipelineInput, AddPipelineSyncResult, MemoryRequestContext
from ..base import AddPipeline, PipelineBase
from ..registry import create_pipeline, register

logger = get_logger(__name__)


@register(type="add", name="mixed_add")
class MixedAddPipeline(PipelineBase):
    """Run the same add input through multiple independent mode pipelines.

    Child calls start concurrently. Results are returned in configuration
    order, so the API response remains deterministic even when the algorithms
    finish in a different order.
    """

    def __init__(self, *, pipelines: Mapping[str, AddPipeline]) -> None:
        if not pipelines:
            raise ValueError("mixed add requires at least one child pipeline")
        self._pipelines = dict(pipelines)

    @classmethod
    def from_config(cls, config: MemoryConfig, **kwargs):
        routing = config.pipelines
        pipelines = {
            mode: create_pipeline(
                type="add",
                name=routing.modes[mode].add_pipeline,
                config=config,
                **kwargs,
            )
            for mode in routing.mixed_add.modes
        }
        task_requiring = [name for name, pipeline in pipelines.items() if getattr(pipeline, "requires_task", False)]
        if task_requiring:
            raise ValueError(
                f"task-requiring add pipelines cannot run via mixed fan-out: {', '.join(task_requiring)}"
            )
        return cls(pipelines=pipelines)

    @property
    def modes(self) -> tuple[str, ...]:
        """Return child modes in deterministic dispatch/result order."""

        return tuple(self._pipelines)

    @traced("add.mixed_add.sync")
    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
    ) -> AddPipelineSyncResult:
        modes = self.modes
        logger.debug("mixed_add_started", modes=modes, request_id=context.request_id)

        # Each child receives an isolated input object and a context carrying
        # its stable public mode. The persistence boundary uses that mode to
        # stamp every memory written by the child.
        calls = [
            pipeline.add_sync(
                inp.model_copy(deep=True),
                context.model_copy(update={"memory_algorithm": mode}),
            )
            for mode, pipeline in self._pipelines.items()
        ]

        # Collect every result before reporting failures. This avoids silently
        # abandoning a slower child after another child has already persisted.
        outcomes = await asyncio.gather(*calls, return_exceptions=True)
        failures = [
            f"{mode}: {outcome}"
            for mode, outcome in zip(modes, outcomes, strict=True)
            if isinstance(outcome, BaseException)
        ]
        failures.extend(
            f"{mode}: returned status {outcome.status!r}"
            for mode, outcome in zip(modes, outcomes, strict=True)
            if isinstance(outcome, AddPipelineSyncResult) and outcome.status != "ok"
        )
        if failures:
            raise RuntimeError("mixed add failed after all modes settled: " + "; ".join(failures))

        memories = []
        for outcome in outcomes:
            assert isinstance(outcome, AddPipelineSyncResult)
            memories.extend(outcome.memories)
        logger.debug(
            "mixed_add_completed",
            modes=modes,
            memory_event_count=len(memories),
            request_id=context.request_id,
        )
        return AddPipelineSyncResult(status="ok", memories=memories)


__all__ = ["MixedAddPipeline"]
