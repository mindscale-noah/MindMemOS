"""Feedback-driven self-evolution pipeline (``feedback_evo`` mode)."""

from __future__ import annotations

from ...components.feedback_evo import EvolutionExecutor
from ...infra.db import FeedbackEventStore
from ...typing import (
    FeedbackEvoPipelineInput,
    FeedbackEvoPipelineResult,
    MemoryRequestContext,
)
from ..registry import register


@register(type="feedback", name="feedback_evo")
class FeedbackEvoPipeline:
    """Read accumulated feedback events and evolve the project's parameters."""

    def __init__(
        self,
        *,
        executor: EvolutionExecutor | None = None,
        event_store: FeedbackEventStore | None = None,
        event_limit: int = 200,
    ) -> None:
        self._executor = executor
        self._event_store = event_store
        self._event_limit = event_limit

    def _executor_impl(self, inp: FeedbackEvoPipelineInput) -> EvolutionExecutor:
        if self._executor is None:
            from ...config import get_config

            fe_config = get_config().feedback_evo
            confidence = fe_config.require_signal_confidence
            threshold: int | None = None
            if inp.force:
                threshold = 0
            elif inp.min_signals_to_evolve is not None:
                threshold = inp.min_signals_to_evolve
            if threshold is not None:
                self._executor = EvolutionExecutor(
                    min_signals_to_evolve=threshold,
                    require_signal_confidence=confidence,
                    max_numeric_change_ratio=fe_config.max_numeric_change_ratio,
                    max_entity_type_delta=fe_config.max_entity_type_delta,
                )
            else:
                self._executor = EvolutionExecutor(
                    require_signal_confidence=confidence,
                    max_numeric_change_ratio=fe_config.max_numeric_change_ratio,
                    max_entity_type_delta=fe_config.max_entity_type_delta,
                )
        return self._executor

    def _event_store_impl(self) -> FeedbackEventStore:
        if self._event_store is None:
            self._event_store = FeedbackEventStore()
        return self._event_store

    async def run(
        self,
        inp: FeedbackEvoPipelineInput,
        context: MemoryRequestContext,
    ) -> FeedbackEvoPipelineResult:
        """Run one evolution round over the project's accumulated events."""

        del context
        events = await self._event_store_impl().list_events(
            inp.project_id,
            user_id=inp.user_id,
            limit=self._event_limit,
        )
        signal_count = sum(len(event.signals) for event in events)
        result = await self._executor_impl(inp).run(inp.project_id, events)
        return FeedbackEvoPipelineResult(
            project_id=inp.project_id,
            evolved=bool(result.changes),
            version=result.version,
            changes=result.changes,
            signal_count=signal_count,
            message=(
                f"evolved to version {result.version}"
                if result.changes
                else "no evolution applied"
            ),
        )
