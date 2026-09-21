"""Feedback-driven self-evolution feedback pipeline (``feedback_evo`` mode)."""

from __future__ import annotations

import asyncio
import uuid

from ....components.feedback_evo import EvolutionExecutor
from ....infra.db import EvolutionStateStore, FeedbackEventStore
from ....typing import (
    FeedbackEvoPipelineInput,
    FeedbackEvoPipelineResult,
    MemoryRequestContext,
)
from ...registry import register


@register(type="feedback", name="feedback_evo")
class FeedbackEvoPipeline:
    """Read accumulated feedback events and evolve the project's parameters."""

    _project_locks: dict[str, asyncio.Lock] = {}

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
        self._state_store: EvolutionStateStore | None = (
            getattr(executor, "_state_store", None) if executor is not None else None
        )

    def _executor_impl(self, inp: FeedbackEvoPipelineInput) -> EvolutionExecutor:
        if self._executor is None:
            from ....config import get_config

            fe_config = get_config().feedback_evo
            confidence = fe_config.require_signal_confidence
            threshold: int | None = None
            if inp.force:
                threshold = 0
            elif inp.min_signals_to_evolve is not None:
                threshold = inp.min_signals_to_evolve
            if threshold is not None:
                self._executor = EvolutionExecutor(
                    state_store=self._state_store_impl(),
                    min_signals_to_evolve=threshold,
                    require_signal_confidence=confidence,
                    max_numeric_change_ratio=fe_config.max_numeric_change_ratio,
                    max_entity_type_delta=fe_config.max_entity_type_delta,
                )
            else:
                self._executor = EvolutionExecutor(
                    state_store=self._state_store_impl(),
                    require_signal_confidence=confidence,
                    max_numeric_change_ratio=fe_config.max_numeric_change_ratio,
                    max_entity_type_delta=fe_config.max_entity_type_delta,
                )
        return self._executor

    def _event_store_impl(self) -> FeedbackEventStore:
        if self._event_store is None:
            self._event_store = FeedbackEventStore()
        return self._event_store

    def _state_store_impl(self) -> EvolutionStateStore:
        if self._state_store is None:
            self._state_store = EvolutionStateStore()
        return self._state_store

    async def run(
        self,
        inp: FeedbackEvoPipelineInput,
        context: MemoryRequestContext,
    ) -> FeedbackEvoPipelineResult:
        """Run one evolution round with optional one-time event consumption."""

        del context
        lock = self._project_locks.setdefault(inp.project_id, asyncio.Lock())
        async with lock:
            state_store = self._state_store_impl()
            if inp.idempotency_key:
                current = await state_store.get_current(inp.project_id)
                if current is not None and current.last_idempotency_key == inp.idempotency_key:
                    return FeedbackEvoPipelineResult(
                        project_id=inp.project_id,
                        evolved=bool(current.last_run_evolved),
                        version=(current.last_run_version if current.last_run_version is not None else current.version),
                        changes=current.last_run_changes,
                        signal_count=current.last_run_signal_count or 0,
                        selected_event_count=current.last_run_selected_event_count or 0,
                        consumed_event_count=current.last_run_consumed_event_count or 0,
                        idempotent_replay=True,
                        message="idempotent replay of completed evolution request",
                    )

            event_store = self._event_store_impl()
            events = await event_store.list_events(
                inp.project_id,
                user_id=inp.user_id,
                limit=self._event_limit,
                unconsumed_only=inp.event_selection == "unconsumed",
            )
            selected_event_count = len(events)
            consumption_id = inp.idempotency_key or str(uuid.uuid4())
            claimed = []
            if inp.event_selection == "unconsumed":
                claimed = await event_store.claim(inp.project_id, events, consumption_id)
                events = claimed
            signal_count = sum(len(event.signals) for event in events)
            try:
                result = await self._executor_impl(inp).run(inp.project_id, events)
            except Exception:
                if claimed:
                    await event_store.release(
                        inp.project_id,
                        claimed,
                        consumption_id=consumption_id,
                    )
                raise

            evolved = bool(result.changes)
            consumed_event_count = 0
            if claimed:
                # The executor intentionally returns version=0 for a no-op.
                # Audit events still need the live configuration version that
                # was evaluated, so retries and round reports can explain which
                # configuration consumed a signal without changing the legacy
                # executor/API version convention.
                consumed_version = result.version
                if consumed_version <= 0:
                    current_after = await state_store.get_current(inp.project_id)
                    consumed_version = current_after.version if current_after is not None else 0
                consumed_event_count = await event_store.finish(
                    inp.project_id,
                    claimed,
                    consumption_id=consumption_id,
                    version=consumed_version,
                    outcome="evolved" if evolved else "no_change",
                )
            if inp.idempotency_key:
                await state_store.record_run(
                    inp.project_id,
                    idempotency_key=inp.idempotency_key,
                    evolved=evolved,
                    version=result.version,
                    signal_count=signal_count,
                    selected_event_count=selected_event_count,
                    consumed_event_count=consumed_event_count,
                    changes=result.changes,
                )
            return FeedbackEvoPipelineResult(
                project_id=inp.project_id,
                evolved=evolved,
                version=result.version,
                changes=result.changes,
                signal_count=signal_count,
                selected_event_count=selected_event_count,
                consumed_event_count=consumed_event_count,
                message=(f"evolved to version {result.version}" if evolved else "no evolution applied"),
            )
