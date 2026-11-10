"""Bounded CPU execution for vanilla search de-duplication."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor
from contextvars import copy_context
from functools import partial
from threading import Lock
from typing import ParamSpec, TypeVar
from weakref import WeakKeyDictionary

from ....logging import get_logger

_P = ParamSpec("_P")
_R = TypeVar("_R")
logger = get_logger(__name__)


class BoundedDedupExecutor:
    """Bound synchronous de-duplication through an async admission gate."""

    def __init__(
        self,
        *,
        executor: Executor | None = None,
        max_workers: int = 2,
        max_inflight: int = 4,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_inflight < 1:
            raise ValueError("max_inflight must be at least 1")

        self._executor = executor
        self._max_workers = max_workers
        self._max_inflight = max_inflight
        self._gates: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()
        self._worker_gates: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()
        self._gates_lock = Lock()

    def _gate_for(self, loop: asyncio.AbstractEventLoop, *, workers: bool = False) -> asyncio.Semaphore:
        with self._gates_lock:
            gates = self._worker_gates if workers else self._gates
            gate = gates.get(loop)
            if gate is None:
                gate = asyncio.Semaphore(self._max_workers if workers else self._max_inflight)
                gates[loop] = gate
            return gate

    async def run(self, func: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Run ``func`` after bounded admission while preserving request context."""
        loop = asyncio.get_running_loop()
        gate = self._gate_for(loop)
        await gate.acquire()
        worker_gate = self._gate_for(loop, workers=True) if self._executor is None else None
        if worker_gate is not None:
            try:
                await worker_gate.acquire()
            except BaseException:
                gate.release()
                raise

        context = copy_context()
        call = partial(func, *args, **kwargs)
        try:
            if self._executor is None:
                future = asyncio.create_task(asyncio.to_thread(context.run, call))
            else:
                future = loop.run_in_executor(self._executor, context.run, call)
        except BaseException:
            if worker_gate is not None:
                worker_gate.release()
            gate.release()
            raise

        def release(_future: asyncio.Future) -> None:
            if worker_gate is not None:
                worker_gate.release()
            gate.release()

        future.add_done_callback(release)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(future)
            except Exception:
                logger.warning("vanilla_dedup_cancelled_worker_failed", exc_info=True)
            raise


vanilla_dedup_executor = BoundedDedupExecutor(max_workers=2, max_inflight=4)

__all__ = ["BoundedDedupExecutor", "vanilla_dedup_executor"]
