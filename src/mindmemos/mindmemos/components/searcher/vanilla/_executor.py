"""Bounded CPU execution for vanilla search de-duplication."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
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
    """Run CPU-bound de-duplication outside asyncio's shared executor."""

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

        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="mindmemos-vanilla-dedup",
        )
        self._max_inflight = max_inflight
        self._gates: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()
        self._gates_lock = Lock()

    def _gate_for(self, loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
        with self._gates_lock:
            gate = self._gates.get(loop)
            if gate is None:
                gate = asyncio.Semaphore(self._max_inflight)
                self._gates[loop] = gate
            return gate

    async def run(self, func: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Run ``func`` after bounded admission while preserving request context."""
        loop = asyncio.get_running_loop()
        gate = self._gate_for(loop)
        await gate.acquire()

        context = copy_context()
        call = partial(func, *args, **kwargs)
        try:
            future = loop.run_in_executor(self._executor, context.run, call)
        except BaseException:
            gate.release()
            raise

        future.add_done_callback(lambda _: gate.release())
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
