"""Bounded task admission and execution shared by local and Kafka managers."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from ...logging import get_logger

logger = get_logger(__name__)
ItemT = TypeVar("ItemT")


class TaskDispatcher(Generic[ItemT]):
    """Deduplicate unfinished task IDs and hold execution slots until completion."""

    def __init__(
        self,
        handler: Callable[[ItemT], Awaitable[None]],
        *,
        max_concurrency: int = 2,
        max_buffered: int = 1000,
    ):
        """Configure one event-loop-owned dispatcher without starting work.

        Args:
            handler: Complete execution callback, including claim and final writes.
            max_concurrency: Maximum callbacks executing simultaneously.
            max_buffered: Maximum unfinished items, including running callbacks.

        Raises:
            ValueError: Either limit is less than one.
        """
        if max_concurrency < 1 or max_buffered < 1:
            raise ValueError("Dispatcher concurrency and buffer limits must be positive")
        self._handler = handler
        self._gate = asyncio.Semaphore(max_concurrency)
        self._max_buffered = max_buffered
        self._tasks: dict[str, asyncio.Task] = {}
        self._capacity_changed = asyncio.Event()
        self._closed = False

    def try_submit(self, task_id: str, item: ItemT) -> bool:
        """Admit work immediately, coalescing duplicate unfinished identities.

        Admission contains no await, so checks and reservation are atomic within
        the owning event loop. Durable callers may retry a full queue later.

        Args:
            task_id: Identity shared by all notifications for the same work.
            item: Input retained until the execution callback completes.

        Returns:
            True if admitted or already pending; False if capacity is exhausted.

        Raises:
            RuntimeError: The dispatcher has closed.
        """
        if self._closed:
            raise RuntimeError("Task dispatcher is closed")
        if task_id in self._tasks:
            return True
        if len(self._tasks) >= self._max_buffered:
            self._capacity_changed.clear()
            return False
        task = asyncio.create_task(self._execute(item), name=f"skill-dispatch-{task_id}")
        self._tasks[task_id] = task

        def finished(done: asyncio.Task) -> None:
            """Release buffer capacity even when cancelled before execution begins."""
            del self._tasks[task_id]
            self._capacity_changed.set()
            if not done.cancelled() and done.exception() is not None:
                logger.error(
                    "task.dispatch.failed",
                    extra={"event_body": {"task_id": task_id}},
                    exc_info=done.exception(),
                )

        task.add_done_callback(finished)
        return True

    async def submit(self, task_id: str, item: ItemT) -> None:
        """Wait for buffer capacity without starting unbounded execution tasks.

        Args:
            task_id: Identity shared by all notifications for the same work.
            item: Input passed to the execution callback after admission.

        Raises:
            RuntimeError: The dispatcher closes before admission.
        """
        while not self.try_submit(task_id, item):
            await self._capacity_changed.wait()

    async def _execute(self, item: ItemT) -> None:
        """Hold the execution slot across the entire callback, including cleanup."""
        async with self._gate:
            await self._handler(item)

    async def close(self) -> None:
        """Reject new work, wake submitters, and cancel all admitted callbacks."""
        self._closed = True
        self._capacity_changed.set()
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
