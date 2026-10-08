"""Async bridges that retain ownership of in-flight synchronous work."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


async def run_sync(function, /, *args, **kwargs):
    """Wait for a synchronous operation to settle even when its caller is cancelled.

    Args:
        function: Synchronous operation executed in the default worker pool.
        *args: Positional arguments for the operation.
        **kwargs: Keyword arguments for the operation.

    Returns:
        The operation's result.

    Raises:
        asyncio.CancelledError: After the active operation has finished.
        Exception: The operation failed during normal execution.
    """

    def call():
        try:
            return function(*args, **kwargs)
        except StopIteration as exc:
            raise RuntimeError("Synchronous operation raised StopIteration") from exc

    task = asyncio.create_task(asyncio.to_thread(call))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


async def map_bounded(function: Callable[[_T], Awaitable[_R]], items: Iterable[_T], *, concurrency: int) -> list[_R]:
    """Run asynchronous calls with bounded admission in input order.

    Args:
        function: Asynchronous operation applied to each item.
        items: Input values in result order.
        concurrency: Maximum number of active operations.

    Returns:
        Results in the same order as the inputs.

    Raises:
        ValueError: The concurrency limit is not positive.
        Exception: An operation failed after active siblings have drained.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    gate = asyncio.Semaphore(concurrency)

    async def run(item: _T) -> _R:
        async with gate:
            return await function(item)

    tasks = [asyncio.create_task(run(item)) for item in items]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def map_sync_bounded(function: Callable[[_T], _R], items: Iterable[_T], *, concurrency: int) -> list[_R]:
    """Run synchronous calls in worker threads with bounded async admission.

    Args:
        function: Synchronous operation applied to each item.
        items: Input values in result order.
        concurrency: Maximum number of active operations.

    Returns:
        Results in the same order as the inputs.
    """

    async def run(item: _T) -> _R:
        return await run_sync(function, item)

    return await map_bounded(run, items, concurrency=concurrency)
