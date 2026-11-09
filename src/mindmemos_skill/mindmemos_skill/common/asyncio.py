"""Bounded structured concurrency for native asynchronous operations."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


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
