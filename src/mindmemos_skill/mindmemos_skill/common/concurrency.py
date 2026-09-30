"""Small concurrency helpers that retain per-call context variables."""

from collections.abc import Callable, Iterable
from concurrent.futures import Executor
from contextvars import copy_context
from typing import TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def map_with_context(
    executor: Executor, function: Callable[[InputT], OutputT], items: Iterable[InputT]
) -> list[OutputT]:
    """Run independent work with a separate copy of the caller's context.

    Args:
        executor: Caller-owned worker pool.
        function: Operation applied to each input.
        items: Ordered inputs to execute.

    Returns:
        Results in input order after every submitted operation completes.
    """
    futures = [executor.submit(copy_context().run, function, item) for item in items]
    return [future.result() for future in futures]
