"""Async bridges that retain ownership of in-flight synchronous work."""

import asyncio


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
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
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
