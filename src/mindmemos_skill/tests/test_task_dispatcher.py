"""Shared admission limits, duplicate notifications, and cancellation ownership."""

import asyncio

import pytest
from mindmemos_skill.infra.task_manager.dispatcher import TaskDispatcher


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2, 4])
async def test_dispatcher_bounds_all_submitters_and_coalesces_pending_ids(limit):
    started = asyncio.Queue()
    finished = asyncio.Queue()
    release = asyncio.Event()
    active = 0
    peak = 0

    async def execute(item):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.put_nowait(item)
        try:
            await release.wait()
        finally:
            active -= 1
            finished.put_nowait(item)

    dispatcher = TaskDispatcher(execute, max_concurrency=limit, max_buffered=8)
    try:
        await asyncio.gather(*(dispatcher.submit(str(index), index) for index in range(8)))
        for _ in range(limit):
            await asyncio.wait_for(started.get(), 1)
        assert started.empty()
        assert dispatcher.try_submit("0", "duplicate")
        assert not dispatcher.try_submit("overflow", 9)
        assert peak == limit
        release.set()
        results = [await asyncio.wait_for(finished.get(), 1) for _ in range(8)]
        assert sorted(results) == list(range(8))
        assert peak == limit
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_buffer_waiter_proceeds_after_failed_handler():
    started = asyncio.Event()
    release = asyncio.Event()
    second_ran = asyncio.Event()

    async def execute(item):
        if item == "first":
            started.set()
            await release.wait()
            raise ValueError("Expected handler failure")
        second_ran.set()

    dispatcher = TaskDispatcher(execute, max_concurrency=1, max_buffered=1)
    try:
        await dispatcher.submit("first", "first")
        await started.wait()
        pending = asyncio.create_task(dispatcher.submit("second", "second"))
        await asyncio.sleep(0)
        assert not pending.done()
        release.set()
        await asyncio.wait_for(pending, 1)
        await asyncio.wait_for(second_ran.wait(), 1)
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_close_cancels_running_and_queued_work_and_wakes_capacity_waiters():
    started = asyncio.Event()
    cancelled = asyncio.Event()
    calls = []

    async def execute(item):
        calls.append(item)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    dispatcher = TaskDispatcher(execute, max_concurrency=1, max_buffered=2)
    await dispatcher.submit("running", "running")
    await started.wait()
    await dispatcher.submit("queued", "queued")
    pending = asyncio.create_task(dispatcher.submit("waiting", "waiting"))
    await asyncio.sleep(0)
    assert not pending.done()
    await asyncio.wait_for(dispatcher.close(), 1)
    with pytest.raises(RuntimeError, match="closed"):
        await pending
    assert cancelled.is_set()
    assert calls == ["running"]
    assert not dispatcher._tasks
    await dispatcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        dispatcher.try_submit("new", "new")


@pytest.mark.asyncio
async def test_cancelled_capacity_waiter_does_not_reserve_an_identity():
    release = asyncio.Event()

    async def execute(item):
        await release.wait()

    dispatcher = TaskDispatcher(execute, max_concurrency=1, max_buffered=1)
    try:
        await dispatcher.submit("first", 1)
        pending = asyncio.create_task(dispatcher.submit("second", 2))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert "second" not in dispatcher._tasks
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_self_cancelled_execution_releases_its_slot():
    completed = asyncio.Event()

    async def execute(item):
        if item == "lost-lease":
            raise asyncio.CancelledError
        completed.set()

    dispatcher = TaskDispatcher(execute, max_concurrency=1, max_buffered=2)
    try:
        await dispatcher.submit("lost-lease", "lost-lease")
        await dispatcher.submit("next", "next")
        await asyncio.wait_for(completed.wait(), 1)
    finally:
        await dispatcher.close()
