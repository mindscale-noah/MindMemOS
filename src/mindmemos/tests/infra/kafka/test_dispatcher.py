"""Tests for ordering, concurrency limits, and backpressure in OrderedKeyedDispatcher."""

from __future__ import annotations

import asyncio

import pytest
from mindmemos.infra.kafka.dispatcher import OrderedKeyedDispatcher


@pytest.mark.asyncio
async def test_same_key_strictly_serial():
    """Process messages for one key serially in submission order."""
    order: list[int] = []
    active = 0
    max_active = 0

    async def process(item):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        order.append(item)
        active -= 1

    async def on_complete(item):
        pass

    d = OrderedKeyedDispatcher(
        global_max_concurrency=8,
        per_key_max_concurrency=1,
        max_buffered=100,
        process=process,
        on_complete=on_complete,
    )
    for i in range(5):
        await d.submit("user-a", i)
    await d.drain()

    assert order == [0, 1, 2, 3, 4]
    assert max_active == 1


@pytest.mark.asyncio
async def test_different_keys_run_concurrently():
    """Process messages for different keys concurrently."""
    started = asyncio.Event()
    both_running = asyncio.Event()
    running = 0

    async def process(item):
        nonlocal running
        running += 1
        if running >= 2:
            both_running.set()
        started.set()
        await asyncio.sleep(0.05)
        running -= 1

    d = OrderedKeyedDispatcher(
        global_max_concurrency=8,
        per_key_max_concurrency=1,
        max_buffered=100,
        process=process,
        on_complete=lambda _i: asyncio.sleep(0),
    )
    await d.submit("user-a", 1)
    await d.submit("user-b", 2)
    await asyncio.wait_for(both_running.wait(), timeout=1.0)
    await d.drain()


@pytest.mark.asyncio
async def test_global_concurrency_cap():
    """Enforce the global concurrency limit even when more keys are active."""
    active = 0
    max_active = 0

    async def process(item):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1

    d = OrderedKeyedDispatcher(
        global_max_concurrency=2,
        per_key_max_concurrency=1,
        max_buffered=100,
        process=process,
        on_complete=lambda _i: asyncio.sleep(0),
    )
    for i in range(6):
        await d.submit(f"user-{i}", i)
    await d.drain()

    assert max_active <= 2


@pytest.mark.asyncio
async def test_shared_global_concurrency_cap_across_dispatchers():
    """Share one semaphore to limit concurrency across dispatchers."""
    active = 0
    max_active = 0
    shared = asyncio.Semaphore(2)

    async def process(item):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1

    dispatchers = [
        OrderedKeyedDispatcher(
            global_max_concurrency=8,
            per_key_max_concurrency=1,
            max_buffered=100,
            process=process,
            on_complete=lambda _i: asyncio.sleep(0),
            shared_global_semaphore=shared,
        )
        for _ in range(2)
    ]

    for index, dispatcher in enumerate(dispatchers):
        for item in range(4):
            await dispatcher.submit(f"{index}:{item}", item)
    await asyncio.gather(*(dispatcher.drain() for dispatcher in dispatchers))

    assert max_active <= 2


@pytest.mark.asyncio
async def test_non_positive_concurrency_config_is_rejected():
    """Reject missing or nonpositive concurrency and buffer limits."""
    with pytest.raises(ValueError, match="global_max_concurrency"):
        OrderedKeyedDispatcher(
            global_max_concurrency=0,
            per_key_max_concurrency=1,
            max_buffered=10,
            process=lambda _i: asyncio.sleep(0),
            on_complete=lambda _i: asyncio.sleep(0),
        )

    with pytest.raises(ValueError, match="per_key_max_concurrency"):
        OrderedKeyedDispatcher(
            global_max_concurrency=1,
            per_key_max_concurrency=0,
            max_buffered=10,
            process=lambda _i: asyncio.sleep(0),
            on_complete=lambda _i: asyncio.sleep(0),
        )

    with pytest.raises(ValueError, match="max_buffered"):
        OrderedKeyedDispatcher(
            global_max_concurrency=1,
            per_key_max_concurrency=1,
            max_buffered=0,
            process=lambda _i: asyncio.sleep(0),
            on_complete=lambda _i: asyncio.sleep(0),
        )


@pytest.mark.asyncio
async def test_backpressure_blocks_submit_without_dropping():
    """Block submission when the buffer is full and eventually process every message."""
    completed: list[int] = []
    release = asyncio.Event()

    async def process(item):
        await release.wait()

    async def on_complete(item):
        completed.append(item)

    d = OrderedKeyedDispatcher(
        global_max_concurrency=1,
        per_key_max_concurrency=1,
        max_buffered=2,
        process=process,
        on_complete=on_complete,
    )
    # Fill the buffer before testing whether the third submission blocks.
    await d.submit("k", 0)
    await d.submit("k", 1)
    blocked = asyncio.create_task(d.submit("k", 2))
    await asyncio.sleep(0.02)
    assert not blocked.done()

    release.set()
    await blocked
    await d.drain()

    assert sorted(completed) == [0, 1, 2]
