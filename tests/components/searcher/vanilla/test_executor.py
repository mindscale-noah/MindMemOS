"""Tests for bounded, isolated vanilla de-duplication execution."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

import pytest
from mindmemos.components.searcher.vanilla._executor import BoundedDedupExecutor


class RecordingExecutor(ThreadPoolExecutor):
    def __init__(self, *, max_workers: int) -> None:
        super().__init__(max_workers=max_workers)
        self._submitted = 0
        self._submitted_lock = threading.Lock()

    @property
    def submitted(self) -> int:
        with self._submitted_lock:
            return self._submitted

    def submit(self, fn, /, *args, **kwargs):
        with self._submitted_lock:
            self._submitted += 1
        return super().submit(fn, *args, **kwargs)


async def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_executor_submits_at_most_the_inflight_limit() -> None:
    release = threading.Event()
    pool = RecordingExecutor(max_workers=2)
    bounded = BoundedDedupExecutor(executor=pool, max_inflight=4)

    try:
        tasks = [asyncio.create_task(bounded.run(release.wait)) for _ in range(6)]
        await _wait_until(lambda: pool.submitted == 4)
        await asyncio.sleep(0.05)

        assert pool.submitted == 4

        release.set()
        await asyncio.gather(*tasks)
        assert pool.submitted == 6
    finally:
        release.set()
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_dedup_workers_do_not_block_the_default_executor() -> None:
    release = threading.Event()
    started = 0
    started_lock = threading.Lock()
    pool = RecordingExecutor(max_workers=2)
    bounded = BoundedDedupExecutor(executor=pool, max_inflight=4)

    def blocking_job() -> None:
        nonlocal started
        with started_lock:
            started += 1
        release.wait()

    try:
        tasks = [asyncio.create_task(bounded.run(blocking_job)) for _ in range(2)]
        await _wait_until(lambda: started == 2)

        assert await asyncio.wait_for(asyncio.to_thread(lambda: "default-ok"), timeout=1) == "default-ok"

        release.set()
        await asyncio.gather(*tasks)
    finally:
        release.set()
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_executor_propagates_context_variables() -> None:
    request_id: ContextVar[str] = ContextVar("request_id", default="missing")
    pool = RecordingExecutor(max_workers=1)
    bounded = BoundedDedupExecutor(executor=pool, max_inflight=1)
    token = request_id.set("req-123")

    try:
        assert await bounded.run(request_id.get) == "req-123"
    finally:
        request_id.reset(token)
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_failed_job_releases_permit() -> None:
    pool = RecordingExecutor(max_workers=1)
    bounded = BoundedDedupExecutor(executor=pool, max_inflight=1)

    def fail() -> None:
        raise RuntimeError("dedup failed")

    try:
        with pytest.raises(RuntimeError, match="dedup failed"):
            await bounded.run(fail)
        assert await bounded.run(lambda: "recovered") == "recovered"
    finally:
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_cancelled_job_holds_permit_until_worker_finishes() -> None:
    release = threading.Event()
    worker_started = threading.Event()
    pool = RecordingExecutor(max_workers=1)
    bounded = BoundedDedupExecutor(executor=pool, max_inflight=1)

    def blocking_job() -> None:
        worker_started.set()
        release.wait()

    try:
        first = asyncio.create_task(bounded.run(blocking_job))
        await asyncio.to_thread(worker_started.wait, 1)
        first.cancel()
        second = asyncio.create_task(bounded.run(lambda: "second"))
        await asyncio.sleep(0.05)

        assert pool.submitted == 1

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == "second"
        assert pool.submitted == 2
    finally:
        release.set()
        pool.shutdown(wait=True)
