import asyncio

import pytest
from mindmemos.infra.db.read_cache import ProjectReadCache


@pytest.mark.asyncio
async def test_cache_coalesces_reads_and_isolates_projects() -> None:
    cache = ProjectReadCache(ttl_seconds=30, max_entries=2)
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return {"value": calls}

    results = await asyncio.gather(*(cache.get("project-a", "key", load) for _ in range(10)))
    assert calls == 1
    results[0]["value"] = 99
    assert await cache.get("project-a", "key", load) == {"value": 1}
    assert await cache.get("project-b", "key", load) == {"value": 2}
    await cache.get("project-c", "key", load)
    await cache.get("project-a", "key", load)
    assert calls == 4


@pytest.mark.asyncio
async def test_cache_expires_and_supports_explicit_refresh(monkeypatch) -> None:
    now = 0.0
    monkeypatch.setattr("mindmemos.infra.db.read_cache.monotonic", lambda: now)
    cache = ProjectReadCache(ttl_seconds=30)
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        return calls

    assert await cache.get("project", "key", load) == 1
    assert await cache.get("project", "key", load) == 1
    assert await cache.get("project", "key", load, force=True) == 2
    now = 31.0
    assert await cache.get("project", "key", load) == 3


@pytest.mark.asyncio
async def test_slow_snapshot_generation_does_not_extend_cache_freshness(monkeypatch) -> None:
    now = 0.0
    monkeypatch.setattr("mindmemos.infra.db.read_cache.monotonic", lambda: now)
    cache = ProjectReadCache(ttl_seconds=30)
    calls = 0

    async def load():
        nonlocal calls, now
        calls += 1
        now += 20
        return calls

    assert await cache.get("project", "key", load) == 1
    assert await cache.get("project", "key", load) == 1
    now = 31.0
    assert await cache.get("project", "key", load) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("force", [False, True])
async def test_write_invalidation_prevents_an_older_inflight_read_repopulating_cache(force) -> None:
    cache = ProjectReadCache()
    started = asyncio.Event()
    release = asyncio.Event()

    async def old_read():
        started.set()
        await release.wait()
        return "old"

    async def new_read():
        return "new"

    task = asyncio.create_task(cache.get("project", "key", old_read, force=force))
    await started.wait()
    cache.invalidate("project")
    assert await cache.get("project", "key", new_read) == "new"
    release.set()
    assert await task == "old"
    assert await cache.get("project", "key", old_read) == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize("older_force", [False, True])
async def test_refresh_starts_a_new_read_and_superseded_read_cannot_overwrite_it(older_force) -> None:
    cache = ProjectReadCache()
    started = asyncio.Event()
    release = asyncio.Event()

    async def old_read():
        started.set()
        await release.wait()
        return "old"

    async def new_read():
        return "new"

    old_task = asyncio.create_task(cache.get("project", "key", old_read, force=older_force))
    await started.wait()
    try:
        assert await asyncio.wait_for(cache.get("project", "key", new_read, force=True), timeout=1) == "new"
    finally:
        release.set()
        assert await old_task == "old"
    assert await cache.get("project", "key", old_read) == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize("max_entries", [1, 2])
async def test_failed_refresh_preserves_valid_snapshot_without_extending_its_ttl(monkeypatch, max_entries) -> None:
    now = 0.0
    monkeypatch.setattr("mindmemos.infra.db.read_cache.monotonic", lambda: now)
    cache = ProjectReadCache(ttl_seconds=30, max_entries=max_entries)

    async def old_read():
        return "old"

    async def fail():
        raise TimeoutError("database unavailable")

    async def new_read():
        return "new"

    assert await cache.get("project", "key", old_read) == "old"
    now = 29.0
    with pytest.raises(TimeoutError, match="database unavailable"):
        await cache.get("project", "key", fail, force=True)
    assert await cache.get("project", "key", new_read) == "old"
    now = 31.0
    assert await cache.get("project", "key", new_read) == "new"


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_read_or_cache_errors() -> None:
    cache = ProjectReadCache()
    started = asyncio.Event()
    release = asyncio.Event()

    async def load():
        started.set()
        await release.wait()
        return None

    task = asyncio.create_task(cache.get("project", "key", load))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await cache.get("project", "key", load) is None

    async def fail():
        raise ValueError("transient")

    with pytest.raises(ValueError, match="transient"):
        await cache.get("project", "error", fail)
    assert await cache.get("project", "error", load) is None
