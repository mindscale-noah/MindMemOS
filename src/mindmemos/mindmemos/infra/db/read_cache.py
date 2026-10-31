"""Bounded, event-loop-local read caches with project-scoped invalidation."""

from __future__ import annotations

from asyncio import Semaphore, Task, create_task, current_task, shield
from collections.abc import Awaitable, Callable, Hashable
from copy import deepcopy
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")
CacheKey = tuple[str, Hashable]


class ProjectReadCache(Generic[T]):
    """Coalesce reads without allowing writes to resurrect an old snapshot.

    Caches are owned by event-loop-scoped repositories. Other processes' writes
    become visible within the TTL; explicit refresh always bypasses cached data.
    """

    def __init__(self, *, ttl_seconds: float = 30, max_entries: int = 2048, max_concurrency: int = 5) -> None:
        self._ttl = max(0, ttl_seconds)
        self._max_entries = max(1, max_entries)
        self._cache: dict[CacheKey, tuple[float, T]] = {}
        self._inflight: dict[CacheKey, Task[T]] = {}
        self._bulkhead = Semaphore(max(1, max_concurrency))

    async def get(
        self, project_id: str, key: Hashable, loader: Callable[[], Awaitable[T]], *, force: bool = False
    ) -> T:
        cache_key = (project_id, key)
        cached = self._cache.get(cache_key)
        if not force and cached is not None and cached[0] > monotonic():
            return deepcopy(cached[1])
        self._cache.pop(cache_key, None)
        task = self._inflight.get(cache_key)
        if task is None:
            task = create_task(self._load(cache_key, loader))
            self._inflight[cache_key] = task
            task.add_done_callback(lambda done: self._finished(cache_key, done))
        return deepcopy(await shield(task))

    def invalidate(self, project_id: str) -> None:
        """Forget cached and in-flight entries without cancelling their readers."""

        for key in list(self._cache):
            if key[0] == project_id:
                self._cache.pop(key, None)
        for key in list(self._inflight):
            if key[0] == project_id:
                self._inflight.pop(key, None)

    async def _load(self, key: CacheKey, loader: Callable[[], Awaitable[T]]) -> T:
        async with self._bulkhead:
            started = monotonic()
            result = await loader()
        if self._inflight.get(key) is current_task():
            now = monotonic()
            if len(self._cache) >= self._max_entries:
                self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
            if len(self._cache) >= self._max_entries:
                self._cache.pop(min(self._cache, key=lambda k: self._cache[k][0]))
            self._cache[key] = (started + self._ttl, deepcopy(result))
        return result

    def _finished(self, key: CacheKey, task: Task[T]) -> None:
        if self._inflight.get(key) is task:
            self._inflight.pop(key, None)
        if not task.cancelled():
            task.exception()  # An aborted HTTP request may no longer await this task.
