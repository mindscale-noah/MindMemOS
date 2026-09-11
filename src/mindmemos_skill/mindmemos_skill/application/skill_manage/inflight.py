"""Application-owned in-flight registry for same-process idempotency.

Keyed by ``(project_id, graph_id, idempotency_key)``. The first call becomes
the leader; identical-hash followers await the same final result via
``asyncio.shield`` so cancelling a follower never cancels the leader. A same-key
different-hash call fails with an idempotency conflict before joining.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


class InflightConflictError(Exception):
    """Same idempotency key carries a different request hash."""

    def __init__(self, key: tuple[str, str, str]) -> None:
        super().__init__(f"idempotency conflict for key {key!r}")
        self.key = key


class InflightLeaderCancelled(Exception):
    """Published to followers when the leader itself was cancelled.

    A native ``CancelledError`` must only propagate out of the task that was
    actually cancelled, so the leader converts its own cancellation into this
    ordinary exception before fanning it out; a follower that receives it
    re-runs the write against the durable operation row instead.
    """


@dataclass(slots=True)
class _Entry:
    request_hash: str
    result_future: asyncio.Future


class InflightRegistry:
    """Tracks one leader per scope+idempotency key and fans out its result."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], _Entry] = {}

    def begin(
        self,
        key: tuple[str, str, str],
        request_hash: str,
    ) -> tuple[str, asyncio.Future | None]:
        """Start or join an in-flight call.

        Returns ``("leader", None)`` when this caller owns the key, or
        ``("follower", result_future)`` when joining an existing leader. Raises
        :class:`InflightConflictError` when the same key has a different hash.
        """

        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(request_hash=request_hash, result_future=asyncio.get_running_loop().create_future())
            self._entries[key] = entry
            return "leader", None
        if entry.request_hash != request_hash:
            raise InflightConflictError(key)
        return "follower", entry.result_future

    async def follow(self, result_future: asyncio.Future) -> object:
        """Await the leader's result without propagating cancellation to it."""

        return await asyncio.shield(result_future)

    def publish(self, key: tuple[str, str, str], result: object | None, error: BaseException | None = None) -> None:
        """Publish the leader's final result/exception to every follower.

        A leader that was cancelled publishes :class:`InflightLeaderCancelled`
        rather than its ``CancelledError`` so followers are not marked
        cancelled themselves. The exception is marked retrieved immediately:
        the leader re-raises it on its own path, so a future nobody awaits
        (no followers joined) must not log "exception was never retrieved".
        """

        entry = self._entries[key]
        if error is None:
            entry.result_future.set_result(result)
            return
        if isinstance(error, asyncio.CancelledError):
            error = InflightLeaderCancelled()
        entry.result_future.set_exception(error)
        entry.result_future.exception()

    def release(self, key: tuple[str, str, str]) -> None:
        """Remove the entry; followers already hold a reference to the future."""

        self._entries.pop(key, None)


__all__ = ["InflightConflictError", "InflightLeaderCancelled", "InflightRegistry"]
