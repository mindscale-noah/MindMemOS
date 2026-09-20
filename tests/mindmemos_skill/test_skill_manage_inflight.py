"""AC-P4-002 (in-flight portion): same-process idempotency registry."""

from __future__ import annotations

import asyncio

import pytest
from mindmemos_skill.application.skill_manage.inflight import InflightConflictError, InflightRegistry


@pytest.mark.asyncio
async def test_leader_and_follower_share_the_same_result() -> None:
    registry = InflightRegistry()
    key = ("p1", "g1", "k1")

    role, future = registry.begin(key, "hash-1")
    assert role == "leader"
    assert future is None

    follower_role, result_future = registry.begin(key, "hash-1")
    assert follower_role == "follower"
    assert result_future is not None

    async def leader() -> str:
        await asyncio.sleep(0)
        registry.publish(key, "committed-result")
        registry.release(key)
        return "committed-result"

    async def follower() -> object:
        assert result_future is not None
        return await registry.follow(result_future)

    leader_task = asyncio.create_task(leader())
    follower_task = asyncio.create_task(follower())
    leader_result, follower_result = await asyncio.gather(leader_task, follower_task)

    assert leader_result == "committed-result"
    assert follower_result == "committed-result"


@pytest.mark.asyncio
async def test_same_key_different_hash_conflicts_before_joining() -> None:
    registry = InflightRegistry()
    key = ("p1", "g1", "k1")
    registry.begin(key, "hash-1")

    with pytest.raises(InflightConflictError):
        registry.begin(key, "hash-2")


@pytest.mark.asyncio
async def test_follower_cancellation_does_not_cancel_leader() -> None:
    registry = InflightRegistry()
    key = ("p1", "g1", "k1")

    role, _ = registry.begin(key, "hash-1")
    assert role == "leader"
    _, result_future = registry.begin(key, "hash-1")
    assert result_future is not None

    leader_cancelled: list[bool] = []

    async def leader() -> None:
        try:
            await asyncio.sleep(0.05)
            registry.publish(key, "done")
        except asyncio.CancelledError:
            leader_cancelled.append(True)
            raise
        finally:
            registry.release(key)

    leader_task = asyncio.create_task(leader())
    follower_task = asyncio.create_task(registry.follow(result_future))

    # Cancel the follower before the leader finishes.
    await asyncio.sleep(0)
    follower_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower_task

    await leader_task
    assert leader_cancelled == []


@pytest.mark.asyncio
async def test_publish_propagates_exception_to_followers() -> None:
    registry = InflightRegistry()
    key = ("p1", "g1", "k1")

    registry.begin(key, "hash-1")
    _, result_future = registry.begin(key, "hash-1")

    error = RuntimeError("boom")
    registry.publish(key, None, error=error)
    registry.release(key)

    with pytest.raises(RuntimeError, match="boom"):
        await registry.follow(result_future)
