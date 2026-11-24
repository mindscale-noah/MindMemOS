"""Deterministic queue fairness, utilization, cancellation and diagnostic checks."""

import asyncio
import json
from contextvars import ContextVar

import pytest
from mindmemos_skill.algos.train.skill_forest.scheduler import RolloutScheduler
from mindmemos_skill.envs.data.typing import DataSample, RewardData
from mindmemos_skill.envs.typing import Reward, RolloutResult, Task


def samples(count):
    return [
        DataSample(task=Task(id=str(index), instruction="public"), reward_data=RewardData()) for index in range(count)
    ]


def result():
    return RolloutResult(output="public", trajectory=[], reward=Reward(score=0, success=False, detail=""))


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


async def test_groups_round_robin_and_late_arrival(tmp_path):
    async def run():
        started = []
        first = asyncio.Event()
        release = asyncio.Event()

        async def execute(*, directory, **kwargs):
            started.append((directory.parent.name, directory.name))
            if len(started) == 1:
                first.set()
                await release.wait()
            return result()

        scheduler = RolloutScheduler(execute, concurrency=1, log_path=tmp_path / "scheduler.jsonl")
        try:
            a = asyncio.create_task(
                scheduler.collect(env=None, samples=samples(4), skill=None, directory=tmp_path / "A")
            )
            await first.wait()
            b = asyncio.create_task(
                scheduler.collect(env=None, samples=samples(2), skill=None, directory=tmp_path / "B")
            )
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(a, b)
            assert [group for group, _ in started] == ["A", "A", "B", "A", "B", "A"]
            assert len(set(started)) == 6
        finally:
            await scheduler.close()

    (await asyncio.wait_for(run(), timeout=5))
    log = events(tmp_path / "scheduler.jsonl")
    assert [event["sequence"] for event in log] == list(range(1, len(log) + 1))
    assert all(event["queue_wait_s"] >= 0 for event in log if event["event"] == "rollout_started")
    assert log[-1]["active"] == log[-1]["queued"] == 0


async def test_one_group_fills_workers_and_refills_without_batch_barrier(tmp_path):
    async def run():
        started = []
        full = asyncio.Event()
        replacement = asyncio.Event()
        release_first = asyncio.Event()
        release_rest = asyncio.Event()

        async def execute(*, sample, **kwargs):
            started.append(sample.task.id)
            if len(started) == 4:
                full.set()
            if sample.task.id == "0":
                await release_first.wait()
            else:
                if sample.task.id == "4":
                    replacement.set()
                await release_rest.wait()
            return result()

        scheduler = RolloutScheduler(execute, concurrency=4)
        try:
            job = asyncio.create_task(scheduler.collect(env=None, samples=samples(8), skill=None, directory=tmp_path))
            await full.wait()
            assert scheduler.active == 4
            release_first.set()
            await replacement.wait()
            assert started == ["0", "1", "2", "3", "4"]
            assert scheduler.active == 4
            release_rest.set()
            assert len(await job) == 8
        finally:
            release_first.set()
            release_rest.set()
            await scheduler.close()

    (await asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("stop", ["rejection", "cancel", "failure"])
async def test_stop_removes_queued_tasks_and_drains_started_calls(tmp_path, stop):
    async def run():
        full = asyncio.Event()
        first_return = asyncio.Event()
        second_return = asyncio.Event()
        rejected = asyncio.Event()
        started = []

        async def execute(*, sample, **kwargs):
            started.append(sample.task.id)
            if len(started) == 2:
                full.set()
            if sample.task.id == "0":
                await first_return.wait()
                if stop == "failure":
                    raise RuntimeError("offline failure")
            else:
                await second_return.wait()
            return result()

        def reject(results):
            if results:
                rejected.set()
                return True
            return False

        scheduler = RolloutScheduler(execute, concurrency=2, log_path=tmp_path / "scheduler.jsonl")
        try:
            job = asyncio.create_task(
                scheduler.collect(
                    env=None,
                    samples=samples(6),
                    skill=None,
                    directory=tmp_path / "validation",
                    stop_when=reject if stop == "rejection" else None,
                )
            )
            await full.wait()
            if stop == "cancel":
                job.cancel()
                await asyncio.sleep(0)
            first_return.set()
            if stop == "rejection":
                await rejected.wait()
            else:
                await asyncio.sleep(0)
            assert not job.done()
            assert started == ["0", "1"]
            second_return.set()
            if stop == "rejection":
                assert len(await job) == 2
            else:
                with pytest.raises(asyncio.CancelledError if stop == "cancel" else RuntimeError):
                    await job
            assert scheduler.active == 0
            assert not scheduler.queue
        finally:
            first_return.set()
            second_return.set()
            await scheduler.close()

    (await asyncio.wait_for(run(), timeout=5))
    log = events(tmp_path / "scheduler.jsonl")
    stopped = next(event for event in log if event["event"] == "group_stopped")
    assert stopped["removed"] == 4
    assert not any(event["event"] == "rollout_started" and event["sequence"] > stopped["sequence"] for event in log)
    assert any(event["event"] == "rollout_completed" and event["after_stop"] for event in log)


async def test_logging_failure_does_not_strand_workers(tmp_path, monkeypatch):
    async def run():
        async def execute(**kwargs):
            return result()

        def fail_log(path, event):
            if event["event"] != "group_submitted":
                raise OSError("disk full")

        monkeypatch.setattr("mindmemos_skill.algos.train.skill_forest.scheduler.append_jsonl", fail_log)
        scheduler = RolloutScheduler(execute, concurrency=2, log_path=tmp_path / "scheduler.jsonl")
        try:
            with pytest.raises(OSError, match="disk full"):
                await scheduler.collect(env=None, samples=samples(5), skill=None, directory=tmp_path)
            assert scheduler.active == 0
            assert not scheduler.queue
        finally:
            await scheduler.close()

    (await asyncio.wait_for(run(), timeout=5))


async def test_workers_preserve_each_submitting_context(tmp_path):
    identity = ContextVar("rollout_test_identity")

    async def run():
        seen = []

        async def execute(*, directory, **kwargs):
            seen.append((directory.parent.name, identity.get()))
            return result()

        scheduler = RolloutScheduler(execute, concurrency=2)

        async def submit(name):
            identity.set(name)
            await scheduler.collect(env=None, samples=samples(3), skill=None, directory=tmp_path / name)

        try:
            await asyncio.gather(submit("A"), submit("B"))
            assert len(seen) == 6
            assert all(name == context for name, context in seen)
        finally:
            await scheduler.close()

    (await asyncio.wait_for(run(), timeout=5))
