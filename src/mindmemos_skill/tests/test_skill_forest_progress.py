"""Forest progress under shared workers, early rejection, failures and recovery."""

import asyncio

import pytest
from mindmemos_skill.algos.train.skill_forest.scheduler import RolloutScheduler
from mindmemos_skill.common.json import read_json
from mindmemos_skill.logging import log_context
from mindmemos_skill.logging.tasks import async_task_log_reporter

from .test_skill_forest import config, run_training
from .test_skill_forest_scheduler import result, samples


async def test_scheduler_live_progress_and_early_stop_accounting(tmp_path):
    rows = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def sink(row):
        rows.append(row)
        if "group_submitted" in row["payload"]["message"]:
            started.set()

    async def execute(**kwargs):
        await release.wait()
        return result()

    scheduler = RolloutScheduler(execute, concurrency=1)
    try:
        async with async_task_log_reporter(sink, project_id="p", task_id="forest", attempt=1):
            task = asyncio.create_task(
                scheduler.collect(
                    env=None,
                    skill=None,
                    samples=samples(4),
                    directory=tmp_path,
                    stage_name="skill_forest.candidate.evaluation",
                    stop_when=lambda results: bool(results),
                )
            )
            try:
                await asyncio.wait_for(started.wait(), 5)
                assert "group_submitted" in rows[-1]["payload"]["message"]
                assert not task.done()
            finally:
                release.set()
                await task
        message = rows[-1]["payload"]["message"]
        assert "group_completed" in message and "stopped=True" in message
        assert "done=1" in message and "running=0 pending=0" in message
    finally:
        await scheduler.close()


async def test_shared_workers_keep_task_and_node_owners(tmp_path):
    rows = {"A": [], "B": []}
    a_started = asyncio.Event()
    release = asyncio.Event()

    async def execute(**kwargs):
        a_started.set()
        await release.wait()
        return result()

    scheduler = RolloutScheduler(execute, concurrency=1)

    async def submit(name):
        async def sink(row):
            rows[name].append(row)

        async with async_task_log_reporter(sink, project_id="p", task_id=name, attempt=1):
            with log_context(tree_id=name, node_id=f"node-{name}"):
                await scheduler.collect(env=None, skill=None, samples=samples(2), directory=tmp_path / name)

    try:
        a = asyncio.create_task(submit("A"))
        await a_started.wait()
        b = asyncio.create_task(submit("B"))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(a, b)
        for name, records in rows.items():
            completions = [row for row in records if "rollout_completed" in row["payload"]["message"]]
            assert len(completions) == 2
            assert all(row["task_id"] == name for row in completions)
            assert all(row["payload"]["node_id"] == f"node-{name}" for row in completions)
            assert all(f"group_id={tmp_path / name}" in row["payload"]["message"] for row in completions)
    finally:
        release.set()
        await scheduler.close()


async def test_scheduler_failure_never_becomes_zero_score(tmp_path):
    rows = []

    async def sink(row):
        rows.append(row)

    async def execute(**kwargs):
        raise RuntimeError("offline failure")

    scheduler = RolloutScheduler(execute, concurrency=1)
    try:
        async with async_task_log_reporter(sink, project_id="p", task_id="forest", attempt=1):
            with pytest.raises(RuntimeError, match="offline failure"):
                await scheduler.collect(env=None, skill=None, samples=samples(2), directory=tmp_path)
        messages = [row["payload"]["message"] for row in rows]
        assert any("rollout_failed" in message and "offline failure" in message for message in messages)
        assert not any("group_completed" in message for message in messages)
    finally:
        await scheduler.close()


async def test_training_summary_and_candidate_metrics(monkeypatch, tmp_path):
    rows = []

    async def sink(row):
        rows.append(row)

    async with async_task_log_reporter(sink, project_id="p", task_id="forest", attempt=1):
        await run_training(tmp_path, settings=config(num_trees=1, expansion_budget=1), monkeypatch=monkeypatch)
    messages = [row["payload"]["message"] for row in rows]
    final = next(message for message in reversed(messages) if "[skill_forest.training] completed" in message)
    report = read_json(tmp_path / "result.json")
    assert f"accepted_candidates={report['training']['summary']['decisions']['accepted']}" in final
    assert any("[skill_forest.candidate] progress" in message and "accepted=True" in message for message in messages)
    assert any(
        "[skill_forest.reflection] progress" in message and "extracted_items=" in message for message in messages
    )
