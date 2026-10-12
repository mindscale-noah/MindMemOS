"""Live TreeSkill progress, stage metrics and shared task/console records."""

import asyncio
import io
import logging

import pytest
from mindmemos_skill.algos.train.treeskill.trainer import sample_tasks
from mindmemos_skill.logging import get_logger
from mindmemos_skill.logging.config import attach_capture_handler, detach_capture_handler
from mindmemos_skill.logging.records import ContextFormatter
from mindmemos_skill.logging.tasks import async_task_log_reporter

from . import treeskill_smoke as smoke
from .test_treeskill_trainer_recovery import source_skill
from .test_treeskill_training import make_sample


@pytest.mark.asyncio
async def test_live_rollout_errors_recovery_and_console_match(tmp_path):
    released = asyncio.Event()
    progress_seen = asyncio.Event()
    rows = []

    async def sink(row):
        rows.append(row)
        if "sample_id=error" in row["payload"]["message"]:
            progress_seen.set()

    class Environment(smoke.SmokeEnvironment):
        async def _run(self, **kwargs):
            if kwargs["task"].id == "error":
                raise RuntimeError("offline execution error")
            if kwargs["task"].id == "slow":
                await released.wait()
            return await super()._run(**kwargs)

    console = io.StringIO()
    handler = logging.StreamHandler(console)
    handler.setFormatter(ContextFormatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    attach_capture_handler(handler)
    try:
        async with async_task_log_reporter(sink, project_id="test", task_id="training", attempt=1) as flush:
            kwargs = dict(
                env=Environment(smoke.SmokeConfig()),
                skill=source_skill(),
                samples=[make_sample(name) for name in ("fast", "error", "slow")],
                directory=tmp_path,
                repeats=1,
                concurrency=3,
                stage_name="treeskill.baseline",
            )
            sampling = asyncio.create_task(sample_tasks(**kwargs))
            try:
                await asyncio.wait_for(progress_seen.wait(), timeout=5)
                assert not sampling.done()
                assert "offline execution error" in rows[-1]["payload"]["message"]
                assert rows[-1]["payload"]["body"] is None
            finally:
                released.set()
                await sampling
            await sample_tasks(**kwargs)
            await flush()
        messages = [row["payload"]["message"] for row in rows]
        assert any("reused=" in message for message in messages)
        assert any("sample_id=slow" in message for message in messages)
        text = console.getvalue()
        assert all(message in text for message in messages)
        assert "project_id=test task_id=training attempt=1" in text
        assert all(row["payload"]["body"] is None for row in rows)
    finally:
        detach_capture_handler(handler)


@pytest.mark.asyncio
async def test_smoke_stage_metrics_match_persisted_artifacts(tmp_path):
    rows = []

    async def sink(row):
        rows.append(row)

    async with async_task_log_reporter(sink, project_id="test", task_id="training", attempt=1):
        summary = await smoke.run_smoke(tmp_path / "experiment")
    messages = [row["payload"]["message"] for row in rows]
    assert any(
        "[treeskill.train.analysis] progress" in message and "extracted_items=1" in message for message in messages
    )
    assert any(
        "[treeskill.train.localization] progress" in message and "located_evidence=1" in message for message in messages
    )
    assert any(
        "[treeskill.train.fusion] progress" in message and "accepted_patches=1" in message for message in messages
    )
    final = next(message for message in reversed(messages) if "[treeskill.training] completed" in message)
    assert f"accepted_patches={summary['training']['accepted_patches']}" in final


@pytest.mark.asyncio
async def test_cancelled_stage_logs_cancellation_without_background_work():
    rows = []
    started = asyncio.Event()

    async def sink(row):
        rows.append(row)

    async def run():
        get_logger(__name__).info("analysis.started | item_id=one")
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            get_logger(__name__).info("analysis.cancelled | item_id=one")
            raise

    async with async_task_log_reporter(sink, project_id="test", task_id="training", attempt=1) as flush:
        task = asyncio.create_task(run())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await flush()
        count = len(rows)
        await asyncio.sleep(0)
        await flush()
        assert len(rows) == count
    assert rows[-1]["payload"]["message"] == "analysis.cancelled | item_id=one"
