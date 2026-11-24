"""Ordinary f-string logs inherit task ownership and share a readable format."""

import asyncio
import io
import logging

from mindmemos_skill.logging import async_task_log_reporter, log_context
from mindmemos_skill.logging.config import attach_capture_handler, detach_capture_handler
from mindmemos_skill.logging.records import ContextFormatter


async def test_plain_logger_child_context_fifo_and_task_isolation():
    logger = logging.getLogger("mindmemos_skill.algorithm.demo")
    messages = {}
    console = io.StringIO()
    handler = logging.StreamHandler(console)
    handler.setFormatter(ContextFormatter())
    attach_capture_handler(handler)

    async def execute(project_id, task_id, marker):
        async def report(row):
            await asyncio.sleep(0)
            assert (row["project_id"], row["task_id"], row["attempt"]) == (project_id, task_id, 1)
            messages.setdefault((project_id, task_id), []).append(row["payload"])

        async def child():
            logger.info(f"analysis.completed | sample_id={marker} | score=1")

        async with async_task_log_reporter(report, project_id=project_id, task_id=task_id, attempt=1):
            with log_context(node_id=marker):
                logger.info(f"analysis.started | sample_id={marker}")
                await asyncio.create_task(child())

    try:
        await asyncio.gather(execute("A", "1", "A1"), execute("A", "2", "A2"), execute("B", "1", "B1"))
        for identity, marker in ((("A", "1"), "A1"), (("A", "2"), "A2"), (("B", "1"), "B1")):
            rows = messages[identity]
            assert [row["message"] for row in rows] == [
                f"analysis.started | sample_id={marker}",
                f"analysis.completed | sample_id={marker} | score=1",
            ]
            assert all(row["node_id"] == marker and row["body"] is None for row in rows)
        text = console.getvalue()
        assert "[INFO] [mindmemos_skill.algorithm.demo] | project_id=A task_id=1 attempt=1 node_id=A1" in text
        assert " | analysis.completed | sample_id=A1 | score=1" in text
        assert len(text.splitlines()) == 6
        logger.info("outside | status=idle")
        assert sum(map(len, messages.values())) == 6
    finally:
        detach_capture_handler(handler)


async def test_event_fields_render_once_without_json_appendix():
    logger = logging.getLogger("mindmemos_skill.algorithm.demo")
    rows = []
    console = io.StringIO()
    handler = logging.StreamHandler(console)
    handler.setFormatter(ContextFormatter())
    attach_capture_handler(handler)

    async def report(row):
        rows.append(row)

    try:
        async with async_task_log_reporter(report, project_id="p", task_id="t", attempt=1):
            logger.info("task.completed", extra={"event_body": {"elapsed": "1.2s", "status": "succeeded"}})
        assert rows[0]["payload"]["message"] == "task.completed | elapsed=1.2s status=succeeded"
        text = console.getvalue()
        assert text.count("elapsed=1.2s") == 1
        assert len(text.splitlines()) == 1
        assert '"status"' not in text
    finally:
        detach_capture_handler(handler)
