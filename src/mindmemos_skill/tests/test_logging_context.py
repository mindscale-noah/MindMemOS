"""Unified correlation, redaction and durable task log behavior."""

import asyncio

import pytest
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.infra.database.builtin_backends.sqlite import SQLiteDatabase
from mindmemos_skill.infra.database.config import SQLiteConfig
from mindmemos_skill.infra.task_manager import TaskSpec
from mindmemos_skill.logging import config, configure_logging, get_logger, log_context
from mindmemos_skill.logging.otel import attach_logs, detach_logs
from mindmemos_skill.logging.tasks import async_task_log_reporter
from mindmemos_skill.persistence.config import SkillPersistenceConfig
from mindmemos_skill.persistence.db_services.tasks import TaskWorkerService
from mindmemos_skill.persistence.store import SkillPersistence
from opentelemetry.sdk._logs import LoggerProvider, LogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture(autouse=True)
def restore_logging():
    logger = get_logger()
    old_level, old_propagate = logger.level, logger.propagate
    yield
    for handler in config._handlers:
        logger.removeHandler(handler)
        handler.close()
    config._handlers.clear()
    logger.setLevel(old_level)
    logger.propagate = old_propagate


async def test_same_log_reaches_console_database_and_otel_without_run(tmp_path, capsys):
    observed = []

    class Capture(LogRecordProcessor):
        def on_emit(self, record):
            observed.append(record.log_record)

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=30000):
            return True

    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'logs.db'}"))
    await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage()
    store = TaskWorkerService(database)
    await store.create(TaskSpec(project_id="p", task_id="t", kind="skill.train", payload={}))
    provider = LoggerProvider(shutdown_on_exit=False)
    provider.add_log_record_processor(Capture())
    handler = attach_logs(provider, context=lambda: None, capture_level="info")
    tracer_provider = TracerProvider()
    configure_logging()
    try:
        with tracer_provider.get_tracer("test").start_as_current_span("request"), log_context(request_id="r"):
            async with async_task_log_reporter(store.append_log, project_id="p", task_id="t", attempt=2):
                try:
                    raise ValueError("password=sample-password-23")
                except ValueError:
                    get_logger("worker").exception(
                        "failed token=sample-token-42",
                        extra={"event_body": {"api_key": "sample-key-81", "input": "x" * 40000}},
                    )
        console = capsys.readouterr().err
        rows = await store.scoped(Scope(fields={"project_id": "p"})).logs("t")
        assert len(rows) == len(observed) == 1
        payload = rows[0].payload
        assert rows[0].attempt == 2
        assert payload["trace_id"] == observed[0].attributes["trace_id"]
        assert payload["span_id"] == observed[0].attributes["span_id"]
        assert observed[0].trace_id == int(payload["trace_id"], 16)
        assert observed[0].span_id == int(payload["span_id"], 16)
        assert payload["trace_id"] in console
        assert "ValueError" in payload["traceback"]
        assert "ValueError" in observed[0].attributes["exception.stacktrace"]
        for text in (console, str(payload), str(observed[0].body), str(observed[0].attributes)):
            assert not any(secret in text for secret in ("sample-password-23", "sample-token-42", "sample-key-81"))
            assert "[REDACTED]" in text
        assert "truncated" in console
        assert await store.scoped(Scope(fields={"project_id": "other"})).logs("t") == []
        get_logger().info("outside task")
        assert len(observed) == 2
        assert len(await store.scoped(Scope(fields={"project_id": "p"})).logs("t")) == 1
    finally:
        detach_logs(handler)
        provider.shutdown()
        tracer_provider.shutdown()
        await database.close()


async def test_concurrent_scopes_are_isolated_and_drained_on_failure():
    rows = []

    async def sink(row):
        await asyncio.sleep(0)
        rows.append(row)

    async def run(task_id):
        with pytest.raises(RuntimeError):
            async with async_task_log_reporter(sink, project_id="p", task_id=task_id, attempt=1):
                for number in range(3):
                    get_logger().info("%s-%s", task_id, number)
                    await asyncio.sleep(0)
                raise RuntimeError("failed")

    await asyncio.gather(run("a"), run("b"))
    for task_id in ("a", "b"):
        assert [row["payload"]["message"] for row in rows if row["task_id"] == task_id] == [
            f"{task_id}-{number}" for number in range(3)
        ]


async def test_sink_failure_is_visible_without_breaking_execution(capsys):
    async def sink(row):
        raise RuntimeError("password=sample-password-23")

    async with async_task_log_reporter(sink, project_id="p", task_id="t", attempt=1):
        get_logger().info("business completed")
    text = capsys.readouterr().err
    assert "Task log persistence failed" in text and "Traceback" in text
    assert "hidden" not in text


async def test_worker_logs_survive_database_reopen(tmp_path):
    from mindmemos_skill.infra.task_manager.kafka import KafkaTaskManager

    path = tmp_path / "worker.db"
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage()
    store = TaskWorkerService(database)
    await store.create(TaskSpec(project_id="p", task_id="worker", kind="skill.train", payload={}))

    async def handler(spec):
        get_logger().info("ordinary algorithm log")
        raise ValueError("token=hidden")

    manager = KafkaTaskManager(store=store, bootstrap_servers="unused", topic="test", group_id="test", handler=handler)
    try:
        await manager._execute("worker")
    finally:
        await manager.close()
        await database.close()
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    try:
        await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage()
        rows = await TaskWorkerService(database).scoped(Scope(fields={"project_id": "p"})).logs("worker")
        assert any(row.payload["message"] == "ordinary algorithm log" for row in rows)
        assert all(row.attempt == 1 for row in rows)
        assert any("ValueError" in (row.payload["traceback"] or "") for row in rows)
        assert "hidden" not in str(rows)
    finally:
        await database.close()


async def test_cancellation_drains_logs():
    rows = []
    ready = asyncio.Event()

    async def sink(row):
        await asyncio.sleep(0.01)
        rows.append(row)

    async def run():
        async with async_task_log_reporter(sink, project_id="p", task_id="cancelled", attempt=1):
            get_logger().info("before cancellation")
            ready.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(rows) == 1
    assert rows[0]["payload"]["message"] == "before cancellation"


async def test_recovered_attempt_keeps_logs_and_paginates_after_reopen(tmp_path):
    from datetime import UTC, datetime, timedelta

    from mindmemos_skill.infra.task_manager.kafka import KafkaTaskManager
    from mindmemos_skill.persistence.constants import SKILL_TASKS_TABLE

    path = tmp_path / "recovery.db"
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage()
    store = TaskWorkerService(database)
    await store.create(TaskSpec(project_id="p", task_id="recover", kind="skill.train", payload={}))
    ready = asyncio.Event()
    interrupted = True

    async def handler(spec):
        if interrupted:
            get_logger().info("first attempt", extra={"event_body": {"stage": "prepare"}})
            ready.set()
            await asyncio.Event().wait()
        get_logger().info("second attempt", extra={"event_body": {"stage": "published"}})
        return {"ok": True}

    manager = KafkaTaskManager(store=store, bootstrap_servers="unused", topic="test", group_id="test", handler=handler)
    try:
        task = asyncio.create_task(manager._execute("recover"))
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        first = await store.scoped(Scope(fields={"project_id": "p"})).logs("recover")
        assert first and all(row.attempt == 1 for row in first)
        assert any(row.payload["message"].split(" | ", 1)[0] == "first attempt" for row in first)
        await database.update(
            SKILL_TASKS_TABLE,
            {"lease_expires_at": datetime.now(UTC) - timedelta(seconds=1)},
            filters={"task_id": "recover"},
        )
        interrupted = False
        await manager._execute("recover")
        snapshot = (await store.scoped(Scope(fields={"project_id": "p"})).get("recover")).snapshot()
        assert snapshot.attempt == 2 and snapshot.status == "succeeded"
        assert "progress" not in snapshot.model_dump()
    finally:
        await manager.close()
        await database.close()
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    try:
        await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage()
        store = TaskWorkerService(database)
        rows = []
        cursor = 0
        while page := await store.scoped(Scope(fields={"project_id": "p"})).logs("recover", after=cursor, limit=1):
            assert page[0].sequence > cursor
            rows.extend(page)
            cursor = page[0].sequence
        messages = [
            row for row in rows if row.payload["message"].split(" | ", 1)[0] in {"first attempt", "second attempt"}
        ]
        assert [row.attempt for row in messages] == [1, 2]
        assert [row.payload["message"].split(" | ", 1)[0] for row in messages] == ["first attempt", "second attempt"]
        assert await store.scoped(Scope(fields={"project_id": "p"})).logs("recover", after=cursor) == []
        assert await store.scoped(Scope(fields={"project_id": "other"})).logs("recover") == []
    finally:
        await database.close()
