"""Verify request tracing, durable propagation, timing and failure boundaries."""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.factory import create_service
from mindmemos_skill.api.typing import ListRequest
from mindmemos_skill.config.config import FastAPIConfig, HttpAuthConfig
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.infra.observability import OpenTelemetryRecorder
from mindmemos_skill.infra.observability.otlp import AsyncOTLPExporter
from mindmemos_skill.infra.task_manager import TaskSpec, TaskStatus
from mindmemos_skill.infra.task_manager.kafka import KafkaTaskManager
from mindmemos_skill.logging import get_logger, log_context, stage, traced
from mindmemos_skill.logging.context import _context
from mindmemos_skill.logging.tracing import bind_tracer
from mindmemos_skill.persistence.constants import SKILL_TASKS_TABLE
from mindmemos_skill.persistence.db_services.tasks import TaskDataService
from mindmemos_skill.persistence.store import SkillPersistence, SkillPersistenceConfig
from opentelemetry.sdk.trace import SpanProcessor
from pydantic import ValidationError

from .service_config import repository_config
from .test_fastapi import GATEWAY_SECRET, gateway_token
from .test_service import deployment_config


@pytest.fixture
async def task_store(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'tasks.db'}"))
    await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage()
    try:
        yield TaskDataService(database)
    finally:
        await database.close()


@pytest.fixture
async def recorder():
    spans = []
    telemetry = OpenTelemetryRecorder(
        service_name="mindmemos-skill",
        instrumentation_name="test",
        on_span_end=spans.append,
        on_log_emit=lambda _: None,
        on_export_error=lambda error: pytest.fail(str(error)),
    )
    try:
        yield telemetry, spans
    finally:
        await telemetry.close()


async def test_saved_trace_parent_headers_recovery_and_attempt_timing(task_store, recorder):
    telemetry, spans = recorder
    headers = []

    class Producer:
        async def send(self, topic, value, *, key, headers):
            captured.append(headers)
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return future

    captured = headers

    async def handler(spec):
        with stage("evaluate"):
            get_logger(__name__).info("candidate.selected")
        return {"ok": True}

    manager = KafkaTaskManager(
        store=task_store,
        bootstrap_servers="unused",
        topic="test",
        group_id="test",
        handler=handler,
        tracer=telemetry.tracer,
    )
    manager._producer = Producer()
    manager._started = True
    with log_context(request_id="request-original"), telemetry.tracer.start_as_current_span("service.submit") as submit:
        spec = TaskSpec(task_id="recover", project_id="project", kind="test", payload={})
        await manager.submit(spec)
    first, _ = await task_store.claim(spec.task_id, lease_seconds=0)
    accepted_at = first.created_at
    await manager._notify(spec.task_id)
    await manager._execute(spec.task_id)
    record = await task_store.load(spec.task_id)
    assert record.created_at == accepted_at
    assert record.attempt == 2
    assert record.finished_at >= record.started_at >= accepted_at
    assert record.snapshot().end_to_end_ms >= record.snapshot().execution_ms
    assert all(dict(value)["traceparent"].decode() == spec.trace_context["traceparent"] for value in headers)
    roots = [span for span in spans if span.name in {"task.kafka.publish", "task.execute"}]
    assert len(roots) == 3
    assert all(span.parent.span_id == submit.get_span_context().span_id for span in roots)
    assert all(span.context.trace_id == submit.get_span_context().trace_id for span in spans)
    logs = await task_store.logs("project", spec.task_id)
    assert all(log.payload["request_id"] == "request-original" for log in logs)
    assert all(log.attempt == 2 for log in logs)
    assert any(log.payload["message"].startswith("stage.completed | ") for log in logs)
    manager._producer = None
    await manager.close()


async def test_result_write_failure_remains_recoverable(task_store, recorder, monkeypatch, caplog):
    telemetry, _ = recorder

    async def handler(spec):
        return {"ok": True}

    manager = KafkaTaskManager(
        store=task_store,
        bootstrap_servers="unused",
        topic="test",
        group_id="test",
        handler=handler,
        tracer=telemetry.tracer,
    )
    spec = TaskSpec(task_id="write-failed", project_id="project", kind="test", payload={})
    await task_store.create(spec)
    monkeypatch.setattr(task_store, "finish", AsyncMock(side_effect=OSError("Result storage unavailable")))
    with caplog.at_level(logging.INFO), pytest.raises(OSError):
        await manager._execute(spec.task_id)
    record = await task_store.load(spec.task_id)
    assert record.status == TaskStatus.RUNNING
    assert record.finished_at is None
    assert not any(
        log.payload["message"].startswith("task.finished | ") for log in await task_store.logs("project", spec.task_id)
    )
    assert any("Result storage unavailable" in (row.exc_text or "") for row in caplog.records)
    await manager.close()


async def test_stage_failure_auto_events_are_redacted_and_traceback_is_not_repeated(recorder, caplog):
    telemetry, spans = recorder

    @traced("outer")
    async def operation():
        with stage("evaluate"):
            raise ValueError("password=private-value")

    with caplog.at_level(logging.INFO), bind_tracer(telemetry.tracer), pytest.raises(ValueError):
        await operation()
    stage_span = next(span for span in spans if span.name == "evaluate")
    assert stage_span.status.status_code.name == "ERROR"
    assert len(stage_span.events) == 1
    event = stage_span.events[0]
    assert "private-value" not in json.dumps(dict(event.attributes))
    assert "Traceback" in event.attributes["exception.stacktrace"]
    assert len([row for row in caplog.records if row.exc_text]) == 1


async def test_http_failure_records_input_output_and_child_service_span(tmp_path, caplog):
    service = await create_service(deployment_config({"home": tmp_path / "home"}, "project"))
    spans = []
    service.telemetry.provider.add_span_processor(_SpanCollector(spans))
    try:
        app = create_app(service)
        with caplog.at_level(logging.INFO):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
                response = await client.post("/v1/skills/list", json={"project_id": "wrong", "password": "hidden"})
        assert response.status_code == 422
        event = next(row for row in caplog.records if row.msg == "http.request.completed")
        assert event.request_id == response.json()["request_id"]
        assert event.event_body["input"]["content"]["password"] == "[REDACTED]"
        assert event.event_body["output"]["content"]["code"] == "error"
        assert event.event_body["completed"]
        assert event.event_body["duration_ms"] >= 0
        with caplog.at_level(logging.INFO), pytest.raises(ValueError):
            await service.list_skill_versions("skill", "wrong")
        assert any(span.name == "service.list_skill_versions" for span in spans)
    finally:
        await service.close()


class _SpanCollector(SpanProcessor):
    def __init__(self, spans):
        self.spans = spans

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        self.spans.append(span)


async def test_http_projects_correlate_logs_and_spans_without_cross_request_leaks(tmp_path, caplog):
    """Use verified identities for concurrent successes, rejections and validation errors."""

    def authorize(principal, project_id):
        assert principal.project_id == project_id
        return Scope(fields={"project_id": project_id})

    service = await create_service(repository_config(tmp_path / "home").service, project_authorizer=authorize)
    spans = []
    service.telemetry.provider.add_span_processor(_SpanCollector(spans))
    app = create_app(
        service,
        api_config=FastAPIConfig(auth=HttpAuthConfig(mode="gateway_jwt", gateway_jwt_secret=GATEWAY_SECRET)),
    )
    cases = [
        ("project-a", ["skill:read"], {}, 200),
        ("project-b", ["skill:read"], {}, 200),
        ("project-c", ["skill:read"], {"project_id": "forged"}, 422),
        ("project-d", ["memory:read"], {}, 403),
        (None, [], {"project_id": "forged"}, 401),
    ]
    try:
        with caplog.at_level(logging.INFO):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
                responses = await asyncio.gather(
                    *[
                        client.post(
                            "/v1/skills/list",
                            json=body,
                            headers={"Authorization": f"Bearer {gateway_token(project_id=project, scopes=scopes)}"}
                            if project
                            else {},
                        )
                        for project, scopes, body, _ in cases
                    ]
                )
        for (project, _, _, status), response in zip(cases, responses, strict=True):
            assert response.status_code == status, response.text
            request_id = response.json()["request_id"]
            request_spans = [span for span in spans if span.attributes.get("request_id") == request_id]
            assert any(span.name == "http.request" for span in request_spans)
            if status == 200:
                assert any(span.name == "service.list_skills" for span in request_spans)
            assert all(span.attributes.get("project_id") == project for span in request_spans)
            records = [record for record in caplog.records if record.__dict__.get("request_id") == request_id]
            assert any(record.msg == "http.request.completed" for record in records)
            assert all(record.__dict__.get("project_id") == project for record in records)
        assert _context.get() == {}
    finally:
        await service.close()


@pytest.mark.parametrize("payload", [ListRequest(project_id="project"), {"project_id": "project"}, {}])
async def test_python_service_calls_bind_project_without_http(tmp_path, caplog, payload):
    """Keep project correlation for typed requests, dictionaries and local defaults."""
    service = await create_service(repository_config(tmp_path / "home", "project"))
    spans = []
    service.telemetry.provider.add_span_processor(_SpanCollector(spans))
    try:
        with caplog.at_level(logging.INFO):
            if payload == {}:
                with pytest.raises(ValidationError):
                    await service.list_skills(payload)
            else:
                assert await service.list_skills(payload) == []
            assert await service.list_graph("project") == []
        assert all(span.attributes.get("project_id") == "project" for span in spans)
        records = [record for record in caplog.records if record.msg in {"traced call", "traced call failed"}]
        assert records
        assert all(record.project_id == "project" for record in records)
        assert _context.get() == {}
    finally:
        await service.close()


@pytest.mark.parametrize("existing,race", [(True, False), (False, False), (False, True)])
async def test_topic_initialization_handles_existing_topics_and_creation_races(
    task_store, monkeypatch, caplog, existing, race
):
    from aiokafka.admin import AIOKafkaAdminClient

    start, close = AsyncMock(), AsyncMock()
    create = AsyncMock(return_value=SimpleNamespace(topic_errors=[("topic", 36 if race else 0, "")]))
    monkeypatch.setattr(AIOKafkaAdminClient, "start", start)
    monkeypatch.setattr(AIOKafkaAdminClient, "close", close)
    monkeypatch.setattr(AIOKafkaAdminClient, "list_topics", AsyncMock(return_value=["topic"] if existing else []))
    monkeypatch.setattr(AIOKafkaAdminClient, "create_topics", create)
    manager = KafkaTaskManager(store=task_store, bootstrap_servers="unused", topic="topic", group_id="group")
    with caplog.at_level(logging.INFO):
        await manager._ensure_topic()
    assert create.await_count == (0 if existing else 1)
    close.assert_awaited_once()
    expected = "task.kafka.topic.ready" if existing or race else "task.kafka.topic.created"
    assert any(row.msg == expected for row in caplog.records)
    await manager.close()


async def test_fenced_finish_cannot_report_success_for_a_reclaimed_attempt(task_store):
    await task_store.create(TaskSpec(task_id="fenced", project_id="project", kind="test", payload={}))
    first, old_token = await task_store.claim("fenced", lease_seconds=0)
    second, token = await task_store.claim("fenced")
    assert second.created_at == first.created_at
    assert not await task_store.finish("fenced", old_token, result={})
    assert (await task_store.load("fenced")).finished_at is None
    assert await task_store.finish("fenced", token, error_type="ValueError")
    snapshot = (await task_store.load("fenced")).snapshot()
    assert snapshot.status == TaskStatus.FAILED
    assert snapshot.finished_at is not None
    assert snapshot.end_to_end_ms >= snapshot.execution_ms


async def test_running_wait_and_execution_time_keep_advancing(task_store):
    spec = TaskSpec(task_id="elapsed", project_id="project", kind="test", payload={})
    await task_store.create(spec)
    await task_store.database.update(
        SKILL_TASKS_TABLE, {"created_at": datetime.now(UTC) - timedelta(seconds=10)}, filters={"task_id": spec.task_id}
    )
    snapshot = (await task_store.load(spec.task_id)).snapshot()
    assert snapshot.waiting_ms >= 10000
    assert snapshot.execution_ms is None
    _, token = await task_store.claim(spec.task_id)
    before = (await task_store.load(spec.task_id)).snapshot().execution_ms
    await asyncio.sleep(0.01)
    assert (await task_store.load(spec.task_id)).snapshot().execution_ms > before
    await task_store.finish(spec.task_id, token, result={})


async def test_otlp_export_failure_prints_traceback_without_recursive_export(monkeypatch, caplog):
    exporter = AsyncOTLPExporter("http://unused", 0.1)
    monkeypatch.setattr(exporter, "_send", AsyncMock(side_effect=OSError("Collector unavailable")))
    monkeypatch.setattr(
        "mindmemos_skill.infra.observability.otlp.encode_spans",
        lambda _: SimpleNamespace(SerializeToString=lambda: b"span"),
    )
    exporter.pending = [("traces", object())]
    try:
        with caplog.at_level(logging.ERROR):
            await exporter._drain()
        record = next(row for row in caplog.records if row.msg == "otel.export.failed")
        assert record.skip_otlp
        assert "Collector unavailable" in (record.exc_text or str(record.exc_info))
        assert exporter.pending == []
    finally:
        await exporter.close()


async def test_worker_interruption_keeps_lease_recovery_without_error_span(task_store, recorder, caplog):
    telemetry, spans = recorder
    ready = asyncio.Event()

    async def handler(spec):
        with stage("train"):
            ready.set()
            await asyncio.Event().wait()

    manager = KafkaTaskManager(
        store=task_store,
        bootstrap_servers="unused",
        topic="test",
        group_id="test",
        handler=handler,
        tracer=telemetry.tracer,
    )
    await task_store.create(TaskSpec(task_id="interrupted", project_id="project", kind="test", payload={}))
    with caplog.at_level(logging.INFO):
        worker = asyncio.create_task(manager._execute("interrupted"))
        await ready.wait()
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
    assert (await task_store.load("interrupted")).finished_at is None
    assert (await task_store.load("interrupted")).status == TaskStatus.RUNNING
    assert all(span.status.status_code.name != "ERROR" for span in spans)
    assert not any(row.exc_text or row.exc_info for row in caplog.records)
    assert any(row.msg == "task.interrupted" for row in caplog.records)
    await manager.close()


async def test_http_file_stream_is_passed_through_with_only_metadata(caplog):
    from mindmemos_skill.logging.http import RequestLoggingMiddleware

    part = b"file-bytes" * 100000
    output = []

    async def application(scope, receive, send):
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/octet-stream")]}
        )
        await send({"type": "http.response.body", "body": part, "more_body": True})
        await send({"type": "http.response.body", "body": part, "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        output.append(message)

    middleware = RequestLoggingMiddleware(application, tracer=lambda: None)
    with caplog.at_level(logging.INFO):
        await middleware({"type": "http", "method": "GET", "path": "/file", "headers": []}, receive, send)
    event = next(row for row in caplog.records if row.msg == "http.request.completed")
    assert event.event_body["output"]["bytes"] == 2 * len(part)
    assert "content" not in event.event_body["output"]
    assert output[1]["body"] is part and output[2]["body"] is part
    assert event.event_body["completed"]
