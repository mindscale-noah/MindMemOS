"""Verify decorator monitoring, correlation isolation and service-owned spans."""

import asyncio
import inspect
import logging

import pytest
from mindmemos_skill.logging import get_logger, traced
from mindmemos_skill.logging.context import _context
from mindmemos_skill.logging.tracing import bind_tracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_sync_metadata_redaction_and_span(caplog):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    @traced("test.call", record_args=True, record_result=True)
    def call(project_id: str, api_key: str):
        get_logger(__name__).info("Inside call")
        return {"token": "secret-result", "value": 3}

    assert call.__name__ == "call"
    assert list(inspect.signature(call).parameters) == ["project_id", "api_key"]
    with caplog.at_level(logging.INFO), bind_tracer(provider.get_tracer("test")):
        assert call("project", "secret-input")["value"] == 3
    record = next(record for record in caplog.records if record.msg == "traced call")
    assert record.event_body["status"] == "succeeded"
    assert record.event_body["duration_ms"] >= 0
    assert "secret-input" not in record.event_body["arguments"]
    assert "secret-result" not in record.event_body["result"]
    span = exporter.get_finished_spans()[0]
    assert span.name == "test.call"
    assert span.status.status_code.name == "OK"
    assert "secret-input" not in span.attributes["arguments"]
    assert _context.get() == {}
    provider.shutdown()


@pytest.mark.asyncio
async def test_async_nested_concurrent_context_and_failure(caplog):
    observed = []

    @traced()
    async def child():
        await asyncio.sleep(0)
        observed.append(_context.get().copy())

    @traced()
    async def parent(task_id: str):
        await child()
        if task_id == "failed":
            raise ValueError("password=private")
        return task_id

    with caplog.at_level(logging.INFO):
        results = await asyncio.gather(parent("success"), parent("failed"), return_exceptions=True)
    assert results[0] == "success"
    assert isinstance(results[1], ValueError)
    assert {row["task_id"] for row in observed} == {"success", "failed"}
    assert _context.get() == {}
    failure = next(record for record in caplog.records if record.msg == "traced call failed")
    assert failure.event_body["error_type"] == "ValueError"
    assert failure.exc_info or failure.exc_text


@pytest.mark.asyncio
async def test_cancellation_is_propagated(caplog):
    ready = asyncio.Event()

    @traced()
    async def wait():
        ready.set()
        await asyncio.Event().wait()

    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(wait())
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    failure = next(record for record in caplog.records if record.msg == "traced call cancelled")
    assert failure.event_body["status"] == "cancelled"
    assert not failure.exc_info and not failure.exc_text
    assert _context.get() == {}


def test_generator_rejected():
    with pytest.raises(TypeError, match="generator"):

        @traced()
        def stream():
            yield 1
