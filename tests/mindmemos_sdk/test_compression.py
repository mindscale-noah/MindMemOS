"""Offline contract tests for the single-snapshot compression API."""

from __future__ import annotations

import io
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest
from mindmemos_sdk.compression import (
    CompressionConfig,
    CompressionError,
    _transport,
    compress_memory,
    compressor,
)

USAGE = {
    "prompt_tokens": 300,
    "completion_tokens": 40,
    "cost": 0.01,
    "completion_tokens_details": {"reasoning_tokens": 15},
}


def envelope(content="summary", **message_fields):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": content, **message_fields}}],
        "usage": deepcopy(USAGE),
        "model": "returned-model",
    }


@pytest.fixture
def inputs():
    return dict(
        task="任务 {{ history }}",
        memory=["old {{ task }}", "earlier claim"],
        messages=[
            {
                "role": "assistant",
                "tool_calls": [{"id": "x", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "x", "content": "observed {{ prev_summary }}"},
        ],
        mode="incremental",
        config=CompressionConfig(api_key="private-key"),
    )


@pytest.fixture
def upstream(monkeypatch):
    calls = []

    def install(data=None):
        value = envelope() if data is None else data
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()

        def open_(url, body, headers, timeout, control):
            control.check()
            control.sending()
            calls.append((url, json.loads(body), headers, timeout))
            return io.BytesIO(raw)

        monkeypatch.setattr(compressor, "_open_upstream", open_)
        return calls

    return install


@pytest.mark.parametrize("mode", ["incremental", "full"])
def test_snapshot_request_and_result_contract(inputs, upstream, mode):
    inputs["mode"] = mode
    before = deepcopy(inputs)
    calls = upstream(envelope("  exact summary\n"))
    events, audit = [], []
    result = compress_memory(**inputs, event_sink=events.append, audit_sink=audit.append)
    assert result.summary == "  exact summary\n"
    assert result.usage == USAGE
    assert result.latency_seconds >= 0
    assert inputs == before
    url, request, headers, timeout = calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert set(request) == {"model", "messages", "stream", "reasoning"}
    assert request["model"] == "qwen/qwen3.5-122b-a10b"
    assert request["reasoning"] == {"effort": "high"}
    assert request["stream"] is False
    assert 239 < timeout <= 240
    assert headers["Authorization"] == "Bearer private-key"
    prompt = request["messages"][0]
    assert prompt["role"] == "user"
    title = "INCREMENTAL MEMORY DELTA" if mode == "incremental" else "FULL MEMORY CONSOLIDATION"
    assert prompt["content"].splitlines()[0] == title
    assert "任务 {{ history }}" in prompt["content"]
    assert "old {{ task }}\n\nearlier claim" in prompt["content"]
    assert "observed {{ prev_summary }}" in prompt["content"]
    assert json.dumps(inputs["messages"], ensure_ascii=False, indent=2) in prompt["content"]
    assert [e["event"] for e in events] == ["compressor_started", "compressor_usage"]
    assert events[-1]["success"] and events[-1]["request_dispatched"]
    assert events[-1]["usage"] == USAGE
    assert audit[0]["request"] == request
    assert json.loads(audit[1]["body"])["choices"][0]["message"]["content"] == result.summary
    assert "private-key" not in json.dumps(events + audit)
    assert "private-key" not in repr(inputs["config"])


@pytest.mark.parametrize(
    "data",
    [
        envelope(""),
        envelope(" \n"),
        envelope(None),
        envelope([{"text": "x"}]),
        envelope(refusal="refused"),
        envelope(tool_calls=[{"id": "x"}]),
        envelope(function_call={"name": "tool"}),
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": None}]},
        {"choices": [envelope()["choices"][0]] * 2},
        [],
        b'{"choices": [], "choices": []}',
        b'{"usage": NaN}',
        b"not json",
    ],
)
def test_invalid_responses_fail(inputs, upstream, data):
    upstream(data)
    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs)
    assert caught.value.code == "invalid_result"


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls", None])
def test_non_stop_finish_retains_billing_evidence(inputs, upstream, finish):
    data = envelope()
    data["choices"][0]["finish_reason"] = finish
    upstream(data)
    events, audit = [], []
    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs, event_sink=events.append, audit_sink=audit.append)
    assert caught.value.usage == USAGE
    assert events[-1]["usage"] == USAGE and not events[-1]["success"]
    assert json.loads(audit[-1]["body"])["usage"] == USAGE


def test_missing_usage_is_unknown(inputs, upstream):
    data = envelope()
    del data["usage"]
    upstream(data)
    assert compress_memory(**inputs).usage is None


def test_usage_only_keeps_finite_nonnegative_numeric_metadata(inputs, upstream):
    data = envelope()
    data["usage"] = {
        "prompt_tokens": True,
        "completion_tokens": -1,
        "cost": "secret",
        "total_tokens": 100,
        "extra": "secret",
        "input_tokens_details": {"cached_tokens": 40},
    }
    upstream(data)
    assert compress_memory(**inputs).usage == {"total_tokens": 100, "input_tokens_details": {"cached_tokens": 40}}


def test_callback_cannot_change_payload_result_or_inputs(inputs, upstream):
    calls = upstream()

    def mutate(event):
        if "request" in event:
            event["request"].clear()
            inputs["memory"].append("too late")
            inputs["messages"].clear()
            inputs["config"].reasoning["effort"] = "low"
        if event.get("usage"):
            event["usage"].clear()

    result = compress_memory(**inputs, audit_sink=mutate, event_sink=mutate)
    assert result.usage == USAGE
    assert calls[0][1]["reasoning"] == {"effort": "high"}
    assert "too late" not in calls[0][1]["messages"][0]["content"]


@pytest.mark.parametrize("fail_on", ["compressor_request", "compressor_response"])
def test_audit_failure_is_explicit_and_keeps_returned_usage(inputs, upstream, fail_on):
    calls = upstream()
    events = []

    def fail(event):
        if event["event"] == fail_on:
            raise OSError("sensitive storage path")

    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs, audit_sink=fail, event_sink=events.append)
    assert caught.value.code == "audit_write_error"
    assert "sensitive" not in str(caught.value)
    assert len(calls) == (fail_on == "compressor_response")
    assert caught.value.usage == (USAGE if calls else None)
    assert events[-2]["event"] == "compression_audit_error"


def test_telemetry_failure_does_not_change_success(inputs, upstream):
    upstream()

    def fail(_event):
        raise RuntimeError("logging failed")

    assert compress_memory(**inputs, event_sink=fail).summary == "summary"


@pytest.mark.parametrize(
    ("exception", "code"),
    [
        (TimeoutError("sensitive"), "timeout"),
        (URLError(TimeoutError("sensitive")), "timeout"),
        (URLError("sensitive"), "transport_error"),
    ],
)
def test_transport_errors_are_safe_and_not_retried(inputs, monkeypatch, exception, code):
    open_ = Mock(side_effect=exception)
    monkeypatch.setattr(compressor, "_open_upstream", open_)
    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs)
    assert caught.value.code == code
    assert "sensitive" not in str(caught.value)
    assert caught.value.usage is None
    assert open_.call_count == 1


def test_http_error_preserves_usage_and_raw_audit(inputs, monkeypatch):
    data = {"error": "upstream rejected", "usage": USAGE}
    error = HTTPError("http://example.test", 429, "provider message", {}, io.BytesIO(json.dumps(data).encode()))
    monkeypatch.setattr(compressor, "_open_upstream", Mock(side_effect=error))
    events, audit = [], []
    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs, event_sink=events.append, audit_sink=audit.append)
    assert caught.value.code == "upstream_http_error"
    assert caught.value.usage == USAGE
    assert events[-1]["status"] == 429
    assert json.loads(audit[-1]["body"]) == data


def test_response_size_is_bounded(inputs, upstream, monkeypatch):
    monkeypatch.setattr(compressor, "_MAX_RESPONSE_BYTES", 10)
    upstream()
    with pytest.raises(CompressionError, match="size limit"):
        compress_memory(**inputs)


def test_cancellation_is_not_a_public_input(inputs, upstream):
    calls = upstream()
    with pytest.raises(TypeError, match="cancellation"):
        compress_memory(**inputs, cancellation=True)
    assert not calls


def test_deadline_after_response_keeps_usage(inputs, monkeypatch):
    expired = threading.Event()
    sock = Mock()
    sock.shutdown.side_effect = lambda _: expired.set()

    def open_(url, body, headers, timeout, control):
        control.register(sock)
        control.sending()
        return io.BytesIO(json.dumps(envelope()).encode())

    monkeypatch.setattr(compressor, "_open_upstream", open_)
    inputs["config"] = CompressionConfig(api_key="", timeout_seconds=0.05)

    def audit(event):
        if event["event"] == "compressor_response":
            assert expired.wait(2)

    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs, audit_sink=audit)
    assert caught.value.code == "timeout"
    assert caught.value.usage == USAGE and caught.value.request_dispatched


def test_deadline_interrupts_only_its_call(inputs, monkeypatch):
    entered, released = threading.Event(), threading.Event()
    sock = Mock()
    sock.shutdown.side_effect = lambda _how: released.set()

    def open_(url, body, headers, timeout, control):
        control.sending()
        if "first-task" in json.loads(body)["messages"][0]["content"]:
            control.register(sock)
            entered.set()
            assert released.wait(2)
            control.check()
        return io.BytesIO(json.dumps(envelope()).encode())

    monkeypatch.setattr(compressor, "_open_upstream", open_)
    with ThreadPoolExecutor(2) as pool:
        pending = pool.submit(
            compress_memory,
            **{**inputs, "task": "first-task", "config": CompressionConfig(api_key="", timeout_seconds=0.05)},
        )
        assert entered.wait(2)
        result = pool.submit(compress_memory, **inputs).result(2)
        with pytest.raises(CompressionError) as caught:
            pending.result(2)
    assert result.summary == "summary" and caught.value.code == "timeout"
    sock.shutdown.assert_called_with(socket.SHUT_RDWR)


def test_total_deadline_interrupts_established_io(inputs, monkeypatch):
    released = threading.Event()
    sock = Mock()
    sock.shutdown.side_effect = lambda _how: released.set()

    def open_(url, body, headers, timeout, control):
        control.register(sock)
        control.sending()
        assert released.wait(2)
        raise OSError("socket interrupted")

    monkeypatch.setattr(compressor, "_open_upstream", open_)
    inputs["config"] = CompressionConfig(api_key="", timeout_seconds=0.05)
    events = []
    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs, event_sink=events.append)
    assert caught.value.code == "timeout"
    assert events[-1]["timed_out"] and events[-1]["request_dispatched"]


def test_transport_refuses_redirects():
    assert _transport._NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test") is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", ""),
        ("memory", "one string"),
        ("memory", [None]),
        ("messages", []),
        ("messages", [{"content": float("nan")}]),
        ("messages", [None]),
        ("mode", "boundary"),
    ],
)
def test_bad_snapshot_fails_before_network(inputs, upstream, field, value):
    calls = upstream()
    inputs[field] = value
    with pytest.raises(ValueError):
        compress_memory(**inputs)
    assert not calls


@pytest.mark.parametrize(
    "config",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": 1e100},
        {"base_url": "ftp://host"},
        {"base_url": "https://user:password@host"},
        {"base_url": "https://host:bad"},
        {"base_url": "https://host?key=secret"},
        {"model": "has space"},
        {"reasoning": {}},
        {"headers": {"bad\nheader": "x"}},
        {"api_key": "key\nheader"},
    ],
)
def test_bad_config_fails_early(config):
    with pytest.raises(ValueError):
        CompressionConfig(**{"api_key": "", **config})


def test_explicit_model_connection_config_does_not_inherit_memory_settings(inputs, upstream):
    calls = upstream()
    inputs["config"] = CompressionConfig(
        api_key="local",
        base_url="http://example.test",
        model="local-model",
        reasoning=None,
        headers={"Authorization": "wrong", "X-Title": "app"},
    )
    compress_memory(**inputs)
    assert calls[0][0] == "http://example.test/v1/chat/completions"
    assert "reasoning" not in calls[0][1]
    assert calls[0][2]["Authorization"] == "Bearer local"


def test_deep_provider_usage_cannot_escape_as_recursion_error(inputs, upstream):
    nested = {"reasoning_tokens": 1}
    for _ in range(600):
        nested = {"completion_tokens_details": nested}
    data = envelope()
    data["usage"] = {"prompt_tokens": 9, "completion_tokens_details": nested}
    upstream(data)
    events = []
    result = compress_memory(**inputs, event_sink=events.append)
    assert result.summary == "summary"
    assert result.usage == {"prompt_tokens": 9}
    assert events[-1]["usage"] == result.usage


def test_invalid_summary_with_deep_usage_returns_safe_error(inputs, upstream):
    nested = {"reasoning_tokens": 1}
    for _ in range(600):
        nested = {"completion_tokens_details": nested}
    data = envelope("")
    data["usage"] = {"prompt_tokens": 9, "completion_tokens_details": nested}
    upstream(data)
    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs)
    assert caught.value.code == "invalid_result"
    assert caught.value.usage == {"prompt_tokens": 9}


@pytest.mark.parametrize("field", ["task", "memory", "messages"])
def test_invalid_input_unicode_is_rejected_before_dispatch(inputs, upstream, field):
    calls = upstream()
    invalid = "bad\ud800"
    inputs[field] = {"task": invalid, "memory": [invalid], "messages": [{"role": "assistant", "content": invalid}]}[
        field
    ]
    with pytest.raises(ValueError, match="UTF-8 JSON") as caught:
        compress_memory(**inputs)
    assert not isinstance(caught.value, UnicodeError)
    assert not calls


def test_invalid_output_unicode_is_rejected_with_usage(inputs, upstream):
    upstream(envelope("bad\ud800"))
    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs)
    assert caught.value.code == "invalid_result"
    assert caught.value.usage == USAGE


def test_deep_config_is_validation_error_not_recursion_error():
    nested = {"effort": "high"}
    for _ in range(600):
        nested = {"nested": nested}
    with pytest.raises(ValueError, match="nesting"):
        CompressionConfig(api_key="", reasoning=nested)


def test_incomplete_http_framing_is_rejected_even_with_valid_json(inputs, monkeypatch):
    class Truncated(io.BytesIO):
        length = 20  # Upstream promised more bytes than it delivered.

    monkeypatch.setattr(compressor, "_open_upstream", lambda *args: Truncated(json.dumps(envelope()).encode()))
    audit = []
    with pytest.raises(CompressionError) as caught:
        compress_memory(**inputs, audit_sink=audit.append)
    assert caught.value.code == "transport_error"
    assert caught.value.usage == USAGE
    assert audit[-1]["event"] == "compressor_response" and not audit[-1]["body_complete"]


def test_dns_stall_obeys_deadline_without_late_connection(inputs, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    connect = Mock(side_effect=AssertionError("must not connect after DNS timeout"))
    slots = threading.BoundedSemaphore(1)

    def stalled_dns(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(2)
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 1))]
        finally:
            finished.set()

    monkeypatch.setattr(_transport, "_RESOLVER_SLOTS", slots)
    monkeypatch.setattr(socket, "getaddrinfo", stalled_dns)
    monkeypatch.setattr(socket.socket, "connect", connect)
    inputs["config"] = CompressionConfig(api_key="", base_url="http://127.0.0.1:1/v1", timeout_seconds=0.1)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(compress_memory, **inputs)
        try:
            assert entered.wait(1)
            with pytest.raises(CompressionError) as caught:
                pending.result(0.5)
            assert caught.value.code == "timeout"
            assert not caught.value.request_dispatched
        finally:
            release.set()
            assert finished.wait(1)
            assert slots.acquire(timeout=1)
            slots.release()
    connect.assert_not_called()


def test_saturated_dns_workers_time_out_without_creating_more(inputs, monkeypatch):
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    resolve = Mock(side_effect=AssertionError("resolver capacity must stay bounded"))
    monkeypatch.setattr(_transport, "_RESOLVER_SLOTS", slots)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    inputs["config"] = CompressionConfig(api_key="", base_url="http://127.0.0.1:1/v1", timeout_seconds=0.05)
    try:
        with pytest.raises(CompressionError) as caught:
            compress_memory(**inputs)
        assert caught.value.code == "timeout"
        assert not caught.value.request_dispatched
        resolve.assert_not_called()
    finally:
        slots.release()
