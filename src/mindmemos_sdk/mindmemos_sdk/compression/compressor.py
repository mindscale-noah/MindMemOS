"""One frozen snapshot in, one validated summary out."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from importlib.resources import files
from typing import Any
from urllib.error import HTTPError, URLError

from ._transport import _deadline, _open_upstream
from .models import CompressionConfig, CompressionError, CompressionMode, CompressionResult, _chat_url

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_USAGE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
    "audio_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
    "cost",
    "prompt_tokens_details",
    "completion_tokens_details",
    "input_tokens_details",
    "output_tokens_details",
    "cost_details",
    "upstream_inference_cost",
    "upstream_inference_prompt_cost",
    "upstream_inference_completions_cost",
}
EventSink = Callable[[dict[str, Any]], None]


def _safe_usage(value, depth=0):
    # Provider metadata must stay small enough for callbacks, results and error
    # copies. Deeply nested JSON must not escape as a RecursionError in finally.
    if not isinstance(value, dict) or depth > 4:
        return None
    result = {}
    for key, item in value.items():
        if key not in _USAGE_KEYS:
            continue
        if item is None:
            result[key] = None
        elif isinstance(item, dict):
            nested = _safe_usage(item, depth + 1)
            if nested:
                result[key] = nested
        elif type(item) in (int, float) and item >= 0:
            try:
                if math.isfinite(item):
                    result[key] = item
            except OverflowError:
                pass
    return result or None


def _strict_json(content):
    def reject_constant(_value):
        raise ValueError("non-finite constant")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result

    try:
        return json.loads(content, parse_constant=reject_constant, object_pairs_hook=unique_object)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise CompressionError("invalid compressor JSON") from None


def _build_messages(task, memory, messages, mode):
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a nonempty string")
    if mode not in ("incremental", "full"):
        raise ValueError("mode must be incremental or full")
    if not isinstance(memory, (list, tuple)) or any(not isinstance(item, str) for item in memory):
        raise ValueError("memory must be a sequence of strings")
    if not isinstance(messages, (list, tuple)) or not messages or any(not isinstance(m, dict) for m in messages):
        raise ValueError("messages must be a nonempty sequence of message dictionaries")
    try:
        history = json.dumps(list(messages), ensure_ascii=False, indent=2, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise ValueError("messages must be JSON serializable with finite values") from None
    template = files(__package__).joinpath("prompts", f"{mode}_prompt.txt").read_text(encoding="utf-8")
    values = {"task": task, "prev_summary": "\n\n".join(memory), "history": history}
    # One pass only: template markers inside TASK, M or R are literal data.
    content = re.sub(r"\{\{ (task|prev_summary|history) \}\}", lambda m: values[m[1]], template)
    return [{"role": "user", "content": content}]


def _emit(sink, event):
    if sink is not None:
        try:
            sink(deepcopy(event))
        except Exception:
            pass  # Optional telemetry cannot change the result of a call.


def _persist(sink, event, event_sink):
    if sink is not None:
        try:
            sink(deepcopy(event))
        except Exception:
            _emit(
                event_sink,
                {"event": "compression_audit_error", "cause": event["event"], "error_type": "AuditWriteError"},
            )
            raise CompressionError("compression audit write failed", code="audit_write_error") from None


def _read_body(response, control):
    chunks, length = [], 0
    while True:
        control.check()
        chunk = response.read1(65536)
        if not chunk:
            control.check()
            return b"".join(chunks), getattr(response, "length", None) in (None, 0)
        length += len(chunk)
        if length > _MAX_RESPONSE_BYTES:
            raise CompressionError("compressor response exceeds size limit")
        chunks.append(chunk)
        if getattr(response, "length", None) == 0:
            # A complete HTTP body can contain usage even if the deadline
            # expires immediately afterwards. Parse that evidence first.
            return b"".join(chunks), True


def _summary(envelope):
    if not isinstance(envelope, dict):
        raise CompressionError("invalid compressor response envelope")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise CompressionError("compressor must return one choice")
    choice = choices[0]
    message = choice.get("message")
    if (
        not isinstance(message, dict)
        or message.get("tool_calls")
        or message.get("function_call")
        or message.get("refusal")
    ):
        raise CompressionError("invalid compressor message")
    content = message.get("content")
    if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
        raise CompressionError("summary incomplete or empty")
    try:
        content.encode("utf-8")
    except UnicodeError:
        raise CompressionError("summary contains invalid Unicode") from None
    return content


def compress_memory(
    *,
    task: str,
    memory: Sequence[str],
    messages: Sequence[dict[str, Any]],
    mode: CompressionMode,
    config: CompressionConfig,
    event_sink: EventSink | None = None,
    audit_sink: EventSink | None = None,
) -> CompressionResult:
    """Compress one caller-selected snapshot; block until success or failure.

    Incremental returns a delta to append to ``memory``; full returns its
    replacement. Inputs are never modified. The caller chooses the range and
    decides whether/when to apply the result. No automatic retries are made.

    ``event_sink`` receives safe started/usage telemetry, including on failure.
    ``audit_sink`` receives request and raw response evidence (no connection
    headers); if persistence fails the call fails. Calls end on completion,
    failure or timeout; there is no cancellation parameter or task registry.
    On timeout the connection is interrupted and partial output is rejected.
    A timeout does not guarantee that the provider stops work or billing.
    """
    started = time.monotonic()
    if not isinstance(config, CompressionConfig):
        raise TypeError("config must be a CompressionConfig")
    # Freeze nested config values as well as the input snapshot before callbacks.
    try:
        config = deepcopy(config)
    except RecursionError:
        raise ValueError("config exceeds supported nesting") from None
    request = {"model": config.model, "messages": _build_messages(task, memory, messages, mode), "stream": False}
    if config.reasoning is not None:
        request["reasoning"] = deepcopy(config.reasoning)
    try:
        body = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise ValueError("compression inputs must be valid UTF-8 JSON with finite values") from None
    headers = {
        k: v
        for k, v in config.headers.items()
        if k.lower() not in {"host", "content-length", "connection", "transfer-encoding", "authorization"}
    }
    headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    if config.api_key:
        headers["Authorization"] = "Bearer " + config.api_key
    control = None
    event = {
        "event": "compressor_usage",
        "source": "compressor",
        "model": config.model,
        "usage": None,
        "success": False,
    }
    error = None
    _emit(event_sink, {**event, "event": "compressor_started"})
    try:
        with _deadline(config.timeout_seconds - (time.monotonic() - started)) as control:
            _persist(
                audit_sink, {"event": "compressor_request", "compression_mode": mode, "request": request}, event_sink
            )
            control.check()
            http_error = None
            try:
                response = _open_upstream(
                    _chat_url(config.base_url),
                    body,
                    headers,
                    max(0.001, config.timeout_seconds - (time.monotonic() - started)),
                    control,
                )
            except HTTPError as exc:
                response = http_error = exc
                event["status"] = exc.code
            with response:
                response_body, body_complete = _read_body(response, control)
            try:
                envelope = _strict_json(response_body)
            except CompressionError:
                envelope = None
            if isinstance(envelope, dict):
                event["usage"] = _safe_usage(envelope.get("usage"))
                if isinstance(envelope.get("model"), str):
                    event["returned_model"] = envelope["model"]
            record = {
                "event": "compressor_response",
                "body": response_body.decode("utf-8", errors="replace"),
                "body_complete": body_complete,
            }
            if http_error is not None:
                record["status"] = http_error.code
            _persist(audit_sink, record, event_sink)
            control.check()
            if not body_complete:
                raise CompressionError("incomplete compressor HTTP response", code="transport_error")
            if http_error is not None:
                raise CompressionError("compressor upstream HTTP error", code="upstream_http_error")
            summary = _summary(envelope)
        event["success"] = True
        return CompressionResult(summary, deepcopy(event["usage"]), time.monotonic() - started)
    except CompressionError as exc:
        error = exc
        raise
    except Exception as exc:
        timeout = isinstance(exc, TimeoutError) or (isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError))
        error = CompressionError(
            "compressor timed out" if timeout else "compressor transport failed",
            code="timeout" if timeout else "transport_error",
        )
        raise error from None
    finally:
        event.update(
            timed_out=error is not None and error.code == "timeout",
            request_dispatched=control is not None and control.request_dispatched,
            latency_seconds=time.monotonic() - started,
        )
        if error is not None:
            error.usage = deepcopy(event["usage"])
            error.latency_seconds = event["latency_seconds"]
            error.request_dispatched = event["request_dispatched"]
        _emit(event_sink, event)
