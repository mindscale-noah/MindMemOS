"""Persistent prompt, stream, result, and retry traces for agent calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.invocation import InvokeError, InvokeResult
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.utils.files import write_json_file, write_text_file


def safe_trace_component(value: str, *, fallback: str = "invoke") -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_")
    if len(sanitized) > 120:
        sanitized = sanitized[:120].rstrip("_")
    return sanitized or fallback


def trace_output_path(trace_name: str, *, logs_root: str | Path | None = None) -> Path:
    """Resolve one trace's file path under a scope's ``logs/`` tree.

    ``logs_root`` wins when given (the call site knows its scope);
    otherwise the bound default cwd's ``logs/`` subdir is used.
    """
    if logs_root is not None:
        base = Path(logs_root)
    else:
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.backend import get_default_cwd

        cwd = get_default_cwd()
        if cwd is None:
            raise InvokeError(
                "agent_backend",
                "trace path requested but no default cwd is bound; call use_backend(default_cwd=...) before invoking",
                retryable=False,
            )
        base = Path(cwd) / "logs"
    parts = [safe_trace_component(part) for part in str(trace_name).replace("\\", "/").split("/") if part.strip()]
    return base.joinpath(*(parts or ["invoke"])).with_suffix(".json")


def trace_result_store_path(trace_name: str, *, logs_root: str | Path | None = None) -> Path:
    """Where one call archives tool results too large to keep inline.

    Beside that call's own trace under ``logs/``, never in cwd (the agent's
    workspace). ``logs_root`` wins when given.
    """
    return trace_output_path(trace_name, logs_root=logs_root).with_suffix(".tool-results")


def _attempt_sidecar_path(path: Path, attempt: int, suffix: str) -> Path:
    if attempt <= 1:
        return path.with_suffix(suffix)
    return path.with_name(f"{path.stem}.attempt_{attempt}{suffix}")


def _trajectory_path(path: Path) -> Path:
    """Return the human-facing event stream next to invoke metadata."""
    if path.stem == "invoke":
        return path.with_name("trajectory.jsonl")
    return path.with_name(f"{path.stem}.trajectory.jsonl")


def _text_line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def _jsonable_arg(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 200:
        return {"length": len(value), "truncated": True}
    if isinstance(value, (list, tuple)):
        return [_jsonable_arg(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_arg(item) for key, item in value.items()}
    return _jsonable(value)


def _write_stream_file(path: Path, content: Any) -> dict | None:
    if content is None:
        return None
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace")
    else:
        text = content if isinstance(content, str) else str(content)
    write_text_file(path, text)
    return {
        "path": str(path),
        "bytes": len(text.encode("utf-8")),
        "lines": _text_line_count(text),
    }


def _write_prompt_file(
    path: Path,
    system: str,
    user: str,
    *,
    attempt: int,
) -> dict:
    content = "===== SYSTEM =====\n" + system + "\n\n===== USER =====\n" + user + "\n"
    prompt_path = _attempt_sidecar_path(path, attempt, ".prompt.txt")
    write_text_file(prompt_path, content)
    return {
        "path": str(prompt_path),
        "bytes": len(content.encode("utf-8")),
        "lines": _text_line_count(content),
    }


def _write_trajectory_file(
    path: Path,
    *,
    backend: str,
    system: str,
    user: str,
    invocation: dict[str, Any],
    result: InvokeResult | None,
    error: Exception | None,
    attempt: int,
    duration_sec: float,
) -> dict | None:
    """Write one flat JSONL execution trajectory: prompt, turns, tool
    calls/results, and the attempt outcome, readable in order."""
    transcript = result.transcript if result is not None else []
    if not transcript:
        return None
    trajectory_path = _trajectory_path(path)
    records: list[dict[str, Any]] = [
        {
            "type": "prompt",
            "attempt": attempt,
            "backend": backend,
            "system": system,
            "user": user,
            "invocation": invocation,
        }
    ]
    for event in transcript:
        payload = _jsonable(event)
        if isinstance(payload, dict):
            records.append({"attempt": attempt, **payload})
        else:
            records.append(
                {
                    "type": "event",
                    "attempt": attempt,
                    "value": payload,
                }
            )
    records.append(
        {
            "type": "attempt_result",
            "attempt": attempt,
            "status": "error" if error is not None else "ok",
            "duration_sec": round(duration_sec, 3),
            "result": _latest_result_summary(result),
            "error": _error_payload(error),
        }
    )
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    if attempt > 1 and trajectory_path.is_file():
        text = trajectory_path.read_text(encoding="utf-8") + text
    write_text_file(trajectory_path, text)
    return {
        "path": str(trajectory_path),
        "bytes": len(text.encode("utf-8")),
        "lines": _text_line_count(text),
    }


def _write_result_streams(
    path: Path,
    result: InvokeResult | None,
    *,
    attempt: int,
) -> dict:
    if result is None or result.raw is None:
        return {}
    raw = result.raw
    streams: dict[str, Any] = {}
    if hasattr(raw, "stdout"):
        stdout_ref = _write_stream_file(
            _attempt_sidecar_path(path, attempt, ".stdout.jsonl"),
            getattr(raw, "stdout", None),
        )
        if stdout_ref is not None:
            streams["stdout"] = stdout_ref
    if hasattr(raw, "stderr"):
        stderr_ref = _write_stream_file(
            _attempt_sidecar_path(path, attempt, ".stderr.txt"),
            getattr(raw, "stderr", None),
        )
        if stderr_ref is not None:
            streams["stderr"] = stderr_ref
    return streams


def _raw_process_payload(raw: Any) -> dict:
    if raw is None:
        return {}
    payload = {"type": type(raw).__name__}
    if hasattr(raw, "returncode"):
        payload["returncode"] = getattr(raw, "returncode", None)
    if hasattr(raw, "args"):
        payload["args"] = _jsonable_arg(getattr(raw, "args", None))
    return payload


def _invocation_payload(kwargs: dict[str, Any]) -> dict:
    payload = {
        "cwd": _jsonable(kwargs.get("cwd")),
        "tools": _jsonable(kwargs.get("tools") or []),
        "permission_mode": _jsonable(kwargs.get("permission_mode")),
        "read_roots": _jsonable(kwargs.get("read_roots") or []),
        "write_roots": _jsonable(kwargs.get("write_roots") or []),
        "session_id": _jsonable(kwargs.get("session_id")),
    }
    return payload


def _invoke_result_payload(
    result: InvokeResult | None,
    *,
    streams: dict | None = None,
) -> dict:
    if result is None:
        return {}
    payload = {
        "channel": result.channel,
        "attempts": result.attempts,
        "input_tokens": result.input_tokens,
        "cache_creation_input_tokens": result.cache_creation_input_tokens,
        "cache_read_input_tokens": result.cache_read_input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "text_length": len(result.text or ""),
        "permission_denials": _jsonable(result.permission_denials),
        "session_id": result.session_id,
        "finish_reason": result.finish_reason,
        "provider_request_id": result.provider_request_id,
        "raw": _raw_process_payload(result.raw),
    }
    if streams:
        payload["streams"] = streams
    return payload


def _latest_result_summary(result: InvokeResult | None) -> dict:
    """Keep top-level usage/status without duplicating the full transcript."""
    if result is None:
        return {}
    return {
        "channel": result.channel,
        "attempts": result.attempts,
        "input_tokens": result.input_tokens,
        "cache_creation_input_tokens": result.cache_creation_input_tokens,
        "cache_read_input_tokens": result.cache_read_input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "finish_reason": result.finish_reason,
        "provider_request_id": result.provider_request_id,
    }


def _error_payload(error: Exception | None) -> dict:
    if error is None:
        return {}
    payload = {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, InvokeError):
        payload.update(
            {
                "channel": error.channel,
                "retryable": error.retryable,
                "resume_session": error.resume_session,
                "attempts": error.attempts,
                "session_id": error.session_id,
            }
        )
    return payload


def _usage_totals(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum usage across every recorded attempt in one logical invocation."""
    results = [
        item.get("result", {}) for item in attempts if isinstance(item, dict) and isinstance(item.get("result"), dict)
    ]
    costs = [result.get("cost_usd") for result in results if isinstance(result.get("cost_usd"), (int, float))]
    return {
        "input_tokens": sum(int(result.get("input_tokens") or 0) for result in results),
        "cache_creation_input_tokens": sum(int(result.get("cache_creation_input_tokens") or 0) for result in results),
        "cache_read_input_tokens": sum(int(result.get("cache_read_input_tokens") or 0) for result in results),
        "output_tokens": sum(int(result.get("output_tokens") or 0) for result in results),
        "cost_usd": round(sum(float(cost) for cost in costs), 12) if costs else None,
    }


def write_invoke_trace(
    *,
    trace_name: str | None,
    backend: str,
    system: str,
    user: str,
    kwargs: dict[str, Any],
    result: InvokeResult | None,
    error: Exception | None,
    attempt: int,
    max_attempts: int,
    duration_sec: float,
    logs_root: str | Path | None = None,
) -> None:
    if not trace_name:
        return
    path = trace_output_path(trace_name, logs_root=logs_root)
    prompt = _write_prompt_file(path, system, user, attempt=attempt)
    streams = _write_result_streams(path, result, attempt=attempt)
    invocation = _invocation_payload(kwargs)
    trajectory = _write_trajectory_file(
        path,
        backend=backend,
        system=system,
        user=user,
        invocation=invocation,
        result=result,
        error=error,
        attempt=attempt,
        duration_sec=duration_sec,
    )
    attempt_payload = {
        "status": "error" if error is not None else "ok",
        "attempt": attempt,
        "duration_sec": round(duration_sec, 3),
        "invocation": invocation,
        "prompt": prompt,
        "trajectory": trajectory,
        "result": _invoke_result_payload(result, streams=streams),
        "error": _error_payload(error),
    }
    attempts = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            prior_attempts = existing.get("attempts")
            if isinstance(prior_attempts, list):
                attempts.extend(prior_attempts)
        except Exception:
            attempts = []
    attempts.append(attempt_payload)
    write_json_file(
        str(path),
        {
            "trace_name": trace_name,
            "backend": backend,
            "status": "error" if error is not None else "ok",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "duration_sec": round(sum(float(item.get("duration_sec") or 0) for item in attempts), 3),
            "result": _latest_result_summary(result),
            "usage_totals": _usage_totals(attempts),
            "trajectory": trajectory,
            "attempts": attempts,
        },
    )


__all__ = ["trace_output_path", "write_invoke_trace"]
