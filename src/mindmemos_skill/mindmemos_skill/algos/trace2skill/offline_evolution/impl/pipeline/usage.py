"""Backend-agnostic aggregation of persisted LLM invocation traces."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

_TOKEN_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def _number(value: Any) -> int | float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _empty_usage() -> dict[str, Any]:
    return {
        "logical_calls": 0,
        "attempts": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
        "accounted_tokens": 0,
        "recorded_cost_usd": 0.0,
        "attempts_without_usage": 0,
        "attempts_without_cost": 0,
        "tokens_complete": True,
        "cost_complete": True,
        "usage_complete": True,
        "failures": {},
    }


def _failure_kind(error: Any) -> str:
    data = error if isinstance(error, dict) else {}
    message = str(data.get("message") or error or "").lower()
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if "permission" in message:
        return "permission"
    if "schema validation" in message or "structured output" in message:
        return "schema"
    if "semantic validation" in message:
        return "semantic"
    if "argument list too long" in message or "subprocess failed" in message:
        return "transport"
    if "parse" in message or "json" in message:
        return "parse"
    return "other"


def _attempt_may_have_unreported_usage(attempt: dict[str, Any]) -> bool:
    if attempt.get("status") == "ok":
        return True
    error = attempt.get("error") if isinstance(attempt.get("error"), dict) else {}
    return bool(error.get("session_id")) or _failure_kind(error) == "timeout"


def _has_attempt_usage(attempt: dict[str, Any]) -> bool:
    result = attempt.get("result") if isinstance(attempt.get("result"), dict) else {}
    return any(_number(result.get(field)) > 0 for field in _TOKEN_FIELDS) or (
        isinstance(result.get("cost_usd"), (int, float))
    )


def _add_trace(target: dict[str, Any], trace: dict[str, Any]) -> None:
    attempts = trace.get("attempts") if isinstance(trace.get("attempts"), list) else []
    target["logical_calls"] += 1
    target["attempts"] += len(attempts)
    if trace.get("status") == "ok":
        target["successful_calls"] += 1
    else:
        target["failed_calls"] += 1

    totals = trace.get("usage_totals")
    if not isinstance(totals, dict):
        totals = {}
    for field in _TOKEN_FIELDS:
        target[field] += int(_number(totals.get(field)))
    target["recorded_cost_usd"] += float(_number(totals.get("cost_usd")))

    failures = Counter(target.get("failures") or {})
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        failure_kind = ""
        if attempt.get("status") != "ok":
            failure_kind = _failure_kind(attempt.get("error"))
            failures[failure_kind] += 1
        if failure_kind == "timeout" or (
            _attempt_may_have_unreported_usage(attempt) and not _has_attempt_usage(attempt)
        ):
            target["attempts_without_usage"] += 1
        result = attempt.get("result") if isinstance(attempt.get("result"), dict) else {}
        if failure_kind == "timeout" or (
            _attempt_may_have_unreported_usage(attempt) and not isinstance(result.get("cost_usd"), (int, float))
        ):
            target["attempts_without_cost"] += 1
    target["failures"] = dict(sorted(failures.items()))


def _finish_usage(usage: dict[str, Any]) -> dict[str, Any]:
    usage["accounted_tokens"] = sum(int(usage[field]) for field in _TOKEN_FIELDS)
    usage["recorded_cost_usd"] = round(float(usage["recorded_cost_usd"]), 6)
    usage["tokens_complete"] = usage["attempts_without_usage"] == 0
    usage["cost_complete"] = usage["attempts_without_cost"] == 0
    usage["usage_complete"] = usage["tokens_complete"] and usage["cost_complete"]
    return usage


def collect_llm_usage(workspace_dir: str | Path) -> dict[str, Any]:
    """Aggregate trace JSON files under each scope's ``logs/`` tree."""
    root = Path(workspace_dir)
    total = _empty_usage()
    by_stage: dict[str, dict[str, Any]] = {}
    if root.is_dir():
        stage_by_path: dict[Path, str] = {}
        for logs_root in root.glob("*/logs"):
            for path in logs_root.rglob("*.json"):
                stage_by_path[path] = path.relative_to(logs_root).parts[0]
        for path in sorted(stage_by_path):
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(trace, dict) or not {"trace_name", "backend", "attempts"}.issubset(trace):
                continue
            stage = stage_by_path[path]
            stage_usage = by_stage.setdefault(stage, _empty_usage())
            _add_trace(stage_usage, trace)
            _add_trace(total, trace)

    finished_stages = {stage: _finish_usage(usage) for stage, usage in sorted(by_stage.items())}
    result = _finish_usage(total)
    result["by_stage"] = finished_stages
    return result


__all__ = ["collect_llm_usage"]
