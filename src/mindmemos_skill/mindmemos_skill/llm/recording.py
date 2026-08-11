"""Run-scoped persistence for complete LLM request and response payloads."""

from __future__ import annotations

import logging
import math
import re
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Protocol, cast

from pydantic import BaseModel, JsonValue, SecretStr

from ..infra.database import ScopedDatabase
from ..persistence.models import LLMCallRecord
from ..persistence.records import to_database_record
from ..persistence.tables import LLM_CALL_TABLE

logger = logging.getLogger(__name__)

_RUN_ID: ContextVar[str | None] = ContextVar("mindmemos_skill_llm_run_id", default=None)
_SECRET_KEY = re.compile(
    r"(?:authorization|cookie|api[_-]?key|access[_-]?token|refresh[_-]?token|password|credential|secret)",
    re.IGNORECASE,
)


class LLMCallSink(Protocol):
    """Persistence target for one completed or failed logical LLM call."""

    async def write(self, record: LLMCallRecord) -> None: ...


class DatabaseLLMCallSink:
    """Store LLM calls in the canonical Skill database."""

    def __init__(self, database: ScopedDatabase) -> None:
        self._database = database

    async def write(self, record: LLMCallRecord) -> None:
        await self._database.upsert_records(LLM_CALL_TABLE, [to_database_record(record)])


@contextmanager
def llm_run_context(run_id: str) -> Iterator[None]:
    """Associate all nested asynchronous LLM calls with one algorithm run."""

    if not run_id:
        raise ValueError("LLM run_id must not be empty")
    token = _RUN_ID.set(run_id)
    try:
        yield
    finally:
        _RUN_ID.reset(token)


def current_llm_run_id() -> str | None:
    """Return the run identifier visible to the current asynchronous task."""

    return _RUN_ID.get()


async def write_llm_call(
    sink: LLMCallSink | None,
    *,
    call_type: str,
    task: str,
    request: Mapping[str, Any],
    response: Mapping[str, Any] | None,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    status: str,
    error: str | None,
    started_at: datetime,
    finished_at: datetime,
    latency_ms: float,
) -> None:
    """Write one call without allowing observability failures to mask model behavior."""

    run_id = current_llm_run_id()
    if sink is None or run_id is None:
        return
    record = LLMCallRecord(
        call_id=str(uuid.uuid4()),
        run_id=run_id,
        task=task,
        call_type=call_type,
        request=json_mapping(request),
        response=json_mapping(response) if response is not None else None,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        status=status,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        latency_ms=latency_ms,
    )
    try:
        await sink.write(record)
    except Exception:
        logger.exception("Failed to persist LLM call run_id=%s task=%s call_id=%s", run_id, task, record.call_id)


def json_mapping(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Return a detached JSON-compatible payload with transport secrets redacted."""

    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("LLM audit payload must be a mapping")
    return cast(dict[str, JsonValue], normalized)


def _json_value(value: Any, *, field_name: str | None = None) -> JsonValue:
    if isinstance(value, SecretStr):
        return "<redacted>"
    if field_name is not None and _SECRET_KEY.search(field_name):
        return "<redacted>"
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"), field_name=field_name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


__all__ = [
    "DatabaseLLMCallSink",
    "LLMCallSink",
    "current_llm_run_id",
    "json_mapping",
    "llm_run_context",
    "write_llm_call",
]
