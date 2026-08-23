"""In-memory implementation of the asynchronous Memory backend."""

from __future__ import annotations

import importlib
import uuid
from contextlib import nullcontext
from datetime import datetime
from typing import Any

from ...config import DefaultsConfig
from ...connections import InMemoryConnection
from ...errors import LiteExecutionError, LiteUnavailableError
from ...transport import Envelope
from ..core import MemoryRequest
from .base import AsyncMemoryBackend


def _load_schema() -> Any:
    try:
        return importlib.import_module("mindmemos_lite.service.schema")
    except (ImportError, AttributeError) as exc:
        raise LiteUnavailableError(
            "mindmemos_lite with the transport-neutral Memory service schema is required"
        ) from exc


def _load_config_context() -> Any:
    try:
        return importlib.import_module("mindmemos_lite.config")
    except (ImportError, AttributeError) as exc:
        raise LiteUnavailableError(
            "mindmemos_lite with request-scoped config overrides is required"
        ) from exc


class InMemoryMemoryBackend(AsyncMemoryBackend):
    """Map SDK Memory requests directly to a runtime Memory service."""

    _REQUIRED_METHODS = ("add", "search", "get", "update", "delete", "feedback", "dream")

    def __init__(
        self,
        connection: InMemoryConnection,
        *,
        defaults: DefaultsConfig | None = None,
    ) -> None:
        self._connection = connection
        self._defaults = defaults or DefaultsConfig()
        self._schema = _load_schema()

    @property
    def _service(self) -> Any:
        try:
            service = self._connection.runtime.memory
        except Exception as exc:
            raise LiteUnavailableError("in-memory runtime does not expose a runnable Memory service") from exc
        missing = [name for name in self._REQUIRED_METHODS if not callable(getattr(service, name, None))]
        if missing:
            raise LiteUnavailableError(
                "in-memory Memory service is missing required operations: " + ", ".join(missing)
            )
        return service

    async def execute(self, request: MemoryRequest[Any]) -> Any:
        request_id = str(uuid.uuid4())
        context = self._context(request_id, request.body)
        try:
            with self._config_scope():
                result = await self._dispatch(request.operation, context, request.body)
        except Exception as exc:
            if isinstance(exc, (LiteExecutionError, LiteUnavailableError)):
                raise
            raise LiteExecutionError(operation=f"memory.{request.operation}", message=str(exc)) from exc
        return request.parse(self._envelope(request.operation, request_id, result))

    def _config_scope(self) -> Any:
        project_config = self._connection.config.project_override_config
        if not project_config:
            return nullcontext()
        return _load_config_context().bind_config_overrides(project_config=project_config)

    def _context(self, request_id: str, body: dict[str, Any]) -> Any:
        config = self._connection.config
        defaults = self._defaults
        return self._schema.RequestContext(
            request_id=request_id,
            account_id=config.account_id,
            project_id=config.project_id,
            api_key_uuid=config.api_key_uuid,
            user_id=body.get("user_id") or defaults.user_id,
            app_id=body.get("app_id") or defaults.app_id,
            agent_id=body.get("agent_id") or defaults.agent_id,
            session_id=body.get("session_id") or defaults.session_id,
            scopes=("memory:*",),
        )

    async def _dispatch(self, operation: str, context: Any, body: dict[str, Any]) -> Any:
        service = self._service
        if operation == "add":
            return await service.add(
                context,
                self._schema.AddMemoryRequest(
                    messages=tuple(self._message(item) for item in body["messages"]),
                    mode=body.get("mode", "sync"),
                    metadata=body.get("metadata") or {},
                    skill_context=tuple(self._skill_context(item) for item in body.get("skill_context", [])),
                    score=body.get("score"),
                    task_id=body.get("task_id"),
                    task=body.get("task"),
                ),
            )
        if operation == "search":
            return await service.search(
                context,
                self._schema.SearchMemoryRequest(
                    query=body["query"],
                    filters=body.get("filters"),
                    top_k=body.get("top_k"),
                    search_strategy=body.get("search_strategy", "fast"),
                    rerank=body.get("rerank", False),
                    score_threshold=body.get("score_threshold"),
                ),
            )
        if operation == "get":
            return await service.get(
                context,
                self._schema.GetMemoryRequest(
                    filters=body.get("filters"),
                    top_k=body.get("top_k"),
                ),
            )
        if operation == "update":
            return await service.update(
                context,
                self._schema.UpdateMemoryRequest(
                    memory_id=body["memory_id"],
                    content=body["content"],
                ),
            )
        if operation == "delete":
            return await service.delete(
                context,
                self._schema.DeleteMemoryRequest(memory_id=body["memory_id"]),
            )
        if operation == "feedback":
            return await service.feedback(
                context,
                self._schema.FeedbackMemoryRequest(
                    feedback=body.get("feedback"),
                    messages=tuple(self._message(item) for item in body.get("messages", [])),
                    recalled_memories=tuple(
                        self._recalled_memory(item) for item in body.get("recalled_memories", [])
                    ),
                    mode=body.get("mode") or "sync",
                ),
            )
        if operation == "dreaming":
            return await service.dream(
                context,
                self._schema.DreamingMemoryRequest(mode=body.get("mode", "async")),
            )
        raise LiteUnavailableError(f"unsupported in-memory Memory operation: {operation}")

    def _message(self, item: dict[str, Any]) -> Any:
        if "role" in item and "content" in item:
            return self._schema.DialogueMessage(
                role=item["role"],
                content=item["content"],
                timestamp=item.get("timestamp"),
            )
        if "url" in item:
            return self._schema.UrlMessage(url=item["url"])
        if "file_name" in item and "file_path" in item:
            return self._schema.FileMessage(
                file_name=item["file_name"],
                file_path=item["file_path"],
                file_type=item.get("file_type", ""),
            )
        if "text" in item:
            return self._schema.TextMessage(text=item["text"])
        raise TypeError(f"unsupported SDK message shape: {sorted(item)}")

    def _skill_context(self, item: dict[str, Any]) -> Any:
        return self._schema.SkillContext(
            name=item["name"],
            content_hash=item["content_hash"],
            base_version_id=item.get("base_version_id") or item.get("version_id", ""),
            version_label=item.get("version_label"),
            usage=item.get("usage"),
        )

    def _recalled_memory(self, item: dict[str, Any]) -> Any:
        lineage_payload = item.get("lineage")
        lineage = None
        if lineage_payload:
            lineage = self._schema.MemoryLineage(
                role=lineage_payload.get("role", "current"),
                derived_from_memory_ids=tuple(lineage_payload.get("derived_from_memory_ids", [])),
                derived_to_memory_ids=tuple(lineage_payload.get("derived_to_memory_ids", [])),
            )
        return self._schema.MemoryItem(
            memory_id=item.get("memory_id") or item["id"],
            content=item.get("content") or item["memory"],
            memory_type=item.get("memory_type", "fact"),
            updated_at=_parse_datetime(item.get("last_update_at")),
            event_time=_parse_datetime(item.get("event_time")),
            source_timestamp=_parse_datetime(item.get("source_timestamp")),
            lineage=lineage,
        )

    def _envelope(self, operation: str, request_id: str, result: Any) -> Envelope:
        code = result.status
        message = getattr(result, "message", None) or ""
        if operation == "add":
            data = {
                "memories": [
                    {
                        "operation": item.operation,
                        "content": item.content,
                        "memory_id": item.memory_id,
                        "mem_type": item.memory_type,
                        "confidence": item.confidence,
                        "related_memory_ids": list(item.related_memory_ids),
                        "graph_edge_count": item.graph_edge_count,
                    }
                    for item in result.memories
                ]
            }
        elif operation in {"search", "get"}:
            data = {"memories": [self._memory_item(item) for item in result.memories]}
        else:
            data = None
        return Envelope(code=code, message=message, request_id=request_id, data=data)

    @staticmethod
    def _memory_item(item: Any) -> dict[str, Any]:
        lineage = item.lineage
        return {
            "id": item.memory_id,
            "memory": item.content,
            "memory_type": item.memory_type,
            "last_update_at": _format_datetime(item.updated_at),
            "event_time": _format_datetime(item.event_time),
            "source_timestamp": _format_datetime(item.source_timestamp),
            "lineage": (
                {
                    "role": lineage.role,
                    "derived_from_memory_ids": list(lineage.derived_from_memory_ids),
                    "derived_to_memory_ids": list(lineage.derived_to_memory_ids),
                }
                if lineage is not None
                else None
            ),
        }


def _format_datetime(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(normalized)


__all__ = ["InMemoryMemoryBackend"]
