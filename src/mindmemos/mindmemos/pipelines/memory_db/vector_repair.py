"""Idempotent dense-vector repair for preserved memory records."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from typing import Any

from pydantic import BaseModel, Field, model_validator

from ...components.text.vector_repair_state import embedding_model_fingerprint
from ...config import get_config
from ...errors import ConfigNotInitializedError
from ...infra.db import QdrantRecord, build_filter, is_empty, match_value, number_range
from ...logging import get_logger
from ...provider_bindings import provider_config_context
from ...typing import MemoryDbMemoryUpdateCommand, MemoryRequestContext
from .reader import MemoryDbReader
from .writer import MemoryDbWriter

logger = get_logger(__name__)


class VectorRepairPolicy(BaseModel):
    """Safety bounds; per-memory progress remains durable in Qdrant."""

    max_attempts: int = Field(default=8, ge=1, le=100)
    base_backoff_seconds: int = Field(default=30, ge=1, le=86_400)
    max_backoff_seconds: int = Field(default=3_600, ge=1, le=604_800)


class VectorRepairRequest(BaseModel):
    """One bounded automatic or explicitly selected repair request."""

    limit: int = Field(default=50, ge=1, le=100)
    memory_ids: list[str] = Field(default_factory=list, max_length=100)
    force: bool = False

    @model_validator(mode="after")
    def validate_explicit_selection(self) -> "VectorRepairRequest":
        self.memory_ids = list(dict.fromkeys(memory_id.strip() for memory_id in self.memory_ids if memory_id.strip()))
        if len(self.memory_ids) > self.limit:
            raise ValueError("memory_ids must not exceed limit")
        if self.force and not self.memory_ids:
            raise ValueError("force requires explicit memory_ids")
        return self


class VectorRepairItem(BaseModel):
    memory_id: str
    status: str
    code: str | None = None


class VectorRepairResult(BaseModel):
    selected: int = 0
    repaired: int = 0
    failed: int = 0
    skipped: int = 0
    items: list[VectorRepairItem] = Field(default_factory=list)


class VectorRepairStatus(BaseModel):
    pending: int = 0
    due: int = 0
    exhausted: int = 0


ProviderContextFactory = Callable[[MemoryRequestContext], Awaitable[AbstractContextManager[Any]]]


class VectorRepairService:
    """Re-embed stored content without invoking extraction or graph inference."""

    def __init__(
        self,
        *,
        reader: MemoryDbReader | None = None,
        writer: MemoryDbWriter | None = None,
        provider_context_factory: ProviderContextFactory = provider_config_context,
        policy: VectorRepairPolicy | None = None,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._reader = reader or MemoryDbReader()
        self._writer = writer or MemoryDbWriter()
        self._provider_context_factory = provider_context_factory
        self._policy = policy or VectorRepairPolicy()
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    async def repair(self, trigger_ctx: MemoryRequestContext, request: VectorRepairRequest) -> VectorRepairResult:
        now_ms = self._now_ms()
        records = await self._select(trigger_ctx, request, now_ms=now_ms)
        result = VectorRepairResult(selected=len(records))
        for record in records:
            eligibility = self._eligibility(record, request=request, now_ms=now_ms)
            if eligibility is not None:
                result.skipped += 1
                result.items.append(VectorRepairItem(memory_id=record.point_id, status="skipped", code=eligibility))
                continue
            try:
                await self._repair_one(trigger_ctx, record, now_ms=now_ms)
            except Exception as exc:
                code = _safe_error_code(exc)
                await self._record_failure(trigger_ctx, record, code=code, now_ms=now_ms)
                result.failed += 1
                result.items.append(VectorRepairItem(memory_id=record.point_id, status="failed", code=code))
                logger.warning(
                    "memory_vector_repair_failed",
                    project_id=trigger_ctx.project_id,
                    memory_id=record.point_id,
                    error_code=code,
                )
            else:
                result.repaired += 1
                result.items.append(VectorRepairItem(memory_id=record.point_id, status="repaired"))
        return result

    async def status(self, ctx: MemoryRequestContext) -> VectorRepairStatus:
        now_ms = self._now_ms()
        pending = await self._reader.count_memory_records(ctx, filters=_pending_filter())
        due = await self._reader.count_memory_records(ctx, filters=_due_filter(now_ms, self._policy.max_attempts))
        exhausted = await self._reader.count_memory_records(ctx, filters=_exhausted_filter(self._policy.max_attempts))
        return VectorRepairStatus(pending=pending, due=due, exhausted=exhausted)

    async def _select(
        self,
        ctx: MemoryRequestContext,
        request: VectorRepairRequest,
        *,
        now_ms: int,
    ) -> list[QdrantRecord]:
        if request.memory_ids:
            records = [await self._reader.get_memory_record(ctx, memory_id) for memory_id in request.memory_ids]
            return [record for record in records if record is not None]
        records, _ = await self._reader.list_memory_records(
            ctx,
            filters=_due_filter(now_ms, self._policy.max_attempts),
            limit=request.limit,
        )
        return records

    def _eligibility(
        self,
        record: QdrantRecord,
        *,
        request: VectorRepairRequest,
        now_ms: int,
    ) -> str | None:
        metadata = dict(record.payload.get("metadata") or {})
        if request.memory_ids and request.force:
            return None
        if not bool(metadata.get("vector_pending")):
            return "vector_repair.not_pending"
        if int(metadata.get("vector_retry_count") or 0) >= self._policy.max_attempts:
            return "vector_repair.attempts_exhausted"
        if int(metadata.get("vector_next_retry_at_ms") or 0) > now_ms:
            return "vector_repair.not_due"
        return None

    async def _repair_one(
        self,
        trigger_ctx: MemoryRequestContext,
        record: QdrantRecord,
        *,
        now_ms: int,
    ) -> None:
        item_ctx = _context_from_record(trigger_ctx, record)
        config_context = await self._provider_context_factory(item_ctx)
        with config_context:
            existing_metadata = dict(record.payload.get("metadata") or {})
            expected_dimension, model_fingerprint = _configured_embedding_identity(
                fallback_dimension=existing_metadata.get("vector_expected_dimension"),
                fallback_fingerprint=existing_metadata.get("vector_model_fingerprint"),
            )
            existing_metadata["vector_expected_dimension"] = expected_dimension
            existing_metadata["vector_model_fingerprint"] = model_fingerprint
            record.payload["metadata"] = existing_metadata
            mutation = await self._writer.update_memory(
                item_ctx,
                MemoryDbMemoryUpdateCommand(
                    memory_id=record.point_id,
                    refresh_vectors=True,
                    touch_update_at=False,
                    metadata_patch={
                        "vector_pending": False,
                        "vector_expected_dimension": expected_dimension,
                        "vector_retry_count": 0,
                        "vector_next_retry_at_ms": 0,
                        "vector_last_error_code": None,
                        "vector_model_fingerprint": model_fingerprint,
                        "vector_repaired_at": now_ms,
                    },
                    consistency="strong",
                ),
            )
            if not mutation.changed:
                raise RuntimeError("memory disappeared during vector repair")

    async def _record_failure(
        self,
        trigger_ctx: MemoryRequestContext,
        record: QdrantRecord,
        *,
        code: str,
        now_ms: int,
    ) -> None:
        metadata = dict(record.payload.get("metadata") or {})
        retry_count = int(metadata.get("vector_retry_count") or 0) + 1
        delay_seconds = min(
            self._policy.max_backoff_seconds,
            self._policy.base_backoff_seconds * (2 ** max(0, retry_count - 1)),
        )
        await self._writer.update_memory(
            _context_from_record(trigger_ctx, record),
            MemoryDbMemoryUpdateCommand(
                memory_id=record.point_id,
                touch_update_at=False,
                metadata_patch={
                    "vector_pending": True,
                    "vector_expected_dimension": metadata.get("vector_expected_dimension"),
                    "vector_retry_count": retry_count,
                    "vector_next_retry_at_ms": now_ms + delay_seconds * 1000,
                    "vector_last_error_code": code,
                    "vector_model_fingerprint": metadata.get("vector_model_fingerprint"),
                },
                consistency="strong",
            ),
        )


def _context_from_record(trigger_ctx: MemoryRequestContext, record: QdrantRecord) -> MemoryRequestContext:
    payload = record.payload
    return MemoryRequestContext(
        request_id=trigger_ctx.request_id,
        account_id=str(payload.get("account_id") or trigger_ctx.account_id),
        project_id=trigger_ctx.project_id,
        api_key_uuid=str(payload.get("api_key_uuid") or trigger_ctx.api_key_uuid),
        memory_algorithm=payload.get("memory_algorithm") or trigger_ctx.memory_algorithm,
        user_id=payload.get("user_id"),
        app_id=payload.get("app_id"),
        session_id=payload.get("session_id"),
        agent_id=payload.get("agent_id"),
        scopes=list(trigger_ctx.scopes),
    )


def _pending_filter():
    return build_filter(must=[match_value("metadata.vector_pending", True)])


def _due_filter(now_ms: int, max_attempts: int):
    return build_filter(
        must=[
            match_value("metadata.vector_pending", True),
            build_filter(
                should=[
                    is_empty("metadata.vector_retry_count"),
                    number_range("metadata.vector_retry_count", lt=max_attempts),
                ]
            ),
            build_filter(
                should=[
                    is_empty("metadata.vector_next_retry_at_ms"),
                    number_range("metadata.vector_next_retry_at_ms", lte=now_ms),
                ]
            ),
        ]
    )


def _exhausted_filter(max_attempts: int):
    return build_filter(
        must=[
            match_value("metadata.vector_pending", True),
            number_range("metadata.vector_retry_count", gte=max_attempts),
        ]
    )


def _configured_embedding_identity(
    *,
    fallback_dimension: Any = None,
    fallback_fingerprint: Any = None,
) -> tuple[int | None, str]:
    try:
        endpoints = get_config().embed_model_router.endpoints
    except ConfigNotInitializedError:
        dimension = fallback_dimension if isinstance(fallback_dimension, int) and fallback_dimension > 0 else None
        fingerprint = (
            fallback_fingerprint
            if isinstance(fallback_fingerprint, str) and fallback_fingerprint
            else embedding_model_fingerprint(expected_dimension=dimension)
        )
        return dimension, fingerprint
    dimensions = {endpoint.dimensions for endpoint in endpoints if endpoint.dimensions is not None}
    dimension = next(iter(dimensions)) if len(dimensions) == 1 else None
    cfg = get_config()
    if dimension is None and not cfg.provider_binding.enabled:
        configured_size = cfg.database.qdrant.vector_size
        dimension = configured_size if isinstance(configured_size, int) and configured_size > 0 else None
    fingerprint = embedding_model_fingerprint(
        expected_dimension=dimension,
        models=[
            {"model": endpoint.model, "transport": endpoint.transport, "dimensions": endpoint.dimensions}
            for endpoint in endpoints
        ],
    )
    return dimension, fingerprint


def _safe_error_code(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if code in {
        "embedding.dimension_mismatch",
        "provider_binding.model_endpoint_missing",
        "provider_binding.not_found",
    }:
        return str(code)
    return "embedding.provider_unavailable"
