"""Dynamic provider binding rules and resolution."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .config import get_config
from .errors import BadRequestError
from .infra.db import ProviderBindingPoint, QdrantRecord, resolve_database_clients
from .logging import get_logger
from .typing import MemoryRequestContext

logger = get_logger(__name__)

SCOPE_FIELDS = ("user_id", "app_id", "session_id", "agent_id")
IMMUTABLE_EMBEDDING_ENDPOINT_FIELDS = ("model", "dimensions")


class ProviderBindingScope(BaseModel):
    """Optional actor scope fields attached to one provider binding."""

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None

    def specificity(self) -> int:
        """Return how many scope fields are constrained by this binding."""

        return sum(1 for field in SCOPE_FIELDS if getattr(self, field))

    def matches(self, ctx: MemoryRequestContext) -> bool:
        """Whether all non-empty binding scope fields match the request context."""

        for field in SCOPE_FIELDS:
            expected = getattr(self, field)
            if expected and getattr(ctx, field) != expected:
                return False
        return True

    def compact_dict(self) -> dict[str, str]:
        """Return only non-empty scope fields for logs and payloads."""

        return {field: value for field in SCOPE_FIELDS if (value := getattr(self, field))}


class ProviderBindingRecord(BaseModel):
    """Stored provider binding payload after hydration from storage."""

    model_config = ConfigDict(extra="forbid")

    binding_id: str
    project_id: str
    scope: ProviderBindingScope = Field(default_factory=ProviderBindingScope)
    routers: dict[str, Any]
    enabled: bool = True


class ProviderBindingStore(Protocol):
    """Storage contract used by the runtime resolver."""

    async def list_project_bindings(self, project_id: str) -> list[ProviderBindingRecord]: ...


class QdrantProviderBindingStore:
    """Qdrant-backed dynamic provider binding store."""

    def __init__(self, clients: Any | None = None) -> None:
        self._clients = resolve_database_clients(clients)

    async def upsert(self, record: ProviderBindingRecord) -> ProviderBindingRecord:
        now = datetime.now(UTC)
        payload = {
            "binding_id": record.binding_id,
            "project_id": record.project_id,
            **record.scope.compact_dict(),
            "scope": record.scope.compact_dict(),
            "routers": record.routers,
            "enabled": record.enabled,
            "config_hash": _config_hash(record.routers),
            "updated_at": now,
        }
        existing = await self._clients.qdrant.get_provider_binding(record.project_id, record.binding_id)
        payload["created_at"] = existing.payload.get("created_at") if existing is not None else now
        await self._clients.qdrant.upsert_provider_binding(
            ProviderBindingPoint(binding_id=record.binding_id, payload=payload)
        )
        return record

    async def get(self, project_id: str, binding_id: str) -> ProviderBindingRecord | None:
        record = await self._clients.qdrant.get_provider_binding(project_id, binding_id)
        return _record_from_qdrant(record) if record is not None else None

    async def list_project_bindings(self, project_id: str) -> list[ProviderBindingRecord]:
        records: list[ProviderBindingRecord] = []
        cursor = None
        while True:
            page, cursor = await self._clients.qdrant.scroll_provider_bindings(
                project_id,
                limit=100,
                cursor=cursor,
            )
            records.extend(_record_from_qdrant(record) for record in page)
            if cursor is None:
                return records


class ProviderBindingService:
    """Management facade for dynamic provider bindings."""

    def __init__(self, *, store: QdrantProviderBindingStore | None = None, enabled: bool | None = None) -> None:
        self._store = store or QdrantProviderBindingStore()
        self._enabled = get_config().provider_binding.enabled if enabled is None else enabled

    async def create_binding(
        self,
        *,
        project_id: str,
        scope: dict[str, Any],
        routers: dict[str, Any],
        request_id: str | None,
    ) -> dict[str, Any]:
        self._ensure_enabled()
        del request_id
        binding_scope = ProviderBindingScope.model_validate(scope or {})
        record = ProviderBindingRecord(
            binding_id=provider_binding_id(project_id, binding_scope),
            project_id=project_id,
            scope=binding_scope,
            routers=deepcopy(routers),
        )
        stored = await self._store.upsert(record)
        return stored.model_dump(mode="json")

    async def patch_binding(
        self,
        *,
        project_id: str,
        binding_id: str,
        routers: dict[str, Any],
        request_id: str | None,
    ) -> dict[str, Any]:
        self._ensure_enabled()
        existing = await self._store.get(project_id, binding_id)
        if existing is None:
            raise BadRequestError("provider binding not found", code="provider_binding.not_found", status_code=404)
        merged = _deep_merge(existing.routers, routers)
        validate_provider_binding_patch(
            existing.routers,
            merged,
            project_id=project_id,
            binding_id=binding_id,
            scope=existing.scope,
            request_id=request_id,
        )
        stored = await self._store.upsert(existing.model_copy(update={"routers": merged}))
        return stored.model_dump(mode="json")

    async def list_bindings(self, *, project_id: str) -> list[dict[str, Any]]:
        self._ensure_enabled()
        records = await self._store.list_project_bindings(project_id)
        return [record.model_dump(mode="json") for record in records]

    def _ensure_enabled(self) -> None:
        if not self._enabled:
            raise BadRequestError("provider binding is disabled", code="provider_binding.disabled")


class ProviderBindingResolver:
    """Resolve the best dynamic provider config override for one request."""

    def __init__(self, *, store: ProviderBindingStore, enabled: bool) -> None:
        self._store = store
        self._enabled = enabled

    async def resolve(self, ctx: MemoryRequestContext) -> dict[str, Any] | None:
        """Return the most specific matching router config, or None."""

        if not self._enabled:
            return None
        records = await self._store.list_project_bindings(ctx.project_id)
        candidates = [
            record
            for record in records
            if record.enabled and record.project_id == ctx.project_id and record.scope.matches(ctx)
        ]
        if not candidates:
            return None
        selected = max(candidates, key=lambda record: (record.scope.specificity(), record.binding_id))
        return selected.routers


def validate_provider_binding_patch(
    old_routers: dict[str, Any],
    new_routers: dict[str, Any],
    *,
    project_id: str,
    binding_id: str,
    scope: ProviderBindingScope,
    request_id: str | None,
) -> None:
    """Reject provider updates that would invalidate stored embedding vectors."""

    blocked_fields = _changed_embedding_identity_fields(old_routers, new_routers)
    if not blocked_fields:
        return
    logger.warning(
        "provider_binding_immutable_update_blocked",
        project_id=project_id,
        binding_id=binding_id,
        scope=scope.compact_dict(),
        blocked_fields=blocked_fields,
        request_id=request_id,
    )
    fields = ", ".join(blocked_fields)
    raise BadRequestError(
        f"cannot update embedding model identity fields for an existing provider binding: {fields}",
        code="provider_binding.immutable_embedding_config",
    )


def _changed_embedding_identity_fields(old_routers: dict[str, Any], new_routers: dict[str, Any]) -> list[str]:
    old_endpoints = _router_endpoints(old_routers, "embed_model_router")
    new_endpoints = _router_endpoints(new_routers, "embed_model_router")
    blocked: list[str] = []
    max_len = max(len(old_endpoints), len(new_endpoints))
    for index in range(max_len):
        old_endpoint = old_endpoints[index] if index < len(old_endpoints) else {}
        new_endpoint = new_endpoints[index] if index < len(new_endpoints) else {}
        for field in IMMUTABLE_EMBEDDING_ENDPOINT_FIELDS:
            if old_endpoint.get(field) != new_endpoint.get(field):
                blocked.append(f"embed_model_router.endpoints[{index}].{field}")
    return blocked


def _router_endpoints(routers: dict[str, Any], router_name: str) -> list[dict[str, Any]]:
    router = routers.get(router_name) or {}
    endpoints = router.get("endpoints") or []
    return [endpoint for endpoint in endpoints if isinstance(endpoint, dict)]


def provider_binding_id(project_id: str, scope: ProviderBindingScope) -> str:
    """Return a deterministic binding id for a project/scope pair."""

    payload = {"project_id": project_id, "scope": scope.compact_dict()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:32]
    return f"pb_{digest}"


def get_provider_binding_service() -> ProviderBindingService:
    """Return the default provider binding management service."""

    return ProviderBindingService()


def get_provider_binding_resolver() -> ProviderBindingResolver:
    """Return the default runtime provider binding resolver."""

    return ProviderBindingResolver(
        store=QdrantProviderBindingStore(),
        enabled=get_config().provider_binding.enabled,
    )


def _record_from_qdrant(record: QdrantRecord) -> ProviderBindingRecord:
    payload = record.payload
    return ProviderBindingRecord(
        binding_id=str(payload.get("binding_id") or record.point_id),
        project_id=str(payload.get("project_id") or ""),
        scope=ProviderBindingScope.model_validate(payload.get("scope") or {}),
        routers=dict(payload.get("routers") or {}),
        enabled=bool(payload.get("enabled", True)),
    )


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in patch.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        elif isinstance(current, list) and isinstance(value, list):
            merged_list = list(current)
            for index, item in enumerate(value):
                if index < len(merged_list) and isinstance(merged_list[index], dict) and isinstance(item, dict):
                    merged_list[index] = _deep_merge(merged_list[index], item)
                elif index < len(merged_list):
                    merged_list[index] = deepcopy(item)
                else:
                    merged_list.append(deepcopy(item))
            result[key] = merged_list
        else:
            result[key] = deepcopy(value)
    return result
