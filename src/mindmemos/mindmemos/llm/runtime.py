"""Runtime ownership and validation helpers for model clients."""

from __future__ import annotations

from typing import Literal

from ..config import get_config
from ..errors import BadRequestError, ConfigNotInitializedError

ModelCapability = Literal["chat", "embedding", "rerank"]

_ROUTER_BY_CAPABILITY = {
    "chat": "chat_model_router",
    "embedding": "embed_model_router",
    "rerank": "rerank_model_router",
}
_CAPABILITY_LABEL = {"chat": "Chat", "embedding": "Embedding", "rerank": "Rerank"}


def provider_binding_runtime_enabled() -> bool:
    """Return whether model clients must be resolved from request context."""

    try:
        return bool(get_config().provider_binding.enabled)
    except (ConfigNotInitializedError, AttributeError):
        return False


def require_model_endpoint(capability: ModelCapability) -> None:
    """Fail clearly when dynamic routing has no usable required endpoint."""

    if not provider_binding_runtime_enabled():
        return
    router = getattr(get_config(), _ROUTER_BY_CAPABILITY[capability], None)
    endpoints = getattr(router, "endpoints", None) if router is not None else None
    if endpoints:
        return
    raise BadRequestError(
        f"{_CAPABILITY_LABEL[capability]} model endpoint is not configured for this request",
        code="provider_binding.model_endpoint_missing",
        status_code=409,
    )
