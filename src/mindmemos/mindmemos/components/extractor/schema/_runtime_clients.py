"""Runtime client resolution helpers for schema add components."""

from __future__ import annotations

from ....llm import (
    EmbedClient,
    LLMClient,
    get_embed_client,
    get_llm_client,
    provider_binding_runtime_enabled,
    require_model_endpoint,
)


def initial_llm_client(explicit_client: LLMClient | None) -> LLMClient | None:
    """Preserve static-mode eager clients while allowing dynamic mode lazy resolution."""

    if explicit_client is not None:
        return explicit_client
    if provider_binding_runtime_enabled():
        return None
    return get_llm_client()


def initial_embed_client(explicit_client: EmbedClient | None) -> EmbedClient | None:
    """Preserve static-mode eager clients while allowing dynamic mode lazy resolution."""

    if explicit_client is not None:
        return explicit_client
    if provider_binding_runtime_enabled():
        return None
    return get_embed_client()


def resolve_llm_client(client: LLMClient | None) -> LLMClient:
    """Return an injected client or resolve one from the current request context."""
    if client is None:
        require_model_endpoint("chat")
    return client or get_llm_client()


def resolve_embed_client(client: EmbedClient | None) -> EmbedClient:
    """Return an injected client or resolve one from the current request context."""
    if client is None:
        require_model_endpoint("embedding")
    return client or get_embed_client()
