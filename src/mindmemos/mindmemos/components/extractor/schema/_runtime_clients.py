"""Runtime client resolution helpers for schema add components."""

from __future__ import annotations

from ....llm import (
    EmbedClient,
    LLMClient,
    get_embed_client,
    get_llm_client,
    require_model_endpoint,
)


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
