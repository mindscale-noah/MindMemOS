"""Small model-client protocols used by the algorithm."""

from __future__ import annotations

from typing import Any, Protocol


class ChatModel(Protocol):
    async def chat(
        self,
        task: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any: ...


class EmbeddingModel(Protocol):
    async def embed(self, task: str, text: str | list[str], **kwargs: Any) -> Any: ...


async def chat_content(
    model: ChatModel,
    *,
    task: str,
    messages: list[dict[str, Any]],
) -> str:
    response = await model.chat(task=task, messages=messages)
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        return str(response.get("content") or "").strip()
    return str(getattr(response, "content", "") or "").strip()


async def embedding_vectors(model: EmbeddingModel, *, task: str, texts: list[str]) -> list[list[float]]:
    response = await model.embed(task=task, text=texts)
    if isinstance(response, list):
        values = response
    elif isinstance(response, dict):
        values = response.get("embeddings") or []
    else:
        values = getattr(response, "embeddings", []) or []
    return [[float(value) for value in vector] for vector in values]


__all__ = ["ChatModel", "EmbeddingModel", "chat_content", "embedding_vectors"]
