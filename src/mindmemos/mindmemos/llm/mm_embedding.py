"""Multimodal embedding client: text and images encoded into one vector space.

``MmEmbedClient.embed_texts`` / ``embed_images`` are the only entry points. Requests
route through a shared ``litellm.Router`` (built by ``llm.router.get_router``) exactly
like the chat/embedding/rerank clients, so retries, cooldowns and endpoint failover
are owned by litellm instead of this module.

Images ride the Jina-style multimodal extension of the OpenAI embeddings API —
``input`` items of the shape ``{"image": ...}`` — which litellm passes through to
``openai/``-compatible endpoints verbatim::

    request  {"model": M, "input": ["text", ...], "encoding_format": "float"}            # texts
    request  {"model": M, "input": [{"image": "data:image/png;base64,..."}, ...]}        # images
             (``{"image": "https://..."}`` when ``image_transport: url`` and a URL is known)
    response {"data": [{"index": i, "embedding": [...]}], "model": M}

The endpoint ``model`` is a LiteLLM identifier with a provider prefix (e.g.
``openai/siglip2-base-patch16-384``); litellm strips the prefix on the wire, so the
serving endpoint sees the bare model name.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from ..config import MmEmbedEndpointConfig, MmEmbedRouterConfig, get_config
from ..errors import ApiError, EmbeddingDimensionError
from ..logging import get_logger
from .router import ModelRouter, get_router

logger = get_logger(__name__)


@dataclass(frozen=True)
class ImageInput:
    """One image to embed: raw bytes, or a URL the endpoint can fetch."""

    data: bytes | None = None
    mime_type: str = "image/png"
    url: str | None = None

    def data_uri(self) -> str:
        if self.data is None:
            raise ValueError("image has no inline bytes")
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode('ascii')}"


class MmEmbedError(ApiError):
    status_code = 502
    code = "mm_embedding.provider_request_failed"


def _parse_vectors(resp: Any, expected: int) -> list[list[float]]:
    """Read embedding vectors from a litellm response, dict- or attribute-style."""

    data = getattr(resp, "data", None) or []
    vectors: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
        vectors.append([float(x) for x in embedding or []])
    if len(vectors) != expected:
        raise ValueError(f"expected {expected} embeddings, got {len(vectors)}")
    return vectors


def _placeholder_key(ep: MmEmbedEndpointConfig) -> MmEmbedEndpointConfig:
    """Give keyless ``openai/`` endpoints a placeholder api_key.

    litellm's openai provider builds an OpenAI SDK client that refuses to construct
    without credentials, so a genuinely keyless endpoint (empty ``api_key``, e.g. the
    bundled ``mm_embed_server``) would fail every call before any request is sent.
    The placeholder satisfies the client; the endpoint ignores the unused bearer token.
    """

    if ep.api_key or not str(ep.model).startswith("openai/"):
        return ep
    return replace(ep, api_key="none")


class MmEmbedClient:
    """Embeds texts and images through the shared LiteLLM router for the mm endpoints."""

    ALIAS = "mm_embedding"

    def __init__(self, router_cfg: MmEmbedRouterConfig, *, router: ModelRouter | None = None) -> None:
        if not router_cfg.endpoints:
            raise ValueError("mm_embed_model_router has no endpoints")
        self._endpoints: list[MmEmbedEndpointConfig] = [_placeholder_key(ep) for ep in router_cfg.endpoints]
        # Router-level transport details (image_transport, max_batch_size) read from the
        # first endpoint: a litellm Router picks deployments itself, so per-endpoint
        # request shaping is not possible — all mm endpoints must agree on them.
        if router is not None:
            self._router = router
        else:
            self._router = get_router(replace(router_cfg, endpoints=self._endpoints), self.ALIAS)[0]

    @property
    def dimensions(self) -> int:
        return int(self._endpoints[0].dimensions)

    @property
    def model(self) -> str:
        return str(self._endpoints[0].model)

    async def embed_texts(self, texts: list[str], *, task: str = "mm.text") -> list[list[float]]:
        return await self._embed("text", list(texts), task=task)

    async def embed_images(self, images: list[ImageInput], *, task: str = "mm.image") -> list[list[float]]:
        return await self._embed("image", list(images), task=task)

    def _image_items(self, images: list[ImageInput]) -> list[dict[str, str]]:
        use_url = self._endpoints[0].image_transport == "url"
        items: list[dict[str, str]] = []
        for image in images:
            items.append({"image": image.url if (use_url and image.url) else image.data_uri()})
        return items

    async def _embed(self, kind: str, inputs: list[Any], *, task: str) -> list[list[float]]:
        if not inputs:
            return []
        vectors: list[list[float]] = []
        batch_size = max(1, int(self._endpoints[0].max_batch_size))
        for start in range(0, len(inputs), batch_size):
            vectors.extend(await self._embed_batch(kind, inputs[start : start + batch_size], task=task))
        return vectors

    async def _embed_batch(self, kind: str, batch: list[Any], *, task: str) -> list[list[float]]:
        items = batch if kind == "text" else self._image_items(batch)
        start = perf_counter()
        try:
            resp = await self._router.aembedding(model=self.ALIAS, input=items, encoding_format="float")
            vectors = _parse_vectors(resp, len(batch))
        except Exception as exc:  # noqa: BLE001 - one batch failed after litellm retries/failover
            logger.warning(
                "mm_embed_call",
                kind=kind,
                task=task,
                model=self.model,
                status="error",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            raise MmEmbedError(f"multimodal embedding failed: {exc}") from exc
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingDimensionError(expected=self.dimensions, actual=len(vector), model=self.model, task=task)
        logger.info(
            "mm_embed_call",
            kind=kind,
            task=task,
            model=self.model,
            status="ok",
            inputs=len(batch),
            latency_ms=round((perf_counter() - start) * 1000, 2),
        )
        return vectors


def get_mm_embed_client() -> MmEmbedClient | None:
    """Return the multimodal embedding client, or None when no endpoint is configured.

    The underlying litellm Router is cached per resolved endpoint config (see
    ``llm.router.get_router``); the wrapper itself is stateless, so building it per
    call is cheap and always reflects the currently bound config.
    """

    router_cfg = get_config().mm_embed_model_router
    if not router_cfg.endpoints:
        return None
    return MmEmbedClient(router_cfg)
