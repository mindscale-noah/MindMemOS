"""Multimodal embedding client: text and images encoded into one vector space over HTTP.

``MmEmbedClient.embed_texts`` / ``embed_images`` are the only entry points. The wire format is
owned by a protocol adapter selected per endpoint (``mm_embed_model_router.endpoints[].protocol``),
so supporting another provider means adding one adapter, not touching callers.

Protocol ``openai`` (OpenAI-style ``POST {api_base}/embeddings``)::

    request  {"model": M, "input": ["text", ...], "dimensions": D}                # texts
    request  {"model": M, "input": [{"image": "data:image/png;base64,..."}, ...]}  # images
             (``{"image": "https://..."}`` when ``image_transport: url`` and a URL is known)
    response {"data": [{"index": i, "embedding": [...]}, ...], "model": M}

The image item shape follows the Jina-style multimodal extension of the OpenAI embeddings API.
"""

from __future__ import annotations

import asyncio
import base64
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import httpx
from omegaconf import DictConfig, ListConfig, OmegaConf

from ..config import MmEmbedEndpointConfig, MmEmbedRouterConfig, get_config
from ..errors import ApiError, EmbeddingDimensionError
from ..logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


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


class MmEmbedProtocol(Protocol):
    """Wire-format adapter for one family of multimodal embedding APIs."""

    def text_request(self, endpoint: MmEmbedEndpointConfig, texts: list[str]) -> tuple[str, dict[str, Any]]: ...

    def image_request(
        self, endpoint: MmEmbedEndpointConfig, images: list[ImageInput]
    ) -> tuple[str, dict[str, Any]]: ...

    def parse(self, payload: dict[str, Any], expected: int) -> list[list[float]]: ...


class OpenAIMmProtocol:
    """OpenAI-style ``/embeddings`` with Jina-style ``{"image": ...}`` items."""

    def text_request(self, endpoint: MmEmbedEndpointConfig, texts: list[str]) -> tuple[str, dict[str, Any]]:
        return "/embeddings", self._body(endpoint, list(texts))

    def image_request(self, endpoint: MmEmbedEndpointConfig, images: list[ImageInput]) -> tuple[str, dict[str, Any]]:
        items = []
        for image in images:
            use_url = endpoint.image_transport == "url" and image.url
            items.append({"image": image.url if use_url else image.data_uri()})
        return "/embeddings", self._body(endpoint, items)

    def parse(self, payload: dict[str, Any], expected: int) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise ValueError(f"expected {expected} embeddings, got {len(data) if isinstance(data, list) else 'none'}")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [[float(x) for x in item["embedding"]] for item in ordered]

    @staticmethod
    def _body(endpoint: MmEmbedEndpointConfig, inputs: list[Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": endpoint.model, "input": inputs, "encoding_format": "float"}
        if endpoint.dimensions is not None:
            body["dimensions"] = int(endpoint.dimensions)
        extra = endpoint.extra_body
        if isinstance(extra, (DictConfig, ListConfig)):
            extra = OmegaConf.to_container(extra, resolve=True)
        if extra:
            body.update(extra)
        return body


PROTOCOLS: dict[str, MmEmbedProtocol] = {"openai": OpenAIMmProtocol()}


class MmEmbedError(ApiError):
    status_code = 502
    code = "mm_embedding.provider_request_failed"


class MmEmbedClient:
    """Embeds texts and images through the configured endpoints, in order, with retries."""

    def __init__(self, router: MmEmbedRouterConfig, *, http_client: httpx.AsyncClient | None = None) -> None:
        if not router.endpoints:
            raise ValueError("mm_embed_model_router has no endpoints")
        self._endpoints = list(router.endpoints)
        self._http = http_client

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

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _embed(self, kind: str, inputs: list[Any], *, task: str) -> list[list[float]]:
        if not inputs:
            return []
        vectors: list[list[float]] = []
        batch_size = max(1, int(self._endpoints[0].max_batch_size))
        for start in range(0, len(inputs), batch_size):
            vectors.extend(await self._embed_batch(kind, inputs[start : start + batch_size], task=task))
        return vectors

    async def _embed_batch(self, kind: str, batch: list[Any], *, task: str) -> list[list[float]]:
        last_error: Exception | None = None
        for endpoint in self._endpoints:
            protocol = PROTOCOLS[endpoint.protocol]
            path, body = (
                protocol.text_request(endpoint, batch) if kind == "text" else protocol.image_request(endpoint, batch)
            )
            for attempt in range(int(endpoint.num_retries) + 1):
                start = perf_counter()
                try:
                    payload = await self._post(endpoint, path, body)
                    vectors = protocol.parse(payload, len(batch))
                except _Retryable as exc:
                    last_error = exc
                    logger.info(
                        "mm_embed_call",
                        kind=kind,
                        task=task,
                        model=endpoint.model,
                        status="retry",
                        attempt=attempt + 1,
                        latency_ms=round((perf_counter() - start) * 1000, 2),
                        error=str(exc)[:200],
                    )
                    await asyncio.sleep(min(8.0, 0.5 * 2**attempt) * (0.5 + random.random()))
                    continue
                except Exception as exc:  # noqa: BLE001 - non-retryable: try the next endpoint
                    last_error = exc
                    logger.warning(
                        "mm_embed_call",
                        kind=kind,
                        task=task,
                        model=endpoint.model,
                        status="error",
                        error=str(exc)[:200],
                    )
                    break
                for vector in vectors:
                    if endpoint.dimensions is not None and len(vector) != int(endpoint.dimensions):
                        raise EmbeddingDimensionError(
                            expected=int(endpoint.dimensions), actual=len(vector), model=endpoint.model, task=task
                        )
                logger.info(
                    "mm_embed_call",
                    kind=kind,
                    task=task,
                    model=endpoint.model,
                    status="ok",
                    inputs=len(batch),
                    latency_ms=round((perf_counter() - start) * 1000, 2),
                )
                return vectors
        raise MmEmbedError(f"multimodal embedding failed: {last_error}") from last_error

    async def _post(self, endpoint: MmEmbedEndpointConfig, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self._http is None:
            self._http = httpx.AsyncClient()
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        url = endpoint.api_base.rstrip("/") + path
        try:
            response = await self._http.post(url, json=body, headers=headers, timeout=float(endpoint.timeout))
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _Retryable(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code in _RETRYABLE_STATUS:
            raise _Retryable(f"HTTP {response.status_code}: {response.text[:200]}")
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code}: {response.text[:200]}")
        return response.json()


class _Retryable(Exception):
    pass


_CLIENTS: dict[tuple[Any, ...], MmEmbedClient] = {}


def _fingerprint(router: MmEmbedRouterConfig) -> tuple[Any, ...]:
    return tuple(
        (ep.model, ep.api_base, ep.api_key, ep.dimensions, ep.protocol, ep.image_transport, ep.max_batch_size)
        for ep in router.endpoints
    )


def get_mm_embed_client() -> MmEmbedClient | None:
    """Return the multimodal embedding client, or None when no endpoint is configured."""

    router = get_config().mm_embed_model_router
    if not router.endpoints:
        return None
    key = _fingerprint(router)
    client = _CLIENTS.get(key)
    if client is None:
        client = _CLIENTS[key] = MmEmbedClient(router)
    return client


async def close_mm_embed_clients() -> None:
    clients = list(_CLIENTS.values())
    _CLIENTS.clear()
    for client in clients:
        await client.close()
