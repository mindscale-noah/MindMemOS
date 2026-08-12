"""Request-scoped adapter for the Platform-owned private model gateway."""

from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from typing import Any
from urllib.parse import SplitResult, urlsplit

import httpx
from omegaconf import DictConfig, ListConfig, OmegaConf

from ..config import ModelEndpointConfig, ModelRouterConfig

_GATEWAY_HTTP_CLIENT: httpx.AsyncClient | None = None
_DEFAULT_PLATFORM_GATEWAY_ORIGIN = "http://backend:8010"


class GatewayResponse:
    """Attribute-compatible view over an OpenAI-style JSON response."""

    def __init__(self, payload: dict[str, Any], *, attempted_retries: int, max_retries: int) -> None:
        # The parsed response is request-local already. Keep it directly and
        # copy only when callers explicitly request a mutable plain mapping.
        self._payload = payload
        self._hidden_params = {
            "additional_headers": {
                "x-litellm-attempted-retries": str(attempted_retries),
                "x-litellm-max-retries": str(max_retries),
            }
        }

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _attribute_view(value)

    def model_dump(self) -> dict[str, Any]:
        return deepcopy(self._payload)


class _AttributeMapping:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _attribute_view(value)


def _attribute_view(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttributeMapping(value)
    if isinstance(value, list):
        return [_attribute_view(item) for item in value]
    return value


def get_gateway_http_client() -> httpx.AsyncClient:
    """Return the process-wide identity-neutral HTTP connection pool."""

    global _GATEWAY_HTTP_CLIENT
    if _GATEWAY_HTTP_CLIENT is None or _GATEWAY_HTTP_CLIENT.is_closed:
        _GATEWAY_HTTP_CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            trust_env=False,
        )
    return _GATEWAY_HTTP_CLIENT


async def close_gateway_http_client() -> None:
    """Close and forget the shared gateway connection pool."""

    global _GATEWAY_HTTP_CLIENT
    client = _GATEWAY_HTTP_CLIENT
    _GATEWAY_HTTP_CLIENT = None
    if client is not None and not client.is_closed:
        await client.aclose()


class PlatformGatewayRouter:
    """LiteLLM-Router-compatible adapter for one hydrated Platform route."""

    def __init__(self, endpoint: ModelEndpointConfig, *, max_retries: int, retry_after: float) -> None:
        self._endpoint = endpoint
        self._max_retries = max(0, max_retries)
        self._retry_after = max(0.0, float(retry_after))

    async def acompletion(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> GatewayResponse:
        del model
        payload = self._request_payload(
            {
                "model": self._endpoint.model,
                "messages": messages,
                **self._configured_fields("temperature", "top_p", "max_tokens", "max_completion_tokens"),
            },
            kwargs,
        )
        return await self._post("chat/completions", payload)

    async def aembedding(self, *, model: str, input: str | list[str], **kwargs: Any) -> GatewayResponse:
        del model
        payload = self._request_payload(
            {
                "model": self._endpoint.model,
                "input": input,
                **self._configured_fields("encoding_format", "dimensions"),
            },
            kwargs,
        )
        return await self._post("embeddings", payload)

    async def arerank(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        top_n: int,
        **kwargs: Any,
    ) -> GatewayResponse:
        del model
        payload = self._request_payload(
            {
                "model": self._endpoint.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
            kwargs,
        )
        return await self._post("rerank", payload)

    def _configured_fields(self, *names: str) -> dict[str, Any]:
        return {name: value for name in names if (value := getattr(self._endpoint, name, None)) is not None}

    def _request_payload(self, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        payload = {**base, **_plain_mapping(getattr(self._endpoint, "extra_body", None))}
        call_extra = overrides.pop("extra_body", None)
        payload.update(_plain_mapping(call_extra))
        payload.update({key: value for key, value in overrides.items() if value is not None})
        return payload

    async def _post(self, path: str, payload: dict[str, Any]) -> GatewayResponse:
        url = f"{self._endpoint.api_base.rstrip('/')}/{path}"
        headers = {
            "Authorization": f"Bearer {self._endpoint.api_key}",
            "Content-Type": "application/json",
        }
        attempted_retries = 0
        while True:
            try:
                response = await get_gateway_http_client().request(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._endpoint.timeout,
                )
            except httpx.TransportError:
                if attempted_retries >= self._max_retries:
                    raise
                attempted_retries += 1
                await self._wait_before_retry()
                continue

            if _is_retryable_status(response.status_code) and attempted_retries < self._max_retries:
                attempted_retries += 1
                await self._wait_before_retry()
                continue
            response.raise_for_status()
            response_payload = response.json()
            if not isinstance(response_payload, dict):
                raise ValueError("Platform gateway response must be a JSON object")
            return GatewayResponse(
                response_payload,
                attempted_retries=attempted_retries,
                max_retries=self._max_retries,
            )

    async def _wait_before_retry(self) -> None:
        if self._retry_after > 0:
            await asyncio.sleep(self._retry_after)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value) if isinstance(value, dict) else {}


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def _origin(parts: SplitResult) -> tuple[str, str, int | None] | None:
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return parts.scheme.lower(), parts.hostname.lower(), port


def is_trusted_platform_gateway_url(api_base: str) -> bool:
    """Check a hydrated route against the server-owned gateway origin.

    The route path is intentionally not fixed: Platform can rename its private
    endpoint without requiring a MindMemOS code change. Credentials remain
    protected because only the configured origin may receive the service token.
    """

    trusted_raw = os.getenv("MINDMEMOS_PLATFORM_GATEWAY_ORIGIN", _DEFAULT_PLATFORM_GATEWAY_ORIGIN).strip()
    try:
        candidate = urlsplit(api_base)
        trusted = urlsplit(trusted_raw)
    except ValueError:
        return False
    if (
        _origin(candidate) is None
        or _origin(candidate) != _origin(trusted)
        or candidate.username is not None
        or candidate.password is not None
        or candidate.query
        or candidate.fragment
        or trusted.username is not None
        or trusted.password is not None
        or trusted.query
        or trusted.fragment
        or trusted.path.rstrip("/")
    ):
        return False
    return candidate.path.rstrip("/").endswith("/v1")


def build_platform_gateway_router(router_cfg: ModelRouterConfig, alias: str) -> tuple[PlatformGatewayRouter, int]:
    """Build an uncached request-scoped adapter for one Platform route."""

    if len(router_cfg.endpoints) != 1:
        raise ValueError("platform_gateway router requires exactly one endpoint")
    endpoint = router_cfg.endpoints[0]
    if getattr(endpoint, "transport", "litellm") != "platform_gateway":
        raise ValueError("platform_gateway router requires a platform_gateway endpoint")
    api_base = str(endpoint.api_base)
    api_key = str(endpoint.api_key)
    if (
        not is_trusted_platform_gateway_url(api_base)
        or "{userId}" in api_base
        or not api_key
        or api_key == "EMPTY"
    ):
        raise ValueError("platform_gateway router requires a hydrated private Platform route")
    if alias not in {"chat", "embedding", "rerank"}:
        raise ValueError(f"unsupported platform_gateway alias: {alias}")
    max_retries = max(0, int(endpoint.num_retries))
    return (
        PlatformGatewayRouter(endpoint, max_retries=max_retries, retry_after=router_cfg.retry_after),
        max_retries,
    )
