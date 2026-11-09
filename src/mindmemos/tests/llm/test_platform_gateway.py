from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable

import httpx
import pytest
from mindmemos.config.app import ModelEndpointConfig, ModelRouterConfig


def _gateway_module():
    try:
        return importlib.import_module("mindmemos.llm.gateway")
    except ModuleNotFoundError:
        pytest.fail("Platform gateway adapter is not implemented")


class FakeHttpClient:
    def __init__(self, responses: list[httpx.Response] | Callable[[dict], httpx.Response]) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    async def request(self, method: str, url: str, **kwargs):
        call = {"method": method, "url": url, **kwargs}
        self.calls.append(call)
        response = self._responses(call) if callable(self._responses) else self._responses.pop(0)
        if response.request is None:
            response.request = httpx.Request(method, url)
        return response


def _response(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "http://gateway.test"))


def _router_config(
    *,
    alias: str,
    user: str = "user-a",
    token: str = "token-a",
    dimensions: int | None = None,
    retries: int = 0,
) -> ModelRouterConfig:
    return ModelRouterConfig(
        endpoints=[
            ModelEndpointConfig(
                model=f"openai/{alias}-model",
                api_key=token,
                api_base=f"http://backend:8010/litellm_memory_proxy/{user}/v1",
                transport="platform_gateway",
                dimensions=dimensions,
                timeout=17,
                num_retries=retries,
                temperature=0.25 if alias == "chat" else None,
                encoding_format="float" if alias == "embedding" else None,
            )
        ],
        retry_after=0,
    )


@pytest.mark.parametrize(
    ("api_base", "api_key"),
    [
        ("https://provider.example/v1", "service-token"),
        ("http://backend:8010/litellm_memory_proxy/{userId}/v1", "EMPTY"),
    ],
)
def test_gateway_rejects_non_private_or_unhydrated_routes(api_base: str, api_key: str) -> None:
    gateway = _gateway_module()
    cfg = _router_config(alias="chat")
    cfg.endpoints[0].api_base = api_base
    cfg.endpoints[0].api_key = api_key

    with pytest.raises(ValueError, match="hydrated private Platform route"):
        gateway.build_platform_gateway_router(cfg, "chat")


def test_gateway_accepts_renamed_route_on_configured_trusted_origin(monkeypatch) -> None:
    gateway = _gateway_module()
    monkeypatch.setenv("MINDMEMOS_PLATFORM_GATEWAY_ORIGIN", "http://gateway:9090")
    cfg = _router_config(alias="chat")
    cfg.endpoints[0].api_base = "http://gateway:9090/internal/model-gateway/user-a/v1"

    router, retries = gateway.build_platform_gateway_router(cfg, "chat")

    assert router._endpoint.api_base == "http://gateway:9090/internal/model-gateway/user-a/v1"
    assert retries == 0


def test_gateway_url_validation_fails_closed_for_malformed_url() -> None:
    gateway = _gateway_module()

    assert gateway.is_trusted_platform_gateway_url("http://[invalid/v1") is False


@pytest.mark.asyncio
async def test_gateway_chat_forwards_openai_payload_without_outer_litellm(monkeypatch) -> None:
    gateway = _gateway_module()
    client = FakeHttpClient(
        [
            _response(
                200,
                {
                    "model": "actual-chat-model",
                    "choices": [{"message": {"content": "remembered"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )
        ]
    )
    monkeypatch.setattr(gateway, "get_gateway_http_client", lambda: client)
    router, retries = gateway.build_platform_gateway_router(_router_config(alias="chat"), "chat")

    response = await router.acompletion(
        model="chat",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=64,
    )

    assert retries == 0
    assert response.choices[0].message.content == "remembered"
    assert response.model_dump()["usage"]["total_tokens"] == 5
    assert client.calls == [
        {
            "method": "POST",
            "url": "http://backend:8010/litellm_memory_proxy/user-a/v1/chat/completions",
            "headers": {"Authorization": "Bearer token-a", "Content-Type": "application/json"},
            "json": {
                "model": "openai/chat-model",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.25,
                "max_tokens": 64,
            },
            "timeout": 17,
        }
    ]


@pytest.mark.asyncio
async def test_gateway_embedding_forwards_dimensions_and_encoding_exactly(monkeypatch) -> None:
    gateway = _gateway_module()
    client = FakeHttpClient(
        [
            _response(
                200,
                {
                    "model": "actual-embedding-model",
                    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(gateway, "get_gateway_http_client", lambda: client)
    router, _ = gateway.build_platform_gateway_router(
        _router_config(alias="embedding", dimensions=3),
        "embedding",
    )

    response = await router.aembedding(model="embedding", input=["memory"])

    assert response.data[0].embedding == [0.1, 0.2, 0.3]
    assert client.calls[0]["json"] == {
        "model": "openai/embedding-model",
        "input": ["memory"],
        "encoding_format": "float",
        "dimensions": 3,
    }


@pytest.mark.asyncio
async def test_gateway_rerank_forwards_query_documents_and_top_n(monkeypatch) -> None:
    gateway = _gateway_module()
    client = FakeHttpClient(
        [_response(200, {"model": "actual-reranker", "results": [{"index": 1, "relevance_score": 0.9}]})]
    )
    monkeypatch.setattr(gateway, "get_gateway_http_client", lambda: client)
    router, _ = gateway.build_platform_gateway_router(_router_config(alias="rerank"), "rerank")

    response = await router.arerank(
        model="rerank",
        query="where",
        documents=["a", "b"],
        top_n=1,
    )

    assert response.results[0].index == 1
    assert client.calls[0]["url"].endswith("/rerank")
    assert client.calls[0]["json"] == {
        "model": "openai/rerank-model",
        "query": "where",
        "documents": ["a", "b"],
        "top_n": 1,
    }


@pytest.mark.asyncio
async def test_gateway_retries_transient_response_with_same_actor_route(monkeypatch) -> None:
    gateway = _gateway_module()
    client = FakeHttpClient(
        [
            _response(503, {"code": "model.provider_unavailable", "message": "temporarily unavailable"}),
            _response(
                200,
                {"model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            ),
        ]
    )
    monkeypatch.setattr(gateway, "get_gateway_http_client", lambda: client)
    router, _ = gateway.build_platform_gateway_router(_router_config(alias="chat", retries=1), "chat")

    response = await router.acompletion(model="chat", messages=[])

    assert len(client.calls) == 2
    assert client.calls[0]["url"] == client.calls[1]["url"]
    assert client.calls[0]["headers"] == client.calls[1]["headers"]
    assert client.calls[0]["json"] == client.calls[1]["json"]
    headers = response._hidden_params["additional_headers"]
    assert headers["x-litellm-attempted-retries"] == "1"
    assert headers["x-litellm-max-retries"] == "1"


@pytest.mark.asyncio
async def test_gateway_retries_every_server_error(monkeypatch) -> None:
    gateway = _gateway_module()
    client = FakeHttpClient(
        [
            _response(501, {"code": "not_implemented", "message": "temporary upstream response"}),
            _response(
                200,
                {"model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            ),
        ]
    )
    monkeypatch.setattr(gateway, "get_gateway_http_client", lambda: client)
    router, _ = gateway.build_platform_gateway_router(_router_config(alias="chat", retries=1), "chat")

    response = await router.acompletion(model="chat", messages=[])

    assert response.choices[0].message.content == "ok"
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_gateway_does_not_retry_deterministic_client_error(monkeypatch) -> None:
    gateway = _gateway_module()
    client = FakeHttpClient([_response(400, {"code": "model.provider_request_rejected", "message": "bad request"})])
    monkeypatch.setattr(gateway, "get_gateway_http_client", lambda: client)
    router, _ = gateway.build_platform_gateway_router(_router_config(alias="chat", retries=2), "chat")

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await router.acompletion(model="chat", messages=[])

    assert len(client.calls) == 1
    assert exc_info.value.response.json()["code"] == "model.provider_request_rejected"


@pytest.mark.asyncio
async def test_request_scoped_gateway_adapters_share_only_neutral_http_pool(monkeypatch) -> None:
    gateway = _gateway_module()

    def respond(call: dict) -> httpx.Response:
        user = call["url"].split("/litellm_memory_proxy/", 1)[1].split("/", 1)[0]
        return _response(
            200,
            {"model": user, "choices": [{"message": {"content": user}, "finish_reason": "stop"}]},
        )

    client = FakeHttpClient(respond)
    monkeypatch.setattr(gateway, "get_gateway_http_client", lambda: client)
    router_a, _ = gateway.build_platform_gateway_router(
        _router_config(alias="chat", user="user-a", token="token-a"),
        "chat",
    )
    router_b, _ = gateway.build_platform_gateway_router(
        _router_config(alias="chat", user="user-b", token="token-b"),
        "chat",
    )

    response_a, response_b = await asyncio.gather(
        router_a.acompletion(model="chat", messages=[]),
        router_b.acompletion(model="chat", messages=[]),
    )

    assert {response_a.choices[0].message.content, response_b.choices[0].message.content} == {"user-a", "user-b"}
    routes = {(call["url"], call["headers"]["Authorization"]) for call in client.calls}
    assert routes == {
        ("http://backend:8010/litellm_memory_proxy/user-a/v1/chat/completions", "Bearer token-a"),
        ("http://backend:8010/litellm_memory_proxy/user-b/v1/chat/completions", "Bearer token-b"),
    }
