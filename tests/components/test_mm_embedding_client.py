import json

import httpx
import pytest
from mindmemos.config import MmEmbedEndpointConfig, MmEmbedRouterConfig
from mindmemos.errors import EmbeddingDimensionError
from mindmemos.llm.mm_embedding import ImageInput, MmEmbedClient, MmEmbedError


def _router(*endpoints: dict) -> MmEmbedRouterConfig:
    return MmEmbedRouterConfig(endpoints=[MmEmbedEndpointConfig(**ep) for ep in endpoints])


def _ok(request: httpx.Request, dims: int = 3) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200, json={"data": [{"index": i, "embedding": [float(i)] * dims} for i in range(len(body["input"]))]}
    )


@pytest.mark.asyncio
async def test_openai_protocol_request_shapes_and_batching() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization"), json.loads(request.content)))
        return _ok(request)

    client = MmEmbedClient(
        _router({"model": "siglip", "api_base": "http://mm/v1/", "api_key": "k", "dimensions": 3, "max_batch_size": 2}),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    vectors = await client.embed_texts(["a", "b", "c"])
    await client.embed_images([ImageInput(data=b"\x89PNG", mime_type="image/png")])

    assert len(vectors) == 3
    assert [s[0] for s in seen] == ["http://mm/v1/embeddings"] * 3
    assert seen[0][1] == "Bearer k"
    assert seen[0][2] == {"model": "siglip", "input": ["a", "b"], "encoding_format": "float", "dimensions": 3}
    assert seen[2][2]["input"] == [{"image": "data:image/png;base64,iVBORw=="}]


@pytest.mark.asyncio
async def test_url_transport_sends_known_urls() -> None:
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _ok(request)

    client = MmEmbedClient(
        _router({"model": "m", "api_base": "http://mm", "dimensions": 3, "image_transport": "url"}),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await client.embed_images([ImageInput(data=b"x", url="https://cdn/x.png"), ImageInput(data=b"y")])
    assert bodies[0]["input"] == [{"image": "https://cdn/x.png"}, {"image": "data:image/png;base64,eQ=="}]


@pytest.mark.asyncio
async def test_retries_then_fails_over_to_the_next_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("mindmemos.llm.mm_embedding.asyncio.sleep", _no_sleep)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "bad":
            return httpx.Response(503, text="busy")
        return _ok(request)

    client = MmEmbedClient(
        _router(
            {"model": "m", "api_base": "http://bad", "dimensions": 3, "num_retries": 1},
            {"model": "m", "api_base": "http://good", "dimensions": 3},
        ),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert len(await client.embed_texts(["q"])) == 1
    assert calls == ["bad", "bad", "good"]


@pytest.mark.asyncio
async def test_errors_and_dimension_mismatch() -> None:
    bad_request = MmEmbedClient(
        _router({"model": "m", "api_base": "http://mm", "dimensions": 3}),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(400, text="nope"))),
    )
    with pytest.raises(MmEmbedError):
        await bad_request.embed_texts(["q"])
    wrong_dims = MmEmbedClient(
        _router({"model": "m", "api_base": "http://mm", "dimensions": 4}),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_ok)),
    )
    with pytest.raises(EmbeddingDimensionError):
        await wrong_dims.embed_texts(["q"])


async def _no_sleep(_seconds):
    return None
