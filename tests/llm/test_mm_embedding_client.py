from types import SimpleNamespace

import pytest
from mindmemos.config import MmEmbedEndpointConfig, MmEmbedRouterConfig
from mindmemos.errors import EmbeddingDimensionError
from mindmemos.llm.mm_embedding import ImageInput, MmEmbedClient, MmEmbedError


def _router_cfg(*endpoints: dict) -> MmEmbedRouterConfig:
    return MmEmbedRouterConfig(endpoints=[MmEmbedEndpointConfig(**ep) for ep in endpoints])


class _FakeRouter:
    """Records aembedding calls; returns N vectors of the given dimension."""

    def __init__(self, *, dims: int = 3, fail: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._dims = dims
        self._fail = fail

    async def aembedding(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail is not None:
            raise self._fail
        data = [
            {"object": "embedding", "index": i, "embedding": [float(i)] * self._dims}
            for i in range(len(kwargs["input"]))
        ]
        return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_request_shapes_and_batching() -> None:
    router = _FakeRouter()
    client = MmEmbedClient(
        _router_cfg(
            {
                "model": "openai/siglip",
                "api_base": "http://mm/v1/",
                "api_key": "k",
                "dimensions": 3,
                "max_batch_size": 2,
            }
        ),
        router=router,
    )
    vectors = await client.embed_texts(["a", "b", "c"])
    await client.embed_images([ImageInput(data=b"\x89PNG", mime_type="image/png")])

    assert len(vectors) == 3
    assert [call["model"] for call in router.calls] == ["mm_embedding"] * 3
    assert router.calls[0]["input"] == ["a", "b"]
    assert router.calls[1]["input"] == ["c"]
    assert router.calls[0]["encoding_format"] == "float"
    assert router.calls[2]["input"] == [{"image": "data:image/png;base64,iVBORw=="}]


@pytest.mark.asyncio
async def test_url_transport_sends_known_urls() -> None:
    router = _FakeRouter()
    client = MmEmbedClient(
        _router_cfg({"model": "openai/m", "api_base": "http://mm", "dimensions": 3, "image_transport": "url"}),
        router=router,
    )
    await client.embed_images([ImageInput(data=b"x", url="https://cdn/x.png"), ImageInput(data=b"y")])
    assert router.calls[0]["input"] == [{"image": "https://cdn/x.png"}, {"image": "data:image/png;base64,eQ=="}]


@pytest.mark.asyncio
async def test_provider_failure_maps_to_mm_embed_error() -> None:
    client = MmEmbedClient(
        _router_cfg({"model": "openai/m", "api_base": "http://mm", "dimensions": 3}),
        router=_FakeRouter(fail=RuntimeError("HTTP 503: busy")),
    )
    with pytest.raises(MmEmbedError, match="busy"):
        await client.embed_texts(["q"])


@pytest.mark.asyncio
async def test_errors_and_dimension_mismatch() -> None:
    short_response = MmEmbedClient(
        _router_cfg({"model": "openai/m", "api_base": "http://mm", "dimensions": 3}),
        router=_FakeRouter(dims=3, fail=ValueError("expected 1 embeddings, got 0")),
    )
    with pytest.raises(MmEmbedError):
        await short_response.embed_texts(["q"])
    wrong_dims = MmEmbedClient(
        _router_cfg({"model": "openai/m", "api_base": "http://mm", "dimensions": 4}),
        router=_FakeRouter(dims=3),
    )
    with pytest.raises(EmbeddingDimensionError):
        await wrong_dims.embed_texts(["q"])


@pytest.mark.asyncio
async def test_empty_input_skips_the_router() -> None:
    router = _FakeRouter()
    client = MmEmbedClient(_router_cfg({"model": "openai/m", "api_base": "http://mm", "dimensions": 3}), router=router)
    assert await client.embed_texts([]) == []
    assert await client.embed_images([]) == []
    assert router.calls == []
