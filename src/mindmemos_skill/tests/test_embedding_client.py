"""Embedding transport validation using the SDK with a local mock transport."""

import asyncio
import json

import httpx
import pytest
from mindmemos_skill.infra.embedding import OpenAIEmbeddingClient, OpenAIEmbeddingConfig
from openai import AsyncOpenAI


def config(**updates):
    return OpenAIEmbeddingConfig(model="test", base_url="http://test.invalid/v1", dimensions=2, batch_size=2, **updates)


async def test_batches_indices_and_borrowed_client_lifecycle():
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        data = [
            {"object": "embedding", "index": i, "embedding": [float(i + 1), 1.0]} for i in range(len(payload["input"]))
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test",
                "data": list(reversed(data)),
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="http://test.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as sdk:
        client = OpenAIEmbeddingClient(config(send_dimensions=True), client=sdk)
        assert (await client.embed([])) == []
        assert not calls
        assert (await client.embed(["a", "b", "c"])) == [[1, 1], [2, 1], [1, 1]]
        assert sorted(payload["input"] for payload in calls) == [["a", "b"], ["c"]]
        assert all(payload["dimensions"] == 2 and payload["encoding_format"] == "float" for payload in calls)
        assert not sdk.is_closed()


async def test_embedding_batches_overlap_and_preserve_input_order():
    """Start every batch before completion and retain order despite late responses."""
    active = 0
    peak = 0
    all_started = asyncio.Event()
    last_completed = asyncio.Event()
    completed = []

    async def handle_batch(request):
        nonlocal active, peak
        texts = json.loads(request.content)["input"]
        active += 1
        peak = max(peak, active)
        if active == 3:
            all_started.set()
        try:
            await all_started.wait()
            if texts[0] != "5":
                await last_completed.wait()
            completed.append(texts[0])
            last_completed.set()
            return httpx.Response(
                200,
                json={"data": [{"index": i, "embedding": [float(text), 1]} for i, text in enumerate(texts)]},
            )
        finally:
            active -= 1

    async with AsyncOpenAI(
        api_key="test",
        base_url="http://test.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle_batch)),
    ) as sdk:
        vectors = await asyncio.wait_for(
            OpenAIEmbeddingClient(config(), client=sdk).embed(["1", "2", "3", "4", "5"]), timeout=5
        )
        assert peak == 3 and active == 0
        assert completed[0] == "5"
        assert vectors == [[float(index), 1] for index in range(1, 6)]
        assert not sdk.is_closed()


@pytest.mark.parametrize("failure", ["transport", "validation", "cancel"])
async def test_embedding_failure_or_cancellation_drains_pending_batches(failure):
    """Cancel sibling requests before returning errors or caller cancellation."""
    pending_started = asyncio.Event()
    pending_cancelled = asyncio.Event()

    async def handle_batch(request):
        texts = json.loads(request.content)["input"]
        if texts[0] == "pending":
            pending_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pending_cancelled.set()
                raise
        await pending_started.wait()
        if failure == "cancel":
            await asyncio.Future()
        if failure == "transport":
            raise httpx.ConnectError("Controlled failure", request=request)
        return httpx.Response(200, json={"data": []})

    async with AsyncOpenAI(
        api_key="test",
        base_url="http://test.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle_batch)),
    ) as sdk:
        task = asyncio.create_task(OpenAIEmbeddingClient(config(), client=sdk).embed(["fail", "other", "pending"]))
        if failure == "cancel":
            await asyncio.wait_for(pending_started.wait(), timeout=5)
            task.cancel()
        expected_error = {
            "transport": RuntimeError,
            "validation": ValueError,
            "cancel": asyncio.CancelledError,
        }[failure]
        with pytest.raises(expected_error):
            await asyncio.wait_for(task, timeout=5)
        assert pending_cancelled.is_set()
        assert not sdk.is_closed()


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 1, "embedding": [1, 1]}],
        [{"index": 0, "embedding": [1]}],
        [{"index": 0, "embedding": [0, 0]}],
        [],
    ],
)
async def test_invalid_provider_response_is_rejected(data):
    def handler(request):
        return httpx.Response(
            200,
            json={"object": "list", "model": "test", "data": data, "usage": {"prompt_tokens": 1, "total_tokens": 1}},
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="http://test.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as sdk:
        with pytest.raises(ValueError):
            (await OpenAIEmbeddingClient(config(), client=sdk).embed(["text"]))


def test_embedding_registry_constructs_independent_lazy_backends():
    from mindmemos_skill.infra.embedding import EmbeddingModel

    settings = config()
    restored = OpenAIEmbeddingConfig.model_validate_json(settings.model_dump_json())
    first = EmbeddingModel.from_config(restored)
    second = EmbeddingModel.from_config(restored)
    assert isinstance(first, OpenAIEmbeddingClient)
    assert first is not second
    assert first.config == settings
    assert first.config is not settings
    assert first._client is None


def test_embedding_registry_uses_explicit_identity_for_custom_backends(monkeypatch):
    from mindmemos_skill.infra.embedding import EmbeddingConfig, EmbeddingModel

    monkeypatch.setattr(EmbeddingModel, "_registry", EmbeddingModel.registered_types())

    @EmbeddingModel.register("test_embedding")
    class CustomEmbedding(EmbeddingModel[EmbeddingConfig]):
        """Exercise configuration-driven construction without provider access."""

        config_type = EmbeddingConfig
        """Explicit backend identity for the offline implementation."""

        async def embed(self, texts):
            return [[1.0] for _ in texts]

    settings = EmbeddingConfig(class_name="test_embedding")
    assert type(EmbeddingModel.from_config(settings)) is CustomEmbedding
    with pytest.raises(ValueError, match="Unknown EmbeddingModel"):
        EmbeddingModel.from_config(EmbeddingConfig(class_name="missing"))


def test_embedding_backend_identity_is_validated_by_the_config():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="class_name"):
        config(class_name="unsupported_embedding")


def test_builtin_embedding_registration_loads_in_a_fresh_process():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from mindmemos_skill.infra.embedding import EmbeddingModel; "
            "assert 'openai_embedding' in EmbeddingModel.registered_types()",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
