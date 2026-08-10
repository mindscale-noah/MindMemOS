from __future__ import annotations

from types import SimpleNamespace

import pytest
from mindmemos_skill.errors import EmbeddingDimensionError
from mindmemos_skill.llm import EmbedClient, LLMClient, build_litellm_params


class FakeRouter:
    async def acompletion(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                    finish_reason="stop",
                )
            ],
            model="chat-model",
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )

    async def aembedding(self, **kwargs):
        return SimpleNamespace(
            data=[{"embedding": [0.1, 0.2]}],
            model="embedding-model",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )


def test_litellm_params_accept_plain_mapping() -> None:
    params = build_litellm_params(
        {
            "model": "openai/custom-embedding",
            "api_key": "secret",
            "dimensions": 2,
            "unused": "ignored",
        },
        dimensions_supported_models=["custom-embedding"],
    )

    assert params == {
        "model": "openai/custom-embedding",
        "api_key": "secret",
        "dimensions": 2,
        "allowed_openai_params": ["dimensions"],
    }


@pytest.mark.asyncio
async def test_chat_client_returns_local_response_contract() -> None:
    response = await LLMClient(FakeRouter()).chat(task="test", messages=[{"role": "user", "content": "hi"}])

    assert response.content == "ok"
    assert response.model == "chat-model"
    assert response.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_chat_client_preserves_openai_tool_calls() -> None:
    class ToolCallRouter:
        async def acompletion(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    type="function",
                                    function=SimpleNamespace(name="lookup", arguments='{"query":"demo"}'),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                model="chat-model",
                usage=None,
            )

    response = await LLMClient(ToolCallRouter()).chat(task="test", messages=[])

    assert response.to_assistant_message() == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query":"demo"}'},
            }
        ],
    }


@pytest.mark.asyncio
async def test_embedding_client_validates_requested_dimension() -> None:
    client = EmbedClient(FakeRouter())

    response = await client.embed(task="test", text="hello", expected_dim=2)
    assert response.embeddings == [[0.1, 0.2]]

    with pytest.raises(EmbeddingDimensionError):
        await client.embed(task="test", text="hello", expected_dim=3)
