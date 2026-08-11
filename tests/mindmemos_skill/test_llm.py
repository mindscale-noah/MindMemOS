from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from mindmemos_skill.errors import EmbeddingDimensionError
from mindmemos_skill.infra.database import RecordQuery
from mindmemos_skill.llm import (
    DatabaseLLMCallSink,
    EmbedClient,
    LLMClient,
    build_litellm_params,
    llm_run_context,
)
from mindmemos_skill.persistence import (
    LLM_CALL_TABLE,
    LLMCallRecord,
    bootstrap_skill_database,
    from_database_record,
)


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


class CapturingSink:
    def __init__(self) -> None:
        self.records: list[LLMCallRecord] = []

    async def write(self, record: LLMCallRecord) -> None:
        self.records.append(record)


class AuditedChatResponse(SimpleNamespace):
    def model_dump(self) -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "model": self.model,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 1},
            },
        }


class AuditedEmbeddingResponse(SimpleNamespace):
    def model_dump(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 0,
                "total_tokens": 1,
                "provider_extension": {"billable_tokens": 1},
            },
        }


class AuditedRouter(FakeRouter):
    async def acompletion(self, **kwargs):
        del kwargs
        return AuditedChatResponse(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            model="chat-model",
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )

    async def aembedding(self, **kwargs):
        del kwargs
        return AuditedEmbeddingResponse(
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
async def test_chat_client_records_complete_run_scoped_call() -> None:
    sink = CapturingSink()
    client = LLMClient(AuditedRouter(), call_sink=sink)

    with llm_run_context("run-1"):
        await client.chat(
            task="skill_grpo.patch",
            messages=[{"role": "user", "content": "propose"}],
            temperature=0.2,
            api_key="must-not-be-stored",
        )

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.run_id == "run-1"
    assert record.task == "skill_grpo.patch"
    assert record.call_type == "chat"
    assert record.request == {
        "model": "chat",
        "messages": [{"role": "user", "content": "propose"}],
        "temperature": 0.2,
        "api_key": "<redacted>",
    }
    assert record.response is not None
    assert record.response["usage"]["prompt_tokens_details"] == {"cached_tokens": 1}
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (2, 1, 3)
    assert record.status == "succeeded"
    assert record.error is None


@pytest.mark.asyncio
async def test_llm_client_does_not_record_without_run_context() -> None:
    sink = CapturingSink()

    await LLMClient(AuditedRouter(), call_sink=sink).chat(task="outside-run", messages=[])

    assert sink.records == []


@pytest.mark.asyncio
async def test_chat_client_records_failed_call_before_reraising() -> None:
    class FailingRouter:
        async def acompletion(self, **kwargs):
            del kwargs
            raise RuntimeError("provider unavailable")

    sink = CapturingSink()
    with llm_run_context("run-failed"):
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await LLMClient(FailingRouter(), call_sink=sink).chat(task="skill_grpo.patch", messages=[])

    assert len(sink.records) == 1
    assert sink.records[0].status == "failed"
    assert sink.records[0].response is None
    assert sink.records[0].error == "RuntimeError: provider unavailable"


@pytest.mark.asyncio
async def test_database_sink_persists_llm_call_by_run_and_task(tmp_path) -> None:
    database = await bootstrap_skill_database(tmp_path / "state.db")
    try:
        with llm_run_context("run-database"):
            await LLMClient(AuditedRouter(), call_sink=DatabaseLLMCallSink(database)).chat(
                task="skill_grpo.cluster_fusion",
                messages=[{"role": "user", "content": "merge"}],
            )

        rows, cursor = await database.query_records(LLM_CALL_TABLE, RecordQuery())
        assert cursor is None
        assert len(rows) == 1
        record = from_database_record(rows[0], LLMCallRecord)
        assert (record.run_id, record.task) == ("run-database", "skill_grpo.cluster_fusion")
        assert (record.input_tokens, record.output_tokens, record.total_tokens) == (2, 1, 3)
    finally:
        await database.close()


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


@pytest.mark.asyncio
async def test_embedding_client_records_complete_run_scoped_call() -> None:
    sink = CapturingSink()

    with llm_run_context("run-embedding"):
        await EmbedClient(AuditedRouter(), call_sink=sink).embed(
            task="skill_grpo.edit_cluster",
            text=["find", "replace"],
        )

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.run_id == "run-embedding"
    assert record.task == "skill_grpo.edit_cluster"
    assert record.call_type == "embedding"
    assert record.request == {"model": "embedding", "input": ["find", "replace"]}
    assert record.response is not None
    assert record.response["usage"]["provider_extension"] == {"billable_tokens": 1}
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (1, 0, 1)
