from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import mindmemos.provider_bindings as provider_bindings
import pytest
from mindmemos.components.feedback import DefaultExplicitFeedbackPlanner
from mindmemos.config import TextProcessingConfig, bind_config_overrides, get_config, init_config, reset_config
from mindmemos.errors import ApiError
from mindmemos.llm import EmbeddingResponse, require_model_endpoint
from mindmemos.pipelines.add.vanilla import vanilla_add
from mindmemos.pipelines.add.vanilla.vanilla_add import VanillaAddPipeline
from mindmemos.pipelines.dreaming import default as dreaming_mod
from mindmemos.pipelines.dreaming.default import DefaultDreamingPipeline
from mindmemos.pipelines.search.agentic import wrapper as agentic_mod
from mindmemos.pipelines.search.agentic.wrapper import AgenticSearchWrapper
from mindmemos.pipelines.search.vanilla import engine as vanilla_search_mod
from mindmemos.pipelines.search.vanilla.engine import VanillaSearchEngine
from mindmemos.typing import MemoryRequestContext, SearchPipelineInput


class NamedEmbed:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str] = []

    async def embed(self, *, task: str, text, **_kwargs) -> EmbeddingResponse:
        self.calls.append(str(text))
        return EmbeddingResponse(embeddings=[[1.0, 2.0, 3.0]])


class NamedLlm:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def dynamic_config():
    init_config(config_path=Path("config/mindmemos/dev.example.yaml"))
    get_config().provider_binding.enabled = True
    try:
        yield
    finally:
        reset_config()


def routes(name: str) -> dict:
    return {
        "chat_model_router": {
            "endpoints": [
                {
                    "model": f"openai/{name}-chat",
                    "api_key": "test",
                    "api_base": "https://example.test/v1",
                }
            ]
        },
        "embed_model_router": {
            "endpoints": [
                {
                    "model": f"openai/{name}-embed",
                    "api_key": "test",
                    "api_base": "https://example.test/v1",
                    "dimensions": 2560,
                }
            ]
        },
    }


def current_model(kind: str) -> str:
    return str(getattr(get_config(), f"{kind}_model_router").endpoints[0].model)


def context(name: str) -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id=f"request-{name}",
        account_id=name,
        project_id=f"project-{name}",
        api_key_uuid=f"key-{name}",
        memory_algorithm="vanilla",
        user_id=f"user-{name}",
    )


@pytest.mark.asyncio
async def test_cached_vanilla_search_resolves_embedding_per_dynamic_request(
    monkeypatch, dynamic_config
) -> None:
    clients = {
        "openai/alice-embed": NamedEmbed("alice"),
        "openai/bob-embed": NamedEmbed("bob"),
    }
    monkeypatch.setattr(
        vanilla_search_mod,
        "get_embed_client",
        lambda: clients[current_model("embed")],
    )

    with bind_config_overrides(project_config=routes("alice")):
        engine = VanillaSearchEngine(db_reader=SimpleNamespace(), db_writer=SimpleNamespace())

    async def encode(name: str):
        with bind_config_overrides(project_config=routes(name)):
            await asyncio.sleep(0)
            return await engine._encode_dense(name)

    alice, bob = await asyncio.gather(encode("alice"), encode("bob"))

    assert alice == bob == [1.0, 2.0, 3.0]
    assert clients["openai/alice-embed"].calls == ["alice"]
    assert clients["openai/bob-embed"].calls == ["bob"]


@pytest.mark.asyncio
async def test_cached_agentic_wrapper_resolves_llm_per_dynamic_request(monkeypatch, dynamic_config) -> None:
    clients = {
        "openai/alice-chat": NamedLlm("alice"),
        "openai/bob-chat": NamedLlm("bob"),
    }
    monkeypatch.setattr(agentic_mod, "get_llm_client", lambda: clients[current_model("chat")])
    observed: list[tuple[str, str]] = []

    class Planner:
        def __init__(self, *, llm, **_kwargs) -> None:
            observed.append(("planner", llm.name))

    class Sufficiency:
        def __init__(self, *, llm, **_kwargs) -> None:
            observed.append(("sufficiency", llm.name))

    class Loop:
        def __init__(self, **_kwargs) -> None:
            pass

        async def run(self, **_kwargs):
            return []

    monkeypatch.setattr(agentic_mod, "LLMAgenticPlanner", Planner)
    monkeypatch.setattr(agentic_mod, "LLMSufficiencyEvaluator", Sufficiency)
    monkeypatch.setattr(agentic_mod, "AgenticLoop", Loop)

    with bind_config_overrides(project_config=routes("alice")):
        wrapper = AgenticSearchWrapper()

    class Engine:
        name = "vanilla"

    async def run(name: str) -> None:
        with bind_config_overrides(project_config=routes(name)):
            await wrapper.run(
                SearchPipelineInput(query=name, search_pipeline="vanilla", agentic=True),
                context(name),
                Engine(),
            )

    await run("alice")
    await run("bob")

    assert observed == [
        ("planner", "alice"),
        ("sufficiency", "alice"),
        ("planner", "bob"),
        ("sufficiency", "bob"),
    ]


def test_feedback_planner_does_not_cache_dynamic_llm(monkeypatch, dynamic_config) -> None:
    clients = {
        "openai/alice-chat": NamedLlm("alice"),
        "openai/bob-chat": NamedLlm("bob"),
    }
    monkeypatch.setattr(
        "mindmemos.components.feedback.explicit_planner.get_llm_client",
        lambda: clients[current_model("chat")],
    )
    planner = DefaultExplicitFeedbackPlanner()

    with bind_config_overrides(project_config=routes("alice")):
        alice = planner._client
    with bind_config_overrides(project_config=routes("bob")):
        bob = planner._client

    assert alice is clients["openai/alice-chat"]
    assert bob is clients["openai/bob-chat"]


def test_dreaming_pipeline_resolves_dynamic_clients_per_request(monkeypatch, dynamic_config) -> None:
    llms = {
        "openai/alice-chat": NamedLlm("alice"),
        "openai/bob-chat": NamedLlm("bob"),
    }
    embeds = {
        "openai/alice-embed": NamedEmbed("alice"),
        "openai/bob-embed": NamedEmbed("bob"),
    }
    monkeypatch.setattr(dreaming_mod, "get_llm_client", lambda: llms[current_model("chat")])
    monkeypatch.setattr(dreaming_mod, "get_embed_client", lambda: embeds[current_model("embed")])

    with bind_config_overrides(project_config=routes("alice")):
        pipeline = DefaultDreamingPipeline(db_reader=SimpleNamespace(), db_writer=SimpleNamespace())
        alice_llm = pipeline._resolve_llm_client()
        alice_embed = pipeline._resolve_embed_client()
    with bind_config_overrides(project_config=routes("bob")):
        bob_llm = pipeline._resolve_llm_client()
        bob_embed = pipeline._resolve_embed_client()

    assert alice_llm is llms["openai/alice-chat"]
    assert bob_llm is llms["openai/bob-chat"]
    assert alice_embed is embeds["openai/alice-embed"]
    assert bob_embed is embeds["openai/bob-embed"]


def test_static_vanilla_add_reuses_builder(monkeypatch) -> None:
    init_config(config_path=Path("config/mindmemos/dev.example.yaml"))
    get_config().provider_binding.enabled = False
    monkeypatch.setattr(vanilla_add, "get_llm_client", lambda: object())
    monkeypatch.setattr(vanilla_add, "get_embed_client", lambda: object())
    try:
        pipeline = VanillaAddPipeline(
            db_reader=SimpleNamespace(),
            db_writer=SimpleNamespace(),
            text_config=TextProcessingConfig(
                bm25_use_spacy_lemma=False,
                spacy_en_model="missing_en_model",
                spacy_zh_model="missing_zh_model",
                sparse_hash_dim=128,
            ),
        )

        assert pipeline._resolve_builder() is pipeline._resolve_builder()
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_missing_dynamic_binding_preserves_static_router_config(dynamic_config) -> None:
    class MissingBindingResolver:
        async def resolve(self, request_context):
            return None

    before = get_config().chat_model_router.endpoints[0].model
    factory = getattr(provider_bindings, "provider_config_context", None)
    assert callable(factory), "provider context restoration must be shared outside MemoryService"

    config_context = await factory(context("unbound"), resolver=MissingBindingResolver())
    with config_context:
        assert get_config().chat_model_router.endpoints[0].model == before


def test_missing_required_dynamic_endpoint_is_structured_conflict(dynamic_config) -> None:
    with bind_config_overrides(project_config={"embed_model_router": {"endpoints": []}}):
        with pytest.raises(ApiError, match="Embedding model endpoint") as exc_info:
            require_model_endpoint("embedding")

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "provider_binding.model_endpoint_missing"
