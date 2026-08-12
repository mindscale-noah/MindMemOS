import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from mindmemos.api.schemas import AddRequest, AuthContext
from mindmemos.api.services.memory_service import MemoryService
from mindmemos.config import TextProcessingConfig, bind_config_overrides, get_config, init_config, reset_config
from mindmemos.errors import ApiError
from mindmemos.llm import EmbeddingResponse
from mindmemos.pipelines.add.vanilla import vanilla_add
from mindmemos.pipelines.add.vanilla.vanilla_add import VanillaAddPipeline
from mindmemos.typing import AddPipelineInput, MemoryDbSearchResult, MemoryDbWriteResult


class RecordingWriter:
    def __init__(self) -> None:
        self.calls = []

    async def apply_mutation_plan(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        plan = args[1]
        write_plan = plan.to_write_plan()
        return MemoryDbWriteResult(
            memory_ids=[memory.memory_id for memory in write_plan.memories],
            entity_ids=[],
        )


class EmptyReader:
    async def list_memories(self, _context, *, filters=None, limit=50, cursor=None):
        return [], None

    async def search_sparse(self, _context, request, *, indices, values):
        return MemoryDbSearchResult(query=request.query, hits=[], total=0)


class RecordingRecorder:
    async def record_add_input(self, *args, **kwargs):
        return None

    async def mark_add_completed(self, *args, **kwargs):
        return None

    async def mark_add_failed(self, *args, **kwargs):
        return None


class ScopedLlm:
    def __init__(self, name: str, barrier: asyncio.Event, entered: list[str]) -> None:
        self.name = name
        self.calls = []
        self._barrier = barrier
        self._entered = entered

    async def chat(self, *, task, messages, format_parser=None, **kwargs):
        self.calls.append((task, messages))
        self._entered.append(self.name)
        if len(self._entered) >= 2:
            self._barrier.set()
        await self._barrier.wait()
        return SimpleNamespace(
            parsed={
                "memories": [
                    {
                        "ref_id": "m1",
                        "content": f"memory-from-{self.name}",
                        "mem_type": "fact",
                        "confidence": 1.0,
                        "action_hint": "add",
                    }
                ],
                "entities": [],
                "sources": [],
                "property_bindings": [],
            }
        )


class ScopedEmbed:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []

    async def embed(self, task, text, **kwargs):
        self.calls.append((task, text))
        texts = text if isinstance(text, list) else [text]
        return EmbeddingResponse(embeddings=[[1.0, 2.0, 3.0] for _ in texts])


@pytest.fixture
def dynamic_provider_config():
    init_config(config_path=Path("config/mindmemos/dev.example.yaml"))
    config = get_config()
    config.provider_binding.enabled = True
    config.chat_model_router.endpoints.clear()
    config.embed_model_router.endpoints.clear()
    try:
        yield
    finally:
        reset_config()


def _project_routes(name: str, *, include_embed: bool = True) -> dict:
    routes = {
        "chat_model_router": {
            "endpoints": [
                {
                    "model": f"openai/{name}-chat",
                    "api_key": f"sk-{name}",
                    "api_base": f"https://{name}.example.test/v1",
                }
            ]
        },
        "embed_model_router": {"endpoints": []},
    }
    if include_embed:
        routes["embed_model_router"]["endpoints"] = [
            {
                "model": f"openai/{name}-embed",
                "api_key": f"sk-{name}",
                "api_base": f"https://{name}.example.test/v1",
                "dimensions": 2560,
            }
        ]
    return routes


def _pipeline(writer: RecordingWriter | None = None) -> VanillaAddPipeline:
    return VanillaAddPipeline(
        db_reader=EmptyReader(),
        db_writer=writer or RecordingWriter(),
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
        consistency="strong",
        recorder=RecordingRecorder(),
    )


def test_pipeline_created_without_static_endpoints_resolves_clients_from_request(
    monkeypatch, dynamic_provider_config
) -> None:
    clients: dict[tuple[str, str], object] = {}

    def resolve(kind: str):
        endpoints = getattr(get_config(), f"{kind}_model_router").endpoints
        model = str(endpoints[0].model) if endpoints else "missing"
        return clients.setdefault((kind, model), object())

    monkeypatch.setattr(vanilla_add, "get_llm_client", lambda: resolve("chat"))
    monkeypatch.setattr(vanilla_add, "get_embed_client", lambda: resolve("embed"))
    pipeline = _pipeline()

    with bind_config_overrides(project_config=_project_routes("alice")):
        builder = pipeline._resolve_builder()

    assert builder._memory_extractor._llm_client is clients[("chat", "openai/alice-chat")]
    assert builder._llm_client is clients[("chat", "openai/alice-chat")]
    assert builder._vectorizer._embed_client is clients[("embed", "openai/alice-embed")]


@pytest.mark.asyncio
async def test_shared_pipeline_keeps_concurrent_user_clients_isolated(
    monkeypatch, dynamic_provider_config
) -> None:
    clients: dict[tuple[str, str], object] = {}
    ready = asyncio.Event()
    entered = 0

    def resolve(kind: str):
        endpoint = getattr(get_config(), f"{kind}_model_router").endpoints[0]
        model = str(endpoint.model)
        return clients.setdefault((kind, model), object())

    monkeypatch.setattr(vanilla_add, "get_llm_client", lambda: resolve("chat"))
    monkeypatch.setattr(vanilla_add, "get_embed_client", lambda: resolve("embed"))
    pipeline = _pipeline()

    async def resolve_for(name: str):
        nonlocal entered
        with bind_config_overrides(project_config=_project_routes(name)):
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            await asyncio.sleep(0)
            return pipeline._resolve_builder()

    alice_builder, bob_builder = await asyncio.gather(resolve_for("alice"), resolve_for("bob"))

    assert alice_builder._llm_client is clients[("chat", "openai/alice-chat")]
    assert alice_builder._vectorizer._embed_client is clients[("embed", "openai/alice-embed")]
    assert bob_builder._llm_client is clients[("chat", "openai/bob-chat")]
    assert bob_builder._vectorizer._embed_client is clients[("embed", "openai/bob-embed")]
    assert alice_builder._llm_client is not bob_builder._llm_client
    assert alice_builder._vectorizer._embed_client is not bob_builder._vectorizer._embed_client


@pytest.mark.asyncio
async def test_memory_service_executes_concurrent_adds_with_each_users_clients(
    monkeypatch, dynamic_provider_config
) -> None:
    barrier = asyncio.Event()
    entered: list[str] = []
    llm_clients = {
        "openai/alice-chat": ScopedLlm("alice", barrier, entered),
        "openai/bob-chat": ScopedLlm("bob", barrier, entered),
        "openai/charlie-chat": ScopedLlm("charlie", barrier, entered),
    }
    embed_clients = {
        "openai/alice-embed": ScopedEmbed("alice"),
        "openai/bob-embed": ScopedEmbed("bob"),
        "openai/charlie-embed": ScopedEmbed("charlie"),
    }

    def current_model(kind: str) -> str:
        return str(getattr(get_config(), f"{kind}_model_router").endpoints[0].model)

    monkeypatch.setattr(vanilla_add, "get_llm_client", lambda: llm_clients[current_model("chat")])
    monkeypatch.setattr(vanilla_add, "get_embed_client", lambda: embed_clients[current_model("embed")])

    class Resolver:
        async def resolve(self, context):
            return _project_routes(context.account_id)

    writer = RecordingWriter()
    pipeline = _pipeline(writer)
    service = MemoryService(
        add_pipeline=pipeline,
        provider_binding_resolver=Resolver(),
        operation_recorder=RecordingRecorder(),
    )

    async def add_for(name: str):
        return await service.add(
            AuthContext(
                request_id=f"request-{name}",
                account_id=name,
                project_id=f"project-{name}",
                api_key_uuid=f"key-{name}",
                memory_algorithm="vanilla",
            ),
            AddRequest(
                user_id=f"user-{name}",
                mode="sync",
                messages=[{"role": "user", "content": f"hello from {name}"}],
            ),
        )

    alice_result, bob_result = await asyncio.gather(add_for("alice"), add_for("bob"))

    assert alice_result.status == bob_result.status == "ok"
    assert len(llm_clients["openai/alice-chat"].calls) == 1
    assert len(llm_clients["openai/bob-chat"].calls) == 1
    assert embed_clients["openai/alice-embed"].calls == [
        ("memory.add.embed", ["memory-from-alice"])
    ]
    assert embed_clients["openai/bob-embed"].calls == [
        ("memory.add.embed", ["memory-from-bob"])
    ]
    writes = {
        call[0][0].project_id: [memory.content for memory in call[0][1].to_write_plan().memories]
        for call in writer.calls
    }
    assert writes == {
        "project-alice": ["memory-from-alice"],
        "project-bob": ["memory-from-bob"],
    }

    stream_events = [
        event
        async for event in service.add_stream(
            AuthContext(
                request_id="request-charlie",
                account_id="charlie",
                project_id="project-charlie",
                api_key_uuid="key-charlie",
                memory_algorithm="vanilla",
            ),
            AddRequest(
                user_id="user-charlie",
                mode="sync",
                messages=[{"role": "user", "content": "hello from charlie"}],
            ),
        )
    ]

    assert stream_events[-1]["event"] == "completed"
    assert len(llm_clients["openai/charlie-chat"].calls) == 1
    assert embed_clients["openai/charlie-embed"].calls == [
        ("memory.add.embed", ["memory-from-charlie"])
    ]


@pytest.mark.asyncio
async def test_incomplete_dynamic_routes_fail_before_writer(
    monkeypatch, dynamic_provider_config
) -> None:
    writer = RecordingWriter()
    pipeline = _pipeline(writer)
    monkeypatch.setattr(vanilla_add, "get_llm_client", lambda: object())
    monkeypatch.setattr(vanilla_add, "get_embed_client", lambda: object())

    with bind_config_overrides(project_config=_project_routes("alice", include_embed=False)):
        with pytest.raises(ApiError, match="Embedding model endpoint") as exc_info:
            await pipeline.add_sync(
                AddPipelineInput(messages=[{"text": "must not be written"}]),
                SimpleNamespace(request_id="request-alice"),
            )

    assert exc_info.value.code == "provider_binding.model_endpoint_missing"
    assert exc_info.value.status_code == 409
    assert writer.calls == []
