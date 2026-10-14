"""vanilla_mm must behave exactly like vanilla for requests without images.

The same text-only dialogues run through the vanilla and the vanilla_mm add pipelines, then the
same queries through the vanilla and vanilla_mm search engines. Every LLM prompt, every
embedding request, every DB write plan and every result must be identical, with the multimodal
embedding client configured (so the image path is live, just unused).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from mindmemos.config import TextProcessingConfig, VanillaMmConfig
from mindmemos.config.algo.search import VanillaSearchConfig
from mindmemos.llm import EmbeddingResponse
from mindmemos.pipelines.add.vanilla import VanillaAddPipeline
from mindmemos.pipelines.add.vanilla_mm import VanillaMmAddPipeline
from mindmemos.pipelines.search.pipeline import SearchPipelineImpl
from mindmemos.pipelines.search.vanilla import VanillaSearchEngine
from mindmemos.pipelines.search.vanilla_mm import VanillaMmSearchEngine
from mindmemos.typing import AddPipelineInput, MemoryRequestContext, SearchPipelineInput

from tests.pipelines.add.test_default_add_pipeline import InMemoryVanillaStore as _BaseStore

TEXT_CONFIG = TextProcessingConfig(
    bm25_use_spacy_lemma=False,
    spacy_en_model="missing_en_model",
    spacy_zh_model="missing_zh_model",
    sparse_hash_dim=128,
)

DIALOGUES = [
    [
        {"role": "user", "content": "I just adopted a beagle named Biscuit.", "timestamp": 1717000000000},
        {"role": "assistant", "content": "Congratulations! Beagles are lively.", "timestamp": 1717000001000},
    ],
    [
        {"role": "user", "content": "Biscuit hates the vacuum cleaner and loves carrots."},
        {"role": "assistant", "content": "Carrots are a healthy treat for dogs."},
    ],
    [{"text": "Kai moved to Lisbon in March and works on Qdrant tooling."}],
]
QUERIES = ["What is my dog called?", "Where does Kai live?", "Biscuit food preferences"]


class InMemoryVanillaStore(_BaseStore):
    """The shared in-memory store, extended with the reader calls of the dense (hybrid) path."""

    async def search_hybrid(self, context, req, *, dense_vector, sparse_vector, dense_limit=None, sparse_limit=None):
        return await super().search_hybrid(context, req, dense_vector=dense_vector, sparse_vector=sparse_vector)

    async def get_memory_lineage(self, context, memory_ids):
        return {}


class RecordingLlm:
    """Deterministic extractor LLM: derives memories from the prompt and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def chat(self, *, task, messages, format_parser=None, **kwargs):
        prompt = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
        self.calls.append((task, prompt))
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        last = str(messages[-1].get("content", ""))[-120:]
        return SimpleNamespace(
            parsed={
                "memories": [
                    {
                        "ref_id": "m1",
                        "content": f"fact {digest}: {last}",
                        "mem_type": "fact",
                        "confidence": 0.9,
                        "action_hint": "add",
                    }
                ],
                "entities": [],
                "sources": [],
                "property_bindings": [],
            }
        )


class RecordingEmbed:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def embed(self, task, text, **kwargs):
        self.calls.append((task, text))
        texts = text if isinstance(text, list) else [text]
        return EmbeddingResponse(embeddings=[_vector(item) for item in texts])


class RecordingMmClient:
    model = "fake-mm"
    dimensions = 4

    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.image_calls: list[int] = []

    async def embed_texts(self, texts, *, task="mm.text"):
        self.text_calls.append(list(texts))
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

    async def embed_images(self, images, *, task="mm.image"):
        self.image_calls.append(len(images))
        return [[0.5, 0.5, 0.5, 0.5] for _ in images]


class EmptyImageStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search_images(self, ctx, vector, **kwargs):
        self.calls.append("search")
        return []

    async def get_images(self, ctx, image_ids):
        self.calls.append("get")
        return []

    async def upsert_images(self, ctx, items):
        self.calls.append("upsert")

    async def update_images(self, ctx, records):
        self.calls.append("update")


class RecordingRecorder:
    def __init__(self) -> None:
        self.completed: list[tuple[str, dict]] = []

    async def mark_add_completed(self, ctx, add_record_id, result) -> None:
        self.completed.append((add_record_id, result.model_dump(mode="json")))


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [byte / 255 for byte in digest[:8]]


def _context(index: int) -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id=f"req-{index}",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


def _normalized(value):
    """Drop wall-clock timestamps; everything else must match exactly."""

    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items() if not isinstance(item, datetime)}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def _plans(store: InMemoryVanillaStore) -> list:
    return [_normalized(entry.plan.model_dump(mode="python")) for entry in store.mutation_plans]


def _search_config():
    return SimpleNamespace(
        algo_config=SimpleNamespace(
            text_processing=TEXT_CONFIG,
            search=SimpleNamespace(vanilla=VanillaSearchConfig(), retention=None),
            vanilla_mm=VanillaMmConfig(),
        )
    )


async def _run(algorithm: str, monkeypatch) -> dict:
    for target in (
        "mindmemos.pipelines.search.pipeline.get_config",
        "mindmemos.pipelines.search.vanilla.engine.get_config",
        "mindmemos.components.text.preprocessor.get_config",
    ):
        monkeypatch.setattr(target, _search_config)
    store = InMemoryVanillaStore()
    llm, embed, mm, images, recorder = (
        RecordingLlm(),
        RecordingEmbed(),
        RecordingMmClient(),
        EmptyImageStore(),
        RecordingRecorder(),
    )
    common = dict(
        db_reader=store,
        db_writer=store,
        text_config=TEXT_CONFIG,
        consistency="fast",
        llm_client=llm,
        embed_client=embed,
        recorder=recorder,
    )
    if algorithm == "vanilla":
        add = VanillaAddPipeline(**common)
        engine = VanillaSearchEngine(db_reader=store, db_writer=store, text_config=TEXT_CONFIG, embed_client=embed)
    else:
        add = VanillaMmAddPipeline(
            **common, image_store=images, mm_embed_client=mm, vanilla_mm_config=VanillaMmConfig()
        )
        engine = VanillaMmSearchEngine(
            db_reader=store,
            db_writer=store,
            text_config=TEXT_CONFIG,
            embed_client=embed,
            image_store=images,
            mm_embed_client=mm,
            vanilla_mm_config=VanillaMmConfig(),
        )
    search = SearchPipelineImpl(
        db_reader=store, db_writer=store, engines={algorithm: engine}, final_filter=None, rerank_client=None
    )

    add_results = []
    for index, messages in enumerate(DIALOGUES):
        # Pin the request time: it defaults to the wall clock, which differs between any two runs.
        inp = AddPipelineInput(messages=messages, event_timestamp_ms=1717000000000 + index * 60_000)
        result = await add.add_sync(inp, _context(index), add_record_id=f"rec-{index}")
        add_results.append(result.model_dump(mode="json"))
    search_results = []
    for query in QUERIES:
        result = await search.search(SearchPipelineInput(query=query, search_pipeline=algorithm, top_k=5), _context(99))
        search_results.append(result.model_dump(mode="json"))
    return {
        "llm": llm.calls,
        "embed": embed.calls,
        "plans": _plans(store),
        "add": add_results,
        "records": recorder.completed,
        "search": search_results,
        "mm_image_calls": mm.image_calls,
        "image_store_calls": [call for call in images.calls if call != "search"],
    }


@pytest.mark.asyncio
async def test_text_only_requests_match_vanilla_exactly(monkeypatch) -> None:
    vanilla = await _run("vanilla", monkeypatch)
    vanilla_mm = await _run("vanilla_mm", monkeypatch)

    assert vanilla["llm"], "the fake LLM must be exercised"
    assert vanilla["plans"], "memories must be written"
    assert any(result["memories"] for result in vanilla["search"]), "searches must return memories"
    for key in ("llm", "embed", "plans", "add", "records", "search"):
        assert vanilla_mm[key] == vanilla[key], key
    # Nothing image-related is written for text-only requests.
    assert vanilla_mm["mm_image_calls"] == []
    assert vanilla_mm["image_store_calls"] == []
