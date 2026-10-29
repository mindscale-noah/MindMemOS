"""Multimodal vanilla must behave exactly like plain vanilla wherever images are not used.

(a) Text-only adds and searches, with ``algo_config.multimodal.mode`` off, basic or enhanced, run
    through the multimodal wrappers and through plain ``VanillaAddPipeline`` /
    ``VanillaSearchEngine``: every LLM prompt, embedding request, DB write plan, add record and
    search result is identical, with the multimodal embedding client configured.
(b) basic / enhanced, for a user who has stored images: a search that is not multimodal returns
    vanilla's results with vanilla's calls only (no multimodal embedding, no metadata read, no
    image query).
(c) enhanced: what the VLM reads from the pixels stays on the image records; text memory holds
    the dialogue facts only.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.components.multimodal import ImageLoader
from mindmemos.config import BlobStoreConfig, MultimodalConfig, TextProcessingConfig
from mindmemos.config.algo.search import VanillaSearchConfig
from mindmemos.errors import BadRequestError
from mindmemos.infra.blob import LocalBlobStore
from mindmemos.llm import EmbeddingResponse
from mindmemos.pipelines.add.vanilla import VanillaAddPipeline
from mindmemos.pipelines.add.vanilla_multimodal import VanillaMultimodalAddPipeline
from mindmemos.pipelines.search.pipeline import SearchPipelineImpl
from mindmemos.pipelines.search.vanilla import VanillaSearchEngine
from mindmemos.pipelines.search.vanilla_multimodal import VanillaMultimodalSearchEngine
from mindmemos.typing import AddPipelineInput, ImageRef, MemoryRequestContext, SearchPipelineInput

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
    """The shared in-memory store, extended with the reader calls of the dense (hybrid) path; reads are logged."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[tuple[str, object]] = []

    async def search_hybrid(self, context, req, *, dense_vector, sparse_vector, dense_limit=None, sparse_limit=None):
        self.reads.append(("search_hybrid", req.query))
        return await super().search_hybrid(context, req, dense_vector=dense_vector, sparse_vector=sparse_vector)

    async def search_sparse(self, context, req, *, indices, values):
        self.reads.append(("search_sparse", req.query))
        return await super().search_sparse(context, req, indices=indices, values=values)

    async def get_memories(self, context, memory_ids):
        self.reads.append(("get_memories", list(memory_ids)))
        return await super().get_memories(context, memory_ids)

    async def get_memory_lineage(self, context, memory_ids):
        self.reads.append(("get_memory_lineage", list(memory_ids)))
        return {}


VISUAL_FACT = "VISUAL: the sofa is green velvet with three seats"


class RecordingLlm:
    """Deterministic extractor LLM: derives memories from the prompt and records every call.

    A call with images (the multimodal extraction) also returns what the image shows.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def chat(self, *, task, messages, format_parser=None, **kwargs):
        prompt = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
        self.calls.append((task, prompt))
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        last = str(messages[-1].get("content", ""))[-120:]
        parsed = {
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
        if isinstance(messages[-1].get("content"), list):
            parsed["memories"] = [
                {"ref_id": "m1", "content": "The user bought a new sofa.", "mem_type": "fact", "source_refs": ["s0"]}
            ]
            parsed["image_memories"] = [{"image": 1, "description": "A green velvet sofa.", "facts": [VISUAL_FACT]}]
        return SimpleNamespace(parsed=parsed)


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


class RecordingImageStore:
    """In-memory image collection; every call is logged."""

    text_vector_size = 8

    def __init__(self) -> None:
        self.records: dict = {}
        self.text_vectors: dict = {}
        self.calls: list[str] = []

    async def search_images(self, ctx, vector, **kwargs):
        self.calls.append("search")
        return []

    async def get_images(self, ctx, image_ids):
        self.calls.append("get")
        return [self.records[i] for i in image_ids if i in self.records]

    async def upsert_images(self, ctx, items):
        self.calls.append("upsert")
        self.records.update({record.image_id: record for record, _ in items})

    async def update_images(self, ctx, records, *, text_vectors=None):
        self.calls.append("update")
        self.records.update({record.image_id: record for record in records})
        self.text_vectors.update(text_vectors or {})

    async def list_session_images(self, ctx, session_id, *, user_id=None, limit=50):
        self.calls.append("list_session")
        return []


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
        )
    )


_FROZEN_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


class _FrozenDatetime(datetime):
    """The pipelines stamp records with ``datetime.now``; freeze it so the two runs cannot straddle a second."""

    @classmethod
    def now(cls, tz=None):  # noqa: PYI024 - mirrors datetime.now's signature
        return _FROZEN_NOW if tz is None else _FROZEN_NOW.astimezone(tz)


class _Harness:
    """One store and one set of fake clients; plain vanilla or the multimodal wrappers on top."""

    def __init__(self, monkeypatch, tmp_path) -> None:
        for target in (
            "mindmemos.pipelines.search.pipeline.get_config",
            "mindmemos.pipelines.search.vanilla.engine.get_config",
            "mindmemos.components.text.preprocessor.get_config",
            "mindmemos.pipelines.memory_db.writer.datetime",
            "mindmemos.pipelines.add.vanilla_multimodal.vanilla_multimodal_add.datetime",
        ):
            monkeypatch.setattr(target, _search_config if target.endswith("get_config") else _FrozenDatetime)
        self.store = InMemoryVanillaStore()
        self.llm, self.embed, self.mm = RecordingLlm(), RecordingEmbed(), RecordingMmClient()
        self.images, self.recorder = RecordingImageStore(), RecordingRecorder()
        self.blobs = LocalBlobStore(tmp_path / "blobs")

    def add_pipeline(self, mode: str | None):
        common = dict(
            db_reader=self.store,
            db_writer=self.store,
            text_config=TEXT_CONFIG,
            consistency="fast",
            llm_client=self.llm,
            embed_client=self.embed,
            recorder=self.recorder,
        )
        if mode is None:
            return VanillaAddPipeline(**common)
        return VanillaMultimodalAddPipeline(
            **common,
            image_store=self.images,
            blob_store=self.blobs,
            image_loader=ImageLoader(BlobStoreConfig(), self.blobs),
            mm_embed_client=self.mm,
            multimodal_config=MultimodalConfig(mode=mode),
        )

    def search_pipeline(self, mode: str | None) -> SearchPipelineImpl:
        common = dict(db_reader=self.store, db_writer=self.store, text_config=TEXT_CONFIG, embed_client=self.embed)
        if mode is None:
            engine = VanillaSearchEngine(**common)
        else:
            engine = VanillaMultimodalSearchEngine(
                **common,
                image_store=self.images,
                mm_embed_client=self.mm,
                multimodal_config=MultimodalConfig(mode=mode),
                llm_client=self.llm,
                blob_store=self.blobs,
            )
        return SearchPipelineImpl(
            db_reader=self.store,
            db_writer=self.store,
            engines={"vanilla": engine},
            final_filter=None,
            rerank_client=None,
        )

    async def add_dialogues(self, mode: str | None, dialogues) -> list[dict]:
        add = self.add_pipeline(mode)
        results = []
        for index, messages in enumerate(dialogues):
            # Pin the request time: it defaults to the wall clock, which differs between any two runs.
            inp = AddPipelineInput(messages=messages, event_timestamp_ms=1717000000000 + index * 60_000)
            result = await add.add_sync(inp, _context(index), add_record_id=f"rec-{index}")
            results.append(result.model_dump(mode="json"))
        return results

    async def search(self, mode: str | None, queries, *, multimodal: bool = False) -> dict:
        search = self.search_pipeline(mode)
        embed_before, llm_before = len(self.embed.calls), len(self.llm.calls)
        self.store.reads.clear()
        results = []
        for query in queries:
            inp = SearchPipelineInput(query=query, search_pipeline="vanilla", top_k=5, multimodal=multimodal)
            results.append((await search.search(inp, _context(99))).model_dump(mode="json"))
        return {
            "results": results,
            "embed": self.embed.calls[embed_before:],
            "llm": self.llm.calls[llm_before:],
            "reads": list(self.store.reads),
        }


async def _run(mode: str | None, monkeypatch, tmp_path, *, multimodal: bool = False) -> dict:
    harness = _Harness(monkeypatch, tmp_path)
    add_results = await harness.add_dialogues(mode, DIALOGUES)
    searched = await harness.search(mode, QUERIES, multimodal=multimodal)
    return {
        "llm": harness.llm.calls,
        "embed": harness.embed.calls,
        "plans": _plans(harness.store),
        "add": add_results,
        "records": harness.recorder.completed,
        "search": searched["results"],
        "mm_calls": harness.mm.text_calls + [[n] for n in harness.mm.image_calls],
        "image_store_calls": harness.images.calls,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "multimodal"),
    [("off", False), ("off", True), ("basic", False), ("enhanced", False)],
    ids=["off", "off-flag-ignored", "basic", "enhanced"],
)
async def test_text_only_requests_match_vanilla_exactly(monkeypatch, tmp_path, mode: str, multimodal: bool) -> None:
    vanilla = await _run(None, monkeypatch, tmp_path / "vanilla")
    wrapped = await _run(mode, monkeypatch, tmp_path / "wrapped", multimodal=multimodal)

    assert vanilla["llm"], "the fake LLM must be exercised"
    assert vanilla["plans"], "memories must be written"
    assert any(result["memories"] for result in vanilla["search"]), "searches must return memories"
    for key in ("llm", "embed", "plans", "add", "records", "search"):
        assert wrapped[key] == vanilla[key], key
    # Nothing image-related happens for text-only requests.
    assert wrapped["mm_calls"] == []
    assert wrapped["image_store_calls"] == []


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
IMAGE_DIALOGUE = [
    {
        "role": "user",
        "content": "Look at the sofa I bought",
        "timestamp": 1717000300000,
        "images": [{"data": base64.b64encode(PNG).decode()}],
    },
    {"role": "assistant", "content": "It looks comfortable.", "timestamp": 1717000301000},
]
SEARCHES = [*QUERIES, "Which sofa did I buy?"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["basic", "enhanced"])
async def test_searches_that_are_not_multimodal_ignore_stored_images(monkeypatch, tmp_path, mode: str) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    await harness.add_dialogues(mode, [*DIALOGUES, IMAGE_DIALOGUE])
    assert harness.images.records, "the user has stored images"
    if mode == "enhanced":
        assert any(view.metadata.get("image_blob_ids") for view in harness.store.memories.values())
    mm_calls = (list(harness.mm.text_calls), list(harness.mm.image_calls))
    image_calls = list(harness.images.calls)

    vanilla = await harness.search(None, SEARCHES)
    wrapped = await harness.search(mode, SEARCHES)

    assert any(result["memories"] for result in vanilla["results"])
    assert wrapped == vanilla  # results, embedding requests, LLM calls and every DB read
    assert (harness.mm.text_calls, harness.mm.image_calls) == mm_calls
    assert harness.images.calls == image_calls

    multimodal = await harness.search(mode, SEARCHES[-1:], multimodal=True)
    assert harness.images.calls != image_calls, "a multimodal search does use the images"
    assert (
        multimodal["results"][0]["memories"][: len(vanilla["results"][-1]["memories"])]
        == (vanilla["results"][-1]["memories"])
    )


@pytest.mark.asyncio
async def test_visual_facts_stay_out_of_text_memory(monkeypatch, tmp_path) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    await harness.add_dialogues("enhanced", [*DIALOGUES, IMAGE_DIALOGUE])

    contents = [view.content for view in harness.store.memories.values()]
    assert "The user bought a new sofa." in contents
    assert not any("VISUAL" in content or "green velvet" in content for content in contents)
    [record] = harness.images.records.values()
    assert record.visual_facts == [VISUAL_FACT]
    assert record.description == "A green velvet sofa."
    [sofa] = [view for view in harness.store.memories.values() if view.content == "The user bought a new sofa."]
    assert sofa.metadata["image_blob_ids"] == [record.blob_id]
    assert record.anchored_memory_ids == [sofa.memory_id]
    assert list(harness.images.text_vectors) == [record.image_id]  # the image's text channel
    assert (
        "multimodal.add.image_text",
        [f"A green velvet sofa.\n{VISUAL_FACT}\nuser: Look at the sofa I bought\nassistant: It looks comfortable."],
    ) in harness.embed.calls


@pytest.mark.asyncio
async def test_concurrent_adds_of_the_same_image_keep_every_anchor(monkeypatch, tmp_path) -> None:
    """Two enhanced adds of the same image racing on ``_link_images`` must not lose an anchor."""

    class _RacyImageStore(RecordingImageStore):
        # Yield inside every read and write: unless the read-modify-write is striped the same way
        # as ``_index``, the two writes interleave and the later one drops the other's anchor.
        async def get_images(self, ctx, image_ids):
            await asyncio.sleep(0)
            return await super().get_images(ctx, image_ids)

        async def update_images(self, ctx, records, *, text_vectors=None):
            await asyncio.sleep(0)
            await super().update_images(ctx, records, text_vectors=text_vectors)

    class _PerCallLlm(RecordingLlm):
        # A distinct dialogue fact per concurrent add, so the two memories cannot be merged.
        def __init__(self) -> None:
            super().__init__()
            self.image_extractions = 0

        async def chat(self, *, task, messages, **kwargs):
            response = await super().chat(task=task, messages=messages, **kwargs)
            if isinstance(messages[-1].get("content"), list):
                self.image_extractions += 1
                response.parsed["memories"][0]["content"] = f"The user bought a new sofa. #{self.image_extractions}"
            return response

    harness = _Harness(monkeypatch, tmp_path)
    harness.llm = _PerCallLlm()
    harness.images = _RacyImageStore()
    add = harness.add_pipeline("enhanced")
    inp = AddPipelineInput(messages=IMAGE_DIALOGUE, event_timestamp_ms=1717000300000)

    await asyncio.gather(add.add_sync(inp, _context(1)), add.add_sync(inp.model_copy(), _context(2)))

    anchored = {m.memory_id for m in harness.store.memories.values() if m.metadata.get("image_blob_ids")}
    assert len(anchored) == 2, "each add must anchor its own dialogue fact to the image"
    [record] = harness.images.records.values()
    assert set(record.anchored_memory_ids) == anchored


@pytest.mark.asyncio
async def test_add_async_rejects_over_limit_images_before_storing(monkeypatch, tmp_path) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    add = VanillaMultimodalAddPipeline(
        db_reader=harness.store,
        db_writer=harness.store,
        text_config=TEXT_CONFIG,
        consistency="fast",
        llm_client=harness.llm,
        embed_client=harness.embed,
        recorder=harness.recorder,
        image_store=harness.images,
        blob_store=harness.blobs,
        image_loader=ImageLoader(BlobStoreConfig(), harness.blobs),
        mm_embed_client=harness.mm,
        multimodal_config=MultimodalConfig(mode="basic", max_images_per_request=1),
    )
    inp = AddPipelineInput(
        messages=[
            {
                "role": "user",
                "content": "look at these sofas",
                "timestamp": 1717000300000,
                "images": [
                    {"data": base64.b64encode(PNG).decode()},
                    {"data": base64.b64encode(PNG).decode()},
                ],
            }
        ],
        event_timestamp_ms=1717000300000,
    )

    with pytest.raises(BadRequestError) as caught:
        await add.add_async(inp, _context(3))

    assert caught.value.code == "add.too_many_images"
    # Rejected before anything is stored or queued.
    assert harness.images.calls == []
    assert not [path for path in (tmp_path / "blobs").rglob("*") if path.is_file()]


@pytest.mark.asyncio
async def test_image_only_enhanced_add_skips_extraction_loudly(monkeypatch, tmp_path) -> None:
    """Enhanced extraction needs a kept text turn; an image-only request must say it stored
    the pixels without reading them, not stay silent about it."""

    from structlog.testing import capture_logs

    harness = _Harness(monkeypatch, tmp_path)
    add = harness.add_pipeline("enhanced")
    inp = AddPipelineInput(
        messages=[
            {
                "role": "user",
                "content": "",
                "timestamp": 1717000300000,
                "images": [{"data": base64.b64encode(PNG).decode()}],
            }
        ],
        event_timestamp_ms=1717000300000,
    )

    with capture_logs() as logs:
        result = await add.add_sync(inp, _context(4))

    assert [m.operation for m in result.memories] == ["add"]  # the stored image only
    [record] = harness.images.records.values()
    assert record.description is None and record.visual_facts == []  # no VLM extraction ran
    assert any(entry["event"] == "multimodal_enhanced_extraction_skipped" for entry in logs)


def test_caller_supplied_image_strings_are_capped_at_write_time() -> None:
    from mindmemos.components.multimodal.messages import AttachedImage
    from mindmemos.pipelines.add.vanilla_multimodal.vanilla_multimodal_add import _occurrence, _source_url

    image = AttachedImage(ImageRef(data="eA==", image_id="i" * 600, caption="c" * 600), 0, "user", 0, "ctx")
    occurrence = _occurrence(image, _context(9))
    assert occurrence.caller_image_id == "i" * 500
    assert occurrence.caption == "c" * 500
    from_url = AttachedImage(ImageRef(url="http://x.example/" + "u" * 600), 0, "user", 0, "")
    assert _source_url([from_url]) == ("http://x.example/" + "u" * 600)[:500]


def test_the_image_loader_is_cached_per_pipeline(monkeypatch) -> None:
    # One loader per pipeline (add and search alike) so the HTTP pool of query-image fetches
    # is reused instead of being rebuilt per call.
    from mindmemos.pipelines import multimodal_deps

    monkeypatch.setattr(multimodal_deps, "get_config", lambda: SimpleNamespace(blob_store=BlobStoreConfig()))
    engine = VanillaMultimodalSearchEngine(
        db_reader=object(),
        db_writer=object(),
        text_config=TEXT_CONFIG,
        embed_client=object(),
        blob_store=LocalBlobStore("nowhere"),
        multimodal_config=MultimodalConfig(mode="enhanced"),
    )
    first = engine.image_loader()
    assert engine.image_loader() is first
    explicit = ImageLoader(BlobStoreConfig(), LocalBlobStore("nowhere"))
    add = VanillaMultimodalAddPipeline(
        db_reader=object(),
        db_writer=object(),
        text_config=TEXT_CONFIG,
        llm_client=object(),
        embed_client=object(),
        image_loader=explicit,
        multimodal_config=MultimodalConfig(mode="enhanced"),
    )
    assert add.image_loader() is explicit  # an injected loader is never replaced
