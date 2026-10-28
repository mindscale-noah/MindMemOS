"""Multimodal vanilla, mode basic: storage, deduplication, degraded mode and the text-to-image channel."""

from __future__ import annotations

import base64
import hashlib

import pytest
from mindmemos.components.multimodal import ImageLoader
from mindmemos.config import BlobStoreConfig, MultimodalConfig, TextProcessingConfig
from mindmemos.errors import BadRequestError
from mindmemos.infra.blob import LocalBlobStore
from mindmemos.pipelines.add.vanilla_multimodal import VanillaMultimodalAddPipeline, image_point_id
from mindmemos.pipelines.search.vanilla_multimodal import VanillaMultimodalSearchEngine
from mindmemos.pipelines.search.vanilla_multimodal.engine import _image_filter
from mindmemos.typing import (
    AddPipelineInput,
    AddPipelineSyncResult,
    ImageSearchHit,
    MemoryAddEventItem,
    MemoryRequestContext,
    MemorySearchItem,
    SearchPipelineInput,
)

TEXT_CONFIG = TextProcessingConfig(bm25_use_spacy_lemma=False, spacy_en_model="x", spacy_zh_model="y")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PNG_B = b"\x89PNG\r\n\x1a\n" + b"\x01" * 64


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _context(request_id: str = "req-1", session_id: str = "s1") -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id=request_id,
        account_id="acc",
        project_id="proj",
        api_key_uuid="key",
        user_id="u1",
        session_id=session_id,
    )


class FakeMm:
    model = "fake-mm"
    dimensions = 3

    def __init__(self) -> None:
        self.images: list[int] = []
        self.texts: list[list[str]] = []

    async def embed_images(self, images, *, task="mm.image"):
        self.images.append(len(images))
        return [[1.0, 0.0, 0.0] for _ in images]

    async def embed_texts(self, texts, *, task="mm.text"):
        self.texts.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeImageStore:
    def __init__(self) -> None:
        self.records = {}

    async def get_images(self, ctx, image_ids):
        return [self.records[i] for i in image_ids if i in self.records]

    async def upsert_images(self, ctx, items):
        for record, vector in items:
            assert len(vector) == 3
            self.records[record.image_id] = record

    async def update_images(self, ctx, records):
        for record in records:
            assert record.image_id in self.records
            self.records[record.image_id] = record

    async def search_images(self, ctx, vector, *, filters=None, limit=5, score_threshold=None):
        self.last_filters = filters
        return [ImageSearchHit(record=r, score=0.3) for r in list(self.records.values())[:limit]]


class StubVanillaAdd(VanillaMultimodalAddPipeline):
    """The multimodal wrapper whose vanilla parent is replaced by a recorder of the text input it receives."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vanilla_inputs: list[AddPipelineInput] = []

    async def _vanilla_add(self, inp):
        self.vanilla_inputs.append(inp)
        return AddPipelineSyncResult(
            status="ok", memories=[MemoryAddEventItem(operation="add", content="text fact", mem_type="fact")]
        )


def _pipeline(tmp_path, *, mm=None, store=None):
    blobs = LocalBlobStore(tmp_path / "blobs")
    pipeline = StubVanillaAdd(
        db_reader=object(),
        db_writer=object(),
        recorder=object(),
        llm_client=None,
        embed_client=None,
        text_config=TEXT_CONFIG,
        image_store=store or FakeImageStore(),
        blob_store=blobs,
        image_loader=ImageLoader(BlobStoreConfig(), blobs),
        mm_embed_client=mm,
        multimodal_config=MultimodalConfig(mode="basic"),
    )
    return pipeline, blobs


@pytest.fixture(autouse=True)
def _stub_vanilla(monkeypatch):
    async def fake_super_add_sync(self, inp, context, *, add_record_id=None):
        return await self._vanilla_add(inp)

    monkeypatch.setattr("mindmemos.pipelines.add.vanilla.VanillaAddPipeline.add_sync", fake_super_add_sync)


@pytest.mark.asyncio
async def test_images_are_stored_indexed_and_removed_from_the_vanilla_input(tmp_path) -> None:
    mm, store = FakeMm(), FakeImageStore()
    pipeline, blobs = _pipeline(tmp_path, mm=mm, store=store)
    inp = AddPipelineInput(
        messages=[
            {"role": "user", "content": "Look at my new sofa", "images": [{"data": _b64(PNG), "image_id": "IMG1"}]},
            {"role": "assistant", "content": "It is a green velvet sofa."},
            {"role": "user", "content": "", "images": [{"data": _b64(PNG_B)}]},
        ],
        event_timestamp_ms=1717000000000,
    )

    result = await pipeline.add_sync(inp, _context())

    # vanilla sees the text exactly as a text-only client would send it.
    [text_inp] = pipeline.vanilla_inputs
    assert [m.model_dump() for m in text_inp.messages] == [
        {"role": "user", "content": "Look at my new sofa", "timestamp": None},
        {"role": "assistant", "content": "It is a green velvet sofa.", "timestamp": None},
    ]
    assert mm.images == [2]
    assert len(store.records) == 2
    record = store.records[image_point_id("proj", "u1", hashlib.sha256(PNG).hexdigest())]
    assert record.occurrences[0].context == "user: Look at my new sofa\nassistant: It is a green velvet sofa."
    assert record.occurrences[0].caller_image_id == "IMG1"
    assert record.session_ids == ["s1"]
    assert (await blobs.get("proj", record.blob_id))[1] == PNG
    assert [event.memory_type for event in result.memories] == ["fact", "image", "image"]


@pytest.mark.asyncio
async def test_same_image_again_adds_an_occurrence_without_reembedding(tmp_path) -> None:
    mm, store = FakeMm(), FakeImageStore()
    pipeline, _ = _pipeline(tmp_path, mm=mm, store=store)
    message = {"role": "user", "content": "the sofa", "images": [{"data": _b64(PNG)}]}

    await pipeline.add_sync(AddPipelineInput(messages=[message]), _context("r1", "s1"))
    result = await pipeline.add_sync(AddPipelineInput(messages=[message]), _context("r2", "s2"))

    assert mm.images == [1]
    [record] = store.records.values()
    assert record.session_ids == ["s1", "s2"]
    assert record.request_ids == ["r1", "r2"]
    assert result.memories[-1].operation == "reinforcement"


@pytest.mark.asyncio
async def test_without_mm_router_images_are_only_stored(tmp_path) -> None:
    store = FakeImageStore()
    pipeline, blobs = _pipeline(tmp_path, mm=None, store=store)
    inp = AddPipelineInput(messages=[{"role": "user", "content": "hi", "images": [{"data": _b64(PNG)}]}])

    result = await pipeline.add_sync(inp, _context())

    assert store.records == {}
    assert await blobs.stat("proj", hashlib.sha256(PNG).hexdigest()) is not None
    assert result.memories[-1].memory_type == "image"


@pytest.mark.asyncio
async def test_too_many_images_are_rejected(tmp_path) -> None:
    pipeline, _ = _pipeline(tmp_path, mm=FakeMm())
    pipeline._explicit_multimodal_config = MultimodalConfig(mode="basic", max_images_per_request=1)
    inp = AddPipelineInput(
        messages=[{"role": "user", "content": "x", "images": [{"data": _b64(PNG)}, {"data": _b64(PNG_B)}]}]
    )
    with pytest.raises(Exception, match="at most 1 images"):
        await pipeline.add_sync(inp, _context())


@pytest.mark.asyncio
async def test_async_add_queues_blob_references_not_bytes(tmp_path, monkeypatch) -> None:
    pipeline, blobs = _pipeline(tmp_path, mm=FakeMm())
    queued = []

    async def fake_super_add_async(self, inp, context, *, add_record_id=None, record_metadata=None):
        queued.append(inp)

    monkeypatch.setattr("mindmemos.pipelines.add.vanilla.VanillaAddPipeline.add_async", fake_super_add_async)
    inp = AddPipelineInput(
        messages=[{"role": "user", "content": "x", "images": [{"data": _b64(PNG), "image_id": "A"}]}]
    )

    await pipeline.add_async(inp, _context())

    [ref] = queued[0].messages[0].images
    assert ref.data is None and ref.image_id == "A"
    assert (await blobs.get("proj", ref.blob_id))[1] == PNG


def _engine(*, mm, store, cfg=None):
    return VanillaMultimodalSearchEngine(
        db_reader=object(),
        db_writer=object(),
        embed_client=object(),
        text_config=TEXT_CONFIG,
        image_store=store,
        mm_embed_client=mm,
        multimodal_config=cfg or MultimodalConfig(mode="basic"),
    )


TEXT_RESULT = MemorySearchItem(id="m1", memory="text fact", last_update_at="2024-01-01 00:00:00")


@pytest.mark.asyncio
async def test_image_results_follow_the_text_results(tmp_path) -> None:
    mm, store = FakeMm(), FakeImageStore()
    pipeline, _ = _pipeline(tmp_path, mm=mm, store=store)
    await pipeline.add_sync(
        AddPipelineInput(
            messages=[{"role": "user", "content": "my sofa", "images": [{"data": _b64(PNG), "image_id": "I"}]}]
        ),
        _context(),
    )
    engine = _engine(mm=mm, store=store)

    items = await engine.augment_results(
        SearchPipelineInput(query="sofa colour", filters={"user_id": "u1"}, search_pipeline="vanilla", multimodal=True),
        _context(),
        [TEXT_RESULT],
    )

    assert items[0] == TEXT_RESULT
    image = items[1]
    assert image.memory_type == "image"
    assert image.memory == "[image] user: my sofa"
    assert image.metadata["blob_id"]
    assert image.metadata["caller_image_ids"] == ["I"]
    assert mm.texts[-1] == ["sofa colour"]


@pytest.mark.asyncio
async def test_image_channel_is_skipped_when_unavailable_or_filtered_on_text_only_fields() -> None:
    store = FakeImageStore()
    inp = SearchPipelineInput(query="q", search_pipeline="vanilla", multimodal=True)
    assert await _engine(mm=None, store=store).augment_results(inp, _context(), [TEXT_RESULT]) == [TEXT_RESULT]
    off = MultimodalConfig(mode="basic", image_search=False)
    assert await _engine(mm=FakeMm(), store=store, cfg=off).augment_results(inp, _context(), [TEXT_RESULT]) == [
        TEXT_RESULT
    ]
    filtered = inp.model_copy(update={"filters": {"mem_type": "fact"}})
    mm = FakeMm()
    assert await _engine(mm=mm, store=store).augment_results(filtered, _context(), [TEXT_RESULT]) == [TEXT_RESULT]
    assert mm.texts == []


def test_image_filter_keeps_session_and_user_conditions() -> None:
    parsed = _image_filter({"AND": [{"user_id": "u1"}, {"session_id": {"in": ["s1", "s2"]}}]})
    assert parsed is not None
    assert _image_filter({"mem_type": "fact"}) is None
    assert _image_filter(None) is not None


@pytest.mark.asyncio
async def test_mode_off_rejects_images_and_leaves_text_to_vanilla(tmp_path) -> None:
    mm, store = FakeMm(), FakeImageStore()
    pipeline, blobs = _pipeline(tmp_path, mm=mm, store=store)
    pipeline._explicit_multimodal_config = MultimodalConfig(mode="off")
    inp = AddPipelineInput(messages=[{"role": "user", "content": "x", "images": [{"data": _b64(PNG)}]}])

    with pytest.raises(BadRequestError) as error:
        await pipeline.add_sync(inp, _context())
    with pytest.raises(BadRequestError):
        await pipeline.add_async(inp, _context())

    assert error.value.code == "add.images_unsupported"
    assert pipeline.vanilla_inputs == [] and mm.images == [] and store.records == {}
    assert await blobs.stat("proj", hashlib.sha256(PNG).hexdigest()) is None
    await pipeline.add_sync(AddPipelineInput(messages=[{"role": "user", "content": "x"}]), _context())
    assert len(pipeline.vanilla_inputs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "basic", "enhanced"])
async def test_only_multimodal_searches_touch_images(mode: str) -> None:
    mm, store = FakeMm(), FakeImageStore()
    engine = _engine(mm=mm, store=store, cfg=MultimodalConfig(mode=mode))
    plain = SearchPipelineInput(query="sofa", search_pipeline="vanilla")

    assert engine.is_multimodal(plain) is False
    assert engine.owns_agentic(plain.model_copy(update={"agentic": True})) is False
    assert await engine.augment_results(plain, _context(), [TEXT_RESULT]) == [TEXT_RESULT]
    assert mm.texts == [] and mm.images == []

    flagged = plain.model_copy(update={"multimodal": True})
    with_image = plain.model_copy(update={"query_images": [{"data": _b64(PNG)}]})
    assert engine.is_multimodal(flagged) is (mode != "off")
    assert engine.is_multimodal(with_image) is (mode != "off")
