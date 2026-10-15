"""vanilla_mm enhanced mode: VLM extraction with anchoring, anchored evidence, agentic refine."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.components.multimodal.extraction import (
    IMAGE_BLOB_IDS_KEY,
    ExtractionImage,
    MultimodalMemoryExtractor,
    RequestImages,
    request_images,
)
from mindmemos.components.multimodal.messages import AttachedImage
from mindmemos.config import TextProcessingConfig, VanillaMmConfig
from mindmemos.infra.blob import StoredBlob
from mindmemos.pipelines.add.vanilla_mm.vanilla_mm_add import _images_by_text_index
from mindmemos.pipelines.search.vanilla_mm import VanillaMmSearchEngine
from mindmemos.pipelines.search.vanilla_mm.engine import _strong
from mindmemos.typing import (
    AddPipelineInput,
    ExtractionEnvelope,
    ImageOccurrence,
    ImageRecord,
    ImageRef,
    ImageSearchHit,
    MemoryRequestContext,
    MemorySearchItem,
    MemoryView,
    SearchPipelineInput,
    TurnMessageRef,
)

CTX = MemoryRequestContext(request_id="r", account_id="a", project_id="p", api_key_uuid="k", user_id="u")
NOW = datetime(2024, 1, 5, tzinfo=UTC)
TEXT_CONFIG = TextProcessingConfig(bm25_use_spacy_lemma=False, spacy_en_model="x", spacy_zh_model="y")


class FakeInner:
    def __init__(self):
        self.calls = 0

    async def extract_from_envelope(self, envelope, preprocessed_texts, context):
        self.calls += 1
        from mindmemos.components.extractor.vanilla.memory import MemoryExtractionResult

        return MemoryExtractionResult()


class FakeVlm:
    def __init__(self, parsed):
        self.parsed, self.messages = parsed, None

    async def chat(self, *, task, messages, format_parser=None, **kwargs):
        self.messages = messages
        return SimpleNamespace(parsed=self.parsed, content=json.dumps(self.parsed))


def _envelope():
    return ExtractionEnvelope(
        boundary="complete",
        extractable_messages=[
            TurnMessageRef(text="Here is my new sofa", role="user", message_index=0, timestamp=1),
            TurnMessageRef(text="A green velvet sofa.", role="assistant", message_index=1, timestamp=1),
            TurnMessageRef(text="Lunch was pasta", role="user", message_index=2, timestamp=1),
        ],
    )


def _preprocessed(envelope):
    return [SimpleNamespace(normalized_text=m.text, lang="en") for m in envelope.extractable_messages]


@pytest.mark.asyncio
async def test_chunk_with_images_uses_one_vlm_call_and_anchors_facts() -> None:
    vlm = FakeVlm(
        {
            "memories": [
                {"ref_id": "m1", "content": "The user's sofa is green velvet.", "source_refs": ["s1"]},
                {"ref_id": "m2", "content": "The user had pasta for lunch.", "source_refs": ["s2"]},
            ],
            "image_descriptions": [{"image": 1, "description": "A green velvet three-seat sofa."}],
        }
    )
    inner = FakeInner()
    extractor = MultimodalMemoryExtractor(inner, llm_client=vlm)
    images = RequestImages(by_message={0: [ExtractionImage("b" * 64, "image/png", b"\x89PNG")]})

    with request_images(images):
        envelope = _envelope()
        result = await extractor.extract_from_envelope(envelope, _preprocessed(envelope), CTX)

    assert inner.calls == 0
    user = vlm.messages[1]["content"]
    assert user[1]["text"] == "IMAGE 1 (evidence_index 0):"
    assert user[2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert vlm.messages[0]["content"].startswith("## Images in this chunk")  # takes precedence over vanilla rules
    payload = json.loads(user[0]["text"])
    assert payload["extractable"][0]["images"] == ["IMAGE 1"]
    assert "IMAGE 1" in user[-1]["text"]  # closing reminder
    sofa, lunch = result.memories
    assert sofa.metadata[IMAGE_BLOB_IDS_KEY] == ["b" * 64]  # the reply describing the image
    assert IMAGE_BLOB_IDS_KEY not in lunch.metadata
    assert sofa.metadata["extractor"] == "vanilla_mm_vlm"
    assert images.descriptions == {"b" * 64: "A green velvet three-seat sofa."}


@pytest.mark.asyncio
async def test_chunks_without_images_go_to_vanilla_untouched() -> None:
    inner, vlm = FakeInner(), FakeVlm({})
    extractor = MultimodalMemoryExtractor(inner, llm_client=vlm)
    envelope = _envelope()
    await extractor.extract_from_envelope(envelope, _preprocessed(envelope), CTX)  # no request images at all
    with request_images(RequestImages(by_message={7: [ExtractionImage("c" * 64, "image/png", b"x")]})):
        await extractor.extract_from_envelope(envelope, _preprocessed(envelope), CTX)  # images elsewhere
    assert inner.calls == 2 and vlm.messages is None


def test_image_only_turns_hand_their_images_to_the_next_kept_turn() -> None:
    inp = AddPipelineInput(
        messages=[
            {"role": "user", "content": "", "images": [{"data": "eA=="}]},
            {"role": "assistant", "content": "Nice picture."},
            {"role": "user", "content": "And this?", "images": [{"data": "eQ=="}]},
        ]
    )
    stored = [
        (StoredBlob("a" * 64, "image/png", 1), b"a", AttachedImage(ImageRef(data="eA=="), 0, "user", 0, "")),
        (StoredBlob("b" * 64, "image/png", 1), b"b", AttachedImage(ImageRef(data="eQ=="), 2, "user", 0, "")),
    ]
    mapping = _images_by_text_index(inp, stored)
    assert {k: [i.blob_id for i in v] for k, v in mapping.items()} == {0: ["a" * 64], 1: ["b" * 64]}


def _record(blob: str, memory_ids=(), description=None) -> ImageRecord:
    return ImageRecord(
        image_id=f"img-{blob[:4]}",
        blob_id=blob,
        mime_type="image/png",
        size_bytes=1,
        project_id="p",
        user_id="u",
        created_at=NOW,
        updated_at=NOW,
        description=description,
        anchored_memory_ids=list(memory_ids),
        occurrences=[ImageOccurrence(context="user: the sofa", event_timestamp_ms=1704412800000)],
    )


def test_strong_hits_must_stand_out_from_the_rest() -> None:
    flat = [ImageSearchHit(record=_record(str(i) * 64), score=0.14 + i * 0.001) for i in range(6)]
    assert _strong(flat, 0.03) == []
    peaked = [*flat, ImageSearchHit(record=_record("9" * 64), score=0.25)]
    assert [h.record.blob_id for h in _strong(peaked, 0.03)] == ["9" * 64]


class FakeReader:
    def __init__(self, views):
        self.views = {v.memory_id: v for v in views}

    async def get_memories(self, ctx, ids):
        return [self.views[i] for i in ids if i in self.views]


class FakeImages:
    def __init__(self, hits):
        self.hits = hits

    async def search_images(self, ctx, vector, *, filters=None, limit=5, score_threshold=None):
        return self.hits

    async def get_images(self, ctx, ids):
        return [h.record for h in self.hits if h.record.image_id in ids]


class FakeMm:
    model, dimensions = "m", 3

    async def embed_texts(self, texts, *, task=""):
        return [[1.0, 0, 0] for _ in texts]

    async def embed_images(self, images, *, task=""):
        return [[1.0, 0, 0] for _ in images]


def _view(memory_id, content, blobs=()):
    return MemoryView(
        memory_id=memory_id,
        project_id="p",
        content=content,
        user_id="u",
        created_at=NOW,
        mem_type="fact",
        status="active",
        metadata={IMAGE_BLOB_IDS_KEY: list(blobs)} if blobs else {},
    )


def _engine(reader, hits, llm=None, **cfg):
    from mindmemos.infra.blob import LocalBlobStore

    return VanillaMmSearchEngine(
        blob_store=LocalBlobStore("/nonexistent-blob-root"),
        db_reader=reader,
        db_writer=object(),
        embed_client=object(),
        text_config=TEXT_CONFIG,
        image_store=FakeImages(hits),
        mm_embed_client=FakeMm(),
        llm_client=llm,
        vanilla_mm_config=VanillaMmConfig(enhanced=True, **cfg),
    )


def _item(memory_id, text):
    return MemorySearchItem(id=memory_id, memory=text, last_update_at="2024-01-05 00:00:00")


@pytest.mark.asyncio
async def test_enhanced_search_returns_anchored_images_and_image_recalled_facts() -> None:
    sofa, ad = "s" * 64, "d" * 64
    reader = FakeReader(
        [
            _view("m1", "The sofa is green velvet.", blobs=[sofa]),
            _view("m9", "The Ethos Water ad has a white background.", blobs=[ad]),
        ]
    )
    hits = [
        ImageSearchHit(record=_record(ad, memory_ids=["m9"], description="Ethos Water ad"), score=0.25),
        *[ImageSearchHit(record=_record(str(i) * 64), score=0.14) for i in range(5)],
    ]
    engine = _engine(reader, hits)

    out = await engine.augment_results(
        SearchPipelineInput(query="Ethos ad background", search_pipeline="vanilla_mm"),
        CTX,
        [_item("m1", "The sofa is green velvet.")],
    )

    assert [o.id for o in out[:2]] == ["m1", "m9"]
    images = [o for o in out if o.memory_type == "image"]
    # Anchored images of the facts interleave with the best similarity hits, capped at evidence_images.
    assert [(i.metadata["blob_id"], i.metadata["evidence"]) for i in images[:3]] == [
        (sofa, "anchored"),
        (ad, "similar"),
        ("0" * 64, "similar"),
    ]
    assert len(images) == 6
    assert images[1].memory == "[image] Ethos Water ad"


def test_request_local_image_labels_never_reach_memory() -> None:
    from mindmemos.components.multimodal.extraction import strip_labels

    assert strip_labels("IMAGE 1 shows checkpoint 1") == "The image shows checkpoint 1"
    assert strip_labels("A bus (IMAGE 2) on Main St") == "A bus on Main St"


class ScriptedLlm:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    async def chat(self, *, task, messages, **kwargs):
        self.seen.append(list(messages))
        return SimpleNamespace(content=self.replies.pop(0))


@pytest.mark.asyncio
async def test_agentic_refine_puts_submitted_evidence_first_and_falls_back_on_failure() -> None:
    ad = "d" * 64
    hits = [ImageSearchHit(record=_record(ad, memory_ids=["m9"], description="Ethos Water ad"), score=0.25)]
    llm = ScriptedLlm(
        [
            '{"tool": "search_images", "args": {"query": "Ethos Water ad"}}',
            '{"tool": "submit", "args": {"facts": ["F2"], "images": ["I1"]}}',
        ]
    )
    engine = _engine(FakeReader([]), hits, llm=llm)
    fast = [_item("m1", "sofa fact"), _item("m2", "Ethos fact")]

    out = await engine.agentic_refine(
        SearchPipelineInput(query="Ethos background?", search_pipeline="vanilla_mm"), CTX, fast
    )

    assert [o.id for o in out if o.memory_type != "image"] == ["m2", "m1"]
    assert [o.metadata["blob_id"] for o in out if o.memory_type == "image"] == [ad]
    assert "Images found" in json.dumps(llm.seen[1][-1]["content"])

    broken = _engine(FakeReader([]), hits, llm=ScriptedLlm([]))
    assert await broken.agentic_refine(SearchPipelineInput(query="q", search_pipeline="vanilla_mm"), CTX, fast) == fast


@pytest.mark.asyncio
async def test_best_image_match_brings_its_session_siblings_in_order() -> None:
    def in_session(blob, minute, score=0.0):
        record = _record(blob).model_copy(
            update={"occurrences": [ImageOccurrence(session_id="S3", event_timestamp_ms=minute * 60_000)]}
        )
        return ImageSearchHit(record=record, score=score)

    route = [in_session(str(i) * 64, i) for i in range(1, 6)]  # checkpoints 1..5 of one route
    hit = route[3].model_copy(update={"score": 0.17})

    class SessionImages(FakeImages):
        async def list_session_images(self, ctx, session_id, *, user_id=None, limit=50):
            assert session_id == "S3"
            return [h.record for h in route]

    engine = _engine(FakeReader([]), [hit], session_expansion_images=4, evidence_images=6)
    engine._image_store = SessionImages([hit, *route])

    out = await engine.augment_results(
        SearchPipelineInput(query="route with parked trucks", search_pipeline="vanilla_mm"), CTX, []
    )

    images = [o for o in out if o.memory_type == "image"]
    assert [i.metadata["blob_id"][0] for i in images][:4] == ["2", "3", "4", "5"]  # window, then the fill
    assert images[0].memory.startswith("[image 2 of 5 in conversation S3]")
    assert images[2].metadata["evidence"] == "similar" and images[0].metadata["evidence"] == "session"
