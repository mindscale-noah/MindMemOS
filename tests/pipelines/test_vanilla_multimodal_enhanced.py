"""Multimodal vanilla, mode enhanced: VLM extraction (dialogue vs. visual facts), anchoring, hybrid image
search, image-recalled facts, evidence images, agentic refine."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.components.multimodal.extraction import (
    IMAGE_BLOB_IDS_KEY,
    ExtractionImage,
    ImageVisuals,
    MultimodalMemoryExtractor,
    RequestImages,
    request_images,
)
from mindmemos.components.multimodal.messages import AttachedImage
from mindmemos.config import MultimodalConfig, TextProcessingConfig
from mindmemos.infra.blob import StoredBlob
from mindmemos.pipelines.add.vanilla_multimodal.vanilla_multimodal_add import _images_by_text_index
from mindmemos.pipelines.search.vanilla_multimodal import VanillaMultimodalSearchEngine
from mindmemos.pipelines.search.vanilla_multimodal.engine import _mix, _strong, _to_search_item
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
from omegaconf import OmegaConf

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
                {"ref_id": "m1", "content": "The user bought a new sofa.", "source_refs": ["s1"]},
                {"ref_id": "m2", "content": "The user had pasta for lunch.", "source_refs": ["s2"]},
            ],
            "image_memories": [
                {
                    "image": 1,
                    "description": "A green velvet three-seat sofa (IMAGE 1).",
                    "facts": ["On 2024-01-05 the user showed a green velvet sofa.", "The sofa has three seats."],
                }
            ],
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
    assert "IMAGE 1" in user[-1]["text"] and "image_memories" in user[-1]["text"]  # closing reminder
    # Dialogue facts only: visual facts never become memories.
    sofa, lunch = result.memories
    assert sofa.metadata[IMAGE_BLOB_IDS_KEY] == ["b" * 64]  # the reply describing the image
    assert IMAGE_BLOB_IDS_KEY not in lunch.metadata
    assert sofa.metadata["extractor"] == "multimodal_vlm"
    assert images.visuals == {
        "b" * 64: ImageVisuals(
            description="A green velvet three-seat sofa.",
            facts=["On 2024-01-05 the user showed a green velvet sofa.", "The sofa has three seats."],
        )
    }


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


def _record(blob: str, memory_ids=(), description=None, facts=()) -> ImageRecord:
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
        visual_facts=list(facts),
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
    def __init__(self, hits, text_hits=()):
        self.hits, self.text_hits = hits, list(text_hits)

    async def search_images(self, ctx, vector, *, channel="image", filters=None, limit=5, score_threshold=None):
        return self.text_hits if channel == "text" else self.hits

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

    return VanillaMultimodalSearchEngine(
        blob_store=LocalBlobStore("/nonexistent-blob-root"),
        db_reader=reader,
        db_writer=object(),
        embed_client=object(),
        text_config=TEXT_CONFIG,
        image_store=FakeImages(hits),
        mm_embed_client=FakeMm(),
        llm_client=llm,
        multimodal_config=MultimodalConfig(mode="enhanced", **cfg),
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
        SearchPipelineInput(query="Ethos ad background", search_pipeline="vanilla", multimodal=True),
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
        SearchPipelineInput(query="Ethos background?", search_pipeline="vanilla", multimodal=True), CTX, fast
    )

    assert [o.id for o in out if o.memory_type != "image"] == ["m2", "m1"]
    assert [o.metadata["blob_id"] for o in out if o.memory_type == "image"] == [ad]
    assert "Images found" in json.dumps(llm.seen[1][-1]["content"])

    broken = _engine(FakeReader([]), hits, llm=ScriptedLlm([]))
    inp = SearchPipelineInput(query="q", search_pipeline="vanilla", multimodal=True)
    assert await broken.agentic_refine(inp, CTX, fast) == fast


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
        SearchPipelineInput(query="route with parked trucks", search_pipeline="vanilla", multimodal=True), CTX, []
    )

    images = [o for o in out if o.memory_type == "image"]
    assert [i.metadata["blob_id"][0] for i in images][:4] == ["2", "3", "4", "5"]  # window, then the fill
    assert images[0].memory.startswith("[image 2 of 5 in conversation S3]")
    assert images[2].metadata["evidence"] == "similar" and images[0].metadata["evidence"] == "session"


def test_image_scores_mix_normalized_text_and_image_similarity() -> None:
    records = [_record(chr(ord("a") + i) * 64) for i in range(12)]
    # Image similarity is flat (near-identical screenshots); text similarity has one clear match.
    by_image = [ImageSearchHit(record=r, score=0.16 - i * 0.001) for i, r in enumerate(records)]
    by_text = [ImageSearchHit(record=records[7], score=0.71)]
    by_text += [ImageSearchHit(record=r, score=0.40 - i * 0.005) for i, r in enumerate(records) if i != 7]

    def order(hits):
        return [hit.record.blob_id[0] for hit in hits]

    assert order(_mix([], by_image, 0.5))[:3] == ["a", "b", "c"]  # no text vectors yet: image ranking
    assert order(_mix(by_text, by_image, 0.5))[0] == "h"  # the clear text match wins
    assert order(_mix(by_text, by_image, 0.0))[:3] == ["a", "b", "c"]
    partial = _mix(by_text[:1], by_image[:2], 0.5)  # an image missing from a channel scores 0 there
    assert set(order(partial)) == {"a", "b", "h"}


def test_image_items_carry_description_and_visual_facts() -> None:
    record = _record("a" * 64, description="Ethos Water ad", facts=["The ad has a white background."])
    item = _to_search_item(ImageSearchHit(record=record, score=0.3))
    assert item.memory == "[image] Ethos Water ad\n- The ad has a white background."
    assert item.metadata["visual_facts"] == ["The ad has a white background."]


class FakeEmbed:
    def __init__(self):
        self.calls = []

    async def embed(self, task, text, **kwargs):
        from mindmemos.llm import EmbeddingResponse

        self.calls.append((task, text))
        return EmbeddingResponse(embeddings=[[0.1, 0.2] for _ in (text if isinstance(text, list) else [text])])


@pytest.mark.asyncio
async def test_text_channel_finds_images_by_their_visual_facts() -> None:
    ad, other = "d" * 64, "o" * 64
    by_image = [ImageSearchHit(record=_record(other), score=0.15), ImageSearchHit(record=_record(ad), score=0.14)]
    by_text = [ImageSearchHit(record=_record(ad, facts=["The Ethos ad is white."]), score=0.8)]
    embed = FakeEmbed()
    engine = _engine(FakeReader([]), by_image, evidence_images=2, session_expansion_images=0)
    engine._image_store = FakeImages(by_image, text_hits=by_text)
    engine._explicit_embed_client = embed

    hits = await engine.image_hits(
        SearchPipelineInput(query="Ethos ad colour", search_pipeline="vanilla", multimodal=True), CTX, engine._config()
    )

    assert [h.record.blob_id for h in hits.ranked] == [ad, other]
    assert [h.record.blob_id for h in hits.by_image] == [other, ad]  # raw image similarity kept apart
    assert embed.calls == [("search.query", "Ethos ad colour")]


@pytest.mark.asyncio
async def test_archived_facts_are_not_recalled_through_images() -> None:
    sofa = "s" * 64
    archived = _view("m9", "Deleted fact.", blobs=[sofa]).model_copy(update={"status": "archived"})
    hits = [
        ImageSearchHit(record=_record(sofa, memory_ids=["m9"]), score=0.25),
        *[ImageSearchHit(record=_record(str(i) * 64), score=0.14) for i in range(5)],
    ]
    out = await _engine(FakeReader([archived]), hits).augment_results(
        SearchPipelineInput(query="sofa", search_pipeline="vanilla", multimodal=True), CTX, []
    )
    assert all(item.memory_type == "image" for item in out)


class ReversingReranker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, documents, top_n, **kwargs):
        from mindmemos.typing import RerankHit, RerankResponse

        self.calls.append((query, list(documents), top_n))
        order = list(reversed(range(len(documents))))[:top_n]
        return RerankResponse(results=[RerankHit(index=i, relevance_score=1.0 - k / 10) for k, i in enumerate(order)])


@pytest.mark.asyncio
async def test_visual_facts_come_back_as_reranked_results_and_images_keep_descriptions() -> None:
    ad, sofa = "d" * 64, "s" * 64
    hits = [
        ImageSearchHit(
            record=_record(ad, description="Ethos ad", facts=["Ethos ad is white.", "Logo is green."]), score=0.25
        ),
        ImageSearchHit(record=_record(sofa, description="Sofa", facts=["Sofa is red."]), score=0.2),
    ]
    reranker = ReversingReranker()
    engine = _engine(FakeReader([]), hits, visual_fact_limit=2, session_expansion_images=0)
    engine._explicit_rerank_client = reranker

    out = await engine.augment_results(
        SearchPipelineInput(query="Ethos ad colour", search_pipeline="vanilla", multimodal=True, rerank=True), CTX, []
    )

    facts = [o for o in out if o.memory_type == "visual_fact"]
    # Reranker order, limit 2; each fact names the image it was read from.
    assert [f.memory for f in facts] == ["[image: Sofa] Sofa is red.", "[image: Ethos ad] Logo is green."]
    assert reranker.calls == [
        (
            "Ethos ad colour",
            ["[image: Ethos ad] Ethos ad is white.", "[image: Ethos ad] Logo is green.", "[image: Sofa] Sofa is red."],
            2,
        )
    ]
    assert facts[0].metadata["blob_id"] == sofa and len({f.id for f in facts}) == 2
    images = [o for o in out if o.memory_type == "image"]
    assert [i.memory for i in images] == ["[image] Ethos ad", "[image] Sofa"]  # facts are separate results
    assert out.index(facts[-1]) < out.index(images[0])

    without_rerank = await engine.augment_results(
        SearchPipelineInput(query="Ethos ad colour", search_pipeline="vanilla", multimodal=True), CTX, []
    )
    assert [o.memory for o in without_rerank if o.memory_type == "visual_fact"] == [
        "[image: Ethos ad] Ethos ad is white.",
        "[image: Ethos ad] Logo is green.",
    ]


@pytest.mark.asyncio
async def test_session_expansion_seeds_from_the_best_image_similarity_match() -> None:
    def in_session(blob, session, score):
        record = _record(blob, description=blob[0]).model_copy(
            update={"occurrences": [ImageOccurrence(session_id=session, event_timestamp_ms=1)]}
        )
        return ImageSearchHit(record=record, score=score)

    by_image = [in_session("a" * 64, "S3", 0.2), in_session("b" * 64, "S6", 0.1)]
    by_text = [in_session("b" * 64, "S6", 0.9), in_session("a" * 64, "S3", 0.1)]
    seen_sessions = []

    class SessionImages(FakeImages):
        async def list_session_images(self, ctx, session_id, *, user_id=None, limit=50):
            seen_sessions.append(session_id)
            return []

    engine = _engine(FakeReader([]), by_image, image_text_weight=0.9)
    engine._image_store = SessionImages(by_image, text_hits=by_text)
    engine._explicit_embed_client = FakeEmbed()

    await engine.augment_results(
        SearchPipelineInput(query="paint test", search_pipeline="vanilla", multimodal=True), CTX, []
    )

    assert seen_sessions == ["S3"]


@pytest.mark.asyncio
async def test_agent_memory_search_also_finds_visual_facts() -> None:
    ad = "d" * 64
    hits = [
        ImageSearchHit(
            record=_record(ad, description="Ethos ad", facts=["The Ethos ad background is white."]), score=0.3
        )
    ]
    llm = ScriptedLlm(
        [
            '{"tool": "search_memories", "args": {"query": "Ethos ad background"}}',
            '{"tool": "submit", "args": {"facts": ["F3"], "images": []}}',
        ]
    )
    engine = _engine(FakeReader([]), hits, llm=llm)
    # The server's config is an OmegaConf object, not a dataclass instance.
    engine._explicit_multimodal_config = OmegaConf.structured(
        MultimodalConfig(mode="enhanced", session_expansion_images=0)
    )
    searched = []

    async def dialogue_facts(inp, context, *, options=None):
        searched.append(inp.query)
        return [_item("m7", "The user discussed the Ethos campaign.")]

    engine.search_candidates = dialogue_facts
    fast = [_item("m1", "sofa fact")]

    out = await engine.agentic_refine(
        SearchPipelineInput(query="Ethos background?", search_pipeline="vanilla", multimodal=True), CTX, fast
    )

    observation = json.dumps(llm.seen[1][-1]["content"])
    assert "Visual facts:" in observation and "F3" in observation and "background is white" in observation
    assert searched == ["Ethos ad background"]
    assert out[0].memory == "[image: Ethos ad] The Ethos ad background is white."
    assert out[0].memory_type == "visual_fact"


def test_visual_facts_name_their_image_briefly() -> None:
    from mindmemos.pipelines.search.vanilla_multimodal.engine import _image_label, _visual_fact_text

    record = _record(
        "a" * 64,
        description="Daily Glucose Dashboard for Maya Patel dated 2025-04-01 labeled Overview. Six metric boxes.",
        facts=["Late Window 57% with circular chart bottom right"],
    )
    assert _visual_fact_text(record, 0) == (
        "[image: Daily Glucose Dashboard for Maya Patel dated 2025-04-01 labeled Overview] "
        "Late Window 57% with circular chart bottom right"
    )
    long = record.model_copy(update={"description": "word " * 60})
    assert len(_image_label(long)) <= 103 and _image_label(long).endswith("...")
    assert _visual_fact_text(record.model_copy(update={"description": None}), 0) == record.visual_facts[0]


def test_agent_keeps_only_a_few_unsubmitted_visual_facts() -> None:
    from mindmemos.pipelines.search.vanilla_multimodal.agentic import UNSUBMITTED_VISUAL_FACTS, MultimodalAgent

    agent = MultimodalAgent(engine=None, llm_client=None, cfg=MultimodalConfig(mode="enhanced"))
    texts = [_item("m1", "dialogue fact")]
    visual = [
        MemorySearchItem(id=f"v{i}", memory=f"visual {i}", memory_type="visual_fact", last_update_at="")
        for i in range(10)
    ]
    handles = {item.id: agent._fact_handle(item) for item in [*texts, *visual]}

    out = agent._result({"facts": [handles["v9"]], "images": []}, [*texts, *visual], [])

    assert [o.id for o in out] == ["v9", "m1", *[f"v{i}" for i in range(UNSUBMITTED_VISUAL_FACTS)]]
