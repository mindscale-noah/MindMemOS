"""Batched trajectory experience dedup: recall union, verdict parsing, resolution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from mindmemos_lite.components.extractor.task_experience.dedup import ExperienceDeduplicator
from mindmemos_lite.typing import MemoryDbSearchHit, MemoryRequestContext, MemoryView, PreprocessedText


def _ctx() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-dedup",
        account_id="account-1",
        project_id="project-1",
        api_key_uuid="key-1",
        user_id="user-1",
    )


def _pp(text: str) -> PreprocessedText:
    return PreprocessedText(text=text, normalized_text=text, content_hash=f"hash-{text}", bm25_text=text)


def _hit(memory_id: str, content: str) -> MemoryDbSearchHit:
    return MemoryDbSearchHit(
        memory_id=memory_id,
        score=0.9,
        memory=MemoryView(
            memory_id=memory_id,
            project_id="project-1",
            content=content,
            mem_type="experience",
            status="active",
        ),
    )


class _FakeEmbed:
    def __init__(self) -> None:
        self.calls = 0
        self.batches: list[list[str]] = []

    async def embed(self, *, task, text):
        self.calls += 1
        texts = [text] if isinstance(text, str) else list(text)
        self.batches.append(texts)
        return SimpleNamespace(embeddings=[[float(len(item))] for item in texts])


class _FakePersistence:
    def __init__(self, hits_by_query: dict[str, list[MemoryDbSearchHit]]) -> None:
        self._hits = hits_by_query
        self.queries: list[str] = []

    async def search_dense(self, ctx, query, *, dense_vector):
        self.queries.append(query.query)
        return SimpleNamespace(hits=self._hits.get(query.query, []))


class _FakeLLM:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = 0
        self.payloads: list[dict] = []

    async def chat(self, *, task, messages, format_parser):
        self.calls += 1
        self.payloads.append(json.loads(messages[1]["content"]))
        return SimpleNamespace(parsed=self.payload)


def _deduplicator(hits_by_query, payload):
    embed = _FakeEmbed()
    persistence = _FakePersistence(hits_by_query)
    llm = _FakeLLM(payload)
    dedup = ExperienceDeduplicator(persistence=persistence, embed_client=embed, llm_client=llm)
    return dedup, embed, persistence, llm


@pytest.mark.asyncio
async def test_recall_many_embeds_every_candidate_in_one_call() -> None:
    dedup, embed, persistence, _ = _deduplicator({"a": [], "b": []}, None)

    recalled = await dedup.recall_many(_ctx(), [(0, "a"), (1, "b")])

    assert embed.calls == 1, "candidates must share a single embedding round trip"
    assert embed.batches[0] == ["a", "b"]
    assert persistence.queries == ["a", "b"]
    assert recalled == {0: [], 1: []}


@pytest.mark.asyncio
async def test_resolve_many_judges_the_batch_once_and_unions_recalled_hits() -> None:
    hits_by_query = {"a": [_hit("m1", "老经验1")], "b": [_hit("m1", "老经验1"), _hit("m2", "老经验2")]}
    payload = {
        "verdicts": [
            {"candidate_id": 0, "verdict": "different"},
            {"candidate_id": 1, "verdict": "same_with_delta", "match_index": 1, "merged_content": "合并后"},
        ]
    }
    dedup, _, _, llm = _deduplicator(hits_by_query, payload)

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a")), (1, "b", _pp("b"))], lang="zh")

    assert llm.calls == 1, "the whole batch must be judged in one call"
    sent = llm.payloads[0]
    assert [candidate["candidate_id"] for candidate in sent["candidates"]] == [0, 1]
    # m1 is recalled by both candidates but appears once in the shared array,
    # numbered by position so the model's match_index aligns with the input.
    assert [entry["memory_index"] for entry in sent["existing"]] == [0, 1]

    assert out[0].action == "create"
    assert out[1].action == "merge"
    assert out[1].target_memory_id == "m2"
    assert out[1].merged_content == "合并后"


@pytest.mark.asyncio
async def test_candidate_missing_from_the_verdicts_is_stored_as_new() -> None:
    payload = {"verdicts": [{"candidate_id": 0, "verdict": "different"}]}
    dedup, _, _, _ = _deduplicator({"a": [_hit("m1", "老经验1")], "b": [_hit("m1", "老经验1")]}, payload)

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a")), (1, "b", _pp("b"))], lang="zh")

    assert out[0].action == "create"
    # In production the missing candidate makes judge_many re-ask it alone and,
    # if it stays missing, only that candidate falls back to new; the parser-level
    # default must agree.
    assert out[1].action == "create", "a skipped candidate must never be dropped"


class _QueuedLLM:
    """Returns one payload per call in order, recording what was sent each round."""

    def __init__(self, payloads) -> None:
        self._payloads = list(payloads)
        self.calls = 0
        self.sent_candidates: list[list[dict]] = []

    async def chat(self, *, task, messages, format_parser):
        self.calls += 1
        self.sent_candidates.append(json.loads(messages[1]["content"])["candidates"])
        index = min(self.calls - 1, len(self._payloads) - 1)
        return SimpleNamespace(parsed=self._payloads[index])


@pytest.mark.asyncio
async def test_judge_many_retries_the_whole_batch_and_keeps_partial_verdicts() -> None:
    llm = _QueuedLLM(
        [
            {"verdicts": [{"candidate_id": 0, "verdict": "different"}]},  # round 1 drops candidate 1
            {"verdicts": [{"candidate_id": 1, "verdict": "same_no_delta", "match_index": 0}]},  # round 2
        ]
    )
    dedup = ExperienceDeduplicator(persistence=object(), embed_client=None, llm_client=llm)

    out = await dedup.judge_many(_ctx(), [(0, "a"), (1, "b")], [_hit("m1", "老经验1")], lang="zh")

    # The second round re-sends the WHOLE batch (blind retry), and candidate 0's
    # first verdict survives it while candidate 1 is picked up.
    assert llm.calls == 2
    assert [entry["candidate_id"] for entry in llm.sent_candidates[0]] == [0, 1]
    assert [entry["candidate_id"] for entry in llm.sent_candidates[1]] == [0, 1]
    assert out[0].verdict == "different"
    assert out[1].verdict == "same_no_delta"
    assert out[1].match_index == 0


@pytest.mark.asyncio
async def test_judge_many_defaults_candidates_still_missing_after_retries_to_new() -> None:
    # This payload can never cover candidate 1, so every round answers only candidate 0.
    llm = _QueuedLLM([{"verdicts": [{"candidate_id": 0, "verdict": "different"}]}])
    dedup = ExperienceDeduplicator(persistence=object(), embed_client=None, llm_client=llm)

    out = await dedup.judge_many(_ctx(), [(0, "a"), (1, "b")], [_hit("m1", "老经验1")], lang="zh")

    assert llm.calls == 2  # re-sent the whole batch, candidate 1 still absent, gave up
    assert out[0].verdict == "different"
    assert out[1].verdict == "different"


@pytest.mark.asyncio
async def test_out_of_range_match_index_falls_back_to_create() -> None:
    payload = {"verdicts": [{"candidate_id": 0, "verdict": "same_with_delta", "match_index": 9, "merged": "x"}]}
    dedup, _, _, _ = _deduplicator({"a": [_hit("m1", "老经验1")]}, payload)

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a"))], lang="zh")

    assert out[0].action == "create"


@pytest.mark.asyncio
async def test_no_recalled_hits_skips_the_llm_call_entirely() -> None:
    dedup, _, _, llm = _deduplicator({"a": []}, {"verdicts": []})

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a"))], lang="zh")

    assert llm.calls == 0, "with nothing to match against there is no reason to ask the model"
    assert out[0].action == "create"


@pytest.mark.asyncio
async def test_same_no_delta_reuses_the_matched_memory() -> None:
    payload = {"verdicts": [{"candidate_id": 0, "verdict": "same_no_delta", "match_index": 0}]}
    dedup, _, _, _ = _deduplicator({"a": [_hit("m1", "老经验1")]}, payload)

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a"))], lang="zh")

    assert out[0].action == "reuse"
    assert out[0].target_memory_id == "m1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"candidate_id": True, "verdict": "different"}, id="bool-id"),
        pytest.param({"candidate_id": "0", "verdict": "different"}, id="string-id"),
        pytest.param({"candidate_id": 0, "verdict": "banana"}, id="unknown-verdict"),
        pytest.param({"candidate_id": 0, "verdict": "same_no_delta", "match_index": "²"}, id="non-decimal-id"),
        pytest.param({"candidate_id": 0, "verdict": "same_no_delta", "match_index": None}, id="null-index"),
        "not-a-dict",
    ],
)
async def test_malformed_verdict_entries_never_lose_a_candidate(entry) -> None:
    """A bad entry must degrade to 'new', not raise and not drop the candidate."""

    dedup, _, _, _ = _deduplicator({"a": [_hit("m1", "老经验1")]}, {"verdicts": [entry]})

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a"))], lang="zh")

    assert out[0].action == "create"


@pytest.mark.asyncio
async def test_judge_failure_keeps_every_candidate_as_new() -> None:
    class _ExplodingLLM:
        async def chat(self, *, task, messages, format_parser):
            raise RuntimeError("provider down")

    dedup = ExperienceDeduplicator(
        persistence=_FakePersistence({"a": [_hit("m1", "老经验1")], "b": [_hit("m1", "老经验1")]}),
        embed_client=_FakeEmbed(),
        llm_client=_ExplodingLLM(),
    )

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a")), (1, "b", _pp("b"))], lang="zh")

    assert [out[0].action, out[1].action] == ["create", "create"]


@pytest.mark.asyncio
async def test_judge_is_skipped_when_no_llm_client_is_configured() -> None:
    dedup = ExperienceDeduplicator(
        persistence=_FakePersistence({"a": [_hit("m1", "老经验1")]}),
        embed_client=_FakeEmbed(),
        llm_client=None,
    )

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a"))], lang="zh")

    assert out[0].action == "create"


@pytest.mark.asyncio
async def test_zero_recall_candidate_is_created_without_an_llm_call() -> None:
    # candidate 1 recalls nothing, so it must be stored as new without consuming a
    # judge slot (old resolve() created it directly when its hits were empty).
    dedup, _, _, llm = _deduplicator({"a": [_hit("m1", "老经验1")], "b": []}, {"verdicts": []})

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a")), (1, "b", _pp("b"))], lang="zh")

    assert llm.calls == 1
    assert [entry["candidate_id"] for entry in llm.payloads[-1]["candidates"]] == [0]
    assert out[0].action == "create"
    assert out[1].action == "create"


@pytest.mark.asyncio
async def test_match_outside_the_candidates_own_recall_falls_back_to_create() -> None:
    # candidate 0 recalls only m1, but the shared union also carries m2 (recalled
    # by candidate 1). A model match pointing at m2 must not merge into it.
    hits = {"a": [_hit("m1", "老经验1")], "b": [_hit("m2", "老经验2")]}
    payload = {"verdicts": [{"candidate_id": 0, "verdict": "same_with_delta", "match_index": 1, "merged_content": "合并"}]}
    dedup, _, _, _ = _deduplicator(hits, payload)

    out = await dedup.resolve_many(_ctx(), [(0, "a", _pp("a")), (1, "b", _pp("b"))], lang="zh")

    assert out[0].action == "create", "must not merge into a memory it never recalled"
    assert out[1].action == "create"
