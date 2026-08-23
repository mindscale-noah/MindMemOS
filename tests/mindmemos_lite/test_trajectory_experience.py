"""Trajectory task+experience builder: chunking reuse and cross-chunk dedup."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from mindmemos_lite.components.chunker import MessageChunker
from mindmemos_lite.components.extractor.task_experience import (
    ExperienceResolution,
    ExtractedExperienceCandidate,
    TrajectoryExperienceBuilder,
)
from mindmemos_lite.components.text import SparseVectorEncoder, TextPreprocessor
from mindmemos_lite.config import MessageChunkerConfig, TextProcessingConfig, TrajectoryAddConfig
from mindmemos_lite.typing import (
    AddPipelineInput,
    DialogueMessage,
    MemoryRequestContext,
    PreprocessedText,
)


class _NoEntities:
    def extract(self, _text, _lang):
        return []

    def extract_many(self, texts, _langs):
        return [[] for _ in texts]


class _NoopVectorizer:
    async def vectorize_many(self, items, consistency: str = "fast", *, batch_size: int = 32):
        return [], [False] * len(items)

    async def vectorize_entities(
        self, entities, *, memories_by_entity=None, consistency: str = "fast", batch_size: int = 32
    ):
        return [], False


class _PerChunkExtractor:
    """Emits one experience candidate per chunk call; wording differs across chunks."""

    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, task_text, turns, lang, context):
        self.calls += 1
        content = (
            f"在无外网环境中, 直接使用 pip 安装包会失败(call={self.calls}). "
            "需要先确认网络或使用离线镜像。"
        )
        indices = [int(turn["message_index"]) for turn in turns]
        return [
            ExtractedExperienceCandidate(
                ref_id=f"e{self.calls}",
                content=content,
                confidence=0.9,
                source_message_indices=indices,
            )
        ]


class _ReuseFirstDedup:
    """Simulates the LLM judge: once one experience exists in this import, reuse it."""

    def __init__(self) -> None:
        self.creates = 0

    async def resolve(
        self,
        ctx: MemoryRequestContext,
        candidate_text: str,
        preprocessed: PreprocessedText,
        *,
        lang: str,
        import_experiences: list[tuple[str, str]] | None = None,
    ) -> ExperienceResolution:
        for memory_id, content in import_experiences or ():
            if content and "pip" in content and "pip" in candidate_text:
                return ExperienceResolution(action="reuse", target_memory_id=memory_id, preprocessed=preprocessed)
        self.creates += 1
        memory_id = f"exp-{self.creates}"
        return ExperienceResolution(action="create", target_memory_id=memory_id, preprocessed=preprocessed)


def _tiny_chunker() -> MessageChunkerConfig:
    return MessageChunkerConfig(
        chunk_soft_token_budget=600,
        chunk_hard_token_budget=1200,
        turn_hard_token_budget=600,
        history_soft_token_budget=100,
        history_hard_token_budget=150,
        history_min_turn_count=1,
        template_tokens=50,
        recall_budget=50,
        output_headroom=100,
        compaction_soft_token_budget=500,
        compaction_head_tokens=100,
        compaction_tail_tokens=100,
    )


def _turns(count: int) -> list[DialogueMessage]:
    turns: list[DialogueMessage] = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        # ~140 chars per line so a modest trace already exceeds the tiny budget.
        text = f"第{index}轮: 用户尝试通过 pip 安装依赖, 但没有外网访问, 出现连接被拒绝的报错, 需要改用离线方式。" * 1
        turns.append(DialogueMessage(role=role, content=text))
    return turns


def _context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-trajectory",
        account_id="account-1",
        project_id="project-1",
        api_key_uuid="key-1",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_trajectory_builder_splits_long_input_and_dedups_across_chunks() -> None:
    messages = _turns(16)  # far above the tiny chunk soft budget
    # 1) chunker alone really splits.
    chunks = await MessageChunker(_tiny_chunker(), llm_client=None).split(messages)
    assert len(chunks.chunks) > 1, f"expected multiple chunks, got {len(chunks.chunks)}"

    text_config = TextProcessingConfig(bm25_use_spacy_lemma=False)
    preprocessor = TextPreprocessor(text_config, entity_extractor=_NoEntities())
    extractor = _PerChunkExtractor()
    dedup = _ReuseFirstDedup()
    builder = TrajectoryExperienceBuilder(
        text_preprocessor=preprocessor,
        extractor=extractor,
        deduplicator=dedup,
        vectorizer=_NoopVectorizer(),
        chunker_config=_tiny_chunker(),
        llm_client=None,
    )

    inp = AddPipelineInput(messages=messages, task="安装 pandas", mode="sync")
    plan, events, _update_commands = await builder.build(inp, _context(), config=TrajectoryAddConfig())

    # extractor ran once per chunk
    assert extractor.calls == len(chunks.chunks)
    assert extractor.calls > 1, "test is not exercising the multi-chunk path"

    # exactly one task entity, and only ONE experience created despite many chunks
    assert len(plan.entities) == 1
    assert len(plan.memories) == 1, f"expected cross-chunk dedup to keep a single node, got {len(plan.memories)}"

    # every chunk produced a task->experience edge to that single node
    task_entity_id = plan.entities[0].entity_id
    task_edges = [r for r in plan.relationships if r.rel_type == "TASK_EXPERIENCE"]
    assert len(task_edges) == len(chunks.chunks)
    assert {r.source.node_id for r in task_edges} == {task_entity_id}
    assert {r.target.node_id for r in task_edges} == {memory.memory_id for memory in plan.memories}

    # events report create + reuse for the remaining chunks
    operations = {event.operation for event in events}
    assert "add" in operations
    assert len({event.memory_id for event in events}) == 1