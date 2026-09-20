"""Trajectory task+experience builder: single-call extraction, dedup, truncation."""

from __future__ import annotations

import pytest
from mindmemos_lite.components.extractor.task_experience import (
    ExperienceResolution,
    ExtractedExperienceCandidate,
    TrajectoryExperienceBuilder,
)
from mindmemos_lite.components.extractor.task_experience.builder import _truncate_tool_message
from mindmemos_lite.components.text import TextPreprocessor
from mindmemos_lite.config import TextProcessingConfig, TrajectoryAddConfig
from mindmemos_lite.typing import (
    AddPipelineInput,
    DialogueMessage,
    FileMessage,
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


class _RecordingExtractor:
    """Records every call and emits two candidates so dedup has something to fold."""

    def __init__(self) -> None:
        self.calls = 0
        self.turns: list[dict] = []

    async def extract(self, task_text, turns, lang, context):
        self.calls += 1
        self.turns = list(turns)
        indices = [int(turn["message_index"]) for turn in turns]
        return [
            ExtractedExperienceCandidate(
                ref_id=f"e{offset}",
                content=(
                    f"在无外网环境中, 直接使用 pip 安装包会失败(变体{offset}). 需要先确认网络或使用离线镜像。"
                ),
                confidence=0.9,
                source_message_indices=indices,
            )
            for offset in range(2)
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


def _turns(count: int) -> list[DialogueMessage]:
    turns: list[DialogueMessage] = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        text = f"第{index}轮: 用户尝试通过 pip 安装依赖, 但没有外网访问, 出现连接被拒绝的报错, 需要改用离线方式。"
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


def _builder(extractor, dedup) -> TrajectoryExperienceBuilder:
    text_config = TextProcessingConfig(bm25_use_spacy_lemma=False)
    preprocessor = TextPreprocessor(text_config, entity_extractor=_NoEntities())
    return TrajectoryExperienceBuilder(
        text_preprocessor=preprocessor,
        extractor=extractor,
        deduplicator=dedup,
        vectorizer=_NoopVectorizer(),
    )


@pytest.mark.asyncio
async def test_trajectory_builder_extracts_whole_trace_in_one_call_and_dedups() -> None:
    messages = _turns(16)
    extractor = _RecordingExtractor()
    builder = _builder(extractor, _ReuseFirstDedup())

    inp = AddPipelineInput(messages=messages, task="安装 pandas", mode="sync")
    plan, events, _update_commands = await builder.build(inp, _context(), config=TrajectoryAddConfig())

    # The whole trajectory reaches the extractor exactly once: no chunk planning,
    # and every message keeps its original position for source refs.
    assert extractor.calls == 1
    assert [turn["message_index"] for turn in extractor.turns] == list(range(len(messages)))

    # exactly one task entity, and only ONE experience created despite two candidates
    assert len(plan.entities) == 1
    assert len(plan.memories) == 1, f"expected dedup to keep a single node, got {len(plan.memories)}"

    task_entity_id = plan.entities[0].entity_id
    task_edges = [r for r in plan.relationships if r.rel_type == "TASK_EXPERIENCE"]
    assert len(task_edges) == 2
    assert {r.source.node_id for r in task_edges} == {task_entity_id}
    assert {r.target.node_id for r in task_edges} == {memory.memory_id for memory in plan.memories}

    assert len({event.memory_id for event in events}) == 1


@pytest.mark.asyncio
async def test_trajectory_builder_keeps_head_and_tail_of_oversized_tool_message() -> None:
    oversized = "HEAD-MARKER " + ("头 " * 500) + ("日志行 " * 20000) + ("尾 " * 500) + " TAIL-MARKER"
    messages = [
        DialogueMessage(role="user", content="帮我装 pandas"),
        DialogueMessage(role="assistant", content="我先执行安装命令"),
        DialogueMessage(role="tool", content=oversized),
        DialogueMessage(role="assistant", content="安装完成"),
    ]
    extractor = _RecordingExtractor()
    builder = _builder(extractor, _ReuseFirstDedup())

    cfg = TrajectoryAddConfig(tool_message_head_tokens=200, tool_message_tail_tokens=200)
    inp = AddPipelineInput(messages=messages, task="安装 pandas", mode="sync")
    await builder.build(inp, _context(), config=cfg)

    text_by_index = {turn["message_index"]: turn["text"] for turn in extractor.turns}
    tool_text = text_by_index[2]

    assert len(tool_text) < len(oversized)
    assert tool_text.startswith("HEAD-MARKER")
    assert tool_text.endswith("TAIL-MARKER")
    assert "truncated" in tool_text

    # non-tool messages are never trimmed
    assert text_by_index[0] == "帮我装 pandas"
    assert text_by_index[1] == "我先执行安装命令"
    assert text_by_index[3] == "安装完成"


@pytest.mark.asyncio
async def test_trajectory_builder_keeps_original_positions_when_messages_are_dropped() -> None:
    messages = [
        DialogueMessage(role="user", content="任务描述"),
        DialogueMessage(role="system", content="系统提示，不是证据"),
        DialogueMessage(role="assistant", content="开始执行"),
        DialogueMessage(role="user", content="   "),  # 空文本
        FileMessage(file_name="log.txt", file_path="/tmp/log.txt"),  # 附件，不是证据
        DialogueMessage(role="tool", content="工具输出"),
    ]
    extractor = _RecordingExtractor()
    builder = _builder(extractor, _ReuseFirstDedup())

    inp = AddPipelineInput(messages=messages, task="安装 pandas", mode="sync")
    await builder.build(inp, _context(), config=TrajectoryAddConfig())

    # Surviving messages keep pointing at their position in the *original*
    # request, so the index list is sparse rather than re-numbered from zero.
    assert [turn["message_index"] for turn in extractor.turns] == [0, 2, 5]
    assert [turn["role"] for turn in extractor.turns] == ["user", "assistant", "tool"]


def test_truncate_tool_message_boundaries() -> None:
    long_text = ("日志行 " * 500) + "END"
    cfg = TrajectoryAddConfig(tool_message_head_tokens=5, tool_message_tail_tokens=5)

    # non-tool roles are never trimmed, whatever their size
    assert _truncate_tool_message(long_text, "assistant", cfg) == long_text
    assert _truncate_tool_message(long_text, "user", cfg) == long_text

    # empty text and a disabled budget both pass through untouched
    assert _truncate_tool_message("", "tool", cfg) == ""
    disabled = TrajectoryAddConfig(tool_message_head_tokens=0, tool_message_tail_tokens=0)
    assert _truncate_tool_message(long_text, "tool", disabled) == long_text

    # text that already fits the combined budget is returned verbatim
    fits = "词 " * 10
    assert _truncate_tool_message(fits, "tool", cfg) == fits

    # genuinely oversized text keeps both edges and reports the gap
    trimmed = _truncate_tool_message(long_text, "tool", cfg)
    assert len(trimmed) < len(long_text)
    assert trimmed.startswith("日志行")
    assert trimmed.endswith("END")
    assert "truncated" in trimmed
