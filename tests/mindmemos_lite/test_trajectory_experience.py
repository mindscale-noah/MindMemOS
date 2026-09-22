"""Trajectory task+experience builder: single-call extraction, dedup, truncation."""

from __future__ import annotations

import pytest
from mindmemos_lite import prompts
from mindmemos_lite.components.extractor.task_experience.extractor import TrajectoryExperienceExtractor
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
)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_extraction_prompt_defaults_and_empty_plan(lang) -> None:
    expected = (
        prompts.EXPERIENCE_EXTRACTION_SYSTEM_PROMPT_ZH if lang == "zh" else prompts.EXPERIENCE_EXTRACTION_SYSTEM_PROMPT
    )
    assert prompts.get_trajectory_experience_prompt(lang) == expected
    assert prompts.get_trajectory_experience_prompt(lang, extract_type="experience") == expected
    with pytest.raises(ValueError, match="plan extraction prompt is empty"):
        prompts.get_trajectory_experience_prompt(lang, extract_type="plan")


@pytest.mark.asyncio
async def test_plan_prompt_is_request_local_and_keeps_output_contract(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(prompts, "PLAN_EXTRACTION_SYSTEM_PROMPT", "Extract plans as experiences JSON.")
    monkeypatch.setattr(prompts, "PLAN_EXTRACTION_SYSTEM_PROMPT_ZH", "Chinese plan prompt.")
    calls = []

    class Client:
        async def chat(self, **kwargs):
            calls.append(kwargs["messages"])
            return SimpleNamespace(
                parsed={"experiences": [{"content": "Reusable plan", "source_message_indices": [0]}]}
            )

    extractor = TrajectoryExperienceExtractor(llm_client=Client())
    turns = [{"message_index": 0, "role": "user", "text": "Task input"}]
    for lang, extract_type in [("en", "plan"), ("zh", "plan"), ("en", "experience")]:
        candidates = await extractor.extract("Task", turns, lang, _context(), extract_type=extract_type)
        assert candidates[0].content == "Reusable plan"
        assert candidates[0].source_message_indices == [0]
    assert [call[0]["content"] for call in calls] == [
        "Extract plans as experiences JSON.",
        "Chinese plan prompt.",
        prompts.EXPERIENCE_EXTRACTION_SYSTEM_PROMPT,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("extract_type", ["unknown", "", None, {}])
async def test_invalid_extract_type_is_not_swallowed(extract_type) -> None:
    extractor = TrajectoryExperienceExtractor(llm_client=None)
    with pytest.raises(ValueError, match="metadata.extract_type"):
        await extractor.extract("Task", [], "en", _context(), extract_type=extract_type)


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
        self.extract_types = []

    async def extract(self, task_text, turns, lang, context, *, extract_type="experience"):
        self.calls += 1
        self.turns = list(turns)
        self.extract_types.append(extract_type)
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


class _StubDedup:
    """Stand-in batched judge: records the batch and creates one node per candidate.

    ``reuse_first`` mimics the LLM deciding that a later candidate is the same
    experience as an earlier one, collapsing the batch onto a single node.
    """

    def __init__(self, *, reuse_first: bool = False) -> None:
        self.calls = 0
        self.batches: list[list[str]] = []
        self._reuse_first = reuse_first

    async def resolve_many(self, ctx, candidates, *, lang):
        self.calls += 1
        self.batches.append([text for _, text, _ in candidates])
        resolutions: dict[int, ExperienceResolution] = {}
        first_target: str | None = None
        for candidate_id, _text, preprocessed in candidates:
            if self._reuse_first and first_target is not None:
                resolutions[candidate_id] = ExperienceResolution(
                    action="reuse", target_memory_id=first_target, preprocessed=preprocessed
                )
                continue
            target = f"exp-{candidate_id}"
            first_target = first_target or target
            resolutions[candidate_id] = ExperienceResolution(
                action="create", target_memory_id=target, preprocessed=preprocessed
            )
        return resolutions


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
@pytest.mark.parametrize("metadata", [{}, {"extract_type": "experience"}, {"extract_type": "plan"}])
async def test_trajectory_builder_extracts_whole_trace_in_one_call_and_dedups(metadata) -> None:
    messages = _turns(16)
    extractor = _RecordingExtractor()
    dedup = _StubDedup(reuse_first=True)
    builder = _builder(extractor, dedup)

    inp = AddPipelineInput(messages=messages, task="安装 pandas", mode="sync", metadata=metadata)
    plan, events, _update_commands = await builder.build(inp, _context(), config=TrajectoryAddConfig())

    # The whole trajectory reaches the extractor exactly once: no chunk planning,
    # and every message keeps its original position for source refs.
    assert extractor.calls == 1
    assert extractor.extract_types == [metadata.get("extract_type", "experience")]
    assert [turn["message_index"] for turn in extractor.turns] == list(range(len(messages)))

    # Both candidates are judged in ONE batched call rather than one call each.
    assert dedup.calls == 1
    assert len(dedup.batches[0]) == 2

    # exactly one task entity, and only ONE experience node despite two candidates
    assert len(plan.entities) == 1
    assert len(plan.memories) == 1, f"expected the batch to collapse onto one node, got {len(plan.memories)}"

    task_entity_id = plan.entities[0].entity_id
    task_edges = [r for r in plan.relationships if r.rel_type == "TASK_EXPERIENCE"]
    assert len(task_edges) == 2
    assert {r.source.node_id for r in task_edges} == {task_entity_id}
    assert {r.target.node_id for r in task_edges} == {memory.memory_id for memory in plan.memories}

    assert len({event.memory_id for event in events}) == 1


class _ByteIdenticalExtractor:
    """Emits the same experience text twice, so only the hash guard can fold it."""

    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, task_text, turns, lang, context, *, extract_type="experience"):
        self.calls += 1
        return [
            ExtractedExperienceCandidate(
                ref_id=f"e{offset}",
                content="同一条经验: 离线环境需要改用本地包源。",
                confidence=0.9,
                source_message_indices=[0],
            )
            for offset in range(2)
        ]


@pytest.mark.asyncio
async def test_trajectory_builder_folds_byte_identical_candidates_before_judging() -> None:
    extractor = _ByteIdenticalExtractor()
    dedup = _StubDedup()
    builder = _builder(extractor, dedup)

    inp = AddPipelineInput(messages=_turns(4), task="安装 pandas", mode="sync")
    plan, events, _update_commands = await builder.build(inp, _context(), config=TrajectoryAddConfig())

    # The duplicate is dropped before the judge, which therefore sees one candidate.
    assert dedup.calls == 1
    assert len(dedup.batches[0]) == 1

    assert len(plan.memories) == 1
    # Both candidates still produce their own task edge onto the shared node.
    task_edges = [r for r in plan.relationships if r.rel_type == "TASK_EXPERIENCE"]
    assert len(task_edges) == 2
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
    builder = _builder(extractor, _StubDedup())

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
    builder = _builder(extractor, _StubDedup())

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


def test_pipeline_wires_experience_recall_top_k_into_dedup() -> None:
    """The dedup recall width must follow the trajectory config, not the default."""
    from mindmemos_lite.config import build_config
    from mindmemos_lite.pipeline import create_pipeline

    cfg = build_config(config_path="config/mindmemos_lite/dev.example.yaml")
    cfg.algo_config.trajectory.experience_recall_top_k = 2
    pipeline = create_pipeline(type="add", name="trajectory_add", config=cfg, persistence=object())

    assert pipeline._builder._deduplicator._top_k == 2
