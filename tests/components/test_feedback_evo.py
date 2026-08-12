"""Tests for the feedback-driven self-evolution components."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from types import SimpleNamespace

from mindmemos.components.feedback_evo import FeedbackEvoCollector
from mindmemos.components.feedback_evo.evolution import (
    EvolutionExecutor,
    EvolutionPlanner,
    _apply_changes,
    _evolvable_config_view,
    _filter_changes_by_threshold,
    _signal_confidence,
    _valid_signals,
    build_initial_evolution_state,
    ensure_evolution_state,
    is_evolvable_path,
)
from mindmemos.components.feedback_evo.collector import _extract_recalled_memories
from mindmemos.components.extractor.feedback_evo.memory import _normalize_feedback_evo_extraction
from mindmemos.typing import (
    EvolutionResult,
    EvolutionState,
    FeedbackEvoEvent,
    MemoryRequestContext,
    ParameterChange,
)


def _event(event_id: str, paths: list[str], confidence: float = 1.0) -> FeedbackEvoEvent:
    return FeedbackEvoEvent(
        event_id=event_id,
        account_id="acc",
        project_id="proj_1",
        api_key_uuid="key",
        submitted_at=datetime.now(UTC),
        signals=[
            {"evolvable_path": path, "round_index": i, "confidence": confidence}
            for i, path in enumerate(paths)
        ],
    )


def _ctx() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc",
        project_id="proj_1",
        api_key_uuid="key",
        user_id="user-1",
    )


class _FakeLLM:
    def __init__(self, parsed) -> None:
        self._parsed = parsed

    async def chat(self, **kwargs):
        del kwargs
        return SimpleNamespace(parsed=self._parsed)


class _FakeEventStore:
    def __init__(self) -> None:
        self.events: list[FeedbackEvoEvent] = []

    async def append(self, context, event) -> None:
        del context
        self.events.append(event)


def _messages(count: int) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_collector_extracts_recalled_memories_from_round():
    llm = _FakeLLM(
        [
            {"round_index": 0, "evolvable_path": "search_config.top_k", "confidence": 0.9, "reason": "r1"},
        ]
    )
    store = _FakeEventStore()
    collector = FeedbackEvoCollector(
        llm_client=llm,  # type: ignore[arg-type]
        event_store=store,  # type: ignore[arg-type]
    )

    messages = _messages(25)
    messages[1] = {
        "role": "assistant",
        "content": "recall",
        "tool_calls": [
            {"name": "retrieve_learnings", "result": {"learnings": ["memory-A", "memory-B"]}}
        ],
    }
    event = await collector.collect(_ctx(), task_messages=messages, task_id="t1")

    signal = event.signals[0]
    assert signal["round_messages"] == messages[0:20]
    assert signal["related_memories"] == ["memory-A", "memory-B"]


@pytest.mark.asyncio
async def test_collector_related_memories_fallback_to_task_recall():
    llm = _FakeLLM(
        [{"round_index": 0, "evolvable_path": "search_config.top_k", "confidence": 0.9, "reason": "r"}]
    )
    store = _FakeEventStore()
    collector = FeedbackEvoCollector(llm_client=llm, event_store=store)  # type: ignore[arg-type]

    # The recall tool call lives in window 1; the signal is in window 0.
    messages = _messages(25)
    messages[21] = {
        "role": "assistant",
        "content": "recall",
        "tool_calls": [
            {"name": "retrieve_learnings", "result": {"learnings": ["memory-X"]}}
        ],
    }
    event = await collector.collect(_ctx(), task_messages=messages, task_id="t1")

    assert event.signals[0]["related_memories"] == ["memory-X"]


def test_extract_recalled_memories_from_tool_results():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "recall",
            "tool_calls": [
                {"name": "retrieve_learnings", "result": {"learnings": ["m1", "m2"]}},
                {"name": "get_order", "result": {"order_id": "x"}},
            ],
        },
        {
            "role": "assistant",
            "content": "recall again",
            "tool_calls": [
                {"name": "retrieve_learnings", "result": {"learnings": ["m2", "m3"]}}
            ],
        },
    ]

    assert _extract_recalled_memories(messages) == ["m1", "m2", "m3"]
    assert _extract_recalled_memories([]) == []


def test_build_initial_evolution_state_seeds_only_evolvable_fields():
    from mindmemos.config import init_config

    init_config(config_name="product", config_path="config/mindmemos/dev.example.yaml")
    state = build_initial_evolution_state("proj_1")

    assert state.version == 1
    assert state.is_current is True
    assert set(state.add_config) == {"extraction_prompt", "entity_tagging_prompt", "entity_types"}
    assert state.add_config["extraction_prompt"].startswith("You are a memory extractor.")
    assert state.add_config["entity_types"] == []
    # No ghost fields: unused vanilla config must never enter evolution state.
    assert state.search_config == {}


def test_evolvable_config_view_trims_ghost_paths():
    current = EvolutionState(
        project_id="p",
        version=3,
        add_config={"extraction_prompt": "v1", "enable_entities": True},
        search_config={
            "top_k": 10,
            "weights": {"fact": 0.8},
            "recall_size": 20,
            "hybrid_prefetch_max": 300,
        },
    )

    view = _evolvable_config_view(current)

    assert view is not None
    assert set(view["add_config"]) == {"extraction_prompt"}
    assert set(view["search_config"]) == {"top_k", "weights"}
    assert "recall_size" not in view["search_config"]
    assert "hybrid_prefetch_max" not in view["search_config"]
    assert _evolvable_config_view(None) is None


def test_is_evolvable_path_whitelist():
    assert is_evolvable_path("add_config.extraction_prompt")
    assert is_evolvable_path("add_config.entity_types")
    assert is_evolvable_path("search_config.top_k")
    assert is_evolvable_path("search_config.weights")
    assert is_evolvable_path("search_config.weights.fact")
    # Ghost paths from the full vanilla config must be rejected.
    assert not is_evolvable_path("search_config.recall_size")
    assert not is_evolvable_path("search_config.use_reranker")
    assert not is_evolvable_path("search_config.hybrid_prefetch_max")
    assert not is_evolvable_path("add_config.chunk_soft_token_budget")


def test_feedback_evo_extraction_normalization_drops_invalid_memories():
    normalized = _normalize_feedback_evo_extraction(
        {
            "memories": [
                {"entity_type": "fact", "text": "keep this"},
                {"entity_type": "scenario_specific"},  # no content -> dropped
                {"content": "has content but no ref_id"},
            ]
        }
    )
    assert len(normalized["memories"]) == 2
    assert normalized["memories"][0]["content"] == "keep this"
    assert normalized["memories"][0]["ref_id"].startswith("m_")
    assert normalized["memories"][1]["ref_id"].startswith("m_")
    assert _normalize_feedback_evo_extraction("not-a-dict") == {"memories": []}


def test_feedback_evo_extraction_normalization_keeps_source_refs():
    from mindmemos.components.extractor.vanilla.memory import MemoryExtractionResult

    normalized = _normalize_feedback_evo_extraction(
        {
            "memories": [
                {
                    "content": "policy memory",
                    "source_refs": [
                        "s0",
                        {"evidence_index": 1},
                        2,
                        {"ref_id": "s3"},
                        {"bad": True},
                        None,
                        "",
                    ],
                }
            ]
        }
    )
    assert normalized["memories"][0]["source_refs"] == ["s0", "s1", "s2", "s3"]
    result = MemoryExtractionResult.model_validate(normalized)
    assert result.memories[0].source_refs == ["s0", "s1", "s2", "s3"]
    assert _normalize_feedback_evo_extraction({"memories": []}) == {"memories": []}


@pytest.mark.asyncio
async def test_ensure_evolution_state_seeds_once_and_reuses():
    from mindmemos.config import init_config
    from mindmemos.typing import EvolutionResult, EvolutionState

    init_config(config_name="product", config_path="config/mindmemos/dev.example.yaml")

    class _FakeStore:
        def __init__(self) -> None:
            self.current: EvolutionState | None = None
            self.applied = 0

        async def get_current(self, project_id):
            del project_id
            return self.current

        async def apply(self, project_id, *, add_config, search_config, changes, trigger=None, rollback_version=None):
            self.applied += 1
            self.current = EvolutionState(
                project_id=project_id,
                version=1,
                is_current=True,
                add_config=add_config,
                search_config=search_config,
            )
            return EvolutionResult(project_id=project_id, version=1, changes=changes)

    store = _FakeStore()
    first = await ensure_evolution_state(store, "proj_1")  # type: ignore[arg-type]
    second = await ensure_evolution_state(store, "proj_1")  # type: ignore[arg-type]

    assert store.applied == 1
    assert first.version == 1
    assert second.version == 1
    assert first.add_config == second.add_config


def test_apply_changes_updates_add_and_search_config():
    current = EvolutionState(
        project_id="p",
        version=3,
        add_config={"extraction_prompt": "v1"},
        search_config={"weights": {"fact": 0.8}},
    )
    changes = [
        ParameterChange(path="search_config.weights.fact", before=0.8, after=0.6),
        ParameterChange(path="add_config.extraction_prompt", before="v1", after="v2"),
    ]

    add_config, search_config = _apply_changes(current, changes)

    assert add_config["extraction_prompt"] == "v2"
    assert search_config["weights"]["fact"] == 0.6


def test_valid_signals_filters_ghost_paths():
    signals = _valid_signals(
        [_event("e1", ["search_config.top_k", "search_config.recall_size", "not_a_path"])]
    )

    assert [signal["evolvable_path"] for signal in signals] == ["search_config.top_k"]


def test_signal_confidence_mean_and_defaults():
    signals = [
        {"evolvable_path": "search_config.top_k", "confidence": 0.8},
        {"evolvable_path": "search_config.top_k", "confidence": 0.6},
        {"evolvable_path": "add_config.extraction_prompt"},  # missing -> 1.0
    ]

    assert _signal_confidence(signals) == pytest.approx(0.8)
    assert _signal_confidence([]) == 0.0


@pytest.mark.asyncio
async def test_evolution_executor_applies_planned_changes():
    class _FakePlanner:
        async def plan(self, signals, current, *, max_changes=None):
            del signals, current, max_changes
            return [ParameterChange(path="search_config.weights.fact", before=0.8, after=0.6)]

    class _FakeStateStore:
        def __init__(self) -> None:
            self.applied: list[dict] = []

        async def get_current(self, project_id):
            del project_id
            return None

        async def apply(self, project_id, *, add_config, search_config, changes, trigger=None, rollback_version=None):
            self.applied.append(
                {
                    "project_id": project_id,
                    "add_config": add_config,
                    "search_config": search_config,
                    "changes": changes,
                }
            )
            return EvolutionResult(project_id=project_id, version=1, changes=changes)

    store = _FakeStateStore()
    executor = EvolutionExecutor(
        planner=_FakePlanner(),
        state_store=store,  # type: ignore[arg-type]
        min_signals_to_evolve=1,
    )

    result = await executor.run("proj_1", [_event("e1", ["search_config.weights.fact"])])

    assert result.version == 1
    assert result.changes[0].path == "search_config.weights.fact"
    assert store.applied[0]["search_config"]["weights"]["fact"] == 0.6


@pytest.mark.asyncio
async def test_evolution_executor_skips_when_signals_below_threshold():
    executor = EvolutionExecutor(
        planner=EvolutionPlanner(),
        min_signals_to_evolve=3,
    )
    result = await executor.run("proj_1", [_event("e1", ["search_config.top_k"])])
    assert result.version == 0
    assert result.changes == []


@pytest.mark.asyncio
async def test_evolution_executor_skips_when_signal_confidence_below_threshold():
    class _TrackingPlanner:
        def __init__(self) -> None:
            self.called = False

        async def plan(self, signals, current, *, max_changes=None):
            del signals, current, max_changes
            self.called = True
            return []

    class _FakeStateStore:
        async def get_current(self, project_id):
            del project_id
            return None

    planner = _TrackingPlanner()
    executor = EvolutionExecutor(
        planner=planner,  # type: ignore[arg-type]
        state_store=_FakeStateStore(),  # type: ignore[arg-type]
        min_signals_to_evolve=1,
        require_signal_confidence=0.7,
    )

    result = await executor.run("proj_1", [_event("e1", ["search_config.weights.fact"], confidence=0.4)])

    assert result.changes == []
    assert result.version == 0
    assert planner.called is False


@pytest.mark.asyncio
async def test_evolution_executor_drops_out_of_bounds_changes():
    current = EvolutionState(
        project_id="p",
        version=1,
        add_config={"extraction_prompt": "v1"},
        search_config={"top_k": 10},
    )

    class _FakePlanner:
        async def plan(self, signals, current, *, max_changes=None):
            del signals, current, max_changes
            return [
                ParameterChange(path="search_config.top_k", before=10, after=100),
                ParameterChange(path="add_config.extraction_prompt", before="v1", after="v2"),
            ]

    class _FakeStateStore:
        async def get_current(self, project_id):
            del project_id
            return current

        async def apply(self, project_id, *, add_config, search_config, changes, trigger=None, rollback_version=None):
            del project_id, trigger, rollback_version
            return EvolutionResult(
                project_id="p",
                version=2,
                changes=changes,
            )

    store = _FakeStateStore()
    executor = EvolutionExecutor(
        planner=_FakePlanner(),  # type: ignore[arg-type]
        state_store=store,  # type: ignore[arg-type]
        min_signals_to_evolve=1,
        require_signal_confidence=0.7,
        max_numeric_change_ratio=0.5,
    )

    result = await executor.run("p", [_event("e1", ["search_config.top_k"])])

    assert [change.path for change in result.changes] == ["add_config.extraction_prompt"]


@pytest.mark.asyncio
async def test_evolution_executor_passes_signals_with_reasons_to_planner():
    received: dict = {}

    class _TrackingPlanner:
        async def plan(self, signals, current, *, max_changes=None):
            del current, max_changes
            received["signals"] = signals
            return []

    class _FakeStateStore:
        async def get_current(self, project_id):
            del project_id
            return None

    executor = EvolutionExecutor(
        planner=_TrackingPlanner(),  # type: ignore[arg-type]
        state_store=_FakeStateStore(),  # type: ignore[arg-type]
        min_signals_to_evolve=1,
        require_signal_confidence=0.7,
    )

    event = FeedbackEvoEvent(
        event_id="evt",
        account_id="acc",
        project_id="proj_1",
        api_key_uuid="key",
        submitted_at=datetime.now(UTC),
        signals=[
            {"evolvable_path": "search_config.top_k", "confidence": 0.9, "reason": "too few memories returned"},
            {"evolvable_path": "search_config.recall_size", "confidence": 0.9, "reason": "ghost path"},
        ],
    )
    result = await executor.run("proj_1", [event])

    assert result.changes == []
    assert received["signals"] == [
        {"evolvable_path": "search_config.top_k", "confidence": 0.9, "reason": "too few memories returned"}
    ]


def test_filter_changes_by_threshold_numeric_ratio():
    current = EvolutionState(
        project_id="p",
        version=1,
        add_config={"extraction_prompt": "v1"},
        search_config={
            "top_k": 10,
            "score_threshold": 0.5,
            "weights": {"fact": 0.8},
        },
    )

    changes = [
        ParameterChange(path="search_config.top_k", before=10, after=14),
        ParameterChange(path="search_config.top_k", before=10, after=16),
        ParameterChange(path="search_config.weights.fact", before=0.8, after=0.5),
        ParameterChange(path="search_config.weights.fact", before=0.8, after=0.3),
        ParameterChange(path="search_config.score_threshold", before=0.5, after=1.0),
    ]

    kept = _filter_changes_by_threshold(
        current,
        changes,
        max_numeric_change_ratio=0.5,
        max_entity_type_delta=2,
    )

    assert [change.path for change in kept] == [
        "search_config.top_k",
        "search_config.weights.fact",
    ]


def test_filter_changes_by_threshold_entity_types_and_prompts():
    current = EvolutionState(
        project_id="p",
        version=1,
        add_config={"entity_types": ["defect_return", "exchange"], "extraction_prompt": "v1"},
        search_config={},
    )

    changes = [
        ParameterChange(path="add_config.entity_types", before=["defect_return", "exchange"], after=["defect_return", "exchange", "clawback"]),
        ParameterChange(path="add_config.entity_types", before=["defect_return", "exchange"], after=["defect_return", "exchange", "clawback", "promo", "refund"]),
        ParameterChange(path="add_config.extraction_prompt", before="v1", after="v2"),
    ]

    kept = _filter_changes_by_threshold(
        current,
        changes,
        max_numeric_change_ratio=0.5,
        max_entity_type_delta=2,
    )

    assert [change.path for change in kept] == [
        "add_config.entity_types",
        "add_config.extraction_prompt",
    ]
