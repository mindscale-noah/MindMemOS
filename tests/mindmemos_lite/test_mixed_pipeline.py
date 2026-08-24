from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from mindmemos_lite.config import (
    MemoryModePipelineConfig,
    MixedAddPipelineConfig,
    PipelineRoutingConfig,
    validate_tree,
)
from mindmemos_lite.errors import InvalidConfigError
from mindmemos_lite.infra.vector_store import FilterGroup, Predicate
from mindmemos_lite.persistence.memory import MemoryPersistence
from mindmemos_lite.persistence.v2 import MEMORY_TABLE
from mindmemos_lite.pipeline.mixed_memory import MixedAddPipeline, ModeSearchPipeline
from mindmemos_lite.typing import (
    AddPipelineInput,
    AddPipelineSyncResult,
    MemoryAddEventItem,
    MemoryDbMutationPlan,
    MemoryDbWritePlan,
    MemoryRequestContext,
    MemoryWrite,
    SearchPipelineInput,
    SearchPipelineResult,
    TextMessage,
)
from omegaconf import OmegaConf


def _context(*, mode: str | None = None) -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id=str(uuid4()),
        account_id="account",
        project_id="project",
        api_key_uuid=str(uuid4()),
        memory_algorithm=mode,
    )


class _ConcurrentAddPipeline:
    def __init__(self, name: str, started: list[str], both_started: asyncio.Event) -> None:
        self.name = name
        self.started = started
        self.both_started = both_started
        self.contexts: list[MemoryRequestContext] = []
        self.inputs: list[AddPipelineInput] = []

    async def add_sync(self, inp, context):
        self.inputs.append(inp)
        self.contexts.append(context)
        self.started.append(self.name)
        if len(self.started) == 2:
            self.both_started.set()
        await asyncio.wait_for(self.both_started.wait(), timeout=1)
        return AddPipelineSyncResult(
            status="ok",
            memories=[MemoryAddEventItem(operation="add", content=self.name)],
        )


@pytest.mark.asyncio
async def test_mixed_add_runs_children_concurrently_and_preserves_config_order() -> None:
    started: list[str] = []
    both_started = asyncio.Event()
    vanilla = _ConcurrentAddPipeline("vanilla", started, both_started)
    experience = _ConcurrentAddPipeline("experience", started, both_started)
    pipeline = MixedAddPipeline(
        pipelines={
            "vanilla": vanilla,
            "experience": experience,
        }
    )
    inp = AddPipelineInput(messages=[TextMessage(text="remember this")])

    result = await pipeline.add_sync(inp, _context())

    assert started == ["vanilla", "experience"]
    assert [item.content for item in result.memories] == ["vanilla", "experience"]
    assert vanilla.contexts[0].memory_algorithm == "vanilla"
    assert experience.contexts[0].memory_algorithm == "experience"
    assert vanilla.inputs[0] is not inp
    assert experience.inputs[0] is not inp
    assert vanilla.inputs[0] is not experience.inputs[0]


class _CapturingSearchPipeline:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    async def search(self, inp, context):
        self.calls.append((inp, context))
        return SearchPipelineResult(status="ok", memories=[])


@pytest.mark.asyncio
async def test_mode_search_routes_exactly_one_pipeline_and_binds_mode_context() -> None:
    vanilla = _CapturingSearchPipeline("vanilla")
    experience = _CapturingSearchPipeline("experience")
    pipeline = ModeSearchPipeline(
        pipelines={"vanilla": vanilla, "experience": experience},
        default_mode="vanilla",
    )

    await pipeline.search(
        SearchPipelineInput(query="how did it work?", memory_mode="experience"),
        _context(mode="vanilla"),
    )

    assert not vanilla.calls
    assert len(experience.calls) == 1
    child_input, child_context = experience.calls[0]
    assert child_input.memory_mode == "experience"
    assert child_context.memory_algorithm == "experience"


@pytest.mark.asyncio
async def test_mode_search_rejects_unconfigured_mode() -> None:
    pipeline = ModeSearchPipeline(
        pipelines={"vanilla": _CapturingSearchPipeline("vanilla")},
        default_mode="vanilla",
    )

    with pytest.raises(ValueError, match="unknown memory mode 'experience'"):
        await pipeline.search(
            SearchPipelineInput(query="query", memory_mode="experience"),
            _context(),
        )


def test_pipeline_routing_config_rejects_unknown_mixed_add_mode() -> None:
    config = PipelineRoutingConfig(
        default_search_mode="vanilla",
        modes={"vanilla": MemoryModePipelineConfig()},
        mixed_add=MixedAddPipelineConfig(modes=["vanilla", "experience"]),
    )

    with pytest.raises(InvalidConfigError, match="unknown: experience"):
        validate_tree(OmegaConf.structured(config))


class _CapturingVectorService:
    graph_enabled = False

    def __init__(self) -> None:
        self.upserts = []
        self.queries = []

    async def upsert_records(self, table, records):
        self.upserts.append((table, records))

    async def query_records(self, table, query):
        self.queries.append((table, query))
        return [], None


@pytest.mark.asyncio
async def test_persistence_stamps_and_filters_memory_mode_from_context() -> None:
    service = _CapturingVectorService()
    persistence = MemoryPersistence(service)
    context = _context(mode="experience")
    memory = MemoryWrite(
        memory_id=str(uuid4()),
        account_id=context.account_id,
        project_id=context.project_id,
        api_key_uuid=context.api_key_uuid,
        content="worked around a deployment issue",
        mem_extract_version="test",
        created_at=datetime.now(UTC),
    )
    plan = MemoryDbMutationPlan.from_write_plan(MemoryDbWritePlan(memories=[memory]))

    await persistence.apply_mutation_plan(context, plan)
    await persistence.list_memories(context)

    memory_upsert = next(records for table, records in service.upserts if table == MEMORY_TABLE)
    assert memory_upsert[0].payload["memory_mode"] == "experience"
    _, query = service.queries[-1]
    assert _contains_mode_predicate(query.filters, "experience")


def _contains_mode_predicate(value, expected_mode: str) -> bool:
    if isinstance(value, Predicate):
        return value.field == "memory_mode" and value.op == "eq" and value.value == expected_mode
    if isinstance(value, FilterGroup):
        return any(_contains_mode_predicate(clause, expected_mode) for clause in value.clauses)
    return False
