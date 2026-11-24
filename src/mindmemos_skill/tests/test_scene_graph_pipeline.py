"""Scene cache, atomic membership, immutable history and strategy integration."""

import asyncio
import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import jwt
import openai
import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatConfig
from mindmemos_skill.algos.manage.skill_manage.algorithm import SkillManageAlgorithm
from mindmemos_skill.algos.manage.skill_manage.extraction import SceneExtractor
from mindmemos_skill.api.typing import (
    AddGraphMembersRequest,
    CommitRequest,
    OrganizeRequest,
    RecommendRequest,
    RegisterRequest,
    RemoveGraphMembersRequest,
)
from mindmemos_skill.config.config import ServiceConfig
from mindmemos_skill.errors import GraphConflictError, GraphNotFoundError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.infra.model_access import platform
from mindmemos_skill.infra.model_access.platform import PLATFORM_MODEL_SECRET_ENV, PlatformModelError
from mindmemos_skill.persistence.constants import SKILL_GRAPH_VERSIONS_TABLE, SKILL_VERSIONS_TABLE
from mindmemos_skill.persistence.db_services.graphs.typing import GraphNodeType
from mindmemos_skill.persistence.db_services.graphs.writer import SkillGraphStore
from mindmemos_skill.persistence.db_services.skills.typing import SceneData
from mindmemos_skill.scene.typing import ProvidedSceneRecord
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import ValidationError

from mindmemos_skill import create_service

from .service_config import deployment_config
from .skill_sources import register_detail, structured


@pytest.fixture
def scene_record():
    """Use the original complete Scene and logic-step fixture."""
    data = json.loads((Path(__file__).parent / "fixtures/scene_graph_d0.json").read_text())
    return ProvidedSceneRecord.model_validate(data["build_input"]["provided"][0]["record"])


class Embeddings:
    """Deterministic asynchronous vectors with recorded input text."""

    def __init__(self, config):
        self.config = config
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.2 if "csv" in text.lower() else 0.8, 0.1] for text in texts]


@pytest.fixture
async def graph_service(tmp_path, monkeypatch, scene_record):
    """Run the real service and SQLite publication with injected model I/O."""
    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {"database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'db'}"}}},
            "repository_pipeline": {"class_name": "repository"},
            "manage_pipeline": {
                "class_name": "graph_manage",
                "config": {
                    "algorithm": {
                        "embedding": {"model": "fake", "base_url": "https://example.invalid/v1", "dimensions": 3},
                        "chat": {"model": "fake"},
                        "llm_linking": False,
                    }
                },
            },
        },
        "project",
    )
    extracted = []

    async def extract(self, document):
        extracted.append(document)
        return scene_record.model_copy(deep=True)

    monkeypatch.setattr(SceneExtractor, "extract", extract)
    async with await create_service(config) as service:
        pipeline = service._manage_pipeline
        embeddings = Embeddings(pipeline.config.algorithm.embedding)

        from mindmemos_skill.infra.embedding import EmbeddingModel

        monkeypatch.setattr(EmbeddingModel, "from_config", lambda config: embeddings)
        with service.repository.persistence.session(Scope(fields={"project_id": "project"})):
            yield service, embeddings, extracted


async def register(service, name):
    """Register an exact source through the normal authorized API."""
    source = Skill(
        SkillContent(
            content=f"---\nname: {name}\ndescription: Process {name}\n---\n## Inputs\n- table\n## Outputs\n- report\n"
        )
    )
    return await register_detail(service, RegisterRequest(project_id="project", skill=structured(source.snapshot())))


async def test_graph_logs_report_actual_scope_cache_and_topology(graph_service, caplog, monkeypatch):
    """Report measured graph sizes across builds, reuse, and membership changes."""
    from collections import Counter

    from mindmemos_skill.logging import get_logger

    monkeypatch.setattr(get_logger(), "propagate", True)
    caplog.set_level("INFO", logger="mindmemos_skill")
    service, _, _ = graph_service
    one, two = await register(service, "log-one"), await register(service, "log-two")
    request = OrganizeRequest(project_id="project", skill_refs=[one.skill.skill_id])
    built = await service.organize_skills(request)
    preview = await service.preview_graph(built.graph_id)
    nodes, edges = preview["nodes"], preview["edges"]
    summary = next(message for message in caplog.messages if "Graph generation completed" in message)
    assert f"graph_id={built.graph_id}" in summary
    assert f"graph_version_id={built.graph_version_id}" in summary
    assert f"skills=1 scenes={sum(node['node_type'] == GraphNodeType.SCENE for node in nodes)}" in summary
    assert f"nodes={len(nodes)} edges={len(edges)}" in summary
    assert f"edge_types={dict(sorted(Counter(edge['edge_type'] for edge in edges).items()))}" in summary
    assert "Graph organize inputs resolved | skills=1" in caplog.text
    assert "cache_hits=0 extracted=1" in caplog.text
    assert "reused_nodes=0" in caplog.text

    caplog.clear()
    reused = await service.organize_skills(request.model_copy(update={"graph_id": built.graph_id}))
    assert reused.graph_version_id == built.graph_version_id
    assert "Graph snapshot reused" in caplog.text
    assert "Graph Scene processing started" not in caplog.text
    assert "Graph embedding stage started" not in caplog.text

    caplog.clear()
    await service.add_graph_members(
        AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[two.skill.skill_id])
    )
    assert "prior_skills=1 added_skills=1 total_skills=2" in caplog.text
    embedding_summary = next(message for message in caplog.messages if "Graph embedding stage completed" in message)
    reused_nodes = int(re.search(r"reused_nodes=(\d+)", embedding_summary).group(1))
    assert reused_nodes >= len(nodes)

    caplog.clear()
    await service.remove_graph_members(
        RemoveGraphMembersRequest(
            project_id="project", graph_ref=built.graph_id, skill_refs=[one.skill.skill_id, two.skill.skill_id]
        )
    )
    assert "prior_skills=2 removed_skills=2 remaining_skills=0" in caplog.text
    assert "skills=0 scenes=0 nodes=0 edges=0 edge_types={}" in caplog.text
    assert "Graph publication completed" in caplog.text


@pytest.mark.parametrize("mode", ["new", "changed", "reused"])
async def test_generation_does_not_publish_until_explicit_publication(graph_service, mode):
    """Keep graph versions and the head unchanged until the publication step."""
    service, _, _ = graph_service
    pipeline = service._manage_pipeline
    first = await register(service, "generation-first")
    request = OrganizeRequest(project_id="project", skill_refs=[first.skill.skill_id])
    graph = None
    graph_id = "skg-generation-boundary"
    built = None
    if mode != "new":
        built = await service.organize_skills(request)
        graph_id = built.graph_id
        graph = await pipeline._resolve_graph(graph_id)
        request = request.model_copy(update={"graph_id": graph_id})
    if mode == "changed":
        second = await register(service, "generation-second")
        request = request.model_copy(update={"skill_refs": [first.skill.skill_id, second.skill.skill_id]})

    versions = await pipeline._resolve_skill_versions(request.skill_refs)
    sources = pipeline._skills_from_versions(versions)
    records = await pipeline._get_or_extract_scene_records(versions, sources, False)
    prior = await pipeline._read_prior_graph(graph)
    store = pipeline._graph_store(request, graph_id, versions, graph)
    database = service.repository.persistence.database
    filters = {"project_id": "project", "graph_id": graph_id}
    before = await database.select(SKILL_GRAPH_VERSIONS_TABLE, filters=filters, limit=None)

    generated = await pipeline._generate_graph(store, sources, records, prior)

    assert await database.select(SKILL_GRAPH_VERSIONS_TABLE, filters=filters, limit=None) == before
    if built is None:
        with pytest.raises(GraphNotFoundError):
            await service.resolve_graph(graph_id)
    else:
        assert (await service.resolve_graph(graph_id))["graph_version_id"] == built.graph_version_id
        assert (generated.snapshot_id == built.graph_version_id) == (mode == "reused")

    published = await pipeline._publish_graph(store, generated)

    assert published.graph_version_id == generated.snapshot_id
    assert (await service.resolve_graph(graph_id))["graph_version_id"] == published.graph_version_id
    after = await database.select(SKILL_GRAPH_VERSIONS_TABLE, filters=filters, limit=None)
    assert len(after) == len(before) + (mode != "reused")
    assert set(published.skill_version_ids) == {version.version_id for version in versions.values()}


@pytest.mark.parametrize("concurrency", [1, 2, 4])
async def test_scene_processing_respects_concurrency_and_reuses_cache(
    graph_service, monkeypatch, scene_record, concurrency, caplog
):
    service, _, _ = graph_service
    pipeline = service._manage_pipeline
    pipeline.config.scene_concurrency = concurrency
    items = [await register(service, f"parallel-{index}") for index in range(5)]
    versions = await pipeline._resolve_skill_versions([item.version.version_id for item in items])
    sources = pipeline._skills_from_versions(versions)
    active = 0
    peak = 0
    calls = 0
    ready = asyncio.Event()

    async def extract(self, document):
        nonlocal active, peak, calls
        active += 1
        calls += 1
        peak = max(peak, active)
        if active == concurrency:
            ready.set()
        try:
            await ready.wait()
            await asyncio.sleep(0.01)
            return scene_record.model_copy(deep=True)
        finally:
            active -= 1

    monkeypatch.setattr(SceneExtractor, "extract", extract)
    with caplog.at_level("INFO"):
        records = await asyncio.wait_for(pipeline._get_or_extract_scene_records(versions, sources, False), timeout=5)
        cached = await pipeline._get_or_extract_scene_records(versions, sources, False)
    assert peak == concurrency
    assert calls == len(items)
    assert list(records) == sorted(versions)
    assert cached == records
    assert "progress=5/5" in caplog.text
    assert "elapsed_s=" in caplog.text
    for item in items:
        assert f"skill={item.version.name}" in caplog.text
        assert f"version_id={item.version.version_id}" in caplog.text


async def test_scene_processing_cancels_pending_work_on_failure(graph_service, monkeypatch):
    service, _, _ = graph_service
    pipeline = service._manage_pipeline
    pipeline.config.scene_concurrency = 2
    items = [await register(service, f"failure-{index}") for index in range(4)]
    versions = await pipeline._resolve_skill_versions([item.version.version_id for item in items])
    sources = pipeline._skills_from_versions(versions)
    both_started = asyncio.Event()
    never_finished = asyncio.Event()
    calls = 0
    active = 0

    async def extract(self, document):
        nonlocal calls, active
        calls += 1
        active += 1
        first = calls == 1
        if calls == 2:
            both_started.set()
        try:
            await both_started.wait()
            if first:
                raise ValueError("Scene extraction failed")
            await never_finished.wait()
        finally:
            active -= 1

    monkeypatch.setattr(SceneExtractor, "extract", extract)
    with pytest.raises(ValueError, match="Scene extraction failed"):
        await asyncio.wait_for(pipeline._get_or_extract_scene_records(versions, sources, False), timeout=5)
    assert active == 0


def test_scene_concurrency_must_be_positive():
    from mindmemos_skill.pipelines.builtin_pipelines.manage.config import GraphManagePipelineConfig

    with pytest.raises(ValidationError):
        GraphManagePipelineConfig(scene_concurrency=0)


async def test_build_cache_reuse_force_and_history(graph_service):
    service, embeddings, extracted = graph_service
    first = await register(service, "csv")
    second = await register(service, "charts")
    request = OrganizeRequest(
        project_id="project", skill_refs=[first.skill.skill_id, second.skill.skill_id], alias="reports"
    )
    built = await service.organize_skills(request)
    assert built.graph_id.startswith("skg-") and len(extracted) == 2
    before = await service.preview_graph("reports")
    assert {node["node_type"] for node in before["nodes"]} == {GraphNodeType.SKILL, GraphNodeType.SCENE}
    assert "public_id" not in (await service.list_graph())[0]
    calls = len(embeddings.calls)
    reused = await service.organize_skills(
        request.model_copy(
            update={"graph_id": built.graph_id, "alias": None, "skill_refs": list(reversed(request.skill_refs))}
        )
    )
    assert reused.graph_version_id == built.graph_version_id
    assert len(extracted) == 2 and len(embeddings.calls) == calls
    forced = await service.organize_skills(
        request.model_copy(update={"graph_id": built.graph_id, "alias": None, "force_extract_scene": True})
    )
    assert forced.graph_version_id == built.graph_version_id and len(extracted) == 4
    assert await service.get_graph_snapshot(built.graph_id, built.graph_version_id) == before
    for strategy in ("graph", "vector"):
        result = await service.recommend_skills(
            RecommendRequest(project_id="project", graph_id=built.graph_id, task=" csv ", strategy=strategy, top_k=1)
        )
        assert len(result.skills) == len(result.scores) == 1
        assert result.graph_version_id == built.graph_version_id
    assert len(extracted) == 4
    assert embeddings.calls[-1] == ["csv"]


async def test_atomic_batch_members_and_empty_graph(graph_service):
    service, _, extracted = graph_service
    one, two = await register(service, "one"), await register(service, "two")
    built = await service.organize_skills(
        OrganizeRequest(project_id="project", skill_refs=[one.skill.skill_id], alias="members")
    )
    add = AddGraphMembersRequest(project_id="project", graph_ref="members", skill_refs=[two.skill.skill_id])
    changed = await service.add_graph_members(add)
    assert set(changed.skill_version_ids) == {one.version.version_id, two.version.version_id}
    assert len(extracted) == 2
    with pytest.raises(ValueError, match="already"):
        await service.add_graph_members(add)
    with pytest.raises(ValueError, match="same skill id"):
        await service.remove_graph_members(
            RemoveGraphMembersRequest(
                project_id="project", graph_ref="members", skill_refs=[one.skill.skill_id, one.version.version_id]
            )
        )
    with pytest.raises(GraphNotFoundError):
        await service.remove_graph_members(
            RemoveGraphMembersRequest(
                project_id="project", graph_ref=changed.graph_version_id, skill_refs=[one.skill.skill_id]
            )
        )
    remaining = await service.remove_graph_members(
        RemoveGraphMembersRequest(project_id="project", graph_ref="members", skill_refs=[two.skill.skill_id])
    )
    assert remaining.graph_version_id == built.graph_version_id
    empty = await service.remove_graph_members(
        RemoveGraphMembersRequest(project_id="project", graph_ref="members", skill_refs=[one.skill.skill_id])
    )
    assert empty.skill_version_ids == [] and len(extracted) == 2
    for strategy in ("graph", "vector"):
        result = await service.recommend_skills(
            RecommendRequest(project_id="project", graph_id=empty.graph_id, task="one", strategy=strategy)
        )
        assert result.skills == result.scores == [] and result.graph_version_id == empty.graph_version_id
    assert await service.get_graph_snapshot(built.graph_id, built.graph_version_id)
    await service.delete_graph("members")
    with pytest.raises(GraphNotFoundError):
        await service.recommend_skills(RecommendRequest(project_id="project", graph_id=built.graph_id, task="one"))


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        {"schema_version": "1"},
        {"schema_version": True},
        {"schema_version": 2},
        {"schema_version": 1, "record": {}},
    ],
)
async def test_invalid_cache_reextracts_and_merges_metadata(graph_service, invalid):
    service, _, extracted = graph_service
    item = await register(service, "cache")
    db = service.repository.persistence.database
    await db.update(
        SKILL_VERSIONS_TABLE,
        {"metadata": {"keep": "value", "scene_data": invalid}},
        filters={"version_id": item.version.version_id},
    )
    await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.version.version_id]))
    assert len(extracted) == 1
    version = await service.repository.persistence.skills.get(item.version.version_id)
    assert version.metadata["keep"] == "value" and version.metadata["scene_data"]["schema_version"] == 1


async def test_extraction_failure_preserves_cache_and_head(graph_service, monkeypatch):
    service, _, _ = graph_service
    item = await register(service, "failure")
    request = OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id])
    built = await service.organize_skills(request)
    skills = service.repository.persistence.skills
    before = await skills.get_scene_data(item.version.version_id)

    async def fail(self, document):
        raise ValueError("Controlled extraction failure")

    monkeypatch.setattr(SceneExtractor, "extract", fail)
    with pytest.raises(ValueError, match="Controlled"):
        await service.organize_skills(
            request.model_copy(update={"graph_id": built.graph_id, "force_extract_scene": True})
        )
    assert await skills.get_scene_data(item.version.version_id) == before
    assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == built.graph_version_id


async def test_head_check_rejects_stale_publication(graph_service):
    service, _, _ = graph_service
    item = await register(service, "conflict")
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id]))
    store = SkillGraphStore(
        service.repository.persistence.database,
        graph_id=built.graph_id,
        actor_id="project",
        version_ids={item.skill.skill_id: item.version.version_id},
        expected_snapshot="skgv-" + uuid4().hex,
        check_head=True,
    )
    with pytest.raises(GraphConflictError):
        await store.select_snapshot(built.graph_version_id)
    assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == built.graph_version_id


class Chat:
    """Original extraction transport seam adapted to native async."""

    def __init__(self, outputs):
        self.config = OpenAIChatConfig(model="fake", parse_max_attempts=2, parse_retry_mode="feedback")
        self.outputs = iter(outputs)
        self.calls = []

    async def run(self, **request):
        self.calls.append(deepcopy(request))
        return AgentResult(output=next(self.outputs), usage={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7})


async def test_original_extractor_feedback_retry(scene_record):
    chat = Chat(["{}", json.dumps({"status": "ok", "record": scene_record.model_dump(mode="json")})])
    extractor = SceneExtractor(chat)
    assert await extractor.extract("Complete source") == scene_record
    assert extractor.summary.attempts == 2 and extractor.summary.total_tokens == 14
    assert "Repair exchange:" in chat.calls[1]["user"]
    assert "ID/ownership/transition/reachability" in chat.calls[1]["user"]


@pytest.mark.parametrize("version", ["1", True, 0, 2])
def test_scene_data_strict_schema(scene_record, version):
    with pytest.raises(ValidationError):
        SceneData(schema_version=version, record=scene_record)


async def test_force_changed_scene_does_not_reuse_and_history_is_immutable(graph_service, monkeypatch, scene_record):
    service, _, _ = graph_service
    item = await register(service, "changed")
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id]))
    before = await service.get_graph_snapshot(built.graph_id, built.graph_version_id)
    changed = scene_record.model_copy(
        update={"skill": scene_record.skill.model_copy(update={"goal": "Changed public capability"})}, deep=True
    )

    async def extract(self, document):
        return changed

    monkeypatch.setattr(SceneExtractor, "extract", extract)
    rebuilt = await service.organize_skills(
        OrganizeRequest(
            project_id="project", graph_id=built.graph_id, skill_refs=[item.skill.skill_id], force_extract_scene=True
        )
    )
    assert rebuilt.graph_version_id != built.graph_version_id
    assert await service.get_graph_snapshot(built.graph_id, built.graph_version_id) == before


async def test_concurrent_pipeline_additions_detect_conflict_instead_of_losing_members(graph_service):
    service, embeddings, _ = graph_service
    one, two, three = await register(service, "one"), await register(service, "two"), await register(service, "three")
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[one.skill.skill_id]))
    original = embeddings.embed
    barrier = asyncio.Barrier(2)

    async def embed(texts):
        await asyncio.wait_for(barrier.wait(), timeout=2)
        return await original(texts)

    embeddings.embed = embed
    pipeline = service._manage_pipeline
    outcomes = await asyncio.gather(
        pipeline.add_graph_members(
            AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[two.skill.skill_id])
        ),
        pipeline.add_graph_members(
            AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[three.skill.skill_id])
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(value, GraphConflictError) for value in outcomes) == 1
    published = next(value for value in outcomes if not isinstance(value, Exception))
    assert one.version.version_id in published.skill_version_ids and len(published.skill_version_ids) == 2
    assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == published.graph_version_id


async def test_cancelled_build_keeps_graph_head(graph_service):
    service, embeddings, _ = graph_service
    one, two = await register(service, "cancel-one"), await register(service, "cancel-two")
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[one.skill.skill_id]))
    started = asyncio.Event()

    async def embed(texts):
        started.set()
        await asyncio.Event().wait()

    embeddings.embed = embed
    task = asyncio.create_task(
        service.add_graph_members(
            AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[two.skill.skill_id])
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == built.graph_version_id
    rows = await service.repository.persistence.database.select(
        SKILL_GRAPH_VERSIONS_TABLE, filters={"graph_id": built.graph_id}
    )
    assert len(rows) == 1


@pytest.mark.parametrize("corruption", ["entry", "scene_duplicate", "step_owner", "step_target", "terminal"])
async def test_semantically_invalid_cache_reextracts(graph_service, scene_record, corruption):
    service, _, extracted = graph_service
    item = await register(service, "semantic-cache")
    record = scene_record.model_dump(mode="json")
    if corruption == "entry":
        record["entry_scene_id"] = "missing"
    elif corruption == "scene_duplicate":
        record["scenes"].append(deepcopy(record["scenes"][0]))
    elif corruption == "step_owner":
        record["logic_steps"][0]["scene_id"] = "missing"
    elif corruption == "step_target":
        record["logic_steps"][0]["next_step_rules"] = [{"target": "missing", "condition": ""}]
    else:
        record["scenes"][0]["next_scene_rules"] = [{"target": "END_", "condition": ""}]
    await service.repository.persistence.database.update(
        SKILL_VERSIONS_TABLE,
        {"metadata": {"scene_data": {"schema_version": 1, "record": record}}},
        filters={"version_id": item.version.version_id},
    )
    await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id]))
    assert len(extracted) == 1


async def test_invalid_new_scene_result_is_not_persisted(graph_service, monkeypatch, scene_record):
    service, _, _ = graph_service
    item = await register(service, "invalid-output")
    request = OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id])
    built = await service.organize_skills(request)
    before = await service.repository.persistence.skills.get_scene_data(item.version.version_id)

    async def extract(self, document):
        return scene_record.model_copy(update={"entry_scene_id": "missing"})

    monkeypatch.setattr(SceneExtractor, "extract", extract)
    with pytest.raises(ValidationError):
        await service.organize_skills(
            request.model_copy(update={"graph_id": built.graph_id, "force_extract_scene": True})
        )
    assert await service.repository.persistence.skills.get_scene_data(item.version.version_id) == before


async def test_cache_storage_failure_does_not_trigger_extraction(graph_service, monkeypatch):
    service, _, extracted = graph_service
    item = await register(service, "storage-failure")
    from mindmemos_skill.persistence.db_services.skills.service import SkillDataService

    async def get_scene_data(self, version_id):
        raise RuntimeError("Controlled database failure")

    monkeypatch.setattr(SkillDataService, "get_scene_data", get_scene_data)
    with pytest.raises(RuntimeError, match="database"):
        await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id]))
    assert extracted == []


async def test_existing_graph_pins_build_config_and_vector_profile(graph_service, monkeypatch):
    service, embeddings, _ = graph_service
    item = await register(service, "config-change")
    request = OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id])
    built = await service.organize_skills(request)
    pipeline = service._manage_pipeline
    request = request.model_copy(update={"graph_id": built.graph_id})
    original = pipeline.config.algorithm
    query_only = original.model_copy(update={"mode": "fuse", "alpha": 0.7})
    pipeline.config = pipeline.config.model_copy(update={"algorithm": query_only})
    assert (await service.organize_skills(request)).graph_version_id == built.graph_version_id
    build_change = query_only.model_copy(
        update={"joint_linking": query_only.joint_linking.model_copy(update={"dependency_match_threshold": 0.9})}
    )
    pipeline.config = pipeline.config.model_copy(update={"algorithm": build_change})
    rebuilt = await service.organize_skills(request)
    assert rebuilt.graph_version_id == built.graph_version_id
    profile_change = build_change.model_copy(
        update={"embedding": build_change.embedding.model_copy(update={"model": "fake-v2"})}
    )
    pipeline.config = pipeline.config.model_copy(update={"algorithm": profile_change})
    embeddings.config = profile_change.embedding
    changed = await service.organize_skills(request)
    assert changed.graph_version_id == rebuilt.graph_version_id
    fresh = await service.organize_skills(request.model_copy(update={"graph_id": None}))
    assert fresh.graph_id != built.graph_id


async def test_document_limit_is_checked_before_scene_model_io(graph_service):
    service, _, extracted = graph_service
    item = await register(service, "oversized")
    pipeline = service._manage_pipeline
    pipeline.config = pipeline.config.model_copy(
        update={"algorithm": pipeline.config.algorithm.model_copy(update={"max_input_chars": 10})}
    )
    with pytest.raises(ValueError, match="exceeds"):
        await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id]))
    assert extracted == []


async def test_new_vector_response_contract_caps_original_internal_results(graph_service, monkeypatch):
    from mindmemos_skill.algos.manage.skill_manage.typing import SceneGraphVectorRetrieval, SceneGraphVectorScore

    service, _, _ = graph_service
    one, two = await register(service, "cap-one"), await register(service, "cap-two")
    built = await service.organize_skills(
        OrganizeRequest(project_id="project", skill_refs=[one.skill.skill_id, two.skill.skill_id])
    )

    async def retrieve_vector(self, *, task, limit=5, include_scenes=False):
        return SceneGraphVectorRetrieval(
            snapshot_id=self.snapshot_id,
            scores=tuple(
                SceneGraphVectorScore(
                    member_id=item.skill.skill_id,
                    node_id=f"node-{i}",
                    node_type=GraphNodeType.SKILL,
                    skill_version_id=item.version.version_id,
                    metadata={},
                    similarity=1 - i / 10,
                )
                for i, item in enumerate((one, two))
            ),
        )

    monkeypatch.setattr(SkillManageAlgorithm, "retrieve_vector", retrieve_vector)
    result = await service.recommend_skills(
        RecommendRequest(project_id="project", graph_id=built.graph_id, task="cap", strategy="vector", top_k=1)
    )
    assert [item.version_id for item in result.skills] == [one.version.version_id]
    assert len(result.scores) == 1


@pytest.mark.parametrize("strategy", ["graph", "vector"])
@pytest.mark.parametrize("member_count", [3, 12])
async def test_recommend_loads_only_selected_versions(graph_service, monkeypatch, strategy, member_count):
    """Keep full version reads proportional to results, preserving pinned versions and order."""
    from mindmemos_skill.infra.database import Database
    from mindmemos_skill.persistence.constants import SKILLS_TABLE

    service, embeddings, _ = graph_service
    service._manage_pipeline.config.algorithm = service._manage_pipeline.config.algorithm.model_copy(
        update={"build_scene": False}
    )
    members = [await register(service, f"lazy-member-{index}") for index in range(member_count)]
    built = await service.organize_skills(
        OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id for item in members])
    )
    request = RecommendRequest(
        project_id="project", graph_id=built.graph_id, task="Process lazy-member", strategy=strategy, top_k=2
    )
    expected = await service.recommend_skills(request)
    # Advancing a family must not change the versions pinned by the graph.
    await service.commit_skill(
        CommitRequest(
            project_id="project",
            skill_id=members[0].skill.skill_id,
            skill=structured(
                members[0].version.snapshot().model_copy(update={"content": members[0].version.content + "\nUpdated"})
            ),
        )
    )
    reads = []
    original_select = Database.select
    original_embed = embeddings.embed

    async def record_skill_reads(self, table, **kwargs):
        rows = await original_select(self, table, **kwargs)
        if table in (SKILLS_TABLE, SKILL_VERSIONS_TABLE):
            reads.append((table, kwargs.get("fields"), rows))
        return rows

    async def embed_after_visibility_check(texts):
        assert len(reads) == 2
        assert all(fields is not None for _, fields, _ in reads)
        return await original_embed(texts)

    monkeypatch.setattr(Database, "select", record_skill_reads)
    monkeypatch.setattr(embeddings, "embed", embed_after_visibility_check)
    result = await service.recommend_skills(request)
    assert result.model_dump() == expected.model_dump()
    assert len(reads) == 2 + 2 * request.top_k
    assert reads[0][1] == ["version_id", "skill_id"] and len(reads[0][2]) == member_count
    assert reads[1][1] == ["skill_id"] and len(reads[1][2]) == member_count
    loaded_ids = [
        row["version_id"]
        for table, fields, rows in reads
        if table == SKILL_VERSIONS_TABLE and fields is None
        for row in rows
    ]
    assert loaded_ids == [item.version_id for item in result.skills]


@pytest.mark.parametrize("strategy", ["graph", "vector"])
async def test_recommend_rejects_deleted_unselected_member(graph_service, strategy):
    """Validate all owners before model I/O even when a deleted member was outside top-k."""
    from mindmemos_skill.errors import SkillNotFoundError

    service, embeddings, _ = graph_service
    members = [await register(service, f"visibility-{index}") for index in range(3)]
    built = await service.organize_skills(
        OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id for item in members])
    )
    request = RecommendRequest(
        project_id="project", graph_id=built.graph_id, task="visibility", strategy=strategy, top_k=1
    )
    selected = await service.recommend_skills(request)
    omitted = next(item for item in members if item.version.version_id != selected.skills[0].version_id)
    await service.repository.persistence.skills.delete(omitted.skill)
    embeddings.calls.clear()
    with pytest.raises(SkillNotFoundError, match=omitted.version.version_id):
        await service.recommend_skills(request)
    assert embeddings.calls == []


async def test_independent_database_clients_enforce_head_conflict(graph_service, tmp_path):
    from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
    from mindmemos_skill.infra.database.scoped import ScopedDatabase
    from mindmemos_skill.persistence import create_persistence

    service, _, _ = graph_service
    item = await register(service, "two-clients")
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id]))
    databases = [
        service.repository.persistence.database,
        SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'db'}")),
    ]
    create_persistence(databases[1])
    stores = [
        SkillGraphStore(
            ScopedDatabase(database, Scope(fields={"project_id": "project"})),
            graph_id=built.graph_id,
            actor_id="test",
            version_ids={item.skill.skill_id: item.version.version_id},
            expected_snapshot=built.graph_version_id,
            check_head=True,
        )
        for database in databases
    ]
    nodes, edges, metadata = await stores[0].load(built.graph_version_id)

    async def publish(index):
        snapshot = "skgv-" + uuid4().hex
        return await stores[index].publish(
            snapshot,
            request_hash=("b" if index == 0 else "c") * 64,
            nodes=[node.model_copy(update={"graph_version_id": snapshot}) for node in nodes],
            edges=[edge.model_copy(update={"graph_version_id": snapshot, "edge_id": uuid4()}) for edge in edges],
            metadata=metadata,
        )

    try:
        results = await asyncio.gather(publish(0), publish(1), return_exceptions=True)
        assert sum(isinstance(value, GraphConflictError) for value in results) == 1
        successful = next(value for value in results if not isinstance(value, Exception))
        assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == successful
        assert len(await service.list_graph_snapshots(built.graph_id)) == 2
    finally:
        await databases[1].close()


async def test_original_skill_only_build_switch_and_explicit_cache_force(graph_service):
    service, _, extracted = graph_service
    item = await register(service, "without-scenes")
    pipeline = service._manage_pipeline
    original = pipeline.config.algorithm
    pipeline.config = pipeline.config.model_copy(
        update={"algorithm": original.model_copy(update={"build_scene": False})}
    )
    request = OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id])
    built = await service.organize_skills(request)
    assert extracted == []
    assert {node["node_type"] for node in (await service.preview_graph(built.graph_id))["nodes"]} == {
        GraphNodeType.SKILL
    }
    forced = await service.organize_skills(
        request.model_copy(update={"graph_id": built.graph_id, "force_extract_scene": True})
    )
    assert len(extracted) == 1 and forced.graph_version_id == built.graph_version_id
    assert await service.repository.persistence.skills.get_scene_data(item.version.version_id) is not None
    pipeline.config = pipeline.config.model_copy(update={"algorithm": original})
    with_scenes = await service.organize_skills(request.model_copy(update={"graph_id": built.graph_id}))
    assert with_scenes.graph_version_id == built.graph_version_id
    fresh = await service.organize_skills(request)
    assert {node["node_type"] for node in (await service.preview_graph(fresh.graph_id))["nodes"]} == {
        GraphNodeType.SKILL,
        GraphNodeType.SCENE,
    }
    assert len(extracted) == 1


async def test_scene_extraction_order_matches_original_sorted_members(graph_service):
    service, _, extracted = graph_service
    items = [await register(service, name) for name in ("order-one", "order-two", "order-three")]
    inputs = list(reversed(sorted(items, key=lambda item: item.skill.skill_id)))
    await service.organize_skills(
        OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id for item in inputs])
    )
    assert extracted == [item.version.content for item in sorted(items, key=lambda item: item.skill.skill_id)]


@pytest.mark.parametrize("build_scene", [True, False])
async def test_incremental_enrichment_preserves_existing_members(graph_service, monkeypatch, build_scene):
    from mindmemos_skill.algos.manage.skill_manage.enrichment import SkillEnricher

    service, _, _ = graph_service
    one, two = await register(service, "enrich-one"), await register(service, "enrich-two")
    pipeline = service._manage_pipeline
    pipeline.config = pipeline.config.model_copy(
        update={
            "algorithm": pipeline.config.algorithm.model_copy(update={"enrich_skill": True, "build_scene": build_scene})
        }
    )
    phase = {"value": "first"}
    calls = []

    async def enrich(self, node, document):
        calls.append(node.skill_version_id)
        return node.model_copy(update={"example_tasks": node.example_tasks or [phase["value"]]})

    monkeypatch.setattr(SkillEnricher, "enrich", enrich)
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[one.skill.skill_id]))
    phase["value"] = "second"
    added = await service.add_graph_members(
        AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[two.skill.skill_id])
    )
    store = SkillGraphStore(
        service.repository.persistence.database,
        graph_id=built.graph_id,
        actor_id="test",
        version_ids={},
    )
    old_nodes, _, _ = await store.load(built.graph_version_id)
    new_nodes, _, _ = await store.load(added.graph_version_id)
    assert next(node for node in old_nodes if node.node_type == GraphNodeType.SKILL).example_tasks == ["first"]
    assert {
        node.skill_version_id: node.example_tasks for node in new_nodes if node.node_type == GraphNodeType.SKILL
    } == {
        one.version.version_id: ["first"],
        two.version.version_id: ["second"],
    }
    removed = await service.remove_graph_members(
        RemoveGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[one.skill.skill_id])
    )
    remaining, _, _ = await store.load(removed.graph_version_id)
    assert next(node for node in remaining if node.node_type == GraphNodeType.SKILL).example_tasks == ["second"]
    assert calls == [one.version.version_id, two.version.version_id]


@pytest.mark.parametrize("build_scene", [True, False])
async def test_membership_changes_keep_snapshot_build_settings(graph_service, build_scene):
    """Service defaults cannot change the graph's representation during add or remove."""
    service, _, extracted = graph_service
    items = [await register(service, name) for name in ("pinned-one", "pinned-two", "pinned-three")]
    pipeline = service._manage_pipeline
    original = pipeline.config.algorithm.model_copy(update={"build_scene": build_scene})
    pipeline.config = pipeline.config.model_copy(update={"algorithm": original})
    built = await service.organize_skills(
        OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id for item in items[:2]])
    )
    pipeline.config = pipeline.config.model_copy(
        update={
            "algorithm": original.model_copy(
                update={
                    "build_scene": not build_scene,
                    "enrich_skill": True,
                    "embedding": original.embedding.model_copy(update={"model": "changed-default"}),
                }
            )
        }
    )
    added = await service.add_graph_members(
        AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[items[2].skill.skill_id])
    )
    removed = await service.remove_graph_members(
        RemoveGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[items[0].skill.skill_id])
    )
    store = SkillGraphStore(
        service.repository.persistence.database, graph_id=built.graph_id, actor_id="test", version_ids={}
    )
    _, _, baseline = await store.load(built.graph_version_id)
    for result in (added, removed):
        nodes, _, metadata = await store.load(result.graph_version_id)
        assert metadata["build_settings"]["config"] == baseline["build_settings"]["config"]
        assert any(node.node_type == GraphNodeType.SCENE for node in nodes) == build_scene
    assert len(extracted) == (3 if build_scene else 0)


async def test_existing_members_ignore_changed_version_cache(graph_service, scene_record):
    """A shared cache refresh must not replace semantics captured in an existing graph."""
    service, embeddings, extracted = graph_service
    items = [await register(service, name) for name in ("cache-one", "cache-two", "cache-three")]
    request = OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id for item in items[:2]])
    built = await service.organize_skills(request)
    changed = scene_record.model_copy(
        update={"skill": scene_record.skill.model_copy(update={"goal": "Changed in another graph"})}, deep=True
    )
    await service.repository.persistence.skills.set_scene_data(items[0].version.version_id, SceneData(record=changed))
    reused = await service.organize_skills(request.model_copy(update={"graph_id": built.graph_id}))
    assert reused.graph_version_id == built.graph_version_id
    added = await service.add_graph_members(
        AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[items[2].skill.skill_id])
    )
    calls_before_remove = len(embeddings.calls)
    removed = await service.remove_graph_members(
        RemoveGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[items[1].skill.skill_id])
    )
    store = SkillGraphStore(
        service.repository.persistence.database, graph_id=built.graph_id, actor_id="test", version_ids={}
    )
    baseline, _, _ = await store.load(built.graph_version_id)
    expected = [
        node.model_dump(exclude={"graph_version_id"})
        for node in baseline
        if node.skill_version_id == items[0].version.version_id
    ]
    for result in (added, removed):
        nodes, _, _ = await store.load(result.graph_version_id)
        assert [
            node.model_dump(exclude={"graph_version_id"})
            for node in nodes
            if node.skill_version_id == items[0].version.version_id
        ] == expected
    assert len(extracted) == 3
    assert len(embeddings.calls) == calls_before_remove


async def test_snapshot_config_restores_models_and_runtime_options(graph_service):
    """Keep model parameters pinned without persisting credentials or freezing query options."""
    from mindmemos_skill.pipelines.builtin_pipelines.manage.build_config import restore_build_config

    service, _, _ = graph_service
    item = await register(service, "connection-config")
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id]))
    store = SkillGraphStore(
        service.repository.persistence.database, graph_id=built.graph_id, actor_id="test", version_ids={}
    )
    _, _, metadata = await store.load(built.graph_version_id)
    original = service._manage_pipeline.config.algorithm
    current = original.model_copy(
        update={
            "mode": "fuse",
            "alpha": 0.7,
            "embedding": original.embedding.model_copy(update={"model": "new-model", "api_key": "new-secret"}),
            "chat": original.chat.model_copy(update={"model": "new-chat", "api_key": "new-chat-secret"}),
        }
    )
    restored = restore_build_config(current, metadata)
    assert restored.mode == "fuse" and restored.alpha == 0.7
    assert restored.embedding.model == original.embedding.model
    assert restored.chat.model == original.chat.model
    assert restored.embedding.api_key == "new-secret"
    assert restored.chat.api_key == "new-chat-secret"
    for name in ("chat", "embedding"):
        assert "api_key" not in metadata["build_settings"]["config"][name]
        assert "base_url" not in metadata["build_settings"]["config"][name]
    incomplete = deepcopy(metadata)
    incomplete["build_settings"]["config"].pop("chat")
    with pytest.raises(ValueError, match="captured Scene extraction model"):
        restore_build_config(current, incomplete)
    incompatible = current.model_copy(
        update={"embedding": current.embedding.model_copy(update={"base_url": "https://other.invalid/v1"})}
    )
    with pytest.raises(ValueError, match="original embedding endpoint"):
        restore_build_config(incompatible, metadata)


async def test_cross_skill_model_links_are_rebuilt_for_final_scope(graph_service, monkeypatch):
    """Retained nodes must receive new cross-member edges after both mutation types."""
    service, _, extracted = graph_service
    pipeline = service._manage_pipeline
    pipeline.config = pipeline.config.model_copy(
        update={"algorithm": pipeline.config.algorithm.model_copy(update={"llm_linking": True})}
    )
    batches = []

    async def infer(self, pairs):
        pairs = list(pairs)
        batches.append(pairs)
        return [
            {
                "source": left.node_id,
                "target": right.node_id,
                "type": "semantic",
                "weight": 0.5,
                "description": f"build-{len(batches)}",
            }
            for left, right in pairs
        ]

    monkeypatch.setattr(SkillManageAlgorithm, "_infer_joint_relations", infer)
    items = [await register(service, name) for name in ("link-one", "link-two", "link-three", "link-four")]
    built = await service.organize_skills(
        OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id for item in items[:3]])
    )
    added = await service.add_graph_members(
        AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[items[3].skill.skill_id])
    )
    removed = await service.remove_graph_members(
        RemoveGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[items[0].skill.skill_id])
    )
    store = SkillGraphStore(
        service.repository.persistence.database, graph_id=built.graph_id, actor_id="test", version_ids={}
    )
    for index, result in enumerate((built, added, removed), 1):
        nodes, edges, _ = await store.load(result.graph_version_id)
        inferred = [edge for edge in edges if edge.edge_type == "semantic"]
        assert inferred and all(edge.metadata["evidence"] == f"build-{index}" for edge in inferred)
        ids = {node.node_id for node in nodes}
        assert all(edge.source_node_id in ids and edge.target_node_id in ids for edge in edges)
    assert len(batches) == 3 and len(extracted) == 4


async def test_graph_generation_reuses_captured_skills_without_database_reads(graph_service, monkeypatch):
    """Build, reuse, and bind graphs without rereading captured Skill records."""
    from mindmemos_skill.infra.database.scoped import ScopedDatabase
    from mindmemos_skill.persistence.constants import SKILLS_TABLE

    service, _, _ = graph_service
    pipeline = service._manage_pipeline
    detail = await register(service, "captured-source")
    request = OrganizeRequest(project_id="project", skill_refs=[detail.skill.skill_id])
    versions = await pipeline._resolve_skill_versions(request.skill_refs)
    sources = pipeline._skills_from_versions(versions)
    records = await pipeline._get_or_extract_scene_records(versions, sources, False)
    store = pipeline._graph_store(request, "skg-captured-source", versions, None)
    original_select = ScopedDatabase.select

    async def select_without_skill_reads(self, table: str, **kwargs):
        assert table not in {SKILLS_TABLE, SKILL_VERSIONS_TABLE}, "Captured Skill data was queried again"
        return await original_select(self, table, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(ScopedDatabase, "select", select_without_skill_reads)
        restored = pipeline._skills_from_versions(versions)
        assert restored[detail.skill.skill_id].snapshot() == sources[detail.skill.skill_id].snapshot()
        generated = await pipeline._generate_graph(store, sources, records, None)
        owner = next(node for node in generated.nodes if node.node_type == GraphNodeType.SKILL)
        assert owner.metadata["family_id"] == detail.skill.skill_id
        assert owner.skill_version_id == detail.version.version_id

    await pipeline._publish_graph(store, generated)

    with monkeypatch.context() as guarded:
        guarded.setattr(ScopedDatabase, "select", select_without_skill_reads)
        reused = await pipeline._generate_graph(store, sources, records, None)
        assert reused.snapshot_id == generated.snapshot_id
        algorithm = pipeline._algorithm(store)
        await algorithm.load_snapshot(
            reused.snapshot_id, nodes=reused.nodes, edges=reused.edges, metadata=reused.metadata
        )
        assert algorithm.snapshot_id == generated.snapshot_id


@pytest.mark.parametrize("build_scene", [True, False])
async def test_organize_replaces_only_changed_skill_version(graph_service, build_scene):
    """Replacing one version rebuilds that member and preserves other captured members."""
    service, _, extracted = graph_service
    pipeline = service._manage_pipeline
    pipeline.config = pipeline.config.model_copy(
        update={"algorithm": pipeline.config.algorithm.model_copy(update={"build_scene": build_scene})}
    )
    one, two = await register(service, "replace-one"), await register(service, "replace-two")
    request = OrganizeRequest(project_id="project", skill_refs=[one.skill.skill_id, two.skill.skill_id])
    built = await service.organize_skills(request)
    updated = await service.commit_skill(
        CommitRequest(
            project_id="project",
            skill_id=one.skill.skill_id,
            skill=structured(
                SkillContent(content="---\nname: replace-one\ndescription: Changed version\n---\nNew steps")
            ),
        )
    )
    result = await service.organize_skills(request.model_copy(update={"graph_id": built.graph_id}))
    assert result.graph_version_id != built.graph_version_id
    assert updated.version.version_id in result.skill_version_ids
    store = SkillGraphStore(
        service.repository.persistence.database, graph_id=built.graph_id, actor_id="test", version_ids={}
    )
    old_nodes, _, _ = await store.load(built.graph_version_id)
    new_nodes, _, _ = await store.load(result.graph_version_id)
    assert [
        node.model_dump(exclude={"graph_version_id"})
        for node in old_nodes
        if node.skill_version_id == two.version.version_id
    ] == [
        node.model_dump(exclude={"graph_version_id"})
        for node in new_nodes
        if node.skill_version_id == two.version.version_id
    ]
    assert all(node.skill_version_id != one.version.version_id for node in new_nodes)
    assert len(extracted) == (3 if build_scene else 0)


@pytest.mark.parametrize("concurrency", [1, 2, 4])
async def test_enrichment_respects_concurrency(graph_service, monkeypatch, concurrency):
    """Overlap enrichment calls up to the configured limit and retain each result."""
    from mindmemos_skill.algos.manage.skill_manage.enrichment import SkillEnricher

    service, _, _ = graph_service
    items = [await register(service, f"parallel-enrich-{index}") for index in range(4)]
    pipeline = service._manage_pipeline
    pipeline.config = pipeline.config.model_copy(
        update={
            "algorithm": pipeline.config.algorithm.model_copy(
                update={"enrich_skill": True, "enrichment_concurrency": concurrency}
            )
        }
    )
    active = 0
    peak = 0
    calls = []
    admitted = asyncio.Event()

    async def enrich(self, node, document):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append(node.skill_version_id)
        if active == concurrency:
            admitted.set()
        try:
            await asyncio.wait_for(admitted.wait(), timeout=2)
            await asyncio.sleep(0)
            return node.model_copy(update={"example_tasks": [node.skill_version_id]})
        finally:
            active -= 1

    monkeypatch.setattr(SkillEnricher, "enrich", enrich)
    built = await service.organize_skills(
        OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id for item in items])
    )
    store = SkillGraphStore(
        service.repository.persistence.database,
        graph_id=built.graph_id,
        actor_id="test",
        version_ids={},
    )
    nodes, _, _ = await store.load(built.graph_version_id)
    assert peak == concurrency
    assert active == 0
    assert set(calls) == {item.version.version_id for item in items}
    assert len(calls) == len(items)
    assert all(node.example_tasks == [node.skill_version_id] for node in nodes if node.node_type == GraphNodeType.SKILL)


def test_enrichment_concurrency_must_be_positive(graph_service):
    """Reject a concurrency limit that cannot admit enrichment work."""
    service, _, _ = graph_service
    config = service._manage_pipeline.config.algorithm
    with pytest.raises(ValidationError, match="enrichment_concurrency"):
        type(config).model_validate({**config.model_dump(), "enrichment_concurrency": 0})


async def test_joint_edge_records_preserve_first_evidence_and_distinct_relations(graph_service):
    """Deduplicate exact edge keys without collapsing reverse or differently typed edges."""
    service, _, _ = graph_service
    pipeline = service._manage_pipeline
    detail = await register(service, "edge-precedence")
    request = OrganizeRequest(project_id="project", skill_refs=[detail.skill.skill_id])
    versions = await pipeline._resolve_skill_versions(request.skill_refs)
    store = pipeline._graph_store(request, "skg-edge-precedence", versions, None)
    algorithm = pipeline._algorithm(store)
    drafts = [
        {"source": "a", "target": "b", "type": "dependency", "weight": 0.8, "description": "Rule evidence"},
        {"source": "a", "target": "b", "type": "dependency", "weight": 0.9, "description": "Model evidence"},
        {"source": "b", "target": "a", "type": "dependency", "weight": 0.4},
        {"source": "a", "target": "b", "type": "semantic", "weight": 0.3, "confidence": 0.6},
    ]
    before = deepcopy(drafts)
    edges = algorithm._make_joint_edge_records("skgv-edge-precedence", drafts)
    assert [(edge.source_node_id, edge.target_node_id, edge.edge_type) for edge in edges] == [
        ("a", "b", "dependency"),
        ("b", "a", "dependency"),
        ("a", "b", "semantic"),
    ]
    assert edges[0].weight == 0.8
    assert edges[0].metadata["evidence"] == "Rule evidence"
    assert edges[2].metadata["confidence"] == 0.6
    assert all(edge.graph_version_id == "skgv-edge-precedence" for edge in edges)
    assert len({edge.edge_id for edge in edges}) == 3
    assert drafts == before


@pytest.mark.parametrize("mode", ["skill_only", "separate", "unified_ppr", "fuse"])
@pytest.mark.parametrize("all_without_scenes", [False, True])
async def test_insufficient_source_preserves_members_and_retrieval(
    graph_service, monkeypatch, scene_record, mode, all_without_scenes
):
    """Mixed and Scene-free snapshots preserve all members through cache, reload and retrieval."""
    from mindmemos_skill.algos.manage.skill_manage.errors import SceneGraphError, SceneGraphFailure

    service, _, _ = graph_service
    pipeline = service._manage_pipeline
    pipeline.config = pipeline.config.model_copy(
        update={"algorithm": pipeline.config.algorithm.model_copy(update={"mode": mode})}
    )
    calls = []

    async def extract(self, document):
        calls.append(document)
        if all_without_scenes or "name: knowledge" in document:
            raise SceneGraphError(SceneGraphFailure.INSUFFICIENT_SOURCE)
        return scene_record.model_copy(deep=True)

    monkeypatch.setattr(SceneExtractor, "extract", extract)
    knowledge, workflow = await register(service, "knowledge"), await register(service, "workflow")
    request = OrganizeRequest(project_id="project", skill_refs=[knowledge.skill.skill_id, workflow.skill.skill_id])
    built = await service.organize_skills(request)
    preview = await service.preview_graph(built.graph_id)
    skill_nodes = [node for node in preview["nodes"] if node["node_type"] == GraphNodeType.SKILL]
    scene_nodes = [node for node in preview["nodes"] if node["node_type"] == GraphNodeType.SCENE]
    assert len(skill_nodes) == 2
    assert bool(scene_nodes) is not all_without_scenes
    assert set(built.skill_version_ids) == {knowledge.version.version_id, workflow.version.version_id}
    cache = await service.repository.persistence.skills.get_scene_data(knowledge.version.version_id)
    assert cache is not None and cache.record is None
    reused = await service.organize_skills(request.model_copy(update={"graph_id": built.graph_id}))
    assert reused.graph_version_id == built.graph_version_id and len(calls) == 2
    for strategy, include_scenes in [("graph", False), ("vector", False), ("vector", True)]:
        result = await service.recommend_skills(
            RecommendRequest(
                project_id="project",
                graph_id=built.graph_id,
                graph_version_id=built.graph_version_id,
                task="Process knowledge workflow table report",
                top_k=2,
                strategy=strategy,
                include_scenes=include_scenes,
            )
        )
        assert {skill.version_id for skill in result.skills} == set(built.skill_version_ids)
    removed = await service.remove_graph_members(
        RemoveGraphMembersRequest(
            project_id="project",
            graph_ref=built.graph_id,
            skill_refs=[workflow.skill.skill_id],
        )
    )
    assert removed.skill_version_ids == [knowledge.version.version_id]
    assert all(
        node["node_type"] == GraphNodeType.SKILL for node in (await service.preview_graph(built.graph_id))["nodes"]
    )
    added = await service.add_graph_members(
        AddGraphMembersRequest(
            project_id="project",
            graph_ref=built.graph_id,
            skill_refs=[workflow.skill.skill_id],
        )
    )
    assert set(added.skill_version_ids) == set(built.skill_version_ids)
    assert len(calls) == 2


async def test_force_extraction_replaces_negative_scene_cache(graph_service, monkeypatch, scene_record):
    """Explicit refresh can add Scenes to an unchanged Skill after a negative cached result."""
    from mindmemos_skill.algos.manage.skill_manage.errors import SceneGraphError, SceneGraphFailure

    service, _, _ = graph_service
    item = await register(service, "knowledge")

    async def insufficient(self, document):
        raise SceneGraphError(SceneGraphFailure.INSUFFICIENT_SOURCE)

    monkeypatch.setattr(SceneExtractor, "extract", insufficient)
    request = OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id])
    before = await service.organize_skills(request)

    async def extracted(self, document):
        return scene_record.model_copy(deep=True)

    monkeypatch.setattr(SceneExtractor, "extract", extracted)
    after = await service.organize_skills(
        request.model_copy(
            update={
                "graph_id": before.graph_id,
                "force_extract_scene": True,
            }
        )
    )
    assert after.graph_version_id != before.graph_version_id
    assert any(
        node["node_type"] == GraphNodeType.SCENE for node in (await service.preview_graph(before.graph_id))["nodes"]
    )


async def test_model_failure_is_not_cached_as_absent_scenes(graph_service, monkeypatch):
    """Transport or structural failures still abort publication and leave no negative cache."""
    from mindmemos_skill.algos.manage.skill_manage.errors import SceneGraphError, SceneGraphFailure

    service, _, _ = graph_service
    item = await register(service, "broken")

    async def failed(self, document):
        raise SceneGraphError(SceneGraphFailure.MODEL_FAILED)

    monkeypatch.setattr(SceneExtractor, "extract", failed)
    with pytest.raises(SceneGraphError, match="SCENE_GRAPH_MODEL_FAILED"):
        await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id]))
    assert await service.repository.persistence.skills.get_scene_data(item.version.version_id) is None


async def test_platform_embedding_space_stays_with_snapshot(tmp_path, monkeypatch):
    """Builds select the active space while retrieval uses the pinned snapshot space."""
    secret = "test-graph-model-signing-key".ljust(32, "-")
    monkeypatch.setenv(PLATFORM_MODEL_SECRET_ENV, secret)
    active_space = "revision-a"
    available_spaces = {"revision-a", "revision-b"}
    requests = []

    def respond(request):
        token = request.headers["Authorization"].removeprefix("Bearer ")
        claims = jwt.decode(token, secret, algorithms=["HS256"], audience="platform-skill-model")
        assert claims["project_id"] == "project"
        if request.url.path.endswith("/embedding-profile"):
            return httpx.Response(200, json={
                "space_id": active_space, "model": "fixture-model", "dimensions": 3,
            })
        assert request.url.path.endswith("/embeddings")
        payload = json.loads(request.content)
        space = payload["space_id"]
        assert payload["dimensions"] == 3
        requests.append(space)
        if space not in available_spaces:
            return httpx.Response(409, json={"detail": "Snapshot embedding space is unavailable"})
        vector = [1.0, 0.0, 0.0] if space == "revision-a" else [0.0, 1.0, 0.0]
        return httpx.Response(200, json={
            "object": "list", "model": "fixture-model",
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, _ in enumerate(payload["input"])
            ],
        })

    def client(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", client)
    monkeypatch.setattr(platform, "httpx", SimpleNamespace(
        AsyncClient=client, HTTPStatusError=httpx.HTTPStatusError, RequestError=httpx.RequestError,
    ))
    config = ServiceConfig.model_validate({
        "home": tmp_path / "home",
        "platform_model_origin": "http://platform.invalid",
        "persistence": {"database": {"class_name": "sqlite"}},
        "repository_pipeline": {"class_name": "repository"},
        "manage_pipeline": {
            "class_name": "graph_manage",
            "config": {"algorithm": {
                "embedding": {"transport": "platform_gateway", "model": "placeholder", "dimensions": 1},
                "chat": {"model": "unused"},
                "build_scene": False,
                "llm_linking": False,
                "scene_llm_linking": False,
                "enrich_skill": False,
                "query_rewrite": False,
            }},
        },
    })

    def authorize(principal, project):
        assert principal == project == "project"
        return Scope(fields={"project_id": project})

    async with await create_service(config, project_authorizer=authorize) as service:
        with service.request_principal("project"):
            item = await register(service, "platform-space")
            organize = OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id])
            first = await service.organize_skills(organize)
            assert requests and set(requests) == {"revision-a"}
            active_space = "revision-b"
            second = await service.organize_skills(organize.model_copy(update={"graph_id": first.graph_id}))
            assert second.graph_version_id != first.graph_version_id
            assert requests[-1] == "revision-b"

            query = RecommendRequest(
                project_id="project", graph_id=first.graph_id, graph_version_id=first.graph_version_id,
                task="process the table", strategy="vector",
            )
            assert (await service.recommend_skills(query)).skills
            assert requests[-1] == "revision-a"
            available_spaces.remove("revision-a")
            with pytest.raises(PlatformModelError) as failure:
                await service.recommend_skills(query)
            assert failure.value.status_code == 409
