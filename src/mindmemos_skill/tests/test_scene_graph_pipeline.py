"""Scene cache, atomic membership, immutable history and strategy integration."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatConfig
from mindmemos_skill.algos.manage.skill_manage.algorithm import SkillManageAlgorithm
from mindmemos_skill.algos.manage.skill_manage.extraction import SceneExtractor
from mindmemos_skill.api.typing import (
    AddGraphMembersRequest,
    OrganizeRequest,
    RecommendRequest,
    RegisterRequest,
    RemoveGraphMembersRequest,
)
from mindmemos_skill.errors import GraphConflictError, GraphNotFoundError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.constants import SKILL_GRAPH_VERSIONS_TABLE, SKILL_VERSIONS_TABLE
from mindmemos_skill.persistence.db_services.skill_management.writer import SkillGraphStore
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


@pytest.mark.parametrize("concurrency", [1, 2, 4])
async def test_scene_processing_respects_concurrency_and_reuses_cache(
    graph_service, monkeypatch, scene_record, concurrency, caplog
):
    service, _, _ = graph_service
    pipeline = service._manage_pipeline
    pipeline.config.scene_concurrency = concurrency
    items = [await register(service, f"parallel-{index}") for index in range(5)]
    versions = await pipeline._resolve_skill_versions("project", [item.version.version_id for item in items])
    sources = await pipeline._get_skill_by_versions(versions)
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
        records = await asyncio.wait_for(pipeline._scene_records("project", versions, sources, False), timeout=5)
        cached = await pipeline._scene_records("project", versions, sources, False)
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
    versions = await pipeline._resolve_skill_versions("project", [item.version.version_id for item in items])
    sources = await pipeline._get_skill_by_versions(versions)
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
        await asyncio.wait_for(pipeline._scene_records("project", versions, sources, False), timeout=5)
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
    assert {node["node_type"] for node in before["nodes"]} == {"skill", "scene"}
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
                project_id="project", graph_ref=f"skgv-{changed.graph_version_id.hex}", skill_refs=[one.skill.skill_id]
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
    assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == f"skgv-{built.graph_version_id.hex}"


async def test_head_check_rejects_stale_publication(graph_service):
    service, _, _ = graph_service
    item = await register(service, "conflict")
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id]))
    store = SkillGraphStore(
        service.repository.persistence.database,
        project_id="project",
        graph_id=built.graph_id,
        actor_id="project",
        version_ids={item.skill.skill_id: item.version.version_id},
        expected_snapshot=uuid4(),
        check_head=True,
    )
    with pytest.raises(GraphConflictError):
        await store.select_snapshot(built.graph_version_id)
    assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == f"skgv-{built.graph_version_id.hex}"


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


async def test_concurrent_additions_detect_conflict_instead_of_losing_members(graph_service):
    service, embeddings, _ = graph_service
    one, two, three = await register(service, "one"), await register(service, "two"), await register(service, "three")
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[one.skill.skill_id]))
    original = embeddings.embed
    barrier = asyncio.Barrier(2)

    async def embed(texts):
        await barrier.wait()
        return await original(texts)

    embeddings.embed = embed
    outcomes = await asyncio.gather(
        service.add_graph_members(
            AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[two.skill.skill_id])
        ),
        service.add_graph_members(
            AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[three.skill.skill_id])
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(value, GraphConflictError) for value in outcomes) == 1
    published = next(value for value in outcomes if not isinstance(value, Exception))
    assert one.version.version_id in published.skill_version_ids and len(published.skill_version_ids) == 2
    assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == f"skgv-{published.graph_version_id.hex}"


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
    assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == f"skgv-{built.graph_version_id.hex}"
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


async def test_query_config_reuses_but_build_config_and_vector_profile_rebuild(graph_service, monkeypatch):
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
    assert rebuilt.graph_version_id != built.graph_version_id
    profile_change = build_change.model_copy(
        update={"embedding": build_change.embedding.model_copy(update={"model": "fake-v2"})}
    )
    pipeline.config = pipeline.config.model_copy(update={"algorithm": profile_change})
    embeddings.config = profile_change.embedding
    changed = await service.organize_skills(request)
    assert changed.graph_version_id != rebuilt.graph_version_id


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
                    node_type="skill",
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


async def test_independent_database_clients_enforce_head_conflict(graph_service, tmp_path):
    from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
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
            database,
            project_id="project",
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
        snapshot = uuid4()
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
        assert (await service.resolve_graph(built.graph_id))["graph_version_id"] == f"skgv-{successful.hex}"
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
    assert {node["node_type"] for node in (await service.preview_graph(built.graph_id))["nodes"]} == {"skill"}
    forced = await service.organize_skills(
        request.model_copy(update={"graph_id": built.graph_id, "force_extract_scene": True})
    )
    assert len(extracted) == 1 and forced.graph_version_id == built.graph_version_id
    assert await service.repository.persistence.skills.get_scene_data(item.version.version_id) is not None
    pipeline.config = pipeline.config.model_copy(update={"algorithm": original})
    with_scenes = await service.organize_skills(request.model_copy(update={"graph_id": built.graph_id}))
    assert with_scenes.graph_version_id != built.graph_version_id
    assert {node["node_type"] for node in (await service.preview_graph(built.graph_id))["nodes"]} == {"skill", "scene"}
    assert len(extracted) == 1


async def test_scene_extraction_order_matches_original_sorted_members(graph_service):
    service, _, extracted = graph_service
    items = [await register(service, name) for name in ("order-one", "order-two", "order-three")]
    inputs = list(reversed(sorted(items, key=lambda item: item.skill.skill_id)))
    await service.organize_skills(
        OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id for item in inputs])
    )
    assert extracted == [item.version.content for item in sorted(items, key=lambda item: item.skill.skill_id)]


async def test_incremental_enrichment_uses_original_unfilled_fields(graph_service, monkeypatch):
    from mindmemos_skill.algos.manage.skill_manage.enrichment import SkillEnricher

    service, _, _ = graph_service
    one, two = await register(service, "enrich-one"), await register(service, "enrich-two")
    pipeline = service._manage_pipeline
    pipeline.config = pipeline.config.model_copy(
        update={"algorithm": pipeline.config.algorithm.model_copy(update={"enrich_skill": True})}
    )
    phase = {"value": "first"}

    async def enrich(self, node, document):
        return node.model_copy(update={"example_tasks": node.example_tasks or [phase["value"]]})

    monkeypatch.setattr(SkillEnricher, "enrich", enrich)
    built = await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[one.skill.skill_id]))
    phase["value"] = "second"
    added = await service.add_graph_members(
        AddGraphMembersRequest(project_id="project", graph_ref=built.graph_id, skill_refs=[two.skill.skill_id])
    )
    store = SkillGraphStore(
        service.repository.persistence.database,
        project_id="project",
        graph_id=built.graph_id,
        actor_id="test",
        version_ids={},
    )
    old_nodes, _, _ = await store.load(built.graph_version_id)
    new_nodes, _, _ = await store.load(added.graph_version_id)
    assert next(node for node in old_nodes if node.node_type == "skill").example_tasks == ["first"]
    assert all(node.example_tasks == ["second"] for node in new_nodes if node.node_type == "skill")
