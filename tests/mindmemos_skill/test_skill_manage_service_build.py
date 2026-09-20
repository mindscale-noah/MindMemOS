"""P4 orchestration skeleton: build_skill_index over the storage port."""

from __future__ import annotations

import pytest
from mindmemos_skill.application.skill_manage.models import (
    SkillManageBuildRequest,
    SkillManageOperationResult,
    SkillManageRemoveRequest,
    SkillManageRetrieveRequest,
    SkillManageUpsertRequest,
)
from mindmemos_skill.application.skill_manage.service import SkillManageService
from mindmemos_skill.application.skill_manage.state_machine import OperationStatus
from mindmemos_skill.errors import SkillNotFoundError
from mindmemos_skill.infra.skill_manage import generation_id_from_operation
from mindmemos_skill.typing import (
    EmbeddingProfile,
    SkillGraphEdge,
    SkillGraphMembership,
    SkillGraphNode,
    SkillManageBuildInput,
    SkillManageBuildPlan,
    SkillManageEmbedInput,
    SkillManageEmbedOutput,
    SkillManageRemoveInput,
    SkillManageRemoveSelector,
    SkillManageRetrieveInput,
    SkillManageRetrieveOutput,
    SkillManageScope,
    SkillVersionRef,
)
from test_skill_manage_storage_contract import InMemorySkillGraphStorageAdapter

PROFILE = EmbeddingProfile(model="stub", dimension=4)
OPERATION_K1 = "8649e81a-00c6-59ae-93e2-4c95d278cf5c"  # uuid5('skill-manage:p1:g1:k1')


async def _seed_generation(storage, scope, generation_id, *, nodes=(), edges=(), memberships=()) -> None:
    """Publish one generation directly through the adapter (stage -> ready -> active)."""

    await storage.stage_generation(
        scope, generation_id, operation_id="seed-op", attempt_no=1, base_revision=None, target_revision=0
    )
    await storage.write_snapshot(
        scope, generation_id, nodes=list(nodes), edges=list(edges), memberships=list(memberships)
    )
    assert await storage.mark_generation_ready(
        scope, generation_id, node_count=len(nodes), edge_count=len(edges), vector_count=0, counts_hash="seed"
    )
    assert await storage.publish_generation(scope, generation_id, expected_revision=None, new_revision=0)


class FakeBuildAlgorithm:
    def __init__(self) -> None:
        self.requests: list[SkillManageBuildInput] = []

    async def plan_build(self, request: SkillManageBuildInput) -> SkillManageBuildPlan:
        self.requests.append(request)
        return SkillManageBuildPlan(
            scope=request.scope,
            nodes=(
                SkillGraphNode(graph_node_id="n1", node_type="skill", skill_id="demo", version_id="v1", name="Demo"),
                SkillGraphNode(graph_node_id="n2", node_type="skill", skill_id="other", version_id="v1", name="Other"),
            ),
            edges=(SkillGraphEdge(edge_id="e1", edge_type="dependency", source_id="n1", target_id="n2"),),
            memberships=(
                SkillGraphMembership(skill_id="demo", version_id="v1", graph_node_id="n1"),
                SkillGraphMembership(skill_id="other", version_id="v1", graph_node_id="n2"),
            ),
        )

    async def plan_remove(self, request: SkillManageRemoveInput) -> SkillManageBuildPlan:
        target = request.selector.skill_id
        nodes = tuple(node for node in request.snapshot.nodes if node.node_type != "skill" or node.skill_id != target)
        return SkillManageBuildPlan(
            scope=request.scope,
            nodes=nodes,
            edges=tuple(
                edge
                for edge in request.snapshot.edges
                if edge.source_id in {node.graph_node_id for node in nodes}
                and edge.target_id in {node.graph_node_id for node in nodes}
            ),
            memberships=tuple(
                membership for membership in request.snapshot.memberships if membership.skill_id != target
            ),
        )

    async def plan_upsert(self, request):
        return SkillManageBuildPlan(
            scope=request.scope,
            nodes=request.snapshot.nodes,
            edges=request.snapshot.edges,
            memberships=request.snapshot.memberships,
        )

    async def embed(self, request: SkillManageEmbedInput) -> SkillManageEmbedOutput:
        return SkillManageEmbedOutput(
            embedding_profile=request.embedding_profile,
            vectors=tuple((1.0,) + (0.0,) * (request.embedding_profile.dimension - 1) for _ in request.texts),
        )

    async def retrieve(self, request: SkillManageRetrieveInput) -> SkillManageRetrieveOutput:
        return SkillManageRetrieveOutput(
            generation_id=request.snapshot.generation_id,
            revision=request.snapshot.revision,
            hits=(),
        )


@pytest.mark.asyncio
async def test_build_skill_index_persist_commits_and_publishes() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    counter = iter(["op-1"])
    service = SkillManageService(
        algorithm=FakeBuildAlgorithm(),
        storage=storage,
        embedding_profile=PROFILE,
        id_generator=lambda: next(counter),
    )

    result = await service.build_skill_index(
        SkillManageBuildRequest(
            project_id="p1",
            graph_id="g1",
            idempotency_key="k1",
            version_refs=(),
        )
    )

    assert isinstance(result, SkillManageOperationResult)
    assert result.status is OperationStatus.COMMITTED
    assert result.operation_id == OPERATION_K1
    assert result.result_revision == 0
    assert result.node_count == 2
    assert result.edge_count == 1

    head = await storage.get_graph_head(SkillManageScope(project_id="p1", graph_id="g1"))
    assert head is not None
    assert head["active_revision"] == 0
    assert head["active_generation_id"] == generation_id_from_operation(OPERATION_K1, 1)


@pytest.mark.asyncio
async def test_build_skill_index_dry_run_has_no_persistence() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    service = SkillManageService(algorithm=FakeBuildAlgorithm(), storage=storage, embedding_profile=PROFILE)

    result = await service.build_skill_index(
        SkillManageBuildRequest(
            project_id="p1",
            graph_id="g1",
            commit_policy="dry_run",
        )
    )

    assert result.operation_id is None
    assert result.node_count == 2
    assert await storage.get_graph_head(SkillManageScope(project_id="p1", graph_id="g1")) is None
    assert await storage.get_operation(SkillManageScope(project_id="p1", graph_id="g1"), "unused") is None


@pytest.mark.asyncio
async def test_retrieve_skills_reads_active_head_generation() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    counter = iter(["op-1"])
    service = SkillManageService(
        algorithm=FakeBuildAlgorithm(),
        storage=storage,
        embedding_profile=PROFILE,
        id_generator=lambda: next(counter),
    )

    await service.build_skill_index(SkillManageBuildRequest(project_id="p1", graph_id="g1", idempotency_key="k1"))

    result = await service.retrieve_skills(SkillManageRetrieveRequest(project_id="p1", graph_id="g1", query="demo"))

    assert result.query == "demo"
    assert result.generation_id == generation_id_from_operation(OPERATION_K1, 1)
    assert result.revision == 0


@pytest.mark.asyncio
async def test_retrieve_skills_missing_graph_raises() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    service = SkillManageService(algorithm=FakeBuildAlgorithm(), storage=storage, embedding_profile=PROFILE)

    with pytest.raises(SkillNotFoundError):
        await service.retrieve_skills(SkillManageRetrieveRequest(project_id="p1", graph_id="missing", query="x"))


@pytest.mark.asyncio
async def test_remove_skill_from_index_commits_new_generation() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    counter = iter(["op-remove"])
    service = SkillManageService(
        algorithm=FakeBuildAlgorithm(),
        storage=storage,
        embedding_profile=PROFILE,
        id_generator=lambda: next(counter),
    )

    scope = SkillManageScope(project_id="p1", graph_id="g1")
    await _seed_generation(
        storage,
        scope,
        "seed:1",
        nodes=(
            SkillGraphNode(graph_node_id="n-csv", node_type="skill", skill_id="csv", version_id="v1", name="CSV"),
            SkillGraphNode(
                graph_node_id="n-filter", node_type="skill", skill_id="filter", version_id="v1", name="Filter"
            ),
        ),
        edges=(
            SkillGraphEdge(edge_id="e1", edge_type="dependency", source_id="n-csv", target_id="n-filter", weight=1.0),
        ),
        memberships=(
            SkillGraphMembership(skill_id="csv", version_id="v1", graph_node_id="n-csv"),
            SkillGraphMembership(skill_id="filter", version_id="v1", graph_node_id="n-filter"),
        ),
    )

    result = await service.remove_skill_from_index(
        SkillManageRemoveRequest(
            project_id="p1",
            graph_id="g1",
            idempotency_key="k2",
            selector=SkillManageRemoveSelector(kind="family", skill_id="csv"),
        )
    )

    assert result.status is OperationStatus.COMMITTED
    assert result.result_revision == 1
    head = await storage.get_graph_head(scope)
    assert head is not None
    assert head["active_revision"] == 1
    snapshot = await storage.load_snapshot(scope, head["active_generation_id"])
    assert {node.skill_id for node in snapshot.nodes} == {"filter"}


@pytest.mark.asyncio
async def test_upsert_skill_index_commits_new_generation() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    counter = iter(["op-upsert"])
    service = SkillManageService(
        algorithm=FakeBuildAlgorithm(),
        storage=storage,
        embedding_profile=PROFILE,
        id_generator=lambda: next(counter),
    )

    scope = SkillManageScope(project_id="p1", graph_id="g1")
    await _seed_generation(storage, scope, "seed:1")

    result = await service.upsert_skill_index(
        SkillManageUpsertRequest(
            project_id="p1",
            graph_id="g1",
            idempotency_key="k3",
            version_ref=SkillVersionRef(skill_id="csv", version_id="v2"),
        )
    )

    assert result.status is OperationStatus.COMMITTED
    assert result.result_revision == 1
    head = await storage.get_graph_head(scope)
    assert head is not None
    assert head["active_revision"] == 1
