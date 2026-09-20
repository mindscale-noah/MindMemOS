"""SkillManageService embedding path: vector reuse, KNN seeding, profiles and idempotency."""

from __future__ import annotations

import hashlib
import math
from uuid import NAMESPACE_URL, uuid5

import pytest
from mindmemos_skill.application.skill_manage.loader import SkillVersionNotFound
from mindmemos_skill.application.skill_manage.models import (
    SkillManageBuildRequest,
    SkillManageRemoveRequest,
    SkillManageRetrieveRequest,
    SkillManageUpsertRequest,
    compute_skill_manage_request_hash,
)
from mindmemos_skill.application.skill_manage.service import SkillManageService
from mindmemos_skill.application.skill_manage.state_machine import OperationStatus
from mindmemos_skill.errors import SkillConfigurationError, SkillManageProfileMismatchError
from mindmemos_skill.infra.skill_manage import generation_id_from_operation
from mindmemos_skill.typing import (
    EmbeddingProfile,
    SkillDocumentInput,
    SkillGraphEmbedding,
    SkillGraphEmbeddingText,
    SkillGraphMembership,
    SkillGraphNode,
    SkillManageBuildInput,
    SkillManageBuildPlan,
    SkillManageEmbedInput,
    SkillManageEmbedOutput,
    SkillManageHit,
    SkillManageMutationPlan,
    SkillManageRemoveInput,
    SkillManageRemoveSelector,
    SkillManageRetrieveInput,
    SkillManageRetrieveOutput,
    SkillManageScope,
    SkillManageUpsertInput,
    SkillVersionRef,
)
from test_skill_manage_storage_contract import InMemorySkillGraphStorageAdapter

PROFILE = EmbeddingProfile(model="stub", dimension=4)
OTHER_PROFILE = EmbeddingProfile(model="other", dimension=4)
SCOPE = SkillManageScope(project_id="p1", graph_id="g1")
QUERY = "find beta then alpha"

DOCS: dict[tuple[str, str], str] = {
    ("alpha", "v1"): "alpha",
    ("beta", "v1"): "beta",
    ("gamma", "v1"): "gamma",
    ("alpha", "v2"): "alpha v2",
}
REFS = (
    SkillVersionRef(skill_id="alpha", version_id="v1"),
    SkillVersionRef(skill_id="beta", version_id="v1"),
    SkillVersionRef(skill_id="gamma", version_id="v1"),
)
REFS_ALPHA_V2 = (SkillVersionRef(skill_id="alpha", version_id="v2"),) + REFS[1:]

# Unit basis vectors per skill text; the query leans toward beta, then alpha, then gamma.
VECTORS: dict[str, tuple[float, ...]] = {
    "alpha": (1.0, 0.0, 0.0, 0.0),
    "beta": (0.0, 1.0, 0.0, 0.0),
    "gamma": (0.0, 0.0, 1.0, 0.0),
    "alpha v2": (0.0, 0.0, 0.0, 1.0),
    QUERY: (0.5, 0.8, 0.2, 0.0),
}


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_for(text: str, dimension: int) -> tuple[float, ...]:
    known = VECTORS.get(text)
    if known is not None:
        return known
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return tuple(digest[index] / 255.0 for index in range(dimension))


def _node(skill_id: str, version_id: str, content: str) -> SkillGraphNode:
    return SkillGraphNode(
        graph_node_id=f"skill:{skill_id}",
        node_type="skill",
        skill_id=skill_id,
        version_id=version_id,
        name=skill_id,
        description=content,
        content_hash=_content_hash(content),
    )


def _texts(nodes: tuple[SkillGraphNode, ...]) -> tuple[SkillGraphEmbeddingText, ...]:
    return tuple(
        SkillGraphEmbeddingText(
            graph_node_id=node.graph_node_id, content_hash=node.content_hash or "", text=node.description
        )
        for node in nodes
        if node.node_type == "skill"
    )


def _memberships(nodes: tuple[SkillGraphNode, ...]) -> tuple[SkillGraphMembership, ...]:
    return tuple(
        SkillGraphMembership(
            skill_id=node.skill_id or node.name, version_id=node.version_id or "", graph_node_id=node.graph_node_id
        )
        for node in nodes
        if node.node_type == "skill"
    )


class FakeVersionLoader:
    async def load(self, ref: SkillVersionRef, *, project_id: str) -> SkillDocumentInput:
        content = DOCS.get((ref.skill_id, ref.version_id))
        if content is None:
            raise SkillVersionNotFound(ref.skill_id, ref.version_id)
        return SkillDocumentInput(skill_id=ref.skill_id, version_id=ref.version_id, content=content)


class RecordingAlgorithm:
    """Plans from the documents it is given, embeds by lookup table and records every call."""

    def __init__(self) -> None:
        self.plan_build_calls: list[SkillManageBuildInput] = []
        self.plans: list[SkillManageBuildPlan] = []
        self.plan_upsert_calls: list[SkillManageUpsertInput] = []
        self.plan_remove_calls: list[SkillManageRemoveInput] = []
        self.embed_calls: list[SkillManageEmbedInput] = []
        self.retrieve_calls: list[SkillManageRetrieveInput] = []

    async def plan_build(self, request: SkillManageBuildInput) -> SkillManageBuildPlan:
        self.plan_build_calls.append(request)
        nodes = tuple(_node(doc.skill_id, doc.version_id, doc.content) for doc in request.documents)
        plan = SkillManageBuildPlan(
            scope=request.scope,
            nodes=nodes,
            edges=(),
            memberships=_memberships(nodes),
            embedding_texts=_texts(nodes),
        )
        self.plans.append(plan)
        return plan

    async def plan_upsert(self, request: SkillManageUpsertInput) -> SkillManageMutationPlan:
        self.plan_upsert_calls.append(request)
        nodes = request.snapshot.nodes
        document = request.document
        if document is not None:
            replacement = _node(document.skill_id, document.version_id, document.content)
            nodes = tuple(replacement if node.skill_id == document.skill_id else node for node in nodes)
            if replacement not in nodes:
                nodes = (*nodes, replacement)
        return SkillManageMutationPlan(
            scope=request.scope,
            nodes=nodes,
            edges=request.snapshot.edges,
            memberships=_memberships(nodes),
            embedding_texts=_texts(nodes),
        )

    async def plan_remove(self, request: SkillManageRemoveInput) -> SkillManageMutationPlan:
        self.plan_remove_calls.append(request)
        target = request.selector.skill_id
        nodes = tuple(node for node in request.snapshot.nodes if node.skill_id != target)
        return SkillManageMutationPlan(
            scope=request.scope,
            nodes=nodes,
            edges=request.snapshot.edges,
            memberships=_memberships(nodes),
            embedding_texts=_texts(nodes),
        )

    async def embed(self, request: SkillManageEmbedInput) -> SkillManageEmbedOutput:
        self.embed_calls.append(request)
        dimension = request.embedding_profile.dimension
        return SkillManageEmbedOutput(
            embedding_profile=request.embedding_profile,
            vectors=tuple(_vector_for(text, dimension) for text in request.texts),
        )

    async def retrieve(self, request: SkillManageRetrieveInput) -> SkillManageRetrieveOutput:
        self.retrieve_calls.append(request)
        hits = tuple(
            SkillManageHit(
                skill_id=node.skill_id or node.name,
                version_id=node.version_id or "",
                name=node.name,
                graph_node_id=node.graph_node_id,
                score=request.vector_similarity.get(node.name, 0.0),
                vector_score=request.vector_similarity.get(node.name),
            )
            for node in request.snapshot.nodes
            if node.node_type == "skill"
        )
        return SkillManageRetrieveOutput(
            generation_id=request.snapshot.generation_id,
            revision=request.snapshot.revision,
            hits=hits,
        )


class RecordingStorage(InMemorySkillGraphStorageAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.write_embeddings_calls: list[tuple[str, list[SkillGraphEmbedding]]] = []

    async def write_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        embeddings: list[SkillGraphEmbedding],
    ) -> None:
        self.write_embeddings_calls.append((generation_id, list(embeddings)))
        await super().write_embeddings(scope, generation_id, embeddings=embeddings)


def _service(
    storage: InMemorySkillGraphStorageAdapter,
    algorithm: RecordingAlgorithm,
    *,
    profile: EmbeddingProfile | None = PROFILE,
) -> SkillManageService:
    return SkillManageService(
        algorithm=algorithm,
        storage=storage,
        embedding_profile=profile,
        version_loader=FakeVersionLoader(),
    )


def _build(key: str | None, refs: tuple[SkillVersionRef, ...] = REFS, **overrides: object) -> SkillManageBuildRequest:
    return SkillManageBuildRequest(project_id="p1", graph_id="g1", idempotency_key=key, version_refs=refs, **overrides)


def _upsert(key: str, ref: SkillVersionRef, **overrides: object) -> SkillManageUpsertRequest:
    return SkillManageUpsertRequest(project_id="p1", graph_id="g1", idempotency_key=key, version_ref=ref, **overrides)


def _remove(key: str, skill_id: str, **overrides: object) -> SkillManageRemoveRequest:
    return SkillManageRemoveRequest(
        project_id="p1",
        graph_id="g1",
        idempotency_key=key,
        selector=SkillManageRemoveSelector(kind="family", skill_id=skill_id),
        **overrides,
    )


def _retrieve(**overrides: object) -> SkillManageRetrieveRequest:
    return SkillManageRetrieveRequest(project_id="p1", graph_id="g1", query=QUERY, **overrides)


def _operation_id(key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"skill-manage:p1:g1:{key}"))


async def _publish_legacy(
    storage: InMemorySkillGraphStorageAdapter, generation_id: str, nodes: list[SkillGraphNode]
) -> None:
    """Publish a generation straight through the adapter without vectors or a profile."""

    await storage.stage_generation(
        SCOPE, generation_id, operation_id="legacy-op", attempt_no=1, base_revision=None, target_revision=0
    )
    await storage.write_snapshot(SCOPE, generation_id, nodes=nodes, edges=[], memberships=[])
    assert await storage.mark_generation_ready(
        SCOPE, generation_id, node_count=len(nodes), edge_count=0, vector_count=0, counts_hash="legacy"
    )
    assert await storage.publish_generation(SCOPE, generation_id, expected_revision=None, new_revision=0)


async def _head(storage: InMemorySkillGraphStorageAdapter) -> dict[str, object]:
    head = await storage.get_graph_head(SCOPE)
    assert head is not None
    return head


@pytest.mark.asyncio
async def test_build_embeds_every_node_and_publishes_profile() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)

    result = await service.build_skill_index(_build("k1"))

    assert result.status is OperationStatus.COMMITTED
    assert result.result_revision == 0
    assert result.node_count == 3
    assert result.vector_count == 3
    assert result.inserted == 3
    assert result.reused == 0
    assert len(algorithm.embed_calls) == 1
    assert algorithm.embed_calls[0].task == "skill_manage.build"
    assert algorithm.embed_calls[0].embedding_profile == PROFILE

    head = await _head(storage)
    assert head["embedding_profile"] == PROFILE.model_dump(mode="json")
    assert head["active_generation_id"] == result.generation_id

    plan = algorithm.plans[-1]
    expected_hashes = {text.graph_node_id: text.content_hash for text in plan.embedding_texts}
    embeddings = await storage.load_embeddings(SCOPE, result.generation_id)
    assert len(embeddings) == 3
    assert {embedding.graph_node_id: embedding.content_hash for embedding in embeddings} == expected_hashes
    for embedding in embeddings:
        assert embedding.embedding_model == PROFILE.model
        assert embedding.embedding_profile_version == PROFILE.template_version
        assert len(embedding.vector) == PROFILE.dimension
        node = next(node for node in plan.nodes if node.graph_node_id == embedding.graph_node_id)
        assert embedding.skill_id == node.skill_id
        assert embedding.version_id == node.version_id
        assert embedding.vector == VECTORS[node.description]


@pytest.mark.asyncio
async def test_rebuild_reuses_vectors_by_content_hash() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)

    first = await service.build_skill_index(_build("k1"))
    assert len(algorithm.embed_calls) == 1

    second = await service.build_skill_index(_build("k2"))
    assert second.status is OperationStatus.COMMITTED
    assert second.result_revision == 1
    assert second.generation_id != first.generation_id
    assert second.vector_count == 3
    assert second.reused == 3
    assert second.inserted == 0
    assert len(algorithm.embed_calls) == 1

    third = await service.build_skill_index(_build("k3", REFS_ALPHA_V2))
    assert third.status is OperationStatus.COMMITTED
    assert third.result_revision == 2
    assert third.vector_count == 3
    assert third.inserted == 1
    assert third.reused == 2
    assert len(algorithm.embed_calls) == 2
    assert algorithm.embed_calls[-1].texts == ("alpha v2",)

    by_node = {e.graph_node_id: e for e in await storage.load_embeddings(SCOPE, third.generation_id)}
    assert by_node["skill:alpha"].vector == VECTORS["alpha v2"]
    assert by_node["skill:alpha"].version_id == "v2"
    assert by_node["skill:alpha"].content_hash == _content_hash("alpha v2")
    assert by_node["skill:beta"].vector == VECTORS["beta"]
    assert by_node["skill:gamma"].vector == VECTORS["gamma"]


@pytest.mark.asyncio
async def test_retrieve_seeds_algorithm_in_knn_order_with_similarities() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    await service.build_skill_index(_build("k1"))

    result = await service.retrieve_skills(_retrieve())

    assert len(algorithm.embed_calls) == 2
    assert algorithm.embed_calls[-1].task == "skill_manage.query"
    assert algorithm.embed_calls[-1].texts == (QUERY,)
    received = algorithm.retrieve_calls[-1]
    assert received.query == QUERY
    assert received.vector_seed == ("beta", "alpha", "gamma")
    assert set(received.vector_similarity) == {"beta", "alpha", "gamma"}
    assert all(isinstance(score, float) for score in received.vector_similarity.values())
    query_norm = math.sqrt(sum(value * value for value in VECTORS[QUERY]))
    assert received.vector_similarity["beta"] == pytest.approx(0.8 / query_norm)
    assert received.vector_similarity["alpha"] == pytest.approx(0.5 / query_norm)
    assert received.vector_similarity["gamma"] == pytest.approx(0.2 / query_norm)

    head = await _head(storage)
    assert result.revision == 0 == head["active_revision"]
    assert result.generation_id == head["active_generation_id"]
    assert [hit["name"] for hit in result.hits] == ["alpha", "beta", "gamma"]
    assert result.hits[1]["vector_score"] == pytest.approx(0.8 / query_norm)

    upserted = await service.upsert_skill_index(_upsert("u1", REFS_ALPHA_V2[0]))
    assert upserted.status is OperationStatus.COMMITTED
    assert upserted.result_revision == 1

    after = await service.retrieve_skills(_retrieve())
    head = await _head(storage)
    assert after.revision == 1 == head["active_revision"]
    assert after.generation_id == head["active_generation_id"] == upserted.generation_id
    # alpha@v2 lives on an orthogonal axis, so it drops out of the similarity order.
    assert algorithm.retrieve_calls[-1].vector_seed[:2] == ("beta", "gamma")


@pytest.mark.asyncio
async def test_retrieve_seed_top_k_and_score_threshold_bound_the_seeds() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    await service.build_skill_index(_build("k1"))

    await service.retrieve_skills(_retrieve(seed_top_k=2))
    assert algorithm.retrieve_calls[-1].vector_seed == ("beta", "alpha")
    assert set(algorithm.retrieve_calls[-1].vector_similarity) == {"beta", "alpha"}

    calls_before = len(algorithm.retrieve_calls)
    result = await service.retrieve_skills(_retrieve(score_threshold=0.99))
    assert len(algorithm.retrieve_calls) == calls_before + 1
    received = algorithm.retrieve_calls[-1]
    assert received.vector_seed == ()
    assert received.vector_similarity == {}
    assert received.score_threshold == 0.99
    assert result.revision == 0
    assert len(result.hits) == 3
    assert all(hit["vector_score"] is None for hit in result.hits)


@pytest.mark.asyncio
async def test_retrieve_legacy_generation_without_profile_skips_vector_stage() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    await _publish_legacy(storage, "legacy:1", [_node("alpha", "v1", "alpha"), _node("beta", "v1", "beta")])

    result = await service.retrieve_skills(_retrieve())

    assert algorithm.embed_calls == []
    assert len(algorithm.retrieve_calls) == 1
    assert algorithm.retrieve_calls[0].vector_seed == ()
    assert algorithm.retrieve_calls[0].vector_similarity == {}
    assert result.generation_id == "legacy:1"
    assert result.revision == 0
    assert [hit["name"] for hit in result.hits] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_retrieve_rejects_head_published_under_another_profile() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    await _service(storage, RecordingAlgorithm()).build_skill_index(_build("k1"))

    other_algorithm = RecordingAlgorithm()
    other_service = _service(storage, other_algorithm, profile=OTHER_PROFILE)
    with pytest.raises(SkillManageProfileMismatchError) as exc_info:
        await other_service.retrieve_skills(_retrieve())

    assert exc_info.value.error_code == "SKILL_MANAGE_EMBEDDING_PROFILE_MISMATCH"
    assert exc_info.value.expected == OTHER_PROFILE.model_dump(mode="json")
    assert exc_info.value.actual == PROFILE.model_dump(mode="json")
    assert other_algorithm.embed_calls == []
    assert other_algorithm.retrieve_calls == []


@pytest.mark.asyncio
async def test_build_rejects_request_profile_different_from_configured() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)

    with pytest.raises(SkillManageProfileMismatchError) as exc_info:
        await service.build_skill_index(_build("k1", embedding_profile=OTHER_PROFILE))

    assert exc_info.value.error_code == "SKILL_MANAGE_EMBEDDING_PROFILE_MISMATCH"
    assert algorithm.plan_build_calls == []
    assert await storage.get_graph_head(SCOPE) is None
    assert await storage.get_operation(SCOPE, _operation_id("k1")) is None

    with pytest.raises(SkillManageProfileMismatchError):
        await service.upsert_skill_index(
            SkillManageUpsertRequest(
                project_id="p1",
                graph_id="g1",
                idempotency_key="u1",
                version_ref=REFS[0],
                embedding_profile=OTHER_PROFILE,
            )
        )


@pytest.mark.asyncio
async def test_unconfigured_profile_without_request_profile_is_a_configuration_error() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm, profile=None)

    with pytest.raises(SkillConfigurationError):
        await service.build_skill_index(_build("k1"))
    assert algorithm.plan_build_calls == []
    assert await storage.get_graph_head(SCOPE) is None

    # A request-supplied profile is enough when nothing is configured, and the
    # head then carries it for reads and mutations that supply none.
    result = await service.build_skill_index(_build("k2", embedding_profile=PROFILE))
    assert result.status is OperationStatus.COMMITTED
    assert (await _head(storage))["embedding_profile"] == PROFILE.model_dump(mode="json")

    retrieved = await service.retrieve_skills(_retrieve())
    assert retrieved.revision == 0
    assert algorithm.embed_calls[-1].embedding_profile == PROFILE

    removed = await service.remove_skill_from_index(_remove("r1", "alpha"))
    assert removed.status is OperationStatus.COMMITTED
    assert removed.result_revision == 1
    assert removed.reused == 2


@pytest.mark.asyncio
async def test_upsert_same_key_replays_and_different_request_conflicts() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    await service.build_skill_index(_build("k1"))
    embed_calls_after_build = len(algorithm.embed_calls)

    first = await service.upsert_skill_index(_upsert("u1", REFS_ALPHA_V2[0]))
    assert first.status is OperationStatus.COMMITTED
    assert first.operation_id == _operation_id("u1")
    assert first.result_revision == 1
    assert first.base_revision == 0
    assert first.generation_id == generation_id_from_operation(first.operation_id, 1)
    assert first.vector_count == 3
    assert first.inserted == 1
    assert first.reused == 2
    assert len(algorithm.plan_upsert_calls) == 1
    assert len(algorithm.embed_calls) == embed_calls_after_build + 1

    row = await storage.get_operation(SCOPE, first.operation_id)
    assert row is not None
    assert row["result_revision"] == 1
    assert row["current_generation_id"] == first.generation_id
    assert row["base_revision"] == 0

    replay = await service.upsert_skill_index(_upsert("u1", REFS_ALPHA_V2[0]))
    assert replay == first
    assert len(algorithm.plan_upsert_calls) == 1
    assert len(algorithm.embed_calls) == embed_calls_after_build + 1
    assert (await _head(storage))["active_revision"] == 1

    conflict = await service.upsert_skill_index(_upsert("u1", REFS[0]))
    assert conflict.status is OperationStatus.CONFLICT
    assert conflict.error_code == "SKILL_MANAGE_IDEMPOTENCY_CONFLICT"
    assert conflict.operation_id == first.operation_id
    assert len(algorithm.plan_upsert_calls) == 1
    assert (await _head(storage))["active_revision"] == 1


@pytest.mark.asyncio
async def test_remove_same_key_replays_and_different_selector_conflicts() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    await service.build_skill_index(_build("k1"))
    embed_calls_after_build = len(algorithm.embed_calls)

    first = await service.remove_skill_from_index(_remove("r1", "gamma"))
    assert first.status is OperationStatus.COMMITTED
    assert first.operation_id == _operation_id("r1")
    assert first.result_revision == 1
    assert first.generation_id == generation_id_from_operation(first.operation_id, 1)
    assert first.node_count == 2
    assert first.vector_count == 2
    assert first.inserted == 0
    assert first.reused == 2
    assert len(algorithm.plan_remove_calls) == 1
    assert len(algorithm.embed_calls) == embed_calls_after_build

    row = await storage.get_operation(SCOPE, first.operation_id)
    assert row is not None
    assert row["result_revision"] == 1
    assert row["current_generation_id"] == first.generation_id

    replay = await service.remove_skill_from_index(_remove("r1", "gamma"))
    assert replay.status is OperationStatus.COMMITTED
    assert replay.operation_id == first.operation_id
    assert replay.result_revision == first.result_revision
    assert replay.generation_id == first.generation_id
    assert len(algorithm.plan_remove_calls) == 1
    assert (await _head(storage))["active_revision"] == 1

    conflict = await service.remove_skill_from_index(_remove("r1", "beta"))
    assert conflict.status is OperationStatus.CONFLICT
    assert conflict.error_code == "SKILL_MANAGE_IDEMPOTENCY_CONFLICT"
    assert len(algorithm.plan_remove_calls) == 1
    head = await _head(storage)
    assert head["active_revision"] == 1
    snapshot = await storage.load_snapshot(SCOPE, head["active_generation_id"])
    assert {node.skill_id for node in snapshot.nodes} == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_same_key_starts_a_reserved_but_unstarted_operation() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    await service.build_skill_index(_build("k1"))

    # A row reserved by a writer that died before beginning its attempt is
    # started by the next same-key caller instead of being reported as pending.
    request = _remove("r-inflight", "alpha")
    operation_id = _operation_id("r-inflight")
    reserved = await storage.reserve_operation(
        SCOPE,
        operation_id,
        idempotency_key="r-inflight",
        request_hash=compute_skill_manage_request_hash(request),
        operation_type="remove",
    )
    assert reserved is True

    result = await service.remove_skill_from_index(request)

    assert result.status is OperationStatus.COMMITTED
    assert result.operation_id == operation_id
    assert result.result_revision == 1
    assert result.generation_id == generation_id_from_operation(operation_id, 1)
    assert len(algorithm.plan_remove_calls) == 1
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "committed" and row["attempt_no"] == 1
    assert (await _head(storage))["active_revision"] == 1


@pytest.mark.asyncio
async def test_noop_remove_records_result_and_replays_committed() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    build = await service.build_skill_index(_build("k1"))
    embed_calls_after_build = len(algorithm.embed_calls)

    first = await service.remove_skill_from_index(_remove("r-noop", "zeta"))
    assert first.status is OperationStatus.COMMITTED
    assert first.operation_id == _operation_id("r-noop")
    assert first.result_revision == 0
    assert first.base_revision == 0
    assert first.generation_id == build.generation_id
    assert first.node_count == 3
    assert len(algorithm.plan_remove_calls) == 1
    assert len(algorithm.embed_calls) == embed_calls_after_build

    row = await storage.get_operation(SCOPE, first.operation_id)
    assert row is not None
    assert row["result_revision"] == 0
    assert row["current_generation_id"] == build.generation_id

    replay = await service.remove_skill_from_index(_remove("r-noop", "zeta"))
    assert replay.status is OperationStatus.COMMITTED
    assert replay.operation_id == first.operation_id
    assert replay.result_revision == first.result_revision
    assert replay.generation_id == first.generation_id
    assert replay.base_revision == first.base_revision
    assert len(algorithm.plan_remove_calls) == 1

    head = await _head(storage)
    assert head["active_revision"] == 0
    assert head["active_generation_id"] == build.generation_id


@pytest.mark.asyncio
async def test_dry_run_build_plans_without_embedding_or_persisting() -> None:
    storage = RecordingStorage()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)

    result = await service.build_skill_index(_build(None, commit_policy="dry_run"))

    assert result.status is OperationStatus.COMMITTED
    assert result.operation_id is None
    assert result.node_count == 3
    assert result.vector_count == 0
    assert len(algorithm.plan_build_calls) == 1
    assert algorithm.plan_build_calls[0].embedding_profile == PROFILE
    assert algorithm.embed_calls == []
    assert storage.write_embeddings_calls == []
    assert await storage.get_graph_head(SCOPE) is None
