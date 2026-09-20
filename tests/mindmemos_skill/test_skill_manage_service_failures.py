"""Failure bookkeeping, replay of failed operations and degraded-read signalling.

A write that fails after its operation row is reserved must leave a typed
outcome behind: the row goes to ``failed`` (or ``conflict`` for a lost head
CAS) with the error code, its staged generation is discarded, and a same-key
retry replays a non-retryable outcome or opens a new attempt for a retryable
one instead of reporting an in-flight state forever.
"""

from __future__ import annotations

import pytest
from mindmemos_skill.algos.skill_manage.errors import embedding_provider_error
from mindmemos_skill.application.skill_manage.models import SkillManageBuildRequest
from mindmemos_skill.application.skill_manage.state_machine import OperationStatus
from mindmemos_skill.infra.skill_manage import generation_id_from_operation
from mindmemos_skill.typing import EmbeddingProfile, SkillGraphEmbedding, SkillManageEmbedInput, SkillManageScope
from pydantic import ValidationError
from test_skill_manage_service_embedding import (
    REFS,
    SCOPE,
    RecordingAlgorithm,
    _build,
    _head,
    _operation_id,
    _publish_legacy,
    _remove,
    _retrieve,
    _service,
)
from test_skill_manage_storage_contract import InMemorySkillGraphStorageAdapter


class FailingEmbedAlgorithm(RecordingAlgorithm):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    async def embed(self, request: SkillManageEmbedInput):
        if self.failures > 0:
            self.failures -= 1
            raise embedding_provider_error(model=request.embedding_profile.model, task=request.task, reason="Boom")
        return await super().embed(request)


class ConflictingStorage(InMemorySkillGraphStorageAdapter):
    """Loses the head CAS once, as a concurrent writer would make it."""

    def __init__(self) -> None:
        super().__init__()
        self.conflicts = 1

    async def publish_generation(self, scope, generation_id, *, expected_revision, new_revision, **kwargs):
        if self.conflicts > 0 and expected_revision is not None:
            self.conflicts -= 1
            return False
        return await super().publish_generation(
            scope, generation_id, expected_revision=expected_revision, new_revision=new_revision, **kwargs
        )


class BrokenVectorStorage(InMemorySkillGraphStorageAdapter):
    async def write_embeddings(self, scope, generation_id, *, embeddings: list[SkillGraphEmbedding]) -> None:
        raise RuntimeError("vector store unavailable")


class OnceFailingVectorStorage(InMemorySkillGraphStorageAdapter):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error: Exception | None = error
        self.publish_calls = 0

    async def write_embeddings(self, scope, generation_id, *, embeddings):
        error, self.error = self.error, None
        if error is not None:
            raise error
        return await super().write_embeddings(scope, generation_id, embeddings=embeddings)

    async def publish_generation(self, *args, **kwargs):
        self.publish_calls += 1
        return await super().publish_generation(*args, **kwargs)


def _storage_error(kind: str) -> Exception:
    if kind == "connection":
        return ConnectionError("connection reset")
    if kind == "timeout":
        return TimeoutError("storage request timed out")
    if kind == "unknown":
        return RuntimeError("unclassified storage failure")
    if kind == "pool_timeout":
        return pytest.importorskip("psycopg_pool").PoolTimeout("pool checkout timed out")
    return getattr(pytest.importorskip("psycopg.errors"), kind)("storage failure")


def _generation_ids(storage: InMemorySkillGraphStorageAdapter) -> set[str]:
    return {generation_id for (_, _, generation_id) in storage._generations}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "connection",
        "timeout",
        "pool_timeout",
        "ConnectionTimeout",
        "ConnectionFailure",
        "ConnectionDoesNotExist",
        "SerializationFailure",
        "DeadlockDetected",
        "AdminShutdown",
        "CrashShutdown",
        "CannotConnectNow",
    ],
)
async def test_transient_storage_failure_allows_same_key_retry_and_only_one_publish(kind: str) -> None:
    error = _storage_error(kind)
    storage = OnceFailingVectorStorage(error)
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)

    with pytest.raises(type(error)) as exc_info:
        await service.build_skill_index(_build("k1"))
    assert exc_info.value is error
    operation_id = _operation_id("k1")
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "failed" and row["attempt_no"] == 1
    assert row["error_code"] == "SKILL_MANAGE_STORAGE_UNAVAILABLE"
    assert row["error_retryable"] is True
    assert row["error_message"] == type(error).__name__
    assert _generation_ids(storage) == set()
    assert await storage.get_graph_head(SCOPE) is None
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None
    assert storage.publish_calls == 0

    # Even a new service instance must recover through the persisted retry policy.
    service = _service(storage, algorithm)
    result = await service.build_skill_index(_build("k1"))
    assert result.status is OperationStatus.COMMITTED
    assert result.operation_id == operation_id and result.result_revision == 0
    assert result.generation_id == generation_id_from_operation(operation_id, 2)
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "committed" and row["attempt_no"] == 2
    assert row["error_code"] is None and row["pending_repair_reason"] is None
    assert _generation_ids(storage) == {result.generation_id}
    assert len(await storage.load_embeddings(SCOPE, result.generation_id)) == 3
    assert await service.build_skill_index(_build("k1")) == result
    assert storage.publish_calls == 1 and len(algorithm.plan_build_calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["unknown", "OperationalError", "InvalidPassword", "InsufficientPrivilege", "SyntaxError"]
)
async def test_unclassified_and_permanent_storage_failures_do_not_retry(kind: str) -> None:
    error = _storage_error(kind)
    storage = OnceFailingVectorStorage(error)
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    with pytest.raises(type(error)):
        await service.build_skill_index(_build("k1"))

    result = await service.build_skill_index(_build("k1"))
    assert result.status is OperationStatus.FAILED and result.error_retryable is None
    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None and row["attempt_no"] == 1
    assert storage.publish_calls == 0 and len(algorithm.plan_build_calls) == 1


@pytest.mark.asyncio
async def test_algorithm_timeout_is_not_classified_as_a_storage_failure() -> None:
    class TimedOutAlgorithm(RecordingAlgorithm):
        async def plan_build(self, request):
            raise TimeoutError("algorithm deadline")

    storage = InMemorySkillGraphStorageAdapter()
    service = _service(storage, TimedOutAlgorithm())
    with pytest.raises(TimeoutError):
        await service.build_skill_index(_build("k1"))
    result = await service.build_skill_index(_build("k1"))
    assert result.status is OperationStatus.FAILED
    assert result.error_code == "SKILL_MANAGE_OPERATION_FAILED" and result.error_retryable is None


@pytest.mark.asyncio
async def test_explicit_non_retryable_flag_overrides_storage_exception_type() -> None:
    class PermanentConnectionError(ConnectionError):
        error_code = "CUSTOM_PERMANENT_ERROR"
        retryable = False

    storage = OnceFailingVectorStorage(PermanentConnectionError("permanent"))
    service = _service(storage, RecordingAlgorithm())
    with pytest.raises(PermanentConnectionError):
        await service.build_skill_index(_build("k1"))
    result = await service.build_skill_index(_build("k1"))
    assert result.status is OperationStatus.FAILED
    assert result.error_code == "CUSTOM_PERMANENT_ERROR" and result.error_retryable is False


@pytest.mark.asyncio
async def test_retryable_embedding_failure_is_recorded_and_retried_as_a_new_attempt() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = FailingEmbedAlgorithm(failures=2)
    service = _service(storage, algorithm)

    with pytest.raises(Exception) as exc_info:
        await service.build_skill_index(_build("k1"))
    assert exc_info.value.error_code == "SKILL_MANAGE_EMBEDDING_PROVIDER_ERROR"

    operation_id = _operation_id("k1")
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "SKILL_MANAGE_EMBEDDING_PROVIDER_ERROR"
    assert row["error_retryable"] is True
    assert row["error_message"] == "SkillManageError"
    assert row["attempt_no"] == 1
    assert await storage.get_graph_head(SCOPE) is None
    assert _generation_ids(storage) == set()
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None

    # A retryable failure is retried under the same operation as a new attempt
    # (failed -> pending_repair -> running); the provider fails once more.
    with pytest.raises(Exception):
        await service.build_skill_index(_build("k1"))
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "failed" and row["attempt_no"] == 2
    assert len(algorithm.plan_build_calls) == 2

    # The third same-key call opens attempt 3 and commits once the provider is back.
    result = await service.build_skill_index(_build("k1"))
    assert result.status is OperationStatus.COMMITTED
    assert result.operation_id == operation_id
    assert result.result_revision == 0
    assert result.generation_id == generation_id_from_operation(operation_id, 3)
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "committed" and row["attempt_no"] == 3
    assert row["error_code"] is None and row["pending_repair_reason"] is None
    assert _generation_ids(storage) == {result.generation_id}
    assert len(algorithm.plan_build_calls) == 3

    # Once committed, the same key replays without touching the algorithm.
    replay = await service.build_skill_index(_build("k1"))
    assert replay == result
    assert len(algorithm.plan_build_calls) == 3


@pytest.mark.asyncio
async def test_failure_after_staging_discards_the_generation() -> None:
    storage = BrokenVectorStorage()
    service = _service(storage, RecordingAlgorithm())

    with pytest.raises(RuntimeError):
        await service.build_skill_index(_build("k1"))

    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "SKILL_MANAGE_OPERATION_FAILED"
    assert row["error_message"] == "RuntimeError"
    assert row["error_retryable"] is None
    assert _generation_ids(storage) == set()
    assert await storage.get_graph_head(SCOPE) is None

    # An untyped failure is not retryable: the same key replays it.
    replay = await service.build_skill_index(_build("k1"))
    assert replay.status is OperationStatus.FAILED
    assert replay.error_code == "SKILL_MANAGE_OPERATION_FAILED"


@pytest.mark.asyncio
async def test_lost_head_cas_is_conflict_and_the_loser_is_discarded() -> None:
    storage = ConflictingStorage()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    built = await service.build_skill_index(_build("k1"))

    lost = await service.remove_skill_from_index(_remove("r1", "alpha"))
    assert lost.status is OperationStatus.CONFLICT
    assert lost.error_code == "SKILL_MANAGE_REVISION_CONFLICT"
    assert lost.error_retryable is False

    row = await storage.get_operation(SCOPE, _operation_id("r1"))
    assert row is not None
    assert row["status"] == "conflict"
    assert row["error_code"] == "SKILL_MANAGE_REVISION_CONFLICT"
    assert row["error_retryable"] is False
    assert _generation_ids(storage) == {built.generation_id}
    assert (await _head(storage))["active_revision"] == 0
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None

    # conflict is terminal: the same key replays it and never re-runs the plan.
    replay = await service.remove_skill_from_index(_remove("r1", "alpha"))
    assert replay.status is OperationStatus.CONFLICT
    assert replay.error_code == "SKILL_MANAGE_REVISION_CONFLICT"
    assert len(algorithm.plan_remove_calls) == 1

    retried = await service.remove_skill_from_index(_remove("r2", "alpha"))
    assert retried.status is OperationStatus.COMMITTED
    assert retried.result_revision == 1


@pytest.mark.asyncio
async def test_embed_calls_carry_a_run_id_per_operation_and_query() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)

    built = await service.build_skill_index(_build("k1"))
    assert algorithm.embed_calls[-1].run_id == built.operation_id

    await service.retrieve_skills(_retrieve())
    await service.retrieve_skills(_retrieve())
    query_runs = [call.run_id for call in algorithm.embed_calls if call.task == "skill_manage.query"]
    assert len(query_runs) == 2
    assert all(query_runs)
    assert query_runs[0] != query_runs[1]


@pytest.mark.asyncio
async def test_legacy_generation_without_vectors_warns_on_retrieve() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm)
    await _publish_legacy(storage, "legacy:1", [])

    result = await service.retrieve_skills(_retrieve())

    assert algorithm.embed_calls == []
    assert [warning["code"] for warning in result.warnings] == ["SKILL_MANAGE_VECTOR_SEED_SKIPPED"]
    assert result.warnings[0]["details"] == {"generation_id": "legacy:1"}


@pytest.mark.asyncio
async def test_normalization_change_re_embeds_instead_of_reusing() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    await _service(storage, algorithm).build_skill_index(_build("k1"))

    # Reuse is gated on the whole stored profile, not just model/template, so a
    # normalization-only change still re-embeds every node. (The real algorithm
    # additionally rejects a normalization it does not implement.)
    other = EmbeddingProfile(model="stub", dimension=4, normalization="nfkc")
    result = await _service(storage, algorithm, profile=other).build_skill_index(_build("k2"))

    assert result.status is OperationStatus.COMMITTED
    assert result.reused == 0
    assert result.inserted == 3
    assert (await _head(storage))["embedding_profile"] == other.model_dump(mode="json")


@pytest.mark.asyncio
async def test_model_change_re_embeds_everything_instead_of_reusing() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    await _service(storage, algorithm).build_skill_index(_build("k1"))

    other = EmbeddingProfile(model="other", dimension=4)
    result = await _service(storage, algorithm, profile=other).build_skill_index(_build("k2"))

    assert result.status is OperationStatus.COMMITTED
    assert result.reused == 0
    assert result.inserted == 3
    assert (await _head(storage))["embedding_profile"] == other.model_dump(mode="json")
    assert {e.embedding_model for e in await storage.load_embeddings(SCOPE, result.generation_id)} == {"other"}


def test_build_request_rejects_two_versions_of_one_skill() -> None:
    with pytest.raises(ValidationError, match="more than once"):
        SkillManageBuildRequest(
            project_id="p1",
            graph_id="g1",
            version_refs=(REFS[0], REFS[0].model_copy(update={"version_id": "v2"})),
        )


@pytest.mark.asyncio
async def test_discard_generation_never_touches_the_active_one() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    service = _service(storage, RecordingAlgorithm())
    built = await service.build_skill_index(_build("k1"))

    await storage.discard_generation(SkillManageScope(project_id="p1", graph_id="g1"), built.generation_id)

    assert _generation_ids(storage) == {built.generation_id}
    assert len(await storage.load_embeddings(SCOPE, built.generation_id)) == 3
