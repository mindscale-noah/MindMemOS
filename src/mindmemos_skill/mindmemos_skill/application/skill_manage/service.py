"""Stateful ``skill_manage`` application orchestration.

The service owns scope, operation identity, the writer lease, generation
staging/validation/publish and deferred GC; the algorithm owns parsing,
linking, embedding text and model calls; the storage adapter owns durable
state. One write walks::

    reserve -> acquire lease -> begin attempt (pending -> running)
      -> plan -> embed -> stage -> write snapshot/vectors
      -> validate -> generation ready -> operation validating
      -> publish (fenced head CAS, new active, old retired)
      -> operation committed -> release lease -> deferred GC

Every instant is taken from the injected clock and handed to the adapter as
an ISO string, so lease expiry and retention are testable without sleeping.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import NAMESPACE_URL, uuid5
from uuid import uuid4 as _uuid4

from ...errors import (
    SkillConfigurationError,
    SkillManageConsistencyError,
    SkillManageGcPartialFailureError,
    SkillManageOperationCancelledError,
    SkillManageOperationPendingError,
    SkillManageOperationPendingRepairError,
    SkillManageProfileMismatchError,
    SkillManageRevisionConflictError,
    SkillManageValidationError,
    SkillManageWriterConflictError,
    SkillNotFoundError,
)
from ...infra.skill_manage import (
    SkillGraphStorageAdapter,
    generation_counts_hash,
    generation_id_from_operation,
)
from ...infra.skill_manage.errors import is_transient_storage_error
from ...service.protocols import SkillManageAlgorithm
from ...typing.skill_manage import (
    EmbeddingProfile,
    SkillDocumentInput,
    SkillGraphEmbedding,
    SkillGraphSnapshot,
    SkillManageBuildInput,
    SkillManageBuildPlan,
    SkillManageEmbedInput,
    SkillManageRemoveInput,
    SkillManageRetrieveInput,
    SkillManageScope,
    SkillManageUpsertInput,
    SkillManageWarning,
    SkillVersionRef,
)
from .inflight import InflightConflictError, InflightLeaderCancelled, InflightRegistry
from .loader import SkillVersionNotFound
from .models import (
    SkillManageBuildRequest,
    SkillManageMaintenanceReport,
    SkillManageOperationResult,
    SkillManageRemoveRequest,
    SkillManageRepairAction,
    SkillManageRepairReport,
    SkillManageRetrieveRequest,
    SkillManageRetrieveResult,
    SkillManageServiceSettings,
    SkillManageUpsertRequest,
    compute_skill_manage_request_hash,
)
from .state_machine import (
    TERMINAL_OPERATION_STATUSES,
    GenerationStatus,
    OperationStatus,
    PendingRepairReason,
)

_BUILD_EMBED_TASK = "skill_manage.build"
_QUERY_EMBED_TASK = "skill_manage.query"
_VECTOR_SEED_SKIPPED = "SKILL_MANAGE_VECTOR_SEED_SKIPPED"
_GC_DEFERRED = "SKILL_MANAGE_GC_DEFERRED"
_REPAIR_DEFERRED = "SKILL_MANAGE_REPAIR_DEFERRED"
_REVISION_CONFLICT = "SKILL_MANAGE_REVISION_CONFLICT"
_WRITER_CONFLICT = "SKILL_MANAGE_WRITER_CONFLICT"
_IDEMPOTENCY_CONFLICT = "SKILL_MANAGE_IDEMPOTENCY_CONFLICT"
_OPERATION_FAILED = "SKILL_MANAGE_OPERATION_FAILED"
_OPERATION_CANCELLED = "SKILL_MANAGE_OPERATION_CANCELLED"
_STORAGE_UNAVAILABLE = "SKILL_MANAGE_STORAGE_UNAVAILABLE"
_RECONCILABLE_REPAIR_REASONS = frozenset(
    {
        PendingRepairReason.CRASH_RECOVERY.value,
        PendingRepairReason.PUBLISH_OUTCOME_UNKNOWN.value,
        PendingRepairReason.FINALIZE_UNAVAILABLE.value,
    }
)
_RESULT_COUNT_FIELDS = ("node_count", "edge_count", "scene_count", "vector_count", "inserted", "reused")


@runtime_checkable
class SkillManageStoragePort(Protocol):
    """Minimal lifecycle surface the application owns for the storage adapter."""

    async def is_ready(self) -> bool: ...

    async def start(self) -> None: ...

    async def close(self) -> None: ...


PlanAndHead = Callable[[], Awaitable[tuple[SkillManageBuildPlan, dict[str, Any] | None, SkillGraphSnapshot | None]]]
_StorageResult = TypeVar("_StorageResult")


class _TransientStorageFailure(Exception):
    def __init__(self, original: Exception) -> None:
        self.original = original


async def _storage_call(call: Awaitable[_StorageResult]) -> _StorageResult:
    """Carry storage provenance to failure bookkeeping without changing caller errors."""

    try:
        return await call
    except Exception as exc:
        # Explicit domain retry policy always wins over driver classification.
        if getattr(exc, "retryable", None) is None and is_transient_storage_error(exc):
            raise _TransientStorageFailure(exc) from exc
        raise


@dataclass
class _WriterLease:
    """One held writer lease, renewed in the background until stopped.

    ``lost`` flips when a renewal is rejected (the lease expired and another
    writer took it over); the publish fencing check would reject the stale
    token anyway, but the flag lets the write bail out before staging more.
    """

    storage: SkillGraphStorageAdapter
    scope: SkillManageScope
    owner_operation_id: str
    token: int
    ttl_seconds: float
    renew_interval_seconds: float
    now: Callable[[], str]
    sleep: Callable[[float], Awaitable[None]]
    lost: bool = False
    _task: asyncio.Task[None] | None = field(default=None, repr=False)

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._renew_forever())

    async def _renew_forever(self) -> None:
        # A renewal that errors (storage blip) is retried next interval; only
        # a rejected CAS is definitive. If the blip outlasts the TTL, publish's
        # fencing check is what rejects the expired lease.
        # The pool's checkout can consume a cancellation that lands inside its
        # liveness probe; the loop condition sees the request anyway, also on
        # the ``continue`` path, so ``stop()`` never waits for a renew to land.
        while not self.lost and not asyncio.current_task().cancelling():
            await self.sleep(self.renew_interval_seconds)
            try:
                await self.renew()
            except asyncio.CancelledError:
                raise
            except Exception:
                continue

    async def renew(self) -> bool:
        if self.lost:
            return False
        renewed = await self.storage.renew_writer_lease(
            self.scope,
            owner_operation_id=self.owner_operation_id,
            token=self.token,
            ttl_seconds=self.ttl_seconds,
            now=self.now(),
        )
        if not renewed:
            self.lost = True
        return renewed

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def release(self) -> None:
        await self.stop()
        if self.lost:
            return
        try:
            await self.storage.release_writer_lease(
                self.scope, owner_operation_id=self.owner_operation_id, token=self.token, now=self.now()
            )
        except Exception:
            return


@dataclass(frozen=True)
class _Attempt:
    attempt_no: int
    generation_id: str
    lease: _WriterLease


class SkillManageService:
    """Application-owned orchestration service for the ``skill_manage`` capability."""

    def __init__(
        self,
        *,
        algorithm: SkillManageAlgorithm | None = None,
        algorithm_resolver: Callable[[str | None], SkillManageAlgorithm] | None = None,
        storage: SkillGraphStorageAdapter,
        embedding_profile: EmbeddingProfile | None = None,
        settings: SkillManageServiceSettings | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        id_generator: Callable[[], str] | None = None,
        version_loader: object | None = None,
        inflight: InflightRegistry | None = None,
    ) -> None:
        if algorithm is None and algorithm_resolver is None:
            raise ValueError("SkillManageService requires either algorithm or algorithm_resolver")
        if algorithm is not None and algorithm_resolver is not None:
            raise ValueError("SkillManageService accepts algorithm XOR algorithm_resolver, not both")
        self._algorithm = algorithm
        self._algorithm_resolver = algorithm_resolver
        self._storage = storage
        self._embedding_profile = embedding_profile
        self._settings = settings or SkillManageServiceSettings()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep or asyncio.sleep
        self._id_generator = id_generator or (lambda: _uuid4().hex)
        self._version_loader = version_loader
        self._inflight = inflight or InflightRegistry()
        self._started = False
        self._closed = False

    def _resolve_algorithm(self, algorithm_name: str | None) -> SkillManageAlgorithm:
        if self._algorithm_resolver is not None:
            return self._algorithm_resolver(algorithm_name)
        return self._algorithm  # type: ignore[return-value]

    @property
    def algorithm(self) -> SkillManageAlgorithm:
        if self._algorithm is not None:
            return self._algorithm
        return self._algorithm_resolver(None)  # type: ignore[union-attr]

    @property
    def storage(self) -> SkillGraphStorageAdapter:
        return self._storage

    @property
    def embedding_profile(self) -> EmbeddingProfile | None:
        return self._embedding_profile

    @property
    def settings(self) -> SkillManageServiceSettings:
        return self._settings

    def is_ready(self) -> bool:
        """Report whether the storage backend has been started and not closed.

        ``from_config`` bootstraps and opens the vector store before the
        application is returned, so an unstarted service is never exposed; the
        application gates ``skill_manage`` on this flag.
        """

        return self._started and not self._closed

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("SkillManageService is closed")
        if self._started:
            return
        await self._storage.start()
        self._started = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._storage.close()
        self._started = False

    async def __aenter__(self) -> SkillManageService:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _now(self) -> str:
        instant = self._clock()
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("skill_manage clock must return timezone-aware datetimes")
        return instant.isoformat()

    async def build_skill_index(
        self,
        request: SkillManageBuildRequest,
    ) -> SkillManageOperationResult:
        """Run one full build synchronously to committed (or dry-run with no I/O)."""

        scope = SkillManageScope(project_id=request.project_id, graph_id=request.graph_id)
        algorithm = self._resolve_algorithm(request.algorithm_name)
        profile = self._effective_profile(request.embedding_profile)

        async def build_input(documents: tuple[SkillDocumentInput, ...] | None = None) -> SkillManageBuildInput:
            return SkillManageBuildInput(
                scope=scope,
                version_refs=request.version_refs,
                documents=await self._resolve_documents(request) if documents is None else documents,
                embedding_profile=profile,
                build_scene=request.build_scene,
                llm_linking=request.llm_linking,
            )

        if request.commit_policy == "dry_run":
            conflict = await self._dry_run_revision_conflict(scope, request.expected_revision)
            if conflict is not None:
                return conflict
            plan = await algorithm.plan_build(await build_input())
            return SkillManageOperationResult(
                operation_id=None,
                status=OperationStatus.COMMITTED,
                node_count=len(plan.nodes),
                edge_count=len(plan.edges),
                scene_count=0,
                warnings=tuple(warning.model_dump(mode="json") for warning in plan.warnings),
            )

        # Operation ID is deterministic on the idempotency key so a retry with
        # the same key touches the same durable operation row (AC-P4-002).
        idempotency_key = request.idempotency_key or self._id_generator()
        request_hash = compute_skill_manage_request_hash(request)
        # Preflight (spec §2): the canonical documents are resolved before any
        # operation row exists, so a loader failure (version missing, or a
        # repository blip) surfaces as its own exception and never burns the
        # idempotency key. Planning stays inside the attempt so a replayed
        # operation never touches the algorithm — nor the loader: a settled
        # row is replayed even if its versions have since left the repository.
        # The preflight belongs to the in-flight leader so same-key followers
        # share its outcome instead of consulting the loader on their own.
        operation_id = self._operation_id(scope, idempotency_key)

        async def persist() -> SkillManageOperationResult:
            documents = (
                await self._resolve_documents(request) if await self._may_run_attempt(scope, operation_id) else None
            )
            opened = await self._open_attempt(
                scope,
                operation_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                operation_type="build",
            )
            if isinstance(opened, SkillManageOperationResult):
                return opened

            async def plan_and_head() -> tuple[SkillManageBuildPlan, dict[str, Any] | None, SkillGraphSnapshot | None]:
                plan = await algorithm.plan_build(await build_input(documents))
                return plan, await _storage_call(self._storage.get_graph_head(scope)), None

            return await self._commit_generation(
                scope,
                operation_id,
                opened,
                algorithm=algorithm,
                profile=profile,
                plan_and_head=plan_and_head,
                noop_ok=False,
                expected_revision=request.expected_revision,
            )

        return await self._run_inflight(scope, idempotency_key, request_hash, persist)

    async def retrieve_skills(
        self,
        request: SkillManageRetrieveRequest,
    ) -> SkillManageRetrieveResult:
        """Read the active head once and retrieve against its fixed generation.

        The query is embedded with the generation's own profile, KNN over that
        generation seeds PPR in similarity order, and the algorithm reranks.
        A generation published without vectors (empty stored profile) skips the
        vector stage, degrades to lexical-only ranking and says so in warnings.
        The generation is re-checked after every load: a deletion starts by
        tombstoning the row, so a row still ``active``/``retired`` afterwards
        proves the loads saw a complete snapshot; otherwise the read starts
        over from a fresh head rather than reporting a torn one.
        """

        scope = SkillManageScope(project_id=request.project_id, graph_id=request.graph_id)
        algorithm = self._resolve_algorithm(request.algorithm_name)
        for attempt in range(2):
            head = await self._storage.get_graph_head(scope)
            if head is None:
                raise SkillNotFoundError(f"graph not found: {request.graph_id}")
            generation_id = head.get("active_generation_id")
            if generation_id is None:
                raise SkillNotFoundError(f"graph has no active generation: {request.graph_id}")

            snapshot = await self._storage.load_snapshot(scope, generation_id)
            vector_seed: list[str] = []
            vector_similarity: dict[str, float] = {}
            warnings: list[dict[str, Any]] = []
            stored_profile = _stored_profile(head)
            if stored_profile is None:
                warnings.append(
                    SkillManageWarning(
                        code=_VECTOR_SEED_SKIPPED,
                        message="active generation was published without embeddings; ranking is lexical only",
                        details={"generation_id": generation_id},
                    ).model_dump(mode="json")
                )
            else:
                profile = self._effective_profile(None, stored=stored_profile)
                if stored_profile != profile:
                    raise SkillManageProfileMismatchError(
                        expected=profile.model_dump(mode="json"), actual=stored_profile.model_dump(mode="json")
                    )
                embedded = await algorithm.embed(
                    SkillManageEmbedInput(
                        embedding_profile=profile,
                        texts=(request.query,),
                        task=_QUERY_EMBED_TASK,
                        run_id=self._id_generator(),
                    )
                )
                hits = await self._storage.search_embeddings(
                    scope,
                    generation_id,
                    query_vector=embedded.vectors[0],
                    top_k=request.seed_top_k,
                    score_threshold=request.score_threshold,
                )
                name_by_node_id = {
                    node.graph_node_id: node.name for node in snapshot.nodes if node.node_type == "skill"
                }
                for hit in hits:
                    name = name_by_node_id.get(hit.graph_node_id)
                    if name is None or name in vector_similarity:
                        continue
                    vector_seed.append(name)
                    vector_similarity[name] = hit.score
            if await self._generation_readable(scope, generation_id):
                break
            if attempt:
                raise SkillNotFoundError(f"generation {generation_id} was collected while it was being read")

        output = await algorithm.retrieve(
            SkillManageRetrieveInput(
                scope=scope,
                snapshot=snapshot,
                query=request.query,
                top_k=request.top_k,
                mode=request.mode,
                score_threshold=request.score_threshold,
                vector_seed=tuple(vector_seed),
                vector_similarity=vector_similarity,
            )
        )
        warnings.extend(warning.model_dump(mode="json") for warning in output.warnings)
        return SkillManageRetrieveResult(
            query=request.query,
            generation_id=output.generation_id,
            revision=output.revision,
            hits=tuple(hit.model_dump(mode="json") for hit in output.hits),
            warnings=tuple(warnings),
        )

    async def upsert_skill_index(self, request: SkillManageUpsertRequest) -> SkillManageOperationResult:
        algorithm = self._resolve_algorithm(request.algorithm_name)
        resolved: dict[str, SkillDocumentInput | None] = {}

        async def preflight() -> None:
            resolved["document"] = await self._resolve_document(request.version_ref, request.project_id)

        async def plan_factory(scope, snapshot, profile):
            if "document" not in resolved:
                await preflight()
            return await algorithm.plan_upsert(
                SkillManageUpsertInput(
                    scope=scope,
                    version_ref=request.version_ref,
                    snapshot=snapshot,
                    embedding_profile=profile,
                    document=resolved["document"],
                )
            )

        return await self._mutate_index(
            scope=SkillManageScope(project_id=request.project_id, graph_id=request.graph_id),
            request_hash=compute_skill_manage_request_hash(request),
            idempotency_key=request.idempotency_key or self._id_generator(),
            operation_type="upsert",
            noop_ok=False,
            algorithm=algorithm,
            requested_profile=request.embedding_profile,
            plan_factory=plan_factory,
            expected_revision=request.expected_revision,
            dry_run=request.commit_policy == "dry_run",
            preflight=preflight,
        )

    async def remove_skill_from_index(self, request: SkillManageRemoveRequest) -> SkillManageOperationResult:
        algorithm = self._resolve_algorithm(request.algorithm_name)

        async def plan_factory(scope, snapshot, profile):
            return await algorithm.plan_remove(
                SkillManageRemoveInput(
                    scope=scope,
                    selector=request.selector,
                    snapshot=snapshot,
                )
            )

        return await self._mutate_index(
            scope=SkillManageScope(project_id=request.project_id, graph_id=request.graph_id),
            request_hash=compute_skill_manage_request_hash(request),
            idempotency_key=request.idempotency_key or self._id_generator(),
            operation_type="remove",
            noop_ok=True,
            algorithm=algorithm,
            requested_profile=None,
            plan_factory=plan_factory,
            expected_revision=request.expected_revision,
            dry_run=request.commit_policy == "dry_run",
        )

    async def _mutate_index(
        self,
        *,
        scope: SkillManageScope,
        request_hash: str,
        idempotency_key: str,
        operation_type: str,
        noop_ok: bool,
        algorithm: SkillManageAlgorithm,
        requested_profile: EmbeddingProfile | None,
        plan_factory,
        expected_revision: int | None = None,
        dry_run: bool = False,
        preflight: Callable[[], Awaitable[None]] | None = None,
    ) -> SkillManageOperationResult:
        self._check_requested_profile(requested_profile)
        head = await self._storage.get_graph_head(scope)
        if head is None:
            raise SkillNotFoundError(f"graph not found: {scope.graph_id}")
        if head.get("active_generation_id") is None:
            raise SkillNotFoundError(f"graph has no active generation: {scope.graph_id}")
        profile = self._effective_profile(requested_profile, stored=_stored_profile(head))

        if dry_run:
            if expected_revision is not None and head.get("active_revision") != expected_revision:
                return _dry_run_conflict()
            plan, _, _ = await self._plan_against_active(scope, head, plan_factory, profile)
            return SkillManageOperationResult(
                operation_id=None,
                status=OperationStatus.COMMITTED,
                node_count=len(plan.nodes),
                edge_count=len(plan.edges),
                scene_count=0,
                warnings=tuple(warning.model_dump(mode="json") for warning in plan.warnings),
            )

        # Preflight (spec §2): whatever the mutation needs from the canonical
        # store is resolved before any operation row exists, so a loader
        # failure never burns the idempotency key; a settled row is replayed
        # without consulting the loader at all. It runs inside the in-flight
        # leader so same-key followers share its outcome.
        operation_id = self._operation_id(scope, idempotency_key)

        async def persist() -> SkillManageOperationResult:
            if preflight is not None and await self._may_run_attempt(scope, operation_id):
                await preflight()
            opened = await self._open_attempt(
                scope,
                operation_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                operation_type=operation_type,
            )
            if isinstance(opened, SkillManageOperationResult):
                return opened

            async def plan_and_head() -> tuple[SkillManageBuildPlan, dict[str, Any] | None, SkillGraphSnapshot | None]:
                # The head is re-read here so the plan is computed against the
                # generation the CAS in publish will be checked against; the
                # lease held by the attempt keeps that generation from being
                # collected underneath the read.
                current = await _storage_call(self._storage.get_graph_head(scope))
                generation_id = current.get("active_generation_id") if current is not None else None
                if generation_id is None:
                    raise SkillNotFoundError(f"graph has no active generation: {scope.graph_id}")
                snapshot = await _storage_call(self._storage.load_snapshot(scope, generation_id))
                return await plan_factory(scope, snapshot, profile), current, snapshot

            return await self._commit_generation(
                scope,
                operation_id,
                opened,
                algorithm=algorithm,
                profile=profile,
                plan_and_head=plan_and_head,
                noop_ok=noop_ok,
                expected_revision=expected_revision,
            )

        return await self._run_inflight(scope, idempotency_key, request_hash, persist)

    # ------------------------------------------------------------- attempts

    async def _may_run_attempt(self, scope: SkillManageScope, operation_id: str) -> bool:
        """Whether a same-key call could execute (rather than replay) — i.e. whether preflight is needed.

        A settled row (committed / conflict / cancelled / non-retryable
        failure) only ever replays, and a replay must not depend on the
        canonical store still holding the versions it was built from.
        """

        row = await self._storage.get_operation(scope, operation_id)
        if row is None:
            return True
        status = OperationStatus(row.get("status") or OperationStatus.PENDING.value)
        if status in TERMINAL_OPERATION_STATUSES:
            return False
        return status is not OperationStatus.FAILED or row.get("error_retryable") is True

    async def _generation_readable(self, scope: SkillManageScope, generation_id: str) -> bool:
        """Whether a generation just read is still a published one (``active``/``retired``).

        Deleting a generation starts by tombstoning its row, so this check
        passing *after* the loads proves none of them ran into a deletion.
        """

        generation = await self._storage.get_generation(scope, generation_id)
        return generation is not None and generation.get("status") in (
            GenerationStatus.ACTIVE.value,
            GenerationStatus.RETIRED.value,
        )

    async def _plan_against_active(
        self,
        scope: SkillManageScope,
        head: dict[str, Any],
        plan_factory,
        profile: EmbeddingProfile,
    ) -> tuple[SkillManageBuildPlan, SkillGraphSnapshot, dict[str, Any]]:
        """Plan a mutation against the head's active generation, re-reading once if it was collected meanwhile."""

        retried = False
        while True:
            generation_id = head.get("active_generation_id")
            if generation_id is None:
                raise SkillNotFoundError(f"graph has no active generation: {scope.graph_id}")
            snapshot = await self._storage.load_snapshot(scope, generation_id)
            if await self._generation_readable(scope, generation_id):
                return await plan_factory(scope, snapshot, profile), snapshot, head
            if retried:
                raise SkillNotFoundError(f"generation {generation_id} was collected while it was being read")
            retried = True
            refreshed = await self._storage.get_graph_head(scope)
            if refreshed is None:
                raise SkillNotFoundError(f"graph not found: {scope.graph_id}")
            head = refreshed

    async def _dry_run_revision_conflict(
        self, scope: SkillManageScope, expected_revision: int | None
    ) -> SkillManageOperationResult | None:
        """The conflict a persist would report for ``expected_revision``, or ``None`` if it would pass.

        A dry run must not report ``committed`` for a request whose persist
        would be rejected by the optimistic lock; the check reads the head
        only and writes nothing.
        """

        if expected_revision is None:
            return None
        head = await self._storage.get_graph_head(scope)
        base_revision = head.get("active_revision") if head is not None else None
        return None if base_revision == expected_revision else _dry_run_conflict()

    async def _run_inflight(
        self,
        scope: SkillManageScope,
        idempotency_key: str,
        request_hash: str,
        coro: Callable[[], Awaitable[SkillManageOperationResult]],
    ) -> SkillManageOperationResult:
        """Run a same-key write once and merge concurrent callers in-process.

        The first caller becomes the leader and publishes its outcome; later
        callers with the same key+hash join the leader's future via
        ``asyncio.shield`` so cancelling a follower never cancels the leader.
        A different hash while the leader is in flight is the same idempotency
        conflict the durable row would report. When the leader itself is
        cancelled, only its own task propagates ``CancelledError``: followers
        re-enter and run against the durable row, which either replays the
        recorded ``cancelled`` outcome as the typed error or, if the leader
        died before reserving, performs the write. While the leader runs,
        cross-process callers still rely on the durable operation row and
        writer lease (they observe ``running`` and poll/take-over).
        """

        key = (scope.project_id, scope.graph_id, idempotency_key)
        while True:
            try:
                role, result_future = self._inflight.begin(key, request_hash)
            except InflightConflictError:
                return SkillManageOperationResult(
                    operation_id=self._operation_id(scope, idempotency_key),
                    status=OperationStatus.CONFLICT,
                    error_code=_IDEMPOTENCY_CONFLICT,
                )
            if role == "follower":
                assert result_future is not None
                try:
                    return await self._inflight.follow(result_future)
                except InflightLeaderCancelled:
                    continue
            try:
                result = await coro()
            except BaseException as exc:
                self._inflight.publish(key, None, error=exc)
                self._inflight.release(key)
                raise
            self._inflight.publish(key, result)
            self._inflight.release(key)
            return result

    async def _open_attempt(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        idempotency_key: str,
        request_hash: str,
        operation_type: str,
    ) -> _Attempt | SkillManageOperationResult:
        """Reserve the operation and own an attempt on it, or replay its outcome.

        Same-key handling follows the frozen idempotency table: a different
        request hash is a conflict; committed/conflict/failed(non-retryable)
        replay the recorded result and cancelled raises its typed error;
        failed(retryable) opens a new attempt via ``pending_repair``; an
        unstarted ``pending`` row is started by whoever wins
        ``begin_initial_attempt_atomically``; ``running``/``validating`` rows
        are waited on while their writer lease is alive and, once it expired,
        reconciled through crash recovery under a taken-over lease so the
        stale writer's token is fenced out before its rows are touched.
        """

        reserved = await self._storage.reserve_operation(
            scope,
            operation_id,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            operation_type=operation_type,
            now=self._now(),
        )
        existing = None if reserved else await self._storage.get_operation(scope, operation_id)
        deadline = self._clock() + timedelta(seconds=self._settings.operation_wait_timeout_seconds)
        held: _WriterLease | None = None
        try:
            while True:
                if held is not None and not await held.renew():
                    # The lease taken over for a reconcile is not renewed in the
                    # background; if it lapsed meanwhile another same-key caller
                    # may own the row now, so nothing below may assume exclusivity.
                    return self._writer_conflict(operation_id)
                if existing is None:
                    started = await self._start_attempt(scope, operation_id, attempt_no=1, initial=True, lease=held)
                    held = None
                    if started is not None:
                        return started
                    existing = await self._storage.get_operation(scope, operation_id)
                    if existing is None:
                        raise SkillManageOperationPendingRepairError(operation_id, "operation row vanished")
                    continue
                if existing.get("request_hash") != request_hash:
                    return SkillManageOperationResult(
                        operation_id=operation_id,
                        status=OperationStatus.CONFLICT,
                        error_code=_IDEMPOTENCY_CONFLICT,
                    )
                status = OperationStatus(existing.get("status") or OperationStatus.PENDING.value)
                attempt_no = int(existing.get("attempt_no") or 1)
                if status is OperationStatus.CANCELLED:
                    raise SkillManageOperationCancelledError(operation_id)
                if status in TERMINAL_OPERATION_STATUSES:
                    return self._replay(existing)
                if status is OperationStatus.FAILED:
                    if existing.get("error_retryable") is not True:
                        return self._replay(existing)
                    started = await self._start_attempt(
                        scope, operation_id, attempt_no=attempt_no + 1, initial=False, lease=held
                    )
                    held = None
                    if started is not None:
                        return started
                elif status is OperationStatus.PENDING:
                    started = await self._start_attempt(
                        scope, operation_id, attempt_no=attempt_no, initial=True, lease=held
                    )
                    held = None
                    if started is not None:
                        return started
                elif status is OperationStatus.PENDING_REPAIR:
                    reason = existing.get("pending_repair_reason")
                    # The lease names the operation, not the process: while this
                    # caller already holds it, a "live writer" can only be itself.
                    other_writer_alive = held is None and await self._writer_alive(scope, operation_id)
                    if reason == PendingRepairReason.RETRY_ATTEMPT_RESERVATION.value:
                        # The reserver is between its two CAS steps; only resume
                        # the reservation once its lease has lapsed.
                        if other_writer_alive:
                            raise SkillManageOperationPendingRepairError(operation_id, reason)
                        held = held or await self._take_over_lease(scope, operation_id)
                        if held is None:
                            return self._writer_conflict(operation_id)
                        started = await self._start_attempt(
                            scope,
                            operation_id,
                            attempt_no=attempt_no + 1,
                            initial=False,
                            lease=held,
                            resume_reservation=True,
                        )
                        held = None
                        if started is not None:
                            return started
                    elif reason in _RECONCILABLE_REPAIR_REASONS:
                        if other_writer_alive:
                            # Another recoverer holds the lease mid-reconcile:
                            # wait for it like a live attempt.
                            pass
                        else:
                            # Reconcile only under the lease, against a row re-read
                            # under it: a same-key retry that already opened a
                            # newer attempt needs the lease to do so, and settling
                            # the listed attempt would otherwise terminate it.
                            held = held or await self._take_over_lease(scope, operation_id)
                            if held is None:
                                return self._writer_conflict(operation_id)
                            refreshed = await self._storage.get_operation(scope, operation_id)
                            if refreshed is None:
                                raise SkillManageOperationPendingRepairError(operation_id, "operation row vanished")
                            if _attempt_view(refreshed) != _attempt_view(existing):
                                existing = refreshed
                                continue
                            await self._storage.repair_crash_recovery(
                                scope,
                                operation_id,
                                attempt_no=attempt_no,
                                generation_id=existing.get("current_generation_id"),
                                now=self._now(),
                            )
                            refreshed = await self._storage.get_operation(scope, operation_id)
                            if refreshed is not None and _attempt_view(refreshed) != _attempt_view(existing):
                                existing = refreshed
                                continue
                            raise SkillManageOperationPendingRepairError(operation_id, reason)
                    else:
                        raise SkillManageOperationPendingRepairError(operation_id, reason)
                elif held is not None or not await self._writer_alive(scope, operation_id):
                    # running/validating with no live lease: take the lease over
                    # first so the stale writer is fenced, then reconcile.
                    held = held or await self._take_over_lease(scope, operation_id)
                    if held is None:
                        return self._writer_conflict(operation_id)
                    claimed = await self._storage.claim_crash_recovery(
                        scope,
                        operation_id,
                        expected_attempt_no=attempt_no,
                        expected_generation_id=existing.get("current_generation_id"),
                        now=self._now(),
                    )
                    if claimed:
                        await self._storage.repair_crash_recovery(
                            scope,
                            operation_id,
                            attempt_no=attempt_no,
                            generation_id=existing.get("current_generation_id"),
                            now=self._now(),
                        )
                    existing = await self._storage.get_operation(scope, operation_id)
                    continue
                if self._clock() >= deadline:
                    raise SkillManageOperationPendingError(operation_id, status.value)
                await self._sleep(self._settings.operation_poll_interval_seconds)
                existing = await self._storage.get_operation(scope, operation_id)
        finally:
            if held is not None:
                await held.release()

    async def _take_over_lease(self, scope: SkillManageScope, operation_id: str) -> _WriterLease | None:
        """Acquire the graph lease for ``operation_id``; ``None`` while another writer holds it.

        An exception (or cancellation) escaping after the acquire committed
        leaves the lease owned by this operation until its TTL. It is not
        released here: the lease row carries no per-call nonce, so a same-key
        caller that fails inside its own acquire could not tell that lease
        from one a live writer of the same operation holds, and releasing the
        wrong one fences a healthy writer out at publish.
        """

        try:
            token = await self._storage.acquire_writer_lease(
                scope,
                owner_operation_id=operation_id,
                ttl_seconds=self._settings.writer_lease_ttl_seconds,
                now=self._now(),
            )
        except SkillManageWriterConflictError:
            return None
        return _WriterLease(
            storage=self._storage,
            scope=scope,
            owner_operation_id=operation_id,
            token=token,
            ttl_seconds=self._settings.writer_lease_ttl_seconds,
            renew_interval_seconds=self._settings.writer_lease_renew_interval_seconds,
            now=self._now,
            sleep=self._sleep,
        )

    @staticmethod
    def _writer_conflict(operation_id: str) -> SkillManageOperationResult:
        # The row is left as it was, so the same key can simply retry later.
        return SkillManageOperationResult(
            operation_id=operation_id,
            status=OperationStatus.CONFLICT,
            error_code=_WRITER_CONFLICT,
            error_retryable=True,
        )

    async def _start_attempt(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        attempt_no: int,
        initial: bool,
        lease: _WriterLease | None = None,
        resume_reservation: bool = False,
    ) -> _Attempt | SkillManageOperationResult | None:
        """Hold the writer lease and CAS the operation into a running attempt.

        Returns the attempt, a writer-conflict result while another writer
        holds the lease, or ``None`` when the attempt CAS lost (the caller
        re-reads the row and decides again). The lease is released on every
        path that does not hand it to an attempt. A lease handed in by the
        caller was taken over earlier and has not been renewed since, so it
        is renewed first: if it lapsed meanwhile another writer may own the
        row now, and the attempt must not begin under a stale token.

        ``cancelled`` is only ever recorded once the attempt object exists: a
        cancellation landing inside the reserve/begin calls is treated like a
        crash at that point (the lease is released; the row is left
        ``pending``, ``pending_repair(retry_attempt_reservation)`` or
        ``running`` for the next same-key call to converge and redo), since
        whether the CAS landed cannot be told from the cancelled await.
        """

        if lease is not None and not await lease.renew():
            return self._writer_conflict(operation_id)
        lease = lease or await self._take_over_lease(scope, operation_id)
        if lease is None:
            return self._writer_conflict(operation_id)
        try:
            head = await self._storage.get_graph_head(scope)
            base_revision = head.get("active_revision") if head is not None else None
            generation_id = generation_id_from_operation(operation_id, attempt_no)
            if initial:
                began = await self._storage.begin_initial_attempt_atomically(
                    scope,
                    operation_id,
                    generation_id=generation_id,
                    fencing_token=lease.token,
                    base_revision=base_revision,
                    now=self._now(),
                )
            else:
                reserved = resume_reservation or await self._storage.reserve_retry_attempt(
                    scope, operation_id, now=self._now()
                )
                began = reserved and await self._storage.begin_retry_attempt_atomically(
                    scope,
                    operation_id,
                    attempt_no=attempt_no,
                    generation_id=generation_id,
                    fencing_token=lease.token,
                    base_revision=base_revision,
                    now=self._now(),
                )
        except BaseException:
            await lease.release()
            raise
        if not began:
            await lease.release()
            return None
        lease.start()
        return _Attempt(attempt_no=attempt_no, generation_id=generation_id, lease=lease)

    async def _operation_in_flight(self, scope: SkillManageScope, operation_id: str) -> bool:
        """Whether the operation row's current attempt may still publish the candidate it names."""

        row = await self._storage.get_operation(scope, operation_id)
        return row is not None and _holds_candidate(row)

    async def _writer_alive(self, scope: SkillManageScope, operation_id: str) -> bool:
        lease = await self._storage.get_writer_lease(scope)
        if lease is None or lease.get("owner_operation_id") != operation_id:
            return False
        expires_at = lease.get("expires_at")
        if expires_at is None:
            return False
        expires = datetime.fromisoformat(str(expires_at))
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        return expires > now

    def _replay(self, existing: dict[str, Any]) -> SkillManageOperationResult:
        status = OperationStatus(existing["status"])
        if status is OperationStatus.COMMITTED:
            payload = dict(existing.get("result_payload") or {})
            counts = {name: int(payload.get(name) or 0) for name in _RESULT_COUNT_FIELDS}
            return SkillManageOperationResult(
                operation_id=existing["operation_id"],
                status=status,
                base_revision=existing.get("base_revision"),
                result_revision=existing.get("result_revision"),
                generation_id=existing.get("current_generation_id"),
                warnings=tuple(payload.get("warnings") or ()),
                **counts,
            )
        return SkillManageOperationResult(
            operation_id=existing["operation_id"],
            status=status,
            error_code=existing.get("error_code"),
            error_retryable=existing.get("error_retryable"),
        )

    # --------------------------------------------------------------- commit

    async def _commit_generation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        attempt: _Attempt,
        *,
        algorithm: SkillManageAlgorithm,
        profile: EmbeddingProfile,
        plan_and_head: PlanAndHead,
        noop_ok: bool,
        expected_revision: int | None = None,
    ) -> SkillManageOperationResult:
        """Plan, embed, stage, write, validate and publish one attempt's generation.

        Any failure after the attempt began is recorded on the operation row
        (``failed`` with the typed code and retryable flag, ``conflict`` for a
        lost head CAS or a fenced-out lease, ``cancelled`` for a pre-publish
        cancellation) and the staged generation is discarded, so the row never
        strands in ``running``/``validating`` while this process is alive.
        Every row transition is fenced by the attempt's token, so a superseded
        attempt can never overwrite the row of the attempt that replaced it.
        """

        lease = attempt.lease
        generation_id = attempt.generation_id
        staged = False
        try:
            plan, head, base_snapshot = await plan_and_head()
            base_revision = head.get("active_revision") if head is not None else None
            base_generation_id = head.get("active_generation_id") if head is not None else None
            base_profile = _stored_profile(head) if head is not None else None

            if expected_revision is not None and base_revision != expected_revision:
                # Optimistic lock: the caller asserted the head revision it planned
                # against. The writer lease only serialises writers; a divergent
                # snapshot (another writer already advanced the head) must be a
                # conflict, not a silent rebase against a different base.
                raise SkillManageRevisionConflictError(
                    expected_revision=expected_revision, actual_revision=base_revision
                )

            if noop_ok and base_snapshot is not None and base_generation_id is not None:
                if _plan_unchanged(plan, base_snapshot):
                    # No-op mutation (e.g. removing a Skill that is already absent):
                    # do not create a new generation or advance the revision. The
                    # row's current generation becomes the base generation, which
                    # is what this attempt reports; if finalize never lands, crash
                    # recovery then converges from that generation's status
                    # instead of treating the never-staged candidate as lost.
                    #
                    # There is no publish here to fence the attempt, so the lease
                    # is re-asserted first: a writer that stalled past its TTL
                    # may have planned against a base another writer has since
                    # retired (or collected, which reads back as an empty
                    # snapshot and makes any removal look like a no-op).
                    if not await _storage_call(lease.renew()):
                        raise SkillManageWriterConflictError(graph_id=scope.graph_id)
                    result = SkillManageOperationResult(
                        operation_id=operation_id,
                        status=OperationStatus.COMMITTED,
                        base_revision=base_revision,
                        result_revision=base_revision,
                        generation_id=base_generation_id,
                        node_count=len(plan.nodes),
                        edge_count=len(plan.edges),
                        vector_count=int(
                            (await _storage_call(self._storage.get_generation(scope, base_generation_id)) or {}).get(
                                "vector_count"
                            )
                            or 0
                        ),
                    )
                    if not await _storage_call(
                        self._storage.begin_validation(
                            scope,
                            operation_id,
                            fencing_token=lease.token,
                            result_generation_id=base_generation_id,
                            now=self._now(),
                        )
                    ):
                        raise SkillManageWriterConflictError(graph_id=scope.graph_id)
                    return await self._await_switch(self._finalize(scope, operation_id, attempt, result))

            new_revision = 0 if base_revision is None else int(base_revision) + 1
            embeddings, reused, fresh = await self._prepare_embeddings(
                algorithm,
                scope,
                plan,
                profile,
                run_id=operation_id,
                base_generation_id=base_generation_id if base_profile == profile else None,
            )
            # "May be staged" from here on: an exception (or cancellation)
            # delivered after the INSERT landed but before the call returned
            # must still discard the candidate, and discarding an id that
            # never landed is a no-op.
            staged = True
            await _storage_call(
                self._storage.stage_generation(
                    scope,
                    generation_id,
                    operation_id=operation_id,
                    attempt_no=attempt.attempt_no,
                    base_revision=base_revision,
                    target_revision=new_revision,
                    now=self._now(),
                )
            )
            await _storage_call(
                self._storage.write_snapshot(
                    scope,
                    generation_id,
                    nodes=list(plan.nodes),
                    edges=list(plan.edges),
                    memberships=[membership.model_dump(mode="python") for membership in plan.memberships],
                )
            )
            await _storage_call(self._storage.write_embeddings(scope, generation_id, embeddings=embeddings))
            await self._validate_generation(scope, generation_id, plan, embeddings, profile)
            if lease.lost or not await _storage_call(
                self._storage.begin_validation(scope, operation_id, fencing_token=lease.token, now=self._now())
            ):
                raise SkillManageWriterConflictError(graph_id=scope.graph_id)
            result = SkillManageOperationResult(
                operation_id=operation_id,
                status=OperationStatus.COMMITTED,
                base_revision=base_revision,
                result_revision=new_revision,
                generation_id=generation_id,
                inserted=fresh,
                reused=reused,
                node_count=len(plan.nodes),
                edge_count=len(plan.edges),
                scene_count=0,
                vector_count=len(embeddings),
                warnings=tuple(warning.model_dump(mode="json") for warning in plan.warnings),
            )
        except SkillManageWriterConflictError:
            return await self._record_conflict(
                scope,
                operation_id,
                attempt,
                error_code=_WRITER_CONFLICT,
                error_message="writer lease was lost before publish",
                discard_generation_id=generation_id if staged else None,
            )
        except SkillManageRevisionConflictError:
            return await self._record_conflict(
                scope,
                operation_id,
                attempt,
                error_code=_REVISION_CONFLICT,
                error_message="head revision no longer matches the request expected_revision",
                discard_generation_id=generation_id if staged else None,
            )
        except asyncio.CancelledError:
            await self._record_cancelled(
                scope, operation_id, attempt, discard_generation_id=generation_id if staged else None
            )
            raise
        except SkillManageOperationPendingRepairError:
            # Raised only by the no-op path's finalize, after the lease was
            # released: the row is deliberately left ``validating`` for crash
            # recovery, and must not be re-recorded as a failed attempt.
            raise
        except Exception as exc:
            transient_storage = isinstance(exc, _TransientStorageFailure)
            failure = exc.original if transient_storage else exc
            recorded = await self._record_failure(
                scope,
                operation_id,
                attempt,
                error_code=str(
                    getattr(failure, "error_code", None)
                    or (_STORAGE_UNAVAILABLE if transient_storage else _OPERATION_FAILED)
                ),
                error_message=type(failure).__name__,
                error_retryable=True if transient_storage else getattr(failure, "retryable", None),
                discard_generation_id=generation_id if staged else None,
            )
            if recorded is False:
                return await self._superseded_outcome(scope, operation_id)
            if transient_storage:
                raise failure from None
            raise

        return await self._await_switch(
            self._publish_and_finalize(
                scope,
                operation_id,
                attempt,
                algorithm=algorithm,
                profile=profile,
                expected_revision=base_revision,
                result=result,
            )
        )

    @staticmethod
    async def _await_switch(switch: Awaitable[SkillManageOperationResult]) -> SkillManageOperationResult:
        """Run the visibility switch (publish/finalize) to completion whatever the caller does.

        Once the switch has started its real outcome must be reported rather
        than leaving the head and the operation row disagreeing. The shield is
        kept up for as long as the caller keeps being cancelled, because
        awaiting the task unshielded would cancel the switch itself.
        """

        task = asyncio.ensure_future(switch)
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    return task.result()

    async def _publish_and_finalize(
        self,
        scope: SkillManageScope,
        operation_id: str,
        attempt: _Attempt,
        *,
        algorithm: SkillManageAlgorithm,
        profile: EmbeddingProfile,
        expected_revision: int | None,
        result: SkillManageOperationResult,
    ) -> SkillManageOperationResult:
        """Run the visibility switch and settle the operation row from its real outcome.

        Fencing and consistency rejections are raised before the transaction
        writes anything, so they are recorded directly. Any other exception
        leaves the commit outcome unknown: the head and the candidate's status
        decide between finalizing, a retryable failure and pending repair.
        """

        generation_id = attempt.generation_id
        try:
            published = await self._storage.publish_generation(
                scope,
                generation_id,
                expected_revision=expected_revision,
                new_revision=int(result.result_revision or 0),
                embedding_profile=profile.model_dump(mode="json"),
                owner_operation_id=operation_id,
                fencing_token=attempt.lease.token,
                algorithm_name=getattr(algorithm, "algorithm_name", None),
                algorithm_version=getattr(algorithm, "algorithm_version", None),
                now=self._now(),
            )
        except SkillManageWriterConflictError:
            return await self._record_conflict(
                scope,
                operation_id,
                attempt,
                error_code=_WRITER_CONFLICT,
                error_message="fencing token rejected at publish",
                discard_generation_id=generation_id,
            )
        except SkillManageConsistencyError as exc:
            recorded = await self._record_failure(
                scope,
                operation_id,
                attempt,
                error_code=exc.error_code,
                error_message=type(exc).__name__,
                error_retryable=False,
                discard_generation_id=generation_id,
            )
            if recorded is False:
                return await self._superseded_outcome(scope, operation_id)
            raise
        except Exception as exc:
            outcome = await self._publish_outcome(scope, generation_id)
            if outcome is True:
                return await self._finalize(scope, operation_id, attempt, result)
            if outcome is False:
                recorded = await self._record_failure(
                    scope,
                    operation_id,
                    attempt,
                    error_code=_STORAGE_UNAVAILABLE,
                    error_message=type(exc).__name__,
                    error_retryable=True,
                    discard_generation_id=generation_id,
                )
                if recorded is False:
                    return await self._superseded_outcome(scope, operation_id)
                raise
            await attempt.lease.release()
            # The head says nothing about this candidate; the row might still.
            # A same-key retry that took the lapsed lease over may have settled
            # the key meanwhile, and then the reply follows the row like every
            # other superseded outcome. Only a row still in this attempt is
            # genuinely undecided.
            if not await self._row_is_at_attempt(scope, operation_id, attempt):
                return await self._superseded_outcome(scope, operation_id)
            raise SkillManageOperationPendingRepairError(
                operation_id, PendingRepairReason.PUBLISH_OUTCOME_UNKNOWN.value
            ) from exc
        if not published:
            return await self._record_conflict(
                scope,
                operation_id,
                attempt,
                error_code=_REVISION_CONFLICT,
                error_message="head revision changed before publish",
                discard_generation_id=generation_id,
            )
        return await self._finalize(scope, operation_id, attempt, result)

    async def _publish_outcome(self, scope: SkillManageScope, generation_id: str) -> bool | None:
        """Whether an interrupted publish committed: True/False when provable, else ``None``.

        A plain read cannot prove a rollback: a commit whose acknowledgement
        was lost may still be landing and is invisible until it does, while
        the head still shows the base revision. The candidate is therefore
        fenced first — the ``ready -> failed`` CAS queues behind the in-flight
        commit and is re-evaluated against the committed row — so winning it
        proves the publish never happened (and can no longer happen), and
        losing it means the candidate's status now tells the truth.
        """

        try:
            if await self._storage.mark_generation_failed(scope, generation_id, now=self._now()):
                return False
            generation = await self._storage.get_generation(scope, generation_id)
            head = await self._storage.get_graph_head(scope)
        except Exception:
            return None
        if generation is not None and generation.get("status") in ("active", "retired"):
            return True
        if head is not None and head.get("active_generation_id") == generation_id:
            return True
        return None

    async def _finalize(
        self,
        scope: SkillManageScope,
        operation_id: str,
        attempt: _Attempt,
        result: SkillManageOperationResult,
    ) -> SkillManageOperationResult:
        """Mark the operation committed, release the lease and run deferred GC.

        The head already points at this attempt's generation, so the only
        admissible outcomes are ``committed`` or a typed pending-repair error
        that leaves the row ``validating`` for crash recovery to converge.
        """

        payload = {name: getattr(result, name) for name in _RESULT_COUNT_FIELDS}
        payload["warnings"] = list(result.warnings)
        committed = False
        for _ in range(2):
            try:
                committed = await self._storage.record_operation_result(
                    scope,
                    operation_id,
                    base_revision=result.base_revision,
                    result_revision=int(result.result_revision or 0),
                    generation_id=str(result.generation_id),
                    result_payload=payload,
                    fencing_token=attempt.lease.token,
                    now=self._now(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            break
        if not committed:
            # The row moved under us (a crash-recovery claim after our lease
            # lapsed) or storage is unavailable; the head stays the source of
            # truth and repair converges the row from it.
            await attempt.lease.release()
            try:
                existing = await self._storage.get_operation(scope, operation_id)
            except Exception:
                existing = None
            if existing is not None and existing.get("status") == OperationStatus.COMMITTED.value:
                # Our own commit landed and only its acknowledgement was lost:
                # this publish still retired a generation, so the deferred GC
                # runs as it would after a clean commit.
                replayed = self._replay(existing)
                gc_warnings = await self._deferred_gc(scope)
                if gc_warnings:
                    return replayed.model_copy(update={"warnings": (*replayed.warnings, *gc_warnings)})
                return replayed
            raise SkillManageOperationPendingRepairError(operation_id, PendingRepairReason.FINALIZE_UNAVAILABLE.value)
        await attempt.lease.release()
        gc_warnings = await self._deferred_gc(scope)
        if gc_warnings:
            return result.model_copy(update={"warnings": (*result.warnings, *gc_warnings)})
        return result

    async def _validate_generation(
        self,
        scope: SkillManageScope,
        generation_id: str,
        plan: SkillManageBuildPlan,
        embeddings: list[SkillGraphEmbedding],
        profile: EmbeddingProfile,
    ) -> None:
        """Check references, profile and read-back counts, then mark the generation ready."""

        node_ids = {node.graph_node_id for node in plan.nodes}
        if len(node_ids) != len(plan.nodes):
            raise SkillManageValidationError("duplicate graph_node_id in plan")
        # The edge row key is the edge_id, so a duplicate would be collapsed by
        # the upsert and the recorded edge_count would overstate what a reader
        # gets back.
        edge_ids = {edge.edge_id for edge in plan.edges}
        if len(edge_ids) != len(plan.edges):
            raise SkillManageValidationError("duplicate edge_id in plan")
        skill_versions = {
            node.graph_node_id: (node.skill_id, node.version_id) for node in plan.nodes if node.node_type == "skill"
        }
        for edge in plan.edges:
            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                raise SkillManageValidationError(f"edge {edge.edge_id} references a missing node")
        seen_skills: set[str] = set()
        for membership in plan.memberships:
            if skill_versions.get(membership.graph_node_id) != (membership.skill_id, membership.version_id):
                raise SkillManageValidationError(f"membership for {membership.skill_id} does not match its node")
            if membership.skill_id in seen_skills:
                raise SkillManageValidationError(f"duplicate membership for {membership.skill_id}")
            seen_skills.add(membership.skill_id)
        required = {text.graph_node_id for text in plan.embedding_texts}
        embedded = [embedding.graph_node_id for embedding in embeddings]
        if len(set(embedded)) != len(embedded) or set(embedded) != required:
            raise SkillManageValidationError("embedded nodes do not match the nodes requiring embeddings")
        for embedding in embeddings:
            if (
                embedding.embedding_model != profile.model
                or embedding.embedding_profile_version != profile.template_version
                or len(embedding.vector) != profile.dimension
            ):
                raise SkillManageValidationError(f"vector for {embedding.graph_node_id} does not match the profile")

        stored = await _storage_call(self._storage.load_snapshot(scope, generation_id))
        stored_nodes = {node.graph_node_id for node in stored.nodes}
        stored_edges = {edge.edge_id for edge in stored.edges}
        stored_memberships = {(m.skill_id, m.version_id, m.graph_node_id) for m in stored.memberships}
        planned_memberships = {(m.skill_id, m.version_id, m.graph_node_id) for m in plan.memberships}
        if stored_nodes != node_ids or stored_edges != edge_ids:
            raise SkillManageValidationError("stored nodes/edges differ from the plan")
        if stored_memberships != planned_memberships:
            raise SkillManageValidationError("stored memberships differ from the plan")
        vector_count = await _storage_call(self._storage.count_embeddings(scope, generation_id))
        if vector_count != len(embeddings):
            raise SkillManageValidationError(f"stored {vector_count} vectors, expected {len(embeddings)}")
        ready = await _storage_call(
            self._storage.mark_generation_ready(
                scope,
                generation_id,
                node_count=len(plan.nodes),
                edge_count=len(plan.edges),
                vector_count=vector_count,
                counts_hash=generation_counts_hash(
                    node_ids=node_ids,
                    edge_ids=stored_edges,
                    memberships=planned_memberships,
                    vector_node_ids=embedded,
                ),
                now=self._now(),
            )
        )
        if not ready:
            raise SkillManageValidationError(f"generation {generation_id} is no longer staging")

    async def _generation_gone(self, scope: SkillManageScope, generation_id: str) -> bool:
        try:
            return await self._storage.get_generation(scope, generation_id) is None
        except Exception:
            return False

    async def _deferred_gc(self, scope: SkillManageScope) -> list[dict[str, Any]]:
        """Collect retired generations past retention; a failure only warns."""

        try:
            await self._storage.gc_retired_generations(
                scope,
                keep=self._settings.retired_generations_to_keep,
                min_age_seconds=self._settings.retired_generation_min_age_seconds,
                now=self._now(),
            )
        except SkillManageGcPartialFailureError as exc:
            return _gc_deferred_warnings(exc.deferred)
        except Exception as exc:
            return [
                SkillManageWarning(
                    code=_GC_DEFERRED,
                    message="retired generation GC failed and is deferred to maintenance",
                    details={"reason": type(exc).__name__},
                ).model_dump(mode="json")
            ]
        return []

    async def _record_failure(
        self,
        scope: SkillManageScope,
        operation_id: str,
        attempt: _Attempt,
        *,
        error_code: str,
        error_message: str,
        error_retryable: bool | None,
        discard_generation_id: str | None,
    ) -> bool | None:
        """Best-effort bookkeeping for a failed write; never masks the original error.

        Returns whether the fenced ``failed`` CAS applied (``None`` when the
        storage call itself failed). ``False`` means the row no longer belongs
        to this attempt, so the caller reports the row's outcome instead of
        this attempt's private one. The lease is released in ``finally``: a
        cancellation landing during the bookkeeping must still stop the
        renewal task, or the lease would be renewed forever by a task nobody
        owns.
        """

        recorded: bool | None = None
        try:
            recorded = await self._storage.fail_operation(
                scope,
                operation_id,
                error_code=error_code,
                error_message=error_message,
                error_retryable=error_retryable,
                fencing_token=attempt.lease.token,
                now=self._now(),
            )
            if discard_generation_id is not None:
                await self._storage.discard_generation(scope, discard_generation_id)
        except Exception:
            pass
        finally:
            await attempt.lease.release()
        return recorded

    async def _record_conflict(
        self,
        scope: SkillManageScope,
        operation_id: str,
        attempt: _Attempt,
        *,
        error_code: str,
        error_message: str,
        discard_generation_id: str | None,
    ) -> SkillManageOperationResult:
        """Record a terminal conflict (lost CAS or fenced-out lease) and discard the candidate.

        The conflict is only this attempt's outcome while the fenced CAS
        applies; when it is rejected the row already belongs to a newer
        attempt and the reply follows the row (see ``_superseded_outcome``).
        """

        recorded: bool | None = None
        try:
            recorded = await self._storage.conflict_operation(
                scope,
                operation_id,
                error_code=error_code,
                error_message=error_message,
                fencing_token=attempt.lease.token,
                now=self._now(),
            )
            if discard_generation_id is not None:
                await self._storage.discard_generation(scope, discard_generation_id)
        except Exception:
            pass
        finally:
            await attempt.lease.release()
        if recorded is False:
            return await self._superseded_outcome(scope, operation_id)
        if recorded is None:
            # The conflict never reached the row: it is still in flight under
            # this attempt and crash recovery will settle it (usually as a
            # retryable failure), so a terminal, non-retryable answer here
            # would contradict every later reader. The same key can retry.
            return self._writer_conflict(operation_id)
        return SkillManageOperationResult(
            operation_id=operation_id,
            status=OperationStatus.CONFLICT,
            error_code=error_code,
            error_retryable=False,
        )

    async def _row_is_at_attempt(self, scope: SkillManageScope, operation_id: str, attempt: _Attempt) -> bool:
        """Whether the row is still in flight under ``attempt`` (unreadable counts as yes: nothing is known)."""

        try:
            existing = await self._storage.get_operation(scope, operation_id)
        except Exception:
            return True
        if existing is None:
            return True
        status = OperationStatus(existing.get("status") or OperationStatus.PENDING.value)
        return (
            status in (OperationStatus.RUNNING, OperationStatus.VALIDATING)
            and int(existing.get("attempt_no") or 1) == attempt.attempt_no
        )

    async def _superseded_outcome(self, scope: SkillManageScope, operation_id: str) -> SkillManageOperationResult:
        """The reply for an attempt whose fenced row transition was rejected.

        A rejected fence means a same-key retry took the lease over after it
        lapsed and opened a newer attempt, so this attempt's private outcome
        is no longer the key's: a settled row is replayed exactly as a retry
        would replay it, and a row still in flight is the retryable writer
        conflict (the row is left as it is; the same key can simply retry).
        """

        try:
            existing = await self._storage.get_operation(scope, operation_id)
        except Exception:
            existing = None
        if existing is not None:
            status = OperationStatus(existing.get("status") or OperationStatus.PENDING.value)
            if status is OperationStatus.CANCELLED:
                raise SkillManageOperationCancelledError(operation_id)
            if status in TERMINAL_OPERATION_STATUSES or status is OperationStatus.FAILED:
                return self._replay(existing)
        return self._writer_conflict(operation_id)

    async def _record_cancelled(
        self,
        scope: SkillManageScope,
        operation_id: str,
        attempt: _Attempt,
        *,
        discard_generation_id: str | None,
    ) -> None:
        try:
            await self._storage.cancel_operation(
                scope,
                operation_id,
                error_code=_OPERATION_CANCELLED,
                fencing_token=attempt.lease.token,
                now=self._now(),
            )
            if discard_generation_id is not None:
                await self._storage.discard_generation(scope, discard_generation_id)
        except Exception:
            pass
        finally:
            await attempt.lease.release()

    async def _prepare_embeddings(
        self,
        algorithm: SkillManageAlgorithm,
        scope: SkillManageScope,
        plan: SkillManageBuildPlan,
        profile: EmbeddingProfile,
        *,
        run_id: str,
        base_generation_id: str | None,
    ) -> tuple[list[SkillGraphEmbedding], int, int]:
        """Return the new generation's vectors plus (reused, freshly embedded) counts.

        ``base_generation_id`` must only be passed when that generation was
        embedded under exactly ``profile``; a base vector is then reused when
        node identity and content hash match, and everything else is embedded
        again in one batched model call.
        """

        if not plan.embedding_texts:
            return [], 0, 0
        existing: dict[tuple[str, str], SkillGraphEmbedding] = {}
        if base_generation_id is not None:
            for prior in await _storage_call(self._storage.load_embeddings(scope, base_generation_id)):
                if (
                    prior.embedding_model == profile.model
                    and prior.embedding_profile_version == profile.template_version
                    and len(prior.vector) == profile.dimension
                ):
                    existing[(prior.graph_node_id, prior.content_hash)] = prior
        nodes_by_id = {node.graph_node_id: node for node in plan.nodes}
        embeddings: dict[str, SkillGraphEmbedding] = {}
        pending = []
        for item in plan.embedding_texts:
            node = nodes_by_id.get(item.graph_node_id)
            if node is None:
                continue
            prior = existing.get((item.graph_node_id, item.content_hash))
            if prior is not None:
                embeddings[item.graph_node_id] = prior.model_copy(
                    update={"node_type": node.node_type, "skill_id": node.skill_id, "version_id": node.version_id}
                )
            else:
                pending.append(item)
        reused = len(embeddings)
        if pending:
            embedded = await algorithm.embed(
                SkillManageEmbedInput(
                    embedding_profile=profile,
                    texts=tuple(item.text for item in pending),
                    task=_BUILD_EMBED_TASK,
                    run_id=run_id,
                )
            )
            for item, vector in zip(pending, embedded.vectors, strict=True):
                node = nodes_by_id[item.graph_node_id]
                embeddings[item.graph_node_id] = SkillGraphEmbedding(
                    graph_node_id=item.graph_node_id,
                    node_type=node.node_type,
                    skill_id=node.skill_id,
                    version_id=node.version_id,
                    content_hash=item.content_hash,
                    embedding_model=profile.model,
                    embedding_profile_version=profile.template_version,
                    vector=vector,
                )
        ordered = [embeddings[item.graph_node_id] for item in plan.embedding_texts if item.graph_node_id in embeddings]
        return ordered, reused, len(pending)

    def _effective_profile(
        self,
        requested: EmbeddingProfile | None,
        *,
        stored: EmbeddingProfile | None = None,
    ) -> EmbeddingProfile:
        """Resolve the profile for one call: request, then configuration, then the head's own."""

        self._check_requested_profile(requested)
        profile = requested or self._embedding_profile or stored
        if profile is None:
            raise SkillConfigurationError("skill_manage embedding profile is not configured")
        return profile

    def _check_requested_profile(self, requested: EmbeddingProfile | None) -> None:
        configured = self._embedding_profile
        if requested is not None and configured is not None and requested != configured:
            raise SkillManageProfileMismatchError(
                expected=configured.model_dump(mode="json"), actual=requested.model_dump(mode="json")
            )

    @staticmethod
    def _operation_id(scope: SkillManageScope, idempotency_key: str) -> str:
        """Deterministic operation ID for an idempotency key within a graph scope."""

        return str(uuid5(NAMESPACE_URL, f"skill-manage:{scope.project_id}:{scope.graph_id}:{idempotency_key}"))

    async def _resolve_documents(self, request: SkillManageBuildRequest) -> tuple[SkillDocumentInput, ...]:
        """Resolve exact version refs into canonical Skill documents via the loader."""

        if self._version_loader is None:
            return ()
        resolved: list[SkillDocumentInput] = []
        for ref in request.version_refs:
            document = await self._version_loader.load(ref, project_id=request.project_id)
            if isinstance(document, SkillDocumentInput):
                resolved.append(document)
        return tuple(resolved)

    async def _resolve_document(self, ref: SkillVersionRef, project_id: str) -> SkillDocumentInput | None:
        """Resolve one exact version ref into its canonical document, or ``None``.

        A missing loader or an unresolvable ref degrades to ``None`` so the
        upsert algorithm keeps the topology unchanged rather than failing the
        whole mutation.
        """

        if self._version_loader is None:
            return None
        try:
            document = await self._version_loader.load(ref, project_id=project_id)
        except SkillVersionNotFound:
            return None
        return document if isinstance(document, SkillDocumentInput) else None

    async def get_skill_manage_operation(
        self,
        operation_id: str,
        *,
        project_id: str,
        graph_id: str,
    ) -> dict[str, Any] | None:
        scope = SkillManageScope(project_id=project_id, graph_id=graph_id)
        return await self._storage.get_operation(scope, operation_id)

    # ---------------------------------------------------------- maintenance

    async def repair_skill_index(
        self,
        *,
        project_id: str,
        graph_id: str,
        stale_candidate_age_seconds: float | None = None,
    ) -> SkillManageRepairReport:
        """Explicit, idempotent, scope-bound repair (spec §9).

        Converges every stranded ``running``/``validating`` row whose writer
        lease has lapsed and every reconcilable ``pending_repair`` row through
        the frozen crash-recovery rules, and discards ``staging``/``ready``
        candidates that are older than ``stale_candidate_age_seconds`` and no
        longer owned by a live attempt. The head and the canonical Skill
        versions are never touched: a candidate that the head already points
        at is left for the operation reconcile to settle. Rows still held by
        a live writer are reported as skipped and left alone; a row the
        crash-recovery rules cannot settle is reported ``operation_unresolved``
        with a ``SKILL_MANAGE_CONSISTENCY_VIOLATION`` warning and left
        ``pending_repair`` for manual handling.
        """

        scope = SkillManageScope(project_id=project_id, graph_id=graph_id)
        max_age = (
            self._settings.writer_lease_ttl_seconds
            if stale_candidate_age_seconds is None
            else stale_candidate_age_seconds
        )
        actions: list[SkillManageRepairAction] = []
        warnings: list[dict[str, Any]] = []

        stranded = await self._storage.list_operations(
            scope,
            statuses=(
                OperationStatus.RUNNING.value,
                OperationStatus.VALIDATING.value,
                OperationStatus.PENDING_REPAIR.value,
            ),
        )
        for listed in sorted(stranded, key=lambda item: str(item.get("operation_id"))):
            operation_id = str(listed["operation_id"])
            listed_status = str(listed.get("status"))
            skipped_reason: str | None = None
            if listed_status == OperationStatus.PENDING_REPAIR.value:
                reason = listed.get("pending_repair_reason")
                if reason not in _RECONCILABLE_REPAIR_REASONS:
                    skipped_reason = str(reason)
            if skipped_reason is None and await self._writer_alive(scope, operation_id):
                skipped_reason = "writer lease is alive"
            held = None if skipped_reason is not None else await self._take_over_lease(scope, operation_id)
            if skipped_reason is None and held is None:
                skipped_reason = "writer lease is held by another writer"
            if skipped_reason is not None:
                actions.append(
                    SkillManageRepairAction(
                        kind="operation_skipped",
                        operation_id=operation_id,
                        from_status=listed_status,
                        reason=skipped_reason,
                    )
                )
                continue
            assert held is not None
            # The lease is held from here through the reconcile so no same-key
            # retry can open a new attempt underneath, and a slow but not-yet-
            # dead writer stays fenced while its candidate is settled or
            # discarded. The row is re-read under the lease: the listing may be
            # arbitrarily stale, and reconciling against a superseded
            # attempt_no/current_generation_id would settle a live newer
            # attempt as failed/conflict. The pending_repair reason is part of
            # that comparison: a retry reservation taken after the listing
            # keeps the attempt fields and may only be resumed, never settled.
            try:
                row = await self._storage.get_operation(scope, operation_id)
                if row is None:
                    actions.append(
                        SkillManageRepairAction(
                            kind="operation_skipped",
                            operation_id=operation_id,
                            from_status=listed_status,
                            reason="operation row vanished",
                        )
                    )
                    continue
                status = str(row.get("status"))
                attempt_no = int(row.get("attempt_no") or 1)
                generation_id = row.get("current_generation_id")
                if _attempt_view(row) != _attempt_view(listed):
                    actions.append(
                        SkillManageRepairAction(
                            kind="operation_skipped",
                            operation_id=operation_id,
                            from_status=status,
                            reason="operation moved since it was listed",
                        )
                    )
                    continue
                if not await held.renew():
                    # The lease is not renewed in the background here; if it
                    # lapsed since the take-over, the row may be another writer's.
                    actions.append(
                        SkillManageRepairAction(
                            kind="operation_skipped",
                            operation_id=operation_id,
                            from_status=status,
                            reason="writer lease is held by another writer",
                        )
                    )
                    continue
                if status != OperationStatus.PENDING_REPAIR.value:
                    claimed = await self._storage.claim_crash_recovery(
                        scope,
                        operation_id,
                        expected_attempt_no=attempt_no,
                        expected_generation_id=generation_id,
                        now=self._now(),
                    )
                    if not claimed:
                        actions.append(
                            SkillManageRepairAction(
                                kind="operation_skipped",
                                operation_id=operation_id,
                                from_status=status,
                                reason="crash-recovery claim lost",
                            )
                        )
                        continue
                converged = await self._storage.repair_crash_recovery(
                    scope,
                    operation_id,
                    attempt_no=attempt_no,
                    generation_id=generation_id,
                    now=self._now(),
                )
                # The adapter settles only the attempt it was given and
                # otherwise reports the row's current status — possibly a
                # newer attempt's terminal one, or a retry reservation, when
                # the held lease lapsed inside the reconcile and a same-key
                # retry moved on. The row is re-read so the report describes
                # this attempt rather than whatever the row has become.
                after = await self._storage.get_operation(scope, operation_id)
            finally:
                await held.release()
            if after is None or _attempt_fields(after) != (attempt_no, generation_id):
                actions.append(
                    SkillManageRepairAction(
                        kind="operation_skipped",
                        operation_id=operation_id,
                        from_status=status,
                        reason="operation moved during the reconcile",
                    )
                )
                continue
            if converged == OperationStatus.PENDING_REPAIR.value:
                if (after.get("status"), after.get("pending_repair_reason")) != (
                    OperationStatus.PENDING_REPAIR.value,
                    PendingRepairReason.CRASH_RECOVERY.value,
                ):
                    # Same attempt fields but not the row this reconcile
                    # claimed: a retry reservation taken meanwhile is what
                    # made the settle CAS lose, not an ambiguous head.
                    actions.append(
                        SkillManageRepairAction(
                            kind="operation_skipped",
                            operation_id=operation_id,
                            from_status=status,
                            reason="operation moved during the reconcile",
                        )
                    )
                    continue
                actions.append(
                    SkillManageRepairAction(
                        kind="operation_unresolved",
                        operation_id=operation_id,
                        generation_id=generation_id,
                        from_status=status,
                        to_status=converged,
                        reason=SkillManageConsistencyError.error_code,
                    )
                )
                warnings.append(
                    SkillManageWarning(
                        code=SkillManageConsistencyError.error_code,
                        message="operation could not be converged; head/generation state is ambiguous",
                        details={"operation_id": operation_id, "generation_id": generation_id},
                    ).model_dump(mode="json")
                )
                continue
            if converged not in (
                OperationStatus.COMMITTED.value,
                OperationStatus.FAILED.value,
                OperationStatus.CONFLICT.value,
            ):
                # The adapter settles only the attempt it was given and reports
                # the row's actual status otherwise: the row moved underneath.
                actions.append(
                    SkillManageRepairAction(
                        kind="operation_skipped",
                        operation_id=operation_id,
                        from_status=status,
                        reason="operation moved during the reconcile",
                    )
                )
                continue
            actions.append(
                SkillManageRepairAction(
                    kind="operation_reconciled",
                    operation_id=operation_id,
                    generation_id=generation_id,
                    from_status=status,
                    to_status=converged,
                )
            )

        head = await self._storage.get_graph_head(scope)
        active_generation_id = head.get("active_generation_id") if head is not None else None
        live_operation_ids = {
            str(row["operation_id"])
            for row in await self._storage.list_operations(
                scope,
                statuses=(
                    OperationStatus.PENDING.value,
                    OperationStatus.RUNNING.value,
                    OperationStatus.VALIDATING.value,
                    OperationStatus.PENDING_REPAIR.value,
                ),
            )
            if _holds_candidate(row)
        }
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        cutoff = now - timedelta(seconds=max_age)
        candidates = await self._storage.list_generations(
            scope,
            statuses=(
                GenerationStatus.STAGING.value,
                GenerationStatus.READY.value,
                GenerationStatus.FAILED.value,
                GenerationStatus.DELETED.value,
            ),
        )
        for generation in sorted(candidates, key=lambda item: str(item.get("generation_id"))):
            generation_id = str(generation["generation_id"])
            owner = generation.get("operation_id")
            if generation_id == active_generation_id or str(owner) in live_operation_ids:
                continue
            # A failed candidate is one the fence already settled and a deleted
            # one is a tombstoned deletion that broke off: neither can ever be
            # published, so both are collected regardless of age; only a
            # staging/ready one might still belong to a writer that is merely slow.
            if generation.get("status") not in (GenerationStatus.FAILED.value, GenerationStatus.DELETED.value):
                created_at = _parse_instant(generation.get("created_at"))
                if created_at is None or created_at > cutoff:
                    continue
            # The owner is re-read right before the discard: an attempt begun
            # after the live listing (a small age threshold makes the window
            # matter) is still in flight and must keep its candidate.
            if owner is not None and await self._operation_in_flight(scope, str(owner)):
                continue
            try:
                await self._storage.discard_generation(scope, generation_id)
            except Exception as exc:
                # The row is deleted last, so a discard whose acknowledgement
                # was lost after landing leaves no row: that is a discard, not
                # a deferral the next pass would never get to report. A failing
                # re-read (store down) keeps it a deferral.
                if not await self._generation_gone(scope, generation_id):
                    warnings.append(
                        SkillManageWarning(
                            code=_REPAIR_DEFERRED,
                            message="stale candidate generation could not be discarded",
                            details={"generation_id": generation_id, "reason": type(exc).__name__},
                        ).model_dump(mode="json")
                    )
                    continue
            actions.append(
                SkillManageRepairAction(
                    kind="candidate_discarded",
                    operation_id=str(owner) if owner is not None else None,
                    generation_id=generation_id,
                    from_status=str(generation.get("status")),
                )
            )
        # Records without a generation row (spec §9 "orphan vectors"): a
        # superseded writer's late snapshot/vector writes landing after its
        # candidate was discarded, or an interrupted discard. Every live
        # candidate stages its row before writing anything, so the records are
        # listed first: an id seen there has its row in the later listing
        # unless it truly belongs to nobody.
        referenced = await self._storage.list_snapshot_generation_ids(scope)
        known = {str(row["generation_id"]) for row in await self._storage.list_generations(scope)}
        for generation_id in sorted(referenced - known - {str(active_generation_id)}):
            # The listing is paged over an OFFSET cursor and can skip a row
            # when another row is deleted between two pages; a complement
            # computed from it must therefore re-read each id, since any
            # existing row means the candidate belongs to the sweep above.
            if await self._storage.get_generation(scope, generation_id) is not None:
                continue
            try:
                await self._storage.discard_generation(scope, generation_id)
            except Exception as exc:
                warnings.append(
                    SkillManageWarning(
                        code=_REPAIR_DEFERRED,
                        message="orphan generation records could not be deleted",
                        details={"generation_id": generation_id, "reason": type(exc).__name__},
                    ).model_dump(mode="json")
                )
                continue
            actions.append(
                SkillManageRepairAction(
                    kind="candidate_discarded",
                    generation_id=generation_id,
                    reason="orphan records without a generation row",
                )
            )
        return SkillManageRepairReport(
            project_id=project_id, graph_id=graph_id, actions=tuple(actions), warnings=tuple(warnings)
        )

    async def maintain_skill_index(
        self,
        *,
        project_id: str,
        graph_id: str,
    ) -> SkillManageMaintenanceReport:
        """Explicit, idempotent retired-generation GC (spec §10).

        The same policy the write path applies after each publish, callable on
        graphs that are read but no longer written so their retired snapshots
        do not accumulate forever. Never deletes the active generation.
        """

        scope = SkillManageScope(project_id=project_id, graph_id=graph_id)
        try:
            removed = await self._storage.gc_retired_generations(
                scope,
                keep=self._settings.retired_generations_to_keep,
                min_age_seconds=self._settings.retired_generation_min_age_seconds,
                now=self._now(),
            )
        except SkillManageGcPartialFailureError as exc:
            # Some generations are already gone: the report must say so even
            # though the pass did not finish cleanly.
            return SkillManageMaintenanceReport(
                project_id=project_id,
                graph_id=graph_id,
                removed_generation_ids=tuple(exc.removed),
                warnings=tuple(_gc_deferred_warnings(exc.deferred)),
            )
        except Exception as exc:
            return SkillManageMaintenanceReport(
                project_id=project_id,
                graph_id=graph_id,
                warnings=(
                    SkillManageWarning(
                        code=_GC_DEFERRED,
                        message="retired generation GC failed and is deferred to the next maintenance run",
                        details={"reason": type(exc).__name__},
                    ).model_dump(mode="json"),
                ),
            )
        return SkillManageMaintenanceReport(
            project_id=project_id, graph_id=graph_id, removed_generation_ids=tuple(removed)
        )


def _gc_deferred_warnings(deferred: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """One ``SKILL_MANAGE_GC_DEFERRED`` warning per generation a GC pass could not delete."""

    return [
        SkillManageWarning(
            code=_GC_DEFERRED,
            message="retired generation could not be deleted and is deferred to the next maintenance run",
            details={"generation_id": generation_id, "reason": reason},
        ).model_dump(mode="json")
        for generation_id, reason in deferred
    ]


def _stored_profile(head: dict[str, Any] | None) -> EmbeddingProfile | None:
    """The profile a head's active generation was embedded with; ``None`` for legacy heads."""

    raw = (head or {}).get("embedding_profile") or {}
    if not raw:
        return None
    return EmbeddingProfile.model_validate(raw)


def _dry_run_conflict() -> SkillManageOperationResult:
    """Dry-run shape of the optimistic-lock rejection: no operation row exists, so no ``operation_id``."""

    return SkillManageOperationResult(
        operation_id=None,
        status=OperationStatus.CONFLICT,
        error_code=_REVISION_CONFLICT,
        error_retryable=False,
    )


def _holds_candidate(row: dict[str, Any]) -> bool:
    """Whether an operation row's current attempt may still publish the candidate it names.

    A retry reservation keeps the failed attempt's fields while it waits for
    the same key to open the next attempt under a new generation id, so the
    candidate it names is that failed attempt's and can never be published;
    only the same key may resume the row, but nothing needs its candidate.
    """

    status = OperationStatus(row.get("status") or OperationStatus.PENDING.value)
    if status in TERMINAL_OPERATION_STATUSES or status is OperationStatus.FAILED:
        return False
    return (
        status is not OperationStatus.PENDING_REPAIR
        or row.get("pending_repair_reason") != PendingRepairReason.RETRY_ATTEMPT_RESERVATION.value
    )


def _attempt_fields(row: dict[str, Any]) -> tuple[int, Any]:
    """The attempt an operation row currently names; a reconcile settles exactly one of these."""

    return int(row.get("attempt_no") or 1), row.get("current_generation_id")


def _attempt_view(row: dict[str, Any]) -> tuple[Any, Any, int, Any]:
    """The part of an operation row a reconciler must see unchanged before acting on it."""

    return (row.get("status"), row.get("pending_repair_reason"), *_attempt_fields(row))


def _parse_instant(value: Any) -> datetime | None:
    """Parse an adapter timestamp (ISO string or datetime) as aware UTC; ``None`` when absent."""

    if value is None:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _plan_unchanged(plan, snapshot) -> bool:
    """True when a mutation plan leaves the snapshot identical (no-op).

    Nodes, edges and memberships are compared order-independently: the plan
    derives them from node order while a loaded snapshot returns them in
    physical record order. Nodes and edges are sorted by their logical id
    (they carry an unhashable payload dict, so a set is not an option);
    memberships are frozen and hashable so they compare as a multiset.
    """

    return (
        sorted(plan.nodes, key=lambda node: node.graph_node_id)
        == sorted(snapshot.nodes, key=lambda node: node.graph_node_id)
        and sorted(plan.edges, key=lambda edge: edge.edge_id) == sorted(snapshot.edges, key=lambda edge: edge.edge_id)
        and Counter(plan.memberships) == Counter(snapshot.memberships)
    )


__all__ = ["SkillManageService", "SkillManageStoragePort"]
