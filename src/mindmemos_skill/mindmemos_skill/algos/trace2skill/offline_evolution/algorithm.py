"""Offline evolution: host-side contract wrapper around the vendored pipeline.

The vendored pipeline (``impl/``) is a *synchronous* trajectory -> Skill
transformer. This wrapper is the async ``optimize()`` contract: it converts a
``Trace2SkillInput`` into the pipeline's scope-first layout, injects the model
and client from the host context, runs the pipeline on a worker thread, and
returns an unpersisted ``SkillCandidate`` plus a typed audit report.

The vendored pipeline is synchronous while the host ``LLMClient`` is async.
The wrapper therefore derives a synchronous OpenAI client from the host
client's router configuration, then injects that client and its model identity
through contextvars. Credentials stay in the process environment as the host
normally holds them.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from uuid import uuid4

from mindmemos_skill.errors import SkillConfigurationError
from mindmemos_skill.llm import LLMCallSink, llm_run_context
from mindmemos_skill.llm.recording import write_llm_call
from mindmemos_skill.registry import ComponentRequirements, ComponentType, register
from mindmemos_skill.typing import Skill, SkillCandidate, Trace2SkillInput, Trace2SkillOutput

from ..collection import ScheduledTrajectoryCollector, TrajectoryCollectionResult, TrajectoryCollector
from .config import OfflineEvolutionConfig
from .models import OfflineEvolutionReport

# A trajectory id becomes a single JSON filename under the (already scope-
# isolated) trajectory/ directory; a skill name becomes the skill directory
# name. Both must be plain file-system basenames so an untrusted id cannot
# traverse out of the source layout.
_SAFE_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Vendored pipeline imports are resolved on first use so a missing impl/ fails
# with a clear message instead of an import error at package load time.
_IMPL_IMPORT_ERROR: ImportError | None = None


def _vendor_pipeline():
    """Import and return (run_pipeline, PipelineConfig) from the vendored impl."""
    global _IMPL_IMPORT_ERROR
    try:
        from .impl.pipeline.runner import run_pipeline
        from .impl.pipeline.types import PipelineConfig

        return run_pipeline, PipelineConfig
    except ImportError as exc:  # pragma: no cover - exercised when impl/ absent
        _IMPL_IMPORT_ERROR = exc
        raise RuntimeError(
            "offline_evolution vendored pipeline (impl/) is not present. "
            "Copy skill-evolution's src/ into offline_evolution/impl/ before running."
        ) from exc


def _resolve_vendor_config_hash(config: OfflineEvolutionConfig) -> str:
    """Hash the affect-result vendor config together with the host config.

    The experiment-level ``config_hash`` hashes the adapter's CLI arguments
    only; the vendored YAML (parallelism, cluster thresholds, prompt budget,
    llm temperature/tokens/retries, vectorizer) is resolved independently of
    the CLI. Hash the secret-free resolved vendor snapshot *and* the host
    config so two runs that produce a different Skill cannot share a hash.
    """
    from .impl.config import resolved_config_snapshot

    payload = {
        "host": config.model_dump(mode="json"),
        "vendor": resolved_config_snapshot(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _normalize_skill_text(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _safe_basename(value: str, *, label: str) -> str:
    if _SAFE_BASENAME_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(
            f"{label} {value!r} is not a valid basename. Only alphanumeric, dot, underscore, and hyphen are allowed."
        )
    return value


def _contained_path(root: str, *parts: str) -> str:
    """Join path components and reject anything resolving outside ``root``."""
    root_path = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_path, *parts))
    try:
        contained = os.path.commonpath((root_path, candidate)) == root_path
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"resolved path {candidate!r} escapes root {root_path!r}")
    return candidate


class _HostLLMRecorder:
    """Buffer vendored LLM calls from a worker thread for later async flush.

    The vendored pipeline runs synchronously on a worker thread and cannot
    ``await`` the host's async ``write_llm_call``. This recorder collects a
    transport-neutral summary per terminal call on that thread; the wrapper
    awaits the batch inside the event loop *after* the worker returns, still
    under the ``llm_run_context`` so every call carries the run id.
    """

    def __init__(self) -> None:
        self._batch: list[Any] = []

    def record(self, call: Any) -> None:
        # Duck-typed on the vendored RecordedCall fields; no impl import here.
        self._batch.append(call)

    async def flush(self, sink: LLMCallSink | None) -> None:
        """Persist buffered calls through the host sink (async, must be awaited)."""
        if sink is None or not self._batch:
            return
        batch, self._batch = self._batch, []
        for call in batch:
            await _write_recorded_call(sink, call)


async def _write_recorded_call(sink: LLMCallSink, call: Any) -> None:
    """Translate one vendored RecordedCall into a host audit record.

    The host's ``write_llm_call`` reads the run id from a contextvar, so it
    is only meaningful when called inside ``llm_run_context`` (the wrapper
    wraps ``optimize()`` in it when a run id is supplied). Without a run id
    the host drops the record, which is the intended graceful degradation:
    a bare ``optimize()`` with no run id simply records nothing.

    The transaction runs on a worker thread and is flushed later, so we
    reconstruct a plausible window from the call's own ``latency_ms`` rather
    than recording the flush time as the call time.
    """
    # The vendored call carries its wall latency; treated as the window the
    # record describes. A missing latency falls back to a zero window.
    latency_ms = max(0.0, float(getattr(call, "latency_ms", 0.0) or 0.0))
    finished = datetime.now(UTC)
    started = finished - timedelta(milliseconds=latency_ms)
    payload = getattr(call, "request", None) or {}
    response = getattr(call, "response", None)
    model = getattr(call, "model", None) or "unknown"
    status = getattr(call, "status", "succeeded")
    error = getattr(call, "error", None)
    input_tokens = getattr(call, "input_tokens", None)
    output_tokens = getattr(call, "output_tokens", None)
    total_tokens = getattr(call, "total_tokens", None)
    task = getattr(call, "task", None) or "offline_evolution"
    await write_llm_call(
        sink,
        call_type="chat",
        task=task,
        request=payload,
        response=response,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        status=status,
        error=error,
        started_at=started,
        finished_at=finished,
        latency_ms=latency_ms,
    )


def _bind_call_recorder(recorder: Any):
    """Lazily bind a host recorder onto the vendored backend context.

    The vendored ``impl/llm/backend.py`` exposes ``bind_call_recorder``; we
    import it here (first use) rather than at module load so the vendored
    package stays optional for the base install.
    """
    from .impl.llm.backend import bind_call_recorder

    return bind_call_recorder(recorder)


def _build_openai_client_from_host(host_client: Any) -> tuple[Any | None, str | None]:
    """Build the vendored sync client from the host ``LLMClient`` config.

    The host's LLMClient wraps a litellm Router (._router attribute). Vendored
    code expects the synchronous OpenAI SDK interface, so the async host object
    cannot be injected directly. Return both the derived client and configured
    model identity. ``(None, None)`` preserves the vendored env fallback for
    lightweight fake pipelines and standalone diagnostics.
    """
    router = getattr(host_client, "_router", None)
    model_list = getattr(router, "model_list", None)
    if not model_list:
        return None, None

    first_model = router.model_list[0]
    litellm_params = first_model.get("litellm_params", {})
    api_key = litellm_params.get("api_key")
    api_base = litellm_params.get("api_base") or litellm_params.get("base_url")
    timeout = litellm_params.get("timeout", 600)
    max_retries = litellm_params.get("max_retries", litellm_params.get("num_retries", 2))
    model_name = litellm_params.get("model") or getattr(host_client, "_default_model", None)

    if not api_key:
        return None, str(model_name) if model_name else None

    try:
        from openai import OpenAI

    except ImportError:
        return None, str(model_name) if model_name else None

    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=timeout,
        max_retries=max_retries,
    )
    return client, str(model_name) if model_name else None


@contextlib.contextmanager
def _bind_chat_client(client: Any):
    """Inject a chat client into the vendored backend context.

    The vendored ``impl/llm/react/backend/client.py`` exposes ``_INJECTED_CLIENT``
    contextvar; we import it here rather than at module load so the vendored
    package stays optional.
    """
    from .impl.llm.react.backend.client import _INJECTED_CLIENT

    token = _INJECTED_CLIENT.set(client)
    try:
        yield
    finally:
        _INJECTED_CLIENT.reset(token)


@contextlib.contextmanager
def _bind_model_name(model_name: str):
    """Inject a model identity into the vendored backend context."""
    from .impl.llm.react.backend.config import _INJECTED_MODEL

    token = _INJECTED_MODEL.set(model_name)
    try:
        yield
    finally:
        _INJECTED_MODEL.reset(token)


def _inject_credentials() -> None:
    """Map host credentials onto the vendored pipeline's expected env names.

    Chat uses the injected synchronous client, while the vendored embedding
    backend still reads env configuration. Do not write an ``.env`` marker:
    installed package directories may be read-only and vendored config already
    treats a missing file as valid.
    """
    mapping = {
        "OPENAI_API_KEY": ("LLM_API_KEY", "EMBEDDING_API_KEY"),
        "OPENAI_BASE_URL": ("LLM_BASE_URL", "EMBEDDING_BASE_URL"),
        "OPENAI_ENDPOINT": ("LLM_BASE_URL", "EMBEDDING_BASE_URL"),
    }
    for host_name, vendored_names in mapping.items():
        value = os.getenv(host_name)
        if not value:
            continue
        for vendored_name in vendored_names:
            if not os.getenv(vendored_name):
                os.environ[vendored_name] = value


@register(
    type=ComponentType.ALGO,
    name="offline_evolution",
    config_model=OfflineEvolutionConfig,
    capabilities={"optimize"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"chat", "embedding"})),
)
class OfflineEvolution:
    algorithm_name: ClassVar[str] = "offline_evolution"
    algorithm_version: ClassVar[str] = "1"

    def __init__(
        self,
        *,
        config: OfflineEvolutionConfig,
        context: Any,
        collector: TrajectoryCollector | None = None,
    ) -> None:
        self._config = config
        self._context = context
        self._config_hash = getattr(context, "config_hash", None)

        try:
            self._chat_model = context.models["chat"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise SkillConfigurationError("offline_evolution requires the 'chat' model role in context.models") from exc

        self._collector = collector
        if self._collector is None and config.collection is not None:
            self._collector = ScheduledTrajectoryCollector(
                agents=context.agents,
                config=config.collection,
            )

    # -- format conversion -------------------------------------------------

    def _write_source_layout(self, *, base_skill: Skill, trajectories: list, source_dir: str) -> dict:
        """Write the pipeline's scope-first source layout and return trajectory ids."""
        scope_name = _safe_basename(self._config.scope_name, label="scope_name")
        scope_root = _contained_path(source_dir, scope_name)
        trajectory_dir = _contained_path(scope_root, "trajectory")
        skills_dir = _contained_path(scope_root, "skills")
        # The skill directory name must equal the SKILL.md frontmatter `name`
        # (the vendored editor gate enforces that match). base_skill.name is
        # derived from that same frontmatter by build_skill, so use it rather
        # than scope_name, which is just the pipeline's scope isolation label.
        skill_name = _safe_basename(base_skill.name or scope_name, label="skill name")
        skill_dir = _contained_path(skills_dir, skill_name)
        os.makedirs(trajectory_dir, exist_ok=True)
        os.makedirs(skill_dir, exist_ok=True)

        # Skill snapshot: the pipeline expects <scope>/skills/<name>/SKILL.md.
        skill_path = _contained_path(skill_dir, "SKILL.md")
        with open(skill_path, "w", encoding="utf-8") as handle:
            handle.write(_normalize_skill_text(base_skill.content))

        selected: list[tuple[str, Any]] = []
        duplicate_ids: list[str] = []
        seen: set[str] = set()
        for trajectory in sorted(trajectories, key=lambda item: item.trajectory_id):
            tid = _safe_basename(trajectory.trajectory_id, label="trajectory_id")
            if tid in seen:
                if self._config.dedupe_trajectories:
                    duplicate_ids.append(tid)
                    continue
                raise FileExistsError(f"duplicate trajectory_id {tid!r} would overwrite an existing file")
            seen.add(tid)
            selected.append((tid, trajectory))

        skipped_ids = [tid for tid, _ in selected[self._config.max_trajectories :]]
        selected = selected[: self._config.max_trajectories]
        if len(selected) < self._config.min_trajectories:
            raise ValueError(
                "offline_evolution requires at least "
                f"{self._config.min_trajectories} usable trajectories; got {len(selected)} "
                f"after removing {len(duplicate_ids)} duplicates and skipping {len(skipped_ids)} over the maximum"
            )

        used_ids: list[str] = []
        for tid, trajectory in selected:
            trajectory_path = _contained_path(trajectory_dir, f"{tid}.json")
            if os.path.exists(trajectory_path):
                raise FileExistsError(
                    f"trajectory file {trajectory_path!r} already exists. "
                    "Duplicate trajectory_id or file collision detected."
                )

            payload = trajectory.model_dump(
                mode="json",
                # Dropping agents keeps any credentials out of the vendored
                # workspace (and thus out of the extraction prompt). The
                # pipeline only needs events/task/execution metadata, not the
                # agent profile or its injected skill snapshots.
                exclude={"agent", "injected_skills"},
                exclude_none=True,
            )
            with open(trajectory_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            used_ids.append(tid)

        return {
            "used_trajectory_ids": used_ids,
            "duplicate_trajectory_ids": duplicate_ids,
            "skipped_trajectory_ids": skipped_ids,
            "skill_path": skill_path,
        }

    # -- pipeline invocation ----------------------------------------------

    def _run_sync(
        self,
        *,
        source_dir: str,
        run_root: str,
        model_name: str | None,
        skill_name: str | None = None,
        recorder: _HostLLMRecorder | None = None,
    ) -> "Any":
        """Run the synchronous pipeline on the calling (worker) thread."""
        _inject_credentials()

        sync_client, host_model_name = _build_openai_client_from_host(self._chat_model)
        if sync_client is not None:
            cleanup_client = _bind_chat_client(sync_client)
        else:
            cleanup_client = contextlib.nullcontext()

        resolved_model_name = host_model_name or model_name
        if resolved_model_name:
            bare_model_name = resolved_model_name.split("/", 1)[-1]
            cleanup_model = _bind_model_name(bare_model_name)
        else:
            cleanup_model = contextlib.nullcontext()

        # Inject call recorder into vendored context
        if recorder:
            cleanup_recorder = _bind_call_recorder(recorder)
        else:
            cleanup_recorder = contextlib.nullcontext()

        run_pipeline, PipelineConfig = _vendor_pipeline()
        config = PipelineConfig(
            source_dir=source_dir,
            run_root_dir=run_root,
            backend=self._config.backend,
            skill_md_only=self._config.skill_md_only,
        )
        try:
            with cleanup_client, cleanup_model, cleanup_recorder:
                result = run_pipeline(config)
        finally:
            close_client = getattr(sync_client, "close", None)
            if callable(close_client):
                close_client()

        final_skills_dir = result.paths.final_skills_dir
        # The vendored finalize stage exports every scope's evolved skill under
        # final_skills/<scope_id>/<skill_name>/SKILL.md (see export_final_skills).
        # The scope label is the host's configured scope_name; omit it and the
        # path points at a nonexistent file.
        skill_dir = os.path.join(
            final_skills_dir,
            self._config.scope_name,
            skill_name or self._config.scope_name,
        )
        skill_path = os.path.join(skill_dir, "SKILL.md")
        return result, skill_path

    # -- public contract ---------------------------------------------------

    async def optimize(self, request: Trace2SkillInput) -> Trace2SkillOutput[OfflineEvolutionReport]:
        """Run one optimization with all nested LLM calls bound to a run id.

        Tasks-only and hybrid requests collect first (via the injected or
        auto-built collector), merge the newly collected trajectories with any
        offline ones, and return only the newly collected trajectories in the
        output. A missing run id is generated so the collected/recorded calls
        still carry a stable id (aligned with ``trajectory_evidence_patch``).
        """
        run_id = request.run_id or f"offline-evolution-{uuid4().hex}"
        normalized_request = request if request.run_id is not None else request.model_copy(update={"run_id": run_id})
        with llm_run_context(run_id):
            return await self._optimize(normalized_request)

    async def _collect(self, request: Trace2SkillInput) -> TrajectoryCollectionResult:
        if self._collector is None:
            raise SkillConfigurationError("offline_evolution task collection requires the algorithm collection config")
        assert request.run_id is not None
        return await self._collector.collect(
            run_id=request.run_id,
            base_skill=request.base_skill,
            tasks=request.tasks,
        )

    async def _optimize(self, request: Trace2SkillInput) -> Trace2SkillOutput[OfflineEvolutionReport]:
        if not request.trajectories and not request.tasks:
            raise ValueError("offline_evolution requires at least one trajectory source")

        collection = await self._collect(request) if request.tasks else None
        collected_trajectories = collection.trajectories if collection is not None else []
        trajectories = [*collected_trajectories, *request.trajectories]
        vendor_config_hash = _resolve_vendor_config_hash(self._config)

        # Create a recorder to buffer vendored LLM calls for later flush
        recorder = _HostLLMRecorder()
        sink = getattr(self._context, "llm_call_sink", None)

        with llm_run_context(request.run_id):
            # The vendored pipeline is synchronous; push it to a worker thread.
            # Context-local client/model bindings keep concurrent runs isolated.
            loop = asyncio.get_running_loop()

            def convert_and_run() -> tuple[str | None, str | None, Any, dict]:
                with tempfile.TemporaryDirectory() as tmp:
                    # A configured run_root keeps the vendor pipeline's artifacts
                    # (extractions, review, final_skills, report) on disk under the
                    # experiment output_dir. Without one we fall back to a throwaway
                    # directory so a bare optimize() call still works.
                    base = str(self._config.run_root) if self._config.run_root else tmp
                    source_dir = os.path.join(base, "source")
                    run_root = os.path.join(base, "run")
                    os.makedirs(source_dir, exist_ok=True)
                    layout = self._write_source_layout(
                        base_skill=request.base_skill,
                        trajectories=trajectories,
                        source_dir=source_dir,
                    )
                    model_name = self._config.optimizer_model
                    result, skill_path = self._run_sync(
                        source_dir=source_dir,
                        run_root=run_root,
                        model_name=model_name,
                        skill_name=request.base_skill.name,
                        recorder=recorder,
                    )

                    # When run_root is None, we're in a throwaway TemporaryDirectory
                    # that will be deleted when this function returns. Read the skill
                    # content now before it vanishes.
                    skill_content = None
                    if self._config.run_root is None:
                        if not os.path.isfile(skill_path):
                            # Pipeline failed to produce the skill; will raise later
                            pass
                        else:
                            with open(skill_path, "r", encoding="utf-8") as handle:
                                skill_content = _normalize_skill_text(handle.read())

                    return skill_path if self._config.run_root else None, skill_content, result, layout

            try:
                skill_path_or_none, skill_content, result, layout = await loop.run_in_executor(None, convert_and_run)
            finally:
                # Preserve terminal failure records even when the pipeline raises.
                await recorder.flush(sink)

            # When run_root is set, the skill file is on disk and we read it now.
            # When run_root is None, we already captured the content before the
            # TemporaryDirectory was deleted.
            if skill_content is None:
                # Persistent run_root case: file should exist on disk
                if skill_path_or_none is None or not os.path.isfile(skill_path_or_none):
                    raise FileNotFoundError(
                        "offline_evolution did not produce an evolved skill at "
                        f"{skill_path_or_none!r}. Pipeline may have failed to export the "
                        f"scope; check {getattr(result, 'paths', None) and result.paths.final_skills_dir!r} "
                        "for the actual final_skills tree."
                    )
                with open(skill_path_or_none, "r", encoding="utf-8") as handle:
                    final_skill_text = _normalize_skill_text(handle.read())
                final_skill_path_for_report = skill_path_or_none
                report_uri = result.paths.final_report_md
            else:
                # Temporary directory case: content was captured before deletion
                final_skill_text = skill_content
                # Use memory:// URIs for the report since the files no longer exist
                final_skill_path_for_report = f"memory://offline_evolution/{self._config.scope_name}/SKILL.md"
                report_uri = "memory://offline_evolution/final_report.md"

            report = OfflineEvolutionReport(
                algorithm_name=self.algorithm_name,
                algorithm_version=self.algorithm_version,
                run_id=request.run_id,
                collection_run_id=collection.run_id if collection is not None else None,
                input_task_ids=[task.task_id for task in request.tasks],
                input_trajectory_ids=[t.trajectory_id for t in trajectories],
                requested_collection_rollout_ids=(collection.requested_rollout_ids if collection is not None else []),
                failed_collection_rollout_ids=(collection.failed_rollout_ids if collection is not None else []),
                used_trajectory_ids=layout["used_trajectory_ids"],
                duplicate_trajectory_ids=layout["duplicate_trajectory_ids"],
                vendor_counts=dict(result.counts or {}),
                report_uri=report_uri,
                config_hash=self._config_hash,
                vendor_config_hash=vendor_config_hash,
            )

            changed = False
            candidate = None
            if final_skill_text is not None:
                base_text = _normalize_skill_text(request.base_skill.content)
                if final_skill_text.strip() != base_text.strip():
                    changed = True
                    candidate = SkillCandidate(
                        blob={"SKILL.md": final_skill_text},
                        resources=dict(request.base_skill.resources),
                        runtime_type=request.base_skill.runtime_type,
                        runtime_schema_version=request.base_skill.runtime_schema_version,
                        runtime_metadata=dict(request.base_skill.runtime_metadata),
                        metadata={
                            "offline_evolution": {
                                "algorithm": self.algorithm_name,
                                "algorithm_version": self.algorithm_version,
                                "prompt_version": report.prompt_version,
                                "run_id": request.run_id,
                                "config_hash": self._config_hash,
                                "collection_run_id": collection.run_id if collection is not None else None,
                                "trajectory_ids": layout["used_trajectory_ids"],
                                "duplicate_trajectory_ids": layout["duplicate_trajectory_ids"],
                                "skipped_trajectory_ids": layout["skipped_trajectory_ids"],
                            }
                        },
                    )

            report.changed = changed
            report.reason = (
                "SKILL.md changed after offline evolution"
                if changed
                else "SKILL.md unchanged after offline evolution (candidate=None)"
            )
            report.final_skill_path = final_skill_path_for_report

            return Trace2SkillOutput(
                candidate=candidate,
                trajectories=collected_trajectories,
                report=report,
            )
