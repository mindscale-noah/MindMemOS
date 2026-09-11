"""Regression tests for the offline-evolution host wrapper.

The vendored pipeline exports final skills as
``final_skills/<scope_id>/<skill_name>/SKILL.md`` (``export_final_skills``).
The host wrapper must read from that layout: a path that drops the scope
layer points at a file that does not exist, and the wrapper must not treat
"file missing" as "skill unchanged".
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos_skill.algos.trace2skill import (
    OfflineEvolution,
    OfflineEvolutionConfig,
    TaskCollectionConfig,
    TrajectoryCollectionResult,
)
from mindmemos_skill.algos.trace2skill.offline_evolution import algorithm as algo_module
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import (
    PipelinePaths,
    PipelineResult,
)
from mindmemos_skill.typing import (
    ExecutionInfo,
    Rollout,
    Skill,
    Task,
    Trace2SkillInput,
    Trajectory,
    TrajectoryStatus,
    compute_skill_content_hash,
)

_BASE = """\
---
name: livemath
description: Solve MCQs.
---
# Live Mathematical MCQ Heuristics
## Option Comparison
- Compare options.
## Final Answer
"""

_EVOLVED = _BASE + "\n## Extra\n- Check for a meta-option.\n"


def test_config_rejects_resource_evolution():
    with pytest.raises(ValueError, match="supports only SKILL.md evolution"):
        OfflineEvolutionConfig(skill_md_only=False)


def _skill(name: str = "livemath", content: str = _BASE) -> Skill:
    blob = {"SKILL.md": content}
    return Skill(
        skill_id=f"{name}-skill",
        version_id="1.0.0",
        version_label="1.0.0",
        content_hash=compute_skill_content_hash(blob),
        name=name,
        blob=blob,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
    )


def _trajectory(skill: Skill, trajectory_id: str) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        task=Task(task_id=f"{trajectory_id}-task", instruction="Solve"),
        rollout=Rollout(rollout_id=f"{trajectory_id}-rollout"),
        injected_skills=[skill],
        events=[{"role": "user", "content": "Solve"}],
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=datetime(2026, 8, 11, tzinfo=UTC),
            finished_at=datetime(2026, 8, 11, 0, 1, tzinfo=UTC),
        ),
    )


def _mock_chat_client():
    """Return a minimal mock that satisfies the chat model requirement."""
    return "mock_client"


class _FakeCollector:
    """Records collect() calls and returns a canned collection result."""

    def __init__(self, result: TrajectoryCollectionResult) -> None:
        self.result = result
        self.calls: list[tuple[str, Skill, list[Task]]] = []

    async def collect(
        self,
        *,
        run_id: str,
        base_skill: Skill,
        tasks: list[Task],
    ) -> TrajectoryCollectionResult:
        self.calls.append((run_id, base_skill, tasks))
        return self.result


def _fake_vendor_pipeline(*, evolved: str, skill_name: str, scope_name: str = "default"):
    """Return a (run_pipeline, PipelineConfig) pair that writes the evolved
    skill in the real export layout: final_skills/<scope>/<skill>/SKILL.md."""

    def run_pipeline(_config):
        # Real pipeline creates a timestamped subdir under run_root_dir.
        # For testing, use a fixed "x" subdir directly under run_root_dir.
        run_dir = os.path.join(_config.run_root_dir, "x")
        final_skills_dir = os.path.join(run_dir, "final_skills")
        output = os.path.join(final_skills_dir, scope_name, skill_name, "SKILL.md")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(evolved)
        paths = PipelinePaths(
            run_dir=run_dir,
            workspace_dir=os.path.join(run_dir, "workspace"),
            skills_baseline_dir=os.path.join(run_dir, "baseline"),
            extractions_dir=os.path.join(run_dir, "workspace", "extractions"),
            review_dir=os.path.join(run_dir, "review"),
            final_skills_dir=final_skills_dir,
            final_report_md=os.path.join(run_dir, "final_report.md"),
            state_json=os.path.join(run_dir, "state.json"),
        )
        return PipelineResult(
            paths=paths,
            input_trajectory=[],
            counts={"trajectories": 1, "skills_updated": 1},
            meta={"status": "complete"},
        )

    run_pipeline.__name__ = "fake_run_pipeline"
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelineConfig

    return run_pipeline, PipelineConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_name", ["default", "custom-scope"])
async def test_reads_evolved_skill_from_scoped_export_path(tmp_path, monkeypatch, scope_name):
    """A non-trivial evolution must surface as a candidate, even when a
    non-default scope name is configured. Previously the wrapper dropped the
    scope layer and read a nonexistent path, so it silently returned
    candidate=None."""
    base_skill = _skill()
    trajectory = _trajectory(base_skill, "traj-1")

    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name=scope_name
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    config = OfflineEvolutionConfig(
        scope_name=scope_name,
        optimizer_model="gpt-5.4-mini",
        run_root=tmp_path,
    )
    algorithm = OfflineEvolution(
        config=config,
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
    )

    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=[trajectory], run_id="run-1")
    )

    assert output.changed is True
    assert output.candidate is not None
    assert output.candidate.blob["SKILL.md"] == _EVOLVED
    assert output.report.final_skill_path is not None
    assert os.path.isfile(output.report.final_skill_path)


@pytest.mark.asyncio
async def test_evolution_preserves_auxiliary_resources(tmp_path, monkeypatch):
    base = _skill()
    resources = {"references/guide.md": "Keep this guide.", "scripts/check.py": "print('check')\n"}
    base = base.model_copy(
        update={
            "resources": resources,
        }
    )
    run_pipeline, pipeline_config = _fake_vendor_pipeline(evolved=_EVOLVED, skill_name=base.name)
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, pipeline_config))
    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(run_root=tmp_path),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
    )
    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base, trajectories=[_trajectory(base, "traj-1")], run_id="run-1")
    )
    assert output.candidate is not None
    assert output.candidate.blob == {"SKILL.md": _EVOLVED}
    assert output.candidate.resources == resources
    assert base.blob == {"SKILL.md": _BASE}


@pytest.mark.asyncio
async def test_missing_final_skill_raises_instead_of_silent_unchanged(tmp_path, monkeypatch):
    """If the evolved skill file cannot be found, the wrapper must fail
    loudly rather than report a false 'unchanged' result."""
    base_skill = _skill()
    trajectory = _trajectory(base_skill, "traj-1")

    # run_pipeline does not write any final skill at all.
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )

    def run_pipeline_missing_skill(_config):
        paths = PipelinePaths(
            run_dir=os.path.join(_config.run_root_dir, "run", "x"),
            workspace_dir=os.path.join(_config.run_root_dir, "run", "x", "workspace"),
            skills_baseline_dir=os.path.join(_config.run_root_dir, "run", "x", "baseline"),
            extractions_dir=os.path.join(_config.run_root_dir, "run", "x", "workspace", "extractions"),
            review_dir=os.path.join(_config.run_root_dir, "run", "x", "review"),
            final_skills_dir=os.path.join(_config.run_root_dir, "run", "x", "final_skills"),
            final_report_md=os.path.join(_config.run_root_dir, "run", "x", "final_report.md"),
            state_json=os.path.join(_config.run_root_dir, "run", "x", "state.json"),
        )
        return PipelineResult(
            paths=paths,
            input_trajectory=[],
            counts={"trajectories": 1, "skills_updated": 1},
            meta={"status": "complete"},
        )

    run_pipeline_missing_skill.__name__ = "fake_run_pipeline"
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline_missing_skill, PipelineConfig))

    config = OfflineEvolutionConfig(
        scope_name="default",
        optimizer_model="gpt-5.4-mini",
        run_root=tmp_path,
    )
    algorithm = OfflineEvolution(
        config=config,
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
    )

    with pytest.raises(FileNotFoundError):
        await algorithm.optimize(Trace2SkillInput(base_skill=base_skill, trajectories=[trajectory], run_id="run-1"))


@pytest.mark.asyncio
async def test_no_change_when_evolved_equals_base(tmp_path, monkeypatch):
    """When the pipeline genuinely produces identical content, candidate is
    None and changed is False (the legitimate no-op case)."""
    base_skill = _skill()
    trajectory = _trajectory(base_skill, "traj-1")

    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_BASE, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    config = OfflineEvolutionConfig(
        scope_name="default",
        optimizer_model="gpt-5.4-mini",
        run_root=tmp_path,
    )
    algorithm = OfflineEvolution(
        config=config,
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
    )

    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=[trajectory], run_id="run-1")
    )

    assert output.changed is False
    assert output.candidate is None


@pytest.mark.asyncio
async def test_traversal_names_are_rejected_before_any_write(tmp_path, monkeypatch):
    """Path-like scope_name / trajectory_id / skill name must be refused up
    front, not silently joined into the source layout."""
    from mindmemos_skill.algos.trace2skill import OfflineEvolutionConfig

    base_skill = _skill()
    trajectory = _trajectory(base_skill, "traj-1")

    with pytest.raises(ValueError):
        OfflineEvolutionConfig(scope_name="../../evil", run_root=tmp_path)


@pytest.mark.asyncio
async def test_traversal_trajectory_id_is_rejected(tmp_path, monkeypatch):
    """A trajectory_id that would escape the scope's trajectory/ dir is refused."""
    base_skill = _skill()
    evil = _trajectory(base_skill, "../../evil")
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    config = OfflineEvolutionConfig(scope_name="default", run_root=tmp_path)
    algorithm = OfflineEvolution(
        config=config, context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg")
    )
    with pytest.raises(ValueError):
        await algorithm.optimize(Trace2SkillInput(base_skill=base_skill, trajectories=[evil], run_id="run-1"))


@pytest.mark.asyncio
async def test_duplicate_trajectory_id_is_not_overwritten(tmp_path, monkeypatch):
    """A duplicate id that would clobber an earlier file must raise, even with
    dedupe_trajectories=False."""
    base_skill = _skill()
    first = _trajectory(base_skill, "dup")
    second = _trajectory(base_skill, "dup")
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    config = OfflineEvolutionConfig(scope_name="default", dedupe_trajectories=False, run_root=tmp_path)
    algorithm = OfflineEvolution(
        config=config, context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg")
    )
    with pytest.raises(FileExistsError):
        await algorithm.optimize(Trace2SkillInput(base_skill=base_skill, trajectories=[first, second], run_id="run-1"))


@pytest.mark.asyncio
async def test_trajectory_bounds_are_deterministic_and_reported(tmp_path, monkeypatch):
    base_skill = _skill().model_copy(
        update={
            "runtime_type": "python",
            "runtime_schema_version": 2,
            "runtime_metadata": {"entrypoint": "solver.py"},
        }
    )
    trajectories = [
        _trajectory(base_skill, "traj-c"),
        _trajectory(base_skill, "traj-a"),
        _trajectory(base_skill, "traj-b"),
        _trajectory(base_skill, "traj-a"),
    ]
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(
            scope_name="default",
            min_trajectories=2,
            max_trajectories=2,
            run_root=tmp_path,
        ),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
    )
    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=trajectories, run_id="run-1")
    )

    assert output.report.used_trajectory_ids == ["traj-a", "traj-b"]
    assert output.report.duplicate_trajectory_ids == ["traj-a"]
    assert output.candidate is not None
    assert output.candidate.metadata["offline_evolution"]["skipped_trajectory_ids"] == ["traj-c"]
    assert output.candidate.runtime_type == "python"
    assert output.candidate.runtime_schema_version == 2
    assert output.candidate.runtime_metadata == {"entrypoint": "solver.py"}


@pytest.mark.asyncio
async def test_minimum_trajectory_count_is_enforced(tmp_path, monkeypatch):
    base_skill = _skill()
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))
    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(
            scope_name="default",
            min_trajectories=2,
            max_trajectories=2,
            run_root=tmp_path,
        ),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
    )

    with pytest.raises(ValueError, match="requires at least 2 usable trajectories"):
        await algorithm.optimize(
            Trace2SkillInput(
                base_skill=base_skill,
                trajectories=[_trajectory(base_skill, "traj-1")],
                run_id="run-1",
            )
        )


@pytest.mark.asyncio
async def test_run_root_none_still_returns_evolved_skill(tmp_path, monkeypatch):
    """With run_root unset the pipeline lives in a TemporaryDirectory that is
    removed before optimize() returns; the skill content must be captured
    inside that window and the report must not advertise a deleted path."""
    base_skill = _skill()
    trajectory = _trajectory(base_skill, "traj-1")
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    config = OfflineEvolutionConfig(scope_name="default", optimizer_model="gpt-5.4-mini", run_root=None)
    algorithm = OfflineEvolution(
        config=config, context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg")
    )

    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=[trajectory], run_id="run-1")
    )

    assert output.changed is True
    assert output.candidate is not None
    assert output.candidate.blob["SKILL.md"] == _EVOLVED
    assert output.report.final_skill_path.startswith("memory://")
    assert output.report.report_uri.startswith("memory://")


@pytest.mark.asyncio
async def test_vendored_llm_calls_are_flushed_into_host_sink(tmp_path, monkeypatch):
    """Every vendored LLM call made by the pipeline must be buffered and then
    flushed through the host sink under llm_run_context, so the optimizer's
    calls carry the run id and land in the audit table."""
    from mindmemos_skill.persistence.models import LLMCallRecord

    class CapturingSink:
        def __init__(self) -> None:
            self.records: list[LLMCallRecord] = []

        async def write(self, record: LLMCallRecord) -> None:
            self.records.append(record)

    base_skill = _skill()
    trajectory = _trajectory(base_skill, "traj-1")

    # A fake pipeline that drives the real vendored invoke() once, so the
    # recorder hook fires and the call summary is captured.
    def run_pipeline_with_invoke(_config):
        import mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react as react_module
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm import invoke, use_backend
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.invocation import InvokeResult

        # Stub the react inner invoke so no credentials / network are needed;
        # the outer backend.py::invoke still runs and fires the recorder. Patch
        # the react module attribute because backend.py resolves it by name.
        def fake_react_invoke(system, user, **kw):
            return InvokeResult(text="ok", channel="react", input_tokens=5, output_tokens=3)

        monkeypatch.setattr(react_module, "invoke", fake_react_invoke)

        # The react backend validates its cwd before invoking; create it.
        cwd = os.path.join(_config.run_root_dir, "run")
        os.makedirs(cwd, exist_ok=True)

        with use_backend("react", default_cwd=cwd):
            invoke("system", "user", tools=None)
        # Return an unchanged skill so the wrapper still reads a final file.
        final_skills_dir = os.path.join(_config.run_root_dir, "run", "x", "final_skills")
        skill_dir = os.path.join(final_skills_dir, "default", "livemath")
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as handle:
            handle.write(_BASE)
        paths = PipelinePaths(
            run_dir=os.path.join(_config.run_root_dir, "run", "x"),
            workspace_dir=os.path.join(_config.run_root_dir, "run", "x", "workspace"),
            skills_baseline_dir=os.path.join(_config.run_root_dir, "run", "x", "baseline"),
            extractions_dir=os.path.join(_config.run_root_dir, "run", "x", "workspace", "extractions"),
            review_dir=os.path.join(_config.run_root_dir, "run", "x", "review"),
            final_skills_dir=final_skills_dir,
            final_report_md=os.path.join(_config.run_root_dir, "run", "x", "final_report.md"),
            state_json=os.path.join(_config.run_root_dir, "run", "x", "state.json"),
        )
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import (
            PipelineResult,
        )

        return PipelineResult(paths=paths, input_trajectory=[], counts={"trajectories": 1}, meta={"status": "complete"})

    run_pipeline_with_invoke.__name__ = "fake_run_pipeline"
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelineConfig

    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline_with_invoke, PipelineConfig))

    sink = CapturingSink()
    config = OfflineEvolutionConfig(scope_name="default", optimizer_model="gpt-5.4-mini", run_root=tmp_path)
    algorithm = OfflineEvolution(
        config=config,
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg", llm_call_sink=sink),
    )

    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=[trajectory], run_id="run-1")
    )

    assert output.changed is False  # identical content -> no candidate
    # At least one vendored LLM call must have been recorded and flushed.
    assert len(sink.records) >= 1
    record = sink.records[0]
    assert record.run_id == "run-1"
    assert record.model in ("gpt-5.4-mini", "unknown")  # resolved from vendored config when available
    assert record.status in ("succeeded", "failed")


@pytest.mark.asyncio
async def test_buffered_llm_calls_are_flushed_when_pipeline_fails(tmp_path, monkeypatch):
    from mindmemos_skill.persistence.models import LLMCallRecord

    class CapturingSink:
        def __init__(self) -> None:
            self.records: list[LLMCallRecord] = []

        async def write(self, record: LLMCallRecord) -> None:
            self.records.append(record)

    def failing_pipeline(_config):
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.backend import current_call_recorder

        recorder = current_call_recorder()
        assert recorder is not None
        recorder.record(
            SimpleNamespace(
                task="extract",
                model="gpt-5.4-mini",
                request={"user": "prompt"},
                response=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                status="failed",
                error="RuntimeError: pipeline failed",
                latency_ms=1.0,
            )
        )
        raise RuntimeError("pipeline failed")

    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelineConfig

    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (failing_pipeline, PipelineConfig))
    sink = CapturingSink()
    base_skill = _skill()
    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(scope_name="default", run_root=tmp_path),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, config_hash="cfg", llm_call_sink=sink),
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        await algorithm.optimize(
            Trace2SkillInput(
                base_skill=base_skill,
                trajectories=[_trajectory(base_skill, "traj-1")],
                run_id="run-1",
            )
        )

    assert len(sink.records) == 1
    assert sink.records[0].run_id == "run-1"
    assert sink.records[0].status == "failed"


@pytest.mark.asyncio
async def test_injected_client_is_consumed_by_vendor(tmp_path, monkeypatch):
    """The host's injected chat client must be consumed by vendored code rather
    than the vendor building one from env. This validates DI integration.

    The wrapper extracts the litellm Router config from the host's LLMClient
    and builds a sync OpenAI client that vendored code can actually use."""
    base_skill = _skill()
    trajectory = _trajectory(base_skill, "traj-1")

    # Track whether a sync OpenAI client was built from the router config.
    openai_clients_built = []
    injected_models = []

    # Mock the OpenAI constructor to track client creation.
    def mock_openai_constructor(**kwargs):
        # Vendored code calls client.chat.completions.create(), so we need
        # to stub that interface.
        class MockOpenAIClient:
            class Chat:
                class Completions:
                    @staticmethod
                    def create(**kw):
                        openai_clients_built.append(kwargs)

                        # Return a minimal OpenAI response shape
                        class MockResponse:
                            class Choice:
                                class Message:
                                    content = "ok"
                                    tool_calls = []

                                message = Message()
                                finish_reason = "stop"

                            choices = [Choice()]
                            model = "gpt-5.4-mini"
                            usage = None

                        return MockResponse()

                completions = Completions()

            chat = Chat()

        return MockOpenAIClient()

    # Patch OpenAI constructor at the module level before import.
    import sys
    import types

    mock_openai_module = types.ModuleType("openai")
    mock_openai_module.OpenAI = mock_openai_constructor
    monkeypatch.setitem(sys.modules, "openai", mock_openai_module)

    # Build a realistic mock of LLMClient with ._router attribute.
    class MockRouter:
        deployment_names = ["gpt-5.4-mini"]
        model_list = [
            {
                "litellm_params": {
                    "model": "openai/gpt-5.4-mini",
                    "api_key": "test-key",
                    "api_base": "https://api.test.com",
                    "timeout": 600,
                    "num_retries": 3,
                }
            }
        ]

    class MockLLMClient:
        def __init__(self):
            self._router = MockRouter()
            self._default_model = "openai/should-not-win"

    mock_client = MockLLMClient()

    # A fake pipeline that drives the real vendored invoke() once.
    def run_pipeline_with_invoke(_config):
        import os

        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm import invoke, use_backend
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.backend.client import (
            current_injected_client,
        )
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.backend.config import (
            current_injected_model,
        )
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelinePaths

        cwd = os.path.join(_config.run_root_dir, "run")
        os.makedirs(cwd, exist_ok=True)

        assert current_injected_client() is not None
        injected_models.append(current_injected_model())
        with use_backend("react", default_cwd=cwd):
            invoke("system", "user", tools=None)

        # Return an unchanged skill.
        run_dir = os.path.join(_config.run_root_dir, "x")
        final_skills_dir = os.path.join(run_dir, "final_skills")
        skill_dir = os.path.join(final_skills_dir, "default", "livemath")
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as handle:
            handle.write(_BASE)
        paths = PipelinePaths(
            run_dir=run_dir,
            workspace_dir=os.path.join(run_dir, "workspace"),
            skills_baseline_dir=os.path.join(run_dir, "baseline"),
            extractions_dir=os.path.join(run_dir, "workspace", "extractions"),
            review_dir=os.path.join(run_dir, "review"),
            final_skills_dir=final_skills_dir,
            final_report_md=os.path.join(run_dir, "final_report.md"),
            state_json=os.path.join(run_dir, "state.json"),
        )
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelineResult

        return PipelineResult(paths=paths, input_trajectory=[], counts={"trajectories": 1}, meta={"status": "complete"})

    run_pipeline_with_invoke.__name__ = "fake_run_pipeline"
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelineConfig

    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline_with_invoke, PipelineConfig))

    config = OfflineEvolutionConfig(scope_name="default", optimizer_model="gpt-5.4-mini", run_root=tmp_path)
    algorithm = OfflineEvolution(
        config=config,
        context=SimpleNamespace(models={"chat": mock_client}, agents={}, config_hash="cfg"),
    )

    monkeypatch.setenv("LLM_MODEL", "must-remain-unchanged")
    await algorithm.optimize(Trace2SkillInput(base_skill=base_skill, trajectories=[trajectory], run_id="run-1"))

    # Verify that an OpenAI client was built from the router's config.
    assert len(openai_clients_built) >= 1
    built_config = openai_clients_built[0]
    assert built_config["api_key"] == "test-key"
    assert built_config["base_url"] == "https://api.test.com"
    assert built_config["timeout"] == 600
    assert built_config["max_retries"] == 3
    assert injected_models == ["gpt-5.4-mini"]
    assert os.environ["LLM_MODEL"] == "must-remain-unchanged"


@pytest.mark.asyncio
async def test_missing_chat_model_raises_configuration_error(tmp_path):
    """When context.models does not provide 'chat', __init__ must raise
    SkillConfigurationError rather than silently accepting it."""
    from mindmemos_skill.errors import SkillConfigurationError

    config = OfflineEvolutionConfig(scope_name="default", run_root=tmp_path)
    with pytest.raises(SkillConfigurationError, match="requires the 'chat' model role"):
        OfflineEvolution(config=config, context=SimpleNamespace(models={}, config_hash="cfg"))


def test_env_missing_does_not_raise_or_create_marker(tmp_path, monkeypatch):
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl import config as vendor_config

    missing_env = tmp_path / ".env"
    monkeypatch.setattr(vendor_config, "_env_path", lambda: missing_env)
    monkeypatch.setattr(vendor_config, "_ENV_LOADED", False)

    vendor_config.load_project_env_once()

    assert vendor_config._ENV_LOADED is True
    assert not missing_env.exists()


def test_credentials_are_mapped_for_chat_and_embedding(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "host-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://host.example/v1")
    for name in ("LLM_API_KEY", "EMBEDDING_API_KEY", "LLM_BASE_URL", "EMBEDDING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    algo_module._inject_credentials()

    assert os.environ["LLM_API_KEY"] == "host-key"
    assert os.environ["EMBEDDING_API_KEY"] == "host-key"
    assert os.environ["LLM_BASE_URL"] == "https://host.example/v1"
    assert os.environ["EMBEDDING_BASE_URL"] == "https://host.example/v1"


@pytest.mark.asyncio
async def test_task_mode_collects_then_returns_new_trajectories(tmp_path, monkeypatch):
    """Tasks-only requests collect first (bound to the request run id), feed the
    collected trajectories to the pipeline, and return only those newly
    collected trajectories in the output."""
    base_skill = _skill()
    tasks = [
        Task(task_id="task-a", instruction="Do task A"),
        Task(task_id="task-b", instruction="Do task B"),
    ]
    collector = _FakeCollector(
        TrajectoryCollectionResult(
            run_id="run-1",
            requested_rollout_ids=["rollout-1", "rollout-2", "rollout-failed"],
            trajectories=[_trajectory(base_skill, "traj-2"), _trajectory(base_skill, "traj-1")],
            failed_rollout_ids=["rollout-failed"],
        )
    )
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    config = OfflineEvolutionConfig(scope_name="default", run_root=tmp_path)
    algorithm = OfflineEvolution(
        config=config,
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
        collector=collector,
    )

    output = await algorithm.optimize(Trace2SkillInput(base_skill=base_skill, tasks=tasks, run_id="run-1"))

    assert collector.calls == [("run-1", base_skill, tasks)]
    assert output.changed is True
    assert output.report.collection_run_id == "run-1"
    assert output.report.input_task_ids == ["task-a", "task-b"]
    assert output.report.requested_collection_rollout_ids == ["rollout-1", "rollout-2", "rollout-failed"]
    assert output.report.failed_collection_rollout_ids == ["rollout-failed"]
    assert output.report.input_trajectory_ids == ["traj-2", "traj-1"]
    assert output.report.used_trajectory_ids == ["traj-1", "traj-2"]  # sorted by id
    assert [t.trajectory_id for t in output.trajectories] == ["traj-2", "traj-1"]


@pytest.mark.asyncio
async def test_hybrid_mode_merges_offline_and_collected_trajectories(tmp_path, monkeypatch):
    """A request carrying both tasks and offline trajectories collects first,
    then merges collected + offline, returning only the collected batch."""
    base_skill = _skill()
    task = Task(task_id="task-new", instruction="Collect one more trace")
    collected = _trajectory(base_skill, "traj-2")
    offline = _trajectory(base_skill, "traj-1")
    collector = _FakeCollector(
        TrajectoryCollectionResult(run_id="run-1", requested_rollout_ids=["rollout-new"], trajectories=[collected])
    )
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(scope_name="default", run_root=tmp_path),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
        collector=collector,
    )

    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=[offline], tasks=[task], run_id="run-1")
    )

    assert output.report.input_trajectory_ids == ["traj-2", "traj-1"]
    assert output.report.used_trajectory_ids == ["traj-1", "traj-2"]
    assert [t.trajectory_id for t in output.trajectories] == ["traj-2"]


@pytest.mark.asyncio
async def test_hybrid_duplicate_ids_are_deduplicated_and_reported(tmp_path, monkeypatch):
    """A collected trajectory whose id collides with an offline one is collapsed
    (dedupe on) and reported, but still counted in the merged input list."""
    base_skill = _skill()
    task = Task(task_id="task-new", instruction="Collect one more trace")
    collected = _trajectory(base_skill, "dup")
    offline = _trajectory(base_skill, "dup")
    collector = _FakeCollector(
        TrajectoryCollectionResult(run_id="run-1", requested_rollout_ids=["rollout-new"], trajectories=[collected])
    )
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(scope_name="default", min_trajectories=1, run_root=tmp_path),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
        collector=collector,
    )

    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=[offline], tasks=[task], run_id="run-1")
    )

    assert output.report.duplicate_trajectory_ids == ["dup"]
    assert output.report.used_trajectory_ids == ["dup"]
    assert [t.trajectory_id for t in output.trajectories] == ["dup"]


@pytest.mark.asyncio
async def test_task_mode_without_collector_raises_configuration_error(tmp_path, monkeypatch):
    """Tasks without a collector (and no collection config) must fail loudly
    rather than silently produce an empty trajectory/ layout."""
    from mindmemos_skill.errors import SkillConfigurationError

    base_skill = _skill()
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(scope_name="default", run_root=tmp_path),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
    )

    with pytest.raises(SkillConfigurationError, match="requires the algorithm collection config"):
        await algorithm.optimize(
            Trace2SkillInput(
                base_skill=base_skill,
                tasks=[Task(task_id="task-a", instruction="Do task A")],
                run_id="run-1",
            )
        )


@pytest.mark.asyncio
async def test_collected_trajectories_respect_min_max_bounds(tmp_path, monkeypatch):
    """min/max truncation applies to the merged (collected + offline) set, and
    over-max ids are reported as skipped."""
    base_skill = _skill()
    task = Task(task_id="task-new", instruction="Collect one more trace")
    collected = _trajectory(base_skill, "traj-3")
    offline = [_trajectory(base_skill, "traj-2"), _trajectory(base_skill, "traj-1")]
    collector = _FakeCollector(
        TrajectoryCollectionResult(run_id="run-1", requested_rollout_ids=["rollout-new"], trajectories=[collected])
    )
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(scope_name="default", min_trajectories=2, max_trajectories=2, run_root=tmp_path),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
        collector=collector,
    )

    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=offline, tasks=[task], run_id="run-1")
    )

    assert output.report.used_trajectory_ids == ["traj-1", "traj-2"]
    assert output.candidate is not None
    assert output.candidate.metadata["offline_evolution"]["skipped_trajectory_ids"] == ["traj-3"]


@pytest.mark.asyncio
async def test_report_carries_vendor_config_hash(tmp_path, monkeypatch):
    """The report must hash the secret-free resolved vendor config (plus host
    config) so two runs that differ only by vendored YAML cannot share a hash."""
    base_skill = _skill()
    trajectory = _trajectory(base_skill, "traj-1")
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(scope_name="default", run_root=tmp_path),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={}, config_hash="cfg"),
    )

    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=[trajectory], run_id="run-1")
    )

    assert output.report.vendor_config_hash is not None
    assert len(output.report.vendor_config_hash) == 64


@pytest.mark.asyncio
async def test_auto_builds_collector_from_config_and_agents(tmp_path, monkeypatch):
    """When no collector is injected but the config carries a collection spec,
    the wrapper builds a ScheduledTrajectoryCollector from context.agents."""
    base_skill = _skill()
    trajectory = _trajectory(base_skill, "traj-1")
    run_pipeline, PipelineConfig = _fake_vendor_pipeline(
        evolved=_EVOLVED, skill_name=base_skill.name, scope_name="default"
    )
    monkeypatch.setattr(algo_module, "_vendor_pipeline", lambda: (run_pipeline, PipelineConfig))

    algorithm = OfflineEvolution(
        config=OfflineEvolutionConfig(
            scope_name="default",
            run_root=tmp_path,
            collection=TaskCollectionConfig(agent_ref="react", env_ref="livemath"),
        ),
        context=SimpleNamespace(models={"chat": _mock_chat_client()}, agents={"react": object()}, config_hash="cfg"),
    )

    assert algorithm._collector is not None

    output = await algorithm.optimize(
        Trace2SkillInput(base_skill=base_skill, trajectories=[trajectory], run_id="run-1")
    )
    assert output.changed is True
