"""Offline cache, persistence, and failure contracts for the Analyze pipeline."""

import pytest
from mindmemos_skill.algos.analyze.skill_judge.config import SkillJudgeConfig
from mindmemos_skill.algos.analyze.typing import SkillAnalysisResult
from mindmemos_skill.api.typing import (
    EvaluationQueryRequest,
    SkillAnalyzeRequest,
    TrajectoryAnalyzeRequest,
)
from mindmemos_skill.errors import TrajectoryNotFoundError
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.infra.model_access.openai import record_provider_model
from mindmemos_skill.persistence import (
    SkillRecord,
    SkillVersionRecord,
    TrajectoryRecord,
    create_persistence,
    ensure_collection,
)
from mindmemos_skill.persistence.constants import SKILLS_TABLE
from mindmemos_skill.pipelines.builtin_pipelines.analyze.config import AnalyzePipelineConfig
from mindmemos_skill.pipelines.builtin_pipelines.analyze.pipeline import BuiltinAnalyzePipeline
from mindmemos_skill.skill import SkillContent, SkillSnapshot
from pydantic import BaseModel

PROJECT = "local-project"


class _FakeConfig(BaseModel):
    """Minimal analyzer configuration supplying stable version metadata."""

    algorithm_version: str = "1"
    prompt_version: str = "1"


class _FakeAnalyzer:
    """Deterministic offline analyzer that counts its own invocations."""

    def __init__(self, result: SkillAnalysisResult | None = None, error: Exception | None = None):
        self.algorithm_name = "fake_judge"
        self.config = _FakeConfig()
        self.result = result or SkillAnalysisResult(summary="analyzed", metrics={"score_mean": 4.0})
        self.error = error
        self.calls = 0

    async def analyze(self, request):
        """Return the fixed result or raise the configured failure."""

        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
async def store(tmp_path):
    """Provide isolated persistence that each test opens under a project scope."""

    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'analyze.db'}"))
    persistence = create_persistence(database)
    (await ensure_collection(persistence))
    try:
        yield persistence
    finally:
        (await database.close())


def pipeline(store, *, skill_judge=None, trajectory_judge=None) -> BuiltinAnalyzePipeline:
    """Build the pipeline with injected offline analyzers."""

    analyzer = BuiltinAnalyzePipeline(AnalyzePipelineConfig(), persistence=store)
    analyzer.skill_judge = skill_judge
    analyzer.trajectory_judge = trajectory_judge
    return analyzer


def skill_version(version_id="version-1", skill_id="skill-1") -> SkillVersionRecord:
    """Return one stored-version snapshot for analysis."""

    return SkillVersionRecord(
        project_id=PROJECT,
        version_id=version_id,
        skill_id=skill_id,
        **SkillSnapshot.from_content(
            SkillContent(
                content="# Skill\n\nDo the thing.",
                name="demo",
                description="A demo Skill.",
                standard_format=False,
            )
        ).model_dump(),
        version_label=f"display-{version_id}",
    )


def central_skill(record: SkillVersionRecord) -> SkillRecord:
    """Return the visible family row required before a version can be read."""

    return SkillRecord(
        project_id=record.project_id,
        skill_id=record.skill_id,
        name=record.name,
        alias=None,
        registration_source="/original/import/path",
        registration_message="Initial import",
        active_version_id=record.version_id,
    )


def trajectory(trajectory_id="trajectory-1", *, reward_score=1.0, **overrides) -> TrajectoryRecord:
    """Return one stored trajectory attempt."""

    values = {
        "project_id": PROJECT,
        "trajectory_id": trajectory_id,
        "trajectory_hash": "trace-hash",
        "task_id": "task-1",
        "rollout_id": "rollout-1",
        "task_instruction": "Solve the task",
        "env_ref": "livemath",
        "status": "succeeded",
        "trajectory": [{"role": "assistant", "content": "42"}],
        "reward_score": reward_score,
    }
    values.update(overrides)
    return TrajectoryRecord(**values)


async def add_version(scoped, version: SkillVersionRecord) -> None:
    """Persist a version and its visible family within an active scope."""

    await scoped.skills.add(version)
    await scoped.database.insert(SKILLS_TABLE, [central_skill(version).model_dump()])


async def test_completed_result_is_reused_without_reinvoking_judge(store):
    """A completed cache hit returns the stored row without a second model call."""

    analyzer = _FakeAnalyzer()
    analyze = pipeline(store, skill_judge=analyzer)
    version = skill_version()
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await add_version(scoped, version)
        request = SkillAnalyzeRequest(project_id=PROJECT, skill_version_id=version.version_id)
        first = await analyze.analyze_skill(request)
        second = await analyze.analyze_skill(request)
    assert analyzer.calls == 1
    assert first.status == "completed" and second.evaluation_id == first.evaluation_id


async def test_platform_judge_runs_again_and_preserves_prior_model_provenance(store):
    """A pool change produces a new evaluation for the same exact Skill version."""

    class PlatformAnalyzer(_FakeAnalyzer):
        async def analyze(self, request):
            record_provider_model(f"actual-model-{self.calls + 1}")
            return await super().analyze(request)

    analyzer = PlatformAnalyzer()
    analyzer.config = SkillJudgeConfig(judge={"model": "platform-chat", "transport": "platform_gateway"})
    analyze = pipeline(store, skill_judge=analyzer)
    version = skill_version()
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await add_version(scoped, version)
        request = SkillAnalyzeRequest(project_id=PROJECT, skill_version_id=version.version_id)
        first = await analyze.analyze_skill(request)
        second = await analyze.analyze_skill(request)
        rows = await scoped.evaluations.list_for_versions([version.version_id])

    assert analyzer.calls == 2
    assert first.evaluation_id != second.evaluation_id
    assert {row.metadata["provider_models"][0] for row in rows} == {"actual-model-1", "actual-model-2"}


async def test_failed_result_is_persisted_and_retryable(store):
    """A failing analyzer stores a failed row and permits a later retry."""

    analyzer = _FakeAnalyzer(error=RuntimeError("judge exploded"))
    analyze = pipeline(store, skill_judge=analyzer)
    version = skill_version()
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await add_version(scoped, version)
        request = SkillAnalyzeRequest(project_id=PROJECT, skill_version_id=version.version_id)
        with pytest.raises(RuntimeError):
            await analyze.analyze_skill(request)
        failed = await scoped.evaluations.list_for_versions([version.version_id], completed_only=False)
        assert [row.status for row in failed] == ["failed"]
        assert "judge exploded" in (failed[0].error or "")
        analyzer.error = None
        retried = await analyze.analyze_skill(request)
    assert retried.status == "completed"
    assert analyzer.calls == 2


async def test_request_id_does_not_change_cache_identity(store):
    """Only result-affecting inputs participate in the reusable key."""

    analyzer = _FakeAnalyzer()
    analyze = pipeline(store, skill_judge=analyzer)
    version = skill_version()
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await add_version(scoped, version)
        first = await analyze.analyze_skill(
            SkillAnalyzeRequest(project_id=PROJECT, skill_version_id=version.version_id, request_id="run-a")
        )
        second = await analyze.analyze_skill(
            SkillAnalyzeRequest(project_id=PROJECT, skill_version_id=version.version_id, request_id="run-b")
        )
    assert analyzer.calls == 1
    assert first.evaluation_key == second.evaluation_key


async def test_force_refresh_bypasses_completed_cache(store):
    """force_refresh re-runs the analyzer even when a completed row exists."""

    analyzer = _FakeAnalyzer()
    analyze = pipeline(store, skill_judge=analyzer)
    version = skill_version()
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await add_version(scoped, version)
        request = SkillAnalyzeRequest(project_id=PROJECT, skill_version_id=version.version_id)
        first = await analyze.analyze_skill(request)
        refreshed = await analyze.analyze_skill(request.model_copy(update={"force_refresh": True}))
        rows = await scoped.evaluations.list_for_versions([version.version_id])
    assert analyzer.calls == 2
    assert refreshed.evaluation_id == first.evaluation_id
    assert len(rows) == 1


async def test_missing_judges_raise_not_implemented(store):
    """Unconfigured analyzers surface an explicit capability error."""

    analyze = pipeline(store)
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        version = skill_version()
        await add_version(scoped, version)
        with pytest.raises(NotImplementedError):
            await analyze.analyze_skill(SkillAnalyzeRequest(project_id=PROJECT, skill_version_id=version.version_id))
        with pytest.raises(NotImplementedError):
            await analyze.analyze_trajectory(TrajectoryAnalyzeRequest(project_id=PROJECT, trajectory_id="trajectory-1"))


async def test_missing_trajectory_raises_not_found(store):
    """Unknown trajectories fail before any analyzer is invoked."""

    analyzer = _FakeAnalyzer()
    analyze = pipeline(store, trajectory_judge=analyzer)
    with store.session(Scope(fields={"project_id": PROJECT})):
        with pytest.raises(TrajectoryNotFoundError):
            await analyze.analyze_trajectory(TrajectoryAnalyzeRequest(project_id=PROJECT, trajectory_id="missing"))
    assert analyzer.calls == 0


async def test_non_finite_metrics_are_not_persisted(store):
    """NaN and infinity are stored as absent scores instead of invalid numbers."""

    result = SkillAnalysisResult(
        summary="suspect",
        metrics={"score_mean": float("nan"), "skill_usage_rate": float("inf")},
    )
    analyze = pipeline(store, skill_judge=_FakeAnalyzer(result=result))
    version = skill_version()
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await add_version(scoped, version)
        stored = await analyze.analyze_skill(
            SkillAnalyzeRequest(project_id=PROJECT, skill_version_id=version.version_id)
        )
    assert stored.score_mean is None
    assert stored.skill_usage_rate is None


async def test_trajectory_override_changes_identity(store):
    """A different verifier override produces a distinct trajectory result."""

    analyzer = _FakeAnalyzer(result=SkillAnalysisResult(summary="scored", metrics={"success": True}))
    analyze = pipeline(store, trajectory_judge=analyzer)
    record = trajectory()
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await scoped.trajectories.add(record)
        base = TrajectoryAnalyzeRequest(project_id=PROJECT, trajectory_id=record.trajectory_id)
        first = await analyze.analyze_trajectory(base)
        second = await analyze.analyze_trajectory(base.model_copy(update={"verifier_score": 0.0}))
    assert analyzer.calls == 2
    assert first.evaluation_key != second.evaluation_key
    assert first.judge_type == "trajectory"


async def test_query_orders_versions_and_filters_judge_types(store):
    """Queries return completed rows in requested version order."""

    analyzer = _FakeAnalyzer()
    analyze = pipeline(store, skill_judge=analyzer)
    first_version = skill_version("v1", "skill-1")
    second_version = skill_version("v2", "skill-2")
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await add_version(scoped, first_version)
        await add_version(scoped, second_version)
        await analyze.analyze_skill(SkillAnalyzeRequest(project_id=PROJECT, skill_version_id="v1"))
        await analyze.analyze_skill(SkillAnalyzeRequest(project_id=PROJECT, skill_version_id="v2"))
        result = await analyze.query(EvaluationQueryRequest(project_id=PROJECT, skill_version_ids=["v2", "v1"]))
        excluded = await analyze.query(
            EvaluationQueryRequest(project_id=PROJECT, skill_version_ids=["v1", "v2"], judge_types=["trajectory"])
        )
    assert [item.skill_version_id for item in result.items] == ["v2", "v1"]
    assert all(item.status == "completed" for item in result.items)
    assert excluded.items == []
