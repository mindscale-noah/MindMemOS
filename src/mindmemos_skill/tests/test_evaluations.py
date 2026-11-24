"""Independent evaluation storage contracts on a real isolated database."""

import pytest
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence import EvaluationRecord, create_persistence, ensure_collection

PROJECT = "project-a"


@pytest.fixture
async def store(tmp_path):
    """Provide isolated persistence whose project scope is chosen per test."""

    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'evaluations.db'}"))
    persistence = create_persistence(database)
    (await ensure_collection(persistence))
    try:
        yield persistence
    finally:
        (await database.close())


def evaluation(project_id=PROJECT, key="key-a", **overrides):
    """Return one minimal completed evaluation record."""

    values = {
        "evaluation_id": f"eval_{project_id}_{key}",
        "project_id": project_id,
        "evaluation_key": key,
        "judge_type": "static_skill",
        "skill_id": "skill-1",
        "skill_version_id": "version-1",
        "algorithm": "skill_judge",
        "algorithm_version": "1",
        "prompt_version": "1",
        "status": "completed",
        "summary": "usable",
        "score_mean": 4.0,
        "scores": {"safety": 4},
    }
    values.update(overrides)
    return EvaluationRecord(**values)


async def test_evaluation_reads_are_project_scoped(store):
    """The same evaluation key never leaks across projects."""

    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await scoped.evaluations.save(evaluation(key="shared"))
    with store.session(Scope(fields={"project_id": "project-b"})) as scoped:
        assert await scoped.evaluations.get("shared") is None
        await scoped.evaluations.save(evaluation("project-b", "shared"))
        assert (await scoped.evaluations.get("shared")).project_id == "project-b"
    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        assert (await scoped.evaluations.get("shared")).project_id == PROJECT


async def test_upsert_is_idempotent_and_preserves_identity(store):
    """Repeated saves update one logical row without rewriting its identity."""

    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        first = await scoped.evaluations.save(evaluation(key="stable"))
        updated = await scoped.evaluations.save(
            evaluation(key="stable", evaluation_id="eval-replacement", summary="refreshed", score_mean=5.0)
        )
        rows = await scoped.database.select("skill_evaluations_v1", filters={"evaluation_key": "stable"})
    assert len(rows) == 1
    assert updated.evaluation_id == first.evaluation_id
    assert updated.created_at == first.created_at
    assert updated.summary == "refreshed"
    assert updated.score_mean == 5.0


async def test_completed_only_queries_skip_failed_records(store):
    """Failed attempts stay readable but are not reusable cache hits."""

    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await scoped.evaluations.save(evaluation(key="done"))
        await scoped.evaluations.save(evaluation(key="broken", status="failed", error="judge exploded"))
        assert (await scoped.evaluations.get("broken")) is not None
        assert await scoped.evaluations.get("broken", completed_only=True) is None
        assert await scoped.evaluations.get_completed("broken") is None
        assert (await scoped.evaluations.get_completed("done")) is not None


async def test_list_for_versions_filters_versions_and_judge_types(store):
    """Batch reads keep version attribution and apply judge-type filters."""

    with store.session(Scope(fields={"project_id": PROJECT})) as scoped:
        await scoped.evaluations.save(evaluation(key="v1-static", skill_version_id="v1"))
        await scoped.evaluations.save(
            evaluation(key="v1-traj", skill_version_id="v1", judge_type="trajectory", trajectory_id="t1")
        )
        await scoped.evaluations.save(evaluation(key="v2-static", skill_version_id="v2"))
        await scoped.evaluations.save(evaluation(key="v3-failed", skill_version_id="v3", status="failed"))
        rows = await scoped.evaluations.list_for_versions(["v1", "v3"])
        assert {row.evaluation_key for row in rows} == {"v1-static", "v1-traj"}
        only_static = await scoped.evaluations.list_for_versions(["v1", "v2", "v3"], judge_types=["static_skill"])
        assert {row.evaluation_key for row in only_static} == {"v1-static", "v2-static"}
        assert await scoped.evaluations.list_for_versions([]) == []
        assert await scoped.evaluations.list_for_versions(["missing"]) == []


def test_evaluation_record_rejects_non_finite_scores():
    """Stored scores must stay representable in JSON and SQL."""

    with pytest.raises(ValueError):
        evaluation(key="nan", score_mean=float("nan"))
    with pytest.raises(ValueError):
        evaluation(key="inf", skill_usage_rate=float("inf"))
