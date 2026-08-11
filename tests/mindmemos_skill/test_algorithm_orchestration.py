"""Application-owned orchestration for local Skill algorithms."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from mindmemos_skill.application import AlgorithmBuildContext
from mindmemos_skill.application import components as component_module
from mindmemos_skill.management import RegisterSkillRequest
from mindmemos_skill.registry import ComponentType, register
from mindmemos_skill.typing import (
    EvolveInput,
    EvolveOutput,
    ExecutionInfo,
    Rollout,
    Skill,
    SkillCandidate,
    Task,
    Trace2SkillInput,
    Trace2SkillOutput,
    Trajectory,
    TrajectoryStatus,
    compute_skill_content_hash,
)
from pydantic import BaseModel, ConfigDict

from mindmemos_skill import (
    AlgorithmCommitPolicy,
    EvolveRunRequest,
    SkillApplication,
    Trace2SkillRunRequest,
)


class _AlgorithmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@register(
    type=ComponentType.ALGO,
    name="test_orchestrated_trace2skill",
    config_model=_AlgorithmConfig,
    capabilities={"optimize"},
)
class _Trace2SkillAlgorithm:
    def __init__(self, *, config: _AlgorithmConfig, context: AlgorithmBuildContext) -> None:
        del config, context

    async def optimize(self, request: Trace2SkillInput) -> Trace2SkillOutput[dict[str, str]]:
        return Trace2SkillOutput(
            candidate=SkillCandidate(blob={"SKILL.md": request.base_skill.content + "\nTrace2Skill\n"}),
            report={"status": "changed"},
        )


@register(
    type=ComponentType.ALGO,
    name="test_orchestrated_evolve",
    config_model=_AlgorithmConfig,
    capabilities={"evolve"},
)
class _EvolveAlgorithm:
    def __init__(self, *, config: _AlgorithmConfig, context: AlgorithmBuildContext) -> None:
        del config, context

    async def evolve(self, request: EvolveInput) -> EvolveOutput:
        blob = {"SKILL.md": request.base_skill.content + "\nEvolved\n"}
        final_skill = request.base_skill.model_copy(
            update={
                "version_id": f"{request.run_id}-candidate",
                "blob": blob,
                "content_hash": compute_skill_content_hash(blob),
            }
        )
        return EvolveOutput(
            run_id=request.run_id,
            final_skill=final_skill,
            changed=True,
            trajectories=[_trajectory(request.base_skill, f"{request.run_id}-trajectory")],
        )


def _config(tmp_path: Path) -> dict:
    return {
        "local": {
            "root_dir": str(tmp_path / "skill"),
            "database": {"provider": "sqlite", "path": "state.db"},
        },
        "runtime": {
            "algorithms": {
                "trace-patch": {"type": "test_orchestrated_trace2skill"},
                "grpo": {"type": "test_orchestrated_evolve"},
            }
        },
    }


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        'name: demo\ndescription: Demo\nversion: "1.0.0"\n\nOriginal\n',
        encoding="utf-8",
    )
    return source


def _trajectory(skill: Skill, trajectory_id: str) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        task=Task(task_id=f"{trajectory_id}-task", instruction="Improve"),
        rollout=Rollout(rollout_id=f"{trajectory_id}-rollout"),
        injected_skills=[skill],
        events=[{"role": "user", "content": "Improve"}],
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=datetime(2026, 8, 11, tzinfo=UTC),
            finished_at=datetime(2026, 8, 11, 0, 0, 1, tzinfo=UTC),
        ),
    )


@pytest.mark.asyncio
async def test_trace2skill_resolves_trajectory_and_persists_version_and_log(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))
    base = Skill.from_record(await application.get_version("demo", registered.version_id))
    trajectory = _trajectory(base, "existing-trajectory")
    await application.record_trajectory(trajectory)

    result = await application.run_trace2skill(
        Trace2SkillRunRequest(
            run_id="trace-run",
            algorithm_name="trace-patch",
            skill_ref="demo",
            trajectory_ids=[trajectory.trajectory_id],
        )
    )

    assert result.persisted_version_id is not None
    assert result.input_trajectory_ids == ["existing-trajectory"]
    assert result.generated_trajectory_ids == []
    assert result.persisted_trajectory_ids == []
    assert (await application.get_algorithm_log(result.algorithm_log_ids[0])).step.payload["run_id"] == "trace-run"
    assert (await application.get_skill("demo")).skill.version_count == 2
    await application.close()


@pytest.mark.asyncio
async def test_evolve_persists_generated_trajectory_candidate_and_log(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))

    result = await application.run_evolve(
        EvolveRunRequest(
            run_id="evolve-run",
            algorithm_name="grpo",
            skill_ref=registered.skill_id,
            train_tasks=[Task(task_id="train-1", instruction="Improve")],
        )
    )

    assert result.persisted_version_id is not None
    assert result.persisted_version_id != "evolve-run-candidate"
    assert result.input_trajectory_ids == []
    assert result.generated_trajectory_ids == ["evolve-run-trajectory"]
    assert result.persisted_trajectory_ids == ["evolve-run-trajectory"]
    stored = await application.get_trajectory("evolve-run-trajectory")
    assert stored.metadata["algorithm_run_id"] == "evolve-run"
    assert (await application.get_algorithm_log(result.algorithm_log_ids[0])).step.status == "succeeded"
    await application.close()


@pytest.mark.asyncio
async def test_dry_run_does_not_persist_algorithm_side_effects(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))

    result = await application.run_evolve(
        EvolveRunRequest(
            run_id="dry-run",
            algorithm_name="grpo",
            skill_ref=registered.skill_id,
            train_tasks=[Task(task_id="train-1", instruction="Improve")],
            commit_policy=AlgorithmCommitPolicy.DRY_RUN,
        )
    )

    assert result.persisted_version_id is None
    assert result.generated_trajectory_ids == ["dry-run-trajectory"]
    assert result.persisted_trajectory_ids == []
    assert result.algorithm_log_ids == []
    assert (await application.get_skill("demo")).skill.version_count == 1
    await application.close()


@pytest.mark.asyncio
async def test_builtin_evolver_composes_through_skill_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(component_module, "get_router", lambda config, alias: (object(), 0))
    config = _config(tmp_path)
    config["runtime"] = {
        "models": {"target": {"model": "openai/test-model"}},
        "agents": {"executor": {"type": "react", "model_ref": "target"}},
        "algorithms": {
            "grpo": {
                "type": "skill_grpo_with_replay_buffer",
                "model_roles": {"chat": "target"},
                "config": {
                    "algorithm": {"replay": {"use_embeddings": False, "embedding_model_id": None}},
                    "dataset": {"env_ref": "livemath", "agent_ref": "executor"},
                },
            }
        },
    }

    application = await SkillApplication.from_config(config)

    assert "evolve" in application.capabilities
    assert application._runtime.skill_algorithms is not None
    assert application._runtime.skill_algorithms.algorithm_names("evolve") == ("grpo",)
    await application.close()
