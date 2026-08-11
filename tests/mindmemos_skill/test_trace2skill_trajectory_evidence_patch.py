"""Focused contracts for the offline trajectory evidence patch algorithm."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from mindmemos_skill.algos.trace2skill import (
    AnnotationMode,
    TaskCollectionConfig,
    Trace2SkillAlgorithm,
    TrajectoryCollectionResult,
    TrajectoryCollector,
    TrajectoryEvidencePatch,
    TrajectoryEvidencePatchConfig,
)
from mindmemos_skill.application.components import compose_runtime
from mindmemos_skill.config import SkillConfigCompiler
from mindmemos_skill.errors import SkillConfigurationError
from mindmemos_skill.infra.database import RecordQuery
from mindmemos_skill.llm import DatabaseLLMCallSink, LLMClient, current_llm_run_id
from mindmemos_skill.persistence import LLM_CALL_TABLE, LLMCallRecord, bootstrap_skill_database, from_database_record
from mindmemos_skill.registry import ComponentType, get_component
from mindmemos_skill.service import SkillOptimizer
from mindmemos_skill.typing import (
    ExecutionInfo,
    Reward,
    Rollout,
    Skill,
    SkillVersionOrigin,
    Task,
    Trace2SkillInput,
    Trajectory,
    TrajectoryStatus,
    compute_skill_content_hash,
)

NOW = datetime(2026, 8, 11, tzinfo=UTC)


class FakeChatModel:
    def __init__(self, *, no_edits: bool = False, fail_summary_marker: str | None = None) -> None:
        self.no_edits = no_edits
        self.fail_summary_marker = fail_summary_marker
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.run_ids: list[str | None] = []

    async def chat(self, task: str, messages: list[dict[str, Any]], format_parser=None, **kwargs: Any):
        del kwargs
        self.calls.append((task, messages))
        self.run_ids.append(current_llm_run_id())
        if task == "trajectory_evidence_summary":
            if self.fail_summary_marker and self.fail_summary_marker in messages[-1]["content"]:
                raise RuntimeError("summary failed")
            return SimpleNamespace(content="Evidence-based trajectory summary.", parsed=None)
        if task == "trajectory_evidence_patch_propose":
            return SimpleNamespace(content="Add a concrete verification step.", parsed=None)
        if task == "trajectory_evidence_patch_apply":
            raw = (
                json.dumps({"edits": []})
                if self.no_edits
                else json.dumps({"edits": [{"op": "insert", "after": 3, "new": "- Verify the output."}]})
            )
            parsed = format_parser(raw) if format_parser is not None else None
            return SimpleNamespace(content=raw, parsed=parsed)
        if task == "trajectory_evidence_patch_rewrite":
            return SimpleNamespace(content=messages[-1]["content"].split("\n", 1)[1], parsed=None)
        raise AssertionError(f"unexpected task: {task}")


@dataclass(frozen=True)
class FakeContext:
    models: dict[str, FakeChatModel]
    agents: dict[str, object] = field(default_factory=dict)
    config_hash: str = "a" * 64


class FakeCollector:
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


def make_skill(*, skill_id: str = "skill-1", version_id: str = "version-1") -> Skill:
    blob = {"SKILL.md": "# Demo\n\n- Existing guidance.\n"}
    return Skill(
        skill_id=skill_id,
        version_id=version_id,
        version_label="1.0.0",
        content_hash=compute_skill_content_hash(blob),
        origin=SkillVersionOrigin.LOCAL,
        name="demo",
        blob=blob,
        created_at=NOW,
    )


def make_trajectory(
    skill: Skill,
    index: int,
    *,
    score: float | None = None,
    detail: str | None = None,
    marker: str = "",
    injected_skill: Skill | None = None,
) -> Trajectory:
    started_at = NOW + timedelta(minutes=index)
    return Trajectory(
        trajectory_id=f"trajectory-{index}",
        task=Task(task_id=f"task-{index}", instruction=f"Do task {index}"),
        rollout=Rollout(rollout_id=f"rollout-{index}"),
        injected_skills=[injected_skill or skill],
        events=[{"role": "user", "content": f"Do task {index} {marker}"}],
        reward=Reward(score=score, detail=detail),
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=10),
            n_turn=1,
        ),
    )


def make_algorithm(
    chat: Any,
    *,
    collector: TrajectoryCollector | None = None,
    **overrides: Any,
) -> TrajectoryEvidencePatch:
    config = {
        "min_trajectories": 2,
        "max_trajectories": 2,
        "summary_concurrency": 2,
        **overrides,
    }
    return TrajectoryEvidencePatch(
        config=TrajectoryEvidencePatchConfig.model_validate(config),
        context=FakeContext(models={"chat": chat}),
        collector=collector,
    )


def test_input_requires_at_least_one_source_and_allows_tasks_with_trajectories() -> None:
    skill = make_skill()

    with pytest.raises(ValueError, match="at least one non-empty trajectory source"):
        Trace2SkillInput(base_skill=skill)

    assert Trace2SkillInput(base_skill=skill, trajectories=[make_trajectory(skill, 0)]).trajectories
    assert Trace2SkillInput(
        base_skill=skill,
        tasks=[Task(task_id="task-0", instruction="Do task 0")],
    ).tasks
    combined = Trace2SkillInput(
        base_skill=skill,
        trajectories=[make_trajectory(skill, 0)],
        tasks=[Task(task_id="task-0", instruction="Do task 0")],
    )
    assert combined.trajectories
    assert combined.tasks


@pytest.mark.asyncio
async def test_task_mode_collects_then_uses_the_same_patch_pipeline() -> None:
    skill = make_skill()
    tasks = [
        Task(task_id="task-a", instruction="Do task A"),
        Task(task_id="task-b", instruction="Do task B"),
    ]
    collector = FakeCollector(
        TrajectoryCollectionResult(
            run_id="experiment-1",
            requested_rollout_ids=["rollout-0", "rollout-1", "rollout-failed"],
            trajectories=[make_trajectory(skill, 0), make_trajectory(skill, 1)],
            failed_rollout_ids=["rollout-failed"],
        )
    )
    algorithm = make_algorithm(FakeChatModel(), collector=collector)

    result = await algorithm.optimize(Trace2SkillInput(base_skill=skill, tasks=tasks, run_id="experiment-1"))

    assert result.changed is True
    assert collector.calls == [("experiment-1", skill, tasks)]
    assert result.report.collection_run_id == "experiment-1"
    assert result.report.input_task_ids == ["task-a", "task-b"]
    assert result.report.requested_collection_rollout_ids == ["rollout-0", "rollout-1", "rollout-failed"]
    assert result.report.failed_collection_rollout_ids == ["rollout-failed"]
    assert result.report.used_trajectory_ids == ["trajectory-0", "trajectory-1"]
    assert [trajectory.trajectory_id for trajectory in result.trajectories] == ["trajectory-0", "trajectory-1"]


@pytest.mark.asyncio
async def test_hybrid_mode_merges_offline_and_collected_trajectories() -> None:
    skill = make_skill()
    task = Task(task_id="task-new", instruction="Collect one more trace")
    collector = FakeCollector(
        TrajectoryCollectionResult(
            run_id="hybrid-1",
            requested_rollout_ids=["rollout-new"],
            trajectories=[make_trajectory(skill, 1)],
        )
    )
    algorithm = make_algorithm(FakeChatModel(), collector=collector)

    result = await algorithm.optimize(
        Trace2SkillInput(
            base_skill=skill,
            trajectories=[make_trajectory(skill, 0)],
            tasks=[task],
            run_id="hybrid-1",
        )
    )

    assert result.report.input_trajectory_ids == ["trajectory-0", "trajectory-1"]
    assert result.report.used_trajectory_ids == ["trajectory-0", "trajectory-1"]
    assert [trajectory.trajectory_id for trajectory in result.trajectories] == ["trajectory-1"]


@pytest.mark.asyncio
async def test_task_mode_requires_collection_runtime_config() -> None:
    skill = make_skill()
    algorithm = make_algorithm(FakeChatModel())

    with pytest.raises(SkillConfigurationError, match="requires the algorithm collection config"):
        await algorithm.optimize(
            Trace2SkillInput(base_skill=skill, tasks=[Task(task_id="task-0", instruction="Do task 0")])
        )


def test_collection_config_has_one_rollout_budget() -> None:
    config = TaskCollectionConfig(agent_ref="react", env_ref="livemath", samples_per_task=3)

    assert config.samples_per_task == 3
    with pytest.raises(ValueError, match="queue_capacity must be at least max_concurrent_rollouts"):
        TaskCollectionConfig(
            agent_ref="react",
            env_ref="livemath",
            max_concurrent_rollouts=2,
            queue_capacity=1,
        )


@pytest.mark.asyncio
async def test_below_threshold_returns_unchanged_without_llm_calls() -> None:
    skill = make_skill()
    chat = FakeChatModel()
    algorithm = make_algorithm(chat)

    result = await algorithm.optimize(Trace2SkillInput(base_skill=skill, trajectories=[make_trajectory(skill, 0)]))

    assert result.changed is False
    assert result.candidate is None
    assert result.report.reason == "below_minimum_trajectory_count"
    assert chat.calls == []


@pytest.mark.asyncio
async def test_unannotated_evidence_uses_unsupervised_prompt_and_returns_candidate() -> None:
    skill = make_skill()
    chat = FakeChatModel()
    algorithm = make_algorithm(chat)

    result = await algorithm.optimize(
        Trace2SkillInput(
            base_skill=skill,
            trajectories=[make_trajectory(skill, 1), make_trajectory(skill, 0)],
        )
    )

    assert isinstance(algorithm, Trace2SkillAlgorithm)
    assert isinstance(algorithm, SkillOptimizer)
    assert result.changed is True
    assert result.candidate is not None
    assert result.candidate.blob["SKILL.md"].endswith("- Verify the output.\n")
    assert not hasattr(result.candidate, "version_id")
    assert result.report.used_trajectory_ids == ["trajectory-0", "trajectory-1"]
    patch_call = next(messages for task, messages in chat.calls if task == "trajectory_evidence_patch_propose")
    assert "no outcome labels" in patch_call[0]["content"]


@pytest.mark.asyncio
async def test_optimize_binds_every_llm_call_to_the_request_run_id() -> None:
    skill = make_skill()
    chat = FakeChatModel()
    algorithm = make_algorithm(chat)

    assert current_llm_run_id() is None
    result = await algorithm.optimize(
        Trace2SkillInput(
            run_id="application-run-1",
            base_skill=skill,
            trajectories=[make_trajectory(skill, 0), make_trajectory(skill, 1)],
        )
    )

    assert current_llm_run_id() is None
    assert chat.run_ids == ["application-run-1"] * 4
    assert result.report.run_id == "application-run-1"
    assert result.candidate is not None
    assert result.candidate.metadata["trace2skill"]["run_id"] == "application-run-1"


@pytest.mark.asyncio
async def test_optimize_persists_llm_calls_under_the_generated_run_id(tmp_path) -> None:
    class ScriptedRouter:
        def __init__(self) -> None:
            self.responses = [
                "Evidence-based trajectory summary.",
                "Evidence-based trajectory summary.",
                "Add a concrete verification step.",
                json.dumps({"edits": [{"op": "insert", "after": 3, "new": "- Verify the output."}]}),
            ]

        async def acompletion(self, **kwargs: Any) -> SimpleNamespace:
            del kwargs
            content = self.responses.pop(0)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                model="chat-model",
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            )

    database = await bootstrap_skill_database(tmp_path / "state.db")
    try:
        chat = LLMClient(ScriptedRouter(), call_sink=DatabaseLLMCallSink(database))
        algorithm = make_algorithm(chat)
        skill = make_skill()

        result = await algorithm.optimize(
            Trace2SkillInput(
                base_skill=skill,
                trajectories=[make_trajectory(skill, 0), make_trajectory(skill, 1)],
            )
        )

        rows, cursor = await database.query_records(LLM_CALL_TABLE, RecordQuery())
        records = [from_database_record(row, LLMCallRecord) for row in rows]
        assert cursor is None
        assert len(records) == 4
        assert result.report.run_id.startswith("trace2skill-")
        assert {record.run_id for record in records} == {result.report.run_id}
        assert {record.task for record in records} == {
            "trajectory_evidence_summary",
            "trajectory_evidence_patch_propose",
            "trajectory_evidence_patch_apply",
        }
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_auto_mode_uses_annotations_without_treating_missing_score_as_failure() -> None:
    skill = make_skill()
    chat = FakeChatModel()
    algorithm = make_algorithm(chat)

    await algorithm.optimize(
        Trace2SkillInput(
            base_skill=skill,
            trajectories=[
                make_trajectory(skill, 0, score=1.0, detail="passed"),
                make_trajectory(skill, 1),
            ],
        )
    )

    patch_call = next(messages for task, messages in chat.calls if task == "trajectory_evidence_patch_propose")
    assert "available annotations" in patch_call[0]["content"]
    assert "score: unknown" in patch_call[1]["content"]
    assert "unknown is unlabeled, not a failure" in patch_call[0]["content"]


@pytest.mark.asyncio
async def test_required_annotations_fail_before_llm_call() -> None:
    skill = make_skill()
    chat = FakeChatModel()
    algorithm = make_algorithm(chat, annotation_mode=AnnotationMode.REQUIRED)

    with pytest.raises(ValueError, match="missing the required reward score"):
        await algorithm.optimize(
            Trace2SkillInput(
                base_skill=skill,
                trajectories=[make_trajectory(skill, 0), make_trajectory(skill, 1, score=1.0)],
            )
        )

    assert chat.calls == []


@pytest.mark.asyncio
async def test_duplicate_ids_are_deduplicated_and_reported() -> None:
    skill = make_skill()
    first = make_trajectory(skill, 0)
    chat = FakeChatModel()
    algorithm = make_algorithm(chat, min_trajectories=1, max_trajectories=1)

    result = await algorithm.optimize(Trace2SkillInput(base_skill=skill, trajectories=[first, first]))

    assert result.changed is True
    assert result.report.duplicate_trajectory_ids == ["trajectory-0"]
    assert sum(task == "trajectory_evidence_summary" for task, _ in chat.calls) == 1


@pytest.mark.asyncio
async def test_summary_failures_can_drop_batch_below_threshold() -> None:
    skill = make_skill()
    chat = FakeChatModel(fail_summary_marker="FAIL")
    algorithm = make_algorithm(chat)

    result = await algorithm.optimize(
        Trace2SkillInput(
            base_skill=skill,
            trajectories=[make_trajectory(skill, 0), make_trajectory(skill, 1, marker="FAIL")],
        )
    )

    assert result.changed is False
    assert result.report.reason == "insufficient_summaries_after_failures"
    assert result.report.failed_summary_trajectory_ids == ["trajectory-1"]
    assert all(task != "trajectory_evidence_patch_propose" for task, _ in chat.calls)


@pytest.mark.asyncio
async def test_no_edit_payload_returns_unchanged_skill() -> None:
    skill = make_skill()
    algorithm = make_algorithm(FakeChatModel(no_edits=True))

    result = await algorithm.optimize(
        Trace2SkillInput(
            base_skill=skill,
            trajectories=[make_trajectory(skill, 0), make_trajectory(skill, 1)],
        )
    )

    assert result.changed is False
    assert result.candidate is None
    assert result.report.reason == "no_effective_change"


@pytest.mark.asyncio
async def test_rejects_wrong_skill_and_oversized_batch_before_llm() -> None:
    skill = make_skill()
    other = make_skill(skill_id="skill-2", version_id="version-2")
    chat = FakeChatModel()
    algorithm = make_algorithm(chat)

    with pytest.raises(ValueError, match="does not reference Skill family"):
        await algorithm.optimize(
            Trace2SkillInput(
                base_skill=skill,
                trajectories=[make_trajectory(skill, 0, injected_skill=other), make_trajectory(skill, 1)],
            )
        )

    oversized = [make_trajectory(skill, index) for index in range(3)]
    with pytest.raises(ValueError, match="one bounded batch"):
        await algorithm.optimize(Trace2SkillInput(base_skill=skill, trajectories=oversized))
    assert chat.calls == []


def test_algorithm_is_registered_as_optimize_component() -> None:
    component = get_component(type=ComponentType.ALGO, name="trajectory_evidence_patch")

    assert component.factory is TrajectoryEvidencePatch
    assert component.config_model is TrajectoryEvidencePatchConfig
    assert component.capabilities == frozenset({"optimize"})
    assert component.requirements.required_model_roles == frozenset({"chat"})


@pytest.mark.asyncio
async def test_compiler_composes_algorithm_with_chat_role(tmp_path) -> None:
    compiled = SkillConfigCompiler().compile(
        {
            "local": {"root_dir": str(tmp_path / "skill")},
            "runtime": {
                "models": {"optimizer": {"model": "openai/test-model"}},
                "algorithms": {
                    "offline_optimizer": {
                        "type": "trajectory_evidence_patch",
                        "model_roles": {"chat": "optimizer"},
                        "config": {"min_trajectories": 2, "max_trajectories": 4},
                    }
                },
            },
        }
    )

    runtime = compose_runtime(compiled)
    try:
        assert runtime.skill_algorithms is not None
        assert runtime.skill_algorithms.capabilities == frozenset({"optimize"})
        assert runtime.algorithm_owners == {"optimize": "offline_optimizer"}
    finally:
        await runtime.close()
