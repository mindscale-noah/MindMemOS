from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from mindmemos_skill.algos.evolve.skill_grpo_with_experience_validation import (
    ExperienceSource,
    ExperienceValidationDecision,
    ExtractedExperienceSet,
    PatchDecision,
    SkillGrpoWithExperienceValidation,
    SkillGrpoWithExperienceValidationEvolveInput,
    SkillGrpoWithExperienceValidationRunConfig,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_experience_validation.validation import (
    assess_experience,
    render_experience_guidance,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import (
    RolloutAttempt,
    RolloutOutcome,
    RolloutPhase,
    RolloutSpec,
)
from mindmemos_skill.envs import BaseEnv, PreparedRollout
from mindmemos_skill.persistence.enums import TrajectoryStatus
from mindmemos_skill.typing import (
    AgentProfile,
    EnvConfig,
    EvolveInput,
    ExecutionInfo,
    Reward,
    Rollout,
    Skill,
    Task,
    Trajectory,
    compute_skill_content_hash,
)


def make_skill(content: str = "# Demo\n\nold guidance\n") -> Skill:
    blob = {"SKILL.md": content}
    return Skill(
        skill_id="demo",
        version_id="v0",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name="demo",
        blob=blob,
        created_at=datetime.now(UTC),
    )


class FakeChatModel:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del messages, kwargs
        self.tasks.append(task)
        if ".reflection." in task:
            return SimpleNamespace(content="Try a different action sequence and verify the goal.")
        if ".experience.contrast." in task:
            return SimpleNamespace(
                content=json.dumps({"experiences": [{"lesson": "contrast-guide", "reason": "faster path"}]})
            )
        if ".experience.failure." in task:
            return SimpleNamespace(
                content=json.dumps({"experiences": [{"lesson": "failure-guide", "reason": "avoid dead end"}]})
            )
        if ".experience.success." in task:
            return SimpleNamespace(
                content=json.dumps({"experiences": [{"lesson": "success-guide", "reason": "preserve path"}]})
            )
        if task == "skill_grpo_experience_validation.patch":
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "edits": [
                            {
                                "find": "old guidance",
                                "replace": "validated guidance",
                                "supporting_experience_sets": [1, 2, 3],
                            }
                        ]
                    }
                )
            )
        raise AssertionError(f"unexpected chat task {task}")


class ExperimentEnv(BaseEnv[EnvConfig]):
    async def rollout(self, agent, task, skills, *, context):
        del agent
        sample_index = context.metadata["sample_index"]
        content = skills[0].content
        if task.task_id == "test":
            score = float("validated guidance" in content)
        elif "contrast-guide" in content:
            score = float(task.task_id == "contrast" and sample_index >= 1)
        elif "failure-guide" in content:
            score = float(task.task_id == "failed")
        elif "success-guide" in content:
            score = float(task.task_id in {"contrast", "successful"})
        else:
            score = {
                "contrast": float(sample_index >= 2),
                "failed": 0.0,
                "successful": 1.0,
            }[task.task_id]
        now = datetime.now(UTC)
        return Trajectory(
            trajectory_id=f"{context.rollout.rollout_id}:{context.rollout.attempt_no}",
            task=task,
            rollout=context.rollout,
            injected_skills=list(skills),
            agent=AgentProfile(),
            events=[{"role": "assistant", "content": f"answer-{sample_index}"}],
            reward=Reward(score=score),
            execution=ExecutionInfo(
                status=TrajectoryStatus.SUCCEEDED,
                started_at=now,
                finished_at=now,
                n_turn=1,
            ),
        )

    async def _evaluate(self, *, trajectory: Trajectory, prepared: PreparedRollout) -> Reward:
        del trajectory, prepared
        return Reward(score=0.0)


class FakeEnvFactory:
    def create(self, ref: str, config) -> ExperimentEnv:
        assert ref == "fake"
        return ExperimentEnv(config)


class FakeAgentResolver:
    def resolve(self, ref: str):
        assert ref == "fake"
        return object()


def make_config() -> SkillGrpoWithExperienceValidationRunConfig:
    return SkillGrpoWithExperienceValidationRunConfig.model_validate(
        {
            "algorithm": {
                "reflection": {"enabled": True},
                "experience": {"max_experiences_per_task": 1},
            },
            "training": {"epochs": 1, "batch_size": 3, "mini_batch_size": 8, "seed": 7},
            "rollout": {
                "max_concurrent_rollouts": 4,
                "queue_capacity": 12,
                "train": {"name": "fixed_group", "params": {"group_size": 4}},
                "experience_validation": {"name": "fixed_group", "params": {"group_size": 1}},
            },
            "dataset": {"env_ref": "fake", "agent_ref": "fake"},
        }
    )


def test_generic_evolve_input_keeps_test_tasks_during_normalization() -> None:
    algorithm = SkillGrpoWithExperienceValidation(
        config=make_config(),
        chat_model=FakeChatModel(),
        agent_resolver=FakeAgentResolver(),
        env_factory=FakeEnvFactory(),
    )
    test_task = Task(task_id="test", instruction="test task")

    normalized = algorithm._normalize_request(
        EvolveInput(
            run_id="application-run",
            base_skill=make_skill(),
            train_tasks=[Task(task_id="train", instruction="train task")],
            test_tasks=[test_task],
        )
    )

    assert normalized.test_tasks == [test_task]


@pytest.mark.asyncio
async def test_all_three_sources_must_pass_targeted_reruns_before_patch() -> None:
    chat = FakeChatModel()
    algorithm = SkillGrpoWithExperienceValidation(
        chat_model=chat,
        agent_resolver=FakeAgentResolver(),
        env_factory=FakeEnvFactory(),
    )

    result = await algorithm.evolve(
        SkillGrpoWithExperienceValidationEvolveInput(
            run_id="experience-validation",
            base_skill=make_skill(),
            train_tasks=[
                Task(task_id="contrast", instruction="contrast task"),
                Task(task_id="failed", instruction="failed task"),
                Task(task_id="successful", instruction="successful task"),
            ],
            test_tasks=[Task(task_id="test", instruction="test task")],
            config=make_config(),
        )
    )

    batch = result.batches[0]
    assert [item.source for item in batch.experiences] == [
        ExperienceSource.CONTRAST,
        ExperienceSource.FAILURE,
        ExperienceSource.SUCCESS,
    ]
    assert [item.decision for item in batch.experience_validations] == [
        ExperienceValidationDecision.ACCEPTED,
        ExperienceValidationDecision.ACCEPTED,
        ExperienceValidationDecision.ACCEPTED,
    ]
    contrast, failure, success = batch.experience_validations
    assert (contrast.baseline_first_success_attempt, contrast.injected_first_success_attempt) == (3, 2)
    assert (failure.baseline_success_rate, failure.injected_success_rate) == (0.0, 0.5)
    assert (success.baseline_success_rate, success.injected_success_rate) == (1.0, 1.0)
    assert batch.accepted_experiences == batch.experiences
    assert batch.patch_decision is PatchDecision.APPLIED
    assert "validated guidance" in result.final_skill.content
    assert result.metrics.experiences_accepted == 3
    assert result.metrics.experience_validation_rollouts == 6
    assert result.metrics.test_score == 1.0
    test_outcomes = [outcome for outcome in result.rollouts if outcome.spec.phase is RolloutPhase.TEST]
    assert len(test_outcomes) == 1
    assert test_outcomes[0].spec.skills[0].content_hash == result.final_skill.content_hash
    patch_call = chat.tasks.index("skill_grpo_experience_validation.patch")
    assert all(".experience." in task or ".reflection." in task for task in chat.tasks[:patch_call])


def make_outcome(task_id: str, *, sample_index: int, score: float) -> RolloutOutcome:
    now = datetime.now(UTC)
    task = Task(task_id=task_id, instruction=task_id)
    spec = RolloutSpec(
        sequence_no=sample_index,
        rollout_id=f"{task_id}-{sample_index}",
        phase=RolloutPhase.VALIDATION,
        task=task,
        skills=[make_skill()],
        sample_index=sample_index,
        agent_ref="fake",
        env_ref="fake",
    )
    trajectory = Trajectory(
        trajectory_id=f"trajectory-{task_id}-{sample_index}",
        task=task,
        rollout=Rollout(rollout_id=spec.rollout_id, attempt_no=0),
        injected_skills=[make_skill()],
        reward=Reward(score=score),
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            n_turn=1,
        ),
    )
    attempt = RolloutAttempt(attempt_no=0, trajectory=trajectory, started_at=now, finished_at=now)
    return RolloutOutcome(spec=spec, attempts=[attempt], trajectory=trajectory, succeeded=True)


@pytest.mark.parametrize(
    ("source", "task_ids", "outcomes", "baseline_attempt"),
    [
        (ExperienceSource.CONTRAST, ["a"], [make_outcome("a", sample_index=2, score=1.0)], 3),
        (ExperienceSource.FAILURE, ["a", "b"], [make_outcome("a", sample_index=0, score=0.0), make_outcome("b", sample_index=0, score=0.0)], None),
        (ExperienceSource.SUCCESS, ["a", "b"], [make_outcome("a", sample_index=0, score=1.0), make_outcome("b", sample_index=0, score=0.0)], None),
    ],
)
def test_source_gates_reject_no_improvement_or_success_drop(
    source: ExperienceSource,
    task_ids: list[str],
    outcomes: list[RolloutOutcome],
    baseline_attempt: int | None,
) -> None:
    experience = ExtractedExperienceSet(
        task_id="set",
        task_ids=task_ids,
        source=source,
        content='{"experiences": [{"lesson": "guide"}]}',
        rollout_count=len(task_ids),
    )

    record = assess_experience(
        experience,
        experience_index=0,
        injected_outcomes=outcomes,
        baseline_first_success_attempt=baseline_attempt,
        success_reward=1.0,
    )

    assert record.decision is ExperienceValidationDecision.REJECTED


def test_experience_injection_excludes_task_evidence() -> None:
    guidance = render_experience_guidance(
        json.dumps(
            {
                "experiences": [
                    {
                        "lesson": "Use the reusable check.",
                        "reason": "It catches incomplete work.",
                        "evidence": [{"task_id": "secret-task", "observation": "specific object"}],
                    }
                ]
            }
        )
    )

    assert "Use the reusable check." in guidance
    assert "It catches incomplete work." in guidance
    assert "secret-task" not in guidance
    assert "specific object" not in guidance
