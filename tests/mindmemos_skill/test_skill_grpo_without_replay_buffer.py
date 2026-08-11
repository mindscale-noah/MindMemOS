from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer import (
    ExperienceSource,
    ReplayFreeExtractedExperience,
    SkillGrpoWithoutReplayBuffer,
    SkillGrpoWithoutReplayBufferEvolveInput,
    SkillGrpoWithoutReplayBufferRunConfig,
    ValidationDecision,
)
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer import algorithm as replay_free_algorithm
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer.experience import ExperienceExtractor
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer.patch import PatchProposer
from mindmemos_skill.envs import BaseEnv, PreparedRollout
from mindmemos_skill.persistence.enums import TrajectoryStatus
from mindmemos_skill.typing import (
    AgentProfile,
    EnvConfig,
    ExecutionInfo,
    Reward,
    Skill,
    Task,
    Trajectory,
    compute_skill_content_hash,
)
from pydantic import ValidationError


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
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del kwargs
        self.tasks.append(task)
        self.calls.append((task, messages))
        if ".reflection." in task:
            return SimpleNamespace(content="Inspect the failed calculation and use a different method.")
        if ".experience." in task:
            return SimpleNamespace(content='{"experiences": [{"lesson": "use new guidance"}]}')
        if task.startswith("skill_grpo.patch"):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "edits": [
                            {
                                "find": "old guidance",
                                "replace": "new guidance",
                                "supporting_experience_sets": [1],
                                "support_count": 1,
                            }
                        ]
                    }
                )
            )
        raise AssertionError(f"unexpected chat task {task}")


class ScoreEnv(BaseEnv[EnvConfig]):
    def __init__(
        self,
        config: EnvConfig,
        *,
        before: float,
        after: float,
        correct_rollouts_by_task: dict[str, int] | None = None,
    ) -> None:
        super().__init__(config)
        self._before = before
        self._after = after
        self._correct_rollouts_by_task = correct_rollouts_by_task

    async def rollout(self, agent, task, skills, *, context):
        del agent
        now = datetime.now(UTC)
        if self._correct_rollouts_by_task is None:
            score = self._after if "new guidance" in skills[0].content else self._before
        else:
            score = float(context.metadata["sample_index"] < self._correct_rollouts_by_task[task.task_id])
        return Trajectory(
            trajectory_id=f"{context.rollout.rollout_id}:{context.rollout.attempt_no}",
            task=task,
            rollout=context.rollout,
            injected_skills=list(skills),
            agent=AgentProfile(),
            events=[{"role": "assistant", "content": "done"}],
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
    def __init__(
        self,
        *,
        before: float,
        after: float,
        correct_rollouts_by_task: dict[str, int] | None = None,
    ) -> None:
        self._before = before
        self._after = after
        self._correct_rollouts_by_task = correct_rollouts_by_task

    def create(self, ref: str, config) -> ScoreEnv:
        assert ref == "fake"
        return ScoreEnv(
            config,
            before=self._before,
            after=self._after,
            correct_rollouts_by_task=self._correct_rollouts_by_task,
        )


class FakeAgentResolver:
    def resolve(self, ref: str):
        assert ref == "fake"
        return object()


class RecordingAlgorithmLogger:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def log(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class ReflectionEnv(BaseEnv[EnvConfig]):
    async def rollout(self, agent, task, skills, *, context):
        del agent
        now = datetime.now(UTC)
        sample_index = context.metadata["sample_index"]
        return Trajectory(
            trajectory_id=f"{context.rollout.rollout_id}:{context.rollout.attempt_no}",
            task=task,
            rollout=context.rollout,
            injected_skills=list(skills),
            events=[
                {"role": "user", "content": task.instruction},
                {"role": "assistant", "content": f"answer-{sample_index}"},
            ],
            reward=Reward(score=float(sample_index > 0)),
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


class ReflectionEnvFactory:
    def create(self, ref: str, config) -> ReflectionEnv:
        assert ref == "fake"
        return ReflectionEnv(config)


class FirstSuccessEnv(BaseEnv[EnvConfig]):
    async def rollout(self, agent, task, skills, *, context):
        del agent
        now = datetime.now(UTC)
        success_sample = task.metadata.get("success_sample")
        score = float(isinstance(success_sample, int) and context.metadata["sample_index"] >= success_sample)
        return Trajectory(
            trajectory_id=f"{context.rollout.rollout_id}:{context.rollout.attempt_no}",
            task=task,
            rollout=context.rollout,
            injected_skills=list(skills),
            events=[{"role": "assistant", "content": f"answer-{context.metadata['sample_index']}"}],
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


class FirstSuccessEnvFactory:
    def create(self, ref: str, config) -> FirstSuccessEnv:
        assert ref == "fake"
        return FirstSuccessEnv(config)


def make_config(*, validation: bool, reflection: bool = False) -> SkillGrpoWithoutReplayBufferRunConfig:
    return SkillGrpoWithoutReplayBufferRunConfig.model_validate(
        {
            "algorithm": {
                "reflection": {"enabled": reflection},
                "validation": {"enabled": validation},
            },
            "training": {"epochs": 1, "batch_size": 1, "seed": 7},
            "rollout": {
                "max_concurrent_rollouts": 1,
                "train": {"name": "fixed_group", "params": {"group_size": 1}},
                "validation": {"name": "fixed_group", "params": {"group_size": 1}},
            },
            "dataset": {"env_ref": "fake", "agent_ref": "fake"},
        }
    )


def make_request(*, validation: bool) -> SkillGrpoWithoutReplayBufferEvolveInput:
    return SkillGrpoWithoutReplayBufferEvolveInput(
        run_id="no-replay",
        base_skill=make_skill(),
        train_tasks=[Task(task_id="train-1", instruction="train")],
        validation_tasks=[Task(task_id="validation-1", instruction="validate")] if validation else [],
        config=make_config(validation=validation),
    )


@pytest.mark.asyncio
async def test_replay_free_flow_applies_merged_patch_without_validation() -> None:
    chat = FakeChatModel()
    algorithm = SkillGrpoWithoutReplayBuffer(
        chat_model=chat,
        agent_resolver=FakeAgentResolver(),
        env_factory=FakeEnvFactory(before=0.0, after=1.0),
    )

    result = await algorithm.evolve(make_request(validation=False))

    assert result.changed is True
    assert "new guidance" in result.final_skill.content
    assert result.batches[0].validation_decision is ValidationDecision.DISABLED
    assert result.batches[0].candidate_edits == result.batches[0].applied_edits
    assert result.metrics.edits_applied == 1
    assert chat.tasks == ["skill_grpo.experience.failure.0", "skill_grpo.patch"]
    assert replay_free_algorithm.ExperienceExtractor is ExperienceExtractor
    assert replay_free_algorithm.PatchProposer is PatchProposer


@pytest.mark.asyncio
async def test_replay_free_evaluates_final_skill_on_test_tasks() -> None:
    algorithm = SkillGrpoWithoutReplayBuffer(
        chat_model=FakeChatModel(),
        agent_resolver=FakeAgentResolver(),
        env_factory=FakeEnvFactory(before=0.0, after=1.0),
    )
    request = make_request(validation=False).model_copy(
        update={"test_tasks": [Task(task_id="test-1", instruction="test")]},
        deep=True,
    )

    result = await algorithm.evolve(request)

    test_outcomes = [outcome for outcome in result.rollouts if outcome.spec.phase.value == "test"]
    assert len(test_outcomes) == 1
    assert test_outcomes[0].spec.skills[0].content_hash == result.final_skill.content_hash
    assert result.metrics.test_score == 1.0


@pytest.mark.asyncio
async def test_replay_free_events_are_logged_without_event_callback() -> None:
    logger = RecordingAlgorithmLogger()
    algorithm = SkillGrpoWithoutReplayBuffer(
        chat_model=FakeChatModel(),
        agent_resolver=FakeAgentResolver(),
        env_factory=FakeEnvFactory(before=0.0, after=1.0),
        logger=logger,  # type: ignore[arg-type]
    )

    await algorithm.evolve(make_request(validation=False))

    step_names = [call["step_name"] for call in logger.calls]
    assert step_names[0] == "run_started"
    assert {
        "phase_started",
        "rollout_completed",
        "phase_completed",
        "batch_rollout_summary",
        "stage_started",
        "stage_completed",
        "batch_completed",
        "run_finished",
    } <= set(step_names)
    assert step_names.count("batch_rollout_summary") == 1
    rollout_log = next(call for call in logger.calls if call["step_name"] == "rollout_completed")
    assert rollout_log["payload"]["score"] == 0.0
    assert rollout_log["payload"]["attempts"] == 1


@pytest.mark.asyncio
async def test_batch_rollout_summary_reports_early_stopping_histogram(capsys) -> None:
    events = []

    async def capture_event(event) -> None:
        events.append(event)

    config_payload = make_config(validation=False).model_dump(mode="json")
    config_payload["training"]["batch_size"] = 3
    config_payload["rollout"]["max_concurrent_rollouts"] = 4
    config_payload["rollout"]["train"]["params"]["group_size"] = 4
    request = SkillGrpoWithoutReplayBufferEvolveInput(
        run_id="rollout-summary",
        base_skill=make_skill(),
        train_tasks=[
            Task(task_id="correct-4", instruction="four"),
            Task(task_id="correct-3", instruction="three"),
            Task(task_id="correct-0", instruction="zero"),
        ],
        config=SkillGrpoWithoutReplayBufferRunConfig.model_validate(config_payload),
    )
    algorithm = SkillGrpoWithoutReplayBuffer(
        chat_model=FakeChatModel(),
        agent_resolver=FakeAgentResolver(),
        env_factory=FakeEnvFactory(
            before=0.0,
            after=1.0,
            correct_rollouts_by_task={"correct-4": 4, "correct-3": 3, "correct-0": 0},
        ),
        on_event=capture_event,
    )

    await algorithm.evolve(request)

    summary = next(event for event in events if event.name == "batch_rollout_summary")
    assert summary.payload == {
        "batch_index": 0,
        "case_count": 3,
        "rollouts_per_case": 4,
        "success_reward": 1.0,
        "actual_rollout_count": 6,
        "early_stopped_rollout_count": 6,
        "solved_at_attempt_histogram": {"1": 2, "2": 0, "3": 0, "4": 0, "unsolved": 1},
    }
    assert (
        "batch=1 rollout stopping distribution: attempt 1=2 cases, attempt 2=0 cases, "
        "attempt 3=0 cases, attempt 4=0 cases, unsolved=1 cases; actual=6, skipped=6"
    ) in capsys.readouterr().out


@pytest.mark.asyncio
async def test_failed_rollout_reflects_and_passes_answer_and_reflection_to_next_rollout() -> None:
    config_payload = make_config(validation=False, reflection=True).model_dump(mode="json")
    config_payload["rollout"]["train"]["params"]["group_size"] = 2
    chat = FakeChatModel()
    algorithm = SkillGrpoWithoutReplayBuffer(
        chat_model=chat,
        agent_resolver=FakeAgentResolver(),
        env_factory=ReflectionEnvFactory(),
    )

    result = await algorithm.evolve(
        SkillGrpoWithoutReplayBufferEvolveInput(
            run_id="reflective-rollouts",
            base_skill=make_skill(),
            train_tasks=[
                Task(
                    task_id="reflect",
                    instruction="solve the task",
                    metadata={"question": "Which option is correct?"},
                )
            ],
            config=SkillGrpoWithoutReplayBufferRunConfig.model_validate(config_payload),
        )
    )

    first, second = result.rollouts
    assert first.spec.task.instruction == "solve the task"
    assert "answer-0" in second.spec.task.instruction
    assert "Inspect the failed calculation and use a different method." in second.spec.task.instruction
    assert "answer-0" in str(second.spec.task.metadata["question"])
    assert second.spec.metadata["reflection_context"] == {
        "prompt_version": "reflexion-shinn-v1",
        "source_rollout_id": first.spec.rollout_id,
        "previous_answer": "answer-0",
        "content": "Inspect the failed calculation and use a different method.",
    }
    reflection_call = next(call for call in chat.calls if ".reflection." in call[0])
    assert "answer-0" in reflection_call[1][1]["content"]
    assert "expected" not in reflection_call[1][1]["content"]


@pytest.mark.asyncio
async def test_successful_first_rollout_stops_remaining_samples() -> None:
    config_payload = make_config(validation=False, reflection=True).model_dump(mode="json")
    config_payload["rollout"]["train"]["params"]["group_size"] = 2
    chat = FakeChatModel()
    algorithm = SkillGrpoWithoutReplayBuffer(
        chat_model=chat,
        agent_resolver=FakeAgentResolver(),
        env_factory=FakeEnvFactory(
            before=0.0,
            after=1.0,
            correct_rollouts_by_task={"train-1": 2},
        ),
    )

    result = await algorithm.evolve(
        SkillGrpoWithoutReplayBufferEvolveInput(
            run_id="independent-after-success",
            base_skill=make_skill(),
            train_tasks=[Task(task_id="train-1", instruction="train")],
            config=SkillGrpoWithoutReplayBufferRunConfig.model_validate(config_payload),
        )
    )

    assert len(result.rollouts) == 1
    assert result.rollouts[0].spec.task.instruction == "train"
    assert not any(".reflection." in task for task in chat.tasks)


@pytest.mark.asyncio
async def test_three_experience_streams_use_mini_batches_and_priority_order() -> None:
    config_payload = make_config(validation=False, reflection=True).model_dump(mode="json")
    config_payload["training"].update({"batch_size": 10, "mini_batch_size": 3})
    config_payload["rollout"]["max_concurrent_rollouts"] = 10
    config_payload["rollout"]["train"]["params"]["group_size"] = 4
    tasks = [
        *[
            Task(task_id=f"immediate-{index}", instruction="solve", metadata={"success_sample": 0})
            for index in range(4)
        ],
        *[Task(task_id=f"improved-{index}", instruction="solve", metadata={"success_sample": 2}) for index in range(4)],
        *[
            Task(task_id=f"failed-{index}", instruction="solve", metadata={"success_sample": None})
            for index in range(2)
        ],
    ]
    chat = FakeChatModel()
    algorithm = SkillGrpoWithoutReplayBuffer(
        chat_model=chat,
        agent_resolver=FakeAgentResolver(),
        env_factory=FirstSuccessEnvFactory(),
    )

    result = await algorithm.evolve(
        SkillGrpoWithoutReplayBufferEvolveInput(
            run_id="three-streams",
            base_skill=make_skill(),
            train_tasks=tasks,
            config=SkillGrpoWithoutReplayBufferRunConfig.model_validate(config_payload),
        )
    )

    experiences = result.batches[0].experiences
    assert [experience.source for experience in experiences] == [
        *([ExperienceSource.CONTRAST] * 4),
        *([ExperienceSource.FAILURE] * 2),
        *([ExperienceSource.SUCCESS] * 3),
    ]
    assert [experience.rollout_count for experience in experiences[:4]] == [3, 3, 3, 3]
    assert all(len(experience.task_ids) <= 3 for experience in experiences[4:])
    assert len(result.rollouts) == 24
    patch_call = next(messages for task, messages in chat.calls if task == "skill_grpo.patch")
    assert "CONTRAST" in patch_call[0]["content"]
    assert patch_call[1]["content"].index("source=contrast") < patch_call[1]["content"].index("source=failure")
    assert patch_call[1]["content"].index("source=failure") < patch_call[1]["content"].index("source=success")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("candidate_score", "expected_decision", "expected_changed"),
    [
        (0.75, ValidationDecision.ACCEPTED, True),
        (0.5, ValidationDecision.ACCEPTED, True),
        (0.25, ValidationDecision.REJECTED, False),
    ],
)
async def test_validation_gate_accepts_non_regression(
    candidate_score: float,
    expected_decision: ValidationDecision,
    expected_changed: bool,
) -> None:
    algorithm = SkillGrpoWithoutReplayBuffer(
        chat_model=FakeChatModel(),
        agent_resolver=FakeAgentResolver(),
        env_factory=FakeEnvFactory(before=0.5, after=candidate_score),
    )

    result = await algorithm.evolve(make_request(validation=True))

    batch = result.batches[0]
    assert result.changed is expected_changed
    assert batch.validation_score_before == 0.5
    assert batch.validation_score_after == candidate_score
    assert batch.validation_decision is expected_decision
    assert bool(batch.applied_edits) is expected_changed
    assert len(result.rollouts) == 3


def test_replay_configuration_is_not_part_of_replay_free_contract() -> None:
    payload = make_config(validation=False).model_dump(mode="json")
    payload["algorithm"]["replay"] = {"capacity": 10}

    with pytest.raises(ValidationError, match="replay"):
        SkillGrpoWithoutReplayBufferRunConfig.model_validate(payload)


def test_replay_free_defaults_use_large_batch_and_extraction_mini_batch() -> None:
    config = SkillGrpoWithoutReplayBufferRunConfig.model_validate({"dataset": {"env_ref": "fake", "agent_ref": "fake"}})

    assert config.training.batch_size == 40
    assert config.training.mini_batch_size == 8
    assert config.rollout.train.params["group_size"] == 4
    assert config.algorithm.reflection.enabled is True


@pytest.mark.asyncio
async def test_patch_budget_prefers_contrast_then_failure_sources() -> None:
    class PriorityChatModel:
        async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            del messages, kwargs
            assert task == "skill_grpo.patch"
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "edits": [
                            {
                                "find": "success guidance",
                                "replace": "success edit",
                                "supporting_experience_sets": [3],
                            },
                            {
                                "find": "failure guidance",
                                "replace": "failure edit",
                                "supporting_experience_sets": [2],
                            },
                            {
                                "find": "contrast guidance",
                                "replace": "contrast edit",
                                "supporting_experience_sets": [1],
                            },
                        ]
                    }
                )
            )

    skill = make_skill("# Demo\n\ncontrast guidance\nfailure guidance\nsuccess guidance\n")
    experiences = [
        ReplayFreeExtractedExperience(
            task_id=source.value,
            task_ids=[source.value],
            source=source,
            content="evidence",
            rollout_count=1,
        )
        for source in (ExperienceSource.CONTRAST, ExperienceSource.FAILURE, ExperienceSource.SUCCESS)
    ]

    proposal = await PatchProposer(PriorityChatModel(), max_edits=2, max_attempts=1).propose(skill, experiences)

    assert [item.edit.find for item in proposal.edit_support] == ["contrast guidance", "failure guidance"]
