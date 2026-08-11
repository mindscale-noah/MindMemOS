from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime

import pytest
from mindmemos_skill.agents import Agent, AgentConfig, AgentExecutionRequest, SkillInjection
from mindmemos_skill.envs import BaseEnv, EnvRolloutContext, PreparedRollout
from mindmemos_skill.typing import (
    EnvConfig,
    ExecutionInfo,
    Reward,
    Rollout,
    RolloutType,
    Skill,
    SkillBinding,
    SkillInjectionMode,
    Task,
    Trajectory,
    TrajectoryStatus,
)
from pydantic import ValidationError


class ExampleEnvConfig(EnvConfig):
    metric: str = "accuracy"


class ExampleEnv(BaseEnv[ExampleEnvConfig]):
    config_type = ExampleEnvConfig

    def __init__(self, config: ExampleEnvConfig | dict) -> None:
        super().__init__(config)
        self.setup_called = False
        self.cleanup_called = False
        self.teardown_called = False

    async def setup(self) -> None:
        self.setup_called = True

    async def cleanup(self) -> None:
        self.cleanup_called = True

    async def _evaluate(
        self,
        *,
        trajectory: Trajectory,
        prepared: PreparedRollout,
    ) -> Reward:
        assert trajectory.environment == prepared.environment
        return Reward(
            score=0.75,
            detail="evaluated",
            metadata={"metric": self.config.metric, "hard": 1, "soft": 0.75},
        )

    async def _teardown(self, prepared: PreparedRollout) -> None:
        assert prepared.agent_request.task.task_id
        self.teardown_called = True


class FakeAgent(Agent[AgentConfig]):
    def __init__(self) -> None:
        super().__init__({})
        self.request: AgentExecutionRequest | None = None

    def inject_skills(
        self,
        skills: list[Skill],
        *,
        mode: SkillInjectionMode | None = None,
    ):
        return nullcontext(SkillInjection(mode=mode or SkillInjectionMode.SYSTEM_PROMPT))

    def bind_skills(self, trajectory: Trajectory) -> list[SkillBinding]:
        return []

    async def execute(self, request: AgentExecutionRequest) -> Trajectory:
        self.request = request
        now = datetime.now(UTC)
        trajectory = Trajectory(
            trajectory_id=request.trajectory_id,
            task=request.task,
            rollout=request.rollout,
            environment=request.environment,
            injected_skills=request.skills,
            execution=ExecutionInfo(
                status=TrajectoryStatus.SUCCEEDED,
                started_at=now,
                finished_at=now,
                n_turn=1,
            ),
        )
        return trajectory


class FailingAgent(FakeAgent):
    async def execute(self, request: AgentExecutionRequest) -> Trajectory:
        self.request = request
        raise RuntimeError("agent failed")


def test_env_config_is_validated_at_construction() -> None:
    env = ExampleEnv({"metric": "soft"})

    assert env.config == ExampleEnvConfig(metric="soft")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleEnv({"unknown": True})


@pytest.mark.asyncio
async def test_rollout_uses_attempt_context_and_evaluates_trajectory(tmp_path) -> None:
    env = ExampleEnv({})
    agent = FakeAgent()
    task = Task(task_id="../unsafe task", instruction="Complete the task")
    skill = Skill(
        skill_id="skill-1",
        version_id="version-1",
        version_label="1.0.0",
        content_hash="sha256:main",
        name="main",
        blob={"SKILL.md": "# Main"},
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    rollout = Rollout(rollout_id="../rollout 1", attempt_no=2, rollout_type=RolloutType.TRAIN)
    context = EnvRolloutContext(
        rollout=rollout,
        env_ref="example",
        workspace_root=tmp_path,
        workspace_scope="train/epoch 1",
        agent_options={"temperature": 0.2},
        metadata={"batch": 3},
    )

    trajectory = await env.rollout(agent, task, [skill], context=context)

    workspace = tmp_path / "train" / "epoch_1" / "unsafe_task" / "rollout_1" / "2"
    assert workspace.is_dir()
    assert trajectory.rollout == rollout
    assert trajectory.environment.env_ref == "example"
    assert trajectory.environment.running_dir == str(workspace)
    assert trajectory.environment.metadata == {"batch": 3, "workspace_scope": "train/epoch 1"}
    assert trajectory.reward == Reward(
        score=0.75,
        detail="evaluated",
        metadata={"metric": "accuracy", "hard": 1, "soft": 0.75},
    )
    assert agent.request is not None
    assert agent.request.environment.running_dir == str(workspace)
    assert agent.request.rollout == rollout
    assert agent.request.options == {"temperature": 0.2}
    assert agent.request.metadata == {"batch": 3}
    assert env.teardown_called is True


@pytest.mark.asyncio
async def test_rollout_tears_down_attempt_when_execution_fails(tmp_path) -> None:
    env = ExampleEnv({})
    context = EnvRolloutContext(
        rollout=Rollout(rollout_id="rollout-1"),
        workspace_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="agent failed"):
        await env.rollout(
            FailingAgent(),
            Task(task_id="task-1", instruction="Fail"),
            [],
            context=context,
        )

    assert env.teardown_called is True


@pytest.mark.asyncio
async def test_env_async_context_runs_instance_lifecycle() -> None:
    env = ExampleEnv({})

    async with env as active:
        assert active is env
        assert env.setup_called is True
        assert env.cleanup_called is False

    assert env.cleanup_called is True
