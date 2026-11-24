"""Verify rollout execution, durable completion, and resume boundaries."""

from pathlib import Path
from unittest.mock import patch

import pytest
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task, Usage
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.skill import Skill, SkillContent, SkillSet
from pydantic import BaseModel, ValidationError


class Options(BaseModel):
    """Settings that must participate in resume identity."""

    model: str = "test-model"
    """Agent model identity used by this test environment."""


def make_skill(content):
    """Build a static source content for execution tests."""
    return Skill(
        source=SkillContent(content=content, name="test-skill", description="Task instructions", standard_format=False)
    )


class ExampleEnvironment(Environment[Options]):
    """Use local artifacts to simulate an agent without model calls."""

    config_type = Options
    """Configuration accepted by the test environment."""

    calls: int = 0
    """Number of physical executions on this instance."""

    fail: bool = False
    """Whether execution should raise before completion."""

    async def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Write a workspace artifact and return an evaluated trajectory."""
        self.calls += 1
        assert workspace.is_absolute()
        assert workspace.name == "workspace"
        assert not (workspace / "audit").exists()
        assert not (workspace / "request.json").exists()
        assert (workspace.parent / "audit/request.json").exists()
        (workspace / "answer.txt").write_text(task.instruction, encoding="utf-8")
        if self.fail:
            raise RuntimeError("Agent failed")
        return RolloutResult(
            output=task.instruction,
            trajectory=[
                {
                    "task": task.model_dump(mode="json"),
                    "skills": [{"SKILL.md": skill.content} for skill in skills],
                    "turn": 1,
                }
            ],
            reward=Reward(score=0.0, success=False, detail="Answer did not pass"),
        )


@pytest.mark.parametrize("completed", [False, True])
async def test_unsupported_resume_has_no_side_effects(tmp_path, completed):
    """Reject resume before assembly or writes, including when a result exists."""

    class NonResumableEnvironment(ExampleEnvironment):
        supports_resume = False

    env = NonResumableEnvironment(Options())
    task = Task(id="task-001", instruction="Solve this")
    skill = make_skill("Check the answer")
    work_dir = tmp_path / "rollout"
    if completed:
        (
            await env.rollout(
                task,
                work_dir,
                reward_data=RewardData(),
                skill_set=SkillSet({"0": skill}, max_concurrency=1),
                resume=False,
            )
        )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with patch.object(skill, "assemble", side_effect=AssertionError("Unexpected assembly")):
        with pytest.raises(NotImplementedError, match="does not support resume; pass resume=False"):
            (
                await env.rollout(
                    task,
                    work_dir,
                    reward_data=RewardData(),
                    skill_set=SkillSet({"0": skill}, max_concurrency=1),
                    resume=True,
                )
            )
        with pytest.raises(NotImplementedError, match="does not support resume"):
            (
                await env.rollout(
                    task, work_dir, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1)
                )
            )
    assert env.calls == int(completed)
    assert work_dir.exists() == completed
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


async def test_non_resumable_environment_executes_each_explicit_run(tmp_path):
    """Disabling resume retains audit persistence and executes matching reruns."""

    class NonResumableEnvironment(ExampleEnvironment):
        supports_resume = False

    env = NonResumableEnvironment(Options())
    task = Task(id="task-001", instruction="Solve this")
    for _ in range(2):
        result = await env.rollout(task, tmp_path, reward_data=RewardData(), resume=False)
        assert result.output == task.instruction
        assert (tmp_path / "audit/result.json").exists()
    assert env.calls == 2


async def test_resume_from_new_instance_without_execution(tmp_path):
    """A complete zero-reward run survives process-local state loss."""
    env = ExampleEnvironment(Options())
    result = await env.rollout(
        Task(id="task-001", instruction="Solve this"),
        tmp_path,
        reward_data=RewardData(),
        skill_set=SkillSet({"0": make_skill("Check the answer")}, max_concurrency=1),
    )
    assert env.calls == 1
    assert (tmp_path / "workspace/answer.txt").read_text() == "Solve this"
    assert {path.name for path in tmp_path.iterdir()} == {"audit", "workspace"}
    assert {path.name for path in (tmp_path / "audit").iterdir()} == {"request.json", "result.json"}
    assert {path.name for path in (tmp_path / "workspace").iterdir()} == {"answer.txt"}
    assert result.trajectory[0]["skills"] == [{"SKILL.md": "Check the answer"}]
    restored = ExampleEnvironment(Options())
    restored.fail = True
    assert (
        await restored.rollout(
            Task(id="task-001", instruction="Solve this"),
            tmp_path,
            reward_data=RewardData(),
            skill_set=SkillSet({"0": make_skill("Check the answer")}, max_concurrency=1),
        )
    ) == result
    assert restored.calls == 0


@pytest.mark.parametrize("change", ["task", "skills", "config"])
async def test_reject_changed_inputs(tmp_path, change):
    """Different candidates or execution settings cannot reuse one sample."""
    env = ExampleEnvironment(Options())
    (
        await env.rollout(
            Task(id="task-001", instruction="task"),
            tmp_path,
            reward_data=RewardData(),
            skill_set=SkillSet({"0": make_skill("skill")}, max_concurrency=1),
        )
    )
    if change == "config":
        env.config = Options(model="another-model")
    with pytest.raises(ValueError, match="inputs changed"):
        (
            await env.rollout(
                Task(id="task-001", instruction="different" if change == "task" else "task"),
                tmp_path,
                reward_data=RewardData(),
                skill_set=SkillSet(
                    {"0": make_skill("different" if change == "skills" else "skill")}, max_concurrency=1
                ),
            )
        )
    assert env.calls == 1


async def test_failure_is_retryable_and_forced_rerun_invalidates_completion(tmp_path):
    """A failed forced rerun must not resurrect an older completion."""
    env = ExampleEnvironment(Options())
    (await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData()))
    env.fail = True
    with pytest.raises(RuntimeError, match="Agent failed"):
        (await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData(), resume=False))
    assert not (tmp_path / "audit/result.json").exists()
    env.fail = False
    (await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData()))
    assert env.calls == 3


async def test_partial_result_is_not_completion(tmp_path):
    """An interrupted temporary result does not skip execution."""
    record_dir = tmp_path / "audit"
    record_dir.mkdir()
    (record_dir / "result.json.tmp").write_text('{"output":')
    env = ExampleEnvironment(Options())
    (await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData()))
    assert env.calls == 1


async def test_resume_does_not_recreate_agent_workspace(tmp_path):
    """Reading a complete audit result does not prepare an execution directory."""
    env = ExampleEnvironment(Options())
    task = Task(id="task-001", instruction="task")
    result = await env.rollout(task, tmp_path, reward_data=RewardData())
    (tmp_path / "workspace/answer.txt").unlink()
    (tmp_path / "workspace").rmdir()
    assert (await env.rollout(task, tmp_path, reward_data=RewardData())) == result
    assert not (tmp_path / "workspace").exists()
    assert env.calls == 1


@pytest.mark.parametrize("linked_dir", ["audit", "workspace"])
async def test_reject_linked_rollout_directories(tmp_path, linked_dir):
    """Do not allow directory aliases to collapse audit and agent storage."""
    target = tmp_path / ("workspace" if linked_dir == "audit" else "audit")
    target.mkdir()
    (tmp_path / linked_dir).symlink_to(target, target_is_directory=True)
    env = ExampleEnvironment(Options())
    with pytest.raises(ValueError, match="symbolic links"):
        (await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData()))
    assert env.calls == 0
    assert list(target.iterdir()) == []


async def test_corrupt_completed_result_is_reported(tmp_path):
    """Corrupted completion data does not trigger silent paid execution."""
    env = ExampleEnvironment(Options())
    (await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData()))
    (tmp_path / "audit/result.json").write_text("{}")
    with pytest.raises(ValidationError):
        (await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData()))
    assert env.calls == 1


def test_environment_is_abstract_and_reward_must_be_finite():
    """Concrete execution and a usable numeric reward are required."""
    with pytest.raises(TypeError):
        Environment(Options())
    with pytest.raises(ValidationError):
        Reward(score=float("nan"), success=False, detail="Invalid score")


async def test_task_subclass_fields_and_original_objects_are_preserved(tmp_path):
    """Pass live objects to execution and include extended task fields in resume."""

    class FileTask(Task):
        input_file: str
        """Task-specific input artifact identity."""

    task = FileTask(id="task-001", instruction="Read the file", input_file="first.csv")
    skill = make_skill("Check columns")

    class ObjectEnvironment(ExampleEnvironment):
        async def _run(
            self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path
        ) -> RolloutResult:
            """Confirm execution receives the task and complete prepared Skill objects."""
            assert task is original_task
            assert type(skills[0]) is type(skill)
            assert skills[0].content == skill.content
            assert skills[0] is not skill
            return await super()._run(task=task, reward_data=reward_data, skills=skills, workspace=workspace)

    original_task = task
    env = ObjectEnvironment(Options())
    result = await env.rollout(
        task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1)
    )
    assert result.trajectory[0]["task"]["input_file"] == "first.csv"
    with pytest.raises(ValueError, match="inputs changed"):
        (
            await env.rollout(
                task.model_copy(update={"input_file": "second.csv"}),
                tmp_path,
                reward_data=RewardData(),
                skill_set=SkillSet({"0": skill}, max_concurrency=1),
            )
        )


async def test_timing_is_measured_and_preserved_on_resume(tmp_path):
    """Audit actual execution once and retain its timestamps on cache reads."""
    env = ExampleEnvironment(Options())
    with patch("mindmemos_skill.envs.base.perf_counter", side_effect=[10.0, 12.5]):
        result = await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData())
    assert result.duration_s == 2.5
    assert result.started_at.utcoffset().total_seconds() == 0
    assert result.finished_at >= result.started_at
    with patch("mindmemos_skill.envs.base.perf_counter", side_effect=AssertionError("Unexpected execution")):
        cached = await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData())
    assert cached == result
    assert env.calls == 1


@pytest.mark.parametrize("typed", [False, True])
async def test_reward_and_usage_extensions_survive_resume(tmp_path, typed):
    """Keep nested extension data and optionally restore concrete model types."""

    class DetailedReward(Reward):
        passed_cases: int
        """Number of successful test cases."""

    class DetailedUsage(Usage):
        reasoning_tokens: int
        """Provider-reported reasoning token count."""

    class DetailedResult(RolloutResult):
        reward: DetailedReward
        """Environment-specific evaluation schema."""

        usage: DetailedUsage
        """Provider-specific usage schema."""

    class AuditedEnvironment(ExampleEnvironment):
        result_type = DetailedResult if typed else RolloutResult

        async def _run(
            self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path
        ) -> RolloutResult:
            """Add typed and untyped audit extensions to a local execution."""
            result = await super()._run(task=task, reward_data=reward_data, skills=skills, workspace=workspace)
            result.reward = DetailedReward(
                score=0.75,
                success=False,
                detail='{"failed": [4]}',
                passed_cases=3,
                breakdown={"accuracy": 0.75},
            )
            result.usage = DetailedUsage(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                reasoning_tokens=10,
                by_model={"test-model": {"calls": 2}},
            )
            return result

    env = AuditedEnvironment(Options())
    result = await env.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData())
    restored = AuditedEnvironment(Options())
    cached = await restored.rollout(Task(id="task-001", instruction="task"), tmp_path, reward_data=RewardData())
    assert restored.calls == 0
    assert cached == result
    assert cached.reward.model_dump()["passed_cases"] == 3
    assert cached.reward.model_dump()["breakdown"] == {"accuracy": 0.75}
    assert cached.usage.model_dump()["reasoning_tokens"] == 10
    assert cached.usage.model_dump()["by_model"] == {"test-model": {"calls": 2}}
    assert cached.reward.detail == '{"failed": [4]}'
    assert cached.usage.total_tokens == 120
    if typed:
        assert isinstance(cached.reward, DetailedReward)
        assert isinstance(cached.usage, DetailedUsage)


def test_audit_validation_and_defaults():
    """Reject invalid counters, partial timing, and non-JSON extension values."""
    from datetime import datetime, timezone

    assert Usage().model_dump() == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        with pytest.raises(ValidationError):
            Usage.model_validate({field: -1})
    with pytest.raises(ValidationError):
        Reward(score=1, success=True, detail="Passed", unsupported=object())
    with pytest.raises(ValidationError, match="Timing fields"):
        RolloutResult(
            output=None,
            trajectory=[],
            reward=Reward(score=1, success=True, detail="Passed"),
            started_at=datetime.now(timezone.utc),
        )


async def test_private_reward_data_is_audited_and_invalidates_resume(tmp_path):
    """Persist private scoring inputs outside the workspace and check them on resume."""
    import json

    class AnswerData(RewardData):
        answer: str
        """Expected answer used only by the evaluator."""

    private = AnswerData(answer="private-reference")

    class ScoringEnvironment(ExampleEnvironment):
        async def _run(
            self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path
        ) -> RolloutResult:
            """Receive the original private scoring object without copying it to workspace."""
            assert reward_data is private
            return await super()._run(task=task, reward_data=reward_data, skills=skills, workspace=workspace)

    env = ScoringEnvironment(Options())
    task = Task(id="private-test", instruction="Solve this task")
    result = await env.rollout(task, tmp_path, reward_data=private)
    request = json.loads((tmp_path / "audit/request.json").read_text())
    assert request["reward_data"]["data"] == {"answer": "private-reference"}
    assert "private-reference" not in (tmp_path / "workspace/answer.txt").read_text()
    assert (await env.rollout(task, tmp_path, reward_data=AnswerData(answer="private-reference"))) == result
    with pytest.raises(ValueError, match="inputs changed"):
        (await env.rollout(task, tmp_path, reward_data=AnswerData(answer="changed-reference")))
    assert env.calls == 1
