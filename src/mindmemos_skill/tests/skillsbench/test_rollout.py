"""Run the full shared rollout and external-command boundary with a substituted Harbor CLI."""

import json
import subprocess
from pathlib import Path

import pytest
from mindmemos_skill.envs.builtin_envs.skillsbench import SkillsBenchConfig, SkillsBenchEnvironment
from mindmemos_skill.envs.builtin_envs.skillsbench.rollout import _read_result
from mindmemos_skill.skill import Skill, SkillContent, SkillSet


def candidate():
    return Skill(
        SkillContent(
            content="---\nname: replacement\ndescription: Replacement workflow\n---\n\nInspect inputs.\n",
            resources={"assets/example.bin": b"\x00\xff"},
        )
    )


@pytest.mark.parametrize("agent", ["oracle", "claude-code", "codex"])
async def test_native_cli_result_and_base_resume_do_not_mutate_sources(tmp_path, sample, harbor, agent):
    source = {
        path.relative_to(sample.reward_data.task_dir): path.read_bytes()
        for path in sample.reward_data.task_dir.rglob("*")
        if path.is_file()
    }
    config = SkillsBenchConfig(harbor_command=harbor, agent=agent, model=None if agent == "oracle" else "test-model")
    env = SkillsBenchEnvironment(config)
    directory = tmp_path / "run"
    result = await env.rollout(sample.task, directory, reward_data=sample.reward_data)
    assert result.reward.success and result.reward.score == 1
    attempts = list((directory / "audit").glob("harbor-*"))
    assert len(attempts) == 1
    job = json.loads((attempts[0] / "job.json").read_text())
    assert job["agents"][0]["name"] == agent
    assert (attempts[0] / "stdout.log").read_text().strip() == "Native CLI completed"
    staged = attempts[0] / "tasks/demo/environment/skills"
    assert (staged / "original/SKILL.md").is_file() if agent == "oracle" else not list(staged.iterdir())
    assert await env.rollout(sample.task, directory, reward_data=sample.reward_data) == result
    assert len(list((directory / "audit").glob("harbor-*"))) == 1
    assert {
        path.relative_to(sample.reward_data.task_dir): path.read_bytes()
        for path in sample.reward_data.task_dir.rglob("*")
        if path.is_file()
    } == source
    serialized = json.dumps(result.trajectory)
    assert "PRIVATE REFERENCE" not in serialized and "HIDDEN REASONING" not in serialized
    if agent != "oracle":
        assert "PUBLIC OBSERVATION" in serialized
        assert result.usage.input_tokens == 10 and result.usage.total_tokens == 14
        assert result.usage.model_extra["cache_read_input_tokens"] == 6


async def test_supplied_skills_replace_bundled_skills_and_keep_binary_resources(tmp_path, sample, harbor, monkeypatch):
    monkeypatch.setenv("SKILLSBENCH_TEST_API_KEY", "private-key")
    env = SkillsBenchEnvironment(
        SkillsBenchConfig(harbor_command=harbor, agent="codex", model="model", agent_env=["SKILLSBENCH_TEST_API_KEY"])
    )
    result = await env.rollout(
        sample.task,
        tmp_path / "run",
        reward_data=sample.reward_data,
        skill_set=SkillSet({"candidate": candidate()}),
    )
    job_path = next((tmp_path / "run/audit").glob("harbor-*/job.json"))
    job = json.loads(job_path.read_text())
    assert job["agents"][0]["env"] == {"SKILLSBENCH_TEST_API_KEY": "${SKILLSBENCH_TEST_API_KEY}"}
    assert "private-key" not in job_path.read_text() and "private-key" not in json.dumps(result.model_dump(mode="json"))
    skills = job_path.parent / "tasks/demo/environment/skills"
    assert [path.name for path in skills.iterdir()] == ["replacement"]
    assert (skills / "replacement/assets/example.bin").read_bytes() == b"\x00\xff"


async def test_native_exceptions_are_not_scored_or_cached(tmp_path, sample, harbor):
    script = harbor[-1]
    path = Path(script)
    path.write_text(
        path.read_text().replace(
            '"task_name":task.name,',
            '"exception_info":{"exception_type":"VerifierTimeoutError","exception_message":"grading timed out"},"task_name":task.name,',
        )
    )
    env = SkillsBenchEnvironment(SkillsBenchConfig(harbor_command=harbor))
    with pytest.raises(RuntimeError, match="VerifierTimeoutError"):
        await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    assert not (tmp_path / "run/audit/result.json").exists()
    assert next((tmp_path / "run/audit").glob("harbor-*/stdout.log")).is_file()


async def test_source_change_blocks_cached_result_reuse(tmp_path, sample, harbor):
    env = SkillsBenchEnvironment(SkillsBenchConfig(harbor_command=harbor))
    await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    (sample.reward_data.task_dir / "tests/reference.txt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)


@pytest.mark.parametrize("reward", [None, {}, {"other": 1}, {"reward": 2}])
def test_missing_or_invalid_native_rewards_are_not_zero_scores(tmp_path, reward):
    from mindmemos_skill.envs import Task

    trial = tmp_path / "job/trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"task_name": "demo", "verifier_result": {"rewards": reward}}))
    with pytest.raises(ValueError, match="reward"):
        _read_result(tmp_path / "job", Task(id="demo", instruction="Task"), "oracle", tmp_path)


async def test_zero_reward_is_a_completed_failure_not_an_execution_error(tmp_path, sample, harbor):
    script = Path(harbor[-1])
    script.write_text(script.read_text().replace('"rewards":{"reward":1}', '"rewards":{"reward":0}'))
    env = SkillsBenchEnvironment(SkillsBenchConfig(harbor_command=harbor))
    result = await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    assert result.reward.score == 0 and result.reward.success is False
    assert (tmp_path / "run/audit/result.json").is_file()


async def test_missing_usage_remains_unknown(tmp_path, sample, harbor):
    script = Path(harbor[-1])
    script.write_text(script.read_text().replace('"n_input_tokens":10', '"n_input_tokens":None'))
    env = SkillsBenchEnvironment(SkillsBenchConfig(harbor_command=harbor, agent="codex", model="test-model"))
    result = await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    assert result.usage.model_extra["reported"] is False


async def test_outer_timeout_retains_partial_cli_log_without_a_completion(tmp_path, sample, harbor):
    script = Path(harbor[-1])
    source = script.read_text().replace(
        "assert sys.argv[1:3]", 'print("Started", flush=True)\nimport time\ntime.sleep(10)\nassert sys.argv[1:3]'
    )
    script.write_text(source)
    env = SkillsBenchEnvironment(SkillsBenchConfig(harbor_command=harbor, timeout_s=0.4))
    with pytest.raises(subprocess.TimeoutExpired):
        await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    assert not (tmp_path / "run/audit/result.json").exists()
    assert "Started" in next((tmp_path / "run/audit").glob("harbor-*/stdout.log")).read_text()


async def test_wrong_cli_version_fails_before_any_task_attempt(tmp_path, sample, harbor):
    script = Path(harbor[-1])
    script.write_text(script.read_text().replace('print("0.24.0")', 'print("0.0.1")'))
    env = SkillsBenchEnvironment(SkillsBenchConfig(harbor_command=harbor))
    with pytest.raises(RuntimeError, match="harbor==0.24.0"):
        await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    assert not list((tmp_path / "run/audit").glob("harbor-*"))


async def test_missing_model_trajectory_is_not_replaced_by_synthetic_success(tmp_path, sample, harbor):
    script = Path(harbor[-1])
    script.write_text(script.read_text() + '\n(trial / "agent/trajectory.json").unlink()\n')
    env = SkillsBenchEnvironment(SkillsBenchConfig(harbor_command=harbor, agent="codex", model="test-model"))
    with pytest.raises(FileNotFoundError, match="trajectory.json"):
        await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    assert not (tmp_path / "run/audit/result.json").exists()


async def test_cli_failure_does_not_publish_a_rollout_completion(tmp_path, sample, harbor):
    script = Path(harbor[-1])
    script.write_text(script.read_text() + "\nraise SystemExit(2)\n")
    env = SkillsBenchEnvironment(SkillsBenchConfig(harbor_command=harbor))
    with pytest.raises(RuntimeError, match="status 2"):
        await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    assert not (tmp_path / "run/audit/result.json").exists()


async def test_endpoint_changes_invalidate_cache_but_key_rotation_does_not(tmp_path, sample, harbor, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://first.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "first-key")
    env = SkillsBenchEnvironment(
        SkillsBenchConfig(
            harbor_command=harbor, agent="codex", model="test-model", agent_env=["OPENAI_API_KEY", "OPENAI_BASE_URL"]
        )
    )
    result = await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    monkeypatch.setenv("OPENAI_API_KEY", "rotated-key")
    assert await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data) == result
    monkeypatch.setenv("OPENAI_BASE_URL", "https://different.invalid/v1")
    with pytest.raises(ValueError, match="inputs changed"):
        await env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)
    assert "first-key" not in (tmp_path / "run/audit/request.json").read_text()


def test_runtime_filters_ambient_model_settings_only(harbor, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient")
    monkeypatch.setenv("CODEX_HOME", "/ambient/native/home")
    monkeypatch.setenv("OPENAI_API_KEY", "selected")
    env = SkillsBenchEnvironment(
        SkillsBenchConfig(harbor_command=harbor, agent="codex", model="test-model", agent_env=["OPENAI_API_KEY"])
    )
    runtime = env._execution(10)
    assert "ANTHROPIC_API_KEY" in runtime.excluded_environment and "CODEX_HOME" in runtime.excluded_environment
    assert runtime.environment == {"OPENAI_API_KEY": "selected"}
    assert "PATH" not in runtime.excluded_environment and "DOCKER_HOST" not in runtime.excluded_environment
