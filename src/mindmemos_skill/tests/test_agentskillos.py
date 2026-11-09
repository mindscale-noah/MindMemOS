"""End-to-end execution with actual files and the unchanged official scorer."""

import json
from pathlib import Path

import pytest
from mindmemos_skill.agents import AgentResult
from mindmemos_skill.agents.buitin_agents.codex.config import CodexConfig
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.agentskillos import (
    AgentSkillOSConfig,
    AgentSkillOSData,
    AgentSkillOSDataConfig,
    AgentSkillOSEnvironment,
    AgentSkillOSRewardData,
    AgentSkillOSTask,
)
from mindmemos_skill.envs.data import Data
from mindmemos_skill.skill import Skill, SkillContent, SkillSet

SOURCE = Path(__file__).resolve().parents[3] / "data/mindmemos_skill/agentskillos"


@pytest.fixture
def config():
    if not (SOURCE / "benchmark/AgentSkillOS_bench/orchestrator.py").exists():
        pytest.skip(
            "Run scripts/download_datasets.py download skill --name agentskillos to test the official evaluator"
        )
    for module in ("pptx", "docx", "fitz", "jsonschema", "pdfplumber", "openpyxl"):
        pytest.importorskip(module)
    return AgentSkillOSConfig(agent=CodexConfig(model="offline-test", api_key="test"), benchmark_root=SOURCE)


def sample(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("PUBLIC_INPUT")
    record = {
        "task_id": "example",
        "prompt": "Read input.txt and create answer.json.",
        "outputs": ["answer.json"],
        "skills": ["alpha"],
        "evaluators": [
            {
                "id": "exists",
                "type": "objective_usability",
                "op_func": "file_exists",
                "weight": 4,
                "op_args": {"path": "answer.json"},
            },
            {
                "id": "correct",
                "type": "objective_usability",
                "op_func": "json_has_keys",
                "weight": 6,
                "op_args": {"path": "answer.json"},
                "value": ["answer"],
                "depends_on": ["exists"],
            },
        ],
    }
    task = AgentSkillOSTask(
        id="example",
        instruction=record["prompt"],
        outputs=record["outputs"],
        files={"input.txt": source},
    )
    private = AgentSkillOSRewardData(
        task_config=record,
        core_gt_ids=["alpha"],
        relevance_source="official_task_skills",
    )
    return task, private


def skill_set(name="alpha"):
    skill = Skill(
        source=SkillContent(
            content=f"---\nname: {name}\n---\nDo the task.",
            resources={"scripts/helper.txt": b"AUXILIARY"},
            name=name,
            description="Task instructions",
            standard_format=False,
        )
    )
    return SkillSet({"database-id": skill}, max_concurrency=1)


class Executor:
    def __init__(self, config, output=None, fail=False):
        self.config = config
        self.output = output
        self.fail = fail
        self.calls = 0
        self.workspaces = []

    async def run(self, *, system, user, workspace):
        self.calls += 1
        self.workspaces.append(workspace)
        assert "evaluators" not in user and "core_gt_ids" not in user
        assert (workspace / "input.txt").read_text() == "PUBLIC_INPUT"
        assert not list(workspace.rglob("official-task-*.json"))
        if self.fail:
            raise RuntimeError("Execution transport failed")
        if self.output is not None:
            (workspace / "workspace/answer.json").write_text(json.dumps(self.output))
        return AgentResult(
            output="Done",
            trajectory=[{"role": "assistant", "content": "Done"}],
            usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )


@pytest.mark.parametrize(
    "artifact,name,expected,recall",
    [
        ({"answer": 42}, "unrelated", 1, 0),
        (None, "alpha", 0, 1),
        ({"wrong": 42}, "alpha", 0, 1),
    ],
)
async def test_task_success_not_retrieval(config, tmp_path, artifact, name, expected, recall):
    task, private = sample(tmp_path)
    agent = Executor(config.agent, artifact)
    env = AgentSkillOSEnvironment(config, agent=agent)
    result = await env.rollout(task, tmp_path / "run", reward_data=private, skill_set=skill_set(name))
    assert result.reward.score == expected
    assert result.reward.success is bool(expected)
    detail = json.loads(result.reward.detail)
    assert detail["retrieval"]["metrics"]["Recall@10"] == recall
    assert detail["task_evaluation"]["passed"] is bool(expected)
    assert result.usage.total_tokens == 14
    assert result.trajectory == [{"role": "assistant", "content": "Done"}]
    assert (agent.workspaces[0] / f".agents/skills/{name}/scripts/helper.txt").read_bytes() == b"AUXILIARY"


async def test_resume_fresh_attempt_and_input_change(config, tmp_path):
    task, private = sample(tmp_path)
    agent = Executor(config.agent, {"answer": 42})
    env = AgentSkillOSEnvironment(config, agent=agent)
    directory = tmp_path / "run"
    first = await env.rollout(task, directory, reward_data=private)
    assert (await env.rollout(task, directory, reward_data=private)) == first
    assert agent.calls == 1
    agent.output = None
    second = await env.rollout(task, directory, reward_data=private, resume=False)
    assert second.reward.score == 0
    assert first.output["artifact_directory"] != second.output["artifact_directory"]
    assert (agent.workspaces[0] / "workspace/answer.json").exists()
    (directory / "audit/result.json").unlink()
    assert (await env.rollout(task, directory, reward_data=private)).reward.score == 0
    result_path = directory / "audit/result.json"
    saved = result_path.read_bytes()
    task.files["input.txt"].write_text("MUTATED")
    with pytest.raises(ValueError, match="inputs changed"):
        (await env.rollout(task, directory, reward_data=private, resume=False))
    assert result_path.read_bytes() == saved
    assert agent.calls == 3


async def test_transport_and_evaluator_errors_are_not_task_failures(config, tmp_path):
    task, private = sample(tmp_path)
    env = AgentSkillOSEnvironment(config, agent=Executor(config.agent, fail=True))
    with pytest.raises(RuntimeError, match="transport"):
        (await env.rollout(task, tmp_path / "run", reward_data=private))
    assert not (tmp_path / "run/audit/result.json").exists()
    bad = private.model_copy(deep=True)
    bad.task_config["evaluators"][0]["op_func"] = "nonexistent"
    agent = Executor(config.agent)
    env = AgentSkillOSEnvironment(config, agent=agent)
    with pytest.raises(RuntimeError, match="Unsupported evaluator"):
        (await env.rollout(task, tmp_path / "bad", reward_data=bad))
    assert agent.calls == 0


def test_official_release_data_and_registration(config):
    assert Data.resolve("agentskillos") is AgentSkillOSData
    assert Environment.resolve("agentskillos") is AgentSkillOSEnvironment
    samples = AgentSkillOSData(AgentSkillOSDataConfig(benchmark_root=SOURCE)).load()
    assert len(samples) == len({sample.task.id for sample in samples}) == 30
    assert sum(bool(sample.task.files) for sample in samples) == 5
    by_id = {sample.task.id: sample for sample in samples}
    assert "artifacts/penguins.csv" in by_id["data_computation_task1"].task.files
    assert "ml_project/train_baseline.py" in by_id["data_computation_task4"].task.files
    assert not by_id["data_computation_task5"].task.files
    for sample in samples:
        assert "evaluators" not in sample.task.model_dump()
        assert "skills" not in sample.task.model_dump()
        assert "core_gt_ids" not in sample.task.model_dump()
        assert sample.reward_data.relevance_source == "official_task_skills"


def test_removed_source_hash_and_legacy_config_rejected(config):
    with pytest.raises(ValueError, match="Extra inputs"):
        AgentSkillOSConfig(agent=config.agent, benchmark_root=SOURCE, evaluator_sha256="wrong")
    with pytest.raises(ValueError):
        AgentSkillOSConfig(agent=config.agent, benchmark_root=SOURCE, top_n=50)


async def test_scoring_cannot_count_skill_resources_as_outputs(config, tmp_path):
    task, private = sample(tmp_path)
    selected = skill_set()
    source = selected.skills["database-id"]
    selected = SkillSet(
        {"id": source.with_resources({"answer.json": b'{"answer":42}'})},
        max_concurrency=1,
    )
    env = AgentSkillOSEnvironment(config, agent=Executor(config.agent))
    result = await env.rollout(task, tmp_path / "run", reward_data=private, skill_set=selected)
    assert result.reward.score == 0


async def test_official_sixty_percent_threshold_and_summary(config, tmp_path):
    from mindmemos_skill.envs.builtin_envs.agentskillos.reward import summarize_results

    task, private = sample(tmp_path)
    private.task_config["evaluators"][0]["weight"] = 6
    private.task_config["evaluators"][1]["weight"] = 4
    env = AgentSkillOSEnvironment(config, agent=Executor(config.agent, {"wrong": 42}))
    result = await env.rollout(task, tmp_path / "run", reward_data=private)
    assert result.reward.score == 1
    assert json.loads(result.reward.detail)["task_evaluation"]["total_score"] == 60
    summary = summarize_results({"example": result}, {"other": "Infrastructure failure"})
    assert summary["task_success_rate"] == 1
    assert summary["scoring_coverage"] == 0.5
    assert summary["status"] == "partial"


async def test_real_codex_adapter_and_official_scoring(config, tmp_path):
    import sys

    task, private = sample(tmp_path)
    cli = tmp_path / "codex-stub"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        "prompt = sys.stdin.read()\n"
        "assert 'Read input.txt' in prompt\n"
        "Path('workspace/answer.json').write_text('{\"answer\":42}')\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'Done'}}))\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':5,'output_tokens':2}}))\n"
    )
    cli.chmod(0o755)
    settings = AgentSkillOSConfig(
        agent=CodexConfig(model="offline-test", cli_path=str(cli), api_key="test", base_url="http://localhost"),
        benchmark_root=SOURCE,
    )
    result = await AgentSkillOSEnvironment(settings).rollout(task, tmp_path / "run", reward_data=private)
    assert result.reward.score == 1
    assert result.usage.total_tokens == 7
    assert result.output["response"] == "Done"


async def test_evaluator_exception_is_not_a_scored_zero(config, tmp_path):
    task, private = sample(tmp_path)
    del private.task_config["evaluators"][1]["value"]
    env = AgentSkillOSEnvironment(config, agent=Executor(config.agent, {"answer": 42}))
    with pytest.raises(RuntimeError, match="Evaluator error"):
        (await env.rollout(task, tmp_path / "run", reward_data=private))
    assert not (tmp_path / "run/audit/result.json").exists()
