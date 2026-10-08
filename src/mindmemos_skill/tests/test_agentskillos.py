"""End-to-end execution with actual files and the unchanged official scorer."""

import asyncio
import json
from hashlib import sha256
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
from mindmemos_skill.persistence.constants import SKILLS_TABLE
from mindmemos_skill.persistence.db_services.skills.typing import SkillRecord
from mindmemos_skill.skill import Skill, SkillContent, SkillSet

SOURCE = Path(__file__).resolve().parents[3] / "data/agentskillos/source"


@pytest.fixture
def config():
    if not (SOURCE / "benchmark/AgentSkillOS_bench/orchestrator.py").exists():
        pytest.skip("Run scripts/agentskillos/download.py to test the official evaluator")
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
        file_sha256={"input.txt": sha256(source.read_bytes()).hexdigest()},
    )
    private = AgentSkillOSRewardData(
        task_config=record,
        task_sha256="fixture",
        core_gt_ids=["alpha"],
        relevance_source="official_task_skills",
        relevance_sha256="fixture",
    )
    return task, private


def skill_set(name="alpha"):
    skill = Skill(
        source=SkillContent(
            content=f"---\nname: {name}\n---\nDo the task.",
            resources={"scripts/helper.txt": b"AUXILIARY"},
            name="test-skill",
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

    def run(self, *, system, user, workspace):
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
def test_task_success_not_retrieval(config, tmp_path, artifact, name, expected, recall):
    task, private = sample(tmp_path)
    agent = Executor(config.agent, artifact)
    env = AgentSkillOSEnvironment(config, agent=agent)
    result = asyncio.run(env.rollout(task, tmp_path / "run", reward_data=private, skill_set=skill_set(name)))
    assert result.reward.score == expected
    assert result.reward.success is bool(expected)
    detail = json.loads(result.reward.detail)
    assert detail["retrieval"]["metrics"]["Recall@10"] == recall
    assert detail["task_evaluation"]["passed"] is bool(expected)
    assert result.usage.total_tokens == 14
    assert result.trajectory == [{"role": "assistant", "content": "Done"}]
    assert (agent.workspaces[0] / f".agents/skills/{name}/scripts/helper.txt").read_bytes() == b"AUXILIARY"


def test_resume_fresh_attempt_and_input_change(config, tmp_path):
    task, private = sample(tmp_path)
    agent = Executor(config.agent, {"answer": 42})
    env = AgentSkillOSEnvironment(config, agent=agent)
    directory = tmp_path / "run"
    first = asyncio.run(env.rollout(task, directory, reward_data=private))
    assert asyncio.run(env.rollout(task, directory, reward_data=private)) == first
    assert agent.calls == 1
    agent.output = None
    second = asyncio.run(env.rollout(task, directory, reward_data=private, resume=False))
    assert second.reward.score == 0
    assert first.output["artifact_directory"] != second.output["artifact_directory"]
    assert (agent.workspaces[0] / "workspace/answer.json").exists()
    (directory / "audit/result.json").unlink()
    assert asyncio.run(env.rollout(task, directory, reward_data=private)).reward.score == 0
    task.files["input.txt"].write_text("MUTATED")
    with pytest.raises(ValueError, match="Public input changed"):
        asyncio.run(env.rollout(task, directory, reward_data=private, resume=False))
    assert not (directory / "audit/result.json").exists()


def test_transport_and_evaluator_errors_are_not_task_failures(config, tmp_path):
    task, private = sample(tmp_path)
    env = AgentSkillOSEnvironment(config, agent=Executor(config.agent, fail=True))
    with pytest.raises(RuntimeError, match="transport"):
        asyncio.run(env.rollout(task, tmp_path / "run", reward_data=private))
    assert not (tmp_path / "run/audit/result.json").exists()
    bad = private.model_copy(deep=True)
    bad.task_config["evaluators"][0]["op_func"] = "nonexistent"
    agent = Executor(config.agent)
    env = AgentSkillOSEnvironment(config, agent=agent)
    with pytest.raises(RuntimeError, match="Unsupported evaluator"):
        asyncio.run(env.rollout(task, tmp_path / "bad", reward_data=bad))
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


def test_wrong_source_hash_and_legacy_config_rejected(config):
    with pytest.raises(ValueError, match="fingerprint"):
        AgentSkillOSConfig(agent=config.agent, benchmark_root=SOURCE, evaluator_sha256="wrong")
    with pytest.raises(ValueError):
        AgentSkillOSConfig(agent=config.agent, benchmark_root=SOURCE, top_n=50)


def test_scoring_cannot_count_skill_resources_as_outputs(config, tmp_path):
    task, private = sample(tmp_path)
    selected = skill_set()
    source = selected.skills["database-id"]
    selected = SkillSet(
        {"id": source.with_resources({"answer.json": b'{"answer":42}'})},
        max_concurrency=1,
    )
    env = AgentSkillOSEnvironment(config, agent=Executor(config.agent))
    result = asyncio.run(env.rollout(task, tmp_path / "run", reward_data=private, skill_set=selected))
    assert result.reward.score == 0


def test_official_sixty_percent_threshold_and_summary(config, tmp_path):
    from mindmemos_skill.envs.builtin_envs.agentskillos.reward import summarize_results

    task, private = sample(tmp_path)
    private.task_config["evaluators"][0]["weight"] = 6
    private.task_config["evaluators"][1]["weight"] = 4
    env = AgentSkillOSEnvironment(config, agent=Executor(config.agent, {"wrong": 42}))
    result = asyncio.run(env.rollout(task, tmp_path / "run", reward_data=private))
    assert result.reward.score == 1
    assert json.loads(result.reward.detail)["task_evaluation"]["total_score"] == 60
    summary = summarize_results({"example": result}, {"other": "Infrastructure failure"})
    assert summary["task_success_rate"] == 1
    assert summary["scoring_coverage"] == 0.5
    assert summary["status"] == "partial"


def test_real_codex_adapter_and_official_scoring(config, tmp_path):
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
        agent=CodexConfig(model="offline-test", cli_path=str(cli), api_key="test"),
        benchmark_root=SOURCE,
    )
    result = asyncio.run(AgentSkillOSEnvironment(settings).rollout(task, tmp_path / "run", reward_data=private))
    assert result.reward.score == 1
    assert result.usage.total_tokens == 7
    assert result.output["response"] == "Done"


def test_evaluator_exception_is_not_a_scored_zero(config, tmp_path):
    task, private = sample(tmp_path)
    del private.task_config["evaluators"][1]["value"]
    env = AgentSkillOSEnvironment(config, agent=Executor(config.agent, {"answer": 42}))
    with pytest.raises(RuntimeError, match="Evaluator error"):
        asyncio.run(env.rollout(task, tmp_path / "run", reward_data=private))
    assert not (tmp_path / "run/audit/result.json").exists()


def test_skill_manage_to_execution_to_official_reward(config, tmp_path):
    from mindmemos_skill.algos.manage.skill_manage import SkillManageAlgorithm, SkillManageConfig, SkillManageSkillSet
    from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
    from mindmemos_skill.infra.embedding import OpenAIEmbeddingConfig
    from mindmemos_skill.persistence import SkillVersionRecord, create_persistence, ensure_collection
    from mindmemos_skill.persistence.db_services.skill_management.writer import SkillGraphStore

    class Embeddings:
        def __init__(self, settings):
            self.config = settings
            self.calls = []

        def embed(self, texts):
            self.calls.extend(texts)
            return [[1.0, 0.0] for _ in texts]

    task, private = sample(tmp_path)
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'graph.db'}"))
    try:
        persistence = create_persistence(database)
        ensure_collection(persistence)
        skills = skill_set().skills
        for name, source in skills.items():
            persistence.skills.add_skill(
                source,
                SkillVersionRecord(
                    project_id="eval",
                    skill_id=name,
                    version_id=f"v-{name}",
                    **source.snapshot().model_dump(),
                ),
            )
            database.insert(
                SKILLS_TABLE,
                [SkillRecord(project_id="eval", skill_id=name, name=name, active_version_id=f"v-{name}").model_dump()],
            )
        store = SkillGraphStore(
            database,
            project_id="eval",
            graph_id="eval",
            actor_id="test",
            version_ids={name: f"v-{name}" for name in skills},
        )
        settings = SkillManageConfig(
            embedding=OpenAIEmbeddingConfig(model="test", base_url="http://test.invalid/v1", dimensions=2),
            build_scene=False,
            llm_linking=False,
            query_rewrite=False,
            mode="skill_only",
            seed_top_k=5,
        )
        embeddings = Embeddings(settings.embedding)
        algorithm = SkillManageAlgorithm(settings, store=store, embedding_model=embeddings)
        selection = SkillManageSkillSet(skills, algorithm=algorithm, limit=5, max_concurrency=1)
        env = AgentSkillOSEnvironment(config, agent=Executor(config.agent, {"answer": 42}))
        result = asyncio.run(env.rollout(task, tmp_path / "run", reward_data=private, skill_set=selection))
        assert result.reward.score == 1
        assert json.loads(result.reward.detail)["retrieval"]["ranked_skill_ids"] == ["alpha"]
        assert embeddings.calls[-1] == task.instruction
    finally:
        database.close()


def test_batch_cli_actual_official_task_and_resume(config, tmp_path):
    import subprocess
    import sys

    cli = tmp_path / "codex-stub"
    cli.write_text(
        f"#!{sys.executable}\nimport json, sys\nsys.stdin.read()\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'Done'}}))\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':5,'output_tokens':2}}))\n"
    )
    cli.chmod(0o755)
    settings = tmp_path / "config.json"
    settings.write_text(json.dumps({"benchmark_root": str(SOURCE), "agent": {"model": "test", "cli_path": str(cli)}}))
    split = tmp_path / "split.json"
    split.write_text('{"test":["data_computation_task3"]}')
    output = tmp_path / "batch"
    script = Path(__file__).resolve().parents[3] / "scripts/agentskillos/run.py"
    command = [
        sys.executable,
        str(script),
        "--config",
        str(settings),
        "--output",
        str(output),
        "--selection",
        "none",
        "--split-file",
        str(split),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    first = json.loads((output / "summary.json").read_text())
    assert first["status"] == "complete"
    assert first["task_success_rate"] == 0
    assert first["scoring_coverage"] == 1
    subprocess.run([*command, "--resume"], check=True, capture_output=True, text=True)
    assert json.loads((output / "summary.json").read_text()) == first
    assert len(list((output / "task-0000/workspace").glob("attempt-*"))) == 1
