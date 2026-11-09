"""Exercise ranked-Skill execution without calling an external model."""

import json

import yaml
from mindmemos_eval.cli import main
from mindmemos_eval.skills import skill_graph
from mindmemos_skill.agents.buitin_agents.claude.agent import ClaudeAgent
from mindmemos_skill.common.json import write_json
from mindmemos_skill.envs.builtin_envs.agentskillos import AgentSkillOSEnvironment


def test_accuracy_runs_native_tasks_and_keeps_errors_separate(tmp_path, monkeypatch):
    """Dispatch Claude with the requested Skills and count official passes only."""
    source = tmp_path / "official"
    suite = source / "benchmark/AgentSkillOS_bench"
    tasks = suite / "tasks"
    tasks.mkdir(parents=True)
    (suite / "orchestrator.py").touch()
    for task_id in ("pass", "fail", "error"):
        task = {
            "task_id": task_id,
            "prompt": f"Write {task_id}.txt",
            "outputs": [f"{task_id}.txt"],
            "skills": ["private-reference"],
            "evaluators": [{"id": "private-evaluator"}],
        }
        (tasks / f"{task_id}.json").write_text(json.dumps(task))

    skills_dir = tmp_path / "skills"
    selected = skills_dir / "selected"
    selected.mkdir(parents=True)
    (selected / "SKILL.md").write_text("---\nname: selected\ndescription: Example Skill\n---\nWrite an answer.\n")
    (selected / "resource.txt").write_text("Example resource")

    async def retrieve(queries, skills, service_config, output, top_k):
        assert queries == {"pass": "Write pass.txt", "fail": "Write fail.txt", "error": "Write error.txt"}
        assert list(skills) == ["selected"]
        rankings = {"pass": ["selected", "outside-top-k"], "fail": [], "error": []}
        output.mkdir()
        write_json(output / "rankings.json", rankings)
        return rankings

    async def run_cli(self, command, workspace, overrides, prompt):
        assert command[command.index("--model") + 1] == "MiniMax-M2.7"
        assert "private-reference" not in prompt
        assert "private-evaluator" not in prompt
        if "Write error.txt" in prompt:
            raise RuntimeError("Fixture execution failure")
        if "Write pass.txt" in prompt:
            assert ".claude/skills/selected/SKILL.md" in command[command.index("--system-prompt") + 1]
            assert (workspace / ".claude/skills/selected/resource.txt").read_text() == "Example resource"
            (workspace / "workspace/pass.txt").write_text("Pass")
        else:
            assert not (workspace / ".claude/skills").exists()
        return json.dumps({"type": "result", "subtype": "success", "result": "Completed"})

    async def score(self, task_path, output_dir, report_path, *, preflight=False):
        task_id = json.loads(task_path.read_text())["task_id"]
        if not preflight:
            report_path.write_text(
                json.dumps({"task_id": task_id, "passed": (output_dir / f"{task_id}.txt").is_file()})
            )

    monkeypatch.setattr(ClaudeAgent, "_run_cli", run_cli)
    monkeypatch.setattr(AgentSkillOSEnvironment, "_score", score)
    monkeypatch.setattr(skill_graph, "_retrieve", retrieve)
    config = tmp_path / "dev.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(source),
                "agent": {
                    "class_name": "claude",
                    "model": "MiniMax-M2.7",
                    "base_url": "http://localhost",
                    "api_key": "test",
                },
                "execution_top_k": 1,
            }
        )
    )
    output = tmp_path / "run"
    args = [
        "skill",
        "evaluate",
        "manage",
        "--algo",
        "skill-graph",
        "--eval-mode",
        "accuracy",
        "--config",
        str(config),
        "--skills-dir",
        str(skills_dir),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 1
    assert json.loads((output / "rankings.json").read_text())["pass"] == ["selected", "outside-top-k"]
    assert json.loads((output / "selected_skills.json").read_text())["pass"] == ["selected"]
    report = json.loads((output / "summary.json").read_text())
    assert report["model"] == "MiniMax-M2.7"
    assert report["accuracy"] == 0.5
    assert report["scoring_coverage"] == 2 / 3
    assert report["errors"] == {"error": "RuntimeError: Fixture execution failure"}
