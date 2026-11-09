"""Verify new Skill command scope and the unchanged Memory parser."""

from types import SimpleNamespace

import pytest
import yaml
from mindmemos_eval.cli import build_arg_parser, main
from mindmemos_eval.skills import runner, skill_graph
from mindmemos_skill.envs import Environment
from mindmemos_skill.pipelines import Pipeline


def test_rebuild_report_accepts_saved_run_directory():
    report = build_arg_parser().parse_args(["skill", "evaluate", "rebuild-report", "output/one"])
    assert str(report.run_dir) == "output/one"


@pytest.mark.parametrize(
    "command",
    [
        "envs",
        "run",
        "treeskill",
        "evolve",
        "skill-graph",
        "test",
        "report",
        "local",
        "spreadsheetbench",
        "check",
        "status",
        "resume",
        "compare",
        "init",
        "data",
    ],
)
def test_removed_skill_commands_are_rejected(command):
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["skill", command])


def test_memory_parser_keeps_existing_arguments():
    args = build_arg_parser().parse_args(
        [
            "memory",
            "--benchmark-config",
            "one.yaml",
            "--manifest-output",
            "one.jsonl",
            "--benchmark-list",
            "locomo",
            "--algorithm",
            "vanilla",
        ]
    )
    assert args.command == "memory" and args.algorithm == "vanilla"


def test_invalid_run_options_fail_before_execution(tmp_path, capsys):
    args = ["skill", "evaluate", "train", "--algo", "skill-forest", "--test-repeats", "0"]
    assert main([*args, "--output-dir", str(tmp_path / "run")]) == 1
    assert "test_repeats" in capsys.readouterr().out
    assert not (tmp_path / "run").exists()


def test_skill_forest_benchmark_requires_a_dataset(capsys):
    parser = build_arg_parser()
    args = ["skill", "evaluate", "train", "--algo", "skill-forest", "--benchmark"]
    assert parser.parse_args([*args, "officeqa"]).benchmark == "officeqa"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([*args, "all"])
    assert exc.value.code == 2
    assert "invalid choice: 'all'" in capsys.readouterr().err


@pytest.mark.parametrize("environment", ["spreadsheetbench_text_action", "livemath", "searchqa"])
def test_treeskill_accepts_environment(environment):
    args = build_arg_parser().parse_args(
        ["skill", "evaluate", "train", "--algo", "tree-skill", "--env", environment, "--check"]
    )
    assert args.env == environment


def test_show_reads_the_environment_and_pipeline_registries(capsys):
    assert main(["skill", "show", "envs"]) == 0
    assert capsys.readouterr().out.splitlines() == list(Environment.registered_types())

    assert main(["skill", "show", "pipelines"]) == 0
    pipelines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in pipelines] == list(Pipeline.registered_types())
    assert "training (evaluate train)" in pipelines


def test_evaluate_train_rejects_manage_algorithm():
    with pytest.raises(SystemExit) as exc:
        main(["skill", "evaluate", "train", "--algo", "skill-graph"])
    assert exc.value.code == 2


@pytest.mark.parametrize("override_output", [False, True])
def test_forest_uses_registered_environment_and_resolved_output(tmp_path, monkeypatch, override_output):
    yaml_output = tmp_path / "from-yaml"
    cli_output = tmp_path / "from-cli"
    config_file = tmp_path / "forest.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "models": {"target": {"model": "offline"}, "optimizer": {"model": "offline"}},
                "skill_forest": {"num_trees": 1, "expansion_budget": 1},
                "output_dir": str(yaml_output),
            }
        )
    )

    async def run_forest(config, options):
        assert options.benchmark == "alfworld"
        assert options.arms == ["no_skill"]
        assert options.output_dir == (cli_output if override_output else yaml_output)
        options.output_dir.mkdir()
        (options.output_dir / "summary.json").write_text('{"incomplete": false}')
        return SimpleNamespace(output_dir=options.output_dir)

    monkeypatch.setattr(runner, "run", run_forest)
    args = [
        "skill",
        "evaluate",
        "train",
        "--algo",
        "skill-forest",
        "--env",
        "alfworld_bounded_history",
        "--config",
        str(config_file),
        "--arms",
        "no_skill",
    ]
    if override_output:
        args.extend(["--output-dir", str(cli_output)])
    assert main(args) == 0


def test_graph_uses_yaml_and_accepts_cli_mode_and_output_overrides(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    skills = tmp_path / "skills"
    service = tmp_path / "service.yaml"
    yaml_output = tmp_path / "retrieval"
    cli_output = tmp_path / "accuracy"
    config_file = tmp_path / "graph.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "eval_mode": "retrieval",
                "tasks_dir": str(tasks),
                "skills_dir": str(skills),
                "service_config": str(service),
                "output_dir": str(yaml_output),
                "benchmark_root": str(tmp_path / "benchmark"),
                "agent": {"class_name": "claude", "model": "offline"},
            }
        )
    )
    calls = []

    async def run_retrieval(tasks_dir, skills_dir, service_config, output):
        calls.append(("retrieval", output))
        assert (tasks_dir, skills_dir, service_config) == (tasks, skills, service)

    async def run_accuracy(config, skills_dir, service_config, output):
        calls.append(("accuracy", output))
        assert config.benchmark_root == tmp_path / "benchmark"
        assert (skills_dir, service_config) == (skills, service)
        return SimpleNamespace(errors={"task": "execution failed"})

    monkeypatch.setattr(skill_graph, "run_retrieval", run_retrieval)
    monkeypatch.setattr(skill_graph, "run_accuracy", run_accuracy)
    args = ["skill", "evaluate", "manage", "--algo", "skill-graph", "--config", str(config_file)]
    assert main(args) == 0
    assert main([*args, "--eval-mode", "accuracy", "--output-dir", str(cli_output)]) == 1
    assert calls == [("retrieval", yaml_output), ("accuracy", cli_output)]
