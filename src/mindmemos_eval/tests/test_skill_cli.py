"""Verify new Skill command scope and the unchanged Memory parser."""

import pytest
from mindmemos_eval.cli import build_arg_parser, main


def test_defaults_and_report_commands():
    parser = build_arg_parser()
    args = parser.parse_args(["skill", "run"])
    assert args.benchmark == "livemath" and args.algorithm == "skill_forest"
    assert args.test_repeats == 1 and args.arms == "no_skill,initial_skill,trained_skill"
    report = parser.parse_args(["skill", "report", "output/one"])
    assert str(report.run_dir) == "output/one"


@pytest.mark.parametrize(
    "command", ["local", "spreadsheetbench", "check", "status", "resume", "compare", "init", "data"]
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
    assert main(["skill", "run", "--test-repeats", "0", "--output-dir", str(tmp_path / "run")]) == 1
    assert "test_repeats" in capsys.readouterr().out
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("environment", ["spreadsheetbench_text_action", "livemath", "searchqa"])
def test_treeskill_accepts_environment(environment):
    args = build_arg_parser().parse_args(["skill", "treeskill", "--env", environment, "--check"])
    assert args.env == environment
