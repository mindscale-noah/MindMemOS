from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "mindmemos_skill" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_mindmemos_skill_experiment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)
CONFIG_ROOT = SCRIPT_PATH.parents[2] / "config" / "mindmemos_skill"


def write_config(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def empty_env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    return path


def runner_command(invocation: SCRIPT.ExperimentInvocation) -> tuple[str, list[str]]:
    python_index = invocation.command.index("python")
    return invocation.command[python_index + 1], invocation.command[python_index + 2 :]


def test_build_invocation_selects_method_environment_and_boolean_flags(tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "skill_grpo_with_replay_buffer" / "livemath" / "gpt54mini_epoch4.yaml",
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert invocation.method == "skill_grpo_with_replay_buffer"
    assert invocation.environment == "livemath"
    assert invocation.run_id == "livemath_skill_grpo_with_replay_buffer_20260102-030405"
    assert invocation.output_dir == Path(
        "outputs/livemath/skill_grpo_with_replay_buffer/"
        "livemath_skill_grpo_with_replay_buffer_20260102-030405"
    )
    assert invocation.command[:7] == ["uv", "run", "--package", "mindmemos-skill", "--extra", "llm", "python"]
    runner, arguments = runner_command(invocation)
    assert runner.endswith("scripts/mindmemos_skill/runners/evolve.py")
    assert arguments[:2] == ["--algorithm", "skill_grpo_with_replay_buffer"]
    assert invocation.command[invocation.command.index("--benchmark") + 1] == "livemath"
    assert "--no-fail-fast" in invocation.command
    assert "--shuffle-choices" in invocation.command
    assert "--no-use-theorem" in invocation.command


def test_replay_buffer_does_not_restrict_environment_or_dataset(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "experiment.yaml",
        {
            "version": 1,
            "method": "skill_grpo_with_replay_buffer",
            "environment": "custom-environment",
            "parameters": {
                "dataset": {
                    "dataset_ref": "alfworld_path_split",
                    "dataset_options": {
                        "split_dir": "custom/splits",
                        "alfworld_data": "custom/data",
                    },
                    "initial_skill": "SKILL.md",
                },
                "environment_options": {
                    "env_ref": "alfworld",
                    "env_options": {"seed": 7},
                    "max_turns": 50,
                },
            },
        },
    )

    invocation = SCRIPT.build_invocation(
        config_path,
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert invocation.environment == "custom-environment"
    assert invocation.command[invocation.command.index("--benchmark") + 1] == "custom-environment"
    assert invocation.command[invocation.command.index("--dataset-ref") + 1] == "alfworld_path_split"
    assert invocation.command[invocation.command.index("--env-ref") + 1] == "alfworld"
    assert invocation.command[invocation.command.index("--dataset-options") + 1] == (
        '{"split_dir":"custom/splits","alfworld_data":"custom/data"}'
    )
    assert invocation.command[invocation.command.index("--env-options") + 1] == '{"seed":7}'
    assert invocation.resolved_config["resolved"]["extras"] == ["llm", "alfworld"]


def test_runner_directory_contains_only_the_two_algorithm_family_scripts() -> None:
    runner_dir = SCRIPT_PATH.parent / "runners"

    assert sorted(path.name for path in runner_dir.glob("*.py")) == ["evolve.py", "trace2skill.py"]


def test_experiment_adapters_stay_outside_the_algorithm_support_package() -> None:
    package_experiments = SCRIPT_PATH.parents[2] / "src" / "mindmemos_skill" / "mindmemos_skill" / "experiments"
    script_experiments = SCRIPT_PATH.parent / "experiments"

    assert not package_experiments.exists()
    assert all(spec.implementation_path.parent == script_experiments for spec in SCRIPT.EXPERIMENTS.values())


def test_skill_evaluation_can_switch_to_true_no_skill_mode(tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "skill_evaluation" / "alfworld" / "default.yaml",
        env_file_override=empty_env_file(tmp_path),
        overrides=["dataset.skill=null", "dataset.no_skill=true", "evaluation.test_limit=2"],
        timestamp="20260102-030405",
        base_environment={},
    )

    runner, arguments = runner_command(invocation)
    assert runner.endswith("scripts/mindmemos_skill/runners/evolve.py")
    assert arguments[:2] == ["--algorithm", "skill_evaluation"]
    assert "--no-skill" in invocation.command
    assert "--skill" not in invocation.command
    assert invocation.command[invocation.command.index("--test-limit") + 1] == "2"


@pytest.mark.parametrize(
    "config_path",
    [
        CONFIG_ROOT / "skill_grpo_without_replay_buffer" / "alfworld" / "bounded_history.yaml",
        CONFIG_ROOT / "skill_grpo_with_experience_validation" / "alfworld" / "bounded_history.yaml",
        CONFIG_ROOT / "skill_evaluation" / "alfworld" / "bounded_history.yaml",
    ],
)
def test_bounded_history_alfworld_configs_select_the_registered_env(config_path: Path, tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        config_path,
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert invocation.environment == "alfworld"
    assert invocation.command[invocation.command.index("--env-ref") + 1] == "alfworld_bounded_history"
    if config_path.parent.parent.name == "skill_grpo_without_replay_buffer":
        assert invocation.command[invocation.command.index("--env-seed") + 1] == "42"


def test_trace2skill_routes_to_family_runner_and_keeps_test_config(tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "trajectory_evidence_patch" / "alfworld" / "default.yaml",
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    runner, arguments = runner_command(invocation)
    assert runner.endswith("scripts/mindmemos_skill/runners/trace2skill.py")
    assert arguments[:2] == ["--algorithm", "trajectory_evidence_patch"]
    assert invocation.command[invocation.command.index("--test-rollouts") + 1] == "1"


@pytest.mark.parametrize(
    "config_path",
    sorted(path.relative_to(CONFIG_ROOT) for path in CONFIG_ROOT.glob("*/*/*.yaml")),
)
def test_all_shipped_experiment_configs_build(config_path: Path, tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / config_path,
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert invocation.command[0:4] == ["uv", "run", "--package", "mindmemos-skill"]
    if invocation.run_id is None:
        assert invocation.output_dir == Path("results/agent_skill_eval/20260102-030405")
    else:
        assert invocation.output_dir == Path(
            f"outputs/{invocation.environment}/{invocation.method}/{invocation.run_id}"
        )


@pytest.mark.parametrize(
    ("config_path", "expected"),
    [
        (CONFIG_ROOT / "skill_evaluation" / "alfworld" / "default.yaml", "50"),
        (CONFIG_ROOT / "skill_evaluation" / "livemath" / "default.yaml", "1"),
        (CONFIG_ROOT / "skill_evaluation" / "spreadsheetbench" / "default.yaml", "30"),
    ],
)
def test_configs_use_one_public_turn_limit(config_path: Path, expected: str, tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        config_path,
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert invocation.command[invocation.command.index("--max-turns") + 1] == expected
    assert "--max-steps" not in invocation.command


def test_explicit_run_id_and_output_dir_still_override_defaults(tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "skill_evaluation" / "alfworld" / "default.yaml",
        env_file_override=empty_env_file(tmp_path),
        overrides=["runtime.run_id=custom-run", "runtime.output_dir=custom/{run_id}"],
        timestamp="20260102-030405",
        base_environment={},
    )

    assert invocation.run_id == "custom-run"
    assert invocation.output_dir == Path("custom/custom-run")


def test_overrides_and_environment_expansion_are_applied(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "experiment.yaml",
        {
            "version": 1,
            "method": "initial_skill_evaluation",
            "environment": "alfworld",
            "parameters": {
                "dataset": {
                    "data_root": "$DATA_ROOT",
                    "split_dir": "split",
                    "initial_skill": "SKILL.md",
                },
                "environment_options": {"max_turns": 50},
            },
        },
    )
    invocation = SCRIPT.build_invocation(
        config_path,
        overrides=["evaluation.test_limit=2"],
        timestamp="stamp",
        base_environment={"DATA_ROOT": "/datasets/alfworld"},
    )

    assert invocation.command[invocation.command.index("--data-root") + 1] == "/datasets/alfworld"
    assert invocation.command[invocation.command.index("--test-limit") + 1] == "2"
    assert invocation.environment_values["ALFWORLD_DATA"] == "/datasets/alfworld"


def test_rejects_unknown_runner_parameter(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "experiment.yaml",
        {
            "version": 1,
            "method": "initial_skill_evaluation",
            "environment": "alfworld",
            "parameters": {
                "dataset": {
                    "data_root": "data",
                    "split_dir": "split",
                    "initial_skill": "SKILL.md",
                },
                "evaluation": {"unknown_parameter": 1},
            },
        },
    )

    with pytest.raises(ValueError, match="does not accept: --unknown-parameter"):
        SCRIPT.build_invocation(config_path, timestamp="stamp", base_environment={})


def test_rejects_duplicate_leaf_parameters(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "experiment.yaml",
        {
            "version": 1,
            "method": "initial_skill_evaluation",
            "environment": "alfworld",
            "parameters": {
                "dataset": {
                    "data_root": "data",
                    "split_dir": "split",
                    "initial_skill": "SKILL.md",
                    "seed": 1,
                },
                "evaluation": {"seed": 2},
            },
        },
    )

    with pytest.raises(ValueError, match="duplicate runner parameter --seed"):
        SCRIPT.build_invocation(config_path, timestamp="stamp", base_environment={})
