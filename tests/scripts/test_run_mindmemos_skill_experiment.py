from __future__ import annotations

import importlib.util
import json
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
        CONFIG_ROOT / "skill_grpo_with_experience_validation" / "livemath" / "default.yaml",
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert invocation.method == "skill_grpo_with_experience_validation"
    assert invocation.environment == "livemath"
    assert invocation.run_id == "livemath_skill_grpo_with_experience_validation_20260102-030405"
    assert invocation.output_dir == Path(
        "outputs/livemath/skill_grpo_with_experience_validation/livemath_skill_grpo_with_experience_validation_20260102-030405"
    )
    assert invocation.command[:7] == ["uv", "run", "--package", "mindmemos-skill", "--extra", "llm", "python"]
    runner, arguments = runner_command(invocation)
    assert runner.endswith("scripts/mindmemos_skill/runners/evolve.py")
    assert arguments[:2] == ["--algorithm", "skill_grpo_with_experience_validation"]
    assert invocation.command[invocation.command.index("--benchmark") + 1] == "livemath"
    assert "--no-fail-fast" in invocation.command
    assert "--shuffle-choices" in invocation.command
    assert "--no-use-theorem" in invocation.command


def test_replay_buffer_does_not_restrict_environment_or_dataset(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "experiment.yaml",
        {
            "version": 1,
            "method": "skill_grpo_with_experience_validation",
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
                    "env_ref": "alfworld_bounded_history",
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
    assert invocation.command[invocation.command.index("--env-ref") + 1] == "alfworld_bounded_history"
    assert invocation.command[invocation.command.index("--dataset-options") + 1] == (
        '{"split_dir":"custom/splits","alfworld_data":"custom/data"}'
    )
    assert invocation.command[invocation.command.index("--env-options") + 1] == '{"seed":7}'
    assert invocation.resolved_config["resolved"]["extras"] == ["llm", "alfworld"]


@pytest.mark.parametrize(
    "config_path",
    [
        CONFIG_ROOT / "skill_grpo_with_experience_validation" / "alfworld" / "default.yaml",
        CONFIG_ROOT / "offline_evolution" / "livemath" / "initial_skill.yaml",
    ],
)
def test_registered_component_experiments_accept_custom_dataset_and_env(config_path: Path, tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        config_path,
        overrides=[
            "environment=custom-benchmark",
            "dataset.dataset_ref=user_dataset",
            "dataset.env_ref=user_env",
        ],
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert invocation.environment == "custom-benchmark"
    assert invocation.command[invocation.command.index("--benchmark") + 1] == "custom-benchmark"
    assert invocation.command[invocation.command.index("--dataset-ref") + 1] == "user_dataset"
    assert invocation.command[invocation.command.index("--env-ref") + 1] == "user_env"


def test_experience_validation_can_enable_previous_batch_invalid_feedback_by_override(tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "skill_grpo_with_experience_validation" / "alfworld" / "default.yaml",
        overrides=["algorithm.previous_batch_invalid_experience_feedback=true"],
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert "--previous-batch-invalid-experience-feedback" in invocation.command


def test_launcher_passes_external_component_modules_to_runner(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "experiment.yaml",
        {
            "version": 1,
            "method": "skill_grpo_with_experience_validation",
            "environment": "custom-benchmark",
            "launcher": {
                "component_modules": ["acme_skill_components", "team_benchmark.components"],
                "extra_dependencies": ["acme-benchmark"],
            },
            "parameters": {
                "dataset": {
                    "dataset_ref": "user_dataset",
                    "dataset_options": {"source": "fixture"},
                    "initial_skill": "SKILL.md",
                },
                "environment_options": {
                    "env_ref": "user_env",
                    "env_options": {"mode": "test"},
                    "max_turns": 3,
                },
            },
        },
    )

    invocation = SCRIPT.build_invocation(config_path, timestamp="stamp", base_environment={})

    assert json.loads(invocation.environment_values["MINDMEMOS_SKILL_COMPONENT_MODULES"]) == [
        "acme_skill_components",
        "team_benchmark.components",
    ]
    assert invocation.command[:7] == ["uv", "run", "--package", "mindmemos-skill", "--extra", "llm", "--extra"]
    assert "acme-benchmark" in invocation.command


def test_runner_directory_contains_only_the_two_algorithm_family_scripts() -> None:
    runner_dir = SCRIPT_PATH.parent / "runners"

    assert sorted(path.name for path in runner_dir.glob("*.py")) == ["evolve.py", "trace2skill.py"]


def test_experiment_adapters_stay_outside_the_algorithm_support_package() -> None:
    package_experiments = SCRIPT_PATH.parents[2] / "src" / "mindmemos_skill" / "mindmemos_skill" / "experiments"
    script_experiments = SCRIPT_PATH.parent / "experiments"

    assert not package_experiments.exists()
    assert all(spec.implementation_path.parent == script_experiments for spec in SCRIPT.EXPERIMENTS.values())


def test_officeqa_custom_search_uses_env_file_without_putting_secrets_in_command(tmp_path: Path) -> None:
    env_file = tmp_path / ".skill.env"
    env_file.write_text(
        "OFFICEQA_SERPAPI_PROTOCOL=serper\n"
        "OFFICEQA_SERPAPI_BASE_URL=https://search.example.test\n"
        "OFFICEQA_SERPAPI_API_KEY=dummy-secret\n",
        encoding="utf-8",
    )
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "skill_grpo_with_experience_validation" / "officeqa" / "default.yaml",
        env_file_override=env_file,
        overrides=["environment_options.env_options.search_mode=custom_search"],
        timestamp="20260102-030405",
        base_environment={},
    )

    serialized_options = invocation.command[invocation.command.index("--env-options") + 1]
    assert json.loads(serialized_options)["search_mode"] == "custom_search"
    assert invocation.environment_values["OFFICEQA_SERPAPI_PROTOCOL"] == "serper"
    assert invocation.environment_values["OFFICEQA_SERPAPI_BASE_URL"] == "https://search.example.test"
    assert invocation.environment_values["OFFICEQA_SERPAPI_API_KEY"] == "dummy-secret"
    assert "dummy-secret" not in " ".join(invocation.command)


@pytest.mark.parametrize(
    "config_path",
    [
        CONFIG_ROOT / "skill_grpo_with_experience_validation" / "officeqa" / "default.yaml",
    ],
)
def test_officeqa_configs_forward_custom_search_env_options(config_path: Path, tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        config_path,
        env_file_override=empty_env_file(tmp_path),
        overrides=["environment_options.env_options.search_mode=custom_search"],
        timestamp="20260102-030405",
        base_environment={},
    )

    serialized_options = invocation.command[invocation.command.index("--env-options") + 1]
    options = json.loads(serialized_options)
    assert options["search_mode"] == "custom_search"


def test_spreadsheetbench_experience_validation_config_uses_expected_controls(tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "skill_grpo_with_experience_validation" / "spreadsheetbench" / "default.yaml",
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert invocation.environment == "spreadsheetbench"
    assert invocation.resolved_config["resolved"]["extras"] == ["llm", "spreadsheetbench"]
    assert invocation.command[invocation.command.index("--benchmark") + 1] == "spreadsheetbench"
    assert invocation.command[invocation.command.index("--epochs") + 1] == "2"
    assert invocation.command[invocation.command.index("--train-rollouts") + 1] == "4"
    assert invocation.command[invocation.command.index("--test-rollouts") + 1] == "3"
    assert invocation.command[invocation.command.index("--max-experiences-per-task") + 1] == "3"
    assert invocation.command[invocation.command.index("--max-tool-observation-chars") + 1] == "24000"
    assert invocation.command[invocation.command.index("--max-turns") + 1] == "30"


def test_spreadsheetbench_experience_validation_can_disable_tool_observation_limit(tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "skill_grpo_with_experience_validation" / "spreadsheetbench" / "default.yaml",
        overrides=["algorithm.max_tool_observation_chars=null"],
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    assert "--max-tool-observation-chars" not in invocation.command


@pytest.mark.parametrize(
    "config_path",
    [CONFIG_ROOT / "skill_grpo_with_experience_validation" / "alfworld" / "default.yaml"],
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
    assert invocation.command[invocation.command.index("--env-seed") + 1] == "42"


def test_trace2skill_routes_to_family_runner_and_keeps_test_config(tmp_path: Path) -> None:
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "offline_evolution" / "livemath" / "initial_skill.yaml",
        env_file_override=empty_env_file(tmp_path),
        timestamp="20260102-030405",
        base_environment={},
    )

    runner, arguments = runner_command(invocation)
    assert runner.endswith("scripts/mindmemos_skill/runners/trace2skill.py")
    assert arguments[:2] == ["--algorithm", "offline_evolution"]
    assert invocation.resolved_config["resolved"]["extras"] == ["llm", "offline-evolution"]
    assert invocation.command[invocation.command.index("--test-rollouts") + 1] == "1"


@pytest.mark.parametrize("environment,skill_name", [("livemath", "livemath"), ("spreadsheetbench", "xlsx")])
def test_offline_evolution_configs_keep_dependencies_and_initial_skill(environment, skill_name, tmp_path):
    invocation = SCRIPT.build_invocation(
        CONFIG_ROOT / "offline_evolution" / environment / "initial_skill.yaml",
        env_file_override=empty_env_file(tmp_path),
        overrides=[
            "offline_evolution.train_limit=2",
            "offline_evolution.min_trajectories=1",
            "evaluation.test_limit=1",
        ],
        timestamp="20260102-030405",
        base_environment={},
    )
    runner, arguments = runner_command(invocation)
    assert runner.endswith("scripts/mindmemos_skill/runners/trace2skill.py")
    assert arguments[:2] == ["--algorithm", "offline_evolution"]
    expected_extras = ["llm", "offline-evolution"]
    if environment == "spreadsheetbench":
        expected_extras.insert(1, "spreadsheetbench")
    assert invocation.resolved_config["resolved"]["extras"] == expected_extras
    skill_path = invocation.command[invocation.command.index("--initial-skill") + 1]
    assert skill_path == f"resources/mindmemos_skill/skills/{environment}/{skill_name}/SKILL.md"
    assert (SCRIPT.REPO_ROOT / skill_path).is_file()
    assert invocation.command[invocation.command.index("--train-limit") + 1] == "2"
    assert invocation.command[invocation.command.index("--test-limit") + 1] == "1"


def test_offline_evolution_does_not_offer_resource_evolution(tmp_path):
    with pytest.raises(ValueError, match="does not accept: --skill-md-only"):
        SCRIPT.build_invocation(
            CONFIG_ROOT / "offline_evolution" / "livemath" / "initial_skill.yaml",
            env_file_override=empty_env_file(tmp_path),
            overrides=["offline_evolution.skill_md_only=false"],
            base_environment={},
        )


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
        (CONFIG_ROOT / "skill_grpo_with_experience_validation" / "alfworld" / "default.yaml", "50"),
        (CONFIG_ROOT / "skill_grpo_with_experience_validation" / "livemath" / "default.yaml", "1"),
        (CONFIG_ROOT / "skill_grpo_with_experience_validation" / "spreadsheetbench" / "default.yaml", "30"),
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
        CONFIG_ROOT / "skill_grpo_with_experience_validation" / "alfworld" / "default.yaml",
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
            "method": "skill_grpo_with_experience_validation",
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
            "method": "skill_grpo_with_experience_validation",
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
            "method": "skill_grpo_with_experience_validation",
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
