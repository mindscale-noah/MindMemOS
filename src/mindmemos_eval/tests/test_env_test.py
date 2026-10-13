"""Exercise the public test CLI through registered native environments without model calls."""

from pathlib import Path

import pytest
import yaml
from mindmemos_eval.cli import main
from mindmemos_skill.common.json import read_json, read_jsonl, write_json
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data, DataSample, RewardData
from mindmemos_skill.envs.typing import Reward, RolloutResult, Task, Usage
from pydantic import BaseModel


class OfflineConfig(BaseModel):
    """Accept a secret only to verify that report configuration is redacted."""

    api_key: str = "private-test-secret"
    """Dummy credential never used for a request."""


@Environment.register("eval_cli_offline")
class OfflineEnvironment(Environment[OfflineConfig]):
    """Return success, zero reward, or an execution error according to the fixture ID."""

    config_type = OfflineConfig
    """Configuration accepted by the registry."""

    async def _run(self, *, task, reward_data, skills, workspace):
        if task.id == "error":
            raise TimeoutError("API timed out")
        success = task.id == "pass"
        return RolloutResult(
            output="answer",
            trajectory=[{"role": "assistant", "content": "answer", "skill_names": [s.source.name for s in skills]}],
            reward=Reward(score=float(success), success=success, detail="fixture"),
            usage=Usage(input_tokens=10, output_tokens=2, total_tokens=12) if task.id != "unknown_usage" else Usage(),
        )


class OfflineDataConfig(BaseModel):
    """Point the registered fixture loader at a typed sample JSON file."""

    data_path: Path
    """JSON array containing task and reward_data objects."""


@Data.register("eval_cli_offline_data")
class OfflineData(Data[OfflineDataConfig]):
    """Read test samples through the same registered Data contract as real benchmarks."""

    config_type = OfflineDataConfig
    """Source options accepted by the registry."""

    def _load(self):
        return [
            DataSample(task=Task.model_validate(row["task"]), reward_data=RewardData.model_validate(row["reward_data"]))
            for row in read_json(self.config.data_path)
        ]


@pytest.fixture
def inputs(tmp_path):
    samples = tmp_path / "samples.json"
    write_json(
        samples,
        [{"task": {"id": key, "instruction": "Answer"}, "reward_data": {}} for key in ("pass", "zero", "error")],
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "environment": {"class_name": "eval_cli_offline", "config": {"api_key": "private-test-secret"}},
                "samples_file": str(samples),
                "concurrency": 2,
            }
        )
    )
    return config, samples, tmp_path / "run"


def test_no_skill_reports_six_requirements_and_keeps_errors_separate(inputs):
    config, _, output = inputs
    assert main(["skill", "test", "--config", str(config), "--no-skill", "--output-dir", str(output)]) == 1
    summary = read_json(output / "summary.json")
    assert summary["mode"] == "no_skill"
    assert (summary["planned"], summary["attempted"], summary["completed"], summary["errors"]) == (3, 3, 2, 1)
    assert summary["successful"] == 1 and summary["success_rate"] == 0.5
    assert summary["wall_seconds"] > 0
    assert summary["total_tokens"] is None and summary["usage_missing"] == 1
    assert summary["reported_usage"] == {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24}
    traces = list(read_jsonl(summary["trajectory_file"]))
    by_id = {row["task"]["id"]: row for row in traces}
    assert len(traces) == 3
    assert by_id["zero"]["outcome"]["error"] is None
    assert by_id["zero"]["outcome"]["score"] == 0
    assert by_id["error"]["result"] is None
    assert "TimeoutError" in by_id["error"]["outcome"]["error"]
    assert by_id["pass"]["result"]["trajectory"][0]["skill_names"] == []
    assert "private-test-secret" not in (output / "config.json").read_text()
    assert (output / "report.md").is_file()


def test_static_collection_runs_once_with_all_skills_and_exact_usage(inputs, tmp_path):
    config, _, output = inputs
    paths = []
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Fixture.\n---\nAnswer carefully.\n")
        paths.append(str(directory))
    assert (
        main(
            ["skill", "test", "--config", str(config), "--skills", *paths, "--limit", "2", "--output-dir", str(output)]
        )
        == 0
    )
    summary = read_json(output / "summary.json")
    assert summary["mode"] == "static_skill" and summary["attempted"] == 2
    assert summary["total_tokens"] == 24 and summary["usage_missing"] == 0
    assert summary["input_tokens"] == 20 and summary["output_tokens"] == 4
    traces = list(read_jsonl(summary["trajectory_file"]))
    assert len(traces) == 2
    assert all(row["result"]["trajectory"][0]["skill_names"] == ["first", "second"] for row in traces)
    assert len(read_json(output / "skills.json")) == 2
    assert not (output / "no_skill").exists()


def test_registered_data_loader_and_split_order(inputs, tmp_path):
    config, samples, output = inputs
    payload = yaml.safe_load(config.read_text())
    payload.pop("samples_file")
    payload["data"] = {"class_name": "eval_cli_offline_data", "config": {"data_path": str(samples)}}
    config.write_text(yaml.safe_dump(payload))
    split = tmp_path / "split.json"
    write_json(split, {"test": ["zero", "pass"]})
    assert (
        main(
            [
                "skill",
                "test",
                "--config",
                str(config),
                "--split-file",
                str(split),
                "--limit",
                "1",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert read_json(output / "task_ids.json") == ["zero"]
    summary = read_json(output / "summary.json")
    assert summary["success_rate"] == 0 and summary["errors"] == 0


@pytest.mark.parametrize("identifier", ["error", "unknown_usage"])
def test_all_errors_and_unreported_usage_are_not_zero_usage(inputs, identifier):
    config, samples, output = inputs
    write_json(samples, [{"task": {"id": identifier, "instruction": "Answer"}, "reward_data": {}}])
    assert main(["skill", "test", "--config", str(config), "--output-dir", str(output)]) == int(identifier == "error")
    summary = read_json(output / "summary.json")
    assert summary["usage_missing"] == 1 and summary["total_tokens"] is None
    if identifier == "error":
        assert summary["completed"] == 0 and summary["success_rate"] is None


def test_modes_are_mutually_exclusive(inputs):
    config, _, _ = inputs
    with pytest.raises(SystemExit):
        main(["skill", "test", "--config", str(config), "--no-skill", "--skills", "a"])


@pytest.mark.parametrize("override", [["--limit", "0"], ["--concurrency", "0"], ["--env", "unregistered"]])
def test_bad_options_fail_before_output(inputs, override):
    config, _, output = inputs
    assert main(["skill", "test", "--config", str(config), "--output-dir", str(output), *override]) == 1
    assert not output.exists()


def test_existing_output_is_not_overwritten(inputs):
    config, _, output = inputs
    output.mkdir()
    (output / "keep").write_text("original")
    assert main(["skill", "test", "--config", str(config), "--output-dir", str(output)]) == 1
    assert (output / "keep").read_text() == "original"


def test_env_listing_uses_registry(capsys):
    assert main(["skill", "envs"]) == 0
    assert "eval_cli_offline" in capsys.readouterr().out


def test_legacy_skill_body_excludes_frontmatter(inputs, tmp_path):
    """Legacy comparison injects the unchanged Markdown body and retains resources."""
    config, _, output = inputs
    directory = tmp_path / "legacy"
    directory.mkdir()
    body = "# Rules\n\nAnswer carefully.\n"
    (directory / "SKILL.md").write_text("---\nname: legacy\ndescription: Fixture.\n---\n\n" + body)
    (directory / "reference.txt").write_text("reference")
    payload = yaml.safe_load(config.read_text())
    payload["skill_body_only"] = True
    config.write_text(yaml.safe_dump(payload))
    assert (
        main(
            [
                "skill",
                "test",
                "--config",
                str(config),
                "--skills",
                str(directory),
                "--limit",
                "1",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    snapshot = read_json(output / "skills.json")["skill-000"]
    assert snapshot["content"] == body
    assert snapshot["resources"]


@pytest.mark.parametrize(
    "reported, expected", [(False, None), (True, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})]
)
def test_usage_coverage_overrides_nonzero_heuristic(tmp_path, reported, expected):
    """Partial usage is unknown while explicitly reported zero usage is complete."""
    from mindmemos_eval.skills.evaluation import _case

    usage = Usage(input_tokens=10, total_tokens=10, reported=False) if not reported else Usage(reported=True)
    result = RolloutResult(
        output="answer", trajectory=[], reward=Reward(score=0, success=False, detail="fixture"), usage=usage
    )
    assert _case("task", result, tmp_path / "result.json").usage == expected
