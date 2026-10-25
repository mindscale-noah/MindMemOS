"""Validate the exact downloaded source independently of container or model execution."""

import json

import pytest
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.skillsbench import SkillsBenchConfig, SkillsBenchData, SkillsBenchDataConfig
from mindmemos_skill.envs.data import Data
from pydantic import ValidationError


def test_registered_data_uses_public_instructions_and_private_verification(sample):
    assert Data.resolve("skillsbench") is SkillsBenchData
    assert Environment.resolve("skillsbench").config_type is SkillsBenchConfig
    assert sample.task.id == "demo" and "/root/result.txt" in sample.task.instruction
    assert "PRIVATE REFERENCE" not in sample.task.model_dump_json()
    assert sample.reward_data.task_dir.name == "demo"
    assert "tests/reference.txt" in sample.reward_data.checksums


@pytest.mark.parametrize("change", ["revision", "content", "missing", "extra", "link"])
def test_changed_or_untrusted_source_is_rejected(dataset, change):
    receipt = dataset / "download_receipt.json"
    task = dataset / "tasks/demo"
    if change == "revision":
        data = json.loads(receipt.read_text())
        data["source"]["revision"] = "different"
        receipt.write_text(json.dumps(data))
    elif change == "content":
        (task / "tests/reference.txt").write_text("changed")
    elif change == "missing":
        (task / "environment/input.bin").unlink()
    elif change == "extra":
        (task / "unexpected.txt").write_text("unexpected")
    else:
        (task / "linked.txt").symlink_to(task / "tests/reference.txt")
    with pytest.raises(ValueError):
        SkillsBenchData(SkillsBenchDataConfig(benchmark_root=dataset)).load()


def test_explicit_task_selection_does_not_include_extra_tasks(dataset):
    loader = SkillsBenchData(SkillsBenchDataConfig(benchmark_root=dataset, task_ids=["demo"]))
    assert [sample.task.id for sample in loader.load()] == ["demo"]
    with pytest.raises(ValueError, match="missing"):
        SkillsBenchData(SkillsBenchDataConfig(benchmark_root=dataset, task_ids=["extra"])).load()


@pytest.mark.parametrize("names", [[], ["demo", "demo"], ["../demo"]])
def test_invalid_task_selection_is_rejected_by_its_config(dataset, names):
    with pytest.raises(ValidationError):
        SkillsBenchDataConfig(benchmark_root=dataset, task_ids=names)


def test_model_and_environment_values_are_validated_without_runtime_io():
    assert SkillsBenchConfig().agent == "oracle"
    for options in ({"agent": "claude-code"}, {"model": "unexpected"}, {"agent_env": ["KEY=value"]}):
        with pytest.raises(ValidationError):
            SkillsBenchConfig(**options)
