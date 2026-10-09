"""Verify request-owned environment configuration across validation and execution."""

import json

import pytest
from mindmemos_skill.api.typing import RegisterRequest, StructuredSkillInfo, TrainingRequest
from mindmemos_skill.config.config import ServiceConfig
from mindmemos_skill.config.schema import training_request_schema
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.searchqa.config import SearchQAConfig
from mindmemos_skill.envs.builtin_envs.searchqa.typing import SearchQARewardData, SearchQATask
from pydantic import ValidationError

from mindmemos_skill import cli, create_service, demos

from .service_config import deployment_config


def search_request():
    """Return serialized SearchQA inputs with complete task and private scoring fields."""
    return {
        "project_id": "project",
        "algo": "train",
        "env": {"class_name": "searchqa", "config": {"agent": {"model": "offline"}, "max_turns": 2}},
        "init_skill_version_id": "parent",
        "work_dir": "unused",
        "tasks": [
            {
                "task": {"id": "a", "instruction": "Who?", "context": "A passage"},
                "reward_data": {"answers": ["Someone"]},
            }
        ],
        "split": {"train": ["a"]},
    }


def test_request_validation_reuses_environment_models_without_construction(monkeypatch):
    """Validate config and restore data types without opening an agent or environment."""
    monkeypatch.setattr(
        Environment.resolve("searchqa"), "__init__", lambda *args: pytest.fail("Constructed environment")
    )
    request = TrainingRequest.model_validate(search_request())
    assert type(request.env.config) is SearchQAConfig
    assert request.env.config.max_turns == 2
    assert type(request.tasks[0].task) is SearchQATask
    assert type(request.tasks[0].reward_data) is SearchQARewardData
    assert TrainingRequest.model_validate_json(request.model_dump_json()) == request


def test_request_preserves_explicit_environment_credentials(monkeypatch):
    """Keep explicit credentials through ordinary JSON serialization."""
    monkeypatch.setattr(
        Environment.resolve("searchqa"), "__init__", lambda *args: pytest.fail("Constructed environment")
    )
    value = search_request()
    value["env"]["config"]["agent"].update(api_key="request-test-key", base_url="https://test.invalid/v1")
    request = TrainingRequest.model_validate(value)
    assert request.env.config.agent.api_key == "request-test-key"
    assert request.env.config.agent.base_url == "https://test.invalid/v1"
    assert "request-test-key" in request.model_dump_json(warnings="error")


@pytest.mark.parametrize(
    "env",
    [
        "searchqa",
        {"class_name": "missing"},
        {"class_name": "sqlite"},
        {"class_name": "searchqa", "config": {"agent": {"model": "offline"}, "max_turns": 0}},
        {"class_name": "searchqa", "config": {"agent": {"model": "offline"}, "max_turn": 2}},
    ],
)
def test_request_rejects_invalid_environment_selections(env):
    """Reject names without config envelopes, wrong families, typos, and invalid limits."""
    with pytest.raises(ValidationError):
        TrainingRequest.model_validate(search_request() | {"env": env})


def test_service_rejects_removed_environment_table():
    """Keep environments out of the service startup contract."""
    with pytest.raises(ValidationError, match="training_environments"):
        ServiceConfig.model_validate({"home": ".", "training_environments": {}})


def test_request_schema_reuses_original_environment_fields_and_cli(tmp_path):
    """Export the owning environment's descriptions, defaults, and constraints."""
    schema = training_request_schema()
    assert schema["$defs"]["SearchQAConfig"]["properties"] == SearchQAConfig.model_json_schema()["properties"]
    assert schema["$defs"]["Environment_searchqa_Selection"]["properties"]["config"]["$ref"] == "#/$defs/SearchQAConfig"
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(search_request())
    assert not validator.is_valid(search_request() | {"env": "searchqa"})
    invalid = search_request()
    invalid["env"]["config"]["max_turns"] = 0
    assert not validator.is_valid(invalid)
    output = tmp_path / "request.schema.json"
    assert cli.main(["schema", "--kind", "training-request", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == schema


@pytest.mark.asyncio
async def test_real_pipeline_uses_request_environment_without_service_profiles(tmp_path, monkeypatch):
    """Run, evaluate, publish and resume through Python using a request-owned environment."""
    created = []
    original_build = Environment.build

    def build(name, options, **dependencies):
        environment = original_build(name, options, **dependencies)
        created.append(environment)
        return environment

    monkeypatch.setattr(Environment, "build", build)
    config = deployment_config(
        {
            "home": tmp_path,
            "persistence": {"database": {"class_name": "sqlite"}},
            "repository_pipeline": {"class_name": "repository"},
            "training_pipelines": {
                "train": {
                    "class_name": "training",
                    "config": {"trainer": {"class_name": "demo_train", "config": {"delay_seconds": 0}}},
                }
            },
        },
        "project",
    )
    async with await create_service(config) as service:
        initial = await service.register_skill(
            RegisterRequest(
                project_id="project",
                skill=StructuredSkillInfo(
                    content="# Guide\nSave output.", name="Guide", description="Guide", standard_format=False
                ),
            )
        )
        request = TrainingRequest.model_validate(
            {
                **search_request(),
                "env": {"class_name": "demo_offline", "config": {"delay_seconds": 0}},
                "init_skill_version_id": initial.skill_version_id,
                "work_dir": str(tmp_path / "run"),
                "tasks": [
                    {"task": {"id": name, "instruction": "Save"}, "reward_data": {}} for name in ("train", "test")
                ],
                "split": {"train": ["train"], "test": ["test"]},
            }
        )
        version_id = (await service.run_training(request)).version_id
        assert len(created) == 1
        assert created[0].config == request.env.config
        assert created[0].config.delay_seconds == 0
        assert (tmp_path / "run/test/results.json").exists()
        assert (await service.run_training(request)).version_id == version_id
        assert Environment.resolve("demo_offline") is demos.environment.DemoEnvironment
