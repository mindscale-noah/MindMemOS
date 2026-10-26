"""Verify implementation-owned configuration parsing and YAML editor schemas."""

import json
from pathlib import Path

import pytest
import yaml
from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.algos import Algorithm, SkillTrainer
from mindmemos_skill.algos.evolve.base import SkillEvolver
from mindmemos_skill.algos.train.skill_forest.config import SkillForestTrainerConfig
from mindmemos_skill.config.loader import load_skill_config, resolve_service_config
from mindmemos_skill.config.schema import skill_config_schema
from mindmemos_skill.pipelines import Pipeline
from mindmemos_skill.pipelines.builtin_pipelines.training.config import TrainingPipelineConfig
from pydantic import BaseModel, ConfigDict, Field

from mindmemos_skill import cli

EXAMPLE = Path(__file__).resolve().parents[3] / "config/mindmemos_skill/dev.example.yaml"
"""Public deployment example used to verify real nested component definitions."""


def test_deployment_example_covers_pipeline_fields_and_nondefault_agent_options():
    """Detect missing pipelines and ensure omitted Agent options match their defaults."""
    value = yaml.safe_load(EXAMPLE.read_text())
    deployment = load_skill_config(EXAMPLE.read_text())
    service = value["service"]
    specs = [service["repository_pipeline"], service["manage_pipeline"]]
    specs.extend(service["training_pipelines"].values())
    specs.extend(service["evolution_pipelines"].values())
    assert {spec["class_name"] for spec in specs} == {
        name
        for name, implementation in Pipeline.registered_types().items()
        if ".builtin_pipelines." in implementation.__module__
    }
    algorithms = Algorithm.registered_types()
    assert {spec["config"]["trainer"]["class_name"] for spec in service["training_pipelines"].values()} == {
        name
        for name, implementation in algorithms.items()
        if issubclass(implementation, SkillTrainer) and ".demos." not in implementation.__module__
    }
    assert {spec["config"]["evolver"]["class_name"] for spec in service["evolution_pipelines"].values()} == {
        name
        for name, implementation in algorithms.items()
        if (issubclass(implementation, SkillEvolver) or name == "offline_evolution")
        and ".demos." not in implementation.__module__
    }

    def assert_fields_explicit(actual, complete, path):
        """Compare nested mappings with the validated model including its defaults."""
        if isinstance(actual, str) and isinstance(complete, dict) and actual in service["agents"]:
            actual = service["agents"][actual]["config"]
        if isinstance(complete, dict):
            assert isinstance(actual, dict), path
            if "model" in complete and "max_completion_tokens" in complete:
                for key in complete.keys() - actual.keys():
                    field = OpenAIChatConfig.model_fields[key]
                    assert not field.is_required(), f"{path}.{key}"
                    assert complete[key] == field.get_default(call_default_factory=True), f"{path}.{key}"
                complete = {key: value for key, value in complete.items() if key in actual}
            assert actual.keys() == complete.keys(), path
            for key, nested in complete.items():
                assert_fields_explicit(actual[key], nested, f"{path}.{key}")

    assert_fields_explicit(value, deployment.model_dump(mode="json"), "deployment")


def test_editor_schema_reuses_original_algorithm_fields():
    """Preserve the owning algorithm's defaults, constraints, descriptions, and nested types."""
    schema = skill_config_schema()
    original = SkillForestTrainerConfig.model_json_schema()
    definition = schema["$defs"]["SkillForestTrainerConfig"]
    assert definition["properties"] == original["properties"]
    assert definition["required"] == original["required"]
    selection = schema["$defs"]["Algorithm_skill_forest_Selection"]["properties"]
    assert selection["class_name"]["const"] == "skill_forest"
    assert selection["config"]["$ref"] == "#/$defs/SkillForestTrainerConfig"
    for name in ("optimizer", "num_trees", "rollout_concurrency"):
        assert definition["properties"][name]["description"]


def test_preflight_retains_original_models_and_preserves_nested_credentials():
    """Retain actual algorithm models and preserve explicit settings through serialization and reconstruction."""
    value = yaml.safe_load(EXAMPLE.read_text())
    options = value["service"]["training_pipelines"]["skill_forest"]["config"]["trainer"]["config"]
    preset = value["service"]["agents"][options["optimizer"]]["config"]
    preset["api_key"] = "schema-test-secret"
    preset["base_url"] = "https://test.invalid/v1"
    deployment = load_skill_config(yaml.safe_dump(value))
    pipeline = deployment.service.training_pipelines["skill_forest"].config
    assert type(pipeline) is TrainingPipelineConfig
    assert type(pipeline.trainer.config) is SkillForestTrainerConfig
    assert pipeline.trainer.config.optimizer.api_key == "schema-test-secret"
    assert "schema-test-secret" in deployment.model_dump_json(warnings="error")
    resolved = resolve_service_config(deployment.service)
    trainer = resolved.training_pipelines["skill_forest"].config.trainer.config
    assert type(trainer) is SkillForestTrainerConfig
    assert trainer.optimizer.api_key == "schema-test-secret"
    assert "schema-test-secret" in resolved.model_dump_json(warnings="error")


def test_schema_validates_real_example_and_rejects_wrong_algorithm_fields():
    """Reject unknown fields, invalid numeric limits, and wrong algorithm families in the editor schema."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = skill_config_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    value = yaml.safe_load(EXAMPLE.read_text())
    validator.validate(value)
    trainer = value["service"]["training_pipelines"]["skill_forest"]["config"]["trainer"]
    trainer["config"]["rollout_concurrency"] = 0
    assert not validator.is_valid(value)
    trainer["config"]["rollout_concurrency"] = 2
    trainer["config"]["num_tree"] = 3
    assert not validator.is_valid(value)
    del trainer["config"]["num_tree"]
    trainer["class_name"] = "simple_evolve"
    assert not validator.is_valid(value)


def test_explicit_custom_registration_uses_its_own_config_without_construction(monkeypatch):
    """Include an imported extension and reuse its config without creating backend instances."""
    monkeypatch.setattr(Algorithm, "_registry", Algorithm.registered_types())

    class CustomConfig(BaseModel):
        """Configure a test-only trainer."""

        model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
        """Reject fields outside this extension's contract."""

        batch_size: int = Field(default=3, ge=1, description="Training samples processed together.")
        """Maximum samples processed together; defaults to 3."""

    @Algorithm.register("schema_custom")
    class CustomTrainer(SkillTrainer[CustomConfig]):
        """Record a schema extension that must never be constructed during export."""

        config_type = CustomConfig
        """Extension-owned model used for runtime and editor validation."""

        def __init__(self, config):
            raise AssertionError("Schema generation must not construct trainers")

        async def train(self, **kwargs):
            raise NotImplementedError

    schema = skill_config_schema()
    properties = schema["$defs"]["CustomConfig"]["properties"]
    assert properties == CustomConfig.model_json_schema()["properties"]
    assert schema["$defs"]["Algorithm_schema_custom_Selection"]["properties"]["class_name"]["const"] == "schema_custom"


def test_schema_cli_does_not_require_a_deployment_or_open_service(tmp_path, monkeypatch):
    """Export editor metadata without loading credentials, opening storage, or requiring --config."""

    def reject_service(*args, **kwargs):
        raise AssertionError("Schema export must not construct a service")

    monkeypatch.setattr(cli, "create_service", reject_service)
    output = tmp_path / "editor/service.schema.json"
    assert cli.main(["schema", "--output", str(output)]) == 0
    schema = json.loads(output.read_text())
    assert "SkillForestTrainerConfig" in schema["$defs"]
    assert not (tmp_path / "data").exists()


def test_repository_cli_still_requires_explicit_config():
    """Keep explicit deployment requirements for actual repository operations."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["list"])
    assert exc.value.code == 2
