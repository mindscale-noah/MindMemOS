"""Pipeline contracts and composition without database or model execution."""

import pytest
from mindmemos_skill.algos import Algorithm, SkillEvolver
from mindmemos_skill.api.config import ServiceConfig
from mindmemos_skill.persistence import Persistence
from mindmemos_skill.pipelines import (
    EvolutionPipeline,
    Pipeline,
    SkillPipelineRequest,
    SkillPipelineResult,
    TrainingPipeline,
    TrainingRequest,
    TrainingSplit,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mindmemos_skill import create_service


class Options(BaseModel):
    """Strict configuration for resource-free test backends."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=10, gt=0)


class Evolver(SkillEvolver[Options]):
    """Algorithm dependency used only to verify composition."""

    config_type = Options

    def evolve(self, *, skill, trajectories):
        raise NotImplementedError


class Store(Persistence[Options]):
    """Resource-free persistence dependency."""

    config_type = Options


class Evolution(EvolutionPipeline[Options]):
    """Composition-only implementation with no executable business flow."""

    config_type = Options

    def __init__(self, config, *, algorithm, repository):
        super().__init__(config)
        self.algorithm = algorithm
        self.repository = repository

    def evolve(self, request):
        raise NotImplementedError


class Training(TrainingPipeline[Options]):
    """Training implementation used only to verify family registration."""

    config_type = Options

    def train(self, request):
        raise NotImplementedError


@pytest.mark.parametrize("base", [EvolutionPipeline, TrainingPipeline])
def test_pipeline_capabilities_require_implementation(base):
    """Reject incomplete capabilities both at construction and registration."""
    with pytest.raises(TypeError, match="abstract"):
        base(Options())
    with pytest.raises(TypeError, match="concrete subclass"):
        Pipeline.register("incomplete")(base)


def test_pipeline_composition_reuses_dependencies_and_allows_algorithm_selection(monkeypatch):
    """Compose one flow with different algorithms through the existing factory."""
    monkeypatch.setattr(Pipeline, "_registry", {})
    monkeypatch.setattr(Algorithm, "_registry", {})
    monkeypatch.setattr(Persistence, "_registry", {})
    Pipeline.register("evolution")(Evolution)
    Pipeline.register("training")(Training)
    Algorithm.register("evolution")(Evolver)
    Persistence.register("store")(Store)
    config = ServiceConfig.model_validate(
        {
            "evolution_pipeline": "first",
            "backends": {
                "first": {
                    "module": "pipelines",
                    "class_name": "evolution",
                    "dependencies": {"algorithm": "algo_a", "repository": "store"},
                },
                "second": {
                    "module": "pipelines",
                    "class_name": "evolution",
                    "config": {"limit": 20},
                    "dependencies": {"algorithm": "algo_b", "repository": "store"},
                },
                "training": {"module": "pipelines", "class_name": "training"},
                "store": {"module": "persistence", "class_name": "store"},
                "algo_a": {"module": "algos", "class_name": "evolution"},
                "algo_b": {"module": "algos", "class_name": "evolution", "config": {"limit": 3}},
            },
        }
    )
    service = create_service(config)
    first, second = service.backends["first"], service.backends["second"]
    assert first is not second
    assert first.repository is second.repository is service.backends["store"]
    assert first.algorithm is service.backends["algo_a"]
    assert second.algorithm is service.backends["algo_b"]
    assert first.algorithm is not second.algorithm
    assert second.config.limit == 20
    assert second.algorithm.config.limit == 3
    assert isinstance(service.backends["training"], TrainingPipeline)
    assert Pipeline.resolve("evolution") is Evolution
    assert Algorithm.resolve("evolution") is Evolver
    with pytest.raises(ValueError, match="request is required"):
        service.auto_evolve_skill()
    service.close()


def test_pipeline_configuration_is_validated_before_construction(monkeypatch):
    """Do not construct any backends when a pipeline's options are invalid."""
    monkeypatch.setattr(Pipeline, "_registry", {})
    calls = []

    @Pipeline.register("training")
    class ObservedTraining(Training):
        def __init__(self, config):
            calls.append(config)
            super().__init__(config)

    config = ServiceConfig.model_validate(
        {
            "backends": {
                "valid": {"module": "pipelines", "class_name": "training"},
                "invalid": {"module": "pipelines", "class_name": "training", "config": {"limit": 0}},
            }
        }
    )
    with pytest.raises(ValidationError):
        create_service(config)
    assert calls == []


def test_request_requires_scope_and_rejects_implicit_selection_options():
    """Keep common requests strict while allowing explicitly typed extensions."""
    request = SkillPipelineRequest(project_id=" project ", skill_ref=" skill ")
    assert request.project_id == "project"
    assert request.skill_ref == "skill"
    assert request.base_version_id is None
    for values in (
        {"skill_ref": "skill"},
        {"project_id": " ", "skill_ref": "skill"},
        {"project_id": "project", "skill_ref": "skill", "base_version_id": " "},
        {"project_id": "project", "skill_ref": "skill", "trajectory_ids": ["t1"]},
    ):
        with pytest.raises(ValidationError):
            SkillPipelineRequest.model_validate(values)

    class HistoricalRequest(SkillPipelineRequest):
        trajectory_ids: list[str] = Field(min_length=1)

    extended = HistoricalRequest(project_id="project", skill_ref="skill", trajectory_ids=["t1"])
    assert extended.trajectory_ids == ["t1"]


def test_result_requires_persisted_identity_and_execution_provenance():
    """A pending run or candidate alone cannot represent a completed result."""
    values = {
        "project_id": "project",
        "skill_id": "skill",
        "base_version_id": "v1",
        "version_id": "v2",
        "run_id": "run1",
    }
    result = SkillPipelineResult.model_validate(values)
    assert SkillPipelineResult.model_validate_json(result.model_dump_json()) == result
    for missing in ("version_id", "base_version_id", "run_id"):
        with pytest.raises(ValidationError):
            SkillPipelineResult.model_validate({key: value for key, value in values.items() if key != missing})
    with pytest.raises(ValidationError):
        SkillPipelineResult.model_validate(values | {"status": "running"})


def training_input():
    """Return three materialized tasks and a complete held-out partition."""
    return {
        "project_id": "project",
        "base_version_id": "v1",
        "tasks": [
            {"task": {"id": identifier, "instruction": "Solve it"}, "reward_data": {}} for identifier in ("a", "b", "c")
        ],
        "split": {"train": ["a"], "val": ["b"], "test": ["c"]},
    }


@pytest.mark.parametrize("optional", [{}, {"val": ["b"]}, {"test": ["b"]}, {"val": ["b"], "test": ["c"]}])
def test_training_optional_splits_and_round_trip(optional):
    """Allow either held-out split independently and preserve its explicit order."""
    values = training_input()
    selected = {"a", *(identifier for ids in optional.values() for identifier in ids)}
    values["tasks"] = [sample for sample in values["tasks"] if sample["task"]["id"] in selected]
    values["split"] = {"train": ["a"], **optional}
    request = TrainingRequest.model_validate(values)
    assert request.split.val == optional.get("val", [])
    assert request.split.test == optional.get("test", [])
    assert TrainingRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize(
    "split",
    [
        {},
        {"train": []},
        {"train": [" "]},
        {"train": ["a", "a"]},
        {"train": ["a"], "val": ["a"]},
        {"train": ["a"], "val": ["b"], "test": ["b"]},
        {"train": ["a"], "validation": ["b"]},
    ],
)
def test_training_split_rejects_invalid_membership(split):
    """Reject empty training, duplicate membership, and unsupported split names."""
    with pytest.raises(ValidationError):
        TrainingSplit.model_validate(split)


@pytest.mark.parametrize(
    "failure", ["version_missing", "version_null", "version_blank", "duplicate", "unknown", "unassigned", "empty"]
)
def test_training_request_rejects_invalid_version_or_task_partition(failure):
    """Validate the required input version and task-to-split correspondence."""
    values = training_input()
    if failure == "version_missing":
        values.pop("base_version_id")
    elif failure == "version_null":
        values["base_version_id"] = None
    elif failure == "version_blank":
        values["base_version_id"] = " "
    elif failure == "duplicate":
        values["tasks"].append(values["tasks"][0])
    elif failure == "unknown":
        values["split"]["test"] = ["missing"]
    elif failure == "unassigned":
        values["split"].pop("test")
    else:
        values["tasks"] = []
    with pytest.raises(ValidationError):
        TrainingRequest.model_validate(values)


def test_training_preserves_public_task_and_private_scoring_subtypes():
    """Use existing sample contracts without dropping environment-specific data."""
    from mindmemos_skill.envs.data.typing import DataSample, RewardData
    from mindmemos_skill.envs.typing import Task

    class FileTask(Task):
        input_file: str

    class ExpectedAnswer(RewardData):
        answer: str

    sample = DataSample(
        task=FileTask(id="a", instruction="Solve it", input_file="input.csv"),
        reward_data=ExpectedAnswer(answer="private answer"),
    )
    request = TrainingRequest(
        project_id="project",
        base_version_id="v1",
        tasks=[sample],
        split=TrainingSplit(train=["a"]),
    )
    assert isinstance(request.tasks[0].task, FileTask)
    assert isinstance(request.tasks[0].reward_data, ExpectedAnswer)
    payload = request.model_dump()["tasks"][0]
    assert payload["task"]["input_file"] == "input.csv"
    assert "answer" not in payload["task"]
    assert payload["reward_data"]["answer"] == "private answer"
