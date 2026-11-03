"""Generate editor schemas from the configurations owned by registered implementations."""

from copy import deepcopy
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, create_model

from ..agents import Agent
from ..agents.config import AgentName
from ..algos import Algorithm, SkillEvolver, SkillTrainer
from ..api.typing import TrainingRequest
from ..envs import Environment
from ..envs.data import DataSample
from ..infra import Infrastructure
from ..infra.database import Database
from ..pipelines import AnalyzePipeline, EvolutionPipeline, ManagePipeline, Pipeline, TrainingPipeline
from ..pipelines.builtin_pipelines.evolve.config import SimpleEvolutionPipelineConfig
from ..pipelines.builtin_pipelines.repository.repository import RepositoryPipeline
from ..pipelines.builtin_pipelines.training.config import TrainingPipelineConfig
from ..registry import Registrable
from .config import (
    ImplementationConfig,
    Name,
    ObservabilityConfig,
    PersistenceConfig,
    ServiceConfig,
    SkillConfig,
)


def _choice(
    registry: type[Registrable],
    family: type[Registrable],
    overrides: dict[type[BaseModel], type[BaseModel]] | None = None,
) -> Any:
    """Describe each registered selection using its original configuration model."""
    branches = []
    for name, implementation in registry.registered_types().items():
        if not issubclass(implementation, family):
            continue
        original = implementation.config_type
        options = (overrides or {}).get(original, original)
        config_field = (
            Field(description=original.__doc__)
            if any(field.is_required() for field in options.model_fields.values())
            else Field(default_factory=options, description=original.__doc__)
        )
        branches.append(
            create_model(
                f"{registry.__name__}_{name}_Selection",
                __base__=ImplementationConfig,
                class_name=(Literal[name], Field(description="Registered implementation name.")),
                config=(options, config_field),
            )
        )
    if not branches:
        raise ValueError(f"No registered implementations of {family.__name__}")
    if len(branches) == 1:
        return branches[0]
    return Annotated[Union[tuple(branches)], Field(discriminator="class_name")]


def _field(model: type[BaseModel], name: str, annotation: Any) -> tuple[Any, Any]:
    """Keep the owning model's field documentation and defaults when specializing its type."""
    return annotation, deepcopy(model.model_fields[name])


def skill_config_schema() -> dict[str, Any]:
    """Build a complete deployment schema without constructing service resources.

    Returns:
        JSON Schema with class_name branches, implementation-owned parameters,
        descriptions, defaults, and constraints for all currently registered backends.

    Raises:
        ValueError: A required backend family has no registered implementations.
    """
    trainer = _choice(Algorithm, SkillTrainer)
    evolver = _choice(Algorithm, SkillEvolver)
    training = create_model(
        "TrainingPipelineEditorConfig",
        __base__=TrainingPipelineConfig,
        trainer=_field(TrainingPipelineConfig, "trainer", trainer),
    )
    evolution = create_model(
        "SimpleEvolutionPipelineEditorConfig",
        __base__=SimpleEvolutionPipelineConfig,
        evolver=_field(SimpleEvolutionPipelineConfig, "evolver", evolver),
    )
    overrides = {TrainingPipelineConfig: training, SimpleEvolutionPipelineConfig: evolution}
    database = _choice(Infrastructure, Database)
    persistence = create_model(
        "PersistenceEditorConfig",
        __base__=PersistenceConfig,
        database=_field(PersistenceConfig, "database", database),
    )
    observability = create_model(
        "ObservabilityEditorConfig",
        __base__=ObservabilityConfig,
        backend=_field(ObservabilityConfig, "backend", database | None),
    )
    service = create_model(
        "ServiceEditorConfig",
        __base__=ServiceConfig,
        agents=_field(ServiceConfig, "agents", dict[AgentName, _choice(Agent, Agent)]),
        persistence=_field(ServiceConfig, "persistence", persistence | None),
        observability=_field(ServiceConfig, "observability", observability | None),
        repository_pipeline=_field(ServiceConfig, "repository_pipeline", _choice(Pipeline, RepositoryPipeline) | None),
        manage_pipeline=_field(ServiceConfig, "manage_pipeline", _choice(Pipeline, ManagePipeline) | None),
        analyze_pipeline=_field(ServiceConfig, "analyze_pipeline", _choice(Pipeline, AnalyzePipeline) | None),
        training_pipelines=_field(
            ServiceConfig, "training_pipelines", dict[Name, _choice(Pipeline, TrainingPipeline, overrides)]
        ),
        evolution_pipelines=_field(
            ServiceConfig, "evolution_pipelines", dict[Name, _choice(Pipeline, EvolutionPipeline, overrides)]
        ),
    )
    deployment = create_model(
        "SkillDeploymentEditorConfig",
        __base__=SkillConfig,
        service=_field(SkillConfig, "service", service),
    )
    schema = TypeAdapter(deployment).json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "MindMemOS Skill service configuration"
    return schema


def training_request_schema() -> dict[str, Any]:
    """Generate training request hints from the original registered environment models.

    Returns:
        JSON Schema for per-run inputs with environment-specific config branches.
    """
    request = create_model(
        "TrainingRequestEditorConfig",
        __base__=TrainingRequest,
        env=_field(TrainingRequest, "env", _choice(Environment, Environment)),
        tasks=_field(TrainingRequest, "tasks", list[dict[str, JsonValue]]),
    )
    schema = TypeAdapter(request).json_schema()
    schema["allOf"] = []
    for name, implementation in Environment.registered_types().items():
        sample = create_model(
            f"Environment_{name}_Sample",
            __base__=DataSample,
            task=_field(DataSample, "task", implementation.task_type),
            reward_data=_field(DataSample, "reward_data", implementation.reward_data_type),
        )
        sample_schema = sample.model_json_schema()
        schema["$defs"].update(sample_schema.pop("$defs", {}))
        schema["$defs"][sample.__name__] = sample_schema
        schema["allOf"].append(
            {
                "if": {
                    "properties": {"env": {"properties": {"class_name": {"const": name}}, "required": ["class_name"]}},
                    "required": ["env"],
                },
                "then": {"properties": {"tasks": {"items": {"$ref": f"#/$defs/{sample.__name__}"}}}},
            }
        )
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "MindMemOS Skill training request"
    return schema


def environment_config_schema() -> dict[str, Any]:
    """Describe the original environment configurations for standalone selection files.

    Returns:
        JSON Schema with a class_name discriminator and environment-owned models.
    """
    schema = TypeAdapter(_choice(Environment, Environment)).json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema
