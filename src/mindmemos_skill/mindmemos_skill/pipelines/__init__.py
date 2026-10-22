"""Shared pipeline contracts and explicit built-in registrations."""

from ..api.typing import (
    AlgoSkillEvolveResult,
    OrganizeRequest,
    OrganizeResult,
    RecommendRequest,
    RecommendResult,
    SkillPipelineRequest,
    TrainingRequest,
    TrainingSplit,
)
from .analyze import AnalyzePipeline
from .base import Pipeline
from .builtin_pipelines.analyze.pipeline import BuiltinAnalyzePipeline
from .builtin_pipelines.evolve.evolving import SimpleEvolutionPipeline
from .builtin_pipelines.manage.graph_manage import GraphManagePipeline
from .builtin_pipelines.repository.repository import RepositoryPipeline
from .builtin_pipelines.training.training import ConfiguredTrainingPipeline
from .evolve import EvolutionPipeline
from .manage import ManagePipeline
from .train import TrainingPipeline

__all__ = [
    "Pipeline",
    "AnalyzePipeline",
    "BuiltinAnalyzePipeline",
    "EvolutionPipeline",
    "SimpleEvolutionPipeline",
    "ConfiguredTrainingPipeline",
    "ManagePipeline",
    "GraphManagePipeline",
    "OrganizeRequest",
    "OrganizeResult",
    "RecommendRequest",
    "RecommendResult",
    "TrainingPipeline",
    "SkillPipelineRequest",
    "AlgoSkillEvolveResult",
    "TrainingRequest",
    "TrainingSplit",
    "RepositoryPipeline",
]
