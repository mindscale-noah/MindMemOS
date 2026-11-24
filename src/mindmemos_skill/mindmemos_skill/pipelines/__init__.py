"""Shared pipeline contracts and explicit built-in registrations."""

from ..api.typing import (
    AlgoSkillEvolveResult,
    EvolutionRequest,
    OrganizeRequest,
    OrganizeResult,
    RecommendRequest,
    RecommendResult,
    TrainingRequest,
    TrainingSplit,
)
from .analyze import AnalyzePipeline
from .base import Pipeline
from .builtin_pipelines.analyze.pipeline import BuiltinAnalyzePipeline
from .builtin_pipelines.manage.graph_manage import GraphManagePipeline
from .builtin_pipelines.offline_evolution import OfflineEvolutionPipeline
from .builtin_pipelines.repository.repository import RepositoryPipeline
from .builtin_pipelines.training.training import ConfiguredTrainingPipeline
from .evolve import EvolutionPipeline
from .manage import ManagePipeline
from .train import TrainingPipeline

__all__ = [
    "OfflineEvolutionPipeline",
    "Pipeline",
    "AnalyzePipeline",
    "BuiltinAnalyzePipeline",
    "EvolutionPipeline",
    "ConfiguredTrainingPipeline",
    "ManagePipeline",
    "GraphManagePipeline",
    "OrganizeRequest",
    "OrganizeResult",
    "RecommendRequest",
    "RecommendResult",
    "TrainingPipeline",
    "EvolutionRequest",
    "AlgoSkillEvolveResult",
    "TrainingRequest",
    "TrainingSplit",
    "RepositoryPipeline",
]
