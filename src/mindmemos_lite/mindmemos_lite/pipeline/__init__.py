"""Pipeline contracts and registration helpers."""

from .base import (
    AddPipeline,
    DeletePipeline,
    DreamingPipeline,
    FeedbackPipeline,
    GetPipeline,
    MemoryPersistencePipelineMixin,
    PipelineBase,
    SearchPipeline,
    SkillEvolvePipeline,
    UpdatePipeline,
)
from .registry import PipelineType, create_pipeline, load_builtin_pipelines, pipeline_requires_task, register

__all__ = [
    "AddPipeline",
    "DeletePipeline",
    "DreamingPipeline",
    "FeedbackPipeline",
    "GetPipeline",
    "MemoryPersistencePipelineMixin",
    "PipelineBase",
    "PipelineType",
    "SearchPipeline",
    "SkillEvolvePipeline",
    "UpdatePipeline",
    "create_pipeline",
    "load_builtin_pipelines",
    "pipeline_requires_task",
    "register",
]
