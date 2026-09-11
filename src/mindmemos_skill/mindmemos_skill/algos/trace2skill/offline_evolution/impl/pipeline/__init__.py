"""End-to-end pipeline entrypoint."""

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.runner import run_pipeline
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import (
    PipelineConfig,
    PipelinePaths,
    PipelineResult,
)

__all__ = ["PipelineConfig", "PipelinePaths", "PipelineResult", "run_pipeline"]
