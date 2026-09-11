"""Extract per-trajectory index records with LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_int
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import (
    display_artifact,
    extraction_output_paths,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.evidence.extraction import (
    HybridConfig,
    build_extraction_records,
    summarize_extraction_results,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.logging import _display_path, _log
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stage import (
    PipelineStage,
    StageContext,
    StageResult,
    register_stage,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stages.staging import StagingResult
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.utils.parallel import TqdmProgress


@dataclass
class ExtractResult(StageResult):
    """extraction: per-trajectory index records."""

    extraction_paths: list[str] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)

    def validate(self, ctx: StageContext) -> None:
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.state import _require_file

        if not self.extraction_paths:
            raise RuntimeError("extract produced no usable records")
        for rel in self.extraction_paths:
            _require_file(ctx.state, rel, stage_id="extract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_paths": list(self.extraction_paths),
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtractResult":
        return cls(
            extraction_paths=list(payload.get("extraction_paths", [])),
            failures=list(payload.get("failures", [])),
        )


@register_stage("extract")
class ExtractStage(PipelineStage):
    """extraction: one LLM call per trajectory, then supervised score join."""

    title = "extract per-trajectory index records"
    result_type = ExtractResult

    def run(self, ctx: StageContext) -> ExtractResult:
        paths = ctx.paths
        run_dir = ctx.run_dir
        logger = ctx.logger
        prev_result: Any = ctx.prev_result

        if isinstance(prev_result, StagingResult):
            trajectory_paths = [str(Path(paths.run_dir) / rel) for rel in prev_result.workspace_trajectory_paths]
        else:
            raise RuntimeError(
                "extract: missing typed prev_result (state['results'].get('staging') "
                "is empty); cannot resume a run that predates the staging stage"
            )

        parallelism = setting_int("parallelism.extract")
        _log(logger, "")
        _log(
            logger,
            f"input       : {_display_path(display_artifact(paths.workspace_dir, 'trajectory'), root=run_dir)} ({len(trajectory_paths)} file(s))",
        )
        _log(
            logger,
            f"output      : {_display_path(display_artifact(paths.workspace_dir, 'extractions'), root=run_dir)}",
        )
        _log(logger, f"workers     : {parallelism}")

        config = HybridConfig.from_settings()
        ordered = sorted(
            trajectory_paths,
            key=lambda path: Path(path).stat().st_size,
            reverse=True,
        )
        outputs = extraction_output_paths(ordered, workspace_root=paths.workspace_dir)
        progress = TqdmProgress("extraction", len(ordered), enabled=logger is not None)
        try:
            batch = build_extraction_records(
                ordered,
                outputs,
                workspace_root=paths.workspace_dir,
                parallelism=parallelism,
                config=config,
                progress_callback=progress,
            )
        finally:
            progress.close()
        summary = summarize_extraction_results(batch)
        if not summary.output_paths:
            raise RuntimeError("extraction produced no usable records")
        extraction_paths = [
            Path(p).resolve().relative_to(Path(paths.run_dir).resolve()).as_posix() for p in summary.output_paths
        ]
        return ExtractResult(
            extraction_paths=extraction_paths,
            failures=summary.failures,
        )
