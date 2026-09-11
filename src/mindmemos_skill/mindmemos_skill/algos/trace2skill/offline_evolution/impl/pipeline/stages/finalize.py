"""Finalize: format-gate and export edited skills.

The final report is NOT written here: it is a run-level closing artifact
the runner renders after this stage is marked done (final_report.md
covers every stage's timing, including finalize's own).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.export import export_final_skills
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.logging import _display_path, _log
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stage import (
    PipelineStage,
    StageContext,
    StageResult,
    register_stage,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stages.evolve import EvolveResult
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stages.staging import StagingResult


@dataclass
class FinalizeResult(StageResult):
    """finalize: per-scope export outcomes after the format gate."""

    final_skills_dir: str = ""
    exports: dict[str, Any] = field(default_factory=dict)

    def validate(self, ctx: StageContext) -> None:
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.state import _require_dir

        _require_dir(ctx.state, self.final_skills_dir, stage_id="finalize")

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_skills_dir": self.final_skills_dir,
            "exports": dict(self.exports),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FinalizeResult":
        return cls(
            final_skills_dir=str(payload.get("final_skills_dir", "")),
            exports=dict(payload.get("exports", {})),
        )


@register_stage("finalize")
class FinalizeStage(PipelineStage):
    """finalize: format-gate and export."""

    title = "finalize and export"
    result_type = FinalizeResult

    def run(self, ctx: StageContext) -> FinalizeResult:
        paths = ctx.paths
        run_dir = ctx.run_dir
        logger = ctx.logger
        staging_payload = (ctx.state.get("results") or {}).get("staging")
        if not isinstance(staging_payload, dict):
            raise ValueError("finalize requires the persisted staging result")
        scope_ids = StagingResult.from_dict(staging_payload).scope_ids
        if not isinstance(ctx.prev_result, EvolveResult):
            raise RuntimeError("finalize requires the evolve result")

        _log(logger, "")
        _log(logger, f"final skills : {_display_path(paths.final_skills_dir, root=run_dir)}")

        exports = export_final_skills(paths, run_dir, scope_ids=scope_ids)

        run_root = Path(paths.run_dir).resolve()
        return FinalizeResult(
            final_skills_dir=Path(paths.final_skills_dir).resolve().relative_to(run_root).as_posix(),
            exports=exports,
        )
