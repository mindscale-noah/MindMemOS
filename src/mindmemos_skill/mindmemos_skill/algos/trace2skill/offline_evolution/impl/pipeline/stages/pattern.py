"""Pattern: build family/adoption/cluster structures per scope. Zero LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_int
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import display_artifact
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.evidence.patterns import (
    PatternsConfig,
    build_patterns_evidence,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.logging import _display_path, _log
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stage import (
    PipelineStage,
    StageContext,
    StageResult,
    register_stage,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stages.staging import StagingResult


@dataclass
class PatternResult(StageResult):
    """pattern: families / adoption / behavior clusters."""

    patterns_summary: dict[str, Any] = field(default_factory=dict)

    def validate(self, ctx: StageContext) -> None:
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.state import _require_file

        scope_ids = list((self.patterns_summary.get("scopes") or {}).keys())
        if not scope_ids:
            raise RuntimeError("pattern recorded no scope outputs")
        for scope_id in scope_ids:
            _require_file(ctx.state, f"workspace/{scope_id}/patterns.json", stage_id="pattern")

    def to_dict(self) -> dict[str, Any]:
        return {"patterns_summary": dict(self.patterns_summary)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatternResult":
        return cls(patterns_summary=dict(payload.get("patterns_summary", {})))


@register_stage("pattern")
class PatternStage(PipelineStage):
    """pattern: build families, adoption tables, and behavior clusters."""

    title = "build family/adoption/cluster structures"
    result_type = PatternResult

    def run(self, ctx: StageContext) -> PatternResult:
        paths = ctx.paths
        run_dir = ctx.run_dir
        logger = ctx.logger

        workspace_root = Path(paths.workspace_dir)
        staging_payload = (ctx.state.get("results") or {}).get("staging")
        if not isinstance(staging_payload, dict):
            raise ValueError("pattern requires the persisted staging result")
        scope_ids = StagingResult.from_dict(staging_payload).scope_ids
        _log(logger, "")
        _log(logger, f"input  : {_display_path(display_artifact(paths.workspace_dir, 'extractions'), root=run_dir)}")
        _log(logger, f"output : {_display_path(str(workspace_root / '*' / 'patterns.json'), root=run_dir)}")

        config = PatternsConfig.from_settings()
        parallelism = setting_int("parallelism.pattern")
        summary = build_patterns_evidence(
            workspace_root=workspace_root,
            scope_ids=scope_ids,
            config=config,
            parallelism=parallelism,
        )
        return PatternResult(patterns_summary=summary.to_dict())
