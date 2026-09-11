"""Staging: copy source scopes into the run workspace (prepare+preprocess merged).

The former preprocess stage had degenerated to an idempotent byte-for-byte
copy of files prepare had already placed; both are now one stage. Output is
input materialization only -- no analysis artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import trajectory_output_paths
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.logging import _log
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare.inputs import _copy_scope_source
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare.paths import _reset_prepare_outputs
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare.shared import _is_relative_to
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stage import (
    PipelineStage,
    StageContext,
    StageResult,
    register_stage,
)


@dataclass
class StagingResult(StageResult):
    """Copy source scopes into the run workspace."""

    workspace_trajectory_paths: list[str] = field(default_factory=list)
    scope_ids: list[str] = field(default_factory=list)
    scope_count: int = 0
    skill_count: int = 0
    scored_count: int = 0
    unscored_count: int = 0

    def validate(self, ctx: StageContext) -> None:
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.state import _require_dir, _require_file

        _require_dir(ctx.state, "workspace", stage_id="staging")
        _require_dir(ctx.state, "baseline", stage_id="staging")
        if len(self.scope_ids) != self.scope_count:
            raise ValueError("staging scope_ids does not match scope_count")
        if not self.workspace_trajectory_paths:
            raise RuntimeError("staging produced no usable trajectory files")
        for rel in self.workspace_trajectory_paths:
            _require_file(ctx.state, rel, stage_id="staging")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_trajectory_paths": list(self.workspace_trajectory_paths),
            "scope_ids": list(self.scope_ids),
            "scope_count": self.scope_count,
            "skill_count": self.skill_count,
            "scored_count": self.scored_count,
            "unscored_count": self.unscored_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StagingResult":
        return cls(
            workspace_trajectory_paths=list(payload.get("workspace_trajectory_paths", [])),
            scope_ids=[str(value) for value in payload.get("scope_ids", [])],
            scope_count=int(payload.get("scope_count", 0)),
            skill_count=int(payload.get("skill_count", 0)),
            scored_count=int(payload.get("scored_count", 0)),
            unscored_count=int(payload.get("unscored_count", 0)),
        )


@register_stage("staging")
class StagingStage(PipelineStage):
    """Copy source scopes into the run workspace."""

    title = "stage inputs"
    result_type = StagingResult

    def run(self, ctx: StageContext) -> StagingResult:
        state = ctx.state
        paths = ctx.paths
        run_dir = ctx.run_dir
        logger = ctx.logger

        _log(logger, "")
        _log(logger, "Stage inputs")

        source = state["source"]
        source_root = Path(source["source_dir"]).resolve()
        run_root = Path(state["run_root"])
        resolved_run_root = run_root.resolve()
        if _is_relative_to(resolved_run_root, source_root):
            raise ValueError("run_dir must not be inside source_dir; copying inputs would risk recursive artifacts")
        if _is_relative_to(source_root, resolved_run_root):
            raise ValueError(
                "source_dir must not be inside run_dir; raw inputs should not be mixed with execution artifacts"
            )

        _reset_prepare_outputs(paths, run_dir)
        supervision_mode = str(state["settings"]["supervision_mode"])
        copy_artifacts = _copy_scope_source(source_root, paths, supervision_mode=supervision_mode)

        copied_trajectory = copy_artifacts["copied_trajectory"]
        staged = trajectory_output_paths(
            [str(run_dir / t["output_path"]) for t in copied_trajectory],
            workspace_root=paths.workspace_dir,
        )
        run_resolved = Path(paths.run_dir).resolve()
        workspace_trajectory_paths = [Path(path).resolve().relative_to(run_resolved).as_posix() for path in staged]

        _log(logger, f"  scopes       : {copy_artifacts['scopes']}")
        _log(logger, f"  trajectories : {len(copied_trajectory)} file(s)")
        _log(logger, f"  skills       : {copy_artifacts['skills']}")
        _log(logger, f"  supervision  : {supervision_mode}")
        if supervision_mode == "supervised":
            _log(
                logger,
                f"  scores       : {copy_artifacts['scored_trajectories']} scored, "
                f"{copy_artifacts['unscored_trajectories']} unknown",
            )

        _log(logger, "✓ Stage inputs complete")
        return StagingResult(
            workspace_trajectory_paths=workspace_trajectory_paths,
            scope_ids=copy_artifacts["scope_ids"],
            scope_count=copy_artifacts["scopes"],
            skill_count=copy_artifacts["skills"],
            scored_count=copy_artifacts["scored_trajectories"],
            unscored_count=copy_artifacts["unscored_trajectories"],
        )
