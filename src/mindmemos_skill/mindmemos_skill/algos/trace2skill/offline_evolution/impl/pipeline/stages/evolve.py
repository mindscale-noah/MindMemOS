"""Evolve: one read+judge+edit agent session per mutable scope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_int
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import display_artifact
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.editor.scope_workflow import (
    run_scope_evolve_agents,
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
class EvolveResult(StageResult):
    """evolve: investigation + skill editing, one session per scope."""

    scope_outcomes: list[dict[str, Any]] = field(default_factory=list)
    evolve_summary: dict[str, Any] = field(default_factory=dict)

    def validate(self, ctx: StageContext) -> None:
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.state import _require_file

        for outcome in self.scope_outcomes:
            scope_id = str(outcome.get("scope_id") or "")
            if scope_id:
                _require_file(ctx.state, f"review/{scope_id}.json", stage_id="evolve")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_outcomes": list(self.scope_outcomes),
            "evolve_summary": dict(self.evolve_summary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvolveResult":
        return cls(
            scope_outcomes=list(payload.get("scope_outcomes", [])),
            evolve_summary=dict(payload.get("evolve_summary", {})),
        )


@register_stage("evolve")
class EvolveStage(PipelineStage):
    """evolve: one autonomous read+edit Agent per mutable scope."""

    title = "evolve scope skill libraries"
    result_type = EvolveResult

    def run(self, ctx: StageContext) -> EvolveResult:
        paths = ctx.paths
        run_dir = ctx.run_dir
        logger = ctx.logger
        state = ctx.state
        staging_payload = (state.get("results") or {}).get("staging")
        if not isinstance(staging_payload, dict):
            raise ValueError("evolve requires the persisted staging result")
        scope_ids = StagingResult.from_dict(staging_payload).scope_ids

        parallelism = setting_int("parallelism.evolve")
        skill_md_only = bool(state["settings"]["skill_md_only"])
        _log(logger, "")
        _log(logger, f"skills : {_display_path(display_artifact(paths.workspace_dir, 'skills'), root=run_dir)}")
        _log(logger, f"parallelism={parallelism}")
        _log(logger, f"skill_md_only={skill_md_only}")

        results = run_scope_evolve_agents(
            paths.workspace_dir,
            scope_ids=scope_ids,
            baseline_root=paths.skills_baseline_dir,
            llm_parallelism=parallelism,
            skill_md_only=skill_md_only,
            review_root=paths.review_dir,
        )
        outcome_fields = (
            "scope_id",
            "status",
            "error",
            "attempts",
            "validation_retries",
            "validation",
            "constraint_violations",
        )
        scope_outcomes = [
            {field: record.get(field) for field in outcome_fields}
            for record in results.get("scopes", [])
            if isinstance(record, dict)
        ]
        return EvolveResult(
            scope_outcomes=scope_outcomes,
            evolve_summary=results["summary"],
        )
