"""Types for the end-to-end offline pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class PipelinePaths:
    """Canonical artifact paths under one timestamped execution directory."""

    run_dir: str
    workspace_dir: str
    # Received skills/ trees kept outside the workspace; the only baseline
    # left for re-seeds and diffs once the editor mutates in place.
    skills_baseline_dir: str
    extractions_dir: str
    review_dir: str
    final_skills_dir: str
    final_report_md: str
    state_json: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_dir": self.run_dir,
            "workspace_dir": self.workspace_dir,
            "skills_baseline_dir": self.skills_baseline_dir,
            "extractions_dir": self.extractions_dir,
            "review_dir": self.review_dir,
            "final_skills_dir": self.final_skills_dir,
            "final_report_md": self.final_report_md,
            "state_json": self.state_json,
        }


@dataclass
class PipelineConfig:
    """Configuration for one complete pipeline run."""

    # Fields --resume treats as overrides; runner.py and scripts/run_pipeline.py
    # iterate this list directly.
    RESUME_OVERRIDE_FIELDS: ClassVar[tuple[str, ...]] = (
        "backend",
        "supervision_mode",
        "skill_md_only",
    )

    source_dir: str | None = None
    run_root_dir: str | None = None
    resume_run_dir: str | None = None
    # Fork mode: materialize the source run, then re-run from stage N.
    from_stage: int | None = None
    stop_after_stage: int | None = None
    backend: str | None = None
    # unsupervised never copies scores/ into the run workspace.
    supervision_mode: str | None = None
    # Restrict investigation to editing the SKILL.md files of existing Skills.
    skill_md_only: bool | None = None


@dataclass
class PipelineResult:
    """Summary returned after a pipeline run or intentional stop."""

    paths: PipelinePaths
    input_trajectory: list[str]
    counts: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": self.paths.to_dict(),
            "input_trajectory": list(self.input_trajectory),
            "counts": dict(self.counts),
            "meta": dict(self.meta),
        }


__all__ = ["PipelineConfig", "PipelinePaths", "PipelineResult"]
