"""Path construction and validation for pipeline runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare.shared import _remove_tree_under_run_dir
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelinePaths


def _create_timestamped_run_dir(run_root: Path) -> Path:
    resolved_root = run_root.resolve()
    anchor = Path(resolved_root.anchor).resolve()
    if resolved_root == anchor:
        raise ValueError(f"refuse to use filesystem root as run_dir: {run_root}")

    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for i in range(100):
        name = timestamp if i == 0 else f"{timestamp}_{i:02d}"
        candidate = run_root / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a unique timestamped run directory under: {run_root}")


def _build_paths(run_dir: Path) -> PipelinePaths:
    """Construct the canonical layout for a run directory.

    Everything a scope owns lives under ``workspace/<scope>/``; the run
    root keeps only products, ``state.json``, and the immutable ``baseline/``.
    """
    return PipelinePaths(
        run_dir=str(run_dir),
        workspace_dir=str(run_dir / "workspace"),
        skills_baseline_dir=str(run_dir / "baseline"),
        extractions_dir=str(run_dir / "workspace"),
        review_dir=str(run_dir / "review"),
        final_skills_dir=str(run_dir / "final_skills"),
        final_report_md=str(run_dir / "final_report.md"),
        state_json=str(run_dir / "state.json"),
    )


def _reset_prepare_outputs(paths: PipelinePaths, run_dir: Path) -> None:
    for raw in (
        paths.workspace_dir,
        paths.review_dir,
        paths.final_skills_dir,
        paths.final_report_md,
    ):
        path = Path(raw)
        if path.exists():
            _remove_tree_under_run_dir(path, run_dir)


__all__ = ["_create_timestamped_run_dir", "_build_paths", "_reset_prepare_outputs"]
