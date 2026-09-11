"""Input validation and copying for source scopes."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import (
    SCORE_ARTIFACT,
    scope_artifact_dir,
    scope_workspace,
    score_file,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.state import RAW_TRAJECTORY_DIRNAME
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelinePaths


def _iter_source_scopes(source_root: Path) -> list[Path]:
    """List top-level scope directories from the raw source root.

    Trajectory file format is opaque; contents are not filtered by suffix.
    """
    if not source_root.is_dir():
        raise FileNotFoundError(f"source_dir not found or not a directory: {source_root}")
    scopes = [
        path
        for path in sorted(source_root.iterdir())
        if (path.is_dir() and not path.name.startswith(".") and (path / RAW_TRAJECTORY_DIRNAME).is_dir())
    ]
    if not scopes:
        raise FileNotFoundError(
            f"source_dir has no scope directories containing {RAW_TRAJECTORY_DIRNAME}/: {source_root}"
        )
    return scopes


def _scope_scores(scope_dir: Path, trajectories: list[Path]) -> dict[str, Path]:
    """Map trajectory stem -> score file for the scored subset of a scope.

    Coverage may be partial: an unscored trajectory is unknown, not zero.
    """
    scores_dir = scope_dir / SCORE_ARTIFACT
    if not scores_dir.is_dir():
        raise FileNotFoundError(
            f"supervised mode requires a scores/ directory in every source scope; missing: {scores_dir}"
        )
    found: dict[str, Path] = {}
    for trajectory in trajectories:
        candidate = score_file(scores_dir, trajectory.stem)
        if candidate.is_file():
            found[trajectory.stem] = candidate
    return found


def _validate_source_scope(scope_dir: Path) -> tuple[list[Path], list[Path]]:
    """Validate one raw scope directory; return trajectories and Skill dirs.

    Trajectory files are accepted as-is; only an empty ``trajectory/`` fails.
    """
    trajectory_dir = scope_dir / RAW_TRAJECTORY_DIRNAME
    skills_dir = scope_dir / "skills"
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"scope directory missing required {RAW_TRAJECTORY_DIRNAME}/: {scope_dir}")
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"scope directory missing required skills/: {scope_dir}")

    trajectories = [p for p in sorted(trajectory_dir.iterdir()) if p.is_file() and not p.name.startswith(".")]
    if not trajectories:
        raise FileNotFoundError(f"scope trajectory directory is empty: {trajectory_dir}")

    skill_dirs = [p for p in sorted(skills_dir.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    missing_skill_md = [p for p in skill_dirs if not (p / "SKILL.md").is_file()]
    if missing_skill_md:
        examples = ", ".join(str(p) for p in missing_skill_md[:5])
        raise FileNotFoundError(f"skill directory missing SKILL.md: {examples}")
    return trajectories, skill_dirs


def _copy_skills(skills_dir: Path, dst: Path) -> None:
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"skills_dir not found or not a directory: {skills_dir}")
    if dst.exists():
        raise FileExistsError(f"destination skills directory already exists: {dst}")

    def _ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {"__pycache__", ".DS_Store"} or name.endswith(".pyc")}

    shutil.copytree(skills_dir, dst, ignore=_ignore)


def validate_source_layout(
    source_dir: str | Path,
    *,
    supervision_mode: str,
) -> None:
    """Verify ``source_dir`` conforms to the scope-first layout.

    Supervised mode additionally requires ``scores/<stem>.txt`` for every
    trajectory; unsupervised mode ignores any ``scores/`` in the source.
    """
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import SCORE_ARTIFACT

    source = Path(source_dir)
    if not source.is_dir():
        raise ValueError(f"source_dir not found or not a directory: {source}")
    supervised = supervision_mode == "supervised"

    scope_dirs = [path for path in sorted(source.iterdir()) if path.is_dir() and not path.name.startswith(".")]
    if not scope_dirs:
        raise ValueError(f"source_dir has no scope directories: {source}")

    for scope_dir in scope_dirs:
        trajectory_dir = scope_dir / RAW_TRAJECTORY_DIRNAME
        if not trajectory_dir.is_dir():
            raise ValueError(f"scope {scope_dir.name!r}: missing required {RAW_TRAJECTORY_DIRNAME}/")
        if not any(trajectory_dir.iterdir()):
            raise ValueError(f"scope {scope_dir.name!r}: {RAW_TRAJECTORY_DIRNAME}/ is empty")

        skills_dir = scope_dir / "skills"
        if not skills_dir.is_dir():
            raise ValueError(f"scope {scope_dir.name!r}: missing required skills/")
        skill_md_files = list(skills_dir.glob("*/SKILL.md"))
        if not skill_md_files:
            raise ValueError(f"scope {scope_dir.name!r}: skills/ has no */SKILL.md")

        if supervised:
            scores_dir = scope_dir / SCORE_ARTIFACT
            if not scores_dir.is_dir():
                raise ValueError(
                    f"supervised mode requires a scores/ directory in every source scope; missing: {scores_dir}"
                )
            for trajectory in trajectory_dir.iterdir():
                if trajectory.is_file() and not (scores_dir / f"{trajectory.stem}.txt").exists():
                    raise ValueError(
                        f"supervised mode requires a scores/ file matching every "
                        f"trajectory; missing for {trajectory.name!r}"
                    )


def _copy_scope_source(
    source_root: Path, paths: PipelinePaths, *, supervision_mode: str = "unsupervised"
) -> dict[str, Any]:
    """Copy source scopes into per-scope run directories keyed by source name.

    Scope directory names come verbatim from the extraction scripts; anything
    the filesystem rejects fails loudly here. Supervised runs copy score text
    verbatim for extraction outcome matching; unsupervised runs do not copy
    scores.
    """
    supervised = supervision_mode == "supervised"
    workspace_root = Path(paths.workspace_dir)
    baseline_root = Path(paths.skills_baseline_dir)
    workspace_root.mkdir(parents=True, exist_ok=False)
    baseline_root.mkdir(parents=True, exist_ok=False)

    scopes_input = _iter_source_scopes(source_root)
    scope_ids: list[str] = []
    copied_trajectory: list[dict[str, Any]] = []
    total_trajectories = 0
    total_skills = 0
    scored_count = 0

    for scope_dir in scopes_input:
        scope_id = scope_dir.name
        scope_ids.append(scope_id)

        trajectories, skill_dirs = _validate_source_scope(scope_dir)
        scores = _scope_scores(scope_dir, trajectories) if supervised else {}

        total_trajectories += len(trajectories)
        total_skills += len(skill_dirs)
        scored_count += len(scores)

        scope_ws = scope_workspace(workspace_root, scope_id)
        scope_ws.mkdir(parents=True, exist_ok=False)

        trajectory_out = scope_artifact_dir(workspace_root, scope_id, "trajectory")
        trajectory_out.mkdir(parents=True, exist_ok=False)
        if supervised:
            scores_out = scope_artifact_dir(workspace_root, scope_id, SCORE_ARTIFACT)
            scores_out.mkdir()
        for traj in trajectories:
            shutil.copy2(traj, trajectory_out / traj.name)
            stem = traj.stem
            score_path = scores.get(stem)
            trajectory_record: dict[str, Any] = {
                "input_path": str(traj.relative_to(source_root)),
                "output_path": str((trajectory_out / traj.name).relative_to(Path(paths.run_dir))),
                "scope_id": scope_id,
            }
            if score_path:
                trajectory_record["score_input"] = str(score_path.relative_to(source_root))
                # Verbatim score text; never coerce to float.
                trajectory_record["score"] = score_path.read_text(encoding="utf-8").strip()
                shutil.copy2(score_path, score_file(scores_out, stem))
            copied_trajectory.append(trajectory_record)

        skills_out = scope_ws / "skills"
        _copy_skills(scope_dir / "skills", skills_out)
        baseline_out = baseline_root / scope_id
        _copy_skills(scope_dir / "skills", baseline_out)

    return {
        "scope_ids": scope_ids,
        "copied_trajectory": copied_trajectory,
        "scopes": len(scope_ids),
        "skills": total_skills,
        "scored_trajectories": scored_count,
        "unscored_trajectories": total_trajectories - scored_count,
    }


__all__ = [
    "_iter_source_scopes",
    "_scope_scores",
    "_validate_source_scope",
    "_copy_skills",
    "_copy_scope_source",
    "validate_source_layout",
]
