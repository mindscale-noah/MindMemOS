"""Fork materialization: reuse completed stages from a prior run."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.logging import LogFn, _log
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare.paths import (
    _build_paths,
    _create_timestamped_run_dir,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare.shared import _remove_tree_under_run_dir
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stage import (
    MIN_FORK_STAGE,
    PIPELINE_ORDER,
    STAGE_ID_BY_NUMBER,
    STAGE_NUMBER_BY_ID,
    TOTAL_USER_STAGES,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.state import (
    _first_incomplete_stage,
    _load_state,
    _stage_output_paths,
    _validate_completed_prefix,
    _write_state,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelinePaths


def _fork_artifact_stage(relative_path: Path) -> int | None:
    """Return the stage that owns a canonical fork artifact, when known."""
    parts = relative_path.parts
    if not parts:
        return None
    if parts[0] == "workspace" and len(parts) >= 3:
        return {
            "extractions": 1,
            "patterns.json": 2,
        }.get(parts[2])
    return {
        "review": 3,
        "final_skills": 4,
        "final_report.md": 4,
    }.get(parts[0])


def _fork_copy_ignore(
    source_run_dir: Path,
    *,
    from_stage: int,
    stats: dict[str, int],
    reference_dirs: set[Path],
) -> Any:
    """Skip generated tail artifacts and separately linked directories."""
    source_root = source_run_dir.resolve()

    def _ignore(directory: str, names: list[str]) -> set[str]:
        parent = Path(directory).relative_to(source_root)
        ignored: set[str] = set()
        for name in names:
            relative = parent / name
            if name in {"__pycache__", ".DS_Store"} or name.endswith(".pyc"):
                ignored.add(name)
                continue
            if relative in reference_dirs:
                ignored.add(name)
                continue
            owner_stage = _fork_artifact_stage(relative)
            if owner_stage is not None and owner_stage >= from_stage:
                ignored.add(name)
                stats["skipped_tail_entries"] += 1
        return ignored

    return _ignore


def _fork_reference_dirs(source_run_dir: Path, *, from_stage: int) -> list[Path]:
    """Return immutable prefix directories that a fork can reference directly."""
    candidates: list[tuple[Path, int]] = [(Path("trajectory"), 0)]
    return sorted(
        relative for relative, stage in candidates if stage < from_stage and (source_run_dir / relative).is_dir()
    )


def _fork_file_copy(stats: dict[str, int]) -> Any:
    """Copy the small set of control/editable files retained by a fork."""

    def _copy(source: str, target: str) -> str:
        copied = shutil.copy2(source, target)
        stats["copied_files"] += 1
        return copied

    return _copy


def _materialize_directory_refs(
    source_run_dir: Path,
    new_run_dir: Path,
    reference_dirs: list[Path],
    stats: dict[str, Any],
) -> None:
    """Create read-only directory symlinks, copying only when unsupported."""
    refs: list[dict[str, str]] = []
    for relative in reference_dirs:
        source = source_run_dir / relative
        reference_target = source.resolve()
        target = new_run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(reference_target, target_is_directory=True)
            stats["linked_directories"] += 1
            refs.append(
                {
                    "path": relative.as_posix(),
                    "target": str(reference_target),
                }
            )
        except OSError:
            shutil.copytree(source, target)
            stats["fallback_directories"] += 1
    stats["directory_refs"] = refs


def materialize_fork(
    *,
    source_run_dir: Path,
    run_root: Path,
    from_stage: int,
    skill_md_only: bool | None = None,
    logger: LogFn | None = None,
) -> tuple[Path, PipelinePaths, dict[str, Any]]:
    """Reference a prior run from a fresh timestamped dir and reset stages >= N.

    Returns ``(new_run_dir, paths, state)`` ready for the resume loop. The
    source run is never mutated: editable files are copied, immutable history
    is symlinked when the platform allows. Only an investigation-stage fork
    may override ``skill_md_only``.
    """
    if from_stage < MIN_FORK_STAGE or from_stage > TOTAL_USER_STAGES:
        raise ValueError(f"--from-stage must be in [{MIN_FORK_STAGE}, {TOTAL_USER_STAGES}], got {from_stage}")
    if skill_md_only is not None and from_stage != 3:
        raise ValueError("skill_md_only can only be overridden by an evolve-stage fork")
    source_run_dir = source_run_dir.resolve()
    source_paths = _build_paths(source_run_dir)
    source_state = _load_state(source_paths)

    first_incomplete = _first_incomplete_stage(source_state)
    if first_incomplete is not None:
        incomplete_number = STAGE_NUMBER_BY_ID.get(first_incomplete, 0)
        if incomplete_number < from_stage:
            raise ValueError(f"cannot fork from stage {from_stage}: source stage {first_incomplete!r} is not done")
    from_stage_id = STAGE_ID_BY_NUMBER[from_stage]
    _validate_completed_prefix(source_state, source_paths, from_stage_id)

    new_run_dir = _create_timestamped_run_dir(run_root)
    reference_dirs = _fork_reference_dirs(source_run_dir, from_stage=from_stage)
    copy_stats: dict[str, Any] = {
        "linked_directories": 0,
        "copied_files": 0,
        "fallback_directories": 0,
        "skipped_tail_entries": 0,
    }
    _log(logger, f"  fork refs : {source_run_dir} -> {new_run_dir}")
    shutil.copytree(
        source_run_dir,
        new_run_dir,
        dirs_exist_ok=True,
        ignore=_fork_copy_ignore(
            source_run_dir,
            from_stage=from_stage,
            stats=copy_stats,
            reference_dirs=set(reference_dirs),
        ),
        copy_function=_fork_file_copy(copy_stats),
    )
    _materialize_directory_refs(source_run_dir, new_run_dir, reference_dirs, copy_stats)

    paths = _build_paths(new_run_dir)
    state = _load_state(paths)
    state["run_dir"] = str(new_run_dir.resolve())
    state["run_root"] = str(run_root.resolve())
    state["forked_from"] = str(source_run_dir)
    state["from_stage"] = from_stage
    if skill_md_only is not None:
        state.setdefault("settings", {})["skill_md_only"] = bool(skill_md_only)
    state["fork_materialization"] = {
        "mode": "directory_symlink_with_copy_fallback",
        **copy_stats,
    }

    # Prepare is index 0, so the user-facing stage number equals its PIPELINE_ORDER index.
    for stage_id in PIPELINE_ORDER[from_stage:]:
        state.setdefault("stages", {})[stage_id] = {"status": "pending"}
        # Drop typed results for re-run stages; a stale prev_result would name deleted files.
        state.setdefault("results", {}).pop(stage_id, None)
        for pattern in _stage_output_paths(paths, stage_id):
            # Patterns may hold per-scope globs; expand them or the literal ``*`` deletes nothing.
            for path in new_run_dir.glob(pattern):
                _remove_tree_under_run_dir(path, new_run_dir)
    _write_state(paths, state)
    _log(
        logger,
        "  fork materialization: "
        f"directory_links={copy_stats['linked_directories']} "
        f"copied_files={copy_stats['copied_files']} "
        f"fallback_directories={copy_stats['fallback_directories']} "
        f"skipped_tail_entries={copy_stats['skipped_tail_entries']}",
    )
    return new_run_dir, paths, state


__all__ = ["materialize_fork"]
