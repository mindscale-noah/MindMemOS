"""Stage state machine and resumable ``state.json`` I/O.

Persisted artifact paths stay relative to ``state["run_dir"]`` so resume
and fork from a relocated archive need no path rewriting.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stage import PIPELINE_ORDER, PREPARE_STAGE
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelineConfig, PipelinePaths
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.utils.files import read_json_file, write_json_file

RAW_TRAJECTORY_DIRNAME = "trajectory"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _initial_state(
    *,
    paths: PipelinePaths,
    run_root: Path,
    source_dir: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    now = _now_iso()
    return {
        "status": "running",
        "created_at": now,
        "updated_at": now,
        # run_dir anchors every relative artifact path below; a fork rewrites it.
        "run_dir": str(Path(paths.run_dir).resolve()),
        "run_root": str(run_root.resolve()),
        "source": {
            "source_dir": source_dir,
        },
        "settings": dict(settings),
        "stages": {stage_id: {"status": "pending"} for stage_id in PIPELINE_ORDER},
        # Typed StageResult per stage, populated by _mark_stage_done(result=...).
        "results": {},
    }


def _write_state(paths: PipelinePaths, state: dict[str, Any]) -> None:
    state["updated_at"] = _now_iso()
    write_json_file(paths.state_json, state)


def _load_state(paths: PipelinePaths) -> dict[str, Any]:
    if not Path(paths.state_json).is_file():
        raise FileNotFoundError(f"resume workspace has no state.json: {paths.state_json}")
    state = read_json_file(paths.state_json)
    if not isinstance(state, dict):
        raise ValueError(f"{paths.state_json}: expected top-level object")
    return state


def _stage_is_done(state: dict[str, Any], stage_id: str) -> bool:
    stage = state.get("stages", {}).get(stage_id, {})
    return isinstance(stage, dict) and stage.get("status") == "done"


def _first_incomplete_stage(state: dict[str, Any]) -> str | None:
    first: str | None = None
    for stage_id in PIPELINE_ORDER:
        if not _stage_is_done(state, stage_id):
            first = stage_id
            break
    if first is None:
        return None

    first_index = PIPELINE_ORDER.index(first)
    later_done = [stage_id for stage_id in PIPELINE_ORDER[first_index + 1 :] if _stage_is_done(state, stage_id)]
    if later_done:
        raise RuntimeError(
            f"state.json is inconsistent: later stages are done after first incomplete stage {first!r}: {later_done}"
        )
    return first


def _mark_stage_running(state: dict[str, Any], paths: PipelinePaths, stage_id: str) -> None:
    state.setdefault("stages", {})[stage_id] = {
        "status": "running",
        "updated_at": _now_iso(),
    }
    state["status"] = "running"
    _write_state(paths, state)


def _mark_stage_done(
    state: dict[str, Any],
    paths: PipelinePaths,
    stage_id: str,
    *,
    elapsed_sec: float,
    result: Any = None,
) -> None:
    state.setdefault("stages", {})[stage_id] = {
        "status": "done",
        "duration_sec": round(elapsed_sec, 3),
        "updated_at": _now_iso(),
    }
    if result is not None:
        state.setdefault("results", {})[stage_id] = result.to_dict()
    _write_state(paths, state)


def _resolve(state: dict[str, Any], value: str) -> Path:
    """Resolve a state-relative path against the recorded run dir."""
    return (Path(state["run_dir"]) / value).resolve()


def _require_file(state: dict[str, Any], value: str, *, stage_id: str) -> None:
    path = _resolve(state, value)
    if not path.is_file():
        raise FileNotFoundError(f"{stage_id} is marked done but file is missing: {path}")


def _require_dir(state: dict[str, Any], value: str, *, stage_id: str) -> None:
    path = _resolve(state, value)
    if not path.is_dir():
        raise FileNotFoundError(f"{stage_id} is marked done but directory is missing: {path}")


def _validate_made_progress(stage_id: str, state: dict[str, Any], paths: PipelinePaths) -> None:
    """Re-run a done stage's ``result.validate()`` against the live filesystem (resume check)."""
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stage import StageContext, get_stage_class

    payload = (state.get("results") or {}).get(stage_id)
    if not isinstance(payload, dict):
        raise ValueError(f"stage {stage_id!r} marked done but state['results'] is empty")
    try:
        stage_cls = get_stage_class(stage_id)
    except KeyError as exc:
        raise ValueError(f"stage {stage_id!r} marked done but no class is registered for it") from exc
    if stage_cls.result_type is None:
        return
    result = stage_cls.result_type.from_dict(payload)
    ctx = StageContext(
        state=state,
        paths=paths,
        run_dir=Path(paths.run_dir),
        logger=None,
        prev_result=None,
        settings=None,
    )
    result.validate(ctx)


def _validate_completed_prefix(
    state: dict[str, Any],
    paths: PipelinePaths,
    first_incomplete: str | None,
) -> None:
    """Re-validate every stage marked done up to ``first_incomplete``."""
    stop_index = len(PIPELINE_ORDER) if first_incomplete is None else PIPELINE_ORDER.index(first_incomplete)
    for stage_id in PIPELINE_ORDER[:stop_index]:
        if not state.get("stages", {}).get(stage_id, {}).get("status") == "done":
            continue
        _validate_made_progress(stage_id, state, paths)


def _last_done_stage(state: dict[str, Any]) -> str | None:
    """Stage ID of the latest done stage in :data:`PIPELINE_ORDER`, else ``None``."""
    for stage_id in reversed(PIPELINE_ORDER):
        if _stage_is_done(state, stage_id):
            return stage_id
    return None


def _ensure_new_run_config(config: PipelineConfig) -> None:
    if not config.source_dir:
        raise ValueError("source_dir is required for a new pipeline run")
    if not config.run_root_dir:
        raise ValueError("--run-dir is required for a new pipeline run")


def _ensure_resume_has_no_overrides(config: PipelineConfig) -> None:
    overrides = []
    if config.source_dir is not None:
        overrides.append("source_dir")
    if config.run_root_dir is not None:
        overrides.append("run_root_dir")
    for name in PipelineConfig.RESUME_OVERRIDE_FIELDS:
        if getattr(config, name) is not None:
            overrides.append(name)
    if overrides:
        raise ValueError(
            f"--resume reads configuration from state.json; do not provide overrides: {', '.join(overrides)}"
        )


def _read_json_if_file(path: str | Path) -> Any | None:
    if Path(path).is_file():
        return read_json_file(path)
    return None


def _stage_output_paths(paths: PipelinePaths, stage_id: str) -> list[str]:
    """Output artifacts (relative to run_root) a fork deletes when resetting ``stage_id``.

    Tail outputs only; prefix inputs and the workspace tree itself are never
    deleted -- the workspace is working state shared across stages.
    """
    workspace = Path(paths.workspace_dir).relative_to(paths.run_dir).as_posix()
    return {
        "extract": [f"{workspace}/*/extractions"],
        "pattern": [
            f"{workspace}/*/patterns.json",
        ],
        "evolve": [
            "review",
        ],
        "finalize": [
            "final_skills",
            "final_report.md",
        ],
    }.get(stage_id, [])


__all__ = [
    "RAW_TRAJECTORY_DIRNAME",
    "PREPARE_STAGE",
    "_now_iso",
    "_initial_state",
    "_write_state",
    "_load_state",
    "_stage_is_done",
    "_first_incomplete_stage",
    "_mark_stage_running",
    "_mark_stage_done",
    "_resolve",
    "_require_file",
    "_require_dir",
    "_validate_made_progress",
    "_validate_completed_prefix",
    "_ensure_new_run_config",
    "_ensure_resume_has_no_overrides",
    "_read_json_if_file",
    "_last_done_stage",
    "_stage_output_paths",
]
