"""Pipeline runner: owns the stage lifecycle and threads typed results.

Runs each registered :class:`PipelineStage` in :data:`PIPELINE_ORDER`
order, validating and persisting each :class:`StageResult`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import display_artifact
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.logging import (
    LogFn,
    _display_path,
    _log,
    _stage_label,
    _stop_after_stage_id,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare import (
    PipelineSettings,
    _build_paths,
    _create_timestamped_run_dir,
    _is_relative_to,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stage import (
    PIPELINE_ORDER,
    STAGE_ID_BY_NUMBER,
    StageContext,
    get_stage_class,
    stage_index,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.state import (
    _ensure_new_run_config,
    _ensure_resume_has_no_overrides,
    _first_incomplete_stage,
    _initial_state,
    _last_done_stage,
    _load_state,
    _mark_stage_done,
    _mark_stage_running,
    _stage_is_done,
    _validate_completed_prefix,
    _write_state,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import (
    PipelineConfig,
    PipelinePaths,
    PipelineResult,
)


def _result_from_state(
    paths: PipelinePaths,
    state: dict[str, Any],
    *,
    status: str,
    final_summary: dict[str, Any] | None = None,
) -> PipelineResult:
    """Build a :class:`PipelineResult` from the persisted state.json."""
    results = state.get("results") or {}
    extract_payload = results.get("extract") or {}
    staging_payload = results.get("staging") or {}

    copied = staging_payload.get("workspace_trajectory_paths", []) or []
    extraction_paths = extract_payload.get("extraction_paths", []) or []

    counts: dict[str, Any] = {
        "trajectories": len(copied),
        "extraction_files": len(extraction_paths),
        "extraction_skipped": len(extract_payload.get("failures", []) or []),
    }
    counts.update(final_summary or {})
    return PipelineResult(
        paths=paths,
        input_trajectory=[str(p) for p in copied],
        counts=counts,
        meta={
            "status": status,
            "run_root": state.get("run_root"),
            "skills_dir": display_artifact(paths.workspace_dir, "skills"),
            "skills_baseline_dir": paths.skills_baseline_dir,
            "extractions_dir": paths.extractions_dir,
            "review_dir": paths.review_dir,
            "final_skills_dir": paths.final_skills_dir,
            "final_report_md": paths.final_report_md,
            "last_done_stage": _last_done_stage(state),
            "requested_stop_after_stage": state.get("requested_stop_after_stage"),
            "stopped_after_stage": state.get("stopped_after_stage"),
            "timings_sec": _stage_timings(state),
            "settings": dict(state.get("settings", {})),
        },
    )


def _stage_timings(state: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for stage_id in PIPELINE_ORDER:
        stage = state.get("stages", {}).get(stage_id, {})
        if not isinstance(stage, dict):
            continue
        duration = stage.get("duration_sec")
        if isinstance(duration, (int, float)):
            out[stage_id] = float(duration)
    total = sum(out.values())
    out["total"] = total
    return out


def _write_closing_report(state: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    """Render final_report.md after every stage (finalize included) is done.

    Run-level closing artifact: written here rather than inside the finalize
    stage so the report's timing table can truthfully show finalize itself.
    Idempotent -- resuming an already-complete run rewrites the same report.
    """
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.export import write_final_report

    results = state.get("results") or {}
    finalize_payload = results.get("finalize") or {}
    staging_payload = results.get("staging") or {}
    evolve_payload = results.get("evolve") or {}
    if not finalize_payload:
        return {}
    return write_final_report(
        paths,
        scope_ids=list(staging_payload.get("scope_ids", []) or []),
        scope_outcomes=list(evolve_payload.get("scope_outcomes", []) or []),
        exports=dict(finalize_payload.get("exports", {}) or {}),
    )


def _finalize_pipeline(
    *,
    state: dict[str, Any],
    paths: PipelinePaths,
    run_dir: Path,
    logger: LogFn | None,
) -> PipelineResult:
    final_summary = _write_closing_report(state, paths)
    result = _result_from_state(paths, state, status="complete", final_summary=final_summary)
    state["status"] = "complete"
    state["result"] = result.to_dict()
    _write_state(paths, state)
    _log(logger, "")
    _log(logger, "=== Pipeline complete ===")
    _log(logger, f"  state        : {_display_path(paths.state_json, root=run_dir)}")
    _log(logger, f"  final skills : {_display_path(paths.final_skills_dir, root=run_dir)}")
    _log(logger, f"  final report : {_display_path(paths.final_report_md, root=run_dir)}")
    _log(logger, f"  workspace    : {run_dir}")
    _log(logger, f"  total time   : {_stage_timings(state)['total']:.1f}s")
    return result


def _stop_pipeline(
    *,
    state: dict[str, Any],
    paths: PipelinePaths,
    run_dir: Path,
    requested_stop_stage_id: str,
    logger: LogFn | None,
) -> PipelineResult:
    _validate_completed_prefix(state, paths, requested_stop_stage_id)
    state["status"] = "stopped"
    state["requested_stop_after_stage"] = requested_stop_stage_id
    state["stopped_after_stage"] = _last_done_stage(state)
    _write_state(paths, state)
    result = _result_from_state(paths, state, status="stopped")
    _log(logger, "")
    _log(logger, "=== Pipeline stopped ===")
    _log(logger, f"  requested : {_stage_label(requested_stop_stage_id)}")
    _log(logger, f"  completed : {_stage_label(state['stopped_after_stage'])}")
    _log(logger, f"  workspace : {run_dir}")
    return result


def _rehydrate_prev_result(state: dict[str, Any], stage_id: str) -> Any:
    """Reconstruct the typed StageResult for ``stage_id`` from state.json, or ``None``."""
    payload = (state.get("results") or {}).get(stage_id)
    if not isinstance(payload, dict):
        return None
    try:
        stage_cls = get_stage_class(stage_id)
    except KeyError:
        return None
    if stage_cls.result_type is None:
        return None
    return stage_cls.result_type.from_dict(payload)


def run_pipeline(
    config: PipelineConfig,
    *,
    logger: LogFn | None = None,
) -> PipelineResult:
    """Run, resume, or fork the current state schema only."""
    # Importing the stages package registers every concrete Stage.
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline import stages as _stages_pkg  # noqa: F401

    fork = bool(config.resume_run_dir and config.run_root_dir and config.from_stage is not None)
    resume = bool(config.resume_run_dir and not fork)
    stop_after = _stop_after_stage_id(config.stop_after_stage)
    backend = ""
    if fork:
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare import materialize_fork

        source_run = Path(config.resume_run_dir or "").resolve()
        run_root = Path(config.run_root_dir or "")
        _log(logger, "=== Pipeline fork ===")
        _log(logger, f"  source    : {source_run}")
        _log(logger, f"  from stage: {_stage_label(STAGE_ID_BY_NUMBER[int(config.from_stage or 0)])}")
        run_dir, paths, state = materialize_fork(
            source_run_dir=source_run,
            run_root=run_root,
            from_stage=int(config.from_stage or 0),
            skill_md_only=config.skill_md_only,
            logger=logger,
        )
        settings = state["settings"]
    elif resume:
        _ensure_resume_has_no_overrides(config)
        run_dir = Path(config.resume_run_dir or "").resolve()
        paths = _build_paths(run_dir)
        state = _load_state(paths)
        settings = PipelineSettings.from_dict(state["settings"])
        _log(logger, "=== Pipeline resume ===")
        _log(logger, f"  workspace : {run_dir}")
    else:
        _ensure_new_run_config(config)
        settings = PipelineSettings.from_config(config)
        backend = settings.backend
        run_root = Path(config.run_root_dir or "")
        source_dir = Path(config.source_dir or "").resolve()
        if _is_relative_to(run_root.resolve(), source_dir):
            raise ValueError("run_dir must not be inside source_dir")
        # Validate source layout first: illegal input must not leave an empty run_dir.
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare.inputs import (
            validate_source_layout,
        )

        validate_source_layout(source_dir, supervision_mode=settings.supervision_mode)
        run_dir = _create_timestamped_run_dir(run_root)
        paths = _build_paths(run_dir)
        state = _initial_state(
            paths=paths,
            run_root=run_root,
            source_dir=str(source_dir),
            settings=settings.to_dict(),
        )
        _write_state(paths, state)
        _log(logger, "=== Pipeline run ===")
        _log(logger, f"  backend   : {backend}")
        _log(logger, f"  workspace : {run_dir}")

    first_incomplete = _first_incomplete_stage(state)
    _validate_completed_prefix(state, paths, first_incomplete)
    if first_incomplete is None:
        return _finalize_pipeline(state=state, paths=paths, run_dir=run_dir, logger=logger)
    if stop_after and _stage_is_done(state, stop_after):
        return _stop_pipeline(
            state=state,
            paths=paths,
            run_dir=run_dir,
            requested_stop_stage_id=stop_after,
            logger=logger,
        )
    if resume or fork:
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm import normalise_backend

        backend = normalise_backend(settings.backend)
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm import use_backend

    ctx = StageContext(
        state=state,
        paths=paths,
        run_dir=run_dir,
        logger=logger,
        prev_result=None,
        settings=settings,
    )

    with use_backend(
        backend,
        default_cwd=paths.workspace_dir,
    ):
        for stage_id in PIPELINE_ORDER:
            if _stage_is_done(state, stage_id):
                continue
            # On resume the previous stage was skipped; rehydrate its result.
            if ctx.prev_result is None:
                prev_index = stage_index(stage_id) - 1
                if prev_index >= 0:
                    prev_id = PIPELINE_ORDER[prev_index]
                    if _stage_is_done(state, prev_id):
                        ctx.prev_result = _rehydrate_prev_result(state, prev_id)

            stage_cls = get_stage_class(stage_id)
            stage = stage_cls()

            started = time.perf_counter()
            _log(logger, "")
            _log(logger, f"=== Stage {stage_index(stage_id)}: {stage.title} ===")
            _mark_stage_running(state, paths, stage_id)
            try:
                result = stage.run(ctx)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                _log(logger, f"✗ Stage failed after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
                raise
            result.validate(ctx)
            elapsed = time.perf_counter() - started
            _mark_stage_done(
                state,
                paths,
                stage_id,
                elapsed_sec=elapsed,
                result=result,
            )
            _log(
                logger,
                f"✓ Stage {stage_index(stage_id)} complete ({elapsed:.1f}s)",
            )
            ctx.prev_result = result

            if stop_after == stage_id:
                return _stop_pipeline(
                    state=state,
                    paths=paths,
                    run_dir=run_dir,
                    requested_stop_stage_id=stop_after,
                    logger=logger,
                )

    return _finalize_pipeline(state=state, paths=paths, run_dir=run_dir, logger=logger)


__all__ = ["run_pipeline"]
