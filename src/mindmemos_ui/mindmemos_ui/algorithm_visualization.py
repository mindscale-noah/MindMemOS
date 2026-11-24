"""Read local Skill-Forest artifacts without loading runtime configuration."""

import json
import math
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import JsonValue, ValidationError

from .directory_picker import choose_directory
from .typing import (
    AlgorithmDirectoryRequest,
    ForestCheckpoint,
    ForestEvaluationRecord,
    ForestManifest,
    ForestNodeView,
    ForestProgress,
    ForestReport,
    ForestRunView,
    ForestSelectionResult,
)


def _safe_file(path: Path, root: Path) -> bool:
    """Keep evidence reads inside the selected directory and bound single files."""
    return path.resolve().is_relative_to(root) and path.is_file() and path.stat().st_size <= 64 * 1024 * 1024


def _read_json(path: Path, root: Path) -> JsonValue:
    """Read bounded JSON evidence, rejecting paths outside the selected root."""
    if not _safe_file(path, root):
        raise ValueError("Missing, external, or oversized artifact")
    return json.loads(path.read_text(encoding="utf-8"))


def _node_view(node: ForestCheckpoint, directory: Path, root: Path, total: int | None) -> ForestNodeView:
    """Combine checkpoint decisions with incremental validation evidence."""
    scores = dict(node.scores)
    errors = 0
    log = directory / "evaluation.jsonl"
    if node.status == "generated" and log.exists():
        if not _safe_file(log, root):
            errors += 1
        else:
            with log.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = ForestEvaluationRecord.model_validate_json(line)
                        if record.result.error:
                            raise ValueError("Failed evaluation")
                        scores[record.task_id] = float(record.result.reward.success)
                    except (ValueError, KeyError, TypeError):
                        errors += 1
    complete = total is not None and len(scores) == total
    final = node.status in {"root", "accepted", "rejected"}
    return ForestNodeView(
        checkpoint=node,
        completed=len(scores),
        total=total,
        score=sum(scores.values()) / len(scores) if scores else None,
        gain=node.gain if complete and final and node.parent_id else None,
        errors=errors,
    )


def _mark_winners(path: Path, root: Path, nodes: list[ForestNodeView], warnings: list[str]) -> bool:
    """Apply official completed selections without inferring winners from scores."""
    result_path = path / "result.json"
    if not result_path.exists():
        return False
    try:
        result = ForestSelectionResult.model_validate(_read_json(result_path, root))
    except (OSError, ValueError):
        warnings.append(f"Winner evidence unavailable: {result_path.relative_to(root)}")
        return False
    if result.status != "completed":
        return False
    by_id = {node.checkpoint.id: node for node in nodes}
    for winner in result.training.winners:
        node = by_id.get(winner.id)
        if node is not None and node.checkpoint.tree_id == winner.tree_id:
            node.winner = "tree"
        else:
            warnings.append(f"Winner checkpoint unavailable: {winner.id}")
    selected = result.training.algorithm_report.selected_node_id
    if selected is not None:
        node = by_id.get(selected)
        if node is not None and node.winner == "tree":
            node.winner = "final"
        else:
            warnings.append(f"Final winner does not match a saved tree winner: {selected}")
    return True


def _training_progress(
    path: Path, root: Path, nodes: list[ForestNodeView], manifest: ForestManifest | None, completed: bool
) -> ForestProgress:
    """Derive observed milestones without claiming that a worker is alive."""
    progress = ForestProgress(
        roots_completed=sum(node.checkpoint.status == "root" for node in nodes),
        roots_total=len(manifest.splits) if manifest else None,
        expansions_completed=sum(node.checkpoint.expanded for node in nodes),
        expansion_budget=manifest.trainer.expansion_budget if manifest else None,
    )
    if completed:
        progress.stage = "completed"
    elif any(_safe_file(file, root) for file in (path / "skills").glob("skill-tree-*.md")):
        progress.stage = "select"
    elif any(node.checkpoint.parent_id is not None or node.checkpoint.expanded for node in nodes):
        progress.stage = "search"
    elif any(folder.resolve().is_relative_to(root) for folder in (path / "trees").glob("tree-*/node-*/generation")):
        progress.stage = "search"
    elif manifest is not None or nodes:
        progress.stage = "root"
    return progress


def read_forest_report(directory: Path) -> ForestReport:
    """Discover training roots and summarize their persisted nodes.

    Args:
        directory: User-selected local output directory, including nested runs.

    Returns:
        Independent runs with checkpoint decisions and validation progress.

    Raises:
        ValueError: The chosen path is not a directory.
    """
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Choose an existing output directory")
    report = ForestReport(runs=[], warnings=[])
    visited = 0
    for current, children, files in os.walk(root, followlinks=False):
        visited += 1
        if visited > 2000:
            report.warnings.append("Directory scan limit reached. Select a more specific output directory.")
            break
        path = Path(current)
        children[:] = sorted(name for name in children if not name.startswith(".") and not (path / name).is_symlink())
        if "trees" not in children:
            if "config.json" in files:
                try:
                    manifest = ForestManifest.model_validate(_read_json(path / "config.json", root))
                    if manifest.trainer.expansion_budget is not None:
                        progress = _training_progress(path, root, [], manifest, False)
                        report.runs.append(ForestRunView(path=str(path), nodes=[], progress=progress))
                        children.clear()
                        continue
                except (OSError, ValueError):
                    pass
            if len(path.relative_to(root).parts) >= 5:
                children.clear()
            continue
        children.clear()
        checkpoints: list[tuple[Path, ForestCheckpoint]] = []
        for file in sorted((path / "trees").glob("tree-*/node-*/node.json")):
            if len(checkpoints) >= 5000:
                report.warnings.append("Node limit reached. Snapshot truncated to 5,000 nodes.")
                break
            try:
                node = ForestCheckpoint.model_validate(_read_json(file, root))
                if not all(math.isfinite(value) for value in [node.gain, *node.scores.values()]):
                    raise ValueError("Non-finite score")
                checkpoints.append((file.parent, node))
            except (OSError, ValueError, ValidationError):
                report.warnings.append(f"Skipped unreadable checkpoint: {file.relative_to(root)}")
        totals = {node.tree_id: len(node.scores) for _, node in checkpoints if node.status == "root" and node.scores}
        manifest = None
        try:
            manifest = ForestManifest.model_validate(_read_json(path / "config.json", root))
            for index, split in enumerate(manifest.splits):
                totals[index] = len(set(split.validation_ids))
        except (OSError, ValueError, KeyError, TypeError):
            report.warnings.append(f"Validation totals inferred from roots: {path.relative_to(root)}")
        nodes = []
        seen = set()
        for location, node in checkpoints:
            if node.id in seen:
                report.warnings.append(f"Skipped duplicate node ID: {node.id}")
                continue
            seen.add(node.id)
            try:
                nodes.append(_node_view(node, location, root, totals.get(node.tree_id)))
            except (OSError, UnicodeError):
                report.warnings.append(f"Validation evidence unavailable: {node.id}")
        completed = _mark_winners(path, root, nodes, report.warnings)
        progress = _training_progress(path, root, nodes, manifest, completed)
        report.runs.append(ForestRunView(path=str(path), nodes=nodes, progress=progress))
    if not report.runs:
        report.warnings.append("No Skill-Forest trees found. Choose a training directory or its output parent.")
    return report


def algorithm_visualization_router() -> APIRouter:
    """Create the session-protected local visualization endpoints.

    Returns:
        Router for directory selection and read-only artifact snapshots.
    """
    router = APIRouter(prefix="/api/algorithm-visualization")

    @router.post("/report", response_model=ForestReport)
    def report(request: AlgorithmDirectoryRequest) -> ForestReport:
        """Read the selected directory without executing algorithm code."""
        try:
            return read_forest_report(Path(request.path))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/choose-directory")
    def select_directory():
        """Open the host's existing native directory picker."""
        try:
            selected = choose_directory(Path.home())
        except RuntimeError:
            return {"path": None, "unavailable": True}
        return {"path": str(selected) if selected else None, "unavailable": False}

    return router
