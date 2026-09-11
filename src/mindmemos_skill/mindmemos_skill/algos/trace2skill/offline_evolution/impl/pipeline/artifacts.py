"""Run-directory layout and path helpers for scope-first artifacts.

Stages never construct ``workspace/<scope>/<artifact>`` paths inline;
they go through the helpers here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

SCORE_ARTIFACT = "scores"
SCORE_FILE_SUFFIX = ".txt"

SCOPE_ARTIFACTS = {
    "trajectory",
    SCORE_ARTIFACT,
    "skills",
    "logs",
    "extractions",
}


def scope_workspace(workspace_root: str | Path, scope_id: str) -> Path:
    """Return one evolution/evidence scope workspace root."""
    if not scope_id or scope_id in {".", ".."} or "/" in scope_id or "\\" in scope_id:
        raise ValueError(f"unsafe scope_id: {scope_id!r}")
    return Path(workspace_root) / scope_id


def scope_artifact_dir(
    workspace_root: str | Path,
    scope_id: str,
    artifact: str,
) -> Path:
    """Return ``workspace/<scope>/<artifact>`` for a canonical artifact."""
    if artifact not in SCOPE_ARTIFACTS:
        raise ValueError(f"unknown scope artifact: {artifact!r}")
    return scope_workspace(workspace_root, scope_id) / artifact


def score_file(scores_dir: str | Path, stem: str) -> Path:
    """Return the score file for one trajectory stem inside ``scores_dir``.

    ``scores_dir`` is a directory (source or staged), not a workspace root.
    """
    return Path(scores_dir) / f"{stem}{SCORE_FILE_SUFFIX}"


def read_score_text(scores_dir: str | Path, stem: str) -> str | None:
    """Return the score's stripped text, or None when absent or empty.

    Verbatim: no float coercion, no polarity mapping -- both belong downstream.
    """
    path = score_file(scores_dir, stem)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def scope_baseline_skills(baseline_root: str | Path, scope_id: str) -> Path:
    """Return ``<run>/intermediate/skills_baseline/<scope>``.

    Lives outside the workspace so nothing the Agent can see can name it;
    ``skills/`` is the editable tree, this is the immutable diff base.
    """
    if not scope_id or scope_id in {".", ".."} or "/" in scope_id or "\\" in scope_id:
        raise ValueError(f"unsafe scope_id: {scope_id!r}")
    return Path(baseline_root) / scope_id


def scope_id_from_path(
    path: str | Path,
    workspace_root: str | Path,
    artifact: str,
) -> str:
    """Validate a scope-first artifact path and return its scope id."""
    if artifact not in SCOPE_ARTIFACTS:
        raise ValueError(f"unknown scope artifact: {artifact!r}")
    resolved = Path(path).resolve()
    root = Path(workspace_root).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is not under workspace root: {path}") from exc
    if len(relative.parts) < 3 or relative.parts[1] != artifact:
        raise ValueError(f"expected <workspace>/<scope>/{artifact}/..., got: {path}")
    return relative.parts[0]


def trajectory_output_paths(
    inputs: Iterable[str],
    *,
    workspace_root: str | Path,
) -> list[str]:
    """Map workspace trajectories to their staged artifact paths.

    Prepare already copied trajectories into ``<workspace>/<scope>/trajectory/``,
    so each path maps onto itself; suffixes are preserved, not normalized.
    """
    root = Path(workspace_root).resolve()
    outputs: list[str] = []
    for raw in inputs:
        source = Path(raw).resolve()
        try:
            relative = source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"trajectory is not under workspace root: {raw}") from exc
        if len(relative.parts) != 3 or relative.parts[1] != "trajectory":
            raise ValueError(f"trajectory must be <workspace>/<scope>/trajectory/<file>: {raw}")
        outputs.append(str(source))
    if len(outputs) != len(set(outputs)):
        raise ValueError("scope-first trajectory output paths must be unique")
    return outputs


def extraction_output_paths(
    inputs: Iterable[str],
    *,
    workspace_root: str | Path,
) -> list[str]:
    """Map trajectory files to ``<workspace>/<scope>/extractions/<stem>.json``."""
    outputs: list[str] = []
    for raw in inputs:
        scope_id = scope_id_from_path(raw, workspace_root, "trajectory")
        source = Path(raw)
        outputs.append(str(scope_artifact_dir(workspace_root, scope_id, "extractions") / f"{source.stem}.json"))
    if len(outputs) != len(set(outputs)):
        raise ValueError("scope-first extraction output paths must be unique")
    return outputs


def display_artifact(workspace_root: str | Path, artifact: str) -> str:
    if artifact not in SCOPE_ARTIFACTS:
        raise ValueError(f"unknown scope artifact: {artifact!r}")
    return str(Path(workspace_root) / "*" / artifact)


__all__ = [
    "SCOPE_ARTIFACTS",
    "SCORE_ARTIFACT",
    "SCORE_FILE_SUFFIX",
    "display_artifact",
    "extraction_output_paths",
    "read_score_text",
    "scope_artifact_dir",
    "scope_baseline_skills",
    "scope_id_from_path",
    "scope_workspace",
    "score_file",
    "trajectory_output_paths",
]
