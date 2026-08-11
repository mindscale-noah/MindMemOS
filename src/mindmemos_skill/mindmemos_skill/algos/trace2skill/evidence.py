"""Normalize and validate trajectory evidence from any supported source."""

from __future__ import annotations

import json
from typing import Any

from ...persistence.enums import TrajectoryStatus
from ...typing import Skill, Trajectory
from .contracts import AnnotationMode, EvidenceSelection, TraceEvidence


def select_evidence(
    skill: Skill,
    trajectories: list[Trajectory],
    *,
    annotation_mode: AnnotationMode,
    transcript_max_chars: int,
    require_skill_match: bool,
) -> EvidenceSelection:
    """Deduplicate, validate and stably order merged trajectories."""

    unique: dict[str, Trajectory] = {}
    duplicates: list[str] = []
    for trajectory in trajectories:
        if trajectory.trajectory_id in unique:
            duplicates.append(trajectory.trajectory_id)
            continue
        unique[trajectory.trajectory_id] = trajectory

    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item.execution.finished_at or item.execution.started_at,
            item.trajectory_id,
        ),
    )
    evidence: list[TraceEvidence] = []
    for trajectory in ordered:
        if trajectory.execution.status is TrajectoryStatus.RUNNING:
            raise ValueError(f"trajectory {trajectory.trajectory_id!r} is still running")
        if require_skill_match and not trajectory_matches_skill(trajectory, skill):
            raise ValueError(
                f"trajectory {trajectory.trajectory_id!r} does not reference Skill family {skill.skill_id!r}"
            )
        if annotation_mode is AnnotationMode.REQUIRED and trajectory.reward.score is None:
            raise ValueError(f"trajectory {trajectory.trajectory_id!r} is missing the required reward score")

        use_annotation = annotation_mode is not AnnotationMode.IGNORE
        evidence.append(
            TraceEvidence(
                trajectory_id=trajectory.trajectory_id,
                task_id=trajectory.task.task_id,
                transcript=render_trajectory(trajectory.events, transcript_max_chars),
                score=trajectory.reward.score if use_annotation else None,
                annotation_detail=trajectory.reward.detail if use_annotation else None,
                annotation_metadata=trajectory.reward.metadata if use_annotation else {},
            )
        )
    return EvidenceSelection(evidence=evidence, duplicate_trajectory_ids=sorted(set(duplicates)))


def trajectory_matches_skill(trajectory: Trajectory, skill: Skill) -> bool:
    """Return whether a trajectory carries an identity reference to ``skill``."""

    for injected in trajectory.injected_skills:
        if _same_skill(
            skill,
            skill_id=injected.skill_id,
            cloud_skill_id=injected.cloud_skill_id,
            version_id=injected.version_id,
            content_hash=injected.content_hash,
        ):
            return True
    for binding in trajectory.skill_bindings:
        if _same_skill(
            skill,
            skill_id=binding.skill_id,
            cloud_skill_id=binding.cloud_skill_id,
            version_id=binding.version_id or binding.base_version_id,
            content_hash=binding.content_hash,
        ):
            return True
    return False


def render_trajectory(events: list[dict[str, Any]], max_chars: int) -> str:
    """Render ordered trajectory events into the compact LLM transcript format."""

    lines: list[str] = []
    for event in events:
        if "role" in event and "content" in event:
            rendered = f"[{event.get('role', '?')}] {event.get('content', '')}"
        elif "text" in event:
            rendered = f"[text] {event.get('text', '')}"
        elif "url" in event:
            rendered = f"[url] {event.get('url', '')}"
        elif "file_name" in event:
            rendered = f"[file] {event.get('file_name', '')}"
        else:
            rendered = json.dumps(event, ensure_ascii=False, sort_keys=True)
        lines.append(_truncate(rendered, max_chars))
    return "\n".join(lines)


def _same_skill(
    skill: Skill,
    *,
    skill_id: str | None,
    cloud_skill_id: str | None,
    version_id: str | None,
    content_hash: str | None,
) -> bool:
    if skill_id is not None:
        return skill_id == skill.skill_id
    if cloud_skill_id is not None:
        return skill.cloud_skill_id is not None and cloud_skill_id == skill.cloud_skill_id
    if version_id is not None:
        return version_id == skill.version_id
    return content_hash is not None and content_hash == skill.content_hash


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n…[{len(text) - max_chars} chars elided]…\n{tail}"


__all__ = ["render_trajectory", "select_evidence", "trajectory_matches_skill"]
