"""Stage pattern: families / adoption / behavior clusters. Zero LLM.

Builds the neutral comparison structure the evolve agent consumes
(docs/refactor-plan-minimal-contract.md §3):

- families: trajectories grouped by tasks-embedding agglomerative clustering
  (task similarity measured independently of execution structure);
- adoption: per (family x skill) used / not_used trajectory lists, from the
  extraction's agent-asserted ``skills`` key;
- behavior clusters: within each family, behaviors clustered on their
  keywords joined (in raw-text first-occurrence order) into one short text
  and embedded once.

All code here is format-agnostic: exact substring positions, embedding
similarity, set operations. Nothing interprets the raw trajectory.

Output: one ``workspace/<scope>/patterns.json`` per scope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.clustering import cluster_text_records
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import scope_workspace
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.evidence.extraction import anchor_variants
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.utils.files import write_json_file
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.utils.parallel import run_parallel_batch


@dataclass(frozen=True)
class PatternsConfig:
    """Pattern-stage knobs read from the settings file (all keys required)."""

    # Cosine distance threshold for grouping trajectories into task families.
    family_distance_threshold: float
    # Cosine distance threshold for behavior clustering within a family.
    keyword_cluster_threshold: float

    @classmethod
    def from_settings(cls) -> "PatternsConfig":
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_float

        return cls(
            family_distance_threshold=setting_float("pattern.family_distance_threshold"),
            keyword_cluster_threshold=setting_float("pattern.keyword_cluster_threshold"),
        )


@dataclass
class TrajectoryRow:
    """One extraction record prepared for family building."""

    trajectory_id: str
    tasks: str
    skills_used: list[str] = field(default_factory=list)
    # Behaviors in execution order (keyword first-occurrence in raw text).
    behaviors: list[dict[str, Any]] = field(default_factory=list)
    score_outcome: str | None = None
    trajectory_truncated: bool = False


def _fragment_position(raw_text: str, fragment: str) -> int:
    """Earliest offset of the fragment or its escaped anchor variant; -1 if absent."""
    positions = [pos for variant in anchor_variants(fragment) if (pos := raw_text.find(variant)) >= 0]
    return min(positions) if positions else -1


def _first_keyword_position(raw_text: str | None, keywords: list[str]) -> int:
    """Earliest raw-text offset of any keyword; unfound keywords sort last."""
    if not raw_text:
        return -1
    positions = [pos for keyword in keywords if (pos := _fragment_position(raw_text, keyword)) >= 0]
    return min(positions) if positions else len(raw_text)


def _ordered_behaviors(record: dict[str, Any], raw_text: str | None) -> list[dict[str, Any]]:
    """Sort behaviors by keyword first-occurrence; extraction order breaks ties.

    Keywords within each behavior are also sorted by first occurrence, so the
    joined clustering text is deterministic.
    """
    raw_behaviors = [b for b in (record.get("behaviors") or []) if isinstance(b, dict)]
    decorated = []
    for original_index, behavior in enumerate(raw_behaviors):
        keywords = [k for k in (behavior.get("keywords") or []) if isinstance(k, str) and k]
        if raw_text:
            keywords = sorted(
                keywords,
                key=lambda k: pos if (pos := _fragment_position(raw_text, k)) >= 0 else len(raw_text),
            )
        decorated.append(
            (
                _first_keyword_position(raw_text, keywords) if raw_text else original_index,
                original_index,
                {"summary": str(behavior.get("summary") or ""), "keywords": keywords},
            )
        )
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [{"behavior_index": index, **payload} for index, (_pos, _orig, payload) in enumerate(decorated)]


def load_scope_rows(workspace_root: str | Path, scope_id: str) -> list[TrajectoryRow]:
    """Read ``extractions/*.json`` into ordered rows; skip unreadable files."""
    scope = scope_workspace(workspace_root, scope_id)
    extractions_dir = scope / "extractions"
    trajectory_dir = scope / "trajectory"
    if not extractions_dir.is_dir():
        return []
    rows: list[TrajectoryRow] = []
    for path in sorted(extractions_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        stem = path.stem
        raw_text: str | None = None
        trajectory_path = trajectory_dir / f"{stem}.json"
        if trajectory_path.is_file():
            try:
                raw_text = trajectory_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw_text = None
        skills_used = sorted(
            {
                str(entry.get("name") or "").strip()
                for entry in (record.get("skills") or [])
                if isinstance(entry, dict) and str(entry.get("name") or "").strip()
            }
        )
        rows.append(
            TrajectoryRow(
                trajectory_id=str(record.get("trajectory_id") or stem),
                tasks=str(record.get("tasks") or ""),
                skills_used=skills_used,
                behaviors=_ordered_behaviors(record, raw_text),
                score_outcome=(str(record["score_outcome"]) if record.get("score_outcome") is not None else None),
                trajectory_truncated=bool(record.get("trajectory_truncated")),
            )
        )
    return rows


def _library_skill_names(workspace_root: str | Path, scope_id: str) -> list[str]:
    skills_dir = scope_workspace(workspace_root, scope_id) / "skills"
    if not skills_dir.is_dir():
        return []
    return sorted(path.name for path in skills_dir.iterdir() if path.is_dir() and not path.name.startswith("."))


def build_families(rows: list[TrajectoryRow], *, distance_threshold: float) -> list[list[int]]:
    """Group row indices into task families via tasks-embedding clustering."""
    texts = [row.tasks for row in rows]
    clusters = cluster_text_records(
        texts,
        texts,
        action_weight=1.0,
        summary_weight=0.0,
        distance_threshold=distance_threshold,
    )
    return [sorted(cluster.member_indices) for cluster in clusters]


def build_adoption(
    members: list[TrajectoryRow],
    *,
    library_skills: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Per skill: which family members consulted it, which did not."""
    adoption: dict[str, dict[str, list[str]]] = {}
    for skill in library_skills:
        used = sorted(row.trajectory_id for row in members if skill in row.skills_used)
        not_used = sorted(row.trajectory_id for row in members if skill not in row.skills_used)
        adoption[skill] = {"used": used, "not_used": not_used}
    return adoption


def _behavior_cluster_text(behavior: dict[str, Any]) -> str:
    return " ".join(behavior.get("keywords") or [])


def build_behavior_clusters(
    members: list[TrajectoryRow],
    *,
    distance_threshold: float,
) -> list[dict[str, Any]]:
    """Cluster all member behaviors on their joined-keywords embedding."""
    refs: list[tuple[str, int]] = []
    texts: list[str] = []
    for row in members:
        for behavior in row.behaviors:
            refs.append((row.trajectory_id, int(behavior["behavior_index"])))
            texts.append(_behavior_cluster_text(behavior))
    if not refs:
        return []
    clusters = cluster_text_records(
        texts,
        texts,
        action_weight=1.0,
        summary_weight=0.0,
        distance_threshold=distance_threshold,
    )
    out: list[dict[str, Any]] = []
    for cluster in clusters:
        out.append(
            {
                "cluster_id": cluster.cluster_id,
                "members": [
                    {"trajectory_id": refs[i][0], "behavior_index": refs[i][1]} for i in sorted(cluster.member_indices)
                ],
            }
        )
    return out


def build_scope_patterns(
    workspace_root: str | Path,
    scope_id: str,
    *,
    config: PatternsConfig,
) -> dict[str, Any]:
    """Build and persist ``workspace/<scope>/patterns.json``; return its payload."""
    rows = load_scope_rows(workspace_root, scope_id)
    library_skills = _library_skill_names(workspace_root, scope_id)

    families: list[dict[str, Any]] = []
    if rows:
        for family_id, member_indices in enumerate(
            build_families(rows, distance_threshold=config.family_distance_threshold)
        ):
            members = [rows[i] for i in member_indices]
            member_records = []
            for row in members:
                entry: dict[str, Any] = {
                    "trajectory_id": row.trajectory_id,
                    "tasks": row.tasks,
                    "skills_used": list(row.skills_used),
                    "behaviors": row.behaviors,
                }
                if row.score_outcome is not None:
                    entry["score_outcome"] = row.score_outcome
                if row.trajectory_truncated:
                    entry["trajectory_truncated"] = True
                member_records.append(entry)
            families.append(
                {
                    "family_id": family_id,
                    "members": member_records,
                    "adoption": build_adoption(members, library_skills=library_skills),
                    "behavior_clusters": build_behavior_clusters(
                        members,
                        distance_threshold=config.keyword_cluster_threshold,
                    ),
                }
            )

    payload = {
        "schema_version": 3,
        "scope_id": scope_id,
        "n_trajectories": len(rows),
        "n_families": len(families),
        "library_skills": library_skills,
        "families": families,
    }
    output_path = scope_workspace(workspace_root, scope_id) / "patterns.json"
    write_json_file(str(output_path), payload)
    return payload


@dataclass
class PatternsEvidenceSummary:
    """Aggregate counts recorded in state.json."""

    scopes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"scopes": dict(self.scopes)}


def build_patterns_evidence(
    workspace_root: str | Path,
    *,
    scope_ids: list[str],
    config: PatternsConfig | None = None,
    parallelism: int = 1,
) -> PatternsEvidenceSummary:
    """Run pattern building for every explicitly supplied scope."""
    workspace = Path(workspace_root)
    config = config or PatternsConfig.from_settings()
    summary = PatternsEvidenceSummary()
    if not scope_ids:
        return summary

    def invoke_one(_index: int, scope_id: str) -> dict[str, Any]:
        return build_scope_patterns(workspace, scope_id, config=config)

    payloads = run_parallel_batch(scope_ids, invoke_one, workers=max(1, int(parallelism)))
    for scope_id, payload in zip(scope_ids, payloads):
        summary.scopes[scope_id] = {
            "n_trajectories": payload["n_trajectories"],
            "n_families": payload["n_families"],
            "n_library_skills": len(payload["library_skills"]),
        }
    return summary


__all__ = [
    "PatternsConfig",
    "PatternsEvidenceSummary",
    "TrajectoryRow",
    "build_adoption",
    "build_behavior_clusters",
    "build_families",
    "build_patterns_evidence",
    "build_scope_patterns",
    "load_scope_rows",
]
