"""Candidate construction and strict paired-ablation scoring."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from ....typing import Skill, Task
from .config import AblationConfig
from .contracts import (
    CandidateCaseResult,
    CandidateEvaluationRecord,
    RolloutOutcome,
    RolloutPhase,
    SkillTextEdit,
)
from .fileedit import apply_best_effort, validate_edits
from .replay_buffer import TouchedCluster


@dataclass(slots=True)
class AblationCandidate:
    candidate_id: str
    cluster_id: str
    edit: SkillTextEdit
    source_task_id: str
    sampled_tasks: list[Task]
    skill: Skill


class AblationEvaluator:
    def __init__(self, config: AblationConfig, *, success_reward: float) -> None:
        self.config = config
        self.success_reward = success_reward
        self._rng = random.Random(config.seed)

    def random_state(self) -> list[Any]:
        version, internal, gaussian = self._rng.getstate()
        return [version, list(internal), gaussian]

    def load_random_state(self, state: list[Any] | None) -> None:
        if state is None:
            return
        if len(state) != 3 or not isinstance(state[1], list):
            raise ValueError("invalid ablation RNG state")
        self._rng.setstate((int(state[0]), tuple(int(value) for value in state[1]), state[2]))

    def build_candidates(
        self,
        *,
        run_id: str,
        batch_index: int,
        current_skill: Skill,
        touched: list[TouchedCluster],
        task_registry: dict[str, Task],
        case_total_scores: dict[str, float],
        skill_factory,
        sample_counter: int,
        min_cluster_edits: int,
    ) -> tuple[list[AblationCandidate], int]:
        candidates: list[AblationCandidate] = []
        for touched_index, item in enumerate(touched):
            cluster = item.cluster
            if len(cluster.records) < min_cluster_edits or cluster.committed_replace is None:
                continue
            find, source = self._pick_find(item, cluster.committed_replace, current_skill.content, case_total_scores)
            edit = SkillTextEdit(find=find, replace=cluster.committed_replace)
            content, applied = apply_best_effort([edit], current_skill.content)
            if not applied or content == current_skill.content:
                continue
            candidate_id = hashlib.sha256(
                f"{run_id}:{batch_index}:{cluster.cluster_id}:{touched_index}".encode("utf-8")
            ).hexdigest()[:20]
            known = sorted(
                {record.source_task_id for record in cluster.records if record.source_task_id in task_registry}
            )
            count = min(self.config.max_source_cases_per_candidate, len(known))
            sampled_ids = self._rng.sample(known, count) if count else []
            candidates.append(
                AblationCandidate(
                    candidate_id=candidate_id,
                    cluster_id=cluster.cluster_id,
                    edit=edit,
                    source_task_id=source,
                    sampled_tasks=[task_registry[task_id] for task_id in sampled_ids],
                    skill=skill_factory(current_skill, content, f"candidate:{candidate_id}"),
                )
            )
            sample_counter += 1
        return candidates, sample_counter

    def score(
        self,
        candidates: list[AblationCandidate],
        outcomes: list[RolloutOutcome],
    ) -> list[CandidateEvaluationRecord]:
        before = self._rates(outcomes, phase=RolloutPhase.ABLATION_BEFORE, candidate_id=None)
        records: list[CandidateEvaluationRecord] = []
        for candidate in candidates:
            after = self._rates(
                outcomes,
                phase=RolloutPhase.ABLATION_AFTER,
                candidate_id=candidate.candidate_id,
            )
            per_case: list[CandidateCaseResult] = []
            net_effect = 0.0
            for task in candidate.sampled_tasks:
                before_rate = before.get(task.task_id, 0.0)
                after_rate = after.get(task.task_id, 0.0)
                delta = after_rate - before_rate
                net_effect += delta
                per_case.append(
                    CandidateCaseResult(
                        task_id=task.task_id,
                        before=before_rate,
                        after=after_rate,
                        delta=delta,
                    )
                )
            records.append(
                CandidateEvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    cluster_id=candidate.cluster_id,
                    edit=candidate.edit,
                    source_task_id=candidate.source_task_id,
                    sampled_task_ids=[task.task_id for task in candidate.sampled_tasks],
                    net_effect=net_effect,
                    per_case=per_case,
                )
            )
        records.sort(key=lambda item: item.net_effect, reverse=True)
        eligible: list[CandidateEvaluationRecord] = []
        for record in records:
            if not self.config.positive_only:
                eligible.append(record)
                continue
            threshold = self.config.improvement_threshold
            accepted = record.net_effect > 0.0 if threshold is None else record.net_effect >= threshold
            if accepted:
                eligible.append(record)
            else:
                record.rejection_reason = "net_effect_gate"
        for record in eligible[: self.config.commit_topk]:
            record.chosen = True
        for record in eligible[self.config.commit_topk :]:
            record.rejection_reason = "commit_topk"
        return records

    def _rates(
        self,
        outcomes: list[RolloutOutcome],
        *,
        phase: RolloutPhase,
        candidate_id: str | None,
    ) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for outcome in outcomes:
            if outcome.spec.phase is not phase or outcome.spec.candidate_id != candidate_id:
                continue
            score = outcome.trajectory.reward.score if outcome.trajectory is not None else None
            success = float(score is not None and score >= self.success_reward)
            grouped.setdefault(outcome.spec.task.task_id, []).append(success)
        return {task_id: sum(values) / len(values) for task_id, values in grouped.items() if values}

    @staticmethod
    def _pick_find(
        touched: TouchedCluster,
        replace: str,
        skill_text: str,
        case_scores: dict[str, float],
    ) -> tuple[str, str]:
        candidates = sorted(
            touched.find_sources,
            key=lambda item: case_scores.get(item[1], 0.0),
        )
        for find, source in candidates:
            edit = SkillTextEdit(find=find, replace=replace)
            valid, _ = validate_edits([edit], skill_text)
            if valid:
                return find, source
        return candidates[0] if candidates else ("", "")


__all__ = ["AblationCandidate", "AblationEvaluator"]
