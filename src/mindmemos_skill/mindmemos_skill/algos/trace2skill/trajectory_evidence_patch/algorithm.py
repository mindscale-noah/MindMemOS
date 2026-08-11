"""Trajectory evidence aggregation followed by a minimal Skill patch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Protocol
from uuid import uuid4

from ....errors import SkillConfigurationError
from ....llm import llm_run_context
from ....registry import ComponentRequirements, ComponentType, register
from ....typing import (
    SkillCandidate,
    Trace2SkillInput,
    normalize_skill_text,
)
from ..collection import ScheduledTrajectoryCollector, TrajectoryCollectionResult, TrajectoryCollector
from ..evidence import select_evidence
from .config import TrajectoryEvidencePatchConfig
from .models import TrajectoryEvidencePatchOutput, TrajectoryEvidencePatchReport, TrajectorySummary
from .patcher import TrajectoryEvidencePatcher
from .summarizer import ChatModel, TrajectoryEvidenceSummarizer


class AlgorithmContext(Protocol):
    """Neutral subset of the application build context required here."""

    models: Mapping[str, ChatModel]
    agents: Mapping[str, object]
    config_hash: str


@register(
    type=ComponentType.ALGO,
    name="trajectory_evidence_patch",
    config_model=TrajectoryEvidencePatchConfig,
    capabilities={"optimize"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"chat"})),
)
class TrajectoryEvidencePatch:
    """Optimize one Skill from a bounded offline, collected, or hybrid batch."""

    algorithm_name: ClassVar[str] = "trajectory_evidence_patch"
    algorithm_version: ClassVar[str] = "1"

    def __init__(
        self,
        *,
        config: TrajectoryEvidencePatchConfig,
        context: AlgorithmContext,
        collector: TrajectoryCollector | None = None,
    ) -> None:
        self._config = config
        self._context = context
        try:
            self._chat_model = context.models["chat"]
        except KeyError as exc:
            raise SkillConfigurationError("trajectory_evidence_patch requires the 'chat' model role") from exc
        self._collector = collector
        if self._collector is None and config.collection is not None:
            self._collector = ScheduledTrajectoryCollector(agents=context.agents, config=config.collection)

    async def optimize(self, request: Trace2SkillInput) -> TrajectoryEvidencePatchOutput:
        """Run one optimization with all nested LLM calls bound to its run ID."""

        run_id = request.run_id or f"trace2skill-{uuid4().hex}"
        normalized_request = request if request.run_id is not None else request.model_copy(update={"run_id": run_id})
        with llm_run_context(run_id):
            return await self._optimize(normalized_request)

    async def _optimize(self, request: Trace2SkillInput) -> TrajectoryEvidencePatchOutput:
        """Acquire evidence when needed, then summarize and patch without persistence."""

        collection = await self._collect(request) if request.tasks else None
        trajectories = [
            *(collection.trajectories if collection is not None else []),
            *request.trajectories,
        ]

        selection = select_evidence(
            request.base_skill,
            trajectories,
            annotation_mode=self._config.annotation_mode,
            transcript_max_chars=self._config.transcript_max_chars,
            require_skill_match=self._config.require_skill_match,
        )
        input_ids = [item.trajectory_id for item in selection.evidence]
        if len(selection.evidence) < self._config.min_trajectories:
            report = self._report(
                request=request,
                collection=collection,
                input_ids=input_ids,
                duplicate_ids=selection.duplicate_trajectory_ids,
                reason="below_minimum_trajectory_count",
            )
            return self._unchanged(report, collection)
        if len(selection.evidence) > self._config.max_trajectories:
            raise ValueError(
                "trajectory_evidence_patch accepts one bounded batch per optimize call: "
                f"received {len(selection.evidence)}, maximum {self._config.max_trajectories}"
            )

        summarizer = TrajectoryEvidenceSummarizer(
            chat_model=self._chat_model,
            task=self._config.summary_task,
            concurrency=self._config.summary_concurrency,
        )
        summaries, failures = await summarizer.summarize(request.base_skill.name, selection.evidence)
        if len(summaries) < self._config.min_trajectories:
            report = self._report(
                request=request,
                collection=collection,
                input_ids=input_ids,
                duplicate_ids=selection.duplicate_trajectory_ids,
                failures=failures,
                summaries=summaries,
                reason="insufficient_summaries_after_failures",
            )
            return self._unchanged(report, collection)

        patch_plan, candidate_md = await TrajectoryEvidencePatcher(
            chat_model=self._chat_model,
            config=self._config,
        ).patch(
            skill_name=request.base_skill.name,
            skill_md=request.base_skill.content,
            summaries=summaries,
        )
        if not candidate_md.strip():
            raise ValueError("trajectory_evidence_patch produced an empty SKILL.md")

        normalized_candidate = normalize_skill_text(candidate_md)
        changed = normalized_candidate != normalize_skill_text(request.base_skill.content)
        report = self._report(
            request=request,
            collection=collection,
            input_ids=input_ids,
            duplicate_ids=selection.duplicate_trajectory_ids,
            failures=failures,
            summaries=summaries,
            patch_plan=patch_plan,
            changed=changed,
            reason=None if changed else "no_effective_change",
        )
        if not changed:
            return self._unchanged(report, collection)

        blob = {**request.base_skill.blob, "SKILL.md": normalized_candidate}
        candidate = SkillCandidate(
            blob=blob,
            resources=request.base_skill.resources,
            commit_message="optimize: apply trajectory evidence",
            metadata={
                "trace2skill": {
                    "algorithm": self.algorithm_name,
                    "algorithm_version": self._config.algorithm_version,
                    "prompt_version": self._config.prompt_version,
                    "run_id": request.run_id,
                    "config_hash": self._context.config_hash,
                    "collection_run_id": collection.run_id if collection is not None else None,
                    "trajectory_ids": report.used_trajectory_ids,
                }
            },
        )
        return TrajectoryEvidencePatchOutput(
            candidate=candidate,
            trajectories=collection.trajectories if collection is not None else [],
            report=report,
        )

    async def _collect(self, request: Trace2SkillInput) -> TrajectoryCollectionResult:
        if self._collector is None:
            raise SkillConfigurationError(
                "trajectory_evidence_patch task collection requires the algorithm collection config"
            )
        assert request.run_id is not None
        return await self._collector.collect(
            run_id=request.run_id,
            base_skill=request.base_skill,
            tasks=request.tasks,
        )

    def _report(
        self,
        *,
        request: Trace2SkillInput,
        collection: TrajectoryCollectionResult | None,
        input_ids: list[str],
        duplicate_ids: list[str],
        failures: list[str] | None = None,
        summaries: list[TrajectorySummary] | None = None,
        patch_plan: str | None = None,
        changed: bool = False,
        reason: str | None = None,
    ) -> TrajectoryEvidencePatchReport:
        assert request.run_id is not None
        resolved_summaries = summaries or []
        return TrajectoryEvidencePatchReport(
            run_id=request.run_id,
            algorithm_version=self._config.algorithm_version,
            prompt_version=self._config.prompt_version,
            annotation_mode=self._config.annotation_mode,
            collection_run_id=collection.run_id if collection is not None else None,
            input_task_ids=[task.task_id for task in request.tasks],
            requested_collection_rollout_ids=(collection.requested_rollout_ids if collection is not None else []),
            failed_collection_rollout_ids=collection.failed_rollout_ids if collection is not None else [],
            input_trajectory_ids=input_ids,
            used_trajectory_ids=[item.trajectory_id for item in resolved_summaries],
            duplicate_trajectory_ids=duplicate_ids,
            failed_summary_trajectory_ids=failures or [],
            summaries=resolved_summaries,
            patch_plan=patch_plan,
            changed=changed,
            reason=reason,
        )

    @staticmethod
    def _unchanged(
        report: TrajectoryEvidencePatchReport,
        collection: TrajectoryCollectionResult | None,
    ) -> TrajectoryEvidencePatchOutput:
        return TrajectoryEvidencePatchOutput(
            candidate=None,
            trajectories=collection.trajectories if collection is not None else [],
            report=report,
        )


__all__ = ["TrajectoryEvidencePatch"]
