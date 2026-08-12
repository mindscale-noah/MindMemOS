"""Tree-aware offline trajectory fusion implemented as a trace2skill algorithm."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Protocol
from uuid import uuid4

from ....errors import SkillConfigurationError
from ....llm import llm_run_context
from ....registry import ComponentRequirements, ComponentType, register
from ....typing import SkillCandidate, Trace2SkillInput, normalize_skill_text
from ..collection import ScheduledTrajectoryCollector, TrajectoryCollectionResult, TrajectoryCollector
from ..evidence import select_evidence
from .analysis import ChatModel, TreeSkillTrajectoryAnalyzer
from .config import TreeSkillConfig
from .fusion import TreeSkillNodeFuser
from .localization import TreeSkillEvidenceLocator
from .models import TreeSkillOutput, TreeSkillReport
from .tree import compile_tree_metadata, parse_skill_markdown


class AlgorithmContext(Protocol):
    models: Mapping[str, ChatModel]
    agents: Mapping[str, object]
    config_hash: str


@register(
    type=ComponentType.ALGO,
    name="treeskill",
    config_model=TreeSkillConfig,
    capabilities={"optimize"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"chat"})),
)
class TreeSkill:
    """Evolve one Skill by locating trajectory evidence into a Markdown tree."""

    algorithm_name: ClassVar[str] = "treeskill"
    algorithm_version: ClassVar[str] = "1"

    def __init__(
        self,
        *,
        config: TreeSkillConfig,
        context: AlgorithmContext,
        collector: TrajectoryCollector | None = None,
    ) -> None:
        self._config = config
        self._context = context
        try:
            self._chat_model = context.models["chat"]
        except KeyError as exc:
            raise SkillConfigurationError("treeskill requires the 'chat' model role") from exc
        self._collector = collector
        if self._collector is None and config.collection is not None:
            self._collector = ScheduledTrajectoryCollector(agents=context.agents, config=config.collection)

    async def optimize(self, request: Trace2SkillInput) -> TreeSkillOutput:
        run_id = request.run_id or f"treeskill-{uuid4().hex}"
        normalized_request = request if request.run_id is not None else request.model_copy(update={"run_id": run_id})
        with llm_run_context(run_id):
            return await self._optimize(normalized_request)

    async def _optimize(self, request: Trace2SkillInput) -> TreeSkillOutput:
        collection = await self._collect(request) if request.tasks else None
        trajectories = [*(collection.trajectories if collection is not None else []), *request.trajectories]
        selection = select_evidence(
            request.base_skill,
            trajectories,
            annotation_mode=self._config.annotation_mode,
            transcript_max_chars=self._config.transcript_max_chars,
            require_skill_match=self._config.require_skill_match,
        )
        input_ids = tuple(item.trajectory_id for item in selection.evidence)
        if len(selection.evidence) < self._config.min_trajectories:
            return self._unchanged(
                request=request,
                collection=collection,
                input_ids=input_ids,
                duplicate_ids=tuple(selection.duplicate_trajectory_ids),
                initial_node_count=0,
                reason="below_minimum_trajectory_count",
            )
        if len(selection.evidence) > self._config.max_trajectories:
            raise ValueError(
                f"treeskill received {len(selection.evidence)} trajectories; maximum is {self._config.max_trajectories}"
            )

        initial_tree = parse_skill_markdown(request.base_skill.content)
        if not initial_tree.nodes:
            return self._unchanged(
                request=request,
                collection=collection,
                input_ids=input_ids,
                duplicate_ids=tuple(selection.duplicate_trajectory_ids),
                initial_node_count=0,
                reason="skill_has_no_content_bearing_markdown_headings",
            )

        analyses, analysis_failures = await TreeSkillTrajectoryAnalyzer(
            chat_model=self._chat_model,
            task=self._config.analysis_task,
            concurrency=self._config.analysis_concurrency,
            success_score_threshold=self._config.success_score_threshold,
            temperature=self._config.analysis_temperature,
            max_tokens=self._config.analysis_max_tokens,
        ).analyze(selection.evidence)
        if not analyses:
            return self._unchanged(
                request=request,
                collection=collection,
                input_ids=input_ids,
                duplicate_ids=tuple(selection.duplicate_trajectory_ids),
                initial_node_count=len(initial_tree.nodes),
                analysis_failures=tuple(analysis_failures),
                reason="no_valid_trajectory_analysis_records",
            )

        located, localization_failures = await TreeSkillEvidenceLocator(
            chat_model=self._chat_model,
            task=self._config.localization_task,
            concurrency=self._config.localization_concurrency,
            temperature=self._config.localization_temperature,
            max_tokens=self._config.localization_max_tokens,
        ).locate(initial_tree, analyses)
        if not located:
            report = self._report(
                request=request,
                input_ids=input_ids,
                duplicate_ids=tuple(selection.duplicate_trajectory_ids),
                initial_node_count=len(initial_tree.nodes),
                final_node_count=len(initial_tree.nodes),
                analysis_failures=tuple(analysis_failures),
                analyses=tuple(analyses),
                localization_failures=tuple(localization_failures),
                reason="no_reusable_evidence_localized",
            )
            return TreeSkillOutput(
                candidate=None,
                trajectories=collection.trajectories if collection is not None else [],
                report=report,
            )

        final_tree, applied_edits, fusion_failures = await TreeSkillNodeFuser(
            chat_model=self._chat_model,
            task=self._config.fusion_task,
            temperature=self._config.fusion_temperature,
            max_tokens=self._config.fusion_max_tokens,
        ).fuse(initial_tree, located)
        changed = normalize_skill_text(final_tree.full_content) != normalize_skill_text(request.base_skill.content)
        report = self._report(
            request=request,
            input_ids=input_ids,
            duplicate_ids=tuple(selection.duplicate_trajectory_ids),
            initial_node_count=len(initial_tree.nodes),
            final_node_count=len(final_tree.nodes),
            analysis_failures=tuple(analysis_failures),
            analyses=tuple(analyses),
            localization_failures=tuple(localization_failures),
            located=tuple(located),
            applied_edits=tuple(applied_edits),
            fusion_failures=tuple(fusion_failures),
            changed=changed,
            reason=None if changed else "no_effective_change",
        )
        if not changed:
            return TreeSkillOutput(
                candidate=None,
                trajectories=collection.trajectories if collection is not None else [],
                report=report,
            )

        metadata = compile_tree_metadata(final_tree)
        metadata["evolution"] = {
            "algorithm": self.algorithm_name,
            "algorithm_version": self._config.algorithm_version,
            "prompt_version": self._config.prompt_version,
            "run_id": request.run_id,
            "config_hash": self._context.config_hash,
            "collection_run_id": collection.run_id if collection is not None else None,
            "trajectory_ids": list(report.input_trajectory_ids),
        }
        candidate = SkillCandidate(
            blob={"SKILL.md": final_tree.full_content},
            resources=dict(request.base_skill.resources),
            commit_message="optimize: apply TreeSkill trajectory fusion",
            metadata={"treeskill": metadata},
        )
        return TreeSkillOutput(
            candidate=candidate,
            trajectories=collection.trajectories if collection is not None else [],
            report=report,
        )

    async def _collect(self, request: Trace2SkillInput) -> TrajectoryCollectionResult:
        if self._collector is None:
            raise SkillConfigurationError("treeskill task collection requires the algorithm collection config")
        assert request.run_id is not None
        return await self._collector.collect(run_id=request.run_id, base_skill=request.base_skill, tasks=request.tasks)

    def _unchanged(
        self,
        *,
        request: Trace2SkillInput,
        collection: TrajectoryCollectionResult | None,
        input_ids: tuple[str, ...],
        duplicate_ids: tuple[str, ...],
        initial_node_count: int,
        analysis_failures: tuple[str, ...] = (),
        reason: str,
    ) -> TreeSkillOutput:
        report = self._report(
            request=request,
            input_ids=input_ids,
            duplicate_ids=duplicate_ids,
            initial_node_count=initial_node_count,
            final_node_count=initial_node_count,
            analysis_failures=analysis_failures,
            reason=reason,
        )
        return TreeSkillOutput(
            candidate=None,
            trajectories=collection.trajectories if collection is not None else [],
            report=report,
        )

    def _report(
        self,
        *,
        request: Trace2SkillInput,
        input_ids: tuple[str, ...],
        duplicate_ids: tuple[str, ...],
        initial_node_count: int,
        final_node_count: int,
        analysis_failures: tuple[str, ...] = (),
        analyses=(),
        localization_failures=(),
        located=(),
        applied_edits=(),
        fusion_failures=(),
        changed: bool = False,
        reason: str | None = None,
    ) -> TreeSkillReport:
        assert request.run_id is not None
        return TreeSkillReport(
            run_id=request.run_id,
            algorithm_version=self._config.algorithm_version,
            prompt_version=self._config.prompt_version,
            input_trajectory_ids=input_ids,
            duplicate_trajectory_ids=duplicate_ids,
            failed_analysis_trajectory_ids=analysis_failures,
            localization_failures=localization_failures,
            fusion_failures=fusion_failures,
            analysis_records=analyses,
            located_evidence=located,
            applied_edits=applied_edits,
            initial_node_count=initial_node_count,
            final_node_count=final_node_count,
            changed=changed,
            reason=reason,
        )


__all__ = ["TreeSkill"]
