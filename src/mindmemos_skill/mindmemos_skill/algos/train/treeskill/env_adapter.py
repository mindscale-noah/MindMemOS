"""Registered environment adaptation for TreeSkill analysis and routing."""

import json
from functools import partial
from importlib.resources import files
from pathlib import Path

from ....envs.base import Environment
from ....envs.typing import Task
from ....registry.base import Registrable
from .config import TreeSkillEnvAdapterConfig
from .typing import (
    TraceEvidence,
    TrainingRollout,
    TrajectoryAnalyzer,
    TreeSkillAnalysisContext,
    TreeSkillAnalysisRuntime,
    TreeSkillRoutingContext,
)

ANALYSIS_SYSTEM_PROMPT = (
    files(f"{__package__}.construction.prompts").joinpath("analysis_system_prompt.txt").read_text(encoding="utf-8")
)

ROUTING_SYSTEM_PROMPT = (
    files(f"{__package__}.construction.prompts").joinpath("routing_system_prompt.txt").read_text(encoding="utf-8")
)


def analysis_user_prompt(evidence: TraceEvidence, *, source: str) -> str:
    """Analysis user prompt.

    Args:
        evidence: Ordered trajectory-derived evidence available to this stage.
        source: Outcome label determined from the recorded score.
    """
    payload = {
        "instance_id": evidence.trajectory_id,
        "task_id": evidence.task_id,
        "record_source": source,
        "score": evidence.score,
        "evaluator_feedback": evidence.annotation_detail,
        "annotation_metadata": evidence.annotation_metadata,
        "trajectory": evidence.transcript,
    }
    return "Analyze this trajectory.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


class TreeSkillEnvAdapter(Registrable[TreeSkillEnvAdapterConfig]):
    """Provide generic behavior without owning run-local models or files."""

    config_type = TreeSkillEnvAdapterConfig
    """Default options inherited by user adapters."""

    def __init__(self, config: TreeSkillEnvAdapterConfig | None = None) -> None:
        """Bind immutable options without constructing model clients.

        Args:
            config: Literal system prompt overrides, or generic defaults.
        """
        super().__init__(config.model_copy(deep=True) if config is not None else self.config_type())

    def validate_environment(self, env: Environment) -> None:
        """Check compatibility before sampling starts.

        Args:
            env: Caller-owned environment; the generic adapter accepts any environment.
        """

    def render_trajectory(self, context: TraceEvidence) -> str:
        """Render normalized analysis evidence.

        Args:
            context: Completed training evidence without private scoring inputs.

        Returns:
            Recorded transcript text.
        """
        return context.transcript

    def render_task(self, task: Task) -> str:
        """Render public task fields for subtree selection.

        Args:
            task: Public task with no RewardData.

        Returns:
            Human-readable task text.
        """
        return json.dumps(task.model_dump(mode="json"), ensure_ascii=False, indent=2)

    def build_analysis_messages(self, context: TreeSkillAnalysisContext) -> list[dict[str, str]]:
        """Build the default analyzer's complete request.

        Args:
            context: Normalized evidence and the trainer-selected outcome label.

        Returns:
            Complete messages without executing a model.
        """
        evidence = context.evidence.model_copy(update={"transcript": self.render_trajectory(context.evidence)})
        return [
            {
                "role": "system",
                "content": self.config.analysis_system
                if self.config.analysis_system is not None
                else ANALYSIS_SYSTEM_PROMPT,
            },
            {"role": "user", "content": analysis_user_prompt(evidence, source=context.source)},
        ]

    def build_routing_messages(self, context: TreeSkillRoutingContext) -> list[dict[str, str]]:
        """Build environment-neutral task-time routing messages.

        Args:
            context: Public task and recursive tree payload.

        Returns:
            Complete messages containing no private training data.
        """
        return [
            {
                "role": "system",
                "content": self.config.routing_system
                if self.config.routing_system is not None
                else ROUTING_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "Route skill subtrees for this task.\n\nTask:\n"
                + self.render_task(context.task)
                + "\n\nSkill tree:\n"
                + json.dumps(context.tree, ensure_ascii=False, indent=2),
            },
        ]

    def prepare_analysis_workspace(self, trajectory: TrainingRollout, workspace: Path) -> None:
        """Prepare optional tool inputs when requested by the trainer.

        Args:
            trajectory: Training execution with any permitted analysis inputs.
            workspace: Fresh trainer-owned tool workspace.
        """
        # Generic analysis needs no additional tool inputs.

    def analysis_phases(self, evidence: list[TraceEvidence], threshold: float) -> list[list[TraceEvidence]]:
        """Return the environment's required scheduling order.

        Args:
            evidence: Ordered training evidence.
            threshold: Shared protocol cutoff used by outcome-aware adapters.

        Returns:
            Independent items grouped into sequential phases.
        """
        return [evidence]

    def create_analyzer(self, runtime: TreeSkillAnalysisRuntime) -> TrajectoryAnalyzer:
        """Construct the default analyzer with this adapter's message hook.

        Args:
            runtime: Direct agents and fixed stage settings.

        Returns:
            An analyzer ready to run, without making calls during construction.
        """
        from .construction.analysis import analyze_trajectory, parse_analysis

        config = runtime.config
        runtime.agent.output_parser = parse_analysis
        return partial(
            analyze_trajectory,
            agent=runtime.agent,
            success_score_threshold=config.success_score_threshold,
            env_adapter=self,
        )


@TreeSkillEnvAdapter.register("generic")
class TreeSkillGenericEnvAdapter(TreeSkillEnvAdapter):
    """Register inherited generic behavior in the shared subclass-only registry."""
