"""Registered environment adaptation for TreeSkill analysis and routing."""

import json

from ....envs.base import Environment
from ....envs.typing import Task
from ....registry.base import Registrable
from .config import TreeSkillEnvAdapterConfig
from .prompts import ANALYSIS_SYSTEM_PROMPT, ROUTING_SYSTEM_PROMPT, analysis_user_prompt
from .typing import (
    TraceEvidence,
    TrajectoryAnalyzer,
    TreeSkillAnalysisContext,
    TreeSkillAnalysisRuntime,
    TreeSkillRoutingContext,
)


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

    def create_analyzer(self, runtime: TreeSkillAnalysisRuntime) -> TrajectoryAnalyzer:
        """Construct the default analyzer with this adapter's message hook.

        Args:
            runtime: Borrowed recorded models, stage settings and output directory.

        Returns:
            An analyzer ready to run, without making calls during construction.
        """
        from .analysis import TreeSkillTrajectoryAnalyzer

        config = runtime.config
        return TreeSkillTrajectoryAnalyzer(
            chat_model=runtime.chat_model,
            task=config.analysis_task,
            concurrency=config.analysis_concurrency,
            success_score_threshold=config.success_score_threshold,
            temperature=config.analysis_temperature,
            max_tokens=config.analysis_max_tokens,
            env_adapter=self,
        )


@TreeSkillEnvAdapter.register("generic")
class TreeSkillGenericEnvAdapter(TreeSkillEnvAdapter):
    """Register inherited generic behavior in the shared subclass-only registry."""
