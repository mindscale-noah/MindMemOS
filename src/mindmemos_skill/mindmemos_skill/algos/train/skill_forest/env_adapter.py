"""Registered, stateless environment adaptation for Skill-Forest."""

import json

from pydantic import JsonValue

from ....envs.base import Environment
from ....envs.typing import RolloutResult
from ....registry.base import Registrable
from .config import SkillForestEnvAdapterConfig
from .prompts import final_patch_user, load_prompt, patch_merge_user, reflection_user, skill_merge_user
from .trajectory_rendering import render_record, tool_conversation
from .typing import (
    SkillForestFinalPatchContext,
    SkillForestPatchMergeContext,
    SkillForestRankingContext,
    SkillForestReflectionContext,
    SkillForestSkillMergeContext,
    SkillForestTrajectoryContext,
)


class SkillForestEnvAdapter(Registrable[SkillForestEnvAdapterConfig]):
    """Provide usable generic behavior; override only environment-specific methods."""

    RANKING_PROMPT = """\
You are an expert in optimizing skills. You receive a skill document and a pool
of proposed edits. Your job is to RANK the edits by importance and select the top ones.

Ranking criteria (in order of priority):
1. **Systematic impact**: edits that address widespread, recurring failure patterns
   across many tasks should rank highest. A rule that fixes 50%% of failures beats
   one that fixes a single edge case.
2. **Complementarity**: edits that fill gaps in the current skill (not duplicate
   existing content) rank higher.
3. **Generality**: edits phrased as general principles rank higher than those
   tied to specific question types or entities.
4. **Actionability**: edits with clear, concrete guidance rank higher than vague advice.

You will be told how many edits to select (the budget).

Respond ONLY with a valid JSON object:
{
  "reasoning": "<brief justification for your ranking decisions>",
  "selected_indices": [<0-based indices of the top edits, in priority order>]
}
"""
    """Shared instructions for ranking merged candidate edits."""

    config_type = SkillForestEnvAdapterConfig
    """Default prompt overrides inherited by user adapters."""

    def __init__(self, config: SkillForestEnvAdapterConfig | None = None) -> None:
        """Load all prompts before any training starts.

        Args:
            config: Literal prompt overrides, or bundled defaults when omitted.
        """
        super().__init__(config.model_copy(deep=True) if config is not None else self.config_type())
        self.prompts = SkillForestEnvAdapterConfig(
            analyst_error=self.config.analyst_error
            if self.config.analyst_error is not None
            else self._load_prompt("analyst_error"),
            analyst_success=self.config.analyst_success
            if self.config.analyst_success is not None
            else self._load_prompt("analyst_success"),
            merge_failure=self.config.merge_failure
            if self.config.merge_failure is not None
            else self._load_prompt("merge_failure"),
            merge_success=self.config.merge_success
            if self.config.merge_success is not None
            else self._load_prompt("merge_success"),
            merge_final=self.config.merge_final
            if self.config.merge_final is not None
            else self._load_prompt("merge_final"),
            ranking=self.config.ranking if self.config.ranking is not None else self._load_prompt("ranking"),
            hierarchical_merge_skills=self.config.hierarchical_merge_skills
            if self.config.hierarchical_merge_skills is not None
            else self._load_prompt("hierarchical_merge_skills"),
        )

    def _load_prompt(self, name: str) -> str:
        """Load the default resource for one stage."""
        if name == "ranking":
            return self.RANKING_PROMPT
        return load_prompt(name)

    def validate_environment(self, env: Environment) -> None:
        """Validate environment compatibility before creating run artifacts.

        Args:
            env: Caller-owned rollout environment; generic adaptation accepts any environment.
        """

    def render_trajectory(self, context: SkillForestTrajectoryContext) -> str:
        """Render public evidence without evaluator-only details.

        Args:
            context: One completed training trajectory without RewardData.

        Returns:
            A Markdown section with public context and paired tool exchanges.
        """
        task_fields = context.task.model_dump(mode="json")
        events = _public_conversation(context.result)
        system = user = ""
        if events and events[0].get("role") == "system":
            system = events.pop(0).get("content") or ""
        if events and events[0].get("role") == "user":
            user = events.pop(0).get("content") or ""
        conversation = tool_conversation(events, assistant_role="agent", strict=False)
        return render_record(
            context.index,
            context.task.id,
            task_fields.get("task_description", context.task.instruction),
            task_fields.get("task_type", task_fields.get("instruction_type", "")),
            "",
            len(conversation),
            conversation,
            system=system,
            user=user,
        )

    def build_reflection_messages(self, context: SkillForestReflectionContext) -> list[dict[str, str]]:
        """Build a complete generic reflection request.

        Args:
            context: Current Skill, ordered evidence and edit budget.

        Returns:
            System and user messages, without model calls.
        """
        rendered = [self.render_trajectory(item) for item in context.trajectories]
        trajectories = "\n\n---\n\n".join(rendered)
        return [
            {
                "role": "system",
                "content": self.prompts.analyst_success if context.success else self.prompts.analyst_error,
            },
            {
                "role": "user",
                "content": reflection_user(
                    context.skill,
                    trajectories,
                    count=len(context.trajectories),
                    success=context.success,
                    edit_budget=context.edit_budget,
                ),
            },
        ]

    def build_patch_merge_messages(self, context: SkillForestPatchMergeContext) -> list[dict[str, str]]:
        """Build a same-outcome patch reduction request.

        Args:
            context: Ordered patches and reduction level.

        Returns:
            Complete optimizer messages.
        """
        return [
            {
                "role": "system",
                "content": self.prompts.merge_success if context.success else self.prompts.merge_failure,
            },
            {
                "role": "user",
                "content": patch_merge_user(
                    context.skill,
                    [patch.model_dump(exclude_unset=True) for patch in context.patches],
                    level=context.level,
                ),
            },
        ]

    def build_final_patch_messages(self, context: SkillForestFinalPatchContext) -> list[dict[str, str]]:
        """Build the final failure/success reconciliation request.

        Args:
            context: Reduced outcome patches and parent Skill.

        Returns:
            Complete optimizer messages.
        """
        return [
            {"role": "system", "content": self.prompts.merge_final},
            {
                "role": "user",
                "content": final_patch_user(
                    context.skill,
                    [
                        ("failure", context.failure.model_dump(exclude_unset=True)),
                        ("success", context.success.model_dump(exclude_unset=True)),
                    ],
                ),
            },
        ]

    def build_ranking_messages(self, context: SkillForestRankingContext) -> list[dict[str, str]]:
        """Build a request to select the most useful candidate edits.

        Args:
            context: Parent Skill, merged edits and the current depth's edit budget.

        Returns:
            Complete optimizer messages with zero-based edit indices.
        """
        descriptions = []
        for index, edit in enumerate(context.patch.edits):
            parts = [f"op={edit.op}"]
            if edit.target:
                parts.append(f"target={edit.target!r}")
            if edit.content:
                parts.append(f"content={edit.content!r}")
            if edit.support_count is not None:
                parts.append(f"support={edit.support_count}")
            descriptions.append(f"[{index}] " + "  ".join(parts))
        user = (
            f"## Current Skill\n{context.skill}\n\n"
            f"## Edits Pool ({len(context.patch.edits)} edits, budget={context.edit_budget})\n"
            + "\n".join(descriptions)
            + f"\n\nSelect the {context.edit_budget} most important edits. Return their 0-based indices in priority order."
        )
        return [
            {"role": "system", "content": self.prompts.ranking},
            {"role": "user", "content": user},
        ]

    def build_skill_merge_messages(self, context: SkillForestSkillMergeContext) -> list[dict[str, str]]:
        """Build a full-Skill synthesis request.

        Args:
            context: Two candidates and their comparable scores.

        Returns:
            Complete optimizer messages.
        """
        return [
            {"role": "system", "content": self.prompts.hierarchical_merge_skills},
            {
                "role": "user",
                "content": skill_merge_user(context.left, context.right, context.left_score, context.right_score),
            },
        ]


@SkillForestEnvAdapter.register("generic")
class SkillForestGenericEnvAdapter(SkillForestEnvAdapter):
    """Register the inherited generic behavior without changing the shared registry contract."""


def _public_conversation(result: RolloutResult) -> list[dict[str, JsonValue]]:
    """Adapt current event shapes without losing tool arguments or leaking score details."""
    conversation = []
    for event in result.trajectory:
        if event.get("role") == "system" and event.get("content") == result.reward.detail:
            continue
        if "content" in event or "action" in event or "tool_calls" in event or event.get("type") == "tool_call":
            conversation.append(event)
        else:
            conversation.append({"role": "agent", "content": json.dumps(event, ensure_ascii=False)})
    return conversation or [{"role": "assistant", "content": result.output}]
