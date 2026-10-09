"""Passage-grounded SearchQA rollout using the LiveMath lifecycle."""

from pathlib import Path

from ....agents import OpenAIChatAgent
from ....skill import Skill
from ...base import Environment
from ...data import RewardData
from ...typing import Task
from ..common import skill_text, total_usage
from .config import SearchQAConfig
from .prompts import SYSTEM_PROMPT
from .reward import evaluate
from .typing import SearchQAResult, SearchQARewardData, SearchQATask


def _truncate(context: str, limit: int) -> str:
    """Preserve DOC-boundary truncation, including its fallback suffix."""
    if len(context) <= limit:
        return context
    result = ""
    for document in context.split("[DOC]"):
        candidate = result + "[DOC]" + document if result else document
        if len(candidate) > limit:
            break
        result = candidate
    return result or context[:limit] + "\n...[truncated]"


@Environment.register("searchqa")
class SearchQAEnvironment(Environment[SearchQAConfig]):
    """Answer from public passages and privately score the final response."""

    task_type = SearchQATask
    """Public SearchQA task schema."""

    reward_data_type = SearchQARewardData
    """Private SearchQA scoring schema."""

    config_type = SearchQAConfig
    """Model and evaluation configuration."""

    result_type = SearchQAResult
    """Typed resume schema."""

    def __init__(self, config: SearchQAConfig, *, agent: OpenAIChatAgent | None = None) -> None:
        """Initialize a text agent with audited settings.

        Args:
            config: Rollout settings.
            agent: Optional matching backend for dependency injection.
        """
        super().__init__(config)
        if agent is not None and agent.config != config.agent:
            raise ValueError("Injected agent config must match the environment's agent config")
        self.agent = agent if agent is not None else OpenAIChatAgent(config.agent)

    def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> SearchQAResult:
        """Run refinement without giving the answering model evaluation feedback."""
        if not isinstance(task, SearchQATask) or not isinstance(reward_data, SearchQARewardData):
            raise TypeError("SearchQA requires SearchQATask and SearchQARewardData")
        text = skill_text(skills)
        system = SYSTEM_PROMPT.format(skill_section=f"## Skill\n{text.strip()}\n\n" if text.strip() else "")
        user = (
            f"## Context\n{_truncate(task.context, self.config.max_context_chars)}\n\n## Question\n{task.instruction}"
        )
        (workspace / "system.md").write_text(system, encoding="utf-8")
        (workspace / "task.md").write_text(user, encoding="utf-8")
        responses, trajectory = [], []
        output = ""
        for turn in range(self.config.max_turns):
            prompt = (
                user
                if turn == 0
                else (
                    f"Your previous answer was:\n{output}\n\n"
                    "Review it against the context and question. "
                    "If correct, repeat it. If wrong, provide a corrected answer.\n"
                    "Use <answer>...</answer> tags for your final answer."
                )
            )
            response = self.agent.run(system=system, user=prompt)
            if not isinstance(response.output, str):
                raise TypeError("SearchQA requires a text agent output")
            output = response.output
            responses.append(response)
            trajectory.extend(response.trajectory)
            if turn > 0 and "<answer>" in output.lower():
                break
        reward = evaluate(output, task, reward_data)
        trajectory.append({"role": "system", "content": reward.detail})
        (workspace / "response.txt").write_text(output, encoding="utf-8")
        return SearchQAResult(output=output, trajectory=trajectory, reward=reward, usage=total_usage(responses))
