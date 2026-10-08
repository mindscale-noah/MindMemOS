"""One LiveMath model call with public workspace files and private evaluation."""

from pathlib import Path

from ....agents import OpenAIChatAgent
from ....skill import Skill
from ...base import Environment
from ...data import RewardData
from ...typing import Task, Usage
from ..common import skill_text
from .config import LiveMathConfig
from .prompts import SYSTEM_PROMPT
from .reward import evaluate
from .typing import LiveMathResult, LiveMathRewardData, LiveMathTask


@Environment.register("livemath")
class LiveMathEnvironment(Environment[LiveMathConfig]):
    """Solve one math MCQ without exposing private references to the model."""

    task_type = LiveMathTask
    """Public LiveMath task schema."""

    reward_data_type = LiveMathRewardData
    """Private LiveMath scoring schema."""

    config_type = LiveMathConfig
    """Single-call agent and evaluation protocol settings."""

    result_type = LiveMathResult
    """Typed result used by the shared persistence and resume implementation."""

    def __init__(self, config: LiveMathConfig, *, agent: OpenAIChatAgent | None = None) -> None:
        """Create a tool-free backend or reuse an explicitly injected one.

        Args:
            config: Settings persisted as part of rollout identity.
            agent: Optional backend with exactly the same recorded configuration.

        Raises:
            ValueError: Injected agent settings differ from the audited settings.
        """
        super().__init__(config)
        if agent is not None and agent.config != config.agent:
            raise ValueError("Injected agent config must match the environment's agent config")
        self.agent = agent if agent is not None else OpenAIChatAgent(config.agent)

    def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> LiveMathResult:
        """Execute only public prompts, then compare the response with the private key."""
        if not isinstance(task, LiveMathTask) or not isinstance(reward_data, LiveMathRewardData):
            raise TypeError("LiveMath requires LiveMathTask and LiveMathRewardData")
        if reward_data.correct_choice not in task.choices:
            raise ValueError("Reference answer is not one of the displayed choices")
        text = skill_text(skills).strip()
        system = SYSTEM_PROMPT.format(skill_section=f"## Skill\n{text}\n\n" if text else "")
        choices = "\n".join(f"{choice.label}. {choice.text}" for choice in task.choices)
        user = f"## Question\n{task.instruction}\n\n## Choices\n{choices}"
        (workspace / "task.md").write_text(user, encoding="utf-8")
        (workspace / "system.md").write_text(system, encoding="utf-8")
        response = self.agent.run(system=system, user=user)
        if not isinstance(response.output, str):
            raise TypeError("LiveMath requires a text agent output")
        (workspace / "response.txt").write_text(response.output, encoding="utf-8")
        usage = Usage(**{"reported": bool(response.usage), **response.usage})
        return LiveMathResult(
            output=response.output,
            trajectory=response.trajectory,
            reward=evaluate(response.output, task, reward_data),
            usage=usage,
        )
