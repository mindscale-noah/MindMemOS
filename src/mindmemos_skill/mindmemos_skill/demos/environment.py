"""A synthetic environment with real rollout audit and resume behavior."""

from pathlib import Path
from time import sleep

from ..envs import Environment
from ..envs.data import RewardData
from ..envs.typing import Reward, RolloutResult, Task
from ..skill import Skill
from .config import DemoEnvironmentConfig


@Environment.register("demo_offline")
class DemoEnvironment(Environment[DemoEnvironmentConfig]):
    """Return a synthetic successful execution without constructing an Agent."""

    config_type = DemoEnvironmentConfig
    """Local timing settings with no external service configuration."""

    async def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Create synthetic messages and a fixed demo reward."""
        sleep(self.config.delay_seconds)
        return RolloutResult(
            output=f"Demo completed: {task.id}",
            trajectory=[
                {"role": "user", "content": task.instruction},
                {"role": "assistant", "content": f"Synthetic execution with {len(skills)} Skill(s)."},
            ],
            reward=Reward(score=1.0, success=True, detail="Synthetic demo reward; no model was called."),
        )
