"""Typed game selection and private simulator scoring inputs."""

from pathlib import Path

from ...data import RewardData
from ...typing import Task


class ALFWorldTask(Task):
    """Public game selection; simulator files are never exposed to the model."""

    gamefile: Path
    """Exact released game.json selected for this rollout."""


class ALFWorldRewardData(RewardData):
    """No additional private inputs are needed for simulator scoring."""
