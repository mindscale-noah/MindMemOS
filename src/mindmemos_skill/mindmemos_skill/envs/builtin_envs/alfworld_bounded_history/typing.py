"""Typed game identity and private simulator provenance."""

from pathlib import Path

from ...data import RewardData
from ...typing import Task


class ALFWorldTask(Task):
    """Public game selection; simulator files are never exposed to the model."""

    gamefile: Path
    """Exact released game.json selected for this rollout."""


class ALFWorldRewardData(RewardData):
    """Content identities of the selected game and supporting files."""

    source_sha256: str
    """Items-file content identity."""

    game_sha256: dict[str, str]
    """Game-directory filenames mapped to content hashes."""
