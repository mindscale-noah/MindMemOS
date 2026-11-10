"""Adapt released ALFWorld game manifests to shared data splits."""

from pathlib import Path

from ...data import Data, DataSample
from ..common import load_records
from .config import ALFWorldDataConfig
from .typing import ALFWorldRewardData, ALFWorldTask


@Data.register("alfworld_bounded_history")
class ALFWorldData(Data[ALFWorldDataConfig]):
    """Load fixed games without starting simulator resources."""

    config_type = ALFWorldDataConfig
    """Game-manifest settings."""

    def _load(self) -> list[DataSample]:
        """Resolve each selected game and validate its source path."""
        source = self.config.data_path.expanduser().resolve()
        root = (self.config.game_root or source.parent).expanduser().resolve()
        samples = []
        for row in load_records(source):
            game = Path(row["gamefile"]).expanduser()
            game = (root / game).resolve()
            if not game.is_file():
                raise FileNotFoundError(game)
            samples.append(
                DataSample(
                    task=ALFWorldTask(
                        id=str(row["id"]),
                        instruction=row.get("task_description") or "Complete the ALFWorld task.",
                        gamefile=game,
                    ),
                    reward_data=ALFWorldRewardData(),
                )
            )
        return samples
