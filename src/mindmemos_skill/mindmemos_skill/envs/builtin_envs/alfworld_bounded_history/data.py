"""Adapt released ALFWorld game manifests to shared data splits."""

from hashlib import sha256
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
        """Resolve and fingerprint each selected game."""
        source = self.config.data_path.expanduser().resolve()
        root = (self.config.game_root or source.parent).expanduser().resolve()
        digest = sha256(source.read_bytes()).hexdigest()
        samples = []
        for row in load_records(source):
            game = Path(row["gamefile"]).expanduser()
            game = (root / game).resolve()
            if not game.is_file():
                raise FileNotFoundError(game)
            hashes = {
                path.name: sha256(path.read_bytes()).hexdigest()
                for path in sorted(game.parent.iterdir())
                if path.is_file()
            }
            samples.append(
                DataSample(
                    task=ALFWorldTask(
                        id=str(row["id"]),
                        instruction=row.get("task_description") or "Complete the ALFWorld task.",
                        gamefile=game,
                    ),
                    reward_data=ALFWorldRewardData(source_sha256=digest, game_sha256=hashes),
                )
            )
        return samples
