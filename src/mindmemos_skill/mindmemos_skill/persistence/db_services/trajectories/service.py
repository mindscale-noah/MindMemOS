"""Store complete local rollout attempts."""

from ....infra.database import Database
from ...constants import TRAJECTORIES_TABLE
from .typing import TrajectoryRecord


class TrajectoryDataService:
    """Read and store rollout attempts on a borrowed database.

    Args:
        database: Infra database with trajectory tables registered.
    """

    def __init__(self, database: Database):
        self.database = database

    def add(self, record: TrajectoryRecord) -> None:
        """Insert a rollout attempt without overwriting existing evidence.

        Args:
            record: Complete trajectory to store.
        """
        self.database.insert(TRAJECTORIES_TABLE, [record.model_dump(mode="python")])

    def get(self, trajectory_id: str) -> TrajectoryRecord | None:
        """Read a stored attempt.

        Args:
            trajectory_id: Local physical attempt identity.

        Returns:
            The stored attempt, or None.
        """
        row = self.database.get(TRAJECTORIES_TABLE, trajectory_id)
        return TrajectoryRecord.model_validate(row) if row is not None else None
