"""Adapt SearchQA records to the shared split API."""

from ...data import Data, DataSample
from ..common import load_records
from .config import SearchQADataConfig
from .typing import SearchQARewardData, SearchQATask


@Data.register("searchqa")
class SearchQAData(Data[SearchQADataConfig]):
    """Keep reference answers outside the public task."""

    config_type = SearchQADataConfig
    """Local source settings."""

    def _load(self) -> list[DataSample]:
        """Load original IDs, passages, and private answer lists."""
        path = self.config.data_path.expanduser().resolve()
        return [
            DataSample(
                task=SearchQATask(id=str(row["id"]), instruction=row["question"], context=row.get("context", "")),
                reward_data=SearchQARewardData(answers=row.get("answers", [])),
            )
            for row in load_records(path)
        ]
