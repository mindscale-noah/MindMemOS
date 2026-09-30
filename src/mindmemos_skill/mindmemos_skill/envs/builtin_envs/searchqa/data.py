"""Adapt SearchQA records to the shared split API."""

from hashlib import sha256

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
        digest = sha256(path.read_bytes()).hexdigest()
        return [
            DataSample(
                task=SearchQATask(id=str(row["id"]), instruction=row["question"], context=row.get("context", "")),
                reward_data=SearchQARewardData(answers=row.get("answers", []), source_sha256=digest),
            )
            for row in load_records(path)
        ]
