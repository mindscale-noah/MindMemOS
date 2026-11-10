"""Load the pinned SkillsBench v1.0 task set using existing dataset receipts."""

import tomllib

from ....common.json import read_json
from ...data import Data, DataSample
from ...typing import Task
from .config import SkillsBenchDataConfig
from .constants import DOWNLOAD_RECEIPT, SKILLSBENCH_REVISION
from .source import task_identity
from .typing import SkillsBenchRewardData


@Data.register("skillsbench")
class SkillsBenchData(Data[SkillsBenchDataConfig]):
    """Keep original instructions public and verifier/solution paths private."""

    config_type = SkillsBenchDataConfig
    """Pinned source location and optional ordered task selection."""

    def _load(self) -> list[DataSample]:
        """Read tasks/ only and validate selected packages against their download checksums."""
        root = self.config.benchmark_root.expanduser().resolve()
        receipt = read_json(root / DOWNLOAD_RECEIPT)
        if receipt["source"]["revision"] != SKILLSBENCH_REVISION:
            raise ValueError("SkillsBench requires the pinned v1.0 source revision")
        available = {path.name: path for path in (root / "tasks").iterdir() if path.is_dir()}
        selected = self.config.task_ids or tuple(sorted(available))
        if not selected or not set(selected) <= available.keys():
            raise ValueError("Requested SkillsBench tasks are missing from tasks/")
        samples = []
        for identifier in selected:
            directory = available[identifier]
            if directory.is_symlink():
                raise ValueError("SkillsBench task directories must not be symbolic links")
            prefix = f"tasks/{identifier}/"
            private = SkillsBenchRewardData(
                task_dir=directory,
                checksums={
                    name[len(prefix) :]: digest
                    for name, digest in receipt["checksums"].items()
                    if name.startswith(prefix)
                },
            )
            task_identity(private)
            config = tomllib.loads((directory / "task.toml").read_text(encoding="utf-8"))
            if config.get("environment", {}).get("skills_dir"):
                raise ValueError("The pinned SkillsBench task must not declare an additional skills_dir")
            samples.append(
                DataSample(
                    task=Task(id=identifier, instruction=(directory / "instruction.md").read_text(encoding="utf-8")),
                    reward_data=private,
                )
            )
        return samples
