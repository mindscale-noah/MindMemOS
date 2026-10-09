"""Public workbook staging and previews shared by both rollout protocols."""

import shutil
from pathlib import Path

from .typing import SpreadsheetBenchRewardData, SpreadsheetBenchTask


def prepare_cases(task: SpreadsheetBenchTask, reward_data: SpreadsheetBenchRewardData, workspace: Path) -> list[Path]:
    """Match case numbers and stage only public inputs in isolated case folders.

    Args:
        task: Public instruction and input workbook identities.
        reward_data: Private expected workbook identities.
        workspace: Agent-facing rollout directory.

    Returns:
        Case directories in the released order.
    """
    numbers = [case.number for case in task.cases]
    if numbers != [case.number for case in reward_data.cases] or len(set(numbers)) != len(numbers):
        raise ValueError("Public workbook cases must match private reference case numbers")
    directories = []
    for source, reference in zip(task.cases, reward_data.cases):
        directory = workspace / source.number
        if directory.is_symlink():
            raise ValueError("Case workspace must not be a symbolic link")
        directory.mkdir(exist_ok=True)
        destination = directory / "input.xlsx"
        destination.unlink(missing_ok=True)
        shutil.copyfile(source.path, destination)
        (directory / "output.xlsx").unlink(missing_ok=True)
        directories.append(directory)
    return directories
