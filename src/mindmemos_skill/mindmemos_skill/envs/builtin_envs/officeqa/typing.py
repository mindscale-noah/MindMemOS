"""Separate public document access from private OfficeQA answers."""

from pathlib import Path

from pydantic import Field

from ...data import RewardData
from ...typing import Task


class OfficeQATask(Task):
    """A question and an explicit public document allowlist."""

    task_type: str = "officeqa"
    """Released task category used by reflection headers."""

    documents: dict[str, Path]
    """Public relative document names mapped to released text files."""

    source_files: list[str] = Field(default_factory=list)
    """Candidate filenames supplied by the dataset."""

    source_docs: list[str] = Field(default_factory=list)
    """Public source-title hints supplied by the dataset."""

    oracle_context: str = ""
    """Public parsed source pages selected by the released source URL page numbers."""


class OfficeQARewardData(RewardData):
    """Private expected answer used after public document research."""

    ground_truth: str
    """Expected direct answer; never passed to the research agent."""
