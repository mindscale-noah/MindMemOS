"""Public workbook cases and private workbook references."""

from pathlib import Path

from pydantic import BaseModel, Field, JsonValue

from ...data import RewardData
from ...typing import Task, Usage


class WorkbookInput(BaseModel):
    """One public input workbook with stable case identity."""

    number: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    """Released case number, safe for output filenames."""

    path: Path
    """Public initial workbook path."""

    sha256: str
    """Initial workbook content identity."""


class WorkbookReference(BaseModel):
    """One private expected workbook."""

    number: str
    """Case number matching its public input."""

    path: Path
    """Private reference workbook path, never mounted into the agent runtime."""

    sha256: str
    """Expected workbook content identity."""


class SpreadsheetBenchTask(Task):
    """Spreadsheet instruction and public input workbooks."""

    cases: list[WorkbookInput] = Field(min_length=1)
    """Ordered public cases to which the generated script is applied."""

    instruction_type: str = ""
    """Released manipulation category."""

    answer_position: str = ""
    """Public answer-position hint supplied by the released dataset."""


class SpreadsheetBenchRewardData(RewardData):
    """Private references and evaluation regions."""

    cases: list[WorkbookReference] = Field(min_length=1)
    """Expected workbook for each released case."""

    answer_position: str
    """Exact released scoring range."""

    answer_sheet: str | None = None
    """Optional default sheet used by the shared scorer."""

    source_sha256: str
    """Dataset metadata content hash."""


class SpreadsheetInteractionResult(BaseModel):
    """Public interaction output before private workbook evaluation."""

    output: str
    """Final model response text."""

    trajectory: list[dict[str, JsonValue]]
    """Model messages and public tool observations."""

    usage: Usage
    """Aggregated model usage for the interaction."""

    diagnostics: dict[str, JsonValue]
    """Mode-specific completion and skill-loading information."""
