"""Public workbook cases and private workbook references."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, JsonValue

from ...data import RewardData
from ...typing import Task, Usage


class WorkbookInput(BaseModel):
    """One public input workbook with stable case identity."""

    number: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    """Released case number, safe for output filenames."""

    path: Path
    """Public initial workbook path."""


class WorkbookReference(BaseModel):
    """One private expected workbook."""

    number: str
    """Case number matching its public input."""

    path: Path
    """Private reference workbook path, never mounted into the agent runtime."""


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


class SpreadsheetContext(BaseModel):
    """Public, rollout-local inputs for prompt and tool hooks."""

    task: SpreadsheetBenchTask
    """Public instruction and workbook identities; excludes scoring inputs."""

    directory: Path
    """Current case workspace containing input.xlsx and output.xlsx."""

    skill_directory: Path
    """Exported guidance and auxiliary files for this case."""

    skill_text: str
    """Already assembled Skill instructions, empty for a baseline."""

    tools: list[dict] = Field(default_factory=list)
    """Tool schemas exposed to the agent for this case."""


class SpreadsheetObservation(BaseModel):
    """Decoded assistant turn and public feedback consumed by the shared loop."""

    assistant: dict[str, JsonValue]
    """Assistant history message, including native tool-call IDs when present."""

    messages: list[dict[str, JsonValue]] = Field(default_factory=list)
    """Ordered tool observations or format correction messages."""

    output: str = ""
    """Human-readable response for the rollout result."""

    complete: bool = False
    """Whether the assistant requested completion; the loop checks output files."""

    skill_loaded: bool = False
    """Whether this turn successfully loaded guidance through a tool."""

    signature: str = ""
    """Comparable action/result signature for optional stagnation detection."""

    repeated_kind: str = "action/result pair"
    """Description used when identical interactions reach the configured limit."""


class PolicyResponseType(StrEnum):
    """Released textual policy response categories."""

    ACTION = "action"
    """Execute the parsed environment tool invocation."""

    TASK_COMPLETE = "task_complete"
    """Check whether the requested output exists before finishing."""

    FORMAT_ERROR = "format_error"
    """Return the exact correction feedback for invalid action syntax."""


@dataclass(frozen=True, slots=True)
class PolicyAction:
    """One parsed tool invocation from the reference text protocol."""

    name: str
    """Requested tool name from the textual action payload."""

    arguments: dict[str, Any]
    """Tool argument mapping parsed from the action JSON."""


@dataclass(frozen=True, slots=True)
class PolicyResponse:
    """Parsed action, completion or format-error result."""

    response_type: PolicyResponseType
    """Whether the response is an action, completion or format error."""

    action: PolicyAction | None = None
    """Parsed tool invocation, or None when no action is available."""

    error_message: str | None = None
    """Correction feedback for a malformed action payload."""
