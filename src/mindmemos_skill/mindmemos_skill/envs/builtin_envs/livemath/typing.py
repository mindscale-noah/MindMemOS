"""Typed official records and public/private LiveMath task data."""

from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from ...data.typing import RewardData
from ...typing import Reward, RolloutResult, Task


class Choice(BaseModel):
    """One labeled answer option."""

    label: str = Field(min_length=1)
    """Canonical uppercase option label."""

    text: str = Field(min_length=1)
    """Full option wording."""

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        """Normalize labels for evaluation.

        Args:
            value: Source choice label.

        Returns:
            Uppercase label without trailing punctuation.

        Raises:
            ValueError: The normalized label is blank.
        """
        value = value.strip().upper().rstrip(".):")
        if not value:
            raise ValueError("Choice label must not be blank")
        return value


class SourceMCQ(BaseModel):
    """Relevant MCQ fields from the official monthly data."""

    question: str = Field(min_length=1)
    """Question text, without private metadata."""

    choices: list[Choice] = Field(min_length=1, max_length=26)
    """Source options, which can omit the separately stored correct option."""

    correct_choice: Choice
    """Private answer key and missing option wording where applicable."""

    @model_validator(mode="after")
    def validate_choices(self) -> Self:
        """Reject ambiguous labels or conflicting answer wording.

        Returns:
            The validated source MCQ.

        Raises:
            ValueError: Labels repeat or the answer key conflicts with an option.
        """
        options = {choice.label: choice.text for choice in self.choices}
        if len(options) != len(self.choices):
            raise ValueError("Duplicate option labels")
        if self.correct_choice.label in options and options[self.correct_choice.label] != self.correct_choice.text:
            raise ValueError("Correct option text conflicts with the displayed option")
        if self.correct_choice.label not in options and len(options) == 26:
            raise ValueError("At most 26 completed choices are supported")
        return self


class SourceRecord(BaseModel):
    """Only the source fields required for public tasks and private scoring."""

    month: str = Field(min_length=1)
    """Source month used in the stable task ID."""

    no: int | str
    """Original item number within that month."""

    mcq: SourceMCQ
    """Question, options, and private answer key."""

    theorem_type: list[str] = Field(default_factory=list)
    """Public mathematical task categories from the original benchmark."""

    @field_validator("theorem_type", mode="before")
    @classmethod
    def normalize_theorem_type(cls, value):
        """Normalize scalar and list categories from official monthly releases.

        Args:
            value: Source category string, list, or null.

        Returns:
            Nonempty category labels in source order.
        """
        if value is None:
            return []
        items = [value] if isinstance(value, str) else value
        return [str(item).strip() for item in items if str(item).strip()]


class LiveMathTask(Task):
    """Public multiple-choice task; no theorem, sketch, or answer metadata."""

    choices: list[Choice]
    """Complete displayed options after optional deterministic shuffling."""

    task_type: str = "math_mcq"
    """Public task category retained for algorithm-specific reflection rendering."""


class LiveMathRewardData(RewardData):
    """Private reference data used only after the agent has answered."""

    correct_choice: Choice
    """Correct option under the displayed labels."""

    source_file: str
    """Path to the source monthly JSON file."""

    source_sha256: str
    """Source file content hash, preventing stale resume after dataset edits."""

    choice_seed: int | None
    """Choice-shuffle seed used when building the task."""


class LiveMathReward(Reward):
    """Exact-choice scoring details."""

    predicted_label: str
    """Parsed model answer label."""

    predicted_text: str
    """Option wording corresponding to the parsed label, if recognized."""

    correct_label: str
    """Private expected label for auditing."""

    correct_text: str
    """Private expected option wording for auditing."""


class LiveMathResult(RolloutResult):
    """Restore typed LiveMath reward fields when resuming."""

    reward: LiveMathReward
    """Exact-choice reward and detailed answer comparison."""
