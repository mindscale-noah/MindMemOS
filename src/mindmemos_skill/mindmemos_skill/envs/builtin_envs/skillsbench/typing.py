"""Private task identities and the consumed subset of Harbor's result and ATIF contracts."""

from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue, model_validator

from ...data import RewardData

TaskId = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
"""Stable single-component benchmark task identity."""


class SkillsBenchRewardData(RewardData):
    """Keep the original task package and its integrity checks outside the agent's inputs."""

    task_dir: Path
    """Original task directory containing public environment files and private tests/solution."""

    checksums: dict[str, str]
    """Task-relative regular file paths mapped to expected SHA-256 values from the download receipt."""

    @model_validator(mode="after")
    def validate_checksums(self) -> Self:
        """Require contained normalized paths and actual SHA-256 identities."""
        if not self.checksums:
            raise ValueError("A SkillsBench task requires source checksums")
        for name, digest in self.checksums.items():
            path = PurePosixPath(name)
            if not name or path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
                raise ValueError("Task checksum paths must be normalized and relative")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("Task checksums must be lowercase SHA-256 values")
        return self


class HarborUsage(BaseModel):
    """Read native agent totals without inventing missing measurements."""

    n_input_tokens: int | None = Field(default=None, ge=0, strict=True)
    """Input tokens including cached input, as reported by Harbor."""

    n_output_tokens: int | None = Field(default=None, ge=0, strict=True)
    """Reported generated output tokens."""

    n_cache_tokens: int | None = Field(default=None, ge=0, strict=True)
    """Cached subset of input tokens, not added again to the total."""

    cost_usd: FiniteFloat | None = Field(default=None, ge=0)
    """Reported model cost in US dollars, or None when unavailable."""


class HarborFailure(BaseModel):
    """Consume a native infrastructure or agent exception."""

    exception_type: str
    """Native Harbor exception class name."""

    exception_message: str
    """Native diagnostic retained separately from scored task failure."""


class HarborVerifierResult(BaseModel):
    """Read the official numeric reward record."""

    rewards: dict[str, FiniteFloat] | None = None
    """Named verifier rewards; SkillsBench uses the reward key."""


class HarborTrialResult(BaseModel):
    """Read only scoring and usage fields; native configuration remains in private artifacts."""

    task_name: str
    """Task identity that must match the requested source package."""

    agent_result: HarborUsage | None = None
    """Reported agent measurements, when available."""

    verifier_result: HarborVerifierResult | None = None
    """Official verifier rewards, absent when verification did not finish."""

    exception_info: HarborFailure | None = None
    """A native exception; such trials must not be converted into successful scored completions."""


class AtifToolCall(BaseModel):
    """Preserve public tool actions from the pinned Harbor ATIF format."""

    tool_call_id: str
    """Native identifier joining an action to its observation."""

    function_name: str
    """Executed native tool name."""

    arguments: dict[str, JsonValue]
    """Named public tool arguments in their original JSON representation."""


class AtifObservationResult(BaseModel):
    """Public tool output, without native private metadata."""

    source_call_id: str | None = None
    """Corresponding native tool call, or None for an unpaired observation."""

    content: JsonValue = None
    """Text or structured multimodal tool result reported to the agent."""


class AtifObservation(BaseModel):
    """Ordered tool results attached to one interaction step."""

    results: list[AtifObservationResult]
    """Public observations in native order."""


class AtifStep(BaseModel):
    """Select public dialogue while excluding reasoning and unrelated native metadata."""

    source: Literal["system", "user", "agent"]
    """Native dialogue role; agent maps to assistant in the shared trajectory."""

    message: JsonValue
    """Original public dialogue content."""

    tool_calls: list[AtifToolCall] | None = None
    """Structured public actions associated with this step."""

    observation: AtifObservation | None = None
    """Public results received after the actions."""


class AtifTrajectory(BaseModel):
    """Read the main native agent trajectory without importing Harbor."""

    model_config = ConfigDict(extra="ignore")
    """Consume dialogue only; retain the complete native file in the private run directory."""

    steps: list[AtifStep] = Field(min_length=1)
    """Ordered main-agent dialogue and tool interactions."""
