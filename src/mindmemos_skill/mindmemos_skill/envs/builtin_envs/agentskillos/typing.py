"""Public queries and private binary relevance labels."""

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from ...data import RewardData
from ...typing import Task

SkillID = Annotated[str, Field(strict=True, min_length=1, pattern=r"^\S+$")]
"""Exact benchmark identity, with no case or prefix normalization."""


class AgentSkillOSTask(Task):
    """Expose task instructions and public inputs, without evaluation rules."""

    files: dict[str, Path] = Field(default_factory=dict)
    """Public destination names mapped to original input files."""

    outputs: list[str] = Field(default_factory=list)
    """Expected deliverable names supplied to the execution agent."""


class TaskRecord(BaseModel):
    """Read the prepared benchmark task schema."""

    task_id: SkillID
    """Stable original task identifier."""

    instruction_text: str = Field(min_length=1)
    """Public query, without relevance labels or reference outputs."""


class RelevanceRecord(BaseModel):
    """Read binary core labels while ignoring unrelated source annotations."""

    core_gt_ids: list[SkillID]
    """Relevant Skill IDs; empty core sets are rejected by this protocol."""

    @field_validator("core_gt_ids")
    @classmethod
    def validate_ids(cls, values: list[str]) -> list[str]:
        """Require an unambiguous nonempty reference set.

        Args:
            values: Original benchmark core Skill IDs.
        """
        if not values or len(values) != len(set(values)):
            raise ValueError("core_gt_ids must be nonempty and unique")
        return values


class RetrievalRewardData(RewardData, RelevanceRecord):
    """Keep relevance ground truth out of the public task."""


class Ranking(BaseModel):
    """Validate rankings before metric calculation."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared ranking fields."""

    skill_ids: list[SkillID]
    """Ordered unique retrieved Skill IDs, possibly empty."""

    @field_validator("skill_ids")
    @classmethod
    def validate_ids(cls, values: list[str]) -> list[str]:
        """Prevent repeated hits from inflating discounted gains.

        Args:
            values: Ranked benchmark identities.
        """
        if len(values) != len(set(values)):
            raise ValueError("Ranking contains duplicate Skill IDs")
        return values


class AgentSkillOSRewardData(RewardData):
    """Retain the original evaluator definition outside the execution workspace."""

    task_config: dict[str, JsonValue]
    """Complete original task JSON consumed by the official evaluator."""

    core_gt_ids: list[SkillID] | None = None
    """Relevant Skill identities for optional retrieval diagnostics only."""

    relevance_source: str
    """Explicit label origin: legacy_core_gt_ids or official_task_skills."""
