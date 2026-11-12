"""Frozen execution and publication artifacts owned by the evolution pipeline."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ....api.typing import EvolutionRequest
from ....envs.typing import Task
from ....persistence import SkillRecord, SkillVersionRecord


class HistoricalTask(Task):
    """Describe recorded task metadata without requiring a separately extracted instruction."""

    instruction: str = Field(min_length=0)
    """Recorded instruction, preserving empty or whitespace-only values when supplied."""


class EvolutionInputMember(BaseModel):
    """Exact repository ownership and source version captured before evolution."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared captured inputs."""

    owner: SkillRecord
    """Family with its expected active pointer captured before analysis."""

    base: SkillVersionRecord
    """Exact historical version supplied to the algorithm."""

    @model_validator(mode="after")
    def validate_owner(self):
        """Require the source version to belong to its captured visible owner."""
        if self.owner.is_deleted or (self.owner.project_id, self.owner.skill_id) != (
            self.base.project_id,
            self.base.skill_id,
        ):
            raise ValueError("Evolution source does not belong to its captured owner")
        return self


class EvolutionRunManifest(BaseModel):
    """Bind a candidate directory to an authorized immutable execution."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared execution manifest fields."""

    run_id: str
    """Stable run directory identity."""

    project_id: str
    """Project permitted to inspect and publish this run."""

    input_hash: str
    """Compared execution configuration fingerprint."""

    request: EvolutionRequest
    """Original scoped request including mutation and activation policy."""

    algorithm: str
    """Registered evolver name used for repository provenance."""

    configuration: dict[str, JsonValue]
    """Algorithm and evidence collection configuration with credentials redacted."""


class CapturedEvolutionInputs(BaseModel):
    """Exact owners and parent snapshots reused after algorithm interruption."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized capture artifacts."""

    inputs: dict[str, EvolutionInputMember]
    """Original Skill IDs mapped to ownership and exact source versions."""
