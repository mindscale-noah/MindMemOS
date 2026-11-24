"""Dataset samples, private evaluation inputs, and persisted split membership."""

from typing import Self

from pydantic import BaseModel, ConfigDict, RootModel, SerializeAsAny, model_validator

from ..typing import Task


class RewardData(BaseModel):
    """Base for dataset-specific private scoring inputs; extend with declared fields."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared inputs instead of silently dropping evaluation data."""


class DataSample(BaseModel):
    """Bind a public task to its private evaluation inputs."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown sample fields."""

    task: SerializeAsAny[Task]
    """Public task with an ID unique within the loaded dataset."""

    reward_data: SerializeAsAny[RewardData]
    """Private scoring inputs, such as accepted answers or reference file paths."""


class SplitManifest(RootModel[dict[str, list[str]]]):
    """Map split names to ordered, globally unique task IDs."""

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        """Validate names and prohibit duplicate membership.

        Returns:
            The validated manifest.

        Raises:
            ValueError: A name or ID is blank, or an ID occurs more than once.
        """
        if not self.root:
            raise ValueError("At least one split is required")
        seen: set[str] = set()
        for name, identifiers in self.root.items():
            if not name.strip():
                raise ValueError("Split names must not be blank")
            for identifier in identifiers:
                if not identifier.strip():
                    raise ValueError("Task IDs must not be blank")
                if identifier in seen:
                    raise ValueError(f"Duplicate task ID in splits: {identifier!r}")
                seen.add(identifier)
        return self
