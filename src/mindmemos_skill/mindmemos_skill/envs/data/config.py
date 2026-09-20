"""Validation of mutually exclusive random and file-based split options."""

from math import fsum, isclose
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, model_validator


class SplitConfig(BaseModel):
    """Select either seeded proportions or an explicit split manifest."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown split options."""

    ratios: dict[str, Annotated[FiniteFloat, Field(ge=0, le=1)]] | None = None
    """Named proportions summing to one, or None for file-based splitting."""

    seed: StrictInt | None = None
    """Required local random seed for ratio splitting; unused in file mode."""

    split_file: Path | None = None
    """JSON file mapping split names to ordered task IDs."""

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        """Require one complete split mode and valid proportions.

        Returns:
            The validated configuration.

        Raises:
            ValueError: Modes conflict, required options are missing, or ratios are invalid.
        """
        if self.split_file is not None:
            if self.ratios is not None or self.seed is not None:
                raise ValueError("split_file cannot be combined with ratios or seed")
        else:
            if not self.ratios or self.seed is None:
                raise ValueError("Random splitting requires nonempty ratios and an explicit seed")
            if any(not name.strip() for name in self.ratios):
                raise ValueError("Split names must not be blank")
            if not isclose(fsum(self.ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Split ratios must sum to one")
        return self
