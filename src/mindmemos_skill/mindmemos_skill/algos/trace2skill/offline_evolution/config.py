"""Strict configuration for the offline evolution algorithm.

This is the *contract* between the experiment YAML and ``optimize()``. It is
deliberately separate from the vendored pipeline's own ``config.py``: the host
represents every affect-the-result parameter here with Pydantic validation,
then maps them onto the vendored ``PipelineConfig`` internally.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..collection import TaskCollectionConfig

BackendName = Literal["react"]

# A scope name becomes a single path segment under the source layout, so it
# must be an unambiguous slug: no separators, no whitespace normalization
# tricks, and no filesystem traversal. It is not the Skill name.
SCOPE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class OfflineEvolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Vendored pipeline selection.
    backend: BackendName = Field(default="react", min_length=1)
    skill_md_only: bool = Field(default=True)

    # Where the vendored pipeline materialises one run. The adapter resolves it
    # under the experiment output_dir; when unset the wrapper falls back to a
    # throwaway directory so a bare optimize() call still works.
    run_root: Path | None = Field(default=None)

    # Data contract knobs.
    min_trajectories: int = Field(default=1, ge=1)
    max_trajectories: int = Field(default=8, ge=1)
    # Collapse duplicate trajectory ids before handing them to the pipeline.
    dedupe_trajectories: bool = Field(default=True)

    # Active collection for tasks-only / hybrid requests. When None, optimize()
    # must be given offline trajectories (adapter pre-collects); when set and
    # the request carries tasks, the wrapper collects them itself.
    collection: TaskCollectionConfig | None = Field(default=None)

    # Scope name written inside the vendored source_dir layout. The pipeline
    # is scope-first: every trajectory and skill lives under one scope dir.
    # This is a scope isolation label, not the skill name.
    scope_name: str = Field(default="default", min_length=1)

    # Model injection (host side); the vendored pipeline reads the same model
    # from its own config, and this field is mirrored into an env override.
    optimizer_model: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_ranges(self) -> OfflineEvolutionConfig:
        if not self.skill_md_only:
            raise ValueError("offline_evolution currently supports only SKILL.md evolution")
        if self.max_trajectories < self.min_trajectories:
            raise ValueError("max_trajectories must be >= min_trajectories")
        if SCOPE_NAME_RE.fullmatch(self.scope_name) is None:
            raise ValueError(
                "scope_name must be a single path-safe slug: letters, digits, '.', '_', '-'; "
                "it may not contain separators or any '..' traversal"
            )
        if self.scope_name in {".", ".."}:
            raise ValueError("scope_name may not be '.' or '..'")
        return self
