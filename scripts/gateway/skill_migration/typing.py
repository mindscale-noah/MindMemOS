"""Legacy inputs needed to copy registered Skills and their trajectory evidence."""

from datetime import UTC, datetime

from mindmemos.typing.skill import SkillVersion
from pydantic import BaseModel, Field, field_validator


class LegacyVersion(SkillVersion):
    """Registered version whose older payload may omit its display name."""

    skill_name: str | None = None
    """Display name recorded at registration, when present."""

    @field_validator("created_at")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        """Use the legacy service's UTC convention for naive timestamps."""
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class MigrationCounts(BaseModel):
    """Per-project counts for a preview or completed write."""

    families: int = 0
    """Registered Skill families found in the source."""

    versions: int = 0
    """Registered Skill versions found in the source."""

    new_families: int = 0
    """Families missing in the destination at the time of the run."""

    verified_families: int = 0
    """Families successfully checked in preview or committed in apply mode."""

    new_versions: int = 0
    """Versions missing in the destination at the time of the run."""

    trajectories: int = 0
    """Attempted add records, including ones whose data failed validation."""

    new_trajectories: int = 0
    """Eligible add records missing in the destination at the time of the run."""

    verified_trajectories: int = 0
    """Trajectories successfully checked in preview or committed in apply mode."""

    skipped_records: int = 0
    """Add records with bindings but no registered version in the project."""

    skipped_bindings: int = 0
    """Pending or unavailable version references excluded from the import."""

    fallback_versions: int = 0
    """Versions imported as nonstandard to preserve the original document."""


class MigrationFailure(BaseModel):
    """Safe diagnostics for locating a failed source record or scan."""

    reason: str
    """Stable category used for aggregate counts."""

    detail: str
    """Operator-facing description without payloads or connection details."""

    project_id: str | None = None
    """Original project, when its ownership could be identified."""

    skill_id: str | None = None
    """Original Skill family ID, when present in the source."""

    version_ids: list[str] = Field(default_factory=list)
    """Original version point IDs associated with the failure."""

    trajectory_id: str | None = None
    """Original trajectory point ID for an attempted trajectory."""


class ProjectMigration(BaseModel):
    """Observed migration outcomes for one discovered project."""

    project_id: str
    """Original gateway project namespace."""

    counts: MigrationCounts = Field(default_factory=MigrationCounts)
    """Source observations and proposed or committed writes."""

    failed_skills: list[MigrationFailure] = Field(default_factory=list)
    """Failed Skill families, including their source version IDs."""

    failed_trajectories: list[MigrationFailure] = Field(default_factory=list)
    """Failed source trajectories with their point IDs."""

    scan_failures: list[MigrationFailure] = Field(default_factory=list)
    """Interrupted collection scans, for which the project counts are partial."""


class MigrationRun(BaseModel):
    """Collect the outcomes needed for a final server-wide report."""

    projects_discovered: int = 0
    """Number of project namespaces found by the completed discovery scan."""

    projects: list[ProjectMigration] = Field(default_factory=list)
    """Observed results for projects that were processed."""

    unscoped_skills: list[MigrationFailure] = Field(default_factory=list)
    """Invalid registered versions that could not be assigned to a project."""

    run_failures: list[MigrationFailure] = Field(default_factory=list)
    """Fatal setup or discovery failures and operator interruption."""
