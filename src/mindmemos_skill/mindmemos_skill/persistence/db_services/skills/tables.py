"""Business tables owned by the skills data service."""

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)

from ....infra.database import TableSpec
from ....infra.database.typing import UTCDateTime
from ...constants import SKILL_VERSION_LABEL_INDEX, SKILL_VERSIONS_TABLE, SKILLS_TABLE
from .typing import SkillRecord, SkillVersionRecord


def build_tables(metadata: MetaData) -> tuple[TableSpec, ...]:
    """Declare tables without creating resources.

    Args:
        metadata: Shared metadata for cross-service foreign keys.

    Returns:
        Table registrations owned by this service.
    """
    table = Table(
        SKILL_VERSIONS_TABLE,
        metadata,
        Column("project_id", Text, nullable=False, comment="Project owning this local Skill version."),
        Column(
            "version_id",
            Text,
            primary_key=True,
            comment="Local version identity; never inferred from a remote version ID.",
        ),
        Column("skill_id", Text, nullable=False, comment="Local Skill family shared by all of its versions."),
        Column("name", Text, nullable=False, comment="Human-readable Skill name."),
        Column("description", Text, nullable=True, comment="Optional explanation of the Skill."),
        Column(
            "parent_version_ids",
            JSON,
            nullable=False,
            comment="Ordered local parent IDs; remote single-parent projection is separate.",
        ),
        Column(
            "bundle",
            Text,
            nullable=False,
            comment="Serialized Skill bundle, preserved without cloud-specific rewriting.",
        ),
        Column("resources", Text, nullable=False, comment="Serialized auxiliary files outside the Skill bundle."),
        Column(
            "content_hash", Text, nullable=False, comment="Hash of the local canonical bundle; not the remote hash."
        ),
        Column(
            "local_snapshot_hash", Text, nullable=True, comment="Optional hash including local auxiliary resources."
        ),
        Column("implementation", Text, nullable=False, comment="Registered concrete Skill implementation name."),
        Column(
            "config_schema_version", Integer, nullable=False, comment="Version of the fixed Skill configuration format."
        ),
        Column(
            "config",
            JSON,
            nullable=False,
            comment="Fixed Skill implementation options; mutable content belongs in bundle.",
        ),
        Column("status", Text, nullable=False, comment="Local lifecycle state; remote lifecycle is stored separately."),
        Column("version_revision", Integer, nullable=False, comment="Local metadata revision for conditional updates."),
        Column(
            "version_label",
            Text,
            nullable=True,
            comment="Unique semantic display label within the project and Skill.",
        ),
        Column("commit_message", Text, nullable=True, comment="Reason for creating the version."),
        Column("metadata", JSON, nullable=False, comment="Portable Skill metadata."),
        Column("local_metadata", JSON, nullable=False, comment="Local-only annotations."),
        Column("origin", Text, nullable=False, comment="Local provenance such as local, imported, or evolved."),
        Column("created_at", UTCDateTime(), nullable=False, comment="Time this local version was created."),
        Column("updated_at", UTCDateTime(), nullable=False, comment="Time local metadata was last changed."),
        UniqueConstraint("project_id", "version_id"),
        Index("skill_versions_owner_version_uq", "project_id", "skill_id", "version_id", unique=True),
        CheckConstraint("length(project_id) > 0"),
        Index("skill_versions_family_idx", "project_id", "skill_id", "created_at", "version_id"),
        Index("skill_versions_content_idx", "content_hash"),
        Index(SKILL_VERSION_LABEL_INDEX, "project_id", "skill_id", "version_label", unique=True),
        CheckConstraint("version_revision >= 0 AND config_schema_version >= 1"),
    )
    skills = Table(
        SKILLS_TABLE,
        metadata,
        Column("project_id", Text, primary_key=True, comment="Local project owning the Skill."),
        Column("skill_id", Text, primary_key=True, comment="Stable central identity independent of paths."),
        Column("name", Text, nullable=False, comment="Mutable display name; not a unique identity."),
        Column("alias", Text, nullable=True, comment="Optional exact-match shortcut unique within the project."),
        Column("registration_source", Text, nullable=True, comment="Initial source for provenance; never watched."),
        Column("registration_message", Text, nullable=True, comment="Initial registration explanation."),
        Column(
            "current_version_id", Text, nullable=False, comment="Only selected version for default reads and exports."
        ),
        Column("created_at", UTCDateTime(), nullable=False, comment="Central registration time."),
        Column("updated_at", UTCDateTime(), nullable=False, comment="Last metadata or selected-version change."),
        UniqueConstraint("project_id", "alias"),
        ForeignKeyConstraint(
            ["project_id", "skill_id", "current_version_id"],
            [
                f"{SKILL_VERSIONS_TABLE}.project_id",
                f"{SKILL_VERSIONS_TABLE}.skill_id",
                f"{SKILL_VERSIONS_TABLE}.version_id",
            ],
            ondelete="RESTRICT",
        ),
        CheckConstraint("length(project_id) > 0 AND length(skill_id) > 0 AND length(name) > 0"),
        CheckConstraint("length(current_version_id) > 0"),
        CheckConstraint("alias IS NULL OR length(alias) > 0"),
        CheckConstraint("updated_at >= created_at"),
    )
    return (
        TableSpec(name=SKILL_VERSIONS_TABLE, table=table, insert_model=SkillVersionRecord),
        TableSpec(name=SKILLS_TABLE, table=skills, insert_model=SkillRecord),
    )
