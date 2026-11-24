"""Convert a legacy Skill snapshot without rewriting its source text."""

from pydantic import JsonValue

from ..persistence.constants import LEGACY_SKILL_SOURCE
from ..persistence.db_services.skills.legacy import legacy_target_id
from ..persistence.db_services.skills.typing import SkillVersionRecord
from ..skill.content import read_frontmatter
from ..skill.typing import SkillContent, SkillSnapshot, parse_skill_header
from .typing import LegacySkillVersion


def _display_value(value: JsonValue) -> str | None:
    """Keep only nonempty text for required display metadata."""
    if isinstance(value, str):
        return value.strip() or None
    return None


def build_legacy_version(
    project_id: str, source: LegacySkillVersion, *, label: str | None = None
) -> SkillVersionRecord:
    """Convert the same version identity and snapshot for live delivery and backfill.

    Args:
        project_id: Original project namespace from the trusted source.
        source: Legacy version metadata and exact SKILL.md text.
        label: Optional local display label assigned by the historical importer.

    Returns:
        Central snapshot retaining the original status and provenance.
    """
    try:
        name, description, metadata = parse_skill_header(source.content)
        standard = True
    except ValueError:
        try:
            metadata = read_frontmatter(source.content)
        except ValueError:
            metadata = {}
        name = (
            _display_value(metadata.pop("name", None))
            or _display_value(source.skill_name)
            or f"legacy-skill-{source.cloud_skill_id}"
        )
        description = _display_value(metadata.pop("description", None)) or "Imported from Memory."
        standard = False
    snapshot = SkillSnapshot.from_content(
        SkillContent(content=source.content, name=name, description=description, standard_format=standard)
    )
    parent = source.parent_version_id
    return SkillVersionRecord(
        **snapshot.model_dump(),
        project_id=project_id,
        skill_id=legacy_target_id(project_id, source.cloud_skill_id, "sk-"),
        version_id=legacy_target_id(project_id, source.version_id, "skv-"),
        parent_version_ids=[legacy_target_id(project_id, parent, "skv-")] if parent else [],
        version_label=label,
        origin=LEGACY_SKILL_SOURCE,
        status=source.status,
        metadata=metadata,
        local_metadata={
            LEGACY_SKILL_SOURCE: {
                "cloud_skill_id": source.cloud_skill_id,
                "version_id": source.version_id,
                "content_hash": source.content_hash,
                "name": source.skill_name,
                "version_label": source.version_label,
                "origin": source.origin,
            }
        },
        created_at=source.created_at,
        updated_at=source.created_at,
    )
