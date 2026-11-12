"""Shared identity and replay checks for legacy Skill copies."""

import json

from ....common.ids import generate_stable_id
from ...constants import LEGACY_SKILL_SOURCE
from .typing import SkillVersionRecord


def legacy_target_id(project_id: str, legacy_id: str, prefix: str) -> str:
    """Return the same central ID for historical and real-time import.

    Args:
        project_id: Original legacy project namespace.
        legacy_id: Exact old family or version identity.
        prefix: Central ID prefix for the corresponding record type.
    """
    key = json.dumps([LEGACY_SKILL_SOURCE, prefix, project_id, legacy_id], separators=(",", ":"))
    return generate_stable_id(key, prefix)


def same_legacy_version(existing: SkillVersionRecord, incoming: SkillVersionRecord) -> bool:
    """Check immutable lineage and legacy provenance before accepting a replay.

    Args:
        existing: Already persisted snapshot.
        incoming: Snapshot reconstructed from the source version.
    """
    return (
        existing.origin == LEGACY_SKILL_SOURCE
        and existing.skill_id == incoming.skill_id
        and existing.parent_version_ids == incoming.parent_version_ids
        and existing.snapshot() == incoming.snapshot()
        and existing.local_metadata.get(LEGACY_SKILL_SOURCE) == incoming.local_metadata[LEGACY_SKILL_SOURCE]
    )
