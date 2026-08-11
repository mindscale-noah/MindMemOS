"""Cloud Skill bundle serialization helpers."""

from __future__ import annotations

from collections.abc import Mapping

from ..contracts import SkillBundle, parse_skill_bundle

CLOUD_SKILL_ROOT_FILE = "SKILL.md"


def normalize_remote_skill_bundle(blob: Mapping[str, str]) -> dict[str, str]:
    bundle = SkillBundle.from_files(dict(blob))
    return {item.path: item.content for item in bundle.files}


def serialize_remote_skill_content(blob: Mapping[str, str]) -> str:
    return SkillBundle.from_files(dict(blob)).canonical_json()


def deserialize_remote_skill_content(content: str) -> dict[str, str]:
    bundle = parse_skill_bundle(content)
    return {item.path: item.content for item in bundle.files}


def compute_remote_skill_content_hash(content: str) -> str:
    return parse_skill_bundle(content).content_hash


def is_remote_skill_bundle_path(path: str) -> bool:
    try:
        SkillBundle.from_files({path: ""})
    except ValueError:
        return False
    return path == CLOUD_SKILL_ROOT_FILE
