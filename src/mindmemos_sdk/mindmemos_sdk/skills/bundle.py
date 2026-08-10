"""SDK compatibility helpers for the Skill-owned cloud bundle contract."""

from __future__ import annotations

import os
from pathlib import Path

from mindmemos_skill.remote import (
    CLOUD_SKILL_ROOT_FILE,
    compute_remote_skill_content_hash,
    deserialize_remote_skill_content,
    is_remote_skill_bundle_path,
    normalize_remote_skill_bundle,
    serialize_remote_skill_content,
)

from ..errors import SkillBundleError

SKILL_WHITELIST: frozenset[str] = frozenset({CLOUD_SKILL_ROOT_FILE})
CONTENT_HASH_ALGORITHM = "sha256"


def is_whitelisted(path: str) -> bool:
    return is_remote_skill_bundle_path(path)


def resolve_skill_dir(skill_path: str | os.PathLike[str]) -> Path:
    """Resolve a Skill directory from a directory or canonical ``SKILL.md`` path."""

    path = Path(skill_path).expanduser()
    if path.is_file():
        if path.name != CLOUD_SKILL_ROOT_FILE:
            raise SkillBundleError(f"skill file is not a bundle root: {path}")
        return path.parent
    return path


def read_local_bundle(skill_path: str | os.PathLike[str]) -> dict[str, str]:
    """Read only ``SKILL.md``; never inspect scripts or other directories."""

    root = resolve_skill_dir(skill_path)
    if not root.is_dir():
        raise SkillBundleError(f"skill path does not exist or is not a directory/SKILL.md file: {root}")
    files: dict[str, str] = {}
    skill_file = root / CLOUD_SKILL_ROOT_FILE
    if skill_file.is_symlink():
        raise SkillBundleError(f"symbolic links are not supported in cloud bundles: {skill_file}")
    if skill_file.is_file():
        files[CLOUD_SKILL_ROOT_FILE] = _read_text(skill_file)
    return normalize_bundle(files)


def normalize_bundle(files: dict[str, str]) -> dict[str, str]:
    try:
        return normalize_remote_skill_bundle(files)
    except ValueError as exc:
        raise SkillBundleError(str(exc)) from exc


def serialize_bundle(files: dict[str, str]) -> str:
    try:
        return serialize_remote_skill_content(files)
    except ValueError as exc:
        raise SkillBundleError(str(exc)) from exc


def deserialize_bundle(text: str) -> dict[str, str]:
    try:
        return deserialize_remote_skill_content(text)
    except ValueError as exc:
        raise SkillBundleError(str(exc)) from exc


def bundle_files_from_content(content: str) -> dict[str, str]:
    return deserialize_bundle(content)


def compute_content_hash(files: dict[str, str]) -> str:
    return compute_remote_skill_content_hash(serialize_bundle(files))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillBundleError(f"binary files are not supported in cloud bundles: {path}") from exc
