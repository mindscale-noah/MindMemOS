"""Strict cloud Skill bundle validation, serialization and content hashing.

The local and cloud bundle contains exactly one canonical ``SKILL.md`` file.
Scripts, references, resources and arbitrary workspace files are rejected
instead of ignored so the server can never acknowledge an upload whose privacy
boundary differs from the client-side boundary.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath

from ...errors import SkillBundleError
from ..text import digest_text

SKILL_ROOT_FILE = "SKILL.md"
SKILL_WHITELIST: frozenset[str] = frozenset({SKILL_ROOT_FILE})
CONTENT_HASH_ALGORITHM = "sha256"


def is_whitelisted(path: str) -> bool:
    """Return whether ``path`` is a canonical cloud-bundle relative path."""

    try:
        canonical = _validate_path(path)
    except SkillBundleError:
        return False
    return canonical == SKILL_ROOT_FILE


def normalize_bundle(files: dict[str, str]) -> dict[str, str]:
    """Validate an exact cloud bundle and normalize text newlines."""

    normalized: dict[str, str] = {}
    for path, content in files.items():
        if not isinstance(path, str) or not isinstance(content, str):
            raise SkillBundleError("skill bundle must map text paths to text content")
        canonical = _validate_path(path)
        if not is_whitelisted(canonical):
            raise SkillBundleError(f"skill bundle path is not allowed: {path}")
        if canonical in normalized:
            raise SkillBundleError(f"skill bundle path is duplicated: {canonical}")
        normalized[canonical] = _normalize_newlines(content)
    if SKILL_ROOT_FILE not in normalized:
        raise SkillBundleError("skill bundle requires SKILL.md")
    return normalized


def serialize_bundle(files: dict[str, str]) -> str:
    """Serialize a validated bundle to its deterministic wire representation."""

    return _serialize_normalized(normalize_bundle(files))


def deserialize_bundle(text: str) -> dict[str, str]:
    """Decode a canonical bundle and reject every noncanonical representation."""

    try:
        records = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SkillBundleError("skill bundle content is not valid canonical JSON") from exc
    if not isinstance(records, list):
        raise SkillBundleError("skill bundle content must be a list of {path, content} records")
    files: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "content"}:
            raise SkillBundleError("skill bundle record must contain only 'path' and 'content'")
        path, content = record["path"], record["content"]
        if not isinstance(path, str) or not isinstance(content, str):
            raise SkillBundleError("skill bundle record 'path' and 'content' must be strings")
        if path in files:
            raise SkillBundleError(f"skill bundle path is duplicated: {path}")
        files[path] = content
    normalized = normalize_bundle(files)
    if _serialize_normalized(normalized) != text:
        raise SkillBundleError("skill bundle content is not canonically serialized")
    return normalized


def bundle_files_from_content(content: str) -> dict[str, str]:
    """Parse the strict canonical register/upload bundle."""

    return deserialize_bundle(content)


def compute_content_hash(files: dict[str, str]) -> str:
    """Compute SHA-256 over the exact canonical cloud bundle."""

    return digest_text(serialize_bundle(files), algorithm=CONTENT_HASH_ALGORITHM)


def _normalize_newlines(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _validate_path(path: str) -> str:
    if "\\" in path:
        raise SkillBundleError(f"skill bundle path must use POSIX separators: {path}")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise SkillBundleError(f"invalid skill bundle path: {path}")
    canonical = candidate.as_posix()
    if canonical != path:
        raise SkillBundleError(f"skill bundle path is not canonical: {path}")
    return canonical


def _serialize_normalized(files: dict[str, str]) -> str:
    records = [{"path": path, "content": files[path]} for path in sorted(files)]
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
