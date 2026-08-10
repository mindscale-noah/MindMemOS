"""Read and reconstruct immutable UTF-8 Skill snapshots."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from ..errors import SkillSnapshotError
from ..persistence import SkillRecord
from .bundle import compute_content_hash, deserialize_files, normalize_text
from .models import SkillSnapshot, SnapshotFile, SnapshotFileRole

_IGNORED_DIRECTORIES = frozenset({".git", "__pycache__"})
_IGNORED_FILES = frozenset({".DS_Store"})


def read_skill_snapshot(source_path: str | Path) -> SkillSnapshot:
    source = Path(source_path).expanduser()
    root = source.parent if source.is_file() and source.name == "SKILL.md" else source
    root = root.resolve()
    if not root.is_dir():
        raise SkillSnapshotError(f"Skill source does not exist or is not a directory: {root}")

    blob: dict[str, str] = {}
    resources: dict[str, str] = {}
    files: list[SnapshotFile] = []
    for path in _iter_files(root):
        relative = _relative_path(root, path)
        try:
            normalized = normalize_text(path.read_bytes().decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SkillSnapshotError(f"binary files are not supported in Skill snapshots: {relative}") from exc
        raw = normalized.encode("utf-8")
        role = _file_role(relative)
        target = blob if role is SnapshotFileRole.ALGORITHM else resources
        target[relative] = normalized
        files.append(
            SnapshotFile(
                path=relative,
                content_hash=hashlib.sha256(raw).hexdigest(),
                byte_size=len(raw),
                mode=stat.S_IMODE(path.stat().st_mode),
                media_type=mimetypes.guess_type(relative)[0],
                role=role,
            )
        )
    if "SKILL.md" not in blob:
        raise SkillSnapshotError(f"Skill source contains no SKILL.md: {root}")
    return _build_snapshot(blob=blob, resources=resources, files=files)


def snapshot_from_editor(content: str, inherited: SkillSnapshot) -> SkillSnapshot:
    files = inherited.file_contents
    files["SKILL.md"] = content
    return snapshot_from_editor_files(files, inherited)


def snapshot_from_editor_files(files: dict[str, str], inherited: SkillSnapshot) -> SkillSnapshot:
    """Build a complete edited snapshot while preserving its file manifest."""

    normalized = {validate_snapshot_path(path): normalize_text(content) for path, content in files.items()}
    inherited_paths = set(inherited.file_contents)
    if set(normalized) != inherited_paths:
        raise SkillSnapshotError("editor files must preserve the existing Skill file paths")
    if not normalized.get("SKILL.md", "").strip():
        raise SkillSnapshotError("SKILL.md cannot be empty")
    files = [item.model_copy(deep=True) for item in inherited.files]
    for index, item in enumerate(files):
        raw = normalized[item.path].encode("utf-8")
        files[index] = item.model_copy(
            update={"content_hash": hashlib.sha256(raw).hexdigest(), "byte_size": len(raw)}
        )
    return _build_snapshot(
        blob={"SKILL.md": normalized["SKILL.md"]},
        resources={path: content for path, content in normalized.items() if path != "SKILL.md"},
        files=files,
    )


def snapshot_from_cloud_content(
    content: str,
    inherited: SkillSnapshot | None,
) -> SkillSnapshot:
    """Compatibility wrapper for a single-file cloud bundle."""

    return snapshot_from_cloud_bundle({"SKILL.md": content}, inherited)


def snapshot_from_cloud_bundle(
    blob: dict[str, str],
    inherited: SkillSnapshot | None,
) -> SkillSnapshot:
    """Install a complete cloud bundle while retaining private local resources.

    The local and remote bundle both contain only ``SKILL.md``. Scripts,
    references and other resources are inherited locally and never cross the
    remote boundary.
    """

    normalized_blob = {validate_snapshot_path(path): normalize_text(text) for path, text in blob.items()}
    resources = dict(inherited.resources) if inherited is not None else {}
    files = [_snapshot_file(path, text) for path, text in normalized_blob.items()]
    if inherited is not None:
        inherited_by_path = {item.path: item for item in inherited.files}
        files.extend(inherited_by_path[path].model_copy(deep=True) for path in resources)
    return _build_snapshot(
        blob=normalized_blob,
        resources=resources,
        files=files,
    )


def snapshot_from_record(record: SkillRecord) -> SkillSnapshot:
    blob = deserialize_files(record.blob)
    resources = deserialize_files(record.resources)
    metadata = record.local_metadata.get("snapshot")
    if not isinstance(metadata, dict):
        raise SkillSnapshotError(f"version {record.version_id} has no snapshot metadata")
    raw_files = metadata.get("files")
    if not isinstance(raw_files, list):
        raise SkillSnapshotError(f"version {record.version_id} has invalid snapshot files")
    try:
        files = [SnapshotFile.model_validate(item) for item in raw_files]
    except ValueError as exc:
        raise SkillSnapshotError(f"version {record.version_id} has invalid snapshot metadata") from exc
    snapshot = _build_snapshot(blob=blob, resources=resources, files=files)
    expected_hash = metadata.get("local_snapshot_hash")
    if snapshot.content_hash != record.content_hash:
        raise SkillSnapshotError(f"version {record.version_id} content hash is corrupt")
    if snapshot.local_snapshot_hash != expected_hash:
        raise SkillSnapshotError(f"version {record.version_id} local snapshot hash is corrupt")
    return snapshot


def snapshot_metadata(snapshot: SkillSnapshot) -> dict[str, Any]:
    return {
        "local_snapshot_hash": snapshot.local_snapshot_hash,
        "files": [item.model_dump(mode="json") for item in snapshot.files],
    }


def validate_snapshot_path(path: str) -> str:
    candidate = PurePosixPath(path.replace("\\", "/"))
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise SkillSnapshotError(f"invalid snapshot-relative path: {path}")
    return candidate.as_posix()


def _build_snapshot(
    *,
    blob: dict[str, str],
    resources: dict[str, str],
    files: list[SnapshotFile],
) -> SkillSnapshot:
    normalized_blob = {validate_snapshot_path(path): normalize_text(text) for path, text in blob.items()}
    normalized_resources = {validate_snapshot_path(path): normalize_text(text) for path, text in resources.items()}
    if set(normalized_blob) != {"SKILL.md"}:
        raise SkillSnapshotError("Skill bundle must contain exactly one SKILL.md file")
    if set(normalized_blob) & set(normalized_resources):
        raise SkillSnapshotError("snapshot paths may not appear in both blob and resources")
    expected_paths = set(normalized_blob) | set(normalized_resources)
    actual_paths = [validate_snapshot_path(item.path) for item in files]
    if len(actual_paths) != len(set(actual_paths)) or set(actual_paths) != expected_paths:
        raise SkillSnapshotError("snapshot file manifest does not match stored file content")
    normalized_files = sorted(files, key=lambda item: item.path)
    contents = {**normalized_blob, **normalized_resources}
    for item in normalized_files:
        raw = contents[item.path].encode("utf-8")
        if item.content_hash != hashlib.sha256(raw).hexdigest() or item.byte_size != len(raw):
            raise SkillSnapshotError(f"snapshot file metadata does not match content: {item.path}")
    content_hash = compute_content_hash(normalized_blob)
    canonical = json.dumps(
        {
            "content_hash": content_hash,
            "files": [
                {
                    "path": item.path,
                    "content_hash": item.content_hash,
                    "mode": item.mode,
                    "role": item.role.value,
                }
                for item in normalized_files
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return SkillSnapshot(
        blob=normalized_blob,
        resources=normalized_resources,
        files=normalized_files,
        content_hash=content_hash,
        local_snapshot_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _iter_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        directory_names[:] = sorted(name for name in directory_names if name not in _IGNORED_DIRECTORIES)
        for directory_name in directory_names:
            if (current / directory_name).is_symlink():
                raise SkillSnapshotError(f"symbolic links are not supported: {current / directory_name}")
        for file_name in sorted(file_names):
            if file_name in _IGNORED_FILES or file_name.endswith(".pyc"):
                continue
            path = current / file_name
            if path.is_symlink():
                raise SkillSnapshotError(f"symbolic links are not supported: {path}")
            if path.is_file():
                paths.append(path)
    return sorted(paths)


def _relative_path(root: Path, path: Path) -> str:
    try:
        return validate_snapshot_path(path.resolve(strict=True).relative_to(root).as_posix())
    except (OSError, ValueError) as exc:
        raise SkillSnapshotError(f"snapshot file escapes Skill root: {path}") from exc


def _file_role(path: str) -> SnapshotFileRole:
    if path == "SKILL.md":
        return SnapshotFileRole.ALGORITHM
    top_level = PurePosixPath(path).parts[0].lower()
    if top_level == "scripts":
        return SnapshotFileRole.SCRIPT
    if top_level == "references":
        return SnapshotFileRole.REFERENCE
    return SnapshotFileRole.RESOURCE


def _snapshot_file(path: str, content: str) -> SnapshotFile:
    raw = content.encode("utf-8")
    return SnapshotFile(
        path=path,
        content_hash=hashlib.sha256(raw).hexdigest(),
        byte_size=len(raw),
        media_type=mimetypes.guess_type(path)[0],
        role=_file_role(path),
    )


__all__ = [
    "read_skill_snapshot",
    "snapshot_from_cloud_bundle",
    "snapshot_from_cloud_content",
    "snapshot_from_editor",
    "snapshot_from_editor_files",
    "snapshot_from_record",
    "snapshot_metadata",
    "validate_snapshot_path",
]
