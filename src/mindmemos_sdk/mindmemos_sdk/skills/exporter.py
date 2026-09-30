"""Preview and export a Skill entrypoint without changing its registry."""

import hashlib
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from ..errors import SkillInstallerError
from .bundle import bundle_files_from_content, compute_content_hash
from .models import SkillExportPlan


def _target(directory: str) -> Path:
    root = Path(directory).expanduser().absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise SkillInstallerError("Destination must not contain symbolic links")
    if root.exists() and not root.is_dir():
        raise SkillInstallerError("Destination must be a directory")
    target = root / "SKILL.md"
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise SkillInstallerError("SKILL.md must be a regular file")
    return target


def _fingerprint(target: Path) -> str | None:
    return hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None


def plan_export(content: str, directory: str) -> SkillExportPlan:
    """Preview a copy of SKILL.md without writing files.

    Args:
        content: Source bundle obtained from a verified version.
        directory: Final Skill directory, not its parent directory.

    Returns:
        A plan carrying source and destination fingerprints.
    """
    target = _target(directory)
    return SkillExportPlan(
        directory=str(target.parent),
        content_hash=compute_content_hash(bundle_files_from_content(content)),
        previous_hash=_fingerprint(target),
        content=content,
    )


def apply_export(plan: SkillExportPlan, backup_root: str) -> dict[str, str | None]:
    """Verify a preview, back up existing bytes, and atomically export SKILL.md.

    Args:
        plan: Previously reviewed source and destination snapshot.
        backup_root: Directory receiving an original file before replacement.

    Returns:
        Destination and optional backup file paths.

    Raises:
        SkillInstallerError: Source or destination changed since preview.
    """
    target = _target(plan.directory)
    files = bundle_files_from_content(plan.content)
    if compute_content_hash(files) != plan.content_hash or _fingerprint(target) != plan.previous_hash:
        raise SkillInstallerError("Export plan is stale; preview again")
    backup = None
    if target.exists():
        backup = Path(backup_root) / "exports" / uuid4().hex / "SKILL.md"
        backup.parent.mkdir(parents=True, exist_ok=False)
        backup.write_bytes(target.read_bytes())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(files["SKILL.md"])
            stream.flush()
            os.fsync(stream.fileno())
        if _fingerprint(_target(plan.directory)) != plan.previous_hash:
            raise SkillInstallerError("Destination changed during export; preview again")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(target), "backup_path": str(backup) if backup else None}
