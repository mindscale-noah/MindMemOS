"""Recoverable export of managed snapshot files into external directories."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from ..errors import SkillExportError
from .models import SkillSnapshot
from .snapshot import validate_snapshot_path


class SkillInstaller:
    """Materialize a snapshot without deleting files outside that snapshot."""

    def __init__(self, *, managed_root: str | Path | None = None) -> None:
        self._managed_root = Path(managed_root).expanduser().resolve() if managed_root is not None else None

    def export(self, snapshot: SkillSnapshot, target_path: str | Path, *, replace: bool) -> Path:
        target = self._validate_target(target_path)
        if target.exists() and not target.is_dir():
            raise SkillExportError(f"export target is not a directory: {target}")
        if target.exists() and not replace:
            raise SkillExportError(f"export target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}-export-"))
        backup = target.parent / f".{target.name}-backup-{uuid.uuid4().hex}"
        try:
            self._write_staging(staging, snapshot)
            self._verify_files(staging, snapshot)
            if not target.exists():
                os.replace(staging, target)
                return target
            backup.mkdir()
            self._merge_existing(staging, target, backup, snapshot)
            return target
        except SkillExportError:
            raise
        except BaseException as exc:
            raise SkillExportError(f"failed to export Skill snapshot: {exc}") from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)

    def _validate_target(self, target_path: str | Path) -> Path:
        expanded = Path(target_path).expanduser()
        if expanded.is_symlink():
            raise SkillExportError(f"export target cannot be a symbolic link: {expanded}")
        target = expanded.resolve()
        unsafe = {Path(target.anchor), Path.home().resolve()}
        if self._managed_root is not None:
            unsafe.add(self._managed_root)
        if target in unsafe:
            raise SkillExportError(f"refusing unsafe export target: {target}")
        if self._managed_root is not None and (
            target.is_relative_to(self._managed_root) or self._managed_root.is_relative_to(target)
        ):
            raise SkillExportError(f"export target overlaps managed Skill state: {target}")
        return target

    @staticmethod
    def _write_staging(staging: Path, snapshot: SkillSnapshot) -> None:
        contents = snapshot.file_contents
        for entry in snapshot.files:
            destination = staging / validate_snapshot_path(entry.path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(contents[entry.path], encoding="utf-8")
            if entry.mode is not None:
                destination.chmod(entry.mode)

    def _merge_existing(self, staging: Path, target: Path, backup: Path, snapshot: SkillSnapshot) -> None:
        applied: list[tuple[Path, Path | None]] = []
        try:
            for entry in snapshot.files:
                relative = validate_snapshot_path(entry.path)
                destination = self._safe_destination(target, relative)
                source = staging / relative
                saved: Path | None = None
                if destination.exists() or destination.is_symlink():
                    if destination.is_symlink() or not destination.is_file():
                        raise SkillExportError(f"managed export path is not a regular file: {destination}")
                    saved = backup / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, saved)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                applied.append((destination, saved))
            self._verify_files(target, snapshot)
        except BaseException:
            self._restore(applied)
            raise

    @staticmethod
    def _safe_destination(root: Path, relative: str) -> Path:
        current = root
        for part in Path(relative).parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise SkillExportError(f"managed export path traverses a symbolic link: {current}")
            if current.exists() and not current.is_dir():
                raise SkillExportError(f"managed export parent is not a directory: {current}")
        return root / relative

    @staticmethod
    def _restore(applied: list[tuple[Path, Path | None]]) -> None:
        for destination, saved in reversed(applied):
            if saved is not None and saved.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(saved, destination)
            else:
                with contextlib.suppress(FileNotFoundError):
                    destination.unlink()

    @staticmethod
    def _verify_files(root: Path, snapshot: SkillSnapshot) -> None:
        for entry in snapshot.files:
            path = root / validate_snapshot_path(entry.path)
            if not path.is_file() or path.is_symlink():
                raise SkillExportError(f"export verification found a missing or unsafe file: {entry.path}")
            actual = path.read_text(encoding="utf-8")
            if actual != snapshot.file_contents[entry.path]:
                raise SkillExportError(f"export verification failed for file: {entry.path}")


__all__ = ["SkillInstaller"]
