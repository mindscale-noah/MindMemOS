"""Evolution state store: full-version Qdrant storage plus file mirror."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import models as qmodels

from ...logging import get_logger
from ...typing import (
    EvolutionResult,
    EvolutionState,
    EvolutionTrigger,
    ParameterChange,
)
from .collections import EvolutionStateRepository
from .filters import match_value
from .registry import resolve_database_clients

logger = get_logger(__name__)


def _point_id(project_id: str, version: int) -> str:
    """Deterministic Qdrant point id for one evolution version.

    Qdrant requires UUID (or unsigned integer) point ids, so the stable
    ``{project_id}:v{version}`` key is mapped through uuid5. The same mapping
    is used for writes and reads.
    """

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{project_id}:v{version}"))


class EvolutionStateStore:
    """Manage versioned evolution state for one project.

    Qdrant ``evolution_state_v1`` is authoritative: every version is one point
    keyed by a deterministic UUID derived from ``{project_id}:v{version}`` with
    an ``is_current`` marker. The ``config/evolved/{project_id}/`` directory
    mirrors all versions as JSON files for git diff / audit; mirror failures
    are non-fatal.
    """

    def __init__(
        self,
        *,
        repo: EvolutionStateRepository | None = None,
        file_history_dir: str | Path = "config/evolved",
    ) -> None:
        self._repo = repo
        self._file_history_dir = Path(file_history_dir)

    def _repo_impl(self) -> EvolutionStateRepository:
        if self._repo is None:
            self._repo = resolve_database_clients().qdrant.evolution_state
        return self._repo

    async def get_current(self, project_id: str) -> EvolutionState | None:
        """Return the current (``is_current=true``) state for a project."""

        filter_ = qmodels.Filter(must=[match_value("is_current", True)])
        records, _ = await self._repo_impl().scroll(
            project_id,
            filter_=filter_,
            limit=1,
        )
        if not records:
            return None
        return EvolutionState.model_validate(records[0].payload)

    async def get_version(self, project_id: str, version: int) -> EvolutionState | None:
        """Return one specific version for a project."""

        record = await self._repo_impl().get(project_id, _point_id(project_id, version))
        if record is None:
            return None
        return EvolutionState.model_validate(record.payload)

    async def apply(
        self,
        project_id: str,
        *,
        add_config: dict[str, Any],
        search_config: dict[str, Any],
        changes: list[ParameterChange],
        trigger: EvolutionTrigger | None = None,
        rollback_version: int | None = None,
    ) -> EvolutionResult:
        """Create a new evolution version and mark it current."""

        current = await self.get_current(project_id)
        version = (current.version + 1) if current is not None else 1
        state = EvolutionState(
            project_id=project_id,
            version=version,
            is_current=True,
            add_config=add_config,
            search_config=search_config,
            trigger=trigger,
            changes=changes,
            rollback_version=rollback_version,
        )
        repo = self._repo_impl()
        if current is not None:
            previous = current.model_copy(update={"is_current": False})
            await repo.upsert(
                _point_id(project_id, previous.version),
                previous.model_dump(mode="json"),
            )
        await repo.upsert(_point_id(project_id, version), state.model_dump(mode="json"))
        await self._mirror(state)
        return EvolutionResult(
            project_id=project_id,
            version=state.version,
            changes=changes,
        )

    async def rollback(self, project_id: str, version: int) -> EvolutionResult:
        """Roll back to a previous version by flipping ``is_current`` markers."""

        target = await self.get_version(project_id, version)
        if target is None:
            raise ValueError(f"evolution version {version} not found for project {project_id}")
        repo = self._repo_impl()
        current = await self.get_current(project_id)
        if current is not None and current.version != version:
            previous = current.model_copy(update={"is_current": False})
            await repo.upsert(
                _point_id(project_id, previous.version),
                previous.model_dump(mode="json"),
            )
        rolled = target.model_copy(update={"is_current": True})
        await repo.upsert(_point_id(project_id, version), rolled.model_dump(mode="json"))
        await self._mirror_current(rolled)
        return EvolutionResult(
            project_id=project_id,
            version=version,
            is_rollback=True,
            changes=target.changes,
        )

    async def list_versions(self, project_id: str, *, limit: int = 100) -> list[EvolutionState]:
        """Return all stored versions for a project (oldest first)."""

        records, _ = await self._repo_impl().scroll(
            project_id,
            limit=limit,
            order_by="version",
        )
        return [EvolutionState.model_validate(r.payload) for r in records]

    async def _mirror(self, state: EvolutionState) -> None:
        """Append a version file and update ``current.json`` (best-effort)."""

        try:
            history_dir = self._file_history_dir / state.project_id / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            version_file = history_dir / f"{state.version:03d}_v{state.version}.json"
            version_file.write_text(
                json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            await self._mirror_current(state)
        except OSError as exc:
            logger.warning(
                "evolution_state_mirror_failed",
                project_id=state.project_id,
                version=state.version,
                error=str(exc),
            )

    async def _mirror_current(self, state: EvolutionState) -> None:
        """Write ``current.json`` pointing at the current version (best-effort)."""

        try:
            project_dir = self._file_history_dir / state.project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "current.json").write_text(
                json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "evolution_state_current_mirror_failed",
                project_id=state.project_id,
                version=state.version,
                error=str(exc),
            )
