"""SkillVersionLoader port: authorized exact-version resolution (P4 prereq)."""

from __future__ import annotations

from pathlib import Path

import pytest
from mindmemos_skill.application.skill_manage.loader import LocalSkillVersionLoader, SkillVersionNotFound
from mindmemos_skill.management import RegisterSkillRequest
from mindmemos_skill.typing import SkillVersionRef

from mindmemos_skill import SkillApplication


def _config(tmp_path: Path) -> dict:
    return {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        'name: demo\ndescription: Demo description\nversion: "1.0.0"\n\nOriginal\n',
        encoding="utf-8",
    )
    return source


@pytest.mark.asyncio
async def test_local_loader_resolves_exact_version_content(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    try:
        registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))
        loader = LocalSkillVersionLoader(application._manager.repository)

        document = await loader.load(
            SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id),
            project_id="p1",
        )

        assert document.skill_id == registered.skill_id
        assert document.version_id == registered.version_id
        assert "Demo description" in document.content
    finally:
        await application.close()


@pytest.mark.asyncio
async def test_local_loader_rejects_unknown_version(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    try:
        loader = LocalSkillVersionLoader(application._manager.repository)
        with pytest.raises(SkillVersionNotFound):
            await loader.load(SkillVersionRef(skill_id="s", version_id="missing"), project_id="p1")
    finally:
        await application.close()


@pytest.mark.asyncio
async def test_local_loader_rejects_skill_version_mismatch(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    try:
        registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))
        loader = LocalSkillVersionLoader(application._manager.repository)
        with pytest.raises(SkillVersionNotFound):
            await loader.load(
                SkillVersionRef(skill_id="wrong-skill", version_id=registered.version_id),
                project_id="p1",
            )
    finally:
        await application.close()


@pytest.mark.asyncio
async def test_local_loader_keeps_a_repository_failure_distinct_from_not_found(tmp_path: Path) -> None:
    """A storage blip must not be reported as a missing version, which the caller would never retry."""

    application = await SkillApplication.from_config(_config(tmp_path))
    try:
        registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))
        repository = application._manager.repository
        loader = LocalSkillVersionLoader(repository)
        original = repository.get_version

        async def get_version_unavailable(version_id: str):
            raise ConnectionResetError("repository unavailable")

        repository.get_version = get_version_unavailable  # type: ignore[method-assign]
        try:
            with pytest.raises(ConnectionResetError):
                await loader.load(
                    SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id), project_id="p1"
                )
        finally:
            repository.get_version = original  # type: ignore[method-assign]
        document = await loader.load(
            SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id), project_id="p1"
        )
        assert document.version_id == registered.version_id
    finally:
        await application.close()
