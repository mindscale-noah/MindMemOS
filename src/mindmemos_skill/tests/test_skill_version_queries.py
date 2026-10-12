"""Verify active selection, hash ambiguity, and stable repository queries."""

from datetime import timedelta

import pytest
from mindmemos_skill.api.typing import (
    ActiveRequest,
    CommitRequest,
    DiffRequest,
    HistoryRequest,
    ListRequest,
    RegisterRequest,
    SetMetaRequest,
    ShowRequest,
    SkillHistoryCursor,
    SkillListCursor,
    SkillPageRequest,
    VersionResolveRequest,
)
from mindmemos_skill.common.time_utils import utcnow
from mindmemos_skill.errors import (
    SkillAmbiguousVersionError,
    SkillConflictError,
    SkillNotFoundError,
    SkillVersionNotFoundError,
)
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.constants import SKILL_VERSIONS_TABLE, SKILLS_TABLE
from pydantic import ValidationError

from .test_skill_content_persistence import plain, service  # noqa: F401


@pytest.mark.asyncio
async def test_history_includes_commit_messages(service):
    """History preserves stored multiline messages and absent messages."""
    first = await service.register_skill(RegisterRequest(project_id="p", skill=plain("first")))
    message = "Validate order identifiers.\nPreserve the original evidence."
    await service.commit_skill(
        CommitRequest(project_id="p", skill_id=first.skill_id, skill=plain("second", commit_message=message))
    )
    history = await service.skill_history(HistoryRequest(project_id="p", skill_id=first.skill_id))
    assert [version.commit_message for version in history] == [None, message]


@pytest.mark.asyncio
async def test_active_selection_and_stale_activation(service):
    first = await service.register_skill(RegisterRequest(project_id="p", skill=plain("first")))
    second = await service.commit_skill(
        CommitRequest(
            project_id="p",
            skill_id=first.skill_id,
            skill=plain("second"),
            active=False,
        )
    )
    assert second.skill.active_version_id == first.skill_version_id
    activated = await service.active_skill(
        ActiveRequest(
            project_id="p",
            version_id=second.version.version_id,
            expected_active_version_id=first.skill_version_id,
        )
    )
    assert activated.active_version_id == second.version.version_id
    assert "current_version_id" not in activated.model_dump()
    shown = await service.show_skill(ShowRequest(project_id="p", skill_id=first.skill_id))
    assert shown.version.content == "second"
    with pytest.raises(SkillConflictError):
        await service.active_skill(
            ActiveRequest(
                project_id="p",
                version_id=first.skill_version_id,
                expected_active_version_id=first.skill_version_id,
            )
        )
    with pytest.raises(SkillNotFoundError) as failure:
        await service.active_skill(ActiveRequest(project_id="other", version_id=first.skill_version_id))
    assert "not found" in str(failure.value).lower()
    history = await service.skill_history(HistoryRequest(project_id="p", skill_id=first.skill_id))
    assert len(history) == 2
    await service.active_skill(ActiveRequest(project_id="p", version_id=first.skill_version_id))
    diff = await service.diff_skill(DiffRequest(project_id="p", to_version_id=second.version.version_id))
    assert diff.from_version_id == first.skill_version_id
    assert diff.to_version_id == second.version.version_id
    assert diff.files


@pytest.mark.asyncio
async def test_hash_resolution_requires_unique_version(service):
    first = await service.register_skill(RegisterRequest(project_id="p", skill=plain(), alias="example"))
    second = await service.commit_skill(
        CommitRequest(
            project_id="p",
            skill_id=first.skill_id,
            skill=plain(resources={"file": "YQ=="}),
            active=False,
        )
    )
    with pytest.raises(SkillAmbiguousVersionError):
        await service.resolve_skill_version(VersionResolveRequest(project_id="p", content_hash=first.content_hash))
    resolved = await service.resolve_skill_version(VersionResolveRequest(project_id="p", skill_hash=first.skill_hash))
    assert resolved.is_active and resolved.alias == "example"
    assert resolved.version == "v1.0.0" and resolved.commit_message is None
    assert resolved.skill_version_id == first.skill_version_id
    other = await service.resolve_skill_version(
        VersionResolveRequest(
            project_id="p",
            skill_version_id=second.version.version_id,
        )
    )
    assert not other.is_active and other.skill_hash == second.version.skill_hash
    with pytest.raises(SkillVersionNotFoundError):
        await service.resolve_skill_version(
            VersionResolveRequest(
                project_id="p",
                skill_version_id=second.version.version_id,
                skill_hash=first.skill_hash,
            )
        )
    with pytest.raises(SkillVersionNotFoundError):
        await service.resolve_skill_version(VersionResolveRequest(project_id="other", skill_hash=first.skill_hash))
    await service.commit_skill(CommitRequest(project_id="p", skill_id=first.skill_id, skill=plain()))
    with pytest.raises(SkillAmbiguousVersionError):
        await service.resolve_skill_version(VersionResolveRequest(project_id="p", skill_hash=first.skill_hash))
    exact = await service.resolve_skill_version(
        VersionResolveRequest(
            project_id="p",
            skill_version_id=first.skill_version_id,
            content_hash=first.content_hash,
        )
    )
    assert not exact.is_active


@pytest.mark.asyncio
async def test_stable_history_with_ties_and_new_insert(service):
    first = await service.register_skill(RegisterRequest(project_id="p", skill=plain()))
    for i in range(4):
        await service.commit_skill(CommitRequest(project_id="p", skill_id=first.skill_id, skill=plain(str(i))))
    cutoff = utcnow()
    tied = cutoff - timedelta(seconds=1)
    with service._persistence.session(Scope(fields={"project_id": "p"})) as session:
        (
            await session.database.update(
                SKILL_VERSIONS_TABLE, {"created_at": tied}, filters={"skill_id": first.skill_id}
            )
        )
    initial = await service.skill_history(
        HistoryRequest(
            project_id="p",
            skill_id=first.skill_id,
            limit=2,
            cursor=SkillHistoryCursor(snapshot_at=cutoff),
        )
    )
    await service.commit_skill(CommitRequest(project_id="p", skill_id=first.skill_id, skill=plain("later")))
    remaining = await service.skill_history(
        HistoryRequest(
            project_id="p",
            skill_id=first.skill_id,
            cursor=SkillHistoryCursor(
                snapshot_at=cutoff, after_created_at=initial[-1].created_at, after_version_id=initial[-1].version_id
            ),
        )
    )
    ids = [row.version_id for row in initial + remaining]
    assert len(ids) == len(set(ids)) == 5
    assert ids == sorted(ids)
    assert len(await service.skill_history(HistoryRequest(project_id="p", skill_id=first.skill_id))) == 6
    with service._persistence.session(Scope(fields={"project_id": "p"})) as session:
        family = await session.skills.resolve_family("p", first.skill_id)
        projected = await session.skills.history(family, limit=None, fields=["version_id"])
        assert len(projected) == 6 and set(projected[0]) == {"version_id"}


@pytest.mark.asyncio
async def test_family_cursor_and_metadata(service):
    families = []
    for name in ("A", "B", "C", "D"):
        families.append(await service.register_skill(RegisterRequest(project_id="p", skill=plain(name))))
    cutoff = utcnow()
    with service._persistence.session(Scope(fields={"project_id": "p"})) as session:
        (
            await session.database.update(
                SKILLS_TABLE, {"created_at": cutoff - timedelta(seconds=1)}, filters={"project_id": "p"}
            )
        )
    first = await service.list_skill_page(SkillPageRequest(project_id="p", snapshot_at=cutoff, limit=2))
    assert first.has_more
    await service.register_skill(RegisterRequest(project_id="p", skill=plain("later")))
    last = first.items[-1]
    second = await service.list_skill_page(
        SkillPageRequest(
            project_id="p",
            snapshot_at=cutoff,
            after_created_at=last.created_at,
            after_skill_id=last.skill_id,
        )
    )
    assert not second.has_more
    ids = [row.skill_id for row in first.items + second.items]
    assert ids == sorted([row.skill_id for row in families], reverse=True)
    assert len(await service.list_skills(ListRequest(project_id="p"))) == 5
    changed = await service.set_skill_meta(
        SetMetaRequest(project_id="p", skill_id=families[0].skill_id, name="Renamed")
    )
    assert changed.name == "Renamed"


@pytest.mark.parametrize(
    "values",
    [
        {"after_created_at": utcnow()},
        {"after_skill_id": "sk-a"},
        {"after_created_at": utcnow(), "after_skill_id": "sk-a"},
        {"snapshot_at": utcnow(), "offset": 1},
    ],
)
def test_invalid_list_cursor(values):
    with pytest.raises(ValidationError):
        SkillListCursor(**values)


def test_lookup_and_history_require_selectors():
    with pytest.raises(ValidationError):
        VersionResolveRequest(project_id="p")
    with pytest.raises(ValidationError):
        HistoryRequest(project_id="p")
    with pytest.raises(ValidationError):
        SkillHistoryCursor(after_version_id="skv-a")


@pytest.mark.asyncio
async def test_unlimited_queries_and_hash_search_past_deleted_rows(service):
    removed = await service.register_skill(RegisterRequest(project_id="p", skill=plain()))
    template = await service.show_skill(ShowRequest(project_id="p", skill_id=removed.skill_id))
    with service._persistence.session(Scope(fields={"project_id": "p"})) as session:
        copies = [
            template.version.model_copy(update={"version_id": f"skv-bulk-{i:04d}", "version_label": f"v2.0.{i}"})
            for i in range(1001)
        ]
        (await session.database.insert(SKILL_VERSIONS_TABLE, [copy.model_dump() for copy in copies]))
    history = await service.skill_history(HistoryRequest(project_id="p", skill_id=removed.skill_id))
    assert len(history) == 1002
    assert len(await service.list_skill_versions(removed.skill_id, "p")) == 1002
    with service._persistence.session(Scope(fields={"project_id": "p"})) as session:
        (await session.skills.delete(template.skill))
    visible = await service.register_skill(RegisterRequest(project_id="p", skill=plain()))
    resolved = await service.resolve_skill_version(VersionResolveRequest(project_id="p", skill_hash=visible.skill_hash))
    assert resolved.skill_version_id == visible.skill_version_id
    with service._persistence.session(Scope(fields={"project_id": "p"})) as session:
        family = await session.skills.resolve_family("p", visible.skill_id)
        families = []
        versions = []
        version = await session.skills.version(family)
        for i in range(1001):
            family_id, version_id = f"sk-bulk-{i:04d}", f"skv-family-{i:04d}"
            families.append(
                family.model_copy(update={"skill_id": family_id, "active_version_id": version_id}).model_dump()
            )
            versions.append(version.model_copy(update={"skill_id": family_id, "version_id": version_id}).model_dump())
        async with session.database.transaction() as tx:
            (await tx.insert(SKILL_VERSIONS_TABLE, versions))
            (await tx.insert(SKILLS_TABLE, families))
    assert len(await service.list_skills(ListRequest(project_id="p"))) == 1002
    page = await service.list_skill_page(SkillPageRequest(project_id="p", snapshot_at=utcnow()))
    assert len(page.items) == 1002 and not page.has_more


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", range(5))
async def test_concurrent_activation_compare_and_set(service, attempt):
    import asyncio

    first = await service.register_skill(RegisterRequest(project_id="p", skill=plain()))
    candidates = []
    for content in ("second", "third"):
        candidates.append(
            await service.commit_skill(
                CommitRequest(
                    project_id="p",
                    skill_id=first.skill_id,
                    skill=plain(content),
                    active=False,
                )
            )
        )
    results = await asyncio.gather(
        *[
            service.active_skill(
                ActiveRequest(
                    project_id="p",
                    version_id=candidate.version.version_id,
                    expected_active_version_id=first.skill_version_id,
                )
            )
            for candidate in candidates
        ],
        return_exceptions=True,
    )
    assert sum(isinstance(result, SkillConflictError) for result in results) == 1, results
    winners = [result for result in results if not isinstance(result, Exception)]
    assert len(winners) == 1
    detail = await service.show_skill(ShowRequest(project_id="p", skill_id=first.skill_id))
    assert detail.skill.active_version_id == winners[0].active_version_id
    assert len(await service.skill_history(HistoryRequest(project_id="p", skill_id=first.skill_id))) == 3
