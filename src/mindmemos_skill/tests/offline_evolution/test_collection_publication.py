"""Verify atomic publication through shared persistence interfaces."""

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from mindmemos_skill.algos.evolve.offline_evolution.typing import EvolutionSnapshot
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
from mindmemos_skill.api.typing import EvolutionRequest
from mindmemos_skill.common.checkpoint import fingerprint
from mindmemos_skill.errors import SkillConflictError, SkillNotFoundError
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence import SkillPersistence
from mindmemos_skill.persistence.config import SkillPersistenceConfig
from mindmemos_skill.persistence.constants import SKILL_VERSIONS_TABLE, SKILLS_TABLE
from mindmemos_skill.persistence.db_services.skills.service import SkillDataService
from mindmemos_skill.pipelines.builtin_pipelines.offline_evolution import publication
from mindmemos_skill.pipelines.builtin_pipelines.offline_evolution.typing import (
    CapturedEvolutionInputs,
    EvolutionInputMember,
    EvolutionRunManifest,
)
from mindmemos_skill.skill import Skill, SkillContent


def skill(name, body="Original", resources=None):
    """Create a complete standard source with optional binary resources."""
    return Skill(
        SkillContent(
            content=f"---\nname: {name}\ndescription: Test {name} workflows.\n---\n\n{body}\n",
            resources=resources or {},
        )
    )


@pytest.fixture
async def repository(tmp_path):
    """Open an isolated real SQLite repository with all business tables."""
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'publication.db'}"))
    store = SkillPersistence(SkillPersistenceConfig(), database=database)
    await store.initialize_storage()
    try:
        with store.session(Scope(fields={"project_id": "project"})):
            yield store
    finally:
        await database.close()


async def capture(repository, name):
    """Register and capture one exact family and its original immutable source."""
    owner = await repository.skills.register(
        name=name, alias=name, source=None, message=None, snapshot=skill(name).snapshot()
    )
    return EvolutionInputMember(owner=owner, base=await repository.skills.get(owner.active_version_id))


@pytest.mark.parametrize("dynamic", [False, True])
async def test_repository_does_not_impose_offline_creation_formats(repository, dynamic):
    captured = await capture(repository, "context")
    created = (
        TreeSkill(TreeSkillConfig(router={"model": "offline"}), source=skill("new-tree").source)
        if dynamic
        else Skill(
            SkillContent(
                content="# New source\r\n", name="new-source", description="New workflow", standard_format=False
            )
        )
    ).snapshot()
    inputs = {captured.owner.skill_id: captured}
    candidate = EvolutionSnapshot(
        skills={captured.owner.skill_id: captured.base.snapshot()},
        created_skills={"new/source": created},
    )
    receipt = await publish(repository, inputs, candidate)
    member = next(item for item in receipt.members if item.action == "created")
    assert member.snapshot == created
    assert (await repository.skills.get(member.version_id)).snapshot() == created


async def publish(repository, inputs, candidate, run_id=None, *, auto_accept=True):
    """Publish through the original repositories and reuse the run directory on retry."""
    request = EvolutionRequest(
        project_id="project",
        auto_accept=auto_accept,
        skill_version_ids=[item.base.version_id for item in inputs.values()],
        run_id=run_id or "evolve_" + uuid4().hex,
    )
    manifest = EvolutionRunManifest(
        run_id=request.run_id,
        project_id="project",
        input_hash=fingerprint(request.model_dump(mode="json")),
        request=request,
        algorithm="offline_evolution",
        configuration={},
    )
    directory = Path(repository.local_database_path()).parent / "runs" / request.run_id
    return publication.read_publication(directory, manifest) or await publication.publish_candidate(
        repository, directory, manifest, CapturedEvolutionInputs(inputs=inputs), candidate
    )


async def test_update_create_are_atomic_and_replay_after_external_deletion(repository):
    first, second, readonly = [await capture(repository, name) for name in ("alpha", "beta", "readonly")]
    inputs = {item.owner.skill_id: item for item in (first, second, readonly)}
    changed = skill("alpha", "Improved", {"asset.bin": b"\x00\xff"}).snapshot()
    candidate = EvolutionSnapshot(
        skills={
            first.owner.skill_id: changed,
            second.owner.skill_id: second.base.snapshot(),
            readonly.owner.skill_id: readonly.base.snapshot(),
        },
        created_skills={"new/delta": skill("delta").snapshot()},
    )
    run_id = "evolve_" + uuid4().hex
    receipt = await publish(repository, inputs, candidate, run_id)
    assert [item.action for item in receipt.members].count("created") == 1
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 5
    assert (await repository.skills.resolve_family(first.owner.skill_id)).active_version_id != first.base.version_id
    assert (await repository.skills.resolve_family(second.owner.skill_id)).active_version_id == second.base.version_id
    await repository.skills.delete(second.owner)
    with pytest.raises(SkillNotFoundError):
        await repository.skills.resolve_family(second.owner.skill_id)
    assert await publish(repository, inputs, candidate, run_id) == receipt
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 5


async def test_missing_input_candidates_are_rejected_without_repository_changes(repository):
    member = await capture(repository, "alpha")
    key = member.owner.skill_id
    inputs = {key: member}
    candidate = EvolutionSnapshot(skills={})
    run_id = "evolve_" + uuid4().hex
    with pytest.raises(ValueError, match="retain every input Skill"):
        await publish(repository, inputs, candidate, run_id)
    assert (await repository.skills.resolve_family(key)).active_version_id == member.base.version_id
    assert await repository.database.count(SKILLS_TABLE) == 1
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 1


@pytest.mark.parametrize("auto_accept", [False, True])
async def test_transaction_failure_rolls_back_all_effects_and_can_resume(repository, monkeypatch, auto_accept):
    members = [await capture(repository, name) for name in ("alpha", "beta")]
    inputs = {item.owner.skill_id: item for item in members}
    candidate = EvolutionSnapshot(
        skills={key: skill(item.base.name, "Improved").snapshot() for key, item in inputs.items()},
        created_skills={"new/delta": skill("delta").snapshot()},
    )
    original = SkillDataService._insert_child
    calls = 0

    async def fail_second(self, tx, owner, version, naming, *, active=True):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted transaction")
        await original(self, tx, owner, version, naming, active=active)

    run_id = "evolve_" + uuid4().hex
    with monkeypatch.context() as patch:
        patch.setattr(SkillDataService, "_insert_child", fail_second)
        with pytest.raises(OSError, match="interrupted"):
            await publish(repository, inputs, candidate, run_id, auto_accept=auto_accept)
    assert await repository.database.count(SKILLS_TABLE) == 2
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 2
    for item in members:
        assert (await repository.skills.resolve_family(item.owner.skill_id)).active_version_id == item.base.version_id
    result = await publish(repository, inputs, candidate, run_id, auto_accept=auto_accept)
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 5
    outcomes = {member.skill_id: member for member in result.members}
    for item in members:
        active = (await repository.skills.resolve_family(item.owner.skill_id)).active_version_id
        assert active == (outcomes[item.owner.skill_id].version_id if auto_accept else item.base.version_id)


async def test_conflict_on_later_member_rolls_back_earlier_updates(repository):
    members = [await capture(repository, name) for name in ("alpha", "beta")]
    inputs = {item.owner.skill_id: item for item in members}
    candidate = EvolutionSnapshot(
        skills={key: skill(item.base.name, "Improved").snapshot() for key, item in inputs.items()},
    )
    later = inputs[sorted(inputs)[-1]]
    await repository.skills.commit(later.owner, skill(later.base.name, "Concurrent").snapshot(), message="other writer")
    with pytest.raises(SkillConflictError):
        await publish(repository, inputs, candidate)
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 3
    earlier = inputs[sorted(inputs)[0]]
    assert (await repository.skills.resolve_family(earlier.owner.skill_id)).active_version_id == earlier.base.version_id


async def test_creation_only_does_not_create_a_version_for_readonly_input(repository):
    member = await capture(repository, "alpha")
    key = member.owner.skill_id
    inputs = {key: member}
    candidate = EvolutionSnapshot(
        skills={key: member.base.snapshot()},
        created_skills={"new/delta": skill("delta").snapshot()},
    )
    receipt = await publish(repository, inputs, candidate)
    assert {item.action for item in receipt.members} == {"unchanged", "created"}
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 2
    assert (await repository.skills.resolve_family(key)).active_version_id == member.base.version_id


@pytest.mark.parametrize("operation", ["create_only", "mixed", "unchanged"])
@pytest.mark.parametrize("auto_accept", [False, True])
async def test_recovers_commit_before_result_file(repository, monkeypatch, operation, auto_accept):
    """Detect committed versions without replaying writes."""
    member = await capture(repository, "alpha")
    key = member.owner.skill_id
    candidate = EvolutionSnapshot(
        skills={key: skill("alpha", "Improved").snapshot() if operation == "mixed" else member.base.snapshot()},
        created_skills={"new/delta": skill("delta").snapshot()} if operation in {"create_only", "mixed"} else {},
    )
    run_id = "evolve_" + uuid4().hex
    original = publication.boundary

    def interrupted(path, identity, *, payload=None):
        if path.name == "publication.json" and payload is not None:
            raise OSError("result file interrupted")
        return original(path, identity, payload=payload)

    with monkeypatch.context() as patch:
        patch.setattr(publication, "boundary", interrupted)
        with pytest.raises(OSError, match="result file interrupted"):
            await publish(repository, {key: member}, candidate, run_id, auto_accept=auto_accept)
    before = await repository.database.count(SKILL_VERSIONS_TABLE)
    if operation == "mixed":
        current = await repository.skills.resolve_family(key)
        await repository.skills.commit(current, skill("alpha", "Later edit").snapshot(), message="later writer")
        before += 1
    result = await publish(repository, {key: member}, candidate, run_id, auto_accept=auto_accept)
    assert result.members
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == before
    assert await publish(repository, {key: member}, candidate, run_id, auto_accept=auto_accept) == result
    if operation == "mixed":
        current = await repository.skills.resolve_family(key)
        assert "Later edit" in (await repository.skills.get(current.active_version_id)).content


@pytest.mark.parametrize("auto_accept", [False, True])
async def test_prepared_journal_is_retried_after_database_rollback(repository, monkeypatch, auto_accept):
    """Discard IDs from an uncommitted journal and safely publish again."""
    member = await capture(repository, "alpha")
    inputs = {member.owner.skill_id: member}
    candidate = EvolutionSnapshot(skills={member.owner.skill_id: skill("alpha", "Improved").snapshot()})
    original = SkillPersistence.transaction

    @asynccontextmanager
    async def interrupted(self):
        async with original(self) as transaction:
            yield transaction
            raise OSError("commit interrupted")

    run_id = "evolve_" + uuid4().hex
    with monkeypatch.context() as patch:
        patch.setattr(SkillPersistence, "transaction", interrupted)
        with pytest.raises(OSError, match="commit interrupted"):
            await publish(repository, inputs, candidate, run_id, auto_accept=auto_accept)
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 1
    result = await publish(repository, inputs, candidate, run_id, auto_accept=auto_accept)
    assert result.members
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 2


async def test_inactive_publication_can_branch_from_a_historical_input(repository):
    """Save the requested output without replacing a concurrently selected version."""
    original = await capture(repository, "alpha")
    current = await repository.skills.commit(
        original.owner, skill("alpha", "Concurrent edit").snapshot(), message="another writer"
    )
    candidate = EvolutionSnapshot(skills={original.owner.skill_id: skill("alpha", "Improved").snapshot()})
    run_id = "evolve_" + uuid4().hex
    result = await publish(repository, {original.owner.skill_id: original}, candidate, run_id, auto_accept=False)
    member = result.members[0]
    assert member.version_id not in {original.base.version_id, current.active_version_id}
    saved = await repository.skills.get(member.version_id)
    assert saved.parent_version_ids == [original.base.version_id]
    assert saved.snapshot() == candidate.skills[original.owner.skill_id]
    assert (
        await repository.skills.resolve_family(original.owner.skill_id)
    ).active_version_id == current.active_version_id
    assert (
        await publish(repository, {original.owner.skill_id: original}, candidate, run_id, auto_accept=False) == result
    )
    assert await repository.database.count(SKILL_VERSIONS_TABLE) == 3
