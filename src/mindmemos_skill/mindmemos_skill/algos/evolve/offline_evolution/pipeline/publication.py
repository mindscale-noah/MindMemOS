"""Publish complete evolution outputs through existing repository transactions."""

from pathlib import Path

from .....api.typing import EvolutionPublishedSkill, EvolutionRunResult
from .....common.checkpoint import boundary
from .....errors import SkillConflictError
from .....persistence import SkillVersionRecord
from .....persistence.constants import SKILL_VERSIONS_TABLE
from .....persistence.db_services.skills.typing import VersionNaming
from .....persistence.store import SkillPersistence
from ..typing import EvolutionSnapshot
from .typing import CapturedEvolutionInputs, EvolutionRunManifest


def read_publication(directory: Path, manifest: EvolutionRunManifest) -> EvolutionRunResult | None:
    """Read an input-bound published result from this execution directory.

    Args:
        directory: Project-checked execution directory.
        manifest: Original request and execution identity.

    Returns:
        Published result, or None before the transaction completes.
    """
    saved = boundary(directory / "publication.json", manifest.input_hash)
    if saved is None:
        return None
    result = EvolutionRunResult.model_validate(saved)
    if result.run_id != manifest.run_id:
        raise SkillConflictError("Evolution publication does not match its execution")
    return result


async def publish_candidate(
    persistence: SkillPersistence,
    directory: Path,
    manifest: EvolutionRunManifest,
    captured: CapturedEvolutionInputs,
    candidate: EvolutionSnapshot,
) -> EvolutionRunResult:
    """Apply complete candidates in the existing persistence transaction.

    Args:
        persistence: Scoped repository with its original transaction and Skill APIs.
        directory: Project-checked execution directory, locked by the caller.
        manifest: Original immutable execution identity.
        captured: Owners and exact source versions captured before analysis.
        candidate: Complete input and newly created candidate snapshots.

    Returns:
        Persisted member versions, including after a committed transaction was interrupted.
    """
    if candidate.skills.keys() != captured.inputs.keys():
        raise ValueError("Evolution must retain every input Skill")
    pending = directory / "publication.pending.json"
    identity = {"input": manifest.input_hash, "candidate": candidate.model_dump(mode="json")}
    saved = boundary(pending, identity)
    if saved is not None:
        result = EvolutionRunResult.model_validate(saved["result"])
        committed = []
        for member in result.members:
            if member.action in {"updated", "created"}:
                row = await persistence.database.get(SKILL_VERSIONS_TABLE, member.version_id)
                if row is not None and (
                    row["skill_id"] != member.skill_id
                    or SkillVersionRecord.model_validate(row).snapshot() != member.snapshot
                ):
                    raise SkillConflictError("A published version differs from its prepared result")
                committed.append(row is not None)
        if all(committed):
            boundary(directory / "publication.json", manifest.input_hash, payload=result.model_dump(mode="json"))
            return result
        if any(committed):
            raise SkillConflictError("Prepared publication records are inconsistent")
        pending.unlink()

    members = []
    naming = VersionNaming(algorithm=manifest.algorithm.replace("_", "-"))
    async with persistence.transaction() as transaction:
        for key, item in sorted(captured.inputs.items()):
            owner, base = item.owner, item.base
            snapshot = candidate.skills[key]
            await transaction.skills.resolve_family(key)
            if snapshot == base.snapshot():
                members.append(
                    EvolutionPublishedSkill(
                        key=key,
                        skill_id=key,
                        action="unchanged",
                        base_version_id=base.version_id,
                        version_id=base.version_id,
                        snapshot=snapshot,
                    )
                )
            else:
                version_id = transaction.skills.new_version_id()
                await transaction.skills.commit(
                    owner,
                    snapshot,
                    message=f"Evolution {manifest.run_id}",
                    naming=naming,
                    base_version_id=base.version_id,
                    active=manifest.request.auto_accept,
                    commit_source=manifest.algorithm,
                    version_id=version_id,
                )
                await transaction.skills.get(version_id)
                members.append(
                    EvolutionPublishedSkill(
                        key=key,
                        skill_id=key,
                        action="updated",
                        base_version_id=base.version_id,
                        version_id=version_id,
                        snapshot=snapshot,
                    )
                )
        for key, snapshot in sorted(candidate.created_skills.items()):
            owner = await transaction.skills.register(
                name=snapshot.name,
                alias=None,
                source=manifest.algorithm,
                message=f"Evolution {manifest.run_id}",
                snapshot=snapshot,
                naming=naming,
            )
            members.append(
                EvolutionPublishedSkill(
                    key=key,
                    skill_id=owner.skill_id,
                    action="created",
                    version_id=owner.active_version_id,
                    snapshot=snapshot,
                )
            )
        result = EvolutionRunResult(
            run_id=manifest.run_id,
            input_version_ids={key: item.base.version_id for key, item in captured.inputs.items()},
            **candidate.model_dump(),
            members=members,
        )
        boundary(pending, identity, payload={"result": result.model_dump(mode="json")})
    boundary(directory / "publication.json", manifest.input_hash, payload=result.model_dump(mode="json"))
    return result
