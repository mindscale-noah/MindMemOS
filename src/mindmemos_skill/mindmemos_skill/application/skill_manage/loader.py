"""Authorized Skill version resolution for ``skill_manage`` builds.

The service never trusts a payload's self-reported ``project_id`` to authorize
access; it resolves exact ``skill_id + version_id`` refs through this port. The
local single-tenant adapter reads canonical versions from ``SkillRepository``;
an HTTP/multi-tenant adapter would load them from an authenticated context
instead.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...typing.skill_manage import SkillDocumentInput, SkillVersionRef


@runtime_checkable
class SkillVersionLoader(Protocol):
    """Resolve one exact immutable version ref into its canonical document."""

    async def load(self, ref: SkillVersionRef, *, project_id: str) -> SkillDocumentInput:
        """Return the parsed Skill document for the exact ref, or raise a typed error."""
        ...


class LocalSkillVersionLoader:
    """Single-tenant adapter backed by :class:`SkillRepository`."""

    def __init__(self, repository) -> None:
        self._repository = repository

    async def load(self, ref: SkillVersionRef, *, project_id: str) -> SkillDocumentInput:
        from ...contracts import SkillBundle
        from ...errors import SkillNotFoundError
        from ...management.repository import SkillRepository
        from ...typing import Skill as SkillModel

        assert isinstance(self._repository, SkillRepository)
        # Only the repository's own "no such version" becomes not-found; a
        # storage failure must stay a storage failure so the caller retries
        # instead of treating the version as missing.
        try:
            record = await self._repository.get_version(ref.version_id)
        except SkillNotFoundError as exc:
            raise SkillVersionNotFound(ref.skill_id, ref.version_id) from exc
        if record.skill_id != ref.skill_id:
            raise SkillVersionNotFound(ref.skill_id, ref.version_id)
        # Prefer the raw bundle content (preserves the frontmatter delimiters)
        # over skill.blob, which strips the leading '---' block.
        content = ""
        try:
            bundle = SkillBundle.model_validate_json(record.bundle)
            for file in bundle.files:
                if file.path == "SKILL.md":
                    content = file.content
                    break
        except Exception:
            content = ""
        if not content:
            skill = SkillModel.from_record(record)
            content = skill.blob.get("SKILL.md", "")
        if not content:
            raise SkillVersionNotFound(ref.skill_id, ref.version_id)
        return SkillDocumentInput(
            skill_id=ref.skill_id,
            version_id=ref.version_id,
            content=content,
            source_path=record.bundle and f"{record.skill_id}",
        )


class SkillVersionNotFound(Exception):
    """Typed not-found error for an unresolvable exact version ref."""

    def __init__(self, skill_id: str, version_id: str) -> None:
        super().__init__(f"Skill version not found: {skill_id}@{version_id}")
        self.skill_id = skill_id
        self.version_id = version_id


__all__ = ["LocalSkillVersionLoader", "SkillVersionLoader", "SkillVersionNotFound"]
