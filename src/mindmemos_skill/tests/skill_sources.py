"""Source factories for repository integration tests."""

from mindmemos_skill.api.typing import RegisterRequest, ShowRequest, StructuredSkillInfo
from mindmemos_skill.skill import SkillSnapshot


def structured(snapshot: SkillSnapshot, **options) -> StructuredSkillInfo:
    """Build structured input from a validated snapshot.

    Args:
        snapshot: Complete source data used by a test.
        options: Explicit metadata or commit options for this request.

    Returns:
        Repository input with canonical encoded resources.
    """
    return StructuredSkillInfo(
        **(snapshot.model_dump(include={"content", "resources", "name", "description", "standard_format"}) | options)
    )


async def register_detail(service, request: RegisterRequest):
    """Register source and fetch its exact persisted version for assertions.

    Args:
        service: Initialized and authorized test service.
        request: Validated registration input.

    Returns:
        Stored family and immutable initial version.
    """
    result = await service.register_skill(request)
    return await service.show_skill(
        ShowRequest(project_id=request.project_id, skill_id=result.skill_id, version_id=result.skill_version_id)
    )
