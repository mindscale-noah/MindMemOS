"""Restore explicit Skill implementations and interrupted task-time assembly."""

from copy import copy

from ..skill import Skill, SkillSnapshot


def load_runtime_skill(snapshot: SkillSnapshot) -> Skill:
    """Restore source content with its recorded runtime behavior.

    Args:
        snapshot: Validated source files and implementation configuration.

    Returns:
        A concrete Skill with lazy runtime dependencies.

    Raises:
        NotImplementedError: No runtime loader exists for the implementation.
    """
    if snapshot.implementation == "static":
        return Skill.from_snapshot(snapshot)
    if snapshot.implementation == "treeskill":
        from ..algos.train.treeskill.treeskill import TreeSkill

        return TreeSkill.from_snapshot(snapshot)
    raise NotImplementedError(f"No runtime loader for Skill implementation {snapshot.implementation!r}")


def restore_assembled_skill(source: Skill, snapshot: SkillSnapshot) -> Skill:
    """Restore task-selected content while retaining the caller's live dependencies.

    Args:
        source: Original runtime Skill, whose router must not run again on resume.
        snapshot: Complete previously assembled text, resources, and metadata.

    Returns:
        A separate instance of the original concrete class.

    Raises:
        ValueError: The recorded runtime configuration differs from the source.
    """
    validated = SkillSnapshot.model_validate(snapshot.model_dump())
    original = source.snapshot()
    if (validated.implementation, validated.config_schema_version, validated.config) != (
        original.implementation,
        original.config_schema_version,
        original.config,
    ):
        raise ValueError("Assembled Skill runtime configuration differs from its source")
    candidate = copy(source)
    Skill.__init__(candidate, validated.to_content(), template=validated)
    return candidate
