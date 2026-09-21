"""Restore runtime behavior explicitly for on-demand workbench assembly."""

from mindmemos_skill.skill import Skill, SkillSnapshot


def load_runtime_skill(snapshot: SkillSnapshot) -> Skill:
    """Restore a supported implementation without changing source metadata.

    Args:
        snapshot: Immutable source and algorithm-owned runtime configuration.

    Returns:
        A Skill implementing the common assemble(task) contract.

    Raises:
        NotImplementedError: The implementation is not installed in this workbench.
    """
    if snapshot.implementation == "static":
        return Skill.from_snapshot(snapshot)
    if snapshot.implementation == "treeskill":
        from mindmemos_skill.algos.train.treeskill import TreeSkill

        restored = snapshot.model_copy(deep=True)
        router = restored.config.get("router", {})
        # Snapshot credentials are masked; resolve them from the runtime environment.
        if router.get("api_key") == "**********":
            router.pop("api_key")
        return TreeSkill.from_snapshot(restored)
    raise NotImplementedError("This Skill implementation has no runtime loader")
