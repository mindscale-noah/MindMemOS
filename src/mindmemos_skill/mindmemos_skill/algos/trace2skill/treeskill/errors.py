"""TreeSkill stage failures that must invalidate an experiment run."""

from __future__ import annotations


class TreeSkillModelRequestError(RuntimeError):
    """Raised when a required TreeSkill model request cannot be completed."""

    def __init__(self, *, stage: str, item_id: str, cause: Exception) -> None:
        self.stage = stage
        self.item_id = item_id
        self.cause_type = type(cause).__name__
        super().__init__(f"TreeSkill {stage} model request failed for {item_id}: {self.cause_type}: {cause}")


__all__ = ["TreeSkillModelRequestError"]
