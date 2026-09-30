"""Atomic, fingerprinted recovery boundaries for a single forest run."""

from pathlib import Path

from ....common.json import append_jsonl
from ....skill import Skill
from ....skill.typing import SkillSnapshot


def load_skill(payload) -> Skill:
    """Restore a validated candidate snapshot.

    Args:
        payload: Serialized Skill snapshot from a committed boundary.

    Returns:
        Reconstructed Skill with its auxiliary files and resources.
    """
    return Skill.from_snapshot(SkillSnapshot.model_validate(payload))


def recovery_event(root: Path, event: str, **details) -> None:
    """Append recovery observations separately from physical-call accounting.

    Args:
        root: Run directory.
        event: Checkpoint or cache event name.
        **details: Non-secret identities and counts.
    """
    append_jsonl(root / "recovery.jsonl", {"event": event, **details})
