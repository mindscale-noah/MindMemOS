"""Convert persisted Memory Add evidence into the shared legacy trajectory format."""

import hashlib
import json

from pydantic import JsonValue

from ..persistence.constants import LEGACY_SKILL_SOURCE
from ..persistence.db_services.skills.typing import SkillVersionRecord
from ..persistence.db_services.trajectories.typing import TrajectoryRecord
from .typing import LegacyBinding, LegacyTrajectory


def build_legacy_binding(binding: LegacyBinding, version: SkillVersionRecord) -> dict[str, JsonValue]:
    """Bind legacy usage to a validated exact destination version.

    Args:
        binding: Old stored usage and content digest.
        version: Version imported from the same legacy namespace.
    """
    original = version.local_metadata.get(LEGACY_SKILL_SOURCE)
    if binding.content_hash and (
        not isinstance(original, dict) or binding.content_hash != original.get("content_hash")
    ):
        raise ValueError("Legacy binding content hash does not match the imported version")
    return {
        "skill_id": version.skill_id,
        "version_id": version.version_id,
        "name": version.name,
        "content_hash": version.content_hash,
        "usage": binding.usage,
    }


def _valid_message(message: dict[str, JsonValue]) -> bool:
    """Recognize the original dialogue and standalone Add attachment shapes."""
    if "role" in message or "content" in message:
        role = message.get("role")
        return isinstance(role, str) and bool(role.strip()) and isinstance(message.get("content"), str)
    return (
        isinstance(message.get("text"), str)
        or isinstance(message.get("url"), str)
        or (isinstance(message.get("file_name"), str) and isinstance(message.get("file_path"), str))
    )


def build_legacy_trajectory(
    project_id: str, add_record_id: str, legacy: LegacyTrajectory, bindings: list[dict[str, JsonValue]]
) -> TrajectoryRecord:
    """Construct the same stable trajectory for live delivery and historical import.

    Args:
        project_id: Trusted original namespace.
        add_record_id: Original persisted point ID.
        legacy: Dialogue and original request metadata.
        bindings: Validated destination version bindings.
    """
    received_at = legacy.request_submitted_at or legacy.event_time or legacy.task_completed_at
    if received_at is None:
        raise ValueError("Add record has no historical timestamp")
    if not all(_valid_message(message) for message in legacy.messages):
        raise ValueError("Add record contains an unsupported message shape")
    encoded = json.dumps(legacy.messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return TrajectoryRecord(
        project_id=project_id,
        trajectory_id=add_record_id,
        trajectory_hash=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        source="reported",
        rollout_id=f"{LEGACY_SKILL_SOURCE}:{add_record_id}",
        task_id=legacy.task_id or add_record_id,
        task_instruction=next((m["content"] for m in legacy.messages if m.get("role") == "user"), ""),
        task_system_prompt=next((m["content"] for m in legacy.messages if m.get("role") == "system"), None),
        status="unknown",
        env_ref="unknown",
        agent_type="harness",
        skill_bindings=bindings,
        trajectory=legacy.messages,
        reward_score=legacy.score,
        n_turn=sum(m.get("role") == "assistant" for m in legacy.messages),
        started_at=None,
        created_at=received_at,
        received_at=received_at,
        user_id=legacy.user_id or None,
        agent_id=legacy.agent_id or None,
        session_id=legacy.session_id or None,
        metadata={
            LEGACY_SKILL_SOURCE: {
                "add_status": legacy.status,
                "memory_completed_at": legacy.task_completed_at.isoformat() if legacy.task_completed_at else None,
                "event_time": legacy.event_time.isoformat() if legacy.event_time else None,
            }
        },
    )
