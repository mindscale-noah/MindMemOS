"""Fixed samples-per-task rollout planning."""

from __future__ import annotations

import hashlib

from ..contracts import RolloutPhase, RolloutSpec
from .strategy import FixedGroupPlan, RolloutPlan


def stable_rollout_id(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class FixedGroupRolloutStrategy:
    name = "fixed_group"

    def plan(self, request: RolloutPlan) -> list[RolloutSpec]:
        if not isinstance(request, FixedGroupPlan):
            raise TypeError("fixed_group requires FixedGroupPlan")
        try:
            phase = RolloutPhase(request.phase)
        except ValueError as exc:
            raise ValueError(f"fixed_group does not support phase {request.phase!r}") from exc

        specs: list[RolloutSpec] = []
        sequence = request.sequence_start
        for task in request.tasks:
            for sample_index in range(request.group_size):
                rollout_id = stable_rollout_id(
                    request.run_id,
                    request.scope,
                    phase.value,
                    task.task_id,
                    sample_index,
                )
                specs.append(
                    RolloutSpec(
                        sequence_no=sequence,
                        rollout_id=rollout_id,
                        phase=phase,
                        task=task,
                        skills=request.skills,
                        sample_index=sample_index,
                        agent_ref=request.agent_ref,
                        env_ref=request.env_ref,
                        seed=request.seed + sequence,
                        temperature=request.temperature,
                        agent_options=request.agent_options,
                        env_options=request.env_options,
                    )
                )
                sequence += 1
        return specs


__all__ = ["FixedGroupRolloutStrategy", "stable_rollout_id"]
