"""Paired Before/After ablation rollout planning."""

from __future__ import annotations

from ..contracts import RolloutPhase, RolloutSpec
from .fixed_group import stable_rollout_id
from .strategy import PairedAblationPlan, RolloutPlan


class PairedAblationRolloutStrategy:
    """Plan one shared Before and one After per candidate/case/sample."""

    name = "paired_ablation"

    def plan(self, request: RolloutPlan) -> list[RolloutSpec]:
        if not isinstance(request, PairedAblationPlan):
            raise TypeError("paired_ablation requires PairedAblationPlan")
        specs: list[RolloutSpec] = []
        sequence = request.sequence_start
        physical_sample_index = request.sample_index_start
        pairs: dict[tuple[str, int], str] = {}

        # Define shared current-skill rollouts first so sequence numbers and
        # persisted outcome order remain stable even when Before/After execute
        # concurrently.
        for task in request.tasks:
            for repeat_index in range(request.samples_per_case):
                pair_id = stable_rollout_id(
                    request.run_id,
                    request.scope,
                    "ablation_pair",
                    task.task_id,
                    repeat_index,
                )
                pairs[(task.task_id, repeat_index)] = pair_id
                specs.append(
                    self._spec(
                        request,
                        sequence=sequence,
                        phase=RolloutPhase.ABLATION_BEFORE,
                        task=task,
                        skills=[request.before_skill],
                        sample_index=physical_sample_index,
                        pair_id=pair_id,
                        candidate_id=None,
                        seed=None,
                    )
                )
                sequence += 1
                physical_sample_index += 1

        # After rollouts retain candidate order and each candidate's sampled-case
        # order, matching asyncio.gather(score(candidate)...) in the source.
        tasks_by_id = {task.task_id: task for task in request.tasks}
        for target in request.targets:
            for task_id in target.task_ids:
                task = tasks_by_id[task_id]
                for repeat_index in range(request.samples_per_case):
                    pair_id = pairs[(task_id, repeat_index)]
                    specs.append(
                        self._spec(
                            request,
                            sequence=sequence,
                            phase=RolloutPhase.ABLATION_AFTER,
                            task=task,
                            skills=[target.skill],
                            sample_index=physical_sample_index,
                            pair_id=pair_id,
                            candidate_id=target.candidate_id,
                            seed=None,
                        )
                    )
                    sequence += 1
                    physical_sample_index += 1
        return specs

    @staticmethod
    def _spec(
        request: PairedAblationPlan,
        *,
        sequence: int,
        phase: RolloutPhase,
        task,
        skills,
        sample_index: int,
        pair_id: str,
        candidate_id: str | None,
        seed: int | None,
    ) -> RolloutSpec:
        rollout_id = stable_rollout_id(
            request.run_id,
            request.scope,
            phase.value,
            task.task_id,
            sample_index,
            candidate_id or "before",
        )
        return RolloutSpec(
            sequence_no=sequence,
            rollout_id=rollout_id,
            phase=phase,
            task=task,
            skills=skills,
            sample_index=sample_index,
            candidate_id=candidate_id,
            pair_id=pair_id,
            agent_ref=request.agent_ref,
            env_ref=request.env_ref,
            seed=seed,
            temperature=request.temperature,
            agent_options=request.agent_options,
            env_options=request.env_options,
        )


__all__ = ["PairedAblationRolloutStrategy"]
