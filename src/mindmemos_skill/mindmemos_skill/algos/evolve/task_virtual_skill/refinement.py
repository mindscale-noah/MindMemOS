"""Continue a completed decomposition run with dynamic retries and Skill-local merges."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ....llm import llm_run_context
from ....persistence.enums import SkillVersionOrigin, SkillVersionStatus
from ....skill_runtime.runtimes.virtual_components import VirtualComponent, VirtualComponentsMetadata
from ....typing import Skill, compute_skill_content_hash
from ..skill_grpo_with_replay_buffer.config import RetryConfig, RolloutConfig
from ..skill_grpo_with_replay_buffer.contracts import RolloutOutcome, RolloutSpec
from ..skill_grpo_with_replay_buffer.models import ChatModel
from ..skill_grpo_with_replay_buffer.rollout import AgentResolver, EnvFactory, RolloutScheduler
from ..skill_grpo_with_replay_buffer.rollout.fixed_group import stable_rollout_id
from ..skill_grpo_without_replay_buffer.reflection import (
    REFLECTION_PROMPT_VERSION,
    ReflectionGenerator,
    previous_answer,
    task_with_reflection,
)
from .config import TaskVirtualSkillRunConfig
from .models import TaskVirtualSkillResult, TrajectoryKeyPoints
from .refinement_models import TaskSkillChange, TaskVirtualSkillRefinementResult, VirtualSkillMerge
from .refinement_prompts import CHANGE_SYSTEM, MERGE_SYSTEM, change_user, merge_user
from .summarizer import TrajectoryKeyPointSummarizer


class TaskVirtualSkillRefiner:
    def __init__(
        self,
        *,
        config: TaskVirtualSkillRunConfig,
        reflection_model: ChatModel,
        summary_model: ChatModel,
        change_model: ChatModel,
        merge_model: ChatModel,
        agent_resolver: AgentResolver,
        env_factory: EnvFactory,
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._reflection_model = reflection_model
        self._summary_model = summary_model
        self._change_model = change_model
        self._merge_model = merge_model
        self._agent_resolver = agent_resolver
        self._env_factory = env_factory
        self._chat_kwargs = dict(chat_kwargs or {})

    async def refine(
        self,
        *,
        run_id: str,
        source: TaskVirtualSkillResult,
    ) -> TaskVirtualSkillRefinementResult:
        if not source.changed or source.candidate is None:
            raise ValueError("source run did not produce virtual Skills to refine")
        with llm_run_context(run_id):
            retry_outcomes = await self._retry_failures(run_id=run_id, source=source)
            summarizer = TrajectoryKeyPointSummarizer(
                chat_model=_ChatWithDefaults(self._summary_model, self._chat_kwargs),
                concurrency=self._config.summary.max_concurrent_summaries,
                transcript_max_chars=self._config.summary.transcript_max_chars,
            )
            summarized_ids = {item.trajectory_id for item in source.trajectory_summaries}
            outcomes_to_summarize = [
                item
                for item in [*source.rollouts, *retry_outcomes]
                if item.trajectory is not None and item.trajectory.trajectory_id not in summarized_ids
            ]
            retry_summaries, summary_failures = await summarizer.summarize(
                outcomes_to_summarize,
                skill_name=source.final_skill.name,
            )
            all_summaries = [*source.trajectory_summaries, *retry_summaries]
            expected_tasks = {item.spec.task.task_id for item in source.rollouts}
            summarized_tasks = {item.task_id for item in all_summaries}
            if missing_tasks := sorted(expected_tasks - summarized_tasks):
                raise RuntimeError(f"cannot produce one change decision per task; missing summaries: {missing_tasks}")
            changes = await self._propose_changes(source.final_skill, all_summaries)
            merges = await self._merge_skills(source.final_skill, changes)
            after_skill = _apply_merges(source.final_skill, merges, run_id=run_id)
        return TaskVirtualSkillRefinementResult(
            source_run_id=source.run_id,
            run_id=run_id,
            before_skill=source.final_skill,
            after_skill=after_skill,
            retry_rollouts=retry_outcomes,
            retry_summaries=retry_summaries,
            failed_summary_trajectory_ids=summary_failures,
            changes=changes,
            merges=merges,
        )

    async def _retry_failures(self, *, run_id: str, source: TaskVirtualSkillResult) -> list[RolloutOutcome]:
        config = self._config
        scheduler = RolloutScheduler(
            agent_resolver=self._agent_resolver,
            env_factory=self._env_factory,
            config=RolloutConfig(
                max_concurrent_rollouts=config.rollout.max_concurrent_rollouts,
                timeout_seconds=config.rollout.timeout_seconds,
                retry=RetryConfig(
                    max_attempts=config.rollout.retry.max_attempts,
                    backoff_seconds=config.rollout.retry.backoff_seconds,
                ),
                fail_fast=False,
                workspace_root=config.rollout.workspace_root,
            ),
        )
        reflection = config.refinement
        reflector = ReflectionGenerator(
            _ChatWithDefaults(self._reflection_model, self._chat_kwargs),
            max_trajectory_chars=reflection.max_trajectory_chars,
            max_reflection_chars=reflection.max_reflection_chars,
            max_concurrency=reflection.max_concurrent_reflections,
        )
        latest: dict[str, RolloutOutcome] = {}
        current_tasks = {}
        for outcome in source.rollouts:
            latest[outcome.spec.task.task_id] = outcome
            current_tasks[outcome.spec.task.task_id] = outcome.spec.task
        retry_outcomes: list[RolloutOutcome] = []
        sequence = max((item.spec.sequence_no for item in source.rollouts), default=-1) + 1
        for retry_round in range(1, reflection.retry_rounds + 1):
            failures = [
                outcome
                for outcome in latest.values()
                if not _is_success(outcome, success_reward=reflection.success_reward)
            ]
            if not failures:
                break

            async def build_spec(outcome: RolloutOutcome, offset: int) -> RolloutSpec | None:
                trajectory = outcome.trajectory
                if trajectory is None:
                    return None
                text = await reflector.reflect(trajectory, sample_index=retry_round)
                if not text:
                    return None
                answer = previous_answer(trajectory, max_chars=reflection.max_previous_answer_chars)
                task_id = outcome.spec.task.task_id
                reflected_task = task_with_reflection(current_tasks[task_id], answer=answer, reflection=text)
                current_tasks[task_id] = reflected_task
                return outcome.spec.model_copy(
                    update={
                        "sequence_no": sequence + offset,
                        "rollout_id": stable_rollout_id(run_id, "dynamic_retry", task_id, retry_round),
                        "sample_index": outcome.spec.sample_index + retry_round,
                        "task": reflected_task,
                        "skills": [source.final_skill],
                        "seed": self._config.batch.seed + sequence + offset,
                        "metadata": {
                            **outcome.spec.metadata,
                            "reflection_context": {
                                "prompt_version": REFLECTION_PROMPT_VERSION,
                                "source_rollout_id": outcome.spec.rollout_id,
                                "previous_answer": answer,
                                "content": text,
                                "retry_round": retry_round,
                            },
                        },
                    },
                    deep=True,
                )

            planned = await asyncio.gather(*(build_spec(outcome, index) for index, outcome in enumerate(failures)))
            specs = [item for item in planned if item is not None]
            if not specs:
                break
            round_outcomes = await scheduler.run(specs)
            retry_outcomes.extend(round_outcomes)
            sequence += len(specs)
            latest.update({item.spec.task.task_id: item for item in round_outcomes})
        retry_outcomes.sort(key=lambda item: item.spec.sequence_no)
        return retry_outcomes

    async def _propose_changes(
        self,
        skill: Skill,
        summaries: list[TrajectoryKeyPoints],
    ) -> list[TaskSkillChange]:
        metadata = VirtualComponentsMetadata.model_validate(skill.runtime_metadata)
        components = [
            {
                "skill_id": item.component_id,
                "name": item.name,
                "description": item.description,
                "content": item.content,
            }
            for item in metadata.components
        ]
        known_ids = {item["skill_id"] for item in components}
        grouped: dict[str, list[TrajectoryKeyPoints]] = defaultdict(list)
        for summary in summaries:
            grouped[summary.task_id].append(summary)
        semaphore = asyncio.Semaphore(self._config.refinement.max_concurrent_changes)

        async def run(task_id: str, items: list[TrajectoryKeyPoints]) -> TaskSkillChange:
            evidence_ids = {item.trajectory_id for item in items}
            messages = [
                {"role": "system", "content": CHANGE_SYSTEM},
                {
                    "role": "user",
                    "content": change_user(task_id=task_id, virtual_skills=components, summaries=items),
                },
            ]
            async with semaphore:
                response = await self._change_model.chat(
                    task=f"task_virtual_skill.change.{task_id}",
                    messages=messages,
                    **self._chat_kwargs,
                )
            result = _parse_change_response(response, task_id=task_id)
            if result.operation == "update" and result.skill_id not in known_ids:
                raise ValueError(f"update names unknown virtual Skill: {result.skill_id}")
            if result.operation == "create" and result.skill_id in known_ids:
                raise ValueError(f"create reuses existing virtual Skill id: {result.skill_id}")
            if set(result.evidence_trajectory_ids) - evidence_ids:
                raise ValueError("change cites a trajectory from another task")
            return result

        results = await asyncio.gather(*(run(task_id, items) for task_id, items in grouped.items()))
        return sorted(results, key=lambda item: item.task_id)

    async def _merge_skills(
        self,
        skill: Skill,
        changes: list[TaskSkillChange],
    ) -> list[VirtualSkillMerge]:
        metadata = VirtualComponentsMetadata.model_validate(skill.runtime_metadata)
        components = {item.component_id: item for item in metadata.components}
        grouped: dict[str, list[TaskSkillChange]] = defaultdict(list)
        for change in changes:
            if change.operation != "noop":
                assert change.skill_id is not None
                grouped[change.skill_id].append(change)
        semaphore = asyncio.Semaphore(self._config.refinement.max_concurrent_merges)

        async def run(skill_id: str, items: list[TaskSkillChange]) -> VirtualSkillMerge:
            operations = {item.operation for item in items}
            if len(operations) != 1:
                raise ValueError(f"conflicting operations for virtual Skill {skill_id}: {sorted(operations)}")
            operation = items[0].operation
            assert operation in {"create", "update"}
            component = components.get(skill_id)
            if operation == "update" and component is None:
                raise ValueError(f"cannot update unknown virtual Skill: {skill_id}")
            if operation == "create" and component is not None:
                raise ValueError(f"cannot create existing virtual Skill: {skill_id}")
            if len(items) == 1:
                item = items[0]
                return VirtualSkillMerge(
                    operation=operation,
                    skill_id=skill_id,
                    name=item.name,
                    description=item.description,
                    source_task_ids=[item.task_id],
                    original_content=component.content if component is not None else None,
                    revised_content=item.content,
                    change_summary=item.diagnosis,
                )
            task_ids = {item.task_id for item in items}

            def parse(content: str) -> VirtualSkillMerge:
                payload = json.loads(content)
                applied = payload.pop("applied_task_ids")
                if not isinstance(applied, list) or not applied or set(applied) - task_ids:
                    raise ValueError("merge applied_task_ids must be a non-empty subset of source tasks")
                name = payload.pop("name")
                description = payload.pop("description")
                return VirtualSkillMerge(
                    operation=operation,
                    skill_id=skill_id,
                    name=name,
                    description=description,
                    source_task_ids=applied,
                    original_content=component.content if component is not None else None,
                    **payload,
                )

            async with semaphore:
                response = await self._merge_model.chat(
                    task=f"task_virtual_skill.merge.{skill_id}",
                    messages=[
                        {"role": "system", "content": MERGE_SYSTEM},
                        {
                            "role": "user",
                            "content": merge_user(
                                operation=operation,
                                skill=(
                                    {
                                        "skill_id": skill_id,
                                        "name": component.name,
                                        "description": component.description,
                                        "content": component.content,
                                    }
                                    if component is not None
                                    else None
                                ),
                                changes=[item.model_dump(mode="json") for item in items],
                            ),
                        },
                    ],
                    format_parser=parse,
                    feedback_on_parse_error=True,
                    **self._chat_kwargs,
                )
            return _parsed(response, parse)

        return await asyncio.gather(*(run(skill_id, items) for skill_id, items in grouped.items()))


class _ChatWithDefaults:
    def __init__(self, client: ChatModel, defaults: Mapping[str, Any]) -> None:
        self._client = client
        self._defaults = dict(defaults)

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self._client.chat(task, messages, **{**self._defaults, **kwargs})


def _parsed(response: Any, parser: Any) -> Any:
    if isinstance(response, str):
        return parser(response)
    if isinstance(response, dict):
        return response.get("parsed") or parser(str(response.get("content") or ""))
    return getattr(response, "parsed", None) or parser(str(getattr(response, "content", "") or ""))


def _parse_change_response(response: Any, *, task_id: str) -> TaskSkillChange:
    if isinstance(response, str):
        content = response
    elif isinstance(response, dict):
        content = str(response.get("content") or "")
    else:
        content = str(getattr(response, "content", "") or "")
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:].lstrip("\r\n")
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise TypeError("Skill change response must be one JSON object")
    return TaskSkillChange(task_id=task_id, **payload)


def _is_success(outcome: RolloutOutcome, *, success_reward: float) -> bool:
    score = outcome.trajectory.reward.score if outcome.trajectory is not None else None
    return score is not None and score >= success_reward


def _apply_merges(skill: Skill, merges: list[VirtualSkillMerge], *, run_id: str) -> Skill:
    if not merges:
        return skill
    metadata = VirtualComponentsMetadata.model_validate(skill.runtime_metadata)
    revisions = {
        item.skill_id: item.revised_content
        for item in merges
        if item.operation == "update" and item.revised_content is not None
    }
    update_merges = {item.skill_id: item for item in merges if item.operation == "update"}
    creations = [item for item in merges if item.operation == "create"]
    components = [
        item.model_copy(
            update=(
                {
                    **(
                        {"name": update_merges[item.component_id].name}
                        if update_merges[item.component_id].name is not None
                        else {}
                    ),
                    **(
                        {"description": update_merges[item.component_id].description}
                        if update_merges[item.component_id].description is not None
                        else {}
                    ),
                    **(
                        {"content": update_merges[item.component_id].revised_content}
                        if update_merges[item.component_id].revised_content is not None
                        else {}
                    ),
                }
                if item.component_id in update_merges
                else {}
            )
        )
        for item in metadata.components
    ]
    components.extend(
        VirtualComponent(
            component_id=item.skill_id,
            name=item.name or "",
            description=item.description or "",
            content=item.revised_content or "",
        )
        for item in creations
    )
    next_metadata = VirtualComponentsMetadata(
        components=components,
        max_initial_components=metadata.max_initial_components,
    )
    resources = dict(skill.resources)
    for skill_id, content in revisions.items():
        path = f"virtual_skills/{skill_id}.md"
        if path in resources:
            resources[path] = content.rstrip() + "\n"
    for item in creations:
        resources[f"virtual_skills/{item.skill_id}.md"] = item.revised_content.rstrip() + "\n"
    merge_summary = "\n".join(f"- `{item.skill_id}`: {item.change_summary}" for item in merges)
    blob = {
        "SKILL.md": skill.content.rstrip()
        + "\n\n## Latest refinement\n\n"
        + merge_summary
        + "\n"
    }
    now = datetime.now(UTC)
    major, minor, patch = (int(value) for value in skill.version_label.split("."))
    return skill.model_copy(
        update={
            "version_id": f"evolve-run:{run_id}:refined-virtual-skills",
            "parent_version_ids": [skill.version_id],
            "version_label": f"{major}.{minor}.{patch + 1}",
            "content_hash": compute_skill_content_hash(blob),
            "status": SkillVersionStatus.DRAFT,
            "origin": SkillVersionOrigin.EVOLUTION,
            "blob": blob,
            "resources": resources,
            "runtime_metadata": next_metadata.model_dump(mode="json"),
            "commit_message": "Merge task-attributed virtual Skill refinements",
            "created_at": now,
            "updated_at": now,
            "metadata": {
                **skill.metadata,
                "refinement": {
                    "algorithm": "task_virtual_skill",
                    "run_id": run_id,
                    "changed_skill_ids": sorted(item.skill_id for item in merges),
                },
            },
        },
        deep=True,
    )


__all__ = ["TaskVirtualSkillRefiner"]
