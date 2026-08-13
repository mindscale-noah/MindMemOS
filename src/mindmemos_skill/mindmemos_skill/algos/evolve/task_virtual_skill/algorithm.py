"""One-batch rollout, summarize, sample, and grounded Skill decomposition."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ....errors import SkillConfigurationError
from ....llm import llm_run_context
from ....persistence.enums import SkillVersionOrigin, SkillVersionStatus
from ....registry import ComponentRequirements, ComponentType, register
from ....typing import EvolveInput, Skill, SkillCandidate, compute_skill_content_hash
from ..base import trajectories_from_rollouts
from ..skill_grpo_with_replay_buffer.batch_planner import TaskBatchPlanner
from ..skill_grpo_with_replay_buffer.config import RetryConfig, RolloutConfig
from ..skill_grpo_with_replay_buffer.models import ChatModel
from ..skill_grpo_with_replay_buffer.rollout import (
    AgentResolver,
    EnvFactory,
    FixedGroupPlan,
    MappingAgentResolver,
    RegistryEnvFactory,
    RolloutScheduler,
    RolloutStrategyRegistry,
)
from .config import TaskVirtualSkillRunConfig
from .models import (
    TaskVirtualSkillInput,
    TaskVirtualSkillPlan,
    TaskVirtualSkillResult,
    VirtualSkillArtifact,
)
from .prompts import DECOMPOSE_SYSTEM, decomposition_user
from .summarizer import TrajectoryKeyPointSummarizer


class AlgorithmContext(Protocol):
    models: Mapping[str, Any]
    agents: Mapping[str, Any]


@register(
    type=ComponentType.ALGO,
    name="task_virtual_skill",
    config_model=TaskVirtualSkillRunConfig,
    capabilities={"evolve"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"summary", "decomposition"})),
)
class TaskVirtualSkillEvolve:
    """Infer independent subtask Skills from one real rollout batch."""

    algorithm_name = "task_virtual_skill"

    def __init__(
        self,
        *,
        config: TaskVirtualSkillRunConfig | None = None,
        context: AlgorithmContext | None = None,
        summary_model: ChatModel | None = None,
        decomposition_model: ChatModel | None = None,
        agent_resolver: AgentResolver | None = None,
        env_factory: EnvFactory | None = None,
        summary_chat_kwargs: Mapping[str, Any] | None = None,
        decomposition_chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if context is not None:
            summary_model = context.models.get("summary")
            decomposition_model = context.models.get("decomposition")
            agent_resolver = MappingAgentResolver(context.agents)
            env_factory = RegistryEnvFactory()
        if summary_model is None or decomposition_model is None or agent_resolver is None or env_factory is None:
            raise SkillConfigurationError(
                "task_virtual_skill requires summary_model, decomposition_model, agent_resolver, and env_factory"
            )
        self._config = config
        self._summary_model = summary_model
        self._decomposition_model = decomposition_model
        self._agent_resolver = agent_resolver
        self._env_factory = env_factory
        self._summary_chat_kwargs = dict(summary_chat_kwargs or {})
        self._decomposition_chat_kwargs = dict(decomposition_chat_kwargs or {})

    async def evolve(self, request: EvolveInput) -> TaskVirtualSkillResult:
        normalized = self._normalize_request(request)
        with llm_run_context(normalized.run_id):
            return await self._evolve(normalized)

    def _normalize_request(self, request: EvolveInput) -> TaskVirtualSkillInput:
        if isinstance(request, TaskVirtualSkillInput):
            return request
        if self._config is None:
            raise SkillConfigurationError("task_virtual_skill has no configured run settings")
        return TaskVirtualSkillInput(**request.model_dump(), config=self._config)

    async def _evolve(self, request: TaskVirtualSkillInput) -> TaskVirtualSkillResult:
        config = request.config
        batches = TaskBatchPlanner().build(
            request.train_tasks,
            epochs=1,
            batch_size=config.batch.batch_size,
            seed=config.batch.seed,
        )
        batch = batches[0]
        specs = RolloutStrategyRegistry.with_builtins().get("fixed_group").plan(
            FixedGroupPlan(
                run_id=request.run_id,
                scope="task_virtual_skill_batch",
                phase="train",
                tasks=list(batch.tasks),
                skills=[request.base_skill],
                sequence_start=0,
                group_size=config.batch.rollouts_per_task,
                agent_ref=config.dataset.agent_ref,
                env_ref=config.dataset.env_ref,
                seed=config.batch.seed,
                agent_options=config.dataset.agent_options,
                env_options=config.dataset.env_options,
            )
        )
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
                fail_fast=config.rollout.fail_fast,
                workspace_root=Path(config.rollout.workspace_root) if config.rollout.workspace_root else None,
            ),
        )
        outcomes = await scheduler.run(specs)
        summarizer = TrajectoryKeyPointSummarizer(
            chat_model=_ChatWithDefaults(self._summary_model, self._summary_chat_kwargs),
            concurrency=config.summary.max_concurrent_summaries,
            transcript_max_chars=config.summary.transcript_max_chars,
        )
        summaries, summary_failures = await summarizer.summarize(outcomes, skill_name=request.base_skill.name)
        if not summaries:
            raise RuntimeError("task_virtual_skill could not summarize any collected trajectory")
        sampled = _sample_summaries(summaries, size=config.summary.sample_size, seed=config.batch.seed)
        sampled_ids = {item.trajectory_id for item in sampled}

        def parse_grounded_plan(content: str) -> TaskVirtualSkillPlan:
            parsed = parse_plan(content)
            _validate_plan(
                parsed,
                source_skill=request.base_skill.content,
                sampled_trajectory_ids=sampled_ids,
                max_virtual_skills=config.decomposition.max_virtual_skills,
            )
            return parsed

        response = await self._decomposition_model.chat(
            task="task_virtual_skill.decompose",
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM},
                {
                    "role": "user",
                    "content": decomposition_user(
                        skill=request.base_skill,
                        summaries=sampled,
                        max_virtual_skills=config.decomposition.max_virtual_skills,
                    ),
                },
            ],
            format_parser=parse_grounded_plan,
            feedback_on_parse_error=True,
            **self._decomposition_chat_kwargs,
        )
        raw_response, plan = _decomposition_response(response, parser=parse_grounded_plan)
        _validate_plan(
            plan,
            source_skill=request.base_skill.content,
            sampled_trajectory_ids=sampled_ids,
            max_virtual_skills=config.decomposition.max_virtual_skills,
        )
        artifacts = _build_artifacts(plan)
        candidate = (
            _build_candidate(
                plan,
                artifacts,
                max_initial_components=config.decomposition.max_initial_components,
                request=request,
            )
            if artifacts
            else None
        )
        final_skill = _candidate_skill(request.base_skill, candidate, run_id=request.run_id)
        return TaskVirtualSkillResult(
            run_id=request.run_id,
            final_skill=final_skill,
            changed=candidate is not None,
            trajectories=trajectories_from_rollouts(outcomes),
            candidate=candidate,
            plan=plan,
            artifacts=artifacts,
            rollouts=outcomes,
            trajectory_summaries=summaries,
            sampled_trajectory_ids=[item.trajectory_id for item in sampled],
            failed_summary_trajectory_ids=summary_failures,
            raw_decomposition_response=raw_response,
        )


class _ChatWithDefaults:
    def __init__(self, client: ChatModel, defaults: Mapping[str, Any]) -> None:
        self._client = client
        self._defaults = dict(defaults)

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self._client.chat(task, messages, **{**self._defaults, **kwargs})


def parse_plan(value: str) -> TaskVirtualSkillPlan:
    text = value.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced is not None:
        text = fenced.group(1)
    return TaskVirtualSkillPlan.model_validate(json.loads(text))


def _decomposition_response(response: Any, *, parser: Any = parse_plan) -> tuple[str, TaskVirtualSkillPlan]:
    if isinstance(response, str):
        return response.strip(), parser(response)
    if isinstance(response, dict):
        content = str(response.get("content") or "").strip()
        parsed = response.get("parsed") or parser(content)
    else:
        content = str(getattr(response, "content", "") or "").strip()
        parsed = getattr(response, "parsed", None) or parser(content)
    return content, parsed if isinstance(parsed, TaskVirtualSkillPlan) else TaskVirtualSkillPlan.model_validate(parsed)


def _sample_summaries(summaries: list[Any], *, size: int, seed: int) -> list[Any]:
    if len(summaries) <= size:
        return list(summaries)
    sampled_ids = {item.trajectory_id for item in random.Random(seed).sample(summaries, size)}
    return [item for item in summaries if item.trajectory_id in sampled_ids]


def _validate_plan(
    plan: TaskVirtualSkillPlan,
    *,
    source_skill: str,
    sampled_trajectory_ids: set[str],
    max_virtual_skills: int,
) -> None:
    if len(plan.virtual_skills) > max_virtual_skills:
        raise ValueError(f"decomposition has {len(plan.virtual_skills)} Skills; maximum is {max_virtual_skills}")
    for skill in plan.virtual_skills:
        unknown = set(skill.supporting_trajectory_ids) - sampled_trajectory_ids
        if unknown:
            raise ValueError(f"virtual Skill {skill.skill_id!r} cites unsampled trajectories: {', '.join(sorted(unknown))}")
        for excerpt in skill.source_excerpts:
            if excerpt not in source_skill:
                raise ValueError(f"virtual Skill {skill.skill_id!r} contains a source excerpt absent from SKILL.md")


def _build_artifacts(plan: TaskVirtualSkillPlan) -> list[VirtualSkillArtifact]:
    artifacts = []
    for skill in plan.virtual_skills:
        content = "\n\n".join(excerpt.strip() for excerpt in skill.source_excerpts)
        markdown = f"# {skill.name}\n\n> {skill.description}\n\n{content}\n"
        artifacts.append(
            VirtualSkillArtifact(
                component_id=skill.skill_id,
                name=skill.name,
                description=skill.description,
                supporting_trajectory_ids=skill.supporting_trajectory_ids,
                source_excerpts=skill.source_excerpts,
                relative_path=f"virtual_skills/{skill.skill_id}.md",
                markdown=markdown,
            )
        )
    return artifacts


def _build_candidate(
    plan: TaskVirtualSkillPlan,
    artifacts: list[VirtualSkillArtifact],
    *,
    max_initial_components: int,
    request: TaskVirtualSkillInput,
) -> SkillCandidate:
    index = [f"# {plan.skill_name}", "", plan.description, "", "## Virtual Skills", ""]
    index.extend(f"- **{item.name}** (`{item.component_id}`): {item.description}" for item in artifacts)
    components = [
        {
            "component_id": item.component_id,
            "name": item.name,
            "description": item.description,
            "content": item.markdown,
        }
        for item in artifacts
    ]
    return SkillCandidate(
        blob={"SKILL.md": "\n".join(index).rstrip() + "\n"},
        resources={item.relative_path: item.markdown for item in artifacts},
        runtime_type="virtual_components",
        runtime_schema_version=1,
        runtime_metadata={"components": components, "max_initial_components": max_initial_components},
        commit_message="Initialize trajectory-grounded virtual Skills",
        metadata={
            "algorithm": TaskVirtualSkillEvolve.algorithm_name,
            "run_id": request.run_id,
            "prompt_version": request.config.prompt_version,
            "sampled_trajectory_ids": [
                trajectory_id for item in artifacts for trajectory_id in item.supporting_trajectory_ids
            ],
        },
    )


def _candidate_skill(base: Skill, candidate: SkillCandidate | None, *, run_id: str) -> Skill:
    if candidate is None:
        return base
    now = datetime.now(UTC)
    major, minor, patch = (int(value) for value in base.version_label.split("."))
    return base.model_copy(
        update={
            "version_id": f"evolve-run:{run_id}:virtual-skills",
            "parent_version_ids": [base.version_id],
            "version_label": f"{major}.{minor}.{patch + 1}",
            "content_hash": compute_skill_content_hash(candidate.blob),
            "status": SkillVersionStatus.DRAFT,
            "origin": SkillVersionOrigin.EVOLUTION,
            "blob": candidate.blob,
            "resources": candidate.resources,
            "runtime_type": candidate.runtime_type,
            "runtime_schema_version": candidate.runtime_schema_version,
            "runtime_metadata": candidate.runtime_metadata,
            "commit_message": candidate.commit_message,
            "created_at": now,
            "updated_at": now,
            "metadata": {**base.metadata, "evolution": candidate.metadata},
        },
        deep=True,
    )


__all__ = ["TaskVirtualSkillEvolve", "parse_plan"]
