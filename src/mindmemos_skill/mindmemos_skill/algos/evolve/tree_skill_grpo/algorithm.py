"""Batch-wise Tree Skill GRPO evolution."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ....agents.base import Agent
from ....envs.base import EnvRolloutContext
from ....llm import llm_run_context
from ....registry import ComponentRequirements, ComponentType, get_env, register
from ....typing import (
    EvolveInput,
    EvolveOutput,
    Rollout,
    RolloutType,
    Skill,
    Task,
    Trajectory,
    compute_skill_content_hash,
)
from ..base import EvolveAlgorithm, EvolveAlgorithmContext
from .config import TreeSkillGrpoConfig
from .models import SkillTree, SkillTreeEdit, TreeEditProposal, TreeFusionResult
from .prompts import reflection_messages, task_with_reflection, tree_edit_messages, tree_fusion_messages


class ChatModel(Protocol):
    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...


@register(
    type=ComponentType.ALGO,
    name="tree_skill_grpo",
    config_model=TreeSkillGrpoConfig,
    capabilities={"evolve"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"chat"})),
)
class TreeSkillGrpoAlgorithm(EvolveAlgorithm):
    """Evolve one Markdown Skill tree after each rollout batch."""

    algorithm_name = "tree_skill_grpo"

    def __init__(
        self,
        *,
        config: TreeSkillGrpoConfig | None = None,
        env_ref: str | None = None,
        chat_model: ChatModel | None = None,
        agent: Agent[Any] | None = None,
        context: EvolveAlgorithmContext | None = None,
        agent_ref: str | None = None,
        env_options: Mapping[str, Any] | None = None,
        agent_options: Mapping[str, Any] | None = None,
        workspace_root: Path | None = None,
        checkpoint_dir: Path | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        max_task_attempts: int | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        agent_ref = agent_ref or (config.agent_ref if config is not None else "react")
        env_ref = env_ref or (config.env_ref if config is not None else None)
        env_options = env_options if env_options is not None else (config.env_options if config is not None else {})
        agent_options = (
            agent_options if agent_options is not None else (config.agent_options if config is not None else {})
        )
        workspace_root = workspace_root or (config.workspace_root if config is not None else None)
        checkpoint_dir = checkpoint_dir or (config.checkpoint_dir if config is not None else None)
        epochs = epochs if epochs is not None else (config.epochs if config is not None else 1)
        batch_size = batch_size if batch_size is not None else (config.batch_size if config is not None else 40)
        max_task_attempts = (
            max_task_attempts
            if max_task_attempts is not None
            else (config.max_task_attempts if config is not None else 4)
        )
        max_concurrency = (
            max_concurrency if max_concurrency is not None else (config.max_concurrency if config is not None else 32)
        )
        if context is not None:
            chat_model = chat_model or context.models.get("chat")
            agent = agent or context.agents.get(agent_ref)
        if chat_model is None:
            raise ValueError("tree_skill_grpo requires a chat model")
        if agent is None:
            raise ValueError(f"tree_skill_grpo requires agent {agent_ref!r}")
        if env_ref is None or not env_ref.strip():
            raise ValueError("env_ref must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if max_task_attempts < 1:
            raise ValueError("max_task_attempts must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._chat_model = chat_model
        self._agent = agent
        self._env_ref = env_ref
        self._env_options = dict(env_options or {})
        self._agent_options = dict(agent_options or {})
        self._workspace_root = workspace_root
        self._checkpoint_dir = checkpoint_dir
        self._epochs = epochs
        self._batch_size = batch_size
        self._max_task_attempts = max_task_attempts
        self._concurrency_limiter = asyncio.Semaphore(max_concurrency)

    async def evolve(self, request: EvolveInput) -> EvolveOutput:
        """Run rollout, reflection, attribution, and tree update batches."""

        with llm_run_context(request.run_id):
            return await self._evolve(request)

    async def _evolve(self, request: EvolveInput) -> EvolveOutput:
        skill_tree = SkillTree.from_markdown(request.base_skill.content)
        if not skill_tree.roots:
            raise ValueError("tree_skill_grpo requires at least one Markdown heading")

        trajectories: list[Trajectory] = []
        batches_per_epoch = (len(request.train_tasks) + self._batch_size - 1) // self._batch_size
        for epoch_index in range(self._epochs):
            for local_batch_index, start in enumerate(range(0, len(request.train_tasks), self._batch_size)):
                batch_index = epoch_index * batches_per_epoch + local_batch_index
                tasks = request.train_tasks[start : start + self._batch_size]
                batch_tree = deepcopy(skill_tree)
                batch_tree_hash = self._tree_hash(batch_tree)
                batch_skill = self._skill_from_tree(request.base_skill, batch_tree)
                cases = await asyncio.gather(
                    *(
                        self._sample_task(
                            request=request,
                            task=task,
                            skill=batch_skill,
                            batch_index=batch_index,
                            task_index=task_index,
                        )
                        for task_index, task in enumerate(tasks)
                    )
                )
                trajectories.extend(trajectory for case in cases for trajectory in case)
                proposal_jobs = [
                    self._propose_tree_edit(
                        run_id=request.run_id,
                        batch_index=batch_index,
                        skill_tree=batch_tree,
                        task_index=task_index,
                        base_tree_hash=batch_tree_hash,
                        trajectories=case,
                    )
                    for task_index, case in enumerate(cases)
                    if len(case) >= 2 and not self._is_success(case[0])
                ]
                proposals = [
                    proposal
                    for proposal in await asyncio.gather(*proposal_jobs)
                    if proposal is not None and proposal.edit.operation != "reject"
                ]
                proposal_groups: dict[str, list[TreeEditProposal]] = {}
                for proposal in proposals:
                    target_node_id = proposal.edit.target_node_id
                    if target_node_id is None:
                        continue
                    proposal_groups.setdefault(target_node_id, []).append(proposal)
                fusion_target_ids = sorted(proposal_groups)
                fusion_results = await asyncio.gather(
                    *(
                        self._fuse_tree_edits(
                            run_id=request.run_id,
                            batch_index=batch_index,
                            skill_tree=batch_tree,
                            target_node_id=target_node_id,
                            proposals=proposal_groups[target_node_id],
                        )
                        for target_node_id in fusion_target_ids
                    )
                )
                skill_tree = self._apply_fusion_results(
                    batch_tree,
                    zip(fusion_target_ids, fusion_results, strict=True),
                )
            self._save_epoch_checkpoint(epoch_index + 1, skill_tree)

        final_skill = self._skill_from_tree(request.base_skill, skill_tree)
        return EvolveOutput(
            run_id=request.run_id,
            final_skill=final_skill,
            changed=final_skill.content != request.base_skill.content,
            trajectories=trajectories,
        )

    async def _sample_task(
        self,
        *,
        request: EvolveInput,
        task: Task,
        skill: Skill,
        batch_index: int,
        task_index: int,
    ) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        retry_task = task
        for attempt_index in range(self._max_task_attempts):
            trajectory = await self._rollout(
                request=request,
                task=retry_task,
                skill=skill,
                batch_index=batch_index,
                task_index=task_index,
                attempt_index=attempt_index,
            )
            trajectories.append(trajectory)
            if self._is_success(trajectory) or attempt_index + 1 == self._max_task_attempts:
                break
            reflection = await self._reflect(trajectory, batch_index=batch_index, attempt_index=attempt_index)
            retry_task = task_with_reflection(
                task,
                previous_answer=self._previous_answer(trajectory),
                reflection=reflection,
            )
        return trajectories

    async def _rollout(
        self,
        *,
        request: EvolveInput,
        task: Task,
        skill: Skill,
        batch_index: int,
        task_index: int,
        attempt_index: int,
    ) -> Trajectory:
        rollout_id = f"{request.run_id}:batch-{batch_index + 1:04d}:task-{task_index + 1:04d}"
        context = EnvRolloutContext(
            rollout=Rollout(
                rollout_id=rollout_id,
                attempt_no=attempt_index,
                rollout_type=RolloutType.TRAIN,
            ),
            env_ref=self._env_ref,
            workspace_root=self._workspace_root,
            workspace_scope=f"train/batch_{batch_index + 1:04d}",
            agent_options=dict(self._agent_options),
            metadata={
                "batch_index": batch_index,
                "task_index": task_index,
                "sample_index": attempt_index,
                "reflection_retry": attempt_index > 0,
            },
        )
        env = get_env(name=self._env_ref, config=self._env_options)
        async with self._concurrency_limiter:
            async with env:
                trajectory = await env.rollout(self._agent, task, [skill], context=context)
        trajectory.environment.env_ref = self._env_ref
        return trajectory

    async def _reflect(self, trajectory: Trajectory, *, batch_index: int, attempt_index: int) -> str:
        return (
            await self._chat_content(
                task=(
                    f"tree_skill_grpo.reflection.batch-{batch_index + 1}."
                    f"{trajectory.task.task_id}.attempt-{attempt_index + 1}"
                ),
                messages=reflection_messages(trajectory),
            )
        )[:4_000].strip()

    async def _propose_tree_edit(
        self,
        run_id: str,
        batch_index: int,
        task_index: int,
        skill_tree: SkillTree,
        base_tree_hash: str,
        trajectories: list[Trajectory],
    ) -> TreeEditProposal | None:
        content = await self._chat_content(
            task=(f"tree_skill_grpo.attribution.{run_id}.batch-{batch_index + 1}.task-{task_index + 1}"),
            messages=tree_edit_messages(skill_tree, trajectories),
        )
        payload = self._json_object(content)
        if payload is None:
            return None
        try:
            edit = SkillTreeEdit.model_validate_json(payload)
        except ValueError:
            return None
        if edit.operation == "reject":
            return None
        if edit.target_node_id is None:
            return None
        try:
            target = skill_tree.node(edit.target_node_id)
        except ValueError:
            return None
        if edit.operation == "edit_node" and target.content.count(edit.find or "") != 1:
            return None
        return TreeEditProposal(
            task_id=trajectories[0].task.task_id,
            base_tree_hash=base_tree_hash,
            outcome_pattern=(
                "failure_to_success" if any(self._is_success(item) for item in trajectories[1:]) else "repeated_failure"
            ),
            edit=edit,
        )

    async def _fuse_tree_edits(
        self,
        *,
        run_id: str,
        batch_index: int,
        skill_tree: SkillTree,
        target_node_id: str,
        proposals: list[TreeEditProposal],
    ) -> TreeFusionResult | None:
        expected_hash = self._tree_hash(skill_tree)
        if any(proposal.base_tree_hash != expected_hash for proposal in proposals):
            raise ValueError("tree fusion received proposals from different base trees")
        content = await self._chat_content(
            task=f"tree_skill_grpo.fusion.{run_id}.batch-{batch_index + 1}.{target_node_id}",
            messages=tree_fusion_messages(skill_tree, target_node_id=target_node_id, proposals=proposals),
        )
        payload = self._json_object(content)
        if payload is None:
            return None
        try:
            result = TreeFusionResult.model_validate_json(payload)
        except ValueError:
            return None
        target = skill_tree.node(target_node_id)
        headings = {child.heading.casefold() for child in target.children}
        for child in result.children:
            normalized_heading = child.heading.casefold()
            if normalized_heading in headings:
                return None
            headings.add(normalized_heading)
        candidate = deepcopy(skill_tree)
        if not self._apply_one_fusion(candidate, target_node_id, result):
            return None
        return result

    @classmethod
    def _apply_fusion_results(
        cls,
        batch_tree: SkillTree,
        results: Iterable[tuple[str, TreeFusionResult | None]],
    ) -> SkillTree:
        candidate = deepcopy(batch_tree)
        for target_node_id, result in results:
            if result is None:
                continue
            proposed = deepcopy(candidate)
            if cls._apply_one_fusion(proposed, target_node_id, result):
                candidate = proposed
        return candidate

    @staticmethod
    def _apply_one_fusion(candidate: SkillTree, target_node_id: str, result: TreeFusionResult) -> bool:
        try:
            target = candidate.node(target_node_id)
            target.content = result.content.strip()
            for child in result.children:
                candidate.apply(SkillTreeEdit(operation="create_child", target_node_id=target_node_id, child=child))
            return SkillTree.from_markdown(candidate.to_markdown()) == candidate
        except ValueError:
            return False

    @staticmethod
    def _json_object(content: str) -> str | None:
        start = content.find("{")
        end = content.rfind("}")
        return content[start : end + 1] if start >= 0 and end >= start else None

    async def _chat_content(self, *, task: str, messages: list[dict[str, Any]]) -> str:
        async with self._concurrency_limiter:
            response = await self._chat_model.chat(task=task, messages=messages)
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            content = response.get("content")
            return (str(content) if content is not None else json.dumps(response, ensure_ascii=False)).strip()
        return str(getattr(response, "content", "") or "").strip()

    def _save_epoch_checkpoint(self, epoch: int, skill_tree: SkillTree) -> None:
        if self._checkpoint_dir is None:
            return
        target = self._checkpoint_dir / f"epoch_{epoch:04d}"
        target.mkdir(parents=True, exist_ok=False)
        (target / "SKILL.md").write_text(skill_tree.to_markdown(), encoding="utf-8")
        (target / "skill_tree.json").write_text(
            json.dumps(asdict(skill_tree), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _is_success(trajectory: Trajectory) -> bool:
        return trajectory.reward.score is not None and trajectory.reward.score > 0.0

    @staticmethod
    def _previous_answer(trajectory: Trajectory) -> str:
        for event in reversed(trajectory.events):
            if event.get("role") != "assistant":
                continue
            content = event.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            return json.dumps(event, ensure_ascii=False, default=str)
        return "(no assistant answer was recorded)"

    @staticmethod
    def _tree_hash(skill_tree: SkillTree) -> str:
        return hashlib.sha256(skill_tree.to_markdown().encode("utf-8")).hexdigest()

    @staticmethod
    def _skill_from_tree(base_skill: Skill, skill_tree: SkillTree) -> Skill:
        content = skill_tree.to_markdown()
        blob = {"SKILL.md": content}
        return base_skill.model_copy(
            update={
                "blob": blob,
                "content_hash": compute_skill_content_hash(blob),
                "updated_at": datetime.now(UTC),
            },
            deep=True,
        )


__all__ = ["ChatModel", "TreeSkillGrpoAlgorithm"]
