"""Extract and hierarchically merge unsupervised historical execution patches."""

import asyncio
import json
from collections.abc import Sequence

from ....agents import OpenAIChatAgent
from ....common.asyncio import run_sync
from ....common.json import parse_model
from ....skill import Skill
from ...base import Algorithm
from ...train.skill_forest.tree import apply_patch
from ...train.skill_forest.typing import Patch
from ..base import SkillEvolver
from ..typing import SkillTrajectory
from .config import SimpleEvolverConfig
from .prompts import EXTRACTION_PROMPT, MERGE_PROMPT
from .trajectory_rendering import clip_message, clip_payload


@Algorithm.register("simple_evolve")
class SimpleSkillEvolver(SkillEvolver[SimpleEvolverConfig]):
    """Extract batch patches concurrently and merge them without running new tasks."""

    config_type = SimpleEvolverConfig
    """Model settings and bounded prompt/patch limits."""

    def __init__(self, config: SimpleEvolverConfig, *, agent_factory=OpenAIChatAgent):
        """Create a detached optimizer from validated settings.

        Args:
            config: Complete model and patch settings.
            agent_factory: Injectable model constructor for verification.
        """
        super().__init__(config)
        self.agent_factory = agent_factory

    async def evolve(self, *, skill: Skill, trajectories: Sequence[SkillTrajectory]) -> Skill:
        """Extract batch patches, merge pairwise, and apply the final patch.

        Args:
            skill: Unmodified parent Skill, including auxiliary resources.
            trajectories: Previously sampled historical execution evidence.

        Returns:
            A detached Skill candidate; an empty patch preserves the original content.

        Raises:
            ValueError: Evidence is empty or a model patch is invalid.
        """
        if not trajectories:
            raise ValueError("Evolution requires at least one trajectory")
        source = skill.content
        if len(source) > self.config.max_skill_chars:
            raise ValueError("Current SKILL.md exceeds the model input limit")
        evidence = []
        for index, item in enumerate(trajectories):
            historical = {name: value.content for name, value in item.skills.items()}
            if any(len(text) > self.config.max_skill_chars for text in historical.values()):
                raise ValueError("Historical SKILL.md exceeds the model input limit")
            evidence.append(
                {
                    "trajectory_id": item.trajectory_id or f"trajectory-{index + 1}",
                    "task": clip_payload(item.task.model_dump(mode="json"), self.config.user_message_chars),
                    "task_system_prompt": clip_payload(item.task_system_prompt, self.config.user_message_chars),
                    "task_metadata": clip_payload(item.task_metadata, self.config.user_message_chars),
                    "skills_used": historical,
                    "trajectory": [clip_message(event, self.config) for event in item.trajectory],
                    "feedback_detail": clip_payload(item.feedback_detail, self.config.tool_message_chars),
                }
            )
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def propose(prompt, payload):
            async with semaphore:
                agent = self.agent_factory(self.config.optimizer)
                response = await run_sync(
                    agent.run,
                    messages=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"skill": source, "L": self.config.max_edits, **payload}, ensure_ascii=False
                            ),
                        },
                    ],
                )
            patch = parse_model(response.output, Patch)
            if len(patch.edits) > self.config.max_edits:
                raise ValueError("Evolution patch exceeds max_edits")
            apply_patch(skill, patch)
            return patch

        groups = [
            evidence[start : start + self.config.batch_size]
            for start in range(0, len(evidence), self.config.batch_size)
        ]
        patches = await asyncio.gather(*[propose(EXTRACTION_PROMPT, {"executions": group}) for group in groups])
        nodes = [
            (patch, [item["trajectory_id"] for item in group]) for patch, group in zip(patches, groups, strict=True)
        ]
        while len(nodes) > 1:
            pairs = [nodes[start : start + 2] for start in range(0, len(nodes) - 1, 2)]
            merged = await asyncio.gather(
                *[
                    propose(
                        MERGE_PROMPT,
                        {
                            "patches": [
                                {"patch": patch.model_dump(mode="json"), "trajectory_ids": ids} for patch, ids in pair
                            ]
                        },
                    )
                    for pair in pairs
                ]
            )
            next_nodes = [
                (patch, [identity for _, ids in pair for identity in ids])
                for patch, pair in zip(merged, pairs, strict=True)
            ]
            if len(nodes) % 2:
                next_nodes.append(nodes[-1])
            nodes = next_nodes
        return apply_patch(skill, nodes[0][0])
