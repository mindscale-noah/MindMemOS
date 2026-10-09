"""Run a complete local Skill training flow with a custom environment."""

import argparse
import asyncio
import re
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import yaml
from mindmemos_skill.agents import OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.algos.train.base import SkillTrainer
from mindmemos_skill.api.typing import (
    HistoryRequest,
    RegisterRequest,
    ShowRequest,
    StructuredSkillInfo,
    TrainingRequest,
    TrainingSplit,
)
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task, Usage
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.infra.task_manager.progress import ProgressReporter
from mindmemos_skill.skill import Skill, SkillContent, SkillSet
from pydantic import BaseModel, ConfigDict

from mindmemos_skill import FastAPIConfig, HttpAuthConfig, ServiceConfig, SkillConfig, create_service


class LookupConfig(BaseModel):
    """Configure the text Agent used to answer each task."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported environment settings."""

    agent: OpenAIChatConfig
    """Model and connection settings recorded with each rollout."""


class LookupRewardData(RewardData):
    """Hold the expected answer outside the public task."""

    data: str
    """Private reference answer used only by the scorer."""


@Environment.register("guide_lookup")
class LookupEnvironment(Environment[LookupConfig]):
    """Ask an Agent to answer using the assembled Skill and score its answer."""

    config_type = LookupConfig
    """Environment configuration validated by the registry."""

    task_type = Task
    """Public task type used when restoring serialized requests."""

    reward_data_type = LookupRewardData
    """Private reference type used when restoring serialized requests."""

    def __init__(self, config: LookupConfig) -> None:
        """Create the Agent configured for this environment.

        Args:
            config: Validated model and connection settings.
        """
        super().__init__(config)
        self.agent = OpenAIChatAgent(config.agent)

    def _run(
        self,
        *,
        task: Task,
        reward_data: RewardData,
        skills: list[Skill],
        workspace: Path,
    ) -> RolloutResult:
        """Extract the Agent's tagged answer and compare it with private data."""
        if not isinstance(reward_data, LookupRewardData):
            raise TypeError("LookupEnvironment requires LookupRewardData")
        skill_text = "\n\n".join(skill.content for skill in skills)
        response = self.agent.run(
            system=f"Follow these Skill instructions. Put the final answer in <answer>...</answer>.\n\n{skill_text}",
            user=task.instruction,
        )
        if not isinstance(response.output, str):
            raise TypeError("LookupEnvironment requires a text Agent response")
        match = re.search(r"<answer>(.*?)</answer>", response.output, re.DOTALL)
        answer = match.group(1).strip() if match else ""
        success = answer == reward_data.data
        return RolloutResult(
            output=answer,
            trajectory=response.trajectory,
            reward=Reward(score=float(success), success=success, detail="correct" if success else "incorrect"),
            usage=Usage(**{"reported": bool(response.usage), **response.usage}),
        )


class LookupTrainerConfig(BaseModel):
    """Configure the deterministic training algorithm."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported trainer settings."""


@Algorithm.register("guide_lookup_trainer")
class LookupTrainer(SkillTrainer[LookupTrainerConfig]):
    """Add mappings for failed training tasks without modifying the input Skill."""

    config_type = LookupTrainerConfig
    """Trainer configuration validated by the registry."""

    async def train(
        self,
        *,
        skill: Skill,
        env: Environment,
        train_samples: Sequence[DataSample],
        validation_samples: Sequence[DataSample],
        work_dir: Path,
        progress: ProgressReporter | None = None,
    ) -> Skill:
        """Train mappings from the supplied samples and environment feedback.

        Args:
            skill: Immutable input Skill.
            env: Environment used to score each training task.
            train_samples: Public tasks and private reference answers for training.
            validation_samples: Reserved samples; this toy trainer does not select on them.
            work_dir: Directory containing independent baseline rollouts.
            progress: Optional pipeline progress reporter.

        Returns:
            A Skill with mappings learned from failed training samples.
        """
        additions: dict[str, str] = {}
        for sample in train_samples:
            if not isinstance(sample.reward_data, LookupRewardData):
                raise TypeError("LookupTrainer requires LookupRewardData")
            result = await env.rollout(
                sample.task,
                work_dir / "baseline" / sample.task.id,
                reward_data=sample.reward_data,
                skill_set=SkillSet({"source": skill}, max_concurrency=1),
            )
            if not result.reward.success:
                additions[sample.task.instruction] = sample.reward_data.data
        if not additions:
            return skill
        new_rules = "".join(f"{key} = {value}\n" for key, value in sorted(additions.items()))
        return skill.with_content(skill.content.rstrip() + "\n" + new_rules)


async def main(model_config: OpenAIChatConfig, forest_model: str | None = None) -> None:
    """Register, train, evaluate, and inspect one local Skill.

    Args:
        model_config: Explicit connection and generation settings for the answering Agent.
        forest_model: Optional optimizer model for Skill-Forest.
    """
    model_config.require_connection()
    run_root = (Path("output/skill-guide") / uuid4().hex).resolve()
    run_root.mkdir(parents=True)
    trainer = (
        {"class_name": "guide_lookup_trainer"}
        if forest_model is None
        else {
            "class_name": "skill_forest",
            "config": {
                "optimizer": model_config.model_copy(update={"model": forest_model}),
                "num_trees": 1,
                "candidates_per_node": 1,
                "max_depth": 1,
                "expansion_budget": 1,
                "expansion_concurrency": 1,
                "rollout_concurrency": 2,
                "model_concurrency": 1,
            },
        }
    )
    config = ServiceConfig.model_validate(
        {
            "home": run_root,
            "persistence": {
                "database": {
                    "class_name": "sqlite",
                    "config": {"url": f"sqlite+pysqlite:///{run_root / 'skills.db'}"},
                },
            },
            "repository_pipeline": {"class_name": "repository"},
            "training_pipelines": {
                "lookup": {
                    "class_name": "training",
                    "config": {"trainer": trainer},
                }
            },
        }
    )
    samples = [
        DataSample(task=Task(id="train-cat", instruction="cat"), reward_data=LookupRewardData(data="猫")),
        DataSample(task=Task(id="train-dog", instruction="dog"), reward_data=LookupRewardData(data="狗")),
        DataSample(task=Task(id="val-cat", instruction="cat"), reward_data=LookupRewardData(data="猫")),
        DataSample(task=Task(id="test-dog", instruction="dog"), reward_data=LookupRewardData(data="狗")),
    ]
    async with await create_service(
        SkillConfig(
            service=config, fastapi=FastAPIConfig(auth=HttpAuthConfig(mode="none", default_project_id="guide-local"))
        )
    ) as service:
        initial = SkillContent(
            content="# Simple dictionary\nAnswer with the value for the given key. If no rule matches, answer unknown.\n",
            name="Demo dictionary",
            description="Dictionary lookup rules",
            standard_format=False,
        )
        registered = await service.register_skill(
            RegisterRequest(
                project_id=service.default_project_id,
                alias="guide-dictionary",
                skill=StructuredSkillInfo(**initial.model_dump()),
            )
        )
        result = await service.run_training(
            TrainingRequest(
                algo="lookup",
                project_id=service.default_project_id,
                env={"class_name": "guide_lookup", "config": {"agent": model_config}},
                init_skill_version_id=registered.skill_version_id,
                work_dir=str(run_root / "training"),
                tasks=samples,
                split=TrainingSplit(train=["train-cat", "train-dog"], val=["val-cat"], test=["test-dog"]),
            )
        )
        current = await service.show_skill(
            ShowRequest(project_id=service.default_project_id, skill_id=registered.skill_id)
        )
        history = await service.skill_history(
            HistoryRequest(project_id=service.default_project_id, skill_id=registered.skill_id)
        )
        print(f"Run directory: {run_root}")
        print(f"Published version: {result.version_id}")
        print(f"Current version: {current.version.version_id}")
        print(f"Version count: {len(history)}")
        print(f"Validation: {(run_root / 'training/val/summary.json').read_text(encoding='utf-8')}")
        print(f"Test: {(run_root / 'training/test/summary.json').read_text(encoding='utf-8')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-config", type=Path, required=True, help="Model YAML with model, api_key, and base_url")
    parser.add_argument("--forest-model", help="Run built-in Skill-Forest with this optimizer model")
    arguments = parser.parse_args()
    model_config = OpenAIChatConfig.model_validate(yaml.safe_load(arguments.llm_config.read_text(encoding="utf-8")))
    asyncio.run(main(model_config, arguments.forest_model))
