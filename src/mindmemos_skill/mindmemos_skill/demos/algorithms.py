"""Offline algorithms used through the real training and evolution pipelines."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from ..algos import Algorithm
from ..algos.evolve.base import SkillEvolver
from ..algos.evolve.typing import SkillTrajectory
from ..algos.train.base import SkillTrainer
from ..common.checkpoint import boundary
from ..common.json import read_json, write_json
from ..envs import Environment
from ..envs.data import DataSample
from ..infra.task_manager.progress import NullProgressReporter, ProgressReporter
from ..skill import Skill, SkillSet, SkillSnapshot
from .config import DemoAlgorithmConfig


@Algorithm.register("demo_train")
class DemoTrainer(SkillTrainer[DemoAlgorithmConfig]):
    """Execute supplied samples and append a deterministic demo instruction."""

    config_type = DemoAlgorithmConfig
    """Delay and appended instruction for this offline trainer."""

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
        """Run synthetic training tasks and save a reusable candidate.

        Args:
            skill: Source Skill copied before adding the demo instruction.
            env: Caller-owned synthetic environment.
            train_samples: Samples to execute in their supplied order.
            validation_samples: Held-out inputs recorded for recovery identity.
            work_dir: Trainer directory containing the input boundary and candidate.
            progress: Optional task event reporter.

        Returns:
            New Skill preserving all original resources and runtime metadata.
        """
        if not train_samples:
            raise ValueError("Demo training requires at least one training sample")
        identity = {
            "skill": skill.snapshot().model_dump(mode="json"),
            "config": self.config.model_dump(mode="json"),
            "environment": f"{type(env).__module__}.{type(env).__qualname__}",
            "environment_config": env.config.model_dump(mode="json"),
            "train": [sample.model_dump(mode="json") for sample in train_samples],
            "val": [sample.model_dump(mode="json") for sample in validation_samples],
        }
        if work_dir.exists() and any(work_dir.iterdir()) and not (work_dir / "inputs.json").exists():
            raise ValueError("Demo training directory has no input record; use a new work_dir")
        boundary(work_dir / "inputs.json", identity, payload={"protocol": 1})
        saved = boundary(work_dir / "candidate.json", identity)
        if saved is not None:
            return Skill.from_snapshot(SkillSnapshot.model_validate(saved["skill"]))
        reporter = progress or NullProgressReporter()
        await reporter.report(
            stage="train", message="Starting offline demo training", completed=0, total=len(train_samples)
        )
        for index, sample in enumerate(train_samples, start=1):
            await asyncio.sleep(self.config.delay_seconds)
            await env.rollout(
                task=sample.task,
                reward_data=sample.reward_data,
                skill_set=SkillSet({"initial": skill}),
                work_dir=work_dir / "rollouts" / f"sample-{index:04d}",
            )
            await reporter.report(
                stage="train", message=f"Processed {sample.task.id}", completed=index, total=len(train_samples)
            )
        candidate = skill.with_content(
            skill.content + f"\n## Demo training\n{self.config.note}\nTraining samples: {len(train_samples)}.\n"
        )
        write_json(work_dir / "report.json", {"demo": True, "train_count": len(train_samples), "llm_calls": 0})
        boundary(work_dir / "candidate.json", identity, payload={"skill": candidate.snapshot().model_dump(mode="json")})
        return candidate

    def build_report(self, work_dir: Path) -> dict:
        """Read the compact report saved by this trainer.

        Args:
            work_dir: Trainer directory containing report.json.

        Returns:
            Synthetic execution count and explicit zero model-call count.
        """
        return read_json(work_dir / "report.json")


@Algorithm.register("demo_evolve")
class DemoEvolver(SkillEvolver[DemoAlgorithmConfig]):
    """Append evidence count and task identities without calling a model."""

    config_type = DemoAlgorithmConfig
    """Delay and appended instruction for this offline evolver."""

    async def evolve(self, *, skill: Skill, trajectories: Sequence[SkillTrajectory]) -> Skill:
        """Create a child using only the already selected evidence.

        Args:
            skill: Input Skill whose source and resources remain unchanged.
            trajectories: Historical attempts sampled by the real evolution pipeline.

        Returns:
            Changed Skill with an auditable evidence count and task IDs.
        """
        await asyncio.sleep(self.config.delay_seconds)
        ids = ", ".join(item.task.id for item in trajectories)
        return skill.with_content(
            skill.content + f"\n## Demo evolution\n{self.config.note}\n"
            f"Sampled trajectories: {len(trajectories)}.\nEvidence tasks: {ids}.\n"
        )
