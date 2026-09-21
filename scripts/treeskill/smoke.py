#!/usr/bin/env python3
"""Exercise active sampling, fusion, routed testing and artifacts without model calls."""

import argparse
import asyncio
import json
from pathlib import Path

from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainer, TreeSkillTrainerConfig
from mindmemos_skill.algos.train.treeskill.experiment import run_experiment
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.skill import Skill, SkillBundle
from pydantic import BaseModel


class SmokeConfig(BaseModel):
    """No external settings are needed for the offline environment."""


class SmokeEnvironment(Environment[SmokeConfig]):
    """Score whether the actively learned verification step appears in the Skill."""

    def _run(self, *, task, reward_data, skills, workspace):
        text = skills[0].bundle.files["SKILL.md"]
        passed = "Verify the saved output." in text
        return RolloutResult(
            output="verified" if passed else "missing verification",
            trajectory=[
                {"role": "user", "content": task.instruction},
                {"role": "assistant", "content": "Save and verify the output."},
            ],
            reward=Reward(score=float(passed), success=passed, detail="Check saved output verification."),
        )


class SmokeAgent(OpenAIChatAgent):
    """Return stage-specific deterministic responses instead of contacting a provider."""

    def run_messages(self, *, messages, tools=None, response_format=None):
        user = messages[-1]["content"]
        payload = json.loads(user.split("\n\n", 1)[1])
        if user.startswith("Analyze"):
            output = {
                "instance_id": payload["instance_id"],
                "task_id": payload["task_id"],
                "record_source": payload["record_source"],
                "items": [{"item_id": "i1", "kind": "failure_memory", "content": "Verify the saved output."}],
            }
        elif user.startswith("Locate"):
            output = {
                "instance_id": payload["analysis_record"]["instance_id"],
                "evidence": [
                    {
                        "evidence_id": "e1",
                        "reusable_lesson": "Verify the saved output.",
                        "target_node_id": "001",
                        "rationale": "Required verification",
                    }
                ],
            }
        elif user.startswith("Fuse"):
            output = {
                "rationale": "Add verification",
                "edits": [
                    {
                        "operation": "update_node",
                        "content": "Save the output.\nVerify the saved output.",
                        "rationale": "Observed omission",
                    }
                ],
            }
        else:
            output = {"selected_subtree_ids": ["001"], "rationale": "Complete workflow"}
        text = json.dumps(output)
        return AgentResult(
            output=text,
            trajectory=[*messages, {"role": "assistant", "content": text}],
            usage={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        )


async def run_smoke(output_dir: Path):
    """Run the full offline training and held-out test flow.

    Args:
        output_dir: Fresh directory to contain the complete experiment.

    Returns:
        The persisted summary report.
    """
    model = OpenAIChatConfig(model="offline-smoke", temperature=0, api_key=None)
    trainer = TreeSkillTrainer(
        TreeSkillTrainerConfig(optimizer=model, router=model, rollout_concurrency=2), agent_factory=SmokeAgent
    )
    samples = [
        DataSample(task=Task(id=name, instruction="Save and verify the output"), reward_data=RewardData())
        for name in ("train", "test")
    ]
    source = Skill(bundle=SkillBundle(files={"SKILL.md": "# Workflow\nSave the output.\n"}))
    await run_experiment(
        trainer=trainer,
        skill=source,
        env=SmokeEnvironment(SmokeConfig()),
        train_samples=samples[:1],
        test_samples=samples[1:],
        work_dir=output_dir,
    )
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["test"]["mean_score"] == 1.0
    assert summary["training"]["accepted_patches"] == 1
    assert source.bundle.files["SKILL.md"] == "# Workflow\nSave the output.\n"
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(asyncio.run(run_smoke(parser.parse_args().output_dir)), indent=2))
