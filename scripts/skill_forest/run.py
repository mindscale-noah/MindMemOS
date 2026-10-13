"""Submit a LiveMath smoke test to the existing make dev service through the SDK."""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import AsyncSkillClient
from mindmemos_sdk.skills.config import EnvironmentConfig
from mindmemos_sdk.skills.typing import (
    CLITrainingRequest,
    RegisterRequest,
    StructuredSkillInfo,
    TaskProgressRequest,
    TaskShowRequest,
    TrainingSplitRequest,
)
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.agents import AgentCatalog, OpenAIChatConfig
from mindmemos_skill.algos.train.skill_forest.config import SkillForestTrainerConfig
from mindmemos_skill.common.json import append_jsonl
from mindmemos_skill.envs.data import Data
from mindmemos_skill.logging import configure_logging, get_logger, log_context

ROOT = Path(__file__).resolve().parents[2]
"""Repository containing the existing service configuration and benchmark inputs."""


async def _run(args: argparse.Namespace) -> None:
    """Reuse configured model credentials and submit to the existing Skill endpoint."""
    document = yaml.safe_load(args.service_config.read_text(encoding="utf-8"))
    profile = document["service"]["training_pipelines"]["skill_forest"]
    trainer = profile["config"]["trainer"]
    if trainer["class_name"] != "skill_forest":
        raise ValueError("The skill_forest profile must select the skill_forest trainer")
    forest = SkillForestTrainerConfig.model_validate(
        trainer["config"], context=AgentCatalog(agents=document["service"].get("agents", {}))
    )
    agent = OpenAIChatConfig(
        api_key=forest.optimizer.api_key,
        base_url=forest.optimizer.base_url,
        model="gpt-5.4-mini",
        reasoning_effort="medium",
        timeout_s=600,
        max_retries=10,
    )
    agent.require_connection()
    if args.max_completion_tokens is not None:
        agent = OpenAIChatConfig.model_validate(
            {**agent.model_dump(), "max_completion_tokens": args.max_completion_tokens}
        )
    data = Data.build("livemath", {"data_path": ROOT / "data/mindmemos_skill/livemath/data"})
    resources = ROOT / "resources/mindmemos_skill_eval/skill_forest-livemath"
    split = data.split(data.load(), split_file=resources / "data_splits/splits.json")
    train, test = split["train"][: args.train_count], split["test"][: args.test_count]
    if len(train) < 2 or not test:
        raise ValueError("At least two train samples and one test sample are required")
    output = args.output.resolve()
    request = CLITrainingRequest(
        init_skill_version_id="pending",
        work_dir=str(output / "training"),
        algo="skill_forest",
        env=EnvironmentConfig(class_name="livemath", config={"agent": agent.model_dump(mode="json")}),
        tasks=[sample.model_dump(mode="json") for sample in train + test],
        split=TrainingSplitRequest(
            train=[sample.task.id for sample in train], test=[sample.task.id for sample in test]
        ),
    )
    sdk = ConfigManager(args.sdk_home).load()
    url, key = sdk.resolve_skill_connection()
    print(
        json.dumps(
            {
                "endpoint": url,
                "train": len(train),
                "test": len(test),
                "target": {
                    "model": agent.model,
                    "reasoning_effort": agent.reasoning_effort,
                    "timeout_s": agent.timeout_s,
                    "max_retries": agent.max_retries,
                    "max_completion_tokens": agent.max_completion_tokens,
                },
                "optimizer_in_config_file": {
                    "model": forest.optimizer.model,
                    "reasoning_effort": forest.optimizer.reasoning_effort,
                    "timeout_s": forest.optimizer.timeout_s,
                    "max_retries": forest.optimizer.max_retries,
                    "max_completion_tokens": forest.optimizer.max_completion_tokens,
                },
                "forest_in_config_file": {
                    "expansion_budget": forest.expansion_budget,
                    "max_depth": forest.max_depth,
                    "num_trees": forest.num_trees,
                    "expansion_concurrency": forest.expansion_concurrency,
                    "rollout_concurrency": forest.rollout_concurrency,
                    "model_concurrency": forest.model_concurrency,
                },
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    async with AsyncHttpTransport(
        base_url=url,
        api_key=key,
        timeout_seconds=sdk.network.timeout_seconds,
        max_retries=0,
        require_api_key=False,
    ) as transport:
        client = AsyncSkillClient(transport)
        if args.check:
            # An authenticated read checks the configured service without submitting training.
            from mindmemos_sdk.skills.typing import ListRequest

            await client.list(ListRequest())
            print("Preflight passed; authenticated service read succeeded. No training submitted.")
            return
        output.mkdir(parents=True, exist_ok=False)
        source = StructuredSkillInfo(
            content=(resources / "initial_skill/SKILL.md").read_text(encoding="utf-8"),
            name="livemath-forest-smoke",
            description="LiveMath initial benchmark Skill",
            standard_format=False,
        )
        registered = await client.register(RegisterRequest(skill=source))
        request.init_skill_version_id = registered["skill_version_id"]
        accepted = await client.train(request)
        (output / "task.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
        print(f"Submitted task: {accepted['task_id']}", flush=True)
        previous = None
        after = 0
        while True:
            state = await client.show_task(TaskShowRequest(task_id=accepted["task_id"]))
            (output / "status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
            while True:
                events = await client.task_progress(TaskProgressRequest(task_id=accepted["task_id"], after=after))
                for event in events:
                    await append_jsonl(output / "service_progress.jsonl", event)
                    after = max(after, event["sequence"])
                    payload = event["payload"]
                    with log_context(
                        project_id=event.get("project_id"),
                        task_id=accepted["task_id"],
                        attempt=event.get("attempt"),
                        request_id=payload.get("request_id"),
                        run_id=payload.get("run_id"),
                        tree_id=payload.get("tree_id"),
                        parent_id=payload.get("parent_id"),
                        node_id=payload.get("node_id"),
                        group_id=payload.get("group_id"),
                    ):
                        get_logger("skill_forest.client").log(
                            logging.getLevelNamesMapping().get(payload.get("level"), logging.INFO),
                            payload["message"],
                        )
                if not events or state["status"] not in {"succeeded", "failed", "cancelled"}:
                    break
            if state["status"] != previous:
                print(f"Task status: {state['status']}", flush=True)
                previous = state["status"]
            if state["status"] == "succeeded":
                print(json.dumps(state["result"], indent=2), flush=True)
                return
            if state["status"] in {"failed", "cancelled"}:
                reason = state.get("error_type") or state["status"]
                try:
                    events = await client.task_progress(TaskProgressRequest(task_id=accepted["task_id"]))
                    for event in events:
                        payload = event["payload"]
                        if payload.get("message") in {"algorithm.item.failed", "stage.failed"}:
                            traceback = payload.get("traceback")
                            if traceback:
                                reason = traceback.strip().splitlines()[-1]
                except Exception as exc:
                    print(f"Could not retrieve failure details: {type(exc).__name__}", flush=True)
                (output / "error.txt").write_text(reason + "\n", encoding="utf-8")
                raise RuntimeError(f"Training ended as {state['status']}: {reason}; inspect {output / 'status.json'}")
            await asyncio.sleep(5)


def main() -> None:
    """Parse smoke-test limits and connect using the development SDK settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-config", type=Path, default=ROOT / "config/mindmemos_skill/dev.yaml")
    parser.add_argument("--sdk-home", type=Path, default=ROOT / ".mindmemos")
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=2)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=16384,
        help="Target output budget including reasoning (default: 16384)",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/skill_forest" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate inputs and service authentication without training"
    )
    args = parser.parse_args()
    if args.train_count < 2 or args.test_count < 1:
        parser.error("train-count must be >= 2 and test-count must be >= 1")
    configure_logging()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
