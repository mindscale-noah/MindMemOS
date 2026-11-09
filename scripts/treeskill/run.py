"""Register, train and test TreeSkill through the existing HTTP SDK."""

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import AsyncSkillClient
from mindmemos_sdk.skills.cli_files import collect_resources, export_source, read_source
from mindmemos_sdk.skills.config import EnvironmentConfig
from mindmemos_sdk.skills.typing import (
    RegisterRequest,
    ShowRequest,
    SkillData,
    StructuredSkillInfo,
    TaskProgressRequest,
    TaskShowRequest,
    TrainingRequest,
    TrainingSplitRequest,
)
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.agents import AgentCatalog
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainerConfig
from mindmemos_skill.common.checkpoint import boundary, fingerprint
from mindmemos_skill.common.json import append_jsonl, read_json, read_jsonl, write_json
from mindmemos_skill.envs.data import Data, DataSample
from mindmemos_skill.logging import configure_logging, get_logger, log_context

from scripts.treeskill.report import build_comparison

logger = get_logger(__name__)


async def _event(root: Path, event: str, **details) -> None:
    """Save and display non-sensitive progress fields."""
    row = {"time": datetime.now(UTC).isoformat(), "event": event, **details}
    await append_jsonl(root / "events.jsonl", row)
    get_logger("treeskill.client").info(
        f"[{details.get('stage', 'treeskill.experiment')}] {event}",
        extra={"event_body": {"stage": "treeskill.experiment", **row}},
    )


async def _train(
    client: AsyncSkillClient,
    *,
    root: Path,
    source: StructuredSkillInfo,
    train: list[DataSample],
    test: list[DataSample],
    env: EnvironmentConfig,
    endpoint: str,
    poll_interval: float,
) -> SkillData:
    """Submit one input-bound task and export its exact published version."""
    tasks = [sample.model_dump(mode="json") for sample in [*train, *test]]
    split = TrainingSplitRequest(train=[sample.task.id for sample in train], test=[sample.task.id for sample in test])
    identity = {
        "endpoint": endpoint,
        "source": source.model_dump(mode="json"),
        "tasks": tasks,
        "split": split.model_dump(mode="json"),
        "env": env.model_dump(mode="json"),
        "work_dir": str(root / "training"),
    }
    boundary(root / "submission.json", identity, payload={"protocol": 1})
    registration_path = root / "registration.json"
    if registration_path.exists():
        registered = read_json(registration_path)
    else:
        registered = await client.register(
            RegisterRequest(alias=f"treeskill-{fingerprint(str(root))[:20]}", skill=source)
        )
        write_json(registration_path, registered)
    accepted_path = root / "training_task.json"
    if accepted_path.exists():
        accepted = read_json(accepted_path)
    else:
        accepted = await client.train(
            TrainingRequest(
                init_skill_version_id=registered["skill_version_id"],
                work_dir=str(root / "training"),
                tasks=tasks,
                split=split,
                algo="treeskill",
                env=env,
            )
        )
        write_json(accepted_path, accepted)
    task_id = accepted["task_id"]
    await _event(root, "training_submitted", task_id=task_id)
    progress_path = root / "service_progress.jsonl"
    after = max((event["sequence"] for event in read_jsonl(progress_path)), default=0) if progress_path.exists() else 0
    failure = {}

    async def save_log(event: dict[str, Any]) -> None:
        """Save a serialized task log containing sequence and payload, retaining its failure diagnosis."""
        nonlocal after, failure
        await append_jsonl(progress_path, event)
        after = max(after, event["sequence"])
        payload = event["payload"]
        body = payload.get("body") or {}
        await append_jsonl(
            root / "events.jsonl",
            {
                "time": datetime.now(UTC).isoformat(),
                "event": "training_progress",
                "stage": body.get("stage"),
                "message": payload["message"],
                "body": body,
            },
        )
        with log_context(
            project_id=event.get("project_id"),
            task_id=task_id,
            attempt=event.get("attempt"),
            request_id=payload.get("request_id"),
            run_id=payload.get("run_id"),
            tree_id=payload.get("tree_id"),
            parent_id=payload.get("parent_id"),
            node_id=payload.get("node_id"),
            group_id=payload.get("group_id"),
        ):
            get_logger("treeskill.client").log(
                logging.getLevelNamesMapping().get(payload.get("level"), logging.INFO),
                payload["message"],
            )
        if payload.get("traceback"):
            failure = payload

    while True:
        progress = await client.task_progress(TaskProgressRequest(task_id=task_id, after=after))
        for event in progress:
            await save_log(event)
        state = await client.show_task(TaskShowRequest(task_id=task_id))
        write_json(root / "training_status.json", state)
        build_comparison(root)
        if state["status"] in {"succeeded", "failed", "cancelled"}:
            # Terminal state is visible only after the task log writer has drained.
            while True:
                final_logs = await client.task_progress(TaskProgressRequest(task_id=task_id, after=after))
                if not final_logs:
                    break
                for event in final_logs:
                    await save_log(event)
        if state["status"] == "succeeded":
            result = state["result"]
            write_json(root / "training_result.json", result)
            detail = await client.show(ShowRequest(skill_version_id=result["version_id"]))
            version = detail["version"]
            write_json(root / "evolved_skill.json", version)
            export_source(version, str(root / "evolved_skill"), overwrite=True)
            return result
        if state["status"] in {"failed", "cancelled"}:
            detail = failure.get("traceback") or failure.get("message") or state.get("error_type")
            raise RuntimeError(f"Training task {task_id} ended as {state['status']}:\n{detail}")
        await asyncio.sleep(poll_interval)


async def _run(args: argparse.Namespace) -> None:
    """Load local input files and delegate all experiment execution to the SDK."""
    root = args.output.resolve()
    document = yaml.safe_load(args.service_config.read_text(encoding="utf-8"))
    pipeline = document["service"]["training_pipelines"]["treeskill"]["config"]
    algorithm = TreeSkillTrainerConfig.model_validate(
        pipeline["trainer"]["config"], context=AgentCatalog(agents=document["service"].get("agents", {}))
    )
    policy = (algorithm.failure_optimizer or algorithm.optimizer).model_copy(
        update={"temperature": 0.7, "stop": ["Observation:"]}
    )
    env = EnvironmentConfig(
        class_name="spreadsheetbench_text_action",
        config={
            "agent": policy.model_dump(mode="json"),
            "max_turns": args.max_turns,
            "transactional_recalculation": True,
        },
    )
    data_root = args.data_root.resolve()
    data = Data.build("spreadsheetbench", {"data_path": data_root / "dataset.json", "data_root": data_root})
    split = data.split(data.load(), split_file=args.split_file.resolve())
    train, test = split["train"][: args.train_count], split["test"][: args.test_count]
    source_root, content, header = read_source(str(args.initial_skill.resolve()))
    source = StructuredSkillInfo(
        content=content,
        resources=collect_resources(source_root),
        name=header["name"],
        description=header["description"],
    )
    sdk = ConfigManager(args.sdk_home).load()
    url, api_key = sdk.resolve_skill_connection()
    manifest = {
        "status": "training",
        "protocol": {
            "baseline_skipped": True,
            "model": policy.model,
            "router_model": algorithm.router.model,
            "train_count": len(train),
            "test_count": len(test),
            "test_execution": "service",
            "max_turns": args.max_turns,
            "transactional_recalculation": True,
        },
        "train_ids": [sample.task.id for sample in train],
        "test_ids": [sample.task.id for sample in test],
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "split.json", {"train": manifest["train_ids"], "val": [], "test": manifest["test_ids"]})
    try:
        async with AsyncHttpTransport(
            base_url=url,
            api_key=api_key,
            timeout_seconds=sdk.network.timeout_seconds,
            max_retries=0,
            require_api_key=False,
        ) as transport:
            await _train(
                AsyncSkillClient(transport),
                root=root,
                source=source,
                train=train,
                test=test,
                env=env,
                endpoint=url,
                poll_interval=args.poll_interval,
            )
            manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        try:
            write_json(root / "manifest.json", manifest)
            build_comparison(root)
        except Exception:
            if manifest["status"] != "failed":
                raise
            logger.exception("Failed to save experiment diagnostics; preserving the original run error")


def main() -> None:
    """Parse input paths and submit TreeSkill training through the configured SDK.

    Raises:
        SystemExit: Required paths or sample counts are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-config", type=Path, default=Path("config/mindmemos_skill/dev.yaml"))
    parser.add_argument("--sdk-home", type=Path, default=Path(".mindmemos"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json"),
    )
    parser.add_argument(
        "--initial-skill",
        type=Path,
        default=Path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench/initial_skill/SKILL.md"),
    )
    args = parser.parse_args()
    if args.report_only:
        print(build_comparison(args.output.resolve()))
        return
    if args.data_root is None:
        parser.error("--data-root is required for training")
    if args.train_count < 1 or args.test_count < 0 or args.max_turns < 1 or args.poll_interval <= 0:
        parser.error("train-count, max-turns and poll-interval must be positive; test-count must be nonnegative")
    asyncio.run(_run(args))


if __name__ == "__main__":
    configure_logging()
    main()
