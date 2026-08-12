#!/usr/bin/env python3
"""Run TreeSkill Evolution followed by routed test evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill import TaskCollectionConfig
from mindmemos_skill.algos.trace2skill.treeskill import TreeSkill, TreeSkillConfig
from mindmemos_skill.envs.registered_envs.spreadsheetbench.recalculation import (
    preflight_recalculation_runtime,
)
from mindmemos_skill.llm import DatabaseLLMCallSink
from mindmemos_skill.persistence import bootstrap_skill_database
from mindmemos_skill.typing import (
    Skill,
    SkillCandidate,
    SkillInjectionMode,
    Trace2SkillInput,
    compute_skill_content_hash,
)

from .skill_evaluation import (
    TestEvaluationConfig,
    build_agent,
    build_client,
    build_dataset,
    build_skill,
    environment_options,
    evaluate_test_tasks,
    limited_test_tasks,
    write_json,
)


@dataclass(frozen=True, slots=True)
class _AlgorithmContext:
    models: dict[str, Any]
    agents: dict[str, Any]
    config_hash: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("spreadsheetbench",), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)

    parser.add_argument("--target-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--optimizer-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)

    parser.add_argument("--train-limit", type=int, default=8)
    parser.add_argument("--collection-rollouts", type=int, default=1)
    parser.add_argument("--min-trajectories", type=int, default=8)
    parser.add_argument("--max-trajectories", type=int, default=8)
    parser.add_argument("--transcript-max-chars", type=int, default=20000)
    parser.add_argument("--annotation-mode", choices=("auto", "required", "ignore"), default="required")
    parser.add_argument("--require-skill-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--success-score-threshold", type=float, default=1.0)
    parser.add_argument("--analysis-concurrency", type=int, default=16)
    parser.add_argument("--localization-concurrency", type=int, default=16)
    parser.add_argument("--analysis-temperature", type=float, default=1.0)
    parser.add_argument("--localization-temperature", type=float, default=0.0)
    parser.add_argument("--fusion-temperature", type=float, default=0.0)
    parser.add_argument("--analysis-max-tokens", type=int, default=4096)
    parser.add_argument("--localization-max-tokens", type=int, default=2048)
    parser.add_argument("--fusion-max-tokens", type=int, default=4096)

    parser.add_argument("--tree-router-temperature", type=float, default=0.0)
    parser.add_argument("--tree-router-max-tokens", type=int, default=512)
    parser.add_argument("--test-rollouts", type=int, default=1)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=16)
    parser.add_argument("--rollout-timeout", type=float)
    parser.add_argument("--rollout-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--shell-timeout", type=int, default=120)
    parser.add_argument(
        "--transactional-recalculation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use-theorem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-sketch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shuffle-choices", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _limited_train_tasks(tasks: list[Any], limit: int) -> list[Any]:
    if limit < 1:
        raise ValueError("train-limit must be at least 1")
    return tasks[:limit]


def _config_hash(args: argparse.Namespace) -> str:
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _candidate_skill(base: Skill, candidate: SkillCandidate | None, *, run_id: str) -> Skill:
    if candidate is None:
        return base
    blob = dict(candidate.blob)
    return base.model_copy(
        update={
            "version_id": f"{run_id}:candidate",
            "parent_version_ids": [base.version_id],
            "content_hash": compute_skill_content_hash(blob),
            "blob": blob,
            "resources": candidate.resources,
            "commit_message": candidate.commit_message,
            "metadata": {**base.metadata, **candidate.metadata},
        },
        deep=True,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    if not args.api_base:
        raise ValueError("--api-base, OPENAI_ENDPOINT, or OPENAI_BASE_URL is required")

    dataset = build_dataset(
        args.benchmark,
        data_root=args.data_root,
        split_dir=args.split_dir,
        seed=args.seed,
        shuffle_choices=args.shuffle_choices,
    )
    train_tasks = _limited_train_tasks(dataset.train_tasks(), args.train_limit)
    test_tasks = limited_test_tasks(dataset, args.test_limit)
    base_skill = build_skill(args.initial_skill, run_id=args.run_id, benchmark=args.benchmark)
    env_options = environment_options(
        args.benchmark,
        max_turns=args.max_turns,
        env_seed=args.env_seed,
        shell_timeout=args.shell_timeout,
        use_theorem=args.use_theorem,
        use_sketch=args.use_sketch,
        transactional_recalculation=args.transactional_recalculation,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "arguments.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    if args.transactional_recalculation:
        preflight_recalculation_runtime(args.output_dir / "recalculation_preflight")
    database = await bootstrap_skill_database(args.output_dir / "state.db")
    sink = DatabaseLLMCallSink(database)
    target_client = build_client(
        model=args.target_model,
        api_base=args.api_base,
        request_timeout=args.request_timeout,
        model_retries=args.model_retries,
        call_sink=sink,
    )
    optimizer_client = build_client(
        model=args.optimizer_model,
        api_base=args.api_base,
        request_timeout=args.request_timeout,
        model_retries=args.model_retries,
        call_sink=sink,
    )
    collection_agent = build_agent(
        client=target_client,
        model=args.target_model,
        max_turns=args.max_turns,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
    )
    routed_agent = build_agent(
        client=target_client,
        model=args.target_model,
        max_turns=args.max_turns,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        skill_injection_mode=SkillInjectionMode.TREE_ROUTED_SYSTEM_PROMPT,
        tree_router_temperature=args.tree_router_temperature,
        tree_router_max_tokens=args.tree_router_max_tokens,
    )
    config = TreeSkillConfig(
        min_trajectories=args.min_trajectories,
        max_trajectories=args.max_trajectories,
        transcript_max_chars=args.transcript_max_chars,
        annotation_mode=args.annotation_mode,
        require_skill_match=args.require_skill_match,
        success_score_threshold=args.success_score_threshold,
        analysis_concurrency=args.analysis_concurrency,
        localization_concurrency=args.localization_concurrency,
        analysis_temperature=args.analysis_temperature,
        localization_temperature=args.localization_temperature,
        fusion_temperature=args.fusion_temperature,
        analysis_max_tokens=args.analysis_max_tokens,
        localization_max_tokens=args.localization_max_tokens,
        fusion_max_tokens=args.fusion_max_tokens,
        collection=TaskCollectionConfig(
            agent_ref="react",
            env_ref=args.benchmark,
            samples_per_task=args.collection_rollouts,
            max_concurrent_rollouts=args.max_concurrent_rollouts,
            timeout_seconds=args.rollout_timeout,
            retry={"max_attempts": args.rollout_retries},
            fail_fast=False,
            workspace_root=args.output_dir / "collection_workspace",
            seed=args.seed,
            env_options=env_options,
        ),
    )
    write_json(args.output_dir / "run_config.json", config.model_dump(mode="json"))
    algorithm = TreeSkill(
        config=config,
        context=_AlgorithmContext(
            models={"chat": optimizer_client},
            agents={"react": collection_agent},
            config_hash=_config_hash(args),
        ),
    )
    try:
        result = await algorithm.optimize(
            Trace2SkillInput(
                run_id=args.run_id,
                base_skill=base_skill,
                tasks=train_tasks,
            )
        )
        final_skill = _candidate_skill(base_skill, result.candidate, run_id=args.run_id)
        test_summary = await evaluate_test_tasks(
            run_id=args.run_id,
            tasks=test_tasks,
            skill=final_skill,
            agent=routed_agent,
            output_dir=args.output_dir / "test",
            config=TestEvaluationConfig(
                benchmark=args.benchmark,
                rollouts=args.test_rollouts,
                max_concurrent_rollouts=args.max_concurrent_rollouts,
                rollout_timeout=args.rollout_timeout,
                rollout_retries=args.rollout_retries,
                seed=args.seed,
                env_options=env_options,
            ),
        )
    finally:
        await database.close()

    write_json(args.output_dir / "result.json", result.model_dump(mode="json"))
    write_json(args.output_dir / "final_skill.json", final_skill.model_dump(mode="json"))
    (args.output_dir / "final_skill.md").write_text(final_skill.content, encoding="utf-8")
    tree_metadata = final_skill.metadata.get("treeskill")
    if tree_metadata is not None:
        write_json(args.output_dir / "tree_metadata.json", tree_metadata)
    summary = {
        "run_id": args.run_id,
        "changed": result.changed,
        "final_skill_hash": final_skill.content_hash,
        "treeskill": result.report.model_dump(mode="json"),
        "test": test_summary,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(asyncio.run(run(parse_args(argv))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
