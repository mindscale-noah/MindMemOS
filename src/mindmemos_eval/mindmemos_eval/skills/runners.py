"""Skill benchmark CLI dispatcher and runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from tqdm.auto import tqdm

from ..llm import LLMClient, LLMConfig
from .agents import ReactAgentFactory, RunResult
from .args import add_common_skill_args, add_spreadsheetbench_args

SkillArgRegistrar = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class SkillBenchmark:
    """CLI binding for one skill benchmark dataset."""

    help: str
    add_args: SkillArgRegistrar
    env_builder: str


@dataclass(frozen=True)
class SkillRunConfig:
    """Generic skill benchmark execution settings."""

    split: str = "all"
    limit: int | None = None
    seed: int | None = None
    show_progress: bool = False
    concurrency: int = 1


SKILL_BENCHMARKS = {
    "spreadsheetbench": SkillBenchmark(
        help="Run SpreadsheetBench skill evaluation.",
        add_args=add_spreadsheetbench_args,
        env_builder="mindmemos_eval.skills.envs.spreadsheetbench.env:build_env",
    ),
}


def _load_attr(path: str) -> Any:
    module_name, attr_name = path.split(":", maxsplit=1)
    return getattr(import_module(module_name), attr_name)


def add_skill_args(parser: argparse.ArgumentParser) -> None:
    """Register shared skill benchmark arguments."""
    common_args = argparse.ArgumentParser(add_help=False)
    add_common_skill_args(common_args)
    subparsers = parser.add_subparsers(
        dest="benchmark",
        metavar="BENCHMARK",
        required=True,
        help="Skill benchmark dataset to run.",
    )
    for name, benchmark in sorted(SKILL_BENCHMARKS.items()):
        subparser = subparsers.add_parser(
            name,
            help=benchmark.help,
            description=benchmark.help,
            parents=[common_args],
        )
        benchmark.add_args(subparser)
        subparser.set_defaults(env_builder=benchmark.env_builder)
    local = subparsers.add_parser("local", help="Run a local Skill training and test benchmark.")
    local.add_argument("--config", required=True, metavar="YAML", help="Single-run evaluation YAML.")
    local.add_argument("--output-dir", required=True, metavar="PATH", help="Isolated output and recovery directory.")
    local.add_argument("--preflight-only", action="store_true", help="Validate inputs without creating a service.")


def run_skill_benchmark(args: argparse.Namespace) -> int:
    """Run the selected skill benchmark environment."""
    if args.benchmark == "local":
        from .local import run_local_eval

        report = asyncio.run(run_local_eval(args.config, args.output_dir, preflight_only=args.preflight_only))
        if report is None:
            print("Local Skill evaluation preflight passed.")
        else:
            print(f"Local Skill evaluation complete: {args.output_dir}/summary.json")
        return 0
    env = _load_attr(args.env_builder)(args)
    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    agent_factory = ReactAgentFactory(
        LLMClient(llm_config),
        max_turns=args.max_turns,
        skill_sources=args.skill or [],
        python_path=args.python_path,
    )
    config = SkillRunConfig(
        limit=args.limit,
        seed=args.seed,
        show_progress=args.progress,
        concurrency=args.concurrency or getattr(env, "default_concurrency", 1),
    )
    result = asyncio.run(SkillEvalRunner(env, agent_factory, config).run())
    _print_result(args.benchmark, env, result, seed=args.seed)
    return 0


class SkillEvalRunner:
    """Run one benchmark environment with an agent factory."""

    def __init__(
        self,
        env: Any,
        agent_factory: ReactAgentFactory,
        config: SkillRunConfig,
    ) -> None:
        self.env = env
        self.agent_factory = agent_factory
        self.config = config

    async def run(self) -> Any:
        cases = self.env.load_cases(self.config.split)
        if self.config.seed is not None:
            random.Random(self.config.seed).shuffle(cases)
        if self.config.limit is not None:
            cases = cases[: self.config.limit]
        if self.config.concurrency > 1:
            results = await self._run_concurrent(cases)
        else:
            results = await self._run_online(cases)
        return self.env.summarize(self.config.split, results)

    async def _run_case(self, case: Any, *, rollout: int = 0, log: bool = True) -> Any:
        workdir = self.env.case_workdir(case, rollout)
        workdir.mkdir(parents=True, exist_ok=True)
        self.env.setup_case(case, workdir)

        agent, _tools = self.agent_factory.build(workdir, self.env.system_prompt())
        messages = self.env.build_messages(case)
        started_at = time.time()
        error: str | None = None
        score = 0.0
        score_message = ""
        try:
            agent_result = await agent.run(messages)
        except Exception as exc:  # noqa: BLE001 - keep one failed case from stopping the run
            agent_result = RunResult(messages=messages, turns=0, finished=False)
            error = f"{type(exc).__name__}: {exc}"

        if error is None:
            try:
                score, score_message = self.env.score(case, workdir)
            except Exception as exc:  # noqa: BLE001 - scoring failures are case failures
                error = f"score error: {type(exc).__name__}: {exc}"
                score = 0.0

        result = self.env.build_result(
            case,
            agent_result,
            workdir=workdir,
            score=score,
            score_message=score_message,
            error=error,
            started_at=started_at,
            ended_at=time.time(),
            rollout=rollout,
        )
        if log:
            self.env.append_trajectory(result)
        return result

    async def _run_online(self, cases: list[Any]) -> list[Any]:
        results: list[Any] = []
        progress = tqdm(cases, desc=self.env.name, unit="task", disable=not self.config.show_progress)
        correct = 0
        for case in progress:
            result = await self._run_case(case)
            results.append(result)
            if result.score >= 1.0:
                correct += 1
            progress.set_postfix(correct=correct, score=f"{correct / len(results):.4f}", last=result.case_id)
        return results

    async def _run_concurrent(self, cases: list[Any]) -> list[Any]:
        results: list[Any | None] = [None] * len(cases)
        progress = tqdm(total=len(cases), desc=self.env.name, unit="task", disable=not self.config.show_progress)
        correct = 0
        completed = 0

        def update_progress(result: Any) -> None:
            nonlocal completed, correct
            completed += 1
            if result.score >= 1.0:
                correct += 1
            progress.update(1)
            progress.set_postfix(correct=correct, score=f"{correct / completed:.4f}", last=result.case_id)

        try:
            ordered = await self._run_cases_concurrently(
                cases,
                concurrency=self.config.concurrency,
                on_result=update_progress,
            )
            for index, result in enumerate(ordered):
                results[index] = result
                self.env.append_trajectory(result)
        finally:
            progress.close()
        return [result for result in results if result is not None]

    async def _run_cases_concurrently(
        self,
        cases: list[Any],
        *,
        concurrency: int,
        on_result: Callable[[Any], None] | None = None,
    ) -> list[Any]:
        semaphore = asyncio.Semaphore(max(1, concurrency))
        results: list[Any | None] = [None] * len(cases)

        async def run_indexed(index: int, case: Any) -> tuple[int, Any]:
            async with semaphore:
                return index, await self._run_case(case, log=False)

        tasks = [asyncio.create_task(run_indexed(index, case)) for index, case in enumerate(cases)]
        for completed in asyncio.as_completed(tasks):
            index, result = await completed
            results[index] = result
            if on_result is not None:
                on_result(result)
        return [result for result in results if result is not None]


def _print_result(benchmark: str, env: Any, result: Any, *, seed: int | None) -> None:
    print(f"{benchmark} result: total={result.total} correct={result.correct} score={result.accuracy:.4f} seed={seed}")
    print(
        json.dumps(
            {
                "env": benchmark,
                "data_root": str(getattr(env, "data_root", "")),
                "run_dir": result.run_dir,
                "summary": f"{result.split}_summary.json",
                "trajectories": str(env.trajectory_path(result.split)),
                "score": result.accuracy,
            },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone skill benchmark parser."""
    parser = argparse.ArgumentParser(description="Run MindMemOS skill benchmark evaluation.")
    add_skill_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI entry point."""
    return run_skill_benchmark(build_parser().parse_args(argv))
